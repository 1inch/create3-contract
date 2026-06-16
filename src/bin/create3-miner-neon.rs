//! ARM NEON (aarch64) build of the CREATE3 vanity miner.
//!
//! Hashes two salts at a time using 128-bit NEON vectors (2 x u64 lanes), so a
//! single keccak-f[1600] permutation advances two independent sponges in
//! lockstep. On non-aarch64 targets this binary compiles to a stub that exits
//! with an error pointing at the scalar `create3-miner`.

#[cfg(target_arch = "aarch64")]
fn main() {
    neon::run();
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!(
        "create3-miner-neon requires an aarch64 (ARM NEON) CPU. \
         Use the scalar `create3-miner` binary on this platform."
    );
    std::process::exit(1);
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod neon {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use core::arch::aarch64::*;
    use rand::Rng;

    use create3_miner::{
        hex_encode_addr, leading_match, parse_cli, run_search, MatchMode,
    };

    /// Rotate-left each 64-bit lane by a compile-time constant `$s` (1..=63).
    ///
    /// NEON has no 64-bit vector rotate, but `vsri` (shift-right-and-insert)
    /// lets us do it in two instructions for *both* lanes at once:
    ///   rotl(x, s) = (x << s) | (x >> (64 - s))
    /// which matches a scalar `ror` on a per-lane basis.
    macro_rules! rotl {
        ($x:expr, $s:literal) => {
            vsriq_n_u64::<{ 64 - $s }>(vshlq_n_u64::<$s>($x), $x)
        };
    }

    // Keccak-f[1600] round constants.
    const RC: [u64; 24] = [
        0x0000000000000001,
        0x0000000000008082,
        0x800000000000808a,
        0x8000000080008000,
        0x000000000000808b,
        0x0000000080000001,
        0x8000000080008081,
        0x8000000000008009,
        0x000000000000008a,
        0x0000000000000088,
        0x0000000080008009,
        0x000000008000000a,
        0x000000008000808b,
        0x800000000000008b,
        0x8000000000008089,
        0x8000000000008003,
        0x8000000000008002,
        0x8000000000000080,
        0x000000000000800a,
        0x800000008000000a,
        0x8000000080008081,
        0x8000000000008080,
        0x0000000080000001,
        0x8000000080008008,
    ];

    /// Keccak-f[1600] permutation over two independent states packed lane-wise
    /// into 25 `uint64x2_t` words (lane 0 = state A, lane 1 = state B).
    ///
    /// The rho/pi step is fully unrolled so every rotation amount is a compile-
    /// time constant (see the [`rotl`] macro).
    #[inline]
    #[allow(unused_assignments)]
    unsafe fn keccak_f1600_x2(a: &mut [uint64x2_t; 25]) {
        let mut array: [uint64x2_t; 5] = [vdupq_n_u64(0); 5];

        for round in 0..24 {
            // Theta
            for x in 0..5 {
                array[x] = veorq_u64(
                    veorq_u64(veorq_u64(a[x], a[x + 5]), veorq_u64(a[x + 10], a[x + 15])),
                    a[x + 20],
                );
            }
            for x in 0..5 {
                let t = veorq_u64(array[(x + 4) % 5], rotl!(array[(x + 1) % 5], 1));
                let mut y = 0;
                while y < 25 {
                    a[x + y] = veorq_u64(a[x + y], t);
                    y += 5;
                }
            }

            // Rho and Pi (unrolled with constant rotation amounts).
            let mut last = a[1];
            macro_rules! step {
                ($pi:literal, $r:literal) => {{
                    let cur = a[$pi];
                    a[$pi] = rotl!(last, $r);
                    last = cur;
                }};
            }
            step!(10, 1);
            step!(7, 3);
            step!(11, 6);
            step!(17, 10);
            step!(18, 15);
            step!(3, 21);
            step!(5, 28);
            step!(16, 36);
            step!(8, 45);
            step!(21, 55);
            step!(24, 2);
            step!(4, 14);
            step!(15, 27);
            step!(23, 41);
            step!(19, 56);
            step!(13, 8);
            step!(12, 25);
            step!(2, 43);
            step!(20, 62);
            step!(14, 18);
            step!(22, 39);
            step!(9, 61);
            step!(6, 20);
            step!(1, 44);

            // Chi
            let mut y = 0;
            while y < 25 {
                for x in 0..5 {
                    array[x] = a[y + x];
                }
                for x in 0..5 {
                    // a = array[x] ^ (!array[x+1] & array[x+2])
                    // vbicq_u64(p, q) = p & !q
                    let notnext = vbicq_u64(array[(x + 2) % 5], array[(x + 1) % 5]);
                    a[y + x] = veorq_u64(array[x], notnext);
                }
                y += 5;
            }

            // Iota
            a[0] = veorq_u64(a[0], vdupq_n_u64(RC[round]));
        }
    }

    /// Builds a single 136-byte (rate) padded keccak256 block from `input`
    /// (which must be shorter than 136 bytes).
    #[inline(always)]
    fn build_block(input: &[u8]) -> [u8; 136] {
        debug_assert!(input.len() < 136);
        let mut b = [0u8; 136];
        b[..input.len()].copy_from_slice(input);
        b[input.len()] ^= 0x01;
        b[135] ^= 0x80;
        b
    }

    /// Computes keccak256 of two single-block inputs simultaneously.
    #[inline]
    unsafe fn keccak256_x2(in_a: &[u8], in_b: &[u8]) -> ([u8; 32], [u8; 32]) {
        let ba = build_block(in_a);
        let bb = build_block(in_b);

        let mut state = [vdupq_n_u64(0); 25];
        for w in 0..17 {
            let wa = u64::from_le_bytes(ba[w * 8..w * 8 + 8].try_into().unwrap());
            let wb = u64::from_le_bytes(bb[w * 8..w * 8 + 8].try_into().unwrap());
            let pair = [wa, wb];
            state[w] = vld1q_u64(pair.as_ptr());
        }

        keccak_f1600_x2(&mut state);

        let mut out_a = [0u8; 32];
        let mut out_b = [0u8; 32];
        for w in 0..4 {
            let mut pair = [0u64; 2];
            vst1q_u64(pair.as_mut_ptr(), state[w]);
            out_a[w * 8..w * 8 + 8].copy_from_slice(&pair[0].to_le_bytes());
            out_b[w * 8..w * 8 + 8].copy_from_slice(&pair[1].to_le_bytes());
        }
        (out_a, out_b)
    }

    /// Derives two CREATE3 addresses (for salts ending in `c0` and `c1`) using
    /// batched NEON keccak. `template` is the 85-byte first-stage buffer with the
    /// fixed prefix/factory/code-hash already filled; only the salt counter
    /// (bytes 45..53) is rewritten per call.
    #[inline(always)]
    unsafe fn addr_pair(template: &[u8; 85], c0: u64, c1: u64) -> ([u8; 20], [u8; 20]) {
        let mut buf_a = *template;
        let mut buf_b = *template;
        buf_a[45..53].copy_from_slice(&c0.to_be_bytes());
        buf_b[45..53].copy_from_slice(&c1.to_be_bytes());

        let (proxy_a, proxy_b) = keccak256_x2(&buf_a, &buf_b);

        let mut s2a = [0u8; 23];
        let mut s2b = [0u8; 23];
        s2a[0] = 0xd6;
        s2a[1] = 0x94;
        s2a[22] = 0x01;
        s2b[0] = 0xd6;
        s2b[1] = 0x94;
        s2b[22] = 0x01;
        s2a[2..22].copy_from_slice(&proxy_a[12..32]);
        s2b[2..22].copy_from_slice(&proxy_b[12..32]);

        let (ah, bh) = keccak256_x2(&s2a, &s2b);

        let mut addr_a = [0u8; 20];
        let mut addr_b = [0u8; 20];
        addr_a.copy_from_slice(&ah[12..32]);
        addr_b.copy_from_slice(&bh[12..32]);
        (addr_a, addr_b)
    }

    pub fn run() {
        let config = parse_cli("create3-miner-neon");
        let factory = config.factory;
        let code_hash = config.code_hash;
        let mode = config.mode.clone();

        run_search(&config, move |stop: &AtomicBool, attempts: &AtomicU64| {
            // Fixed random 24-byte salt prefix; the trailing 8 bytes are a counter.
            let mut salt_prefix = [0u8; 24];
            rand::rng().fill_bytes(&mut salt_prefix);
            let mut counter_bytes = [0u8; 8];
            rand::rng().fill_bytes(&mut counter_bytes);
            let mut counter: u64 = u64::from_le_bytes(counter_bytes);

            let mut template = [0u8; 85];
            template[0] = 0xff;
            template[1..21].copy_from_slice(&factory);
            template[21..45].copy_from_slice(&salt_prefix);
            template[53..85].copy_from_slice(&code_hash);

            let mut hex_buf = [0u8; 40];
            let mut local: u64 = 0;
            const BATCH: u64 = 8192;

            let check = |addr: &[u8; 20], hex_buf: &mut [u8; 40]| -> bool {
                match &mode {
                    MatchMode::Leading(nibbles) => leading_match(addr, nibbles),
                    MatchMode::Regex(re) => {
                        hex_encode_addr(addr, hex_buf);
                        let hex_str = unsafe { std::str::from_utf8_unchecked(hex_buf) };
                        re.is_match(hex_str)
                    }
                }
            };

            let make_salt = |c: u64| -> [u8; 32] {
                let mut s = [0u8; 32];
                s[..24].copy_from_slice(&salt_prefix);
                s[24..32].copy_from_slice(&c.to_be_bytes());
                s
            };

            loop {
                let mut i = 0u64;
                while i < BATCH {
                    let c0 = counter;
                    let c1 = counter.wrapping_add(1);
                    let (addr_a, addr_b) = unsafe { addr_pair(&template, c0, c1) };
                    local += 2;

                    if check(&addr_a, &mut hex_buf) {
                        attempts.fetch_add(local, Ordering::Relaxed);
                        stop.store(true, Ordering::Relaxed);
                        return Some((make_salt(c0), addr_a));
                    }
                    if check(&addr_b, &mut hex_buf) {
                        attempts.fetch_add(local, Ordering::Relaxed);
                        stop.store(true, Ordering::Relaxed);
                        return Some((make_salt(c1), addr_b));
                    }

                    counter = counter.wrapping_add(2);
                    i += 2;
                }
                attempts.fetch_add(local, Ordering::Relaxed);
                local = 0;
                if stop.load(Ordering::Relaxed) {
                    return None;
                }
            }
        });
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use create3_miner::{create3_address, keccak256, parse_address, DEFAULT_PROXY_CODE_HASH};

        #[test]
        fn keccak256_x2_matches_scalar() {
            let a = b"hello world";
            let b = b"the quick brown fox jumps over the lazy dog";
            let (ha, hb) = unsafe { keccak256_x2(a, b) };
            assert_eq!(ha, keccak256(a));
            assert_eq!(hb, keccak256(b));
        }

        #[test]
        fn keccak256_x2_empty_and_short() {
            let a: &[u8] = b"";
            let b: &[u8] = b"x";
            let (ha, hb) = unsafe { keccak256_x2(a, b) };
            assert_eq!(ha, keccak256(a));
            assert_eq!(hb, keccak256(b));
        }

        #[test]
        fn keccak256_x2_known_vector() {
            // keccak256("") = c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
            let (ha, _) = unsafe { keccak256_x2(b"", b"") };
            assert_eq!(
                hex::encode(ha),
                "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
            );
        }

        #[test]
        fn keccak256_x2_first_stage_length_85() {
            // The mining first stage hashes exactly 85 bytes; verify that length.
            let mut buf = [0u8; 85];
            for (i, b) in buf.iter_mut().enumerate() {
                *b = i as u8;
            }
            let (ha, hb) = unsafe { keccak256_x2(&buf, &buf) };
            assert_eq!(ha, keccak256(&buf));
            assert_eq!(hb, keccak256(&buf));
        }

        #[test]
        fn addr_pair_matches_scalar() {
            let factory = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
            let prefix = [0x11u8; 24];

            let mut template = [0u8; 85];
            template[0] = 0xff;
            template[1..21].copy_from_slice(&factory);
            template[21..45].copy_from_slice(&prefix);
            template[53..85].copy_from_slice(&DEFAULT_PROXY_CODE_HASH);

            let c0: u64 = 0x0102030405060708;
            let c1: u64 = c0 + 1;
            let (addr_a, addr_b) = unsafe { addr_pair(&template, c0, c1) };

            let salt = |c: u64| {
                let mut s = [0u8; 32];
                s[..24].copy_from_slice(&prefix);
                s[24..32].copy_from_slice(&c.to_be_bytes());
                s
            };
            assert_eq!(addr_a, create3_address(&factory, &salt(c0), &DEFAULT_PROXY_CODE_HASH));
            assert_eq!(addr_b, create3_address(&factory, &salt(c1), &DEFAULT_PROXY_CODE_HASH));
        }

        #[test]
        fn addr_pair_respects_custom_code_hash() {
            let factory = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
            let prefix = [0x22u8; 24];
            let code_hash = [0xABu8; 32];

            let mut template = [0u8; 85];
            template[0] = 0xff;
            template[1..21].copy_from_slice(&factory);
            template[21..45].copy_from_slice(&prefix);
            template[53..85].copy_from_slice(&code_hash);

            let c0: u64 = 42;
            let (addr_a, _) = unsafe { addr_pair(&template, c0, c0 + 1) };

            let mut salt = [0u8; 32];
            salt[..24].copy_from_slice(&prefix);
            salt[24..32].copy_from_slice(&c0.to_be_bytes());
            assert_eq!(addr_a, create3_address(&factory, &salt, &code_hash));
        }
    }
}
