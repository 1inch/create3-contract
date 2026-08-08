//! ARM NEON (aarch64) build of the CREATE3 vanity miner.
//!
//! Each 128-bit NEON vector holds 2 x u64 lanes, so one keccak-f[1600]
//! permutation advances two independent sponges. To raise IPC on out-of-order
//! cores we additionally multi-buffer `W` independent states per batch
//! (`2*W` salts), and on chips with the ARMv8 SHA3 extension we use the fused
//! EOR3/RAX1/XAR/BCAX instructions (with a runtime fallback to base NEON).
//!
//! On non-aarch64 targets this binary compiles to a stub that exits with an
//! error pointing at the scalar `create3-miner`.

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

    use create3_miner::{hex_encode_addr, mask_match, parse_cli, run_search, MatchMode};

    /// Rotate-left each 64-bit lane by a compile-time constant `$s` (0..=63).
    /// Used only on the base (non-SHA3) path.
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

    /// Full keccak-f[1600] permutation body over `W` independent states, written
    /// once in terms of the op-macros `$xor3 / $rax1 / $xar / $bcax`. It is
    /// instantiated twice (base NEON and SHA3 extension) so each backend gets a
    /// fully inlined, self-contained permutation.
    ///
    /// State layout: `a[s][x + 5*y]` is column x, row y of state s.
    /// Theta-add is fused into the rho/pi step via `xar(a, d, rot) = rol(a^d, rot)`.
    macro_rules! keccak_body {
        ($a:expr, $W:expr, $xor3:ident, $rax1:ident, $xar:ident, $bcax:ident) => {{
            let a = $a;
            let mut c = [[vdupq_n_u64(0); 5]; $W];
            let mut d = [[vdupq_n_u64(0); 5]; $W];
            let mut b = [[vdupq_n_u64(0); 25]; $W];

            for round in 0..24 {
                // Theta: column parities and D = C[x-1] ^ rol(C[x+1], 1).
                for s in 0..$W {
                    c[s][0] = $xor3!($xor3!(a[s][0], a[s][5], a[s][10]), a[s][15], a[s][20]);
                    c[s][1] = $xor3!($xor3!(a[s][1], a[s][6], a[s][11]), a[s][16], a[s][21]);
                    c[s][2] = $xor3!($xor3!(a[s][2], a[s][7], a[s][12]), a[s][17], a[s][22]);
                    c[s][3] = $xor3!($xor3!(a[s][3], a[s][8], a[s][13]), a[s][18], a[s][23]);
                    c[s][4] = $xor3!($xor3!(a[s][4], a[s][9], a[s][14]), a[s][19], a[s][24]);
                    d[s][0] = $rax1!(c[s][4], c[s][1]);
                    d[s][1] = $rax1!(c[s][0], c[s][2]);
                    d[s][2] = $rax1!(c[s][1], c[s][3]);
                    d[s][3] = $rax1!(c[s][2], c[s][4]);
                    d[s][4] = $rax1!(c[s][3], c[s][0]);
                }

                // Theta-add + rho (rotate) + pi (reindex), fused via xar.
                for s in 0..$W {
                    b[s][0] = $xar!(a[s][0], d[s][0], 0);
                    b[s][10] = $xar!(a[s][1], d[s][1], 1);
                    b[s][20] = $xar!(a[s][2], d[s][2], 62);
                    b[s][5] = $xar!(a[s][3], d[s][3], 28);
                    b[s][15] = $xar!(a[s][4], d[s][4], 27);
                    b[s][16] = $xar!(a[s][5], d[s][0], 36);
                    b[s][1] = $xar!(a[s][6], d[s][1], 44);
                    b[s][11] = $xar!(a[s][7], d[s][2], 6);
                    b[s][21] = $xar!(a[s][8], d[s][3], 55);
                    b[s][6] = $xar!(a[s][9], d[s][4], 20);
                    b[s][7] = $xar!(a[s][10], d[s][0], 3);
                    b[s][17] = $xar!(a[s][11], d[s][1], 10);
                    b[s][2] = $xar!(a[s][12], d[s][2], 43);
                    b[s][12] = $xar!(a[s][13], d[s][3], 25);
                    b[s][22] = $xar!(a[s][14], d[s][4], 39);
                    b[s][23] = $xar!(a[s][15], d[s][0], 41);
                    b[s][8] = $xar!(a[s][16], d[s][1], 45);
                    b[s][18] = $xar!(a[s][17], d[s][2], 15);
                    b[s][3] = $xar!(a[s][18], d[s][3], 21);
                    b[s][13] = $xar!(a[s][19], d[s][4], 8);
                    b[s][14] = $xar!(a[s][20], d[s][0], 18);
                    b[s][24] = $xar!(a[s][21], d[s][1], 2);
                    b[s][9] = $xar!(a[s][22], d[s][2], 61);
                    b[s][19] = $xar!(a[s][23], d[s][3], 56);
                    b[s][4] = $xar!(a[s][24], d[s][4], 14);
                }

                // Chi: a[x] = b[x] ^ (~b[x+1] & b[x+2]) per row.
                for s in 0..$W {
                    for y in 0..5 {
                        let r = 5 * y;
                        a[s][r] = $bcax!(b[s][r], b[s][r + 2], b[s][r + 1]);
                        a[s][r + 1] = $bcax!(b[s][r + 1], b[s][r + 3], b[s][r + 2]);
                        a[s][r + 2] = $bcax!(b[s][r + 2], b[s][r + 4], b[s][r + 3]);
                        a[s][r + 3] = $bcax!(b[s][r + 3], b[s][r], b[s][r + 4]);
                        a[s][r + 4] = $bcax!(b[s][r + 4], b[s][r + 1], b[s][r]);
                    }
                    // Iota
                    a[s][0] = veorq_u64(a[s][0], vdupq_n_u64(RC[round]));
                }
            }
        }};
    }

    /// Base NEON permutation: xor3/rax1/xar/bcax built from baseline intrinsics.
    #[inline]
    unsafe fn keccak_perm_base<const W: usize>(a: &mut [[uint64x2_t; 25]; W]) {
        macro_rules! xor3 {
            ($x:expr, $y:expr, $z:expr) => {
                veorq_u64(veorq_u64($x, $y), $z)
            };
        }
        macro_rules! rax1 {
            ($x:expr, $y:expr) => {
                veorq_u64($x, rotl!($y, 1))
            };
        }
        macro_rules! xar {
            ($x:expr, $y:expr, $r:literal) => {
                rotl!(veorq_u64($x, $y), $r)
            };
        }
        macro_rules! bcax {
            ($x:expr, $y:expr, $z:expr) => {
                veorq_u64($x, vbicq_u64($y, $z))
            };
        }
        keccak_body!(a, W, xor3, rax1, xar, bcax);
    }

    /// SHA3-extension permutation: fused EOR3/RAX1/XAR/BCAX instructions.
    #[inline]
    #[target_feature(enable = "sha3")]
    unsafe fn keccak_perm_sha3<const W: usize>(a: &mut [[uint64x2_t; 25]; W]) {
        macro_rules! xor3 {
            ($x:expr, $y:expr, $z:expr) => {
                veor3q_u64($x, $y, $z)
            };
        }
        macro_rules! rax1 {
            ($x:expr, $y:expr) => {
                vrax1q_u64($x, $y)
            };
        }
        // vxarq does ror; rol(v, r) == ror(v, (64 - r) % 64).
        macro_rules! xar {
            ($x:expr, $y:expr, $r:literal) => {
                vxarq_u64::<{ (64 - $r) % 64 }>($x, $y)
            };
        }
        macro_rules! bcax {
            ($x:expr, $y:expr, $z:expr) => {
                vbcaxq_u64($x, $y, $z)
            };
        }
        keccak_body!(a, W, xor3, rax1, xar, bcax);
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

    /// Given the 16-byte window covering buffer bytes 40..56 (with the counter
    /// zeroed), produces the two keccak input words (w5, w6) for counter `c`.
    /// The counter occupies bytes 45..53 (big-endian).
    #[inline(always)]
    fn counter_words(cw_base: &[u8; 16], c: u64) -> (u64, u64) {
        let mut cw = *cw_base;
        cw[5..13].copy_from_slice(&c.to_be_bytes());
        (
            u64::from_le_bytes(cw[0..8].try_into().unwrap()),
            u64::from_le_bytes(cw[8..16].try_into().unwrap()),
        )
    }

    /// Produces the three nonzero keccak input words for the 23-byte second
    /// stage `0xd6 0x94 ++ proxy[12..32] ++ 0x01` (plus padding).
    #[inline(always)]
    fn stage2_words(proxy: &[u8; 32]) -> (u64, u64, u64) {
        let p = &proxy[12..32];
        let w0 = u64::from_le_bytes([0xd6, 0x94, p[0], p[1], p[2], p[3], p[4], p[5]]);
        let w1 = u64::from_le_bytes([p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13]]);
        // bytes: p[14..20], then 0x01 (input terminator) and 0x01 (block pad).
        let w2 = u64::from_le_bytes([p[14], p[15], p[16], p[17], p[18], p[19], 0x01, 0x01]);
        (w0, w1, w2)
    }

    /// Extracts the two 32-byte hash outputs (one per lane) from a permuted state.
    #[inline(always)]
    unsafe fn extract_hash(state: &[uint64x2_t; 25]) -> ([u8; 32], [u8; 32]) {
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        for w in 0..4 {
            let mut pair = [0u64; 2];
            vst1q_u64(pair.as_mut_ptr(), state[w]);
            a[w * 8..w * 8 + 8].copy_from_slice(&pair[0].to_le_bytes());
            b[w * 8..w * 8 + 8].copy_from_slice(&pair[1].to_le_bytes());
        }
        (a, b)
    }

    #[inline(always)]
    fn addr_from_hash(h: &[u8; 32]) -> [u8; 20] {
        let mut a = [0u8; 20];
        a.copy_from_slice(&h[12..32]);
        a
    }

    #[inline(always)]
    fn make_salt(prefix: &[u8; 24], c: u64) -> [u8; 32] {
        let mut s = [0u8; 32];
        s[..24].copy_from_slice(prefix);
        s[24..32].copy_from_slice(&c.to_be_bytes());
        s
    }

    #[inline(always)]
    fn check(mode: &MatchMode, addr: &[u8; 20], hex_buf: &mut [u8; 40]) -> bool {
        match mode {
            MatchMode::Mask { value, mask } => mask_match(addr, value, mask),
            MatchMode::Regex(re) => {
                hex_encode_addr(addr, hex_buf);
                let hex_str = unsafe { std::str::from_utf8_unchecked(hex_buf) };
                re.is_match(hex_str)
            }
        }
    }

    /// One thread's mining loop, multi-buffered over `W` independent states
    /// (`2*W` salts per batch element).
    unsafe fn mine<const W: usize>(
        use_sha3: bool,
        factory: &[u8; 20],
        code_hash: &[u8; 32],
        mode: &MatchMode,
        stop: &AtomicBool,
        attempts: &AtomicU64,
    ) -> Option<([u8; 32], [u8; 20])> {
        // Random 24-byte salt prefix; the trailing 8 bytes are a counter.
        let mut salt_prefix = [0u8; 24];
        rand::rng().fill_bytes(&mut salt_prefix);
        let mut cbytes = [0u8; 8];
        rand::rng().fill_bytes(&mut cbytes);
        let mut counter = u64::from_le_bytes(cbytes);

        // 85-byte first-stage buffer (counter zeroed) and its constant words.
        let mut tmpl = [0u8; 85];
        tmpl[0] = 0xff;
        tmpl[1..21].copy_from_slice(factory);
        tmpl[21..45].copy_from_slice(&salt_prefix);
        tmpl[53..85].copy_from_slice(code_hash);

        let block = build_block(&tmpl);
        let mut s1tmpl = [vdupq_n_u64(0); 25];
        for w in 0..17 {
            let word = u64::from_le_bytes(block[w * 8..w * 8 + 8].try_into().unwrap());
            s1tmpl[w] = vdupq_n_u64(word);
        }
        let mut cw_base = [0u8; 16];
        cw_base.copy_from_slice(&tmpl[40..56]);

        // Second-stage constant state: only the 0x80 block-pad word is fixed.
        let mut s2tmpl = [vdupq_n_u64(0); 25];
        s2tmpl[16] = vdupq_n_u64(0x80u64 << 56);

        let mut hex_buf = [0u8; 40];
        let mut local: u64 = 0;
        const BATCH: u64 = 4096;

        loop {
            let mut i = 0u64;
            while i < BATCH {
                // Stage 1: build W states (2 lanes each) from constant template.
                let mut states = [[vdupq_n_u64(0); 25]; W];
                let mut cnt = [[0u64; 2]; W];
                for s in 0..W {
                    states[s] = s1tmpl;
                    let ca = counter;
                    let cb = counter.wrapping_add(1);
                    cnt[s] = [ca, cb];
                    let (w5a, w6a) = counter_words(&cw_base, ca);
                    let (w5b, w6b) = counter_words(&cw_base, cb);
                    let p5 = [w5a, w5b];
                    let p6 = [w6a, w6b];
                    states[s][5] = vld1q_u64(p5.as_ptr());
                    states[s][6] = vld1q_u64(p6.as_ptr());
                    counter = counter.wrapping_add(2);
                }
                if use_sha3 {
                    keccak_perm_sha3::<W>(&mut states);
                } else {
                    keccak_perm_base::<W>(&mut states);
                }

                // Stage 2: derive proxy addresses and build the second sponge.
                let mut states2 = [[vdupq_n_u64(0); 25]; W];
                for s in 0..W {
                    states2[s] = s2tmpl;
                    let (pa, pb) = extract_hash(&states[s]);
                    let (w0a, w1a, w2a) = stage2_words(&pa);
                    let (w0b, w1b, w2b) = stage2_words(&pb);
                    let p0 = [w0a, w0b];
                    let p1 = [w1a, w1b];
                    let p2 = [w2a, w2b];
                    states2[s][0] = vld1q_u64(p0.as_ptr());
                    states2[s][1] = vld1q_u64(p1.as_ptr());
                    states2[s][2] = vld1q_u64(p2.as_ptr());
                }
                if use_sha3 {
                    keccak_perm_sha3::<W>(&mut states2);
                } else {
                    keccak_perm_base::<W>(&mut states2);
                }

                // Final addresses + matching.
                for s in 0..W {
                    let (ha, hb) = extract_hash(&states2[s]);
                    let addr_a = addr_from_hash(&ha);
                    let addr_b = addr_from_hash(&hb);
                    local += 2;
                    if check(mode, &addr_a, &mut hex_buf) {
                        attempts.fetch_add(local, Ordering::Relaxed);
                        stop.store(true, Ordering::Relaxed);
                        return Some((make_salt(&salt_prefix, cnt[s][0]), addr_a));
                    }
                    if check(mode, &addr_b, &mut hex_buf) {
                        attempts.fetch_add(local, Ordering::Relaxed);
                        stop.store(true, Ordering::Relaxed);
                        return Some((make_salt(&salt_prefix, cnt[s][1]), addr_b));
                    }
                }
                i += (2 * W) as u64;
            }
            attempts.fetch_add(local, Ordering::Relaxed);
            local = 0;
            if stop.load(Ordering::Relaxed) {
                return None;
            }
        }
    }

    pub fn run() {
        let config = parse_cli("create3-miner-neon");
        let use_sha3 =
            !config.force_base_keccak && std::arch::is_aarch64_feature_detected!("sha3");
        eprintln!(
            "Backend:  {} | buffers: {}",
            if use_sha3 { "NEON+SHA3" } else { "NEON" },
            config.neon_buffers
        );

        let factory = config.factory;
        let code_hash = config.code_hash;
        let mode = config.mode.clone();

        macro_rules! launch {
            ($w:literal) => {
                run_search(&config, move |stop: &AtomicBool, attempts: &AtomicU64| unsafe {
                    mine::<$w>(use_sha3, &factory, &code_hash, &mode, stop, attempts)
                })
            };
        }
        match config.neon_buffers {
            1 => launch!(1),
            2 => launch!(2),
            3 => launch!(3),
            _ => launch!(4),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use create3_miner::{create3_address, keccak256, parse_address, DEFAULT_PROXY_CODE_HASH};

        fn sha3_available() -> bool {
            std::arch::is_aarch64_feature_detected!("sha3")
        }

        /// Single-block keccak256 over one input, on the chosen backend.
        unsafe fn keccak256_single(input: &[u8], use_sha3: bool) -> [u8; 32] {
            let block = build_block(input);
            let mut st = [[vdupq_n_u64(0); 25]; 1];
            for w in 0..17 {
                let word = u64::from_le_bytes(block[w * 8..w * 8 + 8].try_into().unwrap());
                st[0][w] = vdupq_n_u64(word);
            }
            if use_sha3 {
                keccak_perm_sha3::<1>(&mut st);
            } else {
                keccak_perm_base::<1>(&mut st);
            }
            let (a, _) = extract_hash(&st[0]);
            a
        }

        fn check_backend(use_sha3: bool) {
            // keccak256("") known vector.
            let empty = unsafe { keccak256_single(b"", use_sha3) };
            assert_eq!(
                hex::encode(empty),
                "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
            );

            for input in [
                &b""[..],
                b"x",
                b"hello world",
                b"the quick brown fox jumps over the lazy dog",
            ] {
                assert_eq!(unsafe { keccak256_single(input, use_sha3) }, keccak256(input));
            }

            // Exact first- and second-stage lengths used by the miner.
            let mut buf85 = [0u8; 85];
            for (i, b) in buf85.iter_mut().enumerate() {
                *b = i as u8;
            }
            assert_eq!(unsafe { keccak256_single(&buf85, use_sha3) }, keccak256(&buf85));

            let mut buf23 = [0u8; 23];
            for (i, b) in buf23.iter_mut().enumerate() {
                *b = (200 - i) as u8;
            }
            assert_eq!(unsafe { keccak256_single(&buf23, use_sha3) }, keccak256(&buf23));
        }

        #[test]
        fn base_keccak_matches_scalar() {
            check_backend(false);
        }

        #[test]
        fn sha3_keccak_matches_scalar() {
            if sha3_available() {
                check_backend(true);
            }
        }

        /// Replicates the miner's per-state derivation for one state (both lanes
        /// fed the same counter) and compares to the scalar reference.
        unsafe fn derive_addr(
            factory: &[u8; 20],
            prefix: &[u8; 24],
            code_hash: &[u8; 32],
            c: u64,
            use_sha3: bool,
        ) -> [u8; 20] {
            let mut tmpl = [0u8; 85];
            tmpl[0] = 0xff;
            tmpl[1..21].copy_from_slice(factory);
            tmpl[21..45].copy_from_slice(prefix);
            tmpl[53..85].copy_from_slice(code_hash);
            let mut cw_base = [0u8; 16];
            cw_base.copy_from_slice(&tmpl[40..56]);

            let block = build_block(&tmpl);
            let mut st = [[vdupq_n_u64(0); 25]; 1];
            for w in 0..17 {
                let word = u64::from_le_bytes(block[w * 8..w * 8 + 8].try_into().unwrap());
                st[0][w] = vdupq_n_u64(word);
            }
            let (w5, w6) = counter_words(&cw_base, c);
            let p5 = [w5, w5];
            let p6 = [w6, w6];
            st[0][5] = vld1q_u64(p5.as_ptr());
            st[0][6] = vld1q_u64(p6.as_ptr());
            if use_sha3 {
                keccak_perm_sha3::<1>(&mut st);
            } else {
                keccak_perm_base::<1>(&mut st);
            }
            let (pa, _) = extract_hash(&st[0]);

            let mut st2 = [[vdupq_n_u64(0); 25]; 1];
            st2[0][16] = vdupq_n_u64(0x80u64 << 56);
            let (w0, w1, w2) = stage2_words(&pa);
            let q0 = [w0, w0];
            let q1 = [w1, w1];
            let q2 = [w2, w2];
            st2[0][0] = vld1q_u64(q0.as_ptr());
            st2[0][1] = vld1q_u64(q1.as_ptr());
            st2[0][2] = vld1q_u64(q2.as_ptr());
            if use_sha3 {
                keccak_perm_sha3::<1>(&mut st2);
            } else {
                keccak_perm_base::<1>(&mut st2);
            }
            let (ha, _) = extract_hash(&st2[0]);
            addr_from_hash(&ha)
        }

        fn check_derivation(use_sha3: bool) {
            let factory = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
            let prefix = [0x11u8; 24];
            for c in [0u64, 1, 42, 0x0102030405060708, u64::MAX - 1] {
                let got = unsafe { derive_addr(&factory, &prefix, &DEFAULT_PROXY_CODE_HASH, c, use_sha3) };
                let salt = make_salt(&prefix, c);
                let want = create3_address(&factory, &salt, &DEFAULT_PROXY_CODE_HASH);
                assert_eq!(got, want, "counter {c}");
            }

            // Custom code hash path.
            let other = [0xABu8; 32];
            let got = unsafe { derive_addr(&factory, &prefix, &other, 7, use_sha3) };
            let want = create3_address(&factory, &make_salt(&prefix, 7), &other);
            assert_eq!(got, want);
        }

        #[test]
        fn base_derivation_matches_scalar() {
            check_derivation(false);
        }

        #[test]
        fn sha3_derivation_matches_scalar() {
            if sha3_available() {
                check_derivation(true);
            }
        }

        /// Verifies the multi-state (W>1) loop indexes states independently.
        #[test]
        fn multibuffer_independent_states() {
            let inputs: [&[u8]; 4] = [
                b"alpha",
                b"bravo input two",
                b"charlie input number three padded",
                b"",
            ];
            unsafe {
                let mut st = [[vdupq_n_u64(0); 25]; 2];
                // state 0 -> inputs[0]/inputs[1], state 1 -> inputs[2]/inputs[3]
                for (s, pair) in [(0usize, (0usize, 1usize)), (1, (2, 3))] {
                    let ba = build_block(inputs[pair.0]);
                    let bb = build_block(inputs[pair.1]);
                    for w in 0..17 {
                        let wa = u64::from_le_bytes(ba[w * 8..w * 8 + 8].try_into().unwrap());
                        let wb = u64::from_le_bytes(bb[w * 8..w * 8 + 8].try_into().unwrap());
                        let p = [wa, wb];
                        st[s][w] = vld1q_u64(p.as_ptr());
                    }
                }
                keccak_perm_base::<2>(&mut st);
                let (h0a, h0b) = extract_hash(&st[0]);
                let (h1a, h1b) = extract_hash(&st[1]);
                assert_eq!(h0a, keccak256(inputs[0]));
                assert_eq!(h0b, keccak256(inputs[1]));
                assert_eq!(h1a, keccak256(inputs[2]));
                assert_eq!(h1b, keccak256(inputs[3]));
            }
        }
    }
}
