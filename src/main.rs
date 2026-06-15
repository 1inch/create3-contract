use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Arg, Command};
use rand::Rng;
use regex::Regex;
use tiny_keccak::{Hasher, Keccak};

/// keccak256 of the CREATE3 proxy child bytecode 0x68363d3d37363d34f0ff3d5260096017f3
const PROXY_CODE_HASH: [u8; 32] = [
    0x8d, 0x04, 0xf2, 0x96, 0xf4, 0x49, 0xa1, 0xe7, 0x95, 0xad, 0x35, 0xf2, 0x7e, 0x6b, 0x1d,
    0x09, 0xaf, 0x5a, 0x24, 0x22, 0xfa, 0x13, 0x7f, 0x3d, 0x6c, 0xbf, 0x52, 0xd7, 0xa9, 0x20,
    0x97, 0x5c,
];

fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak::v256();
    let mut out = [0u8; 32];
    hasher.update(data);
    hasher.finalize(&mut out);
    out
}

/// Computes the CREATE3 address for a given factory and salt:
///   proxy = keccak256(0xff ++ factory ++ salt ++ PROXY_CODE_HASH)[12..]
///   addr  = keccak256(0xd6 ++ 0x94 ++ proxy ++ 0x01)[12..]
///
/// Reference implementation; the mining loop inlines this with reused buffers.
#[cfg(test)]
fn create3_address(factory: &[u8; 20], salt: &[u8; 32]) -> [u8; 20] {
    let mut buf = [0u8; 85];
    buf[0] = 0xff;
    buf[1..21].copy_from_slice(factory);
    buf[21..53].copy_from_slice(salt);
    buf[53..85].copy_from_slice(&PROXY_CODE_HASH);
    let proxy_hash = keccak256(&buf);

    let mut buf2 = [0u8; 23];
    buf2[0] = 0xd6;
    buf2[1] = 0x94;
    buf2[2..22].copy_from_slice(&proxy_hash[12..32]);
    buf2[22] = 0x01;
    let addr_hash = keccak256(&buf2);

    let mut addr = [0u8; 20];
    addr.copy_from_slice(&addr_hash[12..32]);
    addr
}

/// EIP-55 checksummed representation of an address.
fn to_checksum_address(addr: &[u8; 20]) -> String {
    let lower = hex::encode(addr);
    let hash = keccak256(lower.as_bytes());
    let mut out = String::with_capacity(42);
    out.push_str("0x");
    for (i, c) in lower.chars().enumerate() {
        let nibble = (hash[i / 2] >> (if i % 2 == 0 { 4 } else { 0 })) & 0x0f;
        if c.is_ascii_alphabetic() && nibble >= 8 {
            out.push(c.to_ascii_uppercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn parse_address(s: &str) -> Result<[u8; 20], String> {
    let stripped = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    if stripped.len() != 40 {
        return Err(format!("expected 20-byte hex address, got {} hex chars", stripped.len()));
    }
    let bytes = hex::decode(stripped).map_err(|e| format!("invalid hex: {e}"))?;
    let mut out = [0u8; 20];
    out.copy_from_slice(&bytes);
    Ok(out)
}

const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

/// Hex-encodes a 20-byte address into a 40-byte ASCII buffer (no allocation).
#[inline(always)]
fn hex_encode_addr(addr: &[u8; 20], out: &mut [u8; 40]) {
    for (i, b) in addr.iter().enumerate() {
        out[i * 2] = HEX_CHARS[(b >> 4) as usize];
        out[i * 2 + 1] = HEX_CHARS[(b & 0x0f) as usize];
    }
}

fn main() {
    let matches = Command::new("create3-miner")
        .about("Brute-forces a CREATE3 salt so the deployed address matches a regex pattern")
        .arg(
            Arg::new("factory")
                .required(true)
                .help("CREATE3 factory address (0x...)"),
        )
        .arg(Arg::new("pattern").required(true).help(
            "Regex matched against the 40-char lowercase hex address (no 0x). \
             Examples: '^dead' (prefix), 'beef$' (suffix), 'c0ffee' (anywhere), '^0{8}'",
        ))
        .get_matches();

    let factory = match parse_address(matches.get_one::<String>("factory").unwrap()) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: invalid factory address: {e}");
            std::process::exit(1);
        }
    };

    let pattern_str = matches.get_one::<String>("pattern").unwrap();
    let regex = match Regex::new(&format!("(?i){pattern_str}")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: invalid regex pattern: {e}");
            std::process::exit(1);
        }
    };

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("Factory:  {}", to_checksum_address(&factory));
    println!("Pattern:  {pattern_str} (case-insensitive)");
    println!("Threads:  {threads}");
    println!("Mining...");

    let stop = Arc::new(AtomicBool::new(false));
    let attempts = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let stop = Arc::clone(&stop);
        let attempts = Arc::clone(&attempts);
        let regex = regex.clone();
        handles.push(std::thread::spawn(move || -> Option<([u8; 32], [u8; 20], u64)> {
            let mut salt = [0u8; 32];
            rand::rng().fill_bytes(&mut salt);

            // Pre-build the first hash buffer; only the salt portion changes per iteration.
            let mut buf = [0u8; 85];
            buf[0] = 0xff;
            buf[1..21].copy_from_slice(&factory);
            buf[53..85].copy_from_slice(&PROXY_CODE_HASH);

            let mut buf2 = [0u8; 23];
            buf2[0] = 0xd6;
            buf2[1] = 0x94;
            buf2[22] = 0x01;

            let mut hex_buf = [0u8; 40];
            let mut local: u64 = 0;
            const BATCH: u64 = 8192;

            loop {
                for _ in 0..BATCH {
                    // Increment salt (treat last 8 bytes as a big-endian counter).
                    for i in (24..32).rev() {
                        salt[i] = salt[i].wrapping_add(1);
                        if salt[i] != 0 {
                            break;
                        }
                    }

                    buf[21..53].copy_from_slice(&salt);
                    let mut hasher = Keccak::v256();
                    let mut proxy_hash = [0u8; 32];
                    hasher.update(&buf);
                    hasher.finalize(&mut proxy_hash);

                    buf2[2..22].copy_from_slice(&proxy_hash[12..32]);
                    let mut hasher = Keccak::v256();
                    let mut addr_hash = [0u8; 32];
                    hasher.update(&buf2);
                    hasher.finalize(&mut addr_hash);

                    let mut addr = [0u8; 20];
                    addr.copy_from_slice(&addr_hash[12..32]);
                    hex_encode_addr(&addr, &mut hex_buf);
                    local += 1;

                    // hex_buf is always valid ASCII
                    let hex_str = unsafe { std::str::from_utf8_unchecked(&hex_buf) };
                    if regex.is_match(hex_str) {
                        attempts.fetch_add(local, Ordering::Relaxed);
                        stop.store(true, Ordering::Relaxed);
                        return Some((salt, addr, local));
                    }
                }
                attempts.fetch_add(BATCH, Ordering::Relaxed);
                local = 0;
                if stop.load(Ordering::Relaxed) {
                    return None;
                }
            }
        }));
    }

    // Progress reporter
    {
        let stop = Arc::clone(&stop);
        let attempts = Arc::clone(&attempts);
        std::thread::spawn(move || {
            let mut last_count = 0u64;
            let mut last_time = Instant::now();
            loop {
                std::thread::sleep(Duration::from_secs(5));
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                let now = Instant::now();
                let count = attempts.load(Ordering::Relaxed);
                let rate = (count - last_count) as f64 / now.duration_since(last_time).as_secs_f64();
                eprintln!(
                    "  {:>14} attempts | {:.2} MH/s",
                    count,
                    rate / 1_000_000.0
                );
                last_count = count;
                last_time = now;
            }
        });
    }

    let mut result = None;
    for handle in handles {
        if let Some(found) = handle.join().expect("worker thread panicked") {
            result = Some(found);
        }
    }

    let (salt, addr, _) = result.expect("no result despite stop signal");
    let total = attempts.load(Ordering::Relaxed);
    let elapsed = start.elapsed();

    println!();
    println!("Found a match!");
    println!("Salt:     0x{}", hex::encode(salt));
    println!("Address:  {}", to_checksum_address(&addr));
    println!(
        "Attempts: {} in {:.1}s ({:.2} MH/s)",
        total,
        elapsed.as_secs_f64(),
        total as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_code_hash_matches_bytecode() {
        let proxy_bytecode = hex::decode("68363d3d37363d34f0ff3d5260096017f3").unwrap();
        assert_eq!(keccak256(&proxy_bytecode), PROXY_CODE_HASH);
    }

    /// Vector computed independently with foundry:
    ///   cast create2 --deployer 0x9fBB...0ABf --salt 0x00..00 \
    ///     --init-code-hash 0x8d04f296... => proxy 0xA68F3B2839b031e7624C3F3a9e1Fc6843810c236
    ///   cast keccak 0xd694a68f3b2839b031e7624c3f3a9e1fc6843810c23601 => 0x7f35ba6cce28fdd976c66589f2e109a6fb69ad27
    #[test]
    fn create3_address_known_vector() {
        let factory = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
        let salt = [0u8; 32];
        let expected = parse_address("0x7f35ba6cce28fdd976c66589f2e109a6fb69ad27").unwrap();
        assert_eq!(create3_address(&factory, &salt), expected);
    }

    #[test]
    fn checksum_address() {
        let addr = parse_address("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed").unwrap();
        assert_eq!(
            to_checksum_address(&addr),
            "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
        );
    }

    #[test]
    fn hex_encode_matches_hex_crate() {
        let addr = parse_address("0xdeadbeef00112233445566778899aabbccddeeff").unwrap();
        let mut buf = [0u8; 40];
        hex_encode_addr(&addr, &mut buf);
        assert_eq!(std::str::from_utf8(&buf).unwrap(), hex::encode(addr));
    }
}
