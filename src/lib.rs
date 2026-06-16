use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Arg, ArgGroup, Command};
use rand::Rng;
use regex::Regex;
use sha3::{Digest, Keccak256};

/// keccak256 of the default CREATE3 proxy child bytecode
/// `0x68363d3d37363d34f0ff3d5260096017f3`. Used unless overridden via the
/// `-b`/`-B` flags.
pub const DEFAULT_PROXY_CODE_HASH: [u8; 32] = [
    0x8d, 0x04, 0xf2, 0x96, 0xf4, 0x49, 0xa1, 0xe7, 0x95, 0xad, 0x35, 0xf2, 0x7e, 0x6b, 0x1d,
    0x09, 0xaf, 0x5a, 0x24, 0x22, 0xfa, 0x13, 0x7f, 0x3d, 0x6c, 0xbf, 0x52, 0xd7, 0xa9, 0x20,
    0x97, 0x5c,
];

/// Computes Ethereum keccak256 (the original `0x01`-padded Keccak, not NIST
/// SHA3-256) of `data`.
///
/// Convenience wrapper for one-off hashing (tests, checksums, startup). The
/// mining hot loops reuse hashers / batch instead.
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    Digest::update(&mut hasher, data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Computes the CREATE3 address for a given factory, salt and proxy code hash:
///   proxy = keccak256(0xff ++ factory ++ salt ++ code_hash)[12..]
///   addr  = keccak256(0xd6 ++ 0x94 ++ proxy ++ 0x01)[12..]
///
/// Reference (scalar) implementation; the mining loops inline this.
pub fn create3_address(factory: &[u8; 20], salt: &[u8; 32], code_hash: &[u8; 32]) -> [u8; 20] {
    let mut buf = [0u8; 85];
    buf[0] = 0xff;
    buf[1..21].copy_from_slice(factory);
    buf[21..53].copy_from_slice(salt);
    buf[53..85].copy_from_slice(code_hash);
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
pub fn to_checksum_address(addr: &[u8; 20]) -> String {
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

fn strip_hex_prefix(s: &str) -> &str {
    s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s)
}

pub fn parse_address(s: &str) -> Result<[u8; 20], String> {
    let stripped = strip_hex_prefix(s);
    if stripped.len() != 40 {
        return Err(format!("expected 20-byte hex address, got {} hex chars", stripped.len()));
    }
    let bytes = hex::decode(stripped).map_err(|e| format!("invalid hex: {e}"))?;
    let mut out = [0u8; 20];
    out.copy_from_slice(&bytes);
    Ok(out)
}

/// Parses a 32-byte proxy code hash (64 hex chars, optional `0x`).
pub fn parse_bytecode_hash(s: &str) -> Result<[u8; 32], String> {
    let stripped = strip_hex_prefix(s);
    if stripped.len() != 64 {
        return Err(format!(
            "expected 32-byte hash (64 hex chars), got {} hex chars",
            stripped.len()
        ));
    }
    let bytes = hex::decode(stripped).map_err(|e| format!("invalid hex: {e}"))?;
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

/// Parses raw proxy bytecode (hex, optional `0x`) and returns its keccak256.
pub fn parse_bytecode(s: &str) -> Result<[u8; 32], String> {
    let stripped = strip_hex_prefix(s);
    if stripped.is_empty() {
        return Err("bytecode must not be empty".to_string());
    }
    if stripped.len() % 2 != 0 {
        return Err(format!("bytecode must have an even number of hex chars, got {}", stripped.len()));
    }
    let bytes = hex::decode(stripped).map_err(|e| format!("invalid hex: {e}"))?;
    Ok(keccak256(&bytes))
}

pub const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

/// Hex-encodes a 20-byte address into a 40-byte ASCII buffer (no allocation).
#[inline(always)]
pub fn hex_encode_addr(addr: &[u8; 20], out: &mut [u8; 40]) {
    for (i, b) in addr.iter().enumerate() {
        out[i * 2] = HEX_CHARS[(b >> 4) as usize];
        out[i * 2 + 1] = HEX_CHARS[(b & 0x0f) as usize];
    }
}

/// How a candidate address is tested for a match.
#[derive(Clone)]
pub enum MatchMode {
    /// Exact hex prefix as a list of nibbles (0..=15). Compared directly on the
    /// address bytes, skipping hex encoding and the regex engine.
    Leading(Vec<u8>),
    /// Arbitrary case-insensitive regex over the lowercase hex address.
    Regex(Regex),
}

/// Parses a hex prefix (optionally `0x`-prefixed) into a list of nibbles.
pub fn parse_leading(s: &str) -> Result<Vec<u8>, String> {
    let stripped = strip_hex_prefix(s);
    if stripped.is_empty() {
        return Err("leading pattern must not be empty".to_string());
    }
    if stripped.len() > 40 {
        return Err(format!(
            "leading pattern too long: {} hex chars (max 40)",
            stripped.len()
        ));
    }
    let mut nibbles = Vec::with_capacity(stripped.len());
    for c in stripped.chars() {
        match c.to_digit(16) {
            Some(d) => nibbles.push(d as u8),
            None => return Err(format!("invalid hex character '{c}' in leading pattern")),
        }
    }
    Ok(nibbles)
}

/// Returns true if the address starts with the given nibble prefix.
#[inline(always)]
pub fn leading_match(addr: &[u8; 20], nibbles: &[u8]) -> bool {
    for (i, &want) in nibbles.iter().enumerate() {
        let byte = addr[i / 2];
        let got = if i % 2 == 0 { byte >> 4 } else { byte & 0x0f };
        if got != want {
            return false;
        }
    }
    true
}

/// Per-worker scalar mining state with preallocated buffers and a single reused
/// `Keccak256` hasher.
pub struct MiningContext {
    pub salt: [u8; 32],
    buf: [u8; 85],
    buf2: [u8; 23],
    hasher: Keccak256,
    pub hex_buf: [u8; 40],
}

impl MiningContext {
    pub fn new(factory: &[u8; 20], code_hash: &[u8; 32]) -> Self {
        let mut salt = [0u8; 32];
        rand::rng().fill_bytes(&mut salt);

        let mut buf = [0u8; 85];
        buf[0] = 0xff;
        buf[1..21].copy_from_slice(factory);
        buf[53..85].copy_from_slice(code_hash);

        let mut buf2 = [0u8; 23];
        buf2[0] = 0xd6;
        buf2[1] = 0x94;
        buf2[22] = 0x01;

        Self {
            salt,
            buf,
            buf2,
            hasher: Keccak256::new(),
            hex_buf: [0u8; 40],
        }
    }

    /// Advances the salt counter and derives the next CREATE3 address.
    #[inline(always)]
    pub fn next_addr(&mut self) -> [u8; 20] {
        // Increment salt (treat last 8 bytes as a big-endian counter).
        for i in (24..32).rev() {
            self.salt[i] = self.salt[i].wrapping_add(1);
            if self.salt[i] != 0 {
                break;
            }
        }

        self.buf[21..53].copy_from_slice(&self.salt);
        Digest::update(&mut self.hasher, &self.buf);
        let proxy_hash = self.hasher.finalize_reset();

        self.buf2[2..22].copy_from_slice(&proxy_hash[12..32]);
        Digest::update(&mut self.hasher, &self.buf2);
        let addr_hash = self.hasher.finalize_reset();

        let mut addr = [0u8; 20];
        addr.copy_from_slice(&addr_hash[12..32]);
        addr
    }
}

/// Resolved command-line configuration shared by both miners.
pub struct Config {
    pub factory: [u8; 20],
    pub mode: MatchMode,
    pub mode_desc: String,
    pub threads: usize,
    pub code_hash: [u8; 32],
}

/// Builds the CLI, parses args, and resolves them into a [`Config`].
/// Exits the process with a clear message on any invalid input.
pub fn parse_cli(bin_name: &'static str) -> Config {
    let matches = Command::new(bin_name)
        .about("Brute-forces a CREATE3 salt so the deployed address matches a pattern")
        .arg(
            Arg::new("factory")
                .required(true)
                .help("CREATE3 factory address (0x...)"),
        )
        .arg(Arg::new("pattern").help(
            "Regex matched against the 40-char lowercase hex address (no 0x). \
             Examples: '^dead' (prefix), 'beef$' (suffix), 'c0ffee' (anywhere), '^0{8}'",
        ))
        .arg(
            Arg::new("leading")
                .long("leading")
                .value_name("HEX")
                .help(
                    "Fast-path exact hex prefix to match at the start of the address \
                     (case-insensitive), e.g. '000000000' or 'dead'",
                ),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .value_name("N")
                .help("Number of worker threads (default: all available cores)"),
        )
        .arg(
            Arg::new("bytecode")
                .short('b')
                .long("bytecode")
                .value_name("HEX")
                .help(
                    "Proxy child bytecode (hex, 0x optional); its keccak256 is used as the \
                     proxy code hash. Mutually exclusive with --bytecode-hash",
                ),
        )
        .arg(
            Arg::new("bytecode-hash")
                .short('B')
                .long("bytecode-hash")
                .value_name("HEX")
                .help(
                    "Proxy code hash directly (32-byte hex, 0x optional). \
                     Mutually exclusive with --bytecode",
                ),
        )
        .group(
            ArgGroup::new("matcher")
                .args(["pattern", "leading"])
                .required(true),
        )
        .group(
            ArgGroup::new("proxy")
                .args(["bytecode", "bytecode-hash"])
                .required(false),
        )
        .get_matches();

    let factory = match parse_address(matches.get_one::<String>("factory").unwrap()) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: invalid factory address: {e}");
            std::process::exit(1);
        }
    };

    let (mode, mode_desc) = if let Some(leading) = matches.get_one::<String>("leading") {
        match parse_leading(leading) {
            Ok(nibbles) => {
                let desc = format!("Leading:  {} ({} hex chars)", leading, nibbles.len());
                (MatchMode::Leading(nibbles), desc)
            }
            Err(e) => {
                eprintln!("error: invalid leading pattern: {e}");
                std::process::exit(1);
            }
        }
    } else {
        let pattern_str = matches.get_one::<String>("pattern").unwrap();
        match Regex::new(&format!("(?i){pattern_str}")) {
            Ok(r) => (
                MatchMode::Regex(r),
                format!("Pattern:  {pattern_str} (regex, case-insensitive)"),
            ),
            Err(e) => {
                eprintln!("error: invalid regex pattern: {e}");
                std::process::exit(1);
            }
        }
    };

    let threads = match matches.get_one::<String>("threads") {
        Some(s) => match s.parse::<usize>() {
            Ok(n) if n >= 1 => n,
            _ => {
                eprintln!("error: --threads must be a positive integer");
                std::process::exit(1);
            }
        },
        None => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    };

    let code_hash = if let Some(bc) = matches.get_one::<String>("bytecode") {
        match parse_bytecode(bc) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("error: invalid bytecode: {e}");
                std::process::exit(1);
            }
        }
    } else if let Some(bh) = matches.get_one::<String>("bytecode-hash") {
        match parse_bytecode_hash(bh) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("error: invalid bytecode-hash: {e}");
                std::process::exit(1);
            }
        }
    } else {
        DEFAULT_PROXY_CODE_HASH
    };

    Config {
        factory,
        mode,
        mode_desc,
        threads,
        code_hash,
    }
}

/// Orchestrates the multi-threaded search: prints the header, spawns workers,
/// runs the progress reporter, and prints the final result.
///
/// `worker` runs one thread's full mining loop. It must periodically add to the
/// attempts counter, return `Some((salt, addr))` on a match (after setting the
/// stop flag), and return `None` once the stop flag is observed.
pub fn run_search<F>(config: &Config, worker: F)
where
    F: Fn(&AtomicBool, &AtomicU64) -> Option<([u8; 32], [u8; 20])> + Send + Sync + 'static,
{
    println!("Factory:  {}", to_checksum_address(&config.factory));
    println!("{}", config.mode_desc);
    println!("Threads:  {}", config.threads);
    if config.code_hash != DEFAULT_PROXY_CODE_HASH {
        println!("Code hash: 0x{}", hex::encode(config.code_hash));
    }
    println!("Mining...");

    let stop = Arc::new(AtomicBool::new(false));
    let attempts = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let worker = Arc::new(worker);

    let mut handles = Vec::with_capacity(config.threads);
    for _ in 0..config.threads {
        let stop = Arc::clone(&stop);
        let attempts = Arc::clone(&attempts);
        let worker = Arc::clone(&worker);
        handles.push(std::thread::spawn(move || worker(&stop, &attempts)));
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
                eprintln!("  {count:>14} attempts | {:.2} MH/s", rate / 1_000_000.0);
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

    let (salt, addr) = result.expect("no result despite stop signal");
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
        assert_eq!(keccak256(&proxy_bytecode), DEFAULT_PROXY_CODE_HASH);
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
        assert_eq!(create3_address(&factory, &salt, &DEFAULT_PROXY_CODE_HASH), expected);
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

    #[test]
    fn parse_leading_accepts_valid() {
        assert_eq!(parse_leading("dead").unwrap(), vec![13, 14, 10, 13]);
        assert_eq!(parse_leading("DEAD").unwrap(), vec![13, 14, 10, 13]);
        assert_eq!(parse_leading("0xdead").unwrap(), vec![13, 14, 10, 13]);
        assert_eq!(parse_leading("000000000").unwrap(), vec![0u8; 9]);
        assert_eq!(parse_leading(&"f".repeat(40)).unwrap().len(), 40);
    }

    #[test]
    fn parse_leading_rejects_invalid() {
        assert!(parse_leading("").is_err());
        assert!(parse_leading("0x").is_err());
        assert!(parse_leading("xyz").is_err());
        assert!(parse_leading("dead!").is_err());
        assert!(parse_leading(&"0".repeat(41)).is_err());
    }

    #[test]
    fn leading_match_prefixes() {
        let addr = parse_address("0xdeadbeef00112233445566778899aabbccddeeff").unwrap();
        assert!(leading_match(&addr, &parse_leading("dead").unwrap()));
        assert!(leading_match(&addr, &parse_leading("deadbeef").unwrap()));
        assert!(!leading_match(&addr, &parse_leading("beef").unwrap()));
        assert!(!leading_match(&addr, &parse_leading("deae").unwrap()));
    }

    #[test]
    fn leading_match_nibble_boundary() {
        // Bytes: 00 00 00 00 05 ... so 9 leading zero nibbles, then nibble 9 is 5.
        let addr = parse_address("0x0000000005112233445566778899aabbccddeeff").unwrap();
        assert!(leading_match(&addr, &parse_leading("000000000").unwrap()));
        assert!(!leading_match(&addr, &parse_leading("0000000000").unwrap()));
    }

    #[test]
    fn leading_match_agrees_with_regex() {
        let addr = parse_address("0xdead00000000000000000000000000000000beef").unwrap();
        let mut buf = [0u8; 40];
        hex_encode_addr(&addr, &mut buf);
        let hex_str = std::str::from_utf8(&buf).unwrap();
        let re = Regex::new("(?i)^dead").unwrap();
        assert_eq!(
            leading_match(&addr, &parse_leading("dead").unwrap()),
            re.is_match(hex_str)
        );
    }

    #[test]
    fn parse_bytecode_hash_accepts_both_formats() {
        let h = hex::encode(DEFAULT_PROXY_CODE_HASH);
        assert_eq!(parse_bytecode_hash(&h).unwrap(), DEFAULT_PROXY_CODE_HASH);
        assert_eq!(parse_bytecode_hash(&format!("0x{h}")).unwrap(), DEFAULT_PROXY_CODE_HASH);
        assert_eq!(parse_bytecode_hash(&format!("0X{h}")).unwrap(), DEFAULT_PROXY_CODE_HASH);
    }

    #[test]
    fn parse_bytecode_hash_rejects_invalid() {
        assert!(parse_bytecode_hash("").is_err());
        assert!(parse_bytecode_hash("abcd").is_err());
        assert!(parse_bytecode_hash(&"z".repeat(64)).is_err());
    }

    #[test]
    fn parse_bytecode_hashes_proxy_to_default() {
        // The real proxy bytecode hashes to the default code hash, with or without 0x.
        assert_eq!(
            parse_bytecode("68363d3d37363d34f0ff3d5260096017f3").unwrap(),
            DEFAULT_PROXY_CODE_HASH
        );
        assert_eq!(
            parse_bytecode("0x68363d3d37363d34f0ff3d5260096017f3").unwrap(),
            DEFAULT_PROXY_CODE_HASH
        );
    }

    #[test]
    fn parse_bytecode_rejects_invalid() {
        assert!(parse_bytecode("").is_err());
        assert!(parse_bytecode("abc").is_err()); // odd length
        assert!(parse_bytecode("zz").is_err());
    }

    #[test]
    fn non_default_code_hash_changes_address() {
        let factory = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
        let salt = [7u8; 32];
        let other = [0xAAu8; 32];
        assert_ne!(
            create3_address(&factory, &salt, &DEFAULT_PROXY_CODE_HASH),
            create3_address(&factory, &salt, &other)
        );
    }
}
