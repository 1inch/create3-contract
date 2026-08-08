use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Arg, ArgGroup, Command};
use rand::Rng;
use regex::Regex;
use sha3::{Digest, Keccak256};

/// keccak256 of the default CREATE3 proxy child bytecode
/// `0x67363d3d37363d34f03d5260086018f3` (Solady/solmate variant). Used unless
/// overridden via the `-b`/`-B` flags.
pub const DEFAULT_PROXY_CODE_HASH: [u8; 32] = [
    0x21, 0xc3, 0x5d, 0xbe, 0x1b, 0x34, 0x4a, 0x24, 0x88, 0xcf, 0x33, 0x21, 0xd6, 0xce, 0x54,
    0x2f, 0x8e, 0x9f, 0x30, 0x55, 0x44, 0xff, 0x09, 0xe4, 0x99, 0x3a, 0x62, 0x31, 0x9a, 0x49,
    0x7c, 0x1f,
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
    /// Fixed-nibble pattern compared directly on the address bytes: matches
    /// iff `(addr[i] ^ value[i]) & mask[i] == 0` for every byte. Built from
    /// any combination of `--leading`, `--suffix` and `--mask`; skips hex
    /// encoding and the regex engine.
    Mask { value: [u8; 20], mask: [u8; 20] },
    /// Arbitrary case-insensitive regex over the lowercase hex address.
    Regex(Regex),
}

/// Parses a hex nibble string (optionally `0x`-prefixed) into a list of
/// nibble values (0..=15).
pub fn parse_hex_nibbles(s: &str) -> Result<Vec<u8>, String> {
    let stripped = strip_hex_prefix(s);
    if stripped.is_empty() {
        return Err("hex pattern must not be empty".to_string());
    }
    if stripped.len() > 40 {
        return Err(format!(
            "hex pattern too long: {} hex chars (max 40)",
            stripped.len()
        ));
    }
    let mut nibbles = Vec::with_capacity(stripped.len());
    for c in stripped.chars() {
        match c.to_digit(16) {
            Some(d) => nibbles.push(d as u8),
            None => return Err(format!("invalid hex character '{c}' in pattern")),
        }
    }
    Ok(nibbles)
}

/// Parses a hex prefix (optionally `0x`-prefixed) into a list of nibbles.
pub fn parse_leading(s: &str) -> Result<Vec<u8>, String> {
    parse_hex_nibbles(s)
}

/// Returns true if the address matches the fixed-nibble pattern (see
/// [`MatchMode::Mask`]).
#[inline(always)]
pub fn mask_match(addr: &[u8; 20], value: &[u8; 20], mask: &[u8; 20]) -> bool {
    for i in 0..20 {
        if (addr[i] ^ value[i]) & mask[i] != 0 {
            return false;
        }
    }
    true
}

/// Accumulates fixed nibbles from `--leading` / `--suffix` / `--mask` into a
/// single (value, mask) pair over the 20-byte address. Nibble position 0 is
/// the first hex character of the address; a mask half-byte is 0xF where the
/// nibble is constrained.
#[derive(Default, Clone)]
pub struct AddressPattern {
    pub value: [u8; 20],
    pub mask: [u8; 20],
}

impl AddressPattern {
    fn set_nibble(&mut self, pos: usize, nibble: u8) -> Result<(), String> {
        debug_assert!(pos < 40 && nibble <= 0xf);
        let byte = pos / 2;
        let shift = if pos % 2 == 0 { 4 } else { 0 };
        if (self.mask[byte] >> shift) & 0x0f != 0 && (self.value[byte] >> shift) & 0x0f != nibble
        {
            return Err(format!(
                "conflicting constraints at address hex position {pos}"
            ));
        }
        self.mask[byte] |= 0x0f << shift;
        self.value[byte] |= nibble << shift;
        Ok(())
    }

    /// Constrains the first `nibbles.len()` hex chars of the address.
    pub fn add_leading(&mut self, nibbles: &[u8]) -> Result<(), String> {
        for (i, &n) in nibbles.iter().enumerate() {
            self.set_nibble(i, n)?;
        }
        Ok(())
    }

    /// Constrains the last `nibbles.len()` hex chars of the address.
    pub fn add_suffix(&mut self, nibbles: &[u8]) -> Result<(), String> {
        for (i, &n) in nibbles.iter().enumerate() {
            self.set_nibble(40 - nibbles.len() + i, n)?;
        }
        Ok(())
    }

    /// Applies a left-anchored template of hex chars and `.` wildcards over
    /// the 40-char hex address, e.g. `dead....beef`.
    pub fn add_template(&mut self, template: &str) -> Result<(), String> {
        if template.is_empty() {
            return Err("mask template must not be empty".to_string());
        }
        if template.len() > 40 {
            return Err(format!(
                "mask template too long: {} chars (max 40)",
                template.len()
            ));
        }
        for (i, c) in template.chars().enumerate() {
            if c == '.' {
                continue;
            }
            match c.to_digit(16) {
                Some(d) => self.set_nibble(i, d as u8)?,
                None => {
                    return Err(format!(
                        "invalid character '{c}' in mask template (expected hex or '.')"
                    ))
                }
            }
        }
        Ok(())
    }

    /// True if no nibble is constrained.
    pub fn is_empty(&self) -> bool {
        self.mask.iter().all(|&b| b == 0)
    }

    /// Number of constrained nibbles.
    pub fn fixed_nibbles(&self) -> u32 {
        self.mask
            .iter()
            .map(|&b| u32::from(b >> 4 != 0) + u32::from(b & 0x0f != 0))
            .sum()
    }

    /// Canonical 40-char template (hex for fixed nibbles, `.` for free ones).
    pub fn template_string(&self) -> String {
        (0..40)
            .map(|i| {
                let shift = if i % 2 == 0 { 4 } else { 0 };
                if (self.mask[i / 2] >> shift) & 0x0f != 0 {
                    char::from(HEX_CHARS[((self.value[i / 2] >> shift) & 0x0f) as usize])
                } else {
                    '.'
                }
            })
            .collect()
    }
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

/// Default multi-buffering width for the NEON miner (independent keccak states
/// processed per batch). Benchmarking on Apple Silicon shows 1 is fastest: the
/// 25-vector keccak state already spills past 32 NEON registers, so additional
/// buffers only add spill traffic. The flag is kept for experimentation.
pub const DEFAULT_NEON_BUFFERS: usize = 1;

/// Default number of attempts per GPU dispatch for the Metal miner
/// (~8 ms of work at ~500 MH/s: long enough to amortize dispatch overhead,
/// short enough to keep progress reporting and the GPU watchdog happy).
pub const DEFAULT_METAL_BATCH: usize = 1 << 22;

/// Resolved command-line configuration shared by all miners.
pub struct Config {
    pub factory: [u8; 20],
    pub mode: MatchMode,
    pub mode_desc: String,
    pub threads: usize,
    pub code_hash: [u8; 32],
    /// NEON only: number of independent keccak states per batch (1..=4).
    pub neon_buffers: usize,
    /// NEON only: force the base NEON path even if the SHA3 extension exists.
    pub force_base_keccak: bool,
    /// Metal only: attempts per GPU dispatch.
    pub batch: usize,
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
        .arg(
            Arg::new("pattern")
                .help(
                    "Regex matched against the 40-char lowercase hex address (no 0x). \
                     Examples: '^dead' (prefix), 'beef$' (suffix), 'c0ffee' (anywhere), '^0{8}'. \
                     CPU miners only; not supported by create3-miner-metal",
                )
                .conflicts_with_all(["leading", "suffix", "mask"]),
        )
        .arg(
            Arg::new("leading")
                .long("leading")
                .value_name("HEX")
                .help(
                    "Fast-path exact hex prefix to match at the start of the address \
                     (case-insensitive), e.g. '000000000' or 'dead'. \
                     Can be combined with --suffix and --mask",
                ),
        )
        .arg(
            Arg::new("suffix")
                .long("suffix")
                .value_name("HEX")
                .help(
                    "Fast-path exact hex suffix to match at the end of the address \
                     (case-insensitive), e.g. 'beef'. \
                     Can be combined with --leading and --mask",
                ),
        )
        .arg(
            Arg::new("mask")
                .long("mask")
                .value_name("TEMPLATE")
                .help(
                    "Fast-path left-anchored template of hex chars and '.' wildcards \
                     over the 40-char address (case-insensitive), e.g. 'dead....beef'. \
                     Can be combined with --leading and --suffix",
                ),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .value_name("N")
                .help("Number of worker threads (default: all available cores)"),
        )
        .arg(
            Arg::new("buffers")
                .long("buffers")
                .value_name("N")
                .help(
                    "NEON only: independent keccak states processed per batch (1-4). \
                     Higher can raise IPC at the cost of register pressure",
                ),
        )
        .arg(
            Arg::new("no-sha3")
                .long("no-sha3")
                .action(clap::ArgAction::SetTrue)
                .help("NEON only: disable the ARMv8 SHA3 extension path (for benchmarking)"),
        )
        .arg(
            Arg::new("batch")
                .long("batch")
                .value_name("N")
                .help("Metal only: attempts per GPU dispatch (default 4194304)"),
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
                .args(["pattern", "leading", "suffix", "mask"])
                .required(true)
                .multiple(true),
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

    let (mode, mode_desc) = if let Some(pattern_str) = matches.get_one::<String>("pattern") {
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
    } else {
        let mut pattern = AddressPattern::default();
        if let Some(leading) = matches.get_one::<String>("leading") {
            if let Err(e) = parse_hex_nibbles(leading).and_then(|n| pattern.add_leading(&n)) {
                eprintln!("error: invalid --leading: {e}");
                std::process::exit(1);
            }
        }
        if let Some(suffix) = matches.get_one::<String>("suffix") {
            if let Err(e) = parse_hex_nibbles(suffix).and_then(|n| pattern.add_suffix(&n)) {
                eprintln!("error: invalid --suffix: {e}");
                std::process::exit(1);
            }
        }
        if let Some(template) = matches.get_one::<String>("mask") {
            if let Err(e) = pattern.add_template(template) {
                eprintln!("error: invalid --mask: {e}");
                std::process::exit(1);
            }
        }
        // The matcher arg group guarantees at least one flag was given, and
        // every flag rejects empty input, so at least one nibble is fixed.
        let desc = format!(
            "Pattern:  {} ({} fixed nibbles)",
            pattern.template_string(),
            pattern.fixed_nibbles()
        );
        (
            MatchMode::Mask {
                value: pattern.value,
                mask: pattern.mask,
            },
            desc,
        )
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

    let neon_buffers = match matches.get_one::<String>("buffers") {
        Some(s) => match s.parse::<usize>() {
            Ok(n) if (1..=4).contains(&n) => n,
            _ => {
                eprintln!("error: --buffers must be an integer in 1..=4");
                std::process::exit(1);
            }
        },
        None => DEFAULT_NEON_BUFFERS,
    };

    let force_base_keccak = matches.get_flag("no-sha3");

    let batch = match matches.get_one::<String>("batch") {
        Some(s) => match s.parse::<usize>() {
            Ok(n) if (1..=(1usize << 26)).contains(&n) => n,
            _ => {
                eprintln!("error: --batch must be an integer in 1..=67108864");
                std::process::exit(1);
            }
        },
        None => DEFAULT_METAL_BATCH,
    };

    Config {
        factory,
        mode,
        mode_desc,
        threads,
        code_hash,
        neon_buffers,
        force_base_keccak,
        batch,
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
        let proxy_bytecode = hex::decode("67363d3d37363d34f03d5260086018f3").unwrap();
        assert_eq!(keccak256(&proxy_bytecode), DEFAULT_PROXY_CODE_HASH);
    }

    /// Vector computed independently with foundry:
    ///   cast create2 --deployer 0x9fBB...0ABf --salt 0x00..00 \
    ///     --init-code-hash 0x21c35dbe... => proxy 0x932A2198eC22043b9702a6250C8Ad906a3D62131
    ///   cast keccak 0xd694932a2198ec22043b9702a6250c8ad906a3d6213101 => 0x6c8ed9dc3734d7944beddd2fb5acdf5f17247870
    #[test]
    fn create3_address_known_vector() {
        let factory = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
        let salt = [0u8; 32];
        let expected = parse_address("0x6c8ed9dc3734d7944beddd2fb5acdf5f17247870").unwrap();
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

    fn leading_pattern(s: &str) -> AddressPattern {
        let mut p = AddressPattern::default();
        p.add_leading(&parse_leading(s).unwrap()).unwrap();
        p
    }

    fn matches(p: &AddressPattern, addr: &[u8; 20]) -> bool {
        mask_match(addr, &p.value, &p.mask)
    }

    #[test]
    fn mask_match_prefixes() {
        let addr = parse_address("0xdeadbeef00112233445566778899aabbccddeeff").unwrap();
        assert!(matches(&leading_pattern("dead"), &addr));
        assert!(matches(&leading_pattern("deadbeef"), &addr));
        assert!(!matches(&leading_pattern("beef"), &addr));
        assert!(!matches(&leading_pattern("deae"), &addr));
    }

    #[test]
    fn mask_match_nibble_boundary() {
        // Bytes: 00 00 00 00 05 ... so 9 leading zero nibbles, then nibble 9 is 5.
        let addr = parse_address("0x0000000005112233445566778899aabbccddeeff").unwrap();
        assert!(matches(&leading_pattern("000000000"), &addr));
        assert!(!matches(&leading_pattern("0000000000"), &addr));
    }

    #[test]
    fn mask_match_agrees_with_regex() {
        let addr = parse_address("0xdead00000000000000000000000000000000beef").unwrap();
        let mut buf = [0u8; 40];
        hex_encode_addr(&addr, &mut buf);
        let hex_str = std::str::from_utf8(&buf).unwrap();
        let re = Regex::new("(?i)^dead").unwrap();
        assert_eq!(matches(&leading_pattern("dead"), &addr), re.is_match(hex_str));
    }

    #[test]
    fn suffix_pattern_matches_end() {
        let addr = parse_address("0xdead00000000000000000000000000000000beef").unwrap();
        let mut p = AddressPattern::default();
        p.add_suffix(&parse_hex_nibbles("beef").unwrap()).unwrap();
        assert!(matches(&p, &addr));

        // Odd-length suffix lands on the last nibble, not a byte boundary.
        let mut p = AddressPattern::default();
        p.add_suffix(&parse_hex_nibbles("f").unwrap()).unwrap();
        assert!(matches(&p, &addr));

        let mut p = AddressPattern::default();
        p.add_suffix(&parse_hex_nibbles("beee").unwrap()).unwrap();
        assert!(!matches(&p, &addr));
    }

    #[test]
    fn template_pattern() {
        let addr = parse_address("0xdead1234beef0000000000000000000000000000").unwrap();
        let mut p = AddressPattern::default();
        p.add_template("dead....beef").unwrap();
        assert!(matches(&p, &addr));

        let mut p = AddressPattern::default();
        p.add_template("dead....beee").unwrap();
        assert!(!matches(&p, &addr));

        assert!(AddressPattern::default().add_template("").is_err());
        assert!(AddressPattern::default().add_template(&"0".repeat(41)).is_err());
        assert!(AddressPattern::default().add_template("de.z").is_err());
    }

    #[test]
    fn combined_flags_and_conflicts() {
        // leading + suffix + overlapping-but-consistent template combine fine.
        let addr = parse_address("0xdead00000000000000000000000000000000beef").unwrap();
        let mut p = AddressPattern::default();
        p.add_leading(&parse_hex_nibbles("dead").unwrap()).unwrap();
        p.add_suffix(&parse_hex_nibbles("beef").unwrap()).unwrap();
        p.add_template("de.d").unwrap();
        assert!(matches(&p, &addr));
        assert_eq!(p.fixed_nibbles(), 8);

        // Conflicting constraint on the same nibble is rejected.
        let mut p = AddressPattern::default();
        p.add_leading(&parse_hex_nibbles("dead").unwrap()).unwrap();
        assert!(p.add_template("beef").is_err());
    }

    #[test]
    fn template_string_canonical() {
        let mut p = AddressPattern::default();
        p.add_leading(&parse_hex_nibbles("dead").unwrap()).unwrap();
        p.add_suffix(&parse_hex_nibbles("beef").unwrap()).unwrap();
        assert_eq!(p.template_string(), format!("dead{}beef", ".".repeat(32)));
        assert!(AddressPattern::default().is_empty());
        assert!(!p.is_empty());
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
            parse_bytecode("67363d3d37363d34f03d5260086018f3").unwrap(),
            DEFAULT_PROXY_CODE_HASH
        );
        assert_eq!(
            parse_bytecode("0x67363d3d37363d34f03d5260086018f3").unwrap(),
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
