# 1inch Create3

A CREATE3 deployer contract and a companion tool that mines vanity salts for it.

The repo has two parts that work together:

1. **The deployer** ([contracts/](contracts/)) — an `Ownable` `Create3Deployer` that deploys contracts at addresses which depend only on the factory address and a salt, independent of the contract's creation code.
2. **The miner** — a multi-threaded Rust tool that brute-forces a salt so the resulting deploy address matches a pattern you choose. It ships as two binaries that share all logic ([src/lib.rs](src/lib.rs)):
   - `create3-miner` ([src/main.rs](src/main.rs)) — portable scalar miner, runs everywhere.
   - `create3-miner-neon` ([src/bin/create3-miner-neon.rs](src/bin/create3-miner-neon.rs)) — ARM NEON (aarch64) build that hashes two salts per keccak permutation and, when available, uses the ARMv8 SHA3 crypto extension (fused EOR3/RAX1/XAR/BCAX) with a runtime fallback to base NEON. Roughly **2.5× faster** than the scalar miner on Apple Silicon. This is the default binary for `cargo run`.

## The deployer

The factory lives in [contracts/](contracts/): the [`Create3` library](contracts/libraries/Create3.sol) plus an `Ownable` [`Create3Deployer`](contracts/Create3Deployer.sol) wrapper exposing `deploy(salt, code)` and `addressOf(salt)`.

Because it uses the raw `Create3` derivation (raw salt, no `msg.sender` namespacing), the deployed address depends only on the factory address and the salt — not on the contract being deployed. That property is what makes vanity mining possible: find a salt once, and any contract deployed with it lands on the same predictable address.

Proxy init code: `0x68363d3d37363d34f0ff3d5260096017f3`. The address derivation is:

```text
proxy   = keccak256(0xff ++ factory ++ salt ++ keccak256(proxyBytecode))[12:]
address = keccak256(0xd6 ++ 0x94 ++ proxy ++ 0x01)[12:]
```

It is a Foundry project:

```bash
# Build and test (includes addressOf cross-checks against the Rust miner)
forge build
forge test

# Deploy the factory (configure your RPC and key)
forge script script/DeployCreate3Deployer.s.sol --rpc-url <rpc> --private-key <key> --broadcast
```

## The miner

Once the factory is deployed, the miner brute-forces a CREATE3 salt so the deployed contract address matches a pattern. You can match either an exact leading hex prefix (the fast path) or an arbitrary regex.

```bash
cargo run --release -- <factory address> <pattern>
cargo run --release -- <factory address> --leading <hex prefix>
```

### Which binary runs

`cargo run` defaults to the faster NEON binary (`default-run` in [`Cargo.toml`](Cargo.toml)). On Apple Silicon / aarch64 you get the 2-wide NEON miner automatically. To pick a binary explicitly:

```bash
# NEON miner (aarch64 only; the default)
cargo run --release --bin create3-miner-neon -- <factory> --leading <hex>

# Portable scalar miner (works on any CPU)
cargo run --release --bin create3-miner -- <factory> --leading <hex>
```

On non-aarch64 platforms `create3-miner-neon` compiles to a stub that exits with a message telling you to use `create3-miner` instead, so the whole workspace still builds everywhere. Both binaries accept identical flags.

At startup the NEON binary prints which keccak backend it selected (`NEON+SHA3` or `NEON`) and the buffer width. The SHA3 extension is detected at runtime (`is_aarch64_feature_detected!("sha3")`); on chips without it the miner transparently falls back to base NEON. Two NEON-only flags are available:

- `--buffers <N>` (1-4): number of independent keccak states processed per batch. Default is `1`; larger values were slower in benchmarks because the 25-vector keccak state already spills past the 32 NEON registers, so extra buffers only add spill traffic. Kept for experimentation.
- `--no-sha3`: force the base NEON path even when the SHA3 extension exists (useful for benchmarking its contribution).

The repo ships a [`.cargo/config.toml`](.cargo/config.toml) that builds with `-C target-cpu=native`, so release builds automatically use chip-specific instructions (e.g. NEON on Apple Silicon) for extra throughput. No environment variables are needed; a plain `cargo build --release` or `cargo run --release` picks it up. The resulting binary is tuned for the build machine and is not portable to other CPUs.

### Leading prefix (fast path)

For a vanity prefix, prefer `--leading`. It compares the address bytes directly, skipping hex encoding and the regex engine, which is a few percent faster on long searches. The prefix is case-insensitive and may include an optional `0x`.

```bash
# Address starts with nine zeros
cargo run --release -- 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf --leading 000000000

# Address starts with dead
cargo run --release -- 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf --leading dead
```

### Regex (suffixes, anywhere, alternation)

The positional pattern is a regex matched against the 40-character lowercase hex address (without the `0x` prefix). Matching is case-insensitive. Use it for anything `--leading` cannot express, such as suffixes or alternation.

```bash
# Address ends with beef
cargo run --release -- 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf 'beef$'

# c0ffee anywhere in the address
cargo run --release -- 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf 'c0ffee'

# Full regex: 8 leading zeros, or dead/beef suffix
cargo run --release -- 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf '^0{8}'
cargo run --release -- 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf '(dead|beef)$'
```

You must provide exactly one of the positional regex or `--leading`.

### Thread count

By default the miner uses all available cores. On hybrid CPUs (for example Apple Silicon with performance and efficiency cores) the slower efficiency cores can drag down the aggregate hash rate, so restricting to the performance core count is sometimes faster.

```bash
# Use 12 worker threads (e.g. only the performance cores)
cargo run --release -- 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf --leading 000000000 --threads 12
```

### Custom proxy bytecode

By default the miner uses the standard CREATE3 proxy code hash (`keccak256(0x68363d3d37363d34f0ff3d5260096017f3)`). If your factory deploys a different proxy, override it with one of two mutually exclusive flags (both accept hex with or without a `0x` prefix):

- `-B` / `--bytecode-hash` — provide the 32-byte proxy code hash directly.
- `-b` / `--bytecode` — provide the raw proxy bytecode; the miner hashes it for you.

```bash
# Provide the proxy code hash directly
cargo run --release -- <factory> --leading dead -B 0x8d04f296f449a1e795ad35f27e6b1d09af5a2422fa137f3d6cbf52d7a920975c

# Or provide the raw proxy bytecode and let the miner hash it
cargo run --release -- <factory> --leading dead -b 0x68363d3d37363d34f0ff3d5260096017f3
```

When a non-default code hash is used it is echoed in the startup header so you can confirm it.

Example output:

```text
Factory:  0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf
Pattern:  ^dead (regex, case-insensitive)
Threads:  16
Mining...

Found a match!
Salt:     0x085c48e022b858aac59fbe8266fbcc5260705e350c6406c8ae4e96567ff1f7a8
Address:  0xdeAdE579710ce89209e70D6b8aa8d39259AeCb85
Attempts: 123549 in 0.0s (12.08 MH/s)
```

## Putting it together

```bash
# 1. Deploy the factory (configure your RPC and key)
forge script script/DeployCreate3Deployer.s.sol --rpc-url <rpc> --private-key <key> --broadcast

# 2. Mine a vanity address for the deployed factory
cargo run --release -- <deployed factory> '^dead'

# 3. Deploy a contract at the mined address using the found salt
cast send <factory> "deploy(bytes32,bytes)" <salt> <initCode> --rpc-url <rpc> --private-key <key>
```

The contract will land on the printed address regardless of its creation code.

## Performance

By default the miner uses all CPU cores (override with `--threads`); each worker starts from a random salt and increments sequentially, reusing preallocated hash buffers (two keccak256 per attempt). The NEON binary hashes two salts per keccak permutation by packing each state into 128-bit lanes (2 × `u64`), precomputes the constant input words so only the salt counter is rewritten per attempt, and on SHA3-capable chips uses the fused EOR3/RAX1/XAR/BCAX instructions. Every additional constrained hex character multiplies the expected search time by 16.

Indicative throughput at 8 threads on an Apple Silicon laptop (single `--leading` search):

| Build | MH/s | vs scalar |
| ----- | ---- | --------- |
| `create3-miner` (scalar `sha3` crate) | ~25 | 1.0× |
| `create3-miner-neon --no-sha3` (base NEON x2) | ~47 | ~1.9× |
| `create3-miner-neon` (NEON + SHA3 ext) | ~61 | ~2.5× |

Expected attempts per constrained-character count:

| Constrained chars | Expected attempts |
| ----------------- | ----------------- |
| 4                 | ~65 thousand      |
| 6                 | ~17 million       |
| 8                 | ~4.3 billion      |

A progress line with the total attempt count and hash rate is printed to stderr every 5 seconds.

## Tests

```bash
# Rust miner
cargo test --release

# Solidity contracts (addressOf cross-checks against the miner)
forge test
```

The Rust suite includes a derivation vector cross-checked against foundry's `cast create2` / `cast keccak`, NEON-vs-scalar equivalence tests for the batched keccak and address derivation, and the Foundry suite cross-checks `addressOf` against the miner's derivation.
