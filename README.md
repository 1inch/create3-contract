# 1inch Create3

A CREATE3 deployer contract and a companion tool that mines vanity salts for it.

The repo has two parts that work together:

1. **The deployer** ([contracts/](contracts/)) — an `Ownable` `Create3Deployer` that deploys contracts at addresses which depend only on the factory address and a salt, independent of the contract's creation code.
2. **The miner** ([src/main.rs](src/main.rs)) — a multi-threaded Rust tool that brute-forces a salt so the resulting deploy address matches a pattern you choose.

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

By default the miner uses all CPU cores (override with `--threads`); each worker starts from a random salt and increments sequentially, reusing preallocated hash buffers (two keccak256 per attempt). Expect on the order of 1 MH/s per core. Every additional constrained hex character multiplies the expected search time by 16:

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

The Rust suite includes a derivation vector cross-checked against foundry's `cast create2` / `cast keccak`, and the Foundry suite cross-checks `addressOf` against the miner's derivation.
