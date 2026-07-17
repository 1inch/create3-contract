# How our CREATE3 works

This document explains how contract deployment through `Create3Deployer` works: why it exists, how the address is computed, why the address does not depend on the contract code, and how all of this is protected from front-running.

## Why this is needed

Ethereum has three ways to derive the address of a new contract:

| Method | What the address depends on | Problem |
| --- | --- | --- |
| `CREATE` | sender address + its nonce | the nonce keeps growing, so the address cannot be pinned in advance |
| `CREATE2` | factory address + salt + **hash of the contract code** | change a single byte of code (or a constructor argument) and the address changes |
| `CREATE3` | factory address + salt — **nothing else** | — |

CREATE3 removes the main pain of CREATE2: the address no longer depends on the code being deployed. This enables two practical things:

- **The same address on every chain.** Deploy the factory at the same address on each chain and reuse the same salt — the contract lands on an identical address everywhere, even if the code differs slightly between chains.
- **Vanity addresses.** Mine a "pretty" salt once (for example, so the address starts with `0x1111111...`) and later deploy any contract to that address. The code can keep changing right up until the deployment — the address stays the same.

## The big picture

The system has two parts:

1. **The contracts** (`contracts/`) — the `Create3Deployer` factory with `deploy(salt, code)` and `addressOf(salt)`, backed by the `Create3` library.
2. **The miner** (`src/`) — a multi-threaded Rust tool that brute-forces salts until the resulting address matches a chosen pattern.

Both compute the address with the same formula, so a salt found by the miner can be passed straight into `deploy()`.

## The proxy trick: how CREATE3 works inside

The EVM has no native CREATE3 opcode — it is assembled from CREATE2 and CREATE in two steps:

```text
        CREATE2 (salt)                CREATE (nonce = 1)
Factory ───────────────► Proxy ─────────────────────────► Your contract
```

**Step 1.** The factory uses `CREATE2` to deploy a tiny single-use proxy contract. The proxy code is always the same (a constant), so the proxy address depends only on the factory and the salt.

**Step 2.** The factory calls the proxy, passing your contract's init code as calldata. The proxy performs a plain `CREATE`. This is its first (and only) creation, so the proxy's nonce is 1 — and the resulting address depends only on the proxy address.

Putting it together: the proxy address depends on `(factory, salt)`, the contract address depends on `(proxy)`. Your contract's code never enters the chain of derivation.

### The proxy bytecode, opcode by opcode

The proxy init code is 17 bytes: `0x68363d3d37363d34f0ff3d5260096017f3`. It simply puts 9 bytes of runtime code into memory and returns them:

```text
0x00  68 363d3d37363d34f0ff  PUSH9  runtime code
0x0a  3d                     RETURNDATASIZE  (cheap way to get 0)
0x0b  52                     MSTORE          (store the code in memory)
0x0c  6009                   PUSH1 9         (code length)
0x0e  6017                   PUSH1 23        (memory offset)
0x10  f3                     RETURN
```

The proxy runtime code is 9 bytes: `0x363d3d37363d34f0ff`. It copies calldata (your contract's init code) and performs CREATE:

```text
0x00  36  CALLDATASIZE   calldata size
0x01  3d  RETURNDATASIZE 0
0x02  3d  RETURNDATASIZE 0
0x03  37  CALLDATACOPY   copy calldata into memory
0x04  36  CALLDATASIZE   size
0x05  3d  RETURNDATASIZE 0
0x06  34  CALLVALUE      forward ETH, if any
0x07  f0  CREATE         deploy your contract
0x08  ff  SELFDESTRUCT   the proxy destroys itself
```

The final `ff` byte is specific to our version: the proxy destroys itself right after the deployment since it will never be needed again. The canonical variant (solmate/0xSequence/CreateX) is 16 bytes without `SELFDESTRUCT`. Because of this difference our proxy code hash is different, and the miner is configured for it by default.

## The address formula

The address is computed in two steps — exactly the way the EVM derives CREATE2 and CREATE addresses:

```text
proxy   = keccak256(0xff ++ factory ++ salt ++ bytecode_hash)[12:]
address = keccak256(0xd6 ++ 0x94 ++ proxy ++ 0x01)[12:]
```

Notation:

- `++` — byte concatenation (bytes laid out back to back, not addition).
- `[12:]` — "drop the first 12 bytes, keep the last 20": keccak256 yields 32 bytes, while an address is 20 bytes.
- `factory` — the address of the `Create3Deployer` factory (20 bytes).
- `salt` — the salt (32 bytes), used **as is**, with no transformations.
- `bytecode_hash` — `keccak256(proxy init code)`, a constant:

```text
init code:     0x68363d3d37363d34f0ff3d5260096017f3
bytecode_hash: 0x8d04f296f449a1e795ad35f27e6b1d09af5a2422fa137f3d6cbf52d7a920975c
```

The first line is the standard CREATE2 formula ([EIP-1014](https://eips.ethereum.org/EIPS/eip-1014)). The second is the standard CREATE formula: the RLP encoding of the pair `[proxy, nonce]`, where `0xd6` means "a list of 22 bytes", `0x94` means "a string of 20 bytes" (the proxy address), and `0x01` is the proxy's nonce, which is always 1.

Neither the contract code nor the sender address appears in the formula — only the factory and the salt. That is the defining property of CREATE3.

You can verify the address for any salt directly on the factory:

```bash
cast call <factory> "addressOf(bytes32)" <salt> --rpc-url <rpc>
```

## Security: why a raw salt cannot be front-run

A fair question: the salt is visible in the mempool — what stops someone from grabbing it and deploying their own code to our address first?

**Our protection is access control.** The `deploy()` method is restricted to the factory owner (`onlyOwner`). An outsider cannot deploy anything through our factory at all. And deploying the same salt through a *different* factory yields a different address — the factory address is part of the formula. Stealing the address is impossible.

**CreateX protects itself differently.** Its factory is public — anyone can deploy through it, so without extra protection front-running would be real. CreateX mixes `msg.sender` into the salt via re-hashing: the same salt from a different wallet produces a different address. The price is that the salt becomes permanently tied to the deploying wallet.

Comparing the two approaches:

| | CreateX | Our Create3Deployer |
| --- | --- | --- |
| Deploy access | public | owner only |
| Salt | hashed together with `msg.sender` (guarded) | raw, as is |
| Tied to a wallet | yes | no — ownership can be transferred |
| Front-run protection | mixing the sender into the salt | `onlyOwner` + address depends on the factory |
| Proxy init code | `0x67363d3d37363d34f03d5260086018f3` | `0x68363d3d37363d34f0ff3d5260096017f3` |
| bytecode_hash | `0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f` | `0x8d04f296f449a1e795ad35f27e6b1d09af5a2422fa137f3d6cbf52d7a920975c` |

We deliberately chose the raw salt: any wallet that receives ownership can deploy, and mined salts survive a change of deployer.

Things to keep in mind:

- **The factory owner controls all future addresses.** Whoever owns the factory decides which code ends up at which address. The owner key deserves the same care as a deployment key.
- **An address is taken forever.** A second deployment to the same address is impossible — the library reverts with `TargetAlreadyExists`.

## Mining vanity addresses

Since the address depends only on `(factory, salt)`, a pretty address is found by brute-forcing salts. That is the Rust miner's job:

```bash
# Address starts with 1111111
cargo run --release -- <factory> --leading 1111111

# Regex: suffix, match anywhere, alternatives
cargo run --release -- <factory> 'beef$'
cargo run --release -- <factory> '^(a90a|a94a)'
```

Every additional fixed hex character multiplies the search time by 16. Expected attempts and time at different speeds:

| Characters | Attempts (average) | 92 MH/s | 507 MH/s | 3 GH/s |
| --- | --- | --- | --- | --- |
| 8 | 4.3 billion | ~47 s | ~8.5 s | ~1.4 s |
| 9 | 69 billion | ~12 min | ~2.3 min | ~23 s |
| 10 | 1.1 trillion | ~3.3 h | ~36 min | ~6 min |
| 11 | 17.6 trillion | ~2.2 days | ~9.6 h | ~1.6 h |
| 12 | 281 trillion | ~35 days | ~6.4 days | ~26 h |

These are the means of a geometric distribution: with ~63% probability the salt is found within that time, but roughly 1 time in 7 the search takes more than twice the average. A pattern matched "anywhere in the address" is found ~30x faster than a leading prefix of the same length.

If an external party does the mining, they need exactly three parameters:

1. the proxy init code or its hash: `0x8d04f296f449a1e795ad35f27e6b1d09af5a2422fa137f3d6cbf52d7a920975c`;
2. the factory address;
3. the fact that the salt is raw (no `hash(caller, salt)` as in CreateX).

Before a long mining run, cross-check one test salt: the address from the miner must match `addressOf(salt)` on the factory.

## The full deployment cycle

```bash
# 1. Deploy the factory
forge script script/DeployCreate3Deployer.s.sol --rpc-url <rpc> --private-key <key> --broadcast

# 2. Mine a salt for the desired pattern
cargo run --release -- <factory> --leading <hex prefix>

# 3. Deploy the contract to the mined address
cast send <factory> "deploy(bytes32,bytes)" <salt> <initCode> --rpc-url <rpc> --private-key <key>
```

The contract lands on the address printed by the miner — whatever init code is passed in step 3.

## FAQ

**Can different code be deployed to the same address on different chains?**
Yes. The address depends only on the factory and the salt, so on each chain (with the factory at the same address) one salt gives one address — while the code can be anything.

**What happens if the same salt is used twice?**
The transaction reverts with `TargetAlreadyExists` — the address already has code.

**Do constructor arguments affect the address?**
No. The init code (constructor arguments included) is handed to the proxy after the address is already determined. Both code and arguments can change right up until deployment.

**Why does the proxy self-destruct (`SELFDESTRUCT`)?**
The proxy is needed for exactly one operation — it is useless after the deployment, so our bytecode variant destroys it immediately. This does not affect the final contract address.

**Are our salts compatible with CreateX?**
No. CreateX has a different factory address, a different proxy code hash, and it hashes the salt together with the sender before use. All three inputs of the formula differ — salts and addresses do not transfer in either direction.
