# Metal keccak-f[1600] microbenchmark

Measures raw keccak-f[1600] permutation throughput on the Apple GPU, to decide
whether porting the CREATE3 vanity miner to Metal is worth it before writing a
full miner. The miner needs two chained permutations per attempt, so:

```text
expected miner rate (MH/s) ≈ permutations per second / 2
```

Compare against the CPU NEON miner (`cargo run --release -- <factory>
--leading <hex>` prints its rate every 5 s): if the GPU number is only ~1.5x
the CPU number, the port is not worth the complexity; at 3x and above it is.

## Run

```bash
make run
# or
make && ./keccak-bench [--threads N] [--iters N] [--dispatches N] [--variant u64|interleaved]
```

Needs only the Xcode Command Line Tools: the kernel (`keccak.metal`) is
compiled at runtime through the Metal framework, so no `metal` toolchain
install is required. Run plugged in — GPU clocks drop substantially on
battery.

## What it does

Each GPU thread seeds a keccak state from its thread id and runs `--iters`
*chained* permutations (each depends on the previous, modelling the miner's
stage-1 -> stage-2 dependency), then writes an xor-reduction of the final
state so nothing can be dead-code-eliminated. Reported rate = threads x iters
/ GPU time (`gpuStartTime`/`gpuEndTime` of the command buffer).

Two kernel variants, both keeping the whole state in named registers:

- **u64** — lanes as `ulong`. Apple GPU ALUs are 32-bit, so the compiler
  lowers every 64-bit op into 32-bit pairs; 64-bit rotations are the
  expensive part. The simplest possible port.
- **interleaved** — bit-interleaved lanes as `uint2` (even bits in `.x`, odd
  in `.y`): a 64-bit rotation becomes two independent 32-bit rotations. The
  standard trick for keccak on 32-bit hardware, at the cost of converting
  in/out of the interleaved representation (irrelevant in the real miner:
  inputs are near-constant and only 3 output words are needed).

Before timing, both kernels are cross-checked against a scalar reference
implementation (itself checked against the known keccak256("") digest) on the
empty-string vector and on random states with 1 and 3 chained permutations.

## Reading the output

- `maxThreadsPerThreadgroup` (printed per variant): 1024 means the kernel is
  not register-limited; lower values mean occupancy is capped by register
  pressure — expected for keccak (the state alone is 50 x 32-bit registers),
  and the main thing this benchmark exists to quantify.
- If a dispatch fails with a GPU timeout, lower `--iters` (default 128) or
  `--threads` (default 1M); per-dispatch work is threads x iters permutations.
- Apple-silicon reference points for context: the repo's NEON+SHA3 CPU miner
  does ~122 Mperm/s at 8 threads on an M-series laptop (~61 MH/s); the GPU
  ALU ceiling on an M4 Max (40 cores) is very roughly ~1 Gperm/s.

## Measured results (M4 Max, 40-core GPU, Aug 2026)

Defaults (1M threads x 128 iters), plugged in, `maxThreadsPerThreadgroup=1024`
for both variants (not register-limited):

| Kernel        | Mperm/s | Miner-equivalent |
| ------------- | ------- | ---------------- |
| `u64`         | ~739    | ~370 MH/s        |
| `interleaved` | ~1046   | ~523 MH/s        |

CPU baseline on the same machine, NEON+SHA3 miner (`--leading`, 5 s progress
lines): ~101 MH/s on all 16 cores, ~88 MH/s on 12 threads. So the interleaved
GPU kernel runs at ~5.2x the best CPU rate, and sits within ~10-15% of the
theoretical ALU ceiling — a Metal port of the miner is worth building, and
bit interleaving is the representation to use.
