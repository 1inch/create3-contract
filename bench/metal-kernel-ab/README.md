# Metal mining kernel A/B benchmark

Answers one question: is this change to the mining kernel actually faster? It
times single dispatches with the GPU's own clock and, given two kernel sources,
alternates between them so both meet the same clock and thermal state.

## Why not just read the miner's MH/s

Because on a laptop that number moves more than the changes worth measuring. On
an M4 Max the miner reports anywhere from ~510 MH/s cold to ~330 MH/s once the
package is hot, and it drifts throughout. Two consequences:

- A before/after comparison measures the thermal state, not the kernel. Whichever
  binary ran second loses.
- Interleaving runs (ABBA) only cancels drift that is *linear*. Real cooling
  curves are convex: they fall steeply, then flatten. Under that shape the outer
  slots of an ABBA block score about 4% higher than the inner ones no matter what
  code is in them - the same size as the effects being chased. Measured both
  orders of a real comparison and got +2.2% and −5.7% out of identical binaries'
  worth of noise.

Alternating individual dispatches puts the two measurements ~10 ms apart, where
the clock has not moved, and the command buffer reports its own GPU time
(`GPUStartTime`/`GPUEndTime`) so host and queueing overhead stay out of it.

## Run

```bash
# Time the miner's current kernel
cargo run --release --bin metal-kernel-ab

# A/B two kernel sources (baseline first, candidate second)
cargo run --release --bin metal-kernel-ab -- baseline.metal candidate.metal

# Options
cargo run --release --bin metal-kernel-ab -- \
  --pairs 300 --batch 4194304 --leading 0000000000 a.metal b.metal
```

Run plugged in. Paths are relative to the repository root, and with no path the
miner's own [`kernel.metal`](../../src/bin/create3-miner-metal/kernel.metal) is
used.

To try a kernel change, copy `kernel.metal`, edit the copy, and pass both. A
candidate has to keep the host protocol the miner uses - `mine` and
`debug_addresses` taking `MineParams` at buffer 0 and the output at buffer 1, with
the base counter as the last field of `MineParams`. A candidate that changes the
protocol needs `mine_params` in [main.rs](main.rs) updated to match; the
cross-check below will fail loudly rather than quietly time the wrong thing.

## What it does

Before timing anything, each kernel derives 256 addresses through its
`debug_addresses` entry point and they are compared against the scalar
`create3_address` from the library. A timing number is worthless if the kernel is
computing the wrong thing, and the parameter layout here is hand-rolled.

Then each kernel gets `--pairs` dispatches of `--batch` threads, alternating, with
the order swapped every round so a first-versus-second position effect cancels
along with the drift. Reported rate is threads / GPU seconds for that one
dispatch.

The default pattern is ten fixed nibbles, tight enough that no dispatch reports a
hit, which keeps the atomic path out of the measurement. Pass `--leading` /
`--suffix` / `--mask` when the candidate's cost depends on how much of the address
is constrained.

## Reading the output

```text
[pre_pr3_kernel.metal] median 507.1 MH/s (min 472.0, max 511.6)
[kernel.metal]         median 522.5 MH/s (min 487.1, max 526.9)

delta: +3.05% (kernel.metal over pre_pr3_kernel.metal)
kernel.metal faster in 297/300 pairs
```

- **delta** compares medians, and is the number to quote.
- **pairs won** is the statistic to trust. Each pair is two dispatches ~10 ms
  apart, so drift cannot bias it. Near 150/300 means there is no difference,
  whatever the medians say; 297/300 is a real effect even when the delta is small.
- **maxTotalThreadsPerThreadgroup** below 1024 means the kernel is
  register-limited and occupancy is capped. Both current kernels sit at 1024.
- min/max show the thermal spread across the run, which is why the median and the
  pair count are the outputs and the mean is not.

## Measured results (M4 Max, 40-core GPU, Aug 2026)

Two changes measured with this tool, 300 pairs each:

| Change | delta | pairs won |
| ------ | ----- | --------- |
| Splicing stage 2 in the interleaved domain (kept) | +3.05% | 297/300 |
| Specializing the kernel per run, literals for the template and pattern | +0.74% | 260/300 |

The second one is the case for the pair count: a change that is unambiguously
real - it wins 260 of 300 pairs - and still too small to be worth shipping. It
also showed the same +0.7% for a suffix pattern, where the dead-code elimination
it was built around cannot fire, which is how it became clear the gain was the
literals replacing buffer loads rather than the pruning.
