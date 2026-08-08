// Host for the keccak-f[1600] Metal microbenchmark. See README.md.
//
// Compiles keccak.metal at runtime (no Metal toolchain needed, only the
// Command Line Tools), cross-checks both GPU kernel variants against a scalar
// reference implementation, then measures raw permutation throughput.

import Foundation
import Metal

func die(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("error: " + message + "\n").utf8))
    exit(1)
}

// MARK: - CLI

var threads = 1 << 20
var iterations: UInt32 = 128
var dispatches = 5
var variants = ["u64", "interleaved"]

do {
    let args = CommandLine.arguments
    var i = 1
    func value(_ flag: String) -> String {
        i += 1
        guard i < args.count else { die("missing value for \(flag)") }
        return args[i]
    }
    while i < args.count {
        let flag = args[i]
        switch flag {
        case "--threads":
            guard let v = Int(value(flag)), v > 0 else { die("--threads expects a positive integer") }
            threads = v
        case "--iters":
            guard let v = UInt32(value(flag)), v > 0 else { die("--iters expects a positive integer") }
            iterations = v
        case "--dispatches":
            guard let v = Int(value(flag)), v > 0 else { die("--dispatches expects a positive integer") }
            dispatches = v
        case "--variant":
            let v = value(flag)
            guard v == "u64" || v == "interleaved" else { die("--variant must be 'u64' or 'interleaved'") }
            variants = [v]
        case "--help", "-h":
            print("""
            usage: keccak-bench [options]
              --threads N      GPU threads per dispatch (default \(threads))
              --iters N        chained keccak-f[1600] permutations per thread (default \(iterations))
              --dispatches N   timed dispatches per variant (default \(dispatches))
              --variant V      run only one variant: u64 | interleaved (default: both)
            """)
            exit(0)
        default:
            die("unknown flag '\(flag)' (see --help)")
        }
        i += 1
    }
}

// MARK: - Scalar reference keccak-f[1600]

let roundConstants: [UInt64] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
// Flat index = x + 5*y, same layout and tables as the NEON miner.
let rhoOffsets: [UInt64] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43,
    25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
]
let piMapping: [Int] = [
    0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2,
    12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4,
]

func rotl(_ v: UInt64, _ n: UInt64) -> UInt64 {
    n == 0 ? v : (v << n) | (v >> (64 - n))
}

func keccakF(_ a: inout [UInt64]) {
    for rc in roundConstants {
        var c = [UInt64](repeating: 0, count: 5)
        for x in 0..<5 { c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] }
        var d = [UInt64](repeating: 0, count: 5)
        for x in 0..<5 { d[x] = c[(x + 4) % 5] ^ rotl(c[(x + 1) % 5], 1) }
        var b = [UInt64](repeating: 0, count: 25)
        for i in 0..<25 { b[piMapping[i]] = rotl(a[i] ^ d[i % 5], rhoOffsets[i]) }
        for row in stride(from: 0, to: 25, by: 5) {
            for x in 0..<5 {
                a[row + x] = b[row + x] ^ (~b[row + (x + 1) % 5] & b[row + (x + 2) % 5])
            }
        }
        a[0] ^= rc
    }
}

/// Sponge state for keccak256("") after padding, before the permutation.
func keccak256EmptyState() -> [UInt64] {
    var st = [UInt64](repeating: 0, count: 25)
    st[0] = 0x01
    st[16] = 0x80 << 56
    return st
}

func digestHex(_ state: [UInt64]) -> String {
    var out = ""
    for w in state[0..<4] {
        for byte in 0..<8 {
            out += String(format: "%02x", UInt8(truncatingIfNeeded: w >> (8 * UInt64(byte))))
        }
    }
    return out
}

// Guard against a broken reference implementation: known keccak256("") vector.
do {
    var st = keccak256EmptyState()
    keccakF(&st)
    let expected = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    guard digestHex(st) == expected else { die("host reference keccak-f self-test failed") }
}

// MARK: - Bit interleaving (host side of the `interleaved` variant)

func interleave(_ v: UInt64) -> (even: UInt32, odd: UInt32) {
    var e: UInt32 = 0
    var o: UInt32 = 0
    for i in 0..<32 {
        e |= UInt32(truncatingIfNeeded: (v >> (2 * UInt64(i))) & 1) << i
        o |= UInt32(truncatingIfNeeded: (v >> (2 * UInt64(i) + 1)) & 1) << i
    }
    return (e, o)
}

func deinterleave(_ e: UInt32, _ o: UInt32) -> UInt64 {
    var v: UInt64 = 0
    for i in 0..<32 {
        v |= UInt64((e >> UInt32(i)) & 1) << (2 * UInt64(i))
        v |= UInt64((o >> UInt32(i)) & 1) << (2 * UInt64(i) + 1)
    }
    return v
}

// MARK: - Metal setup

guard let device = MTLCreateSystemDefaultDevice() else {
    die("no Metal device available (GPU access blocked?)")
}
guard let queue = device.makeCommandQueue() else { die("cannot create a command queue") }

let executableDir = URL(fileURLWithPath: CommandLine.arguments[0])
    .resolvingSymlinksInPath().deletingLastPathComponent()
let sourceCandidates = [
    executableDir.appendingPathComponent("keccak.metal"),
    URL(fileURLWithPath: "keccak.metal"),
]
guard let sourceURL = sourceCandidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
    die("keccak.metal not found next to the binary or in the current directory")
}

let library: MTLLibrary
do {
    let source = try String(contentsOf: sourceURL, encoding: .utf8)
    library = try device.makeLibrary(source: source, options: MTLCompileOptions())
} catch {
    die("Metal library build failed:\n\(error.localizedDescription)")
}

var pipelineCache: [String: MTLComputePipelineState] = [:]

func makePipeline(_ name: String, iters: UInt32) -> MTLComputePipelineState {
    let key = "\(name)/\(iters)"
    if let cached = pipelineCache[key] { return cached }
    let constants = MTLFunctionConstantValues()
    var value = iters
    constants.setConstantValue(&value, type: .uint, index: 0)
    do {
        let function = try library.makeFunction(name: name, constantValues: constants)
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelineCache[key] = pipeline
        return pipeline
    } catch {
        die("cannot build pipeline '\(name)': \(error.localizedDescription)")
    }
}

func runSingleThread(_ pipeline: MTLComputePipelineState, buffer: MTLBuffer) {
    guard let cb = queue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder() else { die("cannot create a command buffer") }
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(buffer, offset: 0, index: 0)
    let one = MTLSize(width: 1, height: 1, depth: 1)
    enc.dispatchThreads(one, threadsPerThreadgroup: one)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let error = cb.error { die("GPU execution failed: \(error.localizedDescription)") }
}

// MARK: - Correctness

func gpuPermuteU64(_ state: [UInt64], iters: UInt32) -> [UInt64] {
    let pipeline = makePipeline("verify_u64", iters: iters)
    guard let buffer = device.makeBuffer(
        bytes: state, length: 25 * MemoryLayout<UInt64>.stride, options: .storageModeShared
    ) else { die("cannot allocate the verify buffer") }
    runSingleThread(pipeline, buffer: buffer)
    let ptr = buffer.contents().bindMemory(to: UInt64.self, capacity: 25)
    return Array(UnsafeBufferPointer(start: ptr, count: 25))
}

func gpuPermuteInterleaved(_ state: [UInt64], iters: UInt32) -> [UInt64] {
    let pipeline = makePipeline("verify_interleaved", iters: iters)
    var packed = [UInt32](repeating: 0, count: 50)
    for i in 0..<25 {
        let (e, o) = interleave(state[i])
        packed[2 * i] = e
        packed[2 * i + 1] = o
    }
    guard let buffer = device.makeBuffer(
        bytes: packed, length: 50 * MemoryLayout<UInt32>.stride, options: .storageModeShared
    ) else { die("cannot allocate the verify buffer") }
    runSingleThread(pipeline, buffer: buffer)
    let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: 50)
    return (0..<25).map { deinterleave(ptr[2 * $0], ptr[2 * $0 + 1]) }
}

func verify(_ label: String, permute: ([UInt64], UInt32) -> [UInt64]) {
    var expected = keccak256EmptyState()
    keccakF(&expected)
    guard permute(keccak256EmptyState(), 1) == expected else {
        die("[\(label)] keccak256(\"\") vector mismatch — kernel is broken")
    }

    for trial in 0..<4 {
        let state = (0..<25).map { _ in UInt64.random(in: UInt64.min...UInt64.max) }
        for iters in [UInt32(1), 3] {
            var want = state
            for _ in 0..<iters { keccakF(&want) }
            guard permute(state, iters) == want else {
                die("[\(label)] random-state mismatch (trial \(trial), \(iters) chained permutations)")
            }
        }
    }
    print("[\(label)] correctness OK: keccak256(\"\") vector + 4 random states x {1,3} chained permutations")
}

// MARK: - Benchmark

func bench(_ label: String, kernel: String, outStride: Int) {
    let pipeline = makePipeline(kernel, iters: iterations)
    let tgWidth = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
    guard let outBuffer = device.makeBuffer(length: threads * outStride, options: .storageModeShared) else {
        die("cannot allocate the output buffer")
    }

    // maxTotalThreadsPerThreadgroup < 1024 usually means the kernel is
    // register-limited — the key occupancy diagnostic for this experiment.
    print("[\(label)] maxThreadsPerThreadgroup=\(pipeline.maxTotalThreadsPerThreadgroup), "
        + "simdWidth=\(pipeline.threadExecutionWidth), threadgroup=\(tgWidth)")

    func dispatchOnce() -> Double {
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { die("cannot create a command buffer") }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(outBuffer, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error {
            die("GPU execution failed, try lower --iters/--threads: \(error.localizedDescription)")
        }
        return cb.gpuEndTime - cb.gpuStartTime
    }

    _ = dispatchOnce() // warmup: pipeline residency + GPU clock ramp

    let permsPerDispatch = Double(threads) * Double(iterations)
    var rates: [Double] = []
    for n in 1...dispatches {
        let seconds = dispatchOnce()
        let rate = permsPerDispatch / seconds
        rates.append(rate)
        print(String(format: "  run %d/%d: %7.3f s   %8.1f Mperm/s", n, dispatches, seconds, rate / 1e6))
    }
    let best = rates.max()!
    let median = rates.sorted()[rates.count / 2]
    print(String(format: "[%@] best %.1f Mperm/s, median %.1f Mperm/s  =>  ~%.1f MH/s miner-equivalent (2 keccak per attempt)",
                 label, best / 1e6, median / 1e6, best / 2e6))
    print("")
}

// MARK: - Main

print("Device:     \(device.name)")
print("Threads:    \(threads)")
print("Iters:      \(iterations) chained permutations per thread")
print(String(format: "Dispatch:   %.0f M permutations each", Double(threads) * Double(iterations) / 1e6))
print("")

for variant in variants {
    switch variant {
    case "u64":
        verify("u64", permute: gpuPermuteU64)
        bench("u64", kernel: "bench_u64", outStride: MemoryLayout<UInt64>.stride)
    case "interleaved":
        verify("interleaved", permute: gpuPermuteInterleaved)
        bench("interleaved", kernel: "bench_interleaved", outStride: MemoryLayout<UInt32>.stride)
    default:
        break
    }
}
