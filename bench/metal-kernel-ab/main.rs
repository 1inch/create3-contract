//! Paired A/B benchmark for the CREATE3 Metal mining kernel. See README.md.
//!
//! Times single dispatches with the command buffer's own GPU clock and, when
//! given two kernel sources, alternates between them so both meet the same clock
//! and thermal state. On a power-limited Mac that is the difference between
//! resolving a 1% change and not resolving a 10% one.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("metal-kernel-ab requires macOS (Apple GPU / Metal).");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    bench::run();
}

#[cfg(target_os = "macos")]
mod bench {
    use std::ffi::c_void;
    use std::path::Path;
    use std::ptr::NonNull;

    use clap::{Arg, Command};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
        MTLResourceOptions, MTLSize,
    };

    use create3_miner::{
        create3_address, parse_address, parse_leading, AddressPattern, DEFAULT_PROXY_CODE_HASH,
    };

    /// The kernel under test unless another path is given.
    const DEFAULT_KERNEL: &str = "src/bin/create3-miner-metal/kernel.metal";

    const MAX_HITS: usize = 16;
    const HITS_LEN: usize = 8 + MAX_HITS * 8;
    /// `MineParams` in the miner's kernel: 17 interleaved template words, the
    /// pattern and mask over final-state words 1..3, the two plain words the
    /// counter is spliced into, then the dispatch base counter.
    const PARAMS_LEN: usize = 17 * 8 + 3 * 8 + 3 * 8 + 8 + 8 + 8;

    /// Timing inputs are arbitrary - any factory and salt cost the same two
    /// permutations - but they are fixed rather than random so a run is
    /// reproducible, and real enough for the cross-check to mean something.
    const FACTORY: &str = "0x71481c3b9c6fba3066ae84961ea22378a80cabe7";
    const SALT_PREFIX: [u8; 24] = [0x5a; 24];
    const VERIFY_BASE: u64 = 0x0102_0304_0500_0000;

    fn interleave64(x: u64) -> (u32, u32) {
        fn even_bits(mut x: u64) -> u32 {
            x &= 0x5555_5555_5555_5555;
            x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
            x = (x | (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
            x = (x | (x >> 4)) & 0x00ff_00ff_00ff_00ff;
            x = (x | (x >> 8)) & 0x0000_ffff_0000_ffff;
            x = (x | (x >> 16)) & 0x0000_0000_ffff_ffff;
            x as u32
        }
        (even_bits(x), even_bits(x >> 1))
    }

    /// The 136-byte padded keccak block for the 85-byte first stage, counter
    /// zeroed (mirrors `stage1_block` in the miner).
    fn stage1_block(factory: &[u8; 20], code_hash: &[u8; 32]) -> [u8; 136] {
        let mut block = [0u8; 136];
        block[0] = 0xff;
        block[1..21].copy_from_slice(factory);
        block[21..45].copy_from_slice(&SALT_PREFIX);
        // bytes 45..53 (the counter) stay zero
        block[53..85].copy_from_slice(code_hash);
        block[85] ^= 0x01;
        block[135] ^= 0x80;
        block
    }

    /// Serializes `MineParams` for `setBytes`, laid out as the kernel declares it.
    fn mine_params(block: &[u8; 136], value: &[u8; 20], mask: &[u8; 20]) -> Vec<u8> {
        let mut blob = vec![0u8; PARAMS_LEN];
        let mut put = |off: usize, word: u64| {
            let (lo, hi) = interleave64(word);
            blob[off..off + 4].copy_from_slice(&lo.to_le_bytes());
            blob[off + 4..off + 8].copy_from_slice(&hi.to_le_bytes());
        };

        for w in 0..17 {
            put(w * 8, u64::from_le_bytes(block[w * 8..w * 8 + 8].try_into().unwrap()));
        }

        // The address occupies hash bytes 12..32: bytes 0..4 in the high half of
        // final-state word 1, bytes 4..12 in word 2, bytes 12..20 in word 3.
        let words = |bytes: &[u8; 20]| -> [u64; 3] {
            let mut w1 = [0u8; 8];
            w1[4..8].copy_from_slice(&bytes[0..4]);
            [
                u64::from_le_bytes(w1),
                u64::from_le_bytes(bytes[4..12].try_into().unwrap()),
                u64::from_le_bytes(bytes[12..20].try_into().unwrap()),
            ]
        };
        let pattern_off = 17 * 8;
        let mask_off = pattern_off + 3 * 8;
        for (i, (p, m)) in words(value).iter().zip(words(mask).iter()).enumerate() {
            put(pattern_off + i * 8, *p);
            put(mask_off + i * 8, *m);
        }

        let plain_off = mask_off + 3 * 8;
        for (i, range) in [40..48, 48..56].iter().enumerate() {
            let word = u64::from_le_bytes(block[range.clone()].try_into().unwrap());
            blob[plain_off + i * 8..plain_off + i * 8 + 8].copy_from_slice(&word.to_le_bytes());
        }

        blob
    }

    /// The base counter is the trailing field of `MineParams`, so a variant kernel
    /// that keeps it last needs no change here.
    fn set_counter(blob: &mut [u8], counter: u64) {
        let n = blob.len();
        blob[n - 8..].copy_from_slice(&counter.to_le_bytes());
    }

    fn shared_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        len: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .expect("failed to allocate a Metal buffer")
    }

    fn pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
        function: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(source), None)
            .expect("Metal kernel failed to compile");
        let func = library
            .newFunctionWithName(&NSString::from_str(function))
            .unwrap_or_else(|| panic!("kernel function `{function}` not found"));
        device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("failed to create the compute pipeline")
    }

    /// Encodes one dispatch, waits for it, and returns the seconds the command
    /// buffer reports for itself. That clock excludes queueing and host work, so
    /// it measures the kernel rather than the pipeline around it.
    fn dispatch(
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        params: &[u8],
        out: &ProtocolObject<dyn MTLBuffer>,
        threads: usize,
    ) -> f64 {
        let cb = queue.commandBuffer().expect("failed to create a command buffer");
        let enc = cb.computeCommandEncoder().expect("failed to create an encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            let ptr = NonNull::new(params.as_ptr() as *mut c_void).unwrap();
            enc.setBytes_length_atIndex(ptr, params.len(), 0);
            enc.setBuffer_offset_atIndex(Some(out), 0, 1);
        }
        let tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: threads, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        cb.GPUEndTime() - cb.GPUStartTime()
    }

    /// Cross-checks a kernel's derivation against the scalar reference through the
    /// `debug_addresses` entry point, so a timing number can never come from a
    /// kernel that is quietly computing the wrong thing.
    fn verify(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        label: &str,
        source: &str,
        factory: &[u8; 20],
        code_hash: &[u8; 32],
        params: &[u8],
    ) {
        const N: usize = 256;
        let pipeline = pipeline(device, source, "debug_addresses");
        let out = shared_buffer(device, N * 3 * 8);
        let mut blob = params.to_vec();
        set_counter(&mut blob, VERIFY_BASE);
        dispatch(queue, &pipeline, &blob, &out, N);

        let ptr = out.contents().as_ptr() as *const u64;
        for i in 0..N {
            let (w1, w2, w3) = unsafe {
                (
                    ptr.add(3 * i).read_unaligned(),
                    ptr.add(3 * i + 1).read_unaligned(),
                    ptr.add(3 * i + 2).read_unaligned(),
                )
            };
            let mut got = [0u8; 20];
            got[0..4].copy_from_slice(&w1.to_le_bytes()[4..8]);
            got[4..12].copy_from_slice(&w2.to_le_bytes());
            got[12..20].copy_from_slice(&w3.to_le_bytes());

            let counter = VERIFY_BASE + i as u64;
            let mut salt = [0u8; 32];
            salt[..24].copy_from_slice(&SALT_PREFIX);
            salt[24..32].copy_from_slice(&counter.to_be_bytes());
            let want = create3_address(factory, &salt, code_hash);
            assert_eq!(got, want, "[{label}] derivation mismatch at counter {counter}");
        }
        println!("[{label}] correctness OK: {N} addresses match the scalar reference");
    }

    fn median(values: &[f64]) -> f64 {
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).expect("GPU timing produced a NaN"));
        let mid = v.len() / 2;
        if v.len().is_multiple_of(2) {
            (v[mid - 1] + v[mid]) / 2.0
        } else {
            v[mid]
        }
    }

    fn label_of(path: &str) -> String {
        Path::new(path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string())
    }

    fn parse_or_exit(name: &str, raw: &str, max: usize) -> usize {
        match raw.parse::<usize>() {
            Ok(n) if (1..=max).contains(&n) => n,
            _ => {
                eprintln!("error: --{name} must be an integer in 1..={max}");
                std::process::exit(1);
            }
        }
    }

    pub fn run() {
        let matches = Command::new("metal-kernel-ab")
            .about(
                "Times the CREATE3 Metal mining kernel per dispatch, and A/Bs two \
                 kernel sources by alternating between them",
            )
            .arg(
                Arg::new("kernel")
                    .num_args(0..=2)
                    .help(
                        "Kernel source paths. One measures that kernel; two alternate \
                         between them (defaults to the miner's own kernel)",
                    ),
            )
            .arg(
                Arg::new("pairs")
                    .long("pairs")
                    .value_name("N")
                    .help("Dispatches to time per kernel (default 300)"),
            )
            .arg(
                Arg::new("batch")
                    .long("batch")
                    .value_name("N")
                    .help("Threads per dispatch (default 4194304)"),
            )
            .arg(
                Arg::new("leading")
                    .long("leading")
                    .value_name("HEX")
                    .help(
                        "Prefix the kernel matches against, which decides how much of \
                         the address a candidate kernel may skip (default 0000000000)",
                    ),
            )
            .arg(
                Arg::new("suffix")
                    .long("suffix")
                    .value_name("HEX")
                    .help("Suffix the kernel matches against"),
            )
            .arg(
                Arg::new("mask")
                    .long("mask")
                    .value_name("TEMPLATE")
                    .help("Left-anchored hex/'.' template the kernel matches against"),
            )
            .get_matches();

        let pairs = matches
            .get_one::<String>("pairs")
            .map_or(300, |s| parse_or_exit("pairs", s, 100_000));
        let batch = matches
            .get_one::<String>("batch")
            .map_or(4 << 20, |s| parse_or_exit("batch", s, 1 << 26));

        let mut pattern = AddressPattern::default();
        let apply = |flag: &str, result: Result<(), String>| {
            if let Err(e) = result {
                eprintln!("error: invalid --{flag}: {e}");
                std::process::exit(1);
            }
        };
        if let Some(hex) = matches.get_one::<String>("leading") {
            apply("leading", parse_leading(hex).and_then(|n| pattern.add_leading(&n)));
        }
        if let Some(hex) = matches.get_one::<String>("suffix") {
            apply("suffix", parse_leading(hex).and_then(|n| pattern.add_suffix(&n)));
        }
        if let Some(template) = matches.get_one::<String>("mask") {
            apply("mask", pattern.add_template(template));
        }
        if pattern.is_empty() {
            // Ten fixed nibbles: tight enough that no dispatch reports a hit, so
            // the atomic path stays out of the measurement.
            pattern
                .add_leading(&parse_leading("0000000000").expect("valid literal"))
                .expect("empty pattern accepts a prefix");
        }

        let paths: Vec<String> = matches
            .get_many::<String>("kernel")
            .map(|v| v.cloned().collect())
            .unwrap_or_else(|| vec![DEFAULT_KERNEL.to_string()]);

        let device = MTLCreateSystemDefaultDevice().unwrap_or_else(|| {
            eprintln!("error: no Metal device available");
            std::process::exit(1);
        });
        let queue = device.newCommandQueue().expect("failed to create a command queue");

        let factory = parse_address(FACTORY).expect("valid literal address");
        let code_hash = DEFAULT_PROXY_CODE_HASH;
        let block = stage1_block(&factory, &code_hash);
        let mut params = mine_params(&block, &pattern.value, &pattern.mask);

        println!("Device:   {}", device.name());
        println!(
            "Pattern:  {} ({} fixed nibbles)",
            pattern.template_string(),
            pattern.fixed_nibbles()
        );
        println!("Dispatch: {batch} threads x {pairs} per kernel");

        let sources: Vec<(String, String)> = paths
            .iter()
            .map(|p| {
                let src = std::fs::read_to_string(p).unwrap_or_else(|e| {
                    eprintln!("error: cannot read {p}: {e}");
                    std::process::exit(1);
                });
                (label_of(p), src)
            })
            .collect();

        for (label, source) in &sources {
            verify(&device, &queue, label, source, &factory, &code_hash, &params);
        }

        let pipelines: Vec<_> = sources
            .iter()
            .map(|(_, source)| pipeline(&device, source, "mine"))
            .collect();
        for ((label, _), p) in sources.iter().zip(pipelines.iter()) {
            // Below 1024 the kernel is register-limited, which caps occupancy.
            println!(
                "[{label}] maxTotalThreadsPerThreadgroup: {}",
                p.maxTotalThreadsPerThreadgroup()
            );
        }

        let hits = shared_buffer(&device, HITS_LEN);
        unsafe { (hits.contents().as_ptr() as *mut u32).write_unaligned(0) };

        let mut counter = VERIFY_BASE;
        let mut rates: Vec<Vec<f64>> = vec![Vec::with_capacity(pairs); pipelines.len()];

        // Let the pipelines become resident and the GPU clocks ramp.
        for p in &pipelines {
            for _ in 0..5 {
                set_counter(&mut params, counter);
                counter = counter.wrapping_add(batch as u64);
                dispatch(&queue, p, &params, &hits, batch);
            }
        }

        for i in 0..pairs {
            // Swap the order every round so a first-versus-second position effect
            // cancels along with the drift.
            let mut order: Vec<usize> = (0..pipelines.len()).collect();
            if i % 2 == 1 {
                order.reverse();
            }
            for k in order {
                set_counter(&mut params, counter);
                counter = counter.wrapping_add(batch as u64);
                let seconds = dispatch(&queue, &pipelines[k], &params, &hits, batch);
                rates[k].push(batch as f64 / seconds / 1e6);
            }
        }

        println!();
        for ((label, _), r) in sources.iter().zip(rates.iter()) {
            let lo = r.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = r.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "[{label}] median {:.1} MH/s (min {lo:.1}, max {hi:.1})",
                median(r)
            );
        }

        if let [a, b] = rates.as_slice() {
            let (ma, mb) = (median(a), median(b));
            let wins = a.iter().zip(b.iter()).filter(|(x, y)| y > x).count();
            println!();
            println!(
                "delta: {:+.2}% ({} over {})",
                (mb / ma - 1.0) * 100.0,
                sources[1].0,
                sources[0].0
            );
            // Each pair is two dispatches ~10 ms apart, so this count is the
            // drift-proof statistic: near half the pairs means no real difference.
            println!("{} faster in {wins}/{pairs} pairs", sources[1].0);
        }
    }
}
