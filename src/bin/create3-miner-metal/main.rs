//! Metal (Apple GPU) build of the CREATE3 vanity miner.
//!
//! One GPU thread derives one candidate address: two chained keccak-f[1600]
//! permutations in bit-interleaved form (see `kernel.metal`). The host feeds a
//! precomputed stage-1 sponge template plus the match pattern/mask, dispatches
//! batches of `--batch` threads, and re-verifies every GPU-reported hit on the
//! CPU before accepting it.
//!
//! Only `--leading` / `--suffix` / `--mask` are supported. The regex mode is
//! CPU-only; on a regex pattern this binary errors out and points at the
//! scalar `create3-miner`. On non-macOS targets it compiles to a stub.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!(
        "create3-miner-metal requires macOS (Apple GPU / Metal). \
         Use the scalar `create3-miner` binary on this platform."
    );
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    metal::run();
}

#[cfg(target_os = "macos")]
mod metal {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
        MTLResourceOptions, MTLSize,
    };
    use rand::Rng;

    use create3_miner::{create3_address, mask_match, parse_cli, run_search, MatchMode};

    const MAX_HITS: usize = 16;
    /// Byte size of the `MineParams` struct in `kernel.metal` (see there for the
    /// field layout). All fields are 8-byte aligned and tightly packed:
    /// 17*8 (tmpl) + 3*8 (pattern) + 3*8 (mask) + 8 + 8 + 8.
    const PARAMS_LEN: usize = 17 * 8 + 3 * 8 + 3 * 8 + 8 + 8 + 8;
    /// Byte size of the `HitBuffer`: atomic count (u32) + pad (u32) + counters.
    const HITS_LEN: usize = 8 + MAX_HITS * 8;

    /// Interleaves the bits of a 64-bit lane: even bits into `.0`, odd bits into
    /// `.1`. Must match `interleave64` in `kernel.metal`.
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

    /// Builds the 136-byte padded keccak block for the 85-byte first stage, with
    /// the salt counter left zeroed (spliced in per-thread on the GPU).
    fn stage1_block(factory: &[u8; 20], salt_prefix: &[u8; 24], code_hash: &[u8; 32]) -> [u8; 136] {
        let mut tmpl = [0u8; 85];
        tmpl[0] = 0xff;
        tmpl[1..21].copy_from_slice(factory);
        tmpl[21..45].copy_from_slice(salt_prefix);
        // bytes 45..53 (the counter) stay zero
        tmpl[53..85].copy_from_slice(code_hash);

        let mut block = [0u8; 136];
        block[..85].copy_from_slice(&tmpl);
        block[85] ^= 0x01;
        block[135] ^= 0x80;
        block
    }

    /// Serializes `MineParams` into a byte blob for `setBytes`. `base_counter`
    /// is written last and rewritten cheaply per dispatch.
    struct Params {
        blob: [u8; PARAMS_LEN],
    }

    impl Params {
        fn new(
            block: &[u8; 136],
            value: &[u8; 20],
            mask: &[u8; 20],
        ) -> Self {
            let mut blob = [0u8; PARAMS_LEN];

            // tmpl[0..17]: interleaved stage-1 rate words.
            for w in 0..17 {
                let word = u64::from_le_bytes(block[w * 8..w * 8 + 8].try_into().unwrap());
                let (lo, hi) = interleave64(word);
                blob[w * 8..w * 8 + 4].copy_from_slice(&lo.to_le_bytes());
                blob[w * 8 + 4..w * 8 + 8].copy_from_slice(&hi.to_le_bytes());
            }

            // pattern[0..3] and mask[0..3] over final-state words 1..3. The
            // address occupies hash bytes 12..32: bytes 0..4 in the high half
            // of word 1, bytes 4..12 in word 2, bytes 12..20 in word 3.
            let words = |src: &[u8; 20]| -> [u64; 3] {
                let mut w1 = [0u8; 8];
                w1[4..8].copy_from_slice(&src[0..4]);
                let w2 = u64::from_le_bytes(src[4..12].try_into().unwrap());
                let w3 = u64::from_le_bytes(src[12..20].try_into().unwrap());
                [u64::from_le_bytes(w1), w2, w3]
            };
            let pat = words(value);
            let msk = words(mask);
            let pattern_off = 17 * 8;
            let mask_off = pattern_off + 3 * 8;
            for i in 0..3 {
                let (plo, phi) = interleave64(pat[i]);
                blob[pattern_off + i * 8..pattern_off + i * 8 + 4].copy_from_slice(&plo.to_le_bytes());
                blob[pattern_off + i * 8 + 4..pattern_off + i * 8 + 8].copy_from_slice(&phi.to_le_bytes());
                let (mlo, mhi) = interleave64(msk[i]);
                blob[mask_off + i * 8..mask_off + i * 8 + 4].copy_from_slice(&mlo.to_le_bytes());
                blob[mask_off + i * 8 + 4..mask_off + i * 8 + 8].copy_from_slice(&mhi.to_le_bytes());
            }

            // w5_base / w6_base: plain stage-1 words 5 and 6 (counter zeroed).
            let w5_base = u64::from_le_bytes(block[40..48].try_into().unwrap());
            let w6_base = u64::from_le_bytes(block[48..56].try_into().unwrap());
            let w5_off = mask_off + 3 * 8;
            blob[w5_off..w5_off + 8].copy_from_slice(&w5_base.to_le_bytes());
            blob[w5_off + 8..w5_off + 16].copy_from_slice(&w6_base.to_le_bytes());

            Params { blob }
        }

        fn set_base_counter(&mut self, c: u64) {
            let off = PARAMS_LEN - 8;
            self.blob[off..off + 8].copy_from_slice(&c.to_le_bytes());
        }
    }

    /// A shared-storage Metal buffer plus a raw pointer to its contents.
    struct SharedBuffer {
        buf: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    }

    impl SharedBuffer {
        fn new(device: &ProtocolObject<dyn MTLDevice>, len: usize) -> Self {
            let buf = device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                .expect("failed to allocate a Metal buffer");
            SharedBuffer { buf }
        }

        fn as_mut_ptr(&self) -> *mut u8 {
            self.buf.contents().as_ptr().cast()
        }

        /// Reads the hit buffer: number of hits and their counters (clamped to
        /// MAX_HITS), then resets the count to 0 for reuse.
        fn drain_hits(&self) -> Vec<u64> {
            let ptr = self.as_mut_ptr();
            unsafe {
                let count = (ptr as *const u32).read_unaligned() as usize;
                let n = count.min(MAX_HITS);
                let counters = ptr.add(8) as *const u64;
                let hits = (0..n).map(|i| counters.add(i).read_unaligned()).collect();
                (ptr as *mut u32).write_unaligned(0);
                hits
            }
        }
    }

    pub fn run() {
        let mut config = parse_cli("create3-miner-metal");

        let (value, mask) = match &config.mode {
            MatchMode::Mask { value, mask } => (*value, *mask),
            MatchMode::Regex(_) => {
                eprintln!(
                    "error: create3-miner-metal does not support regex patterns. \
                     Use --leading / --suffix / --mask here, or the scalar `create3-miner` \
                     for regex."
                );
                std::process::exit(1);
            }
        };

        let device = match MTLCreateSystemDefaultDevice() {
            Some(d) => d,
            None => {
                eprintln!(
                    "error: no Metal device available. \
                     Use the scalar `create3-miner` binary instead."
                );
                std::process::exit(1);
            }
        };

        eprintln!(
            "Backend:  Metal | {} | batch: {}",
            device.name(),
            config.batch
        );

        // The GPU is the single worker; run_search's threading is unused.
        config.threads = 1;

        let factory = config.factory;
        let code_hash = config.code_hash;
        let batch = config.batch;
        // Retained Metal handles are not Send, so they cannot cross into the
        // worker closure; recreate the device there. Dropping it here also
        // proves the probe above did not leave a live reference behind.
        drop(device);

        run_search(&config, move |stop: &AtomicBool, attempts: &AtomicU64| {
            mine(&factory, &code_hash, &value, &mask, batch, stop, attempts)
        });
    }

    /// One dispatch's worth of GPU work, kept alive while it runs so the CPU
    /// can process the previous dispatch concurrently.
    struct InFlight {
        cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        buf_idx: usize,
    }

    fn mine(
        factory: &[u8; 20],
        code_hash: &[u8; 32],
        value: &[u8; 20],
        mask: &[u8; 20],
        batch: usize,
        stop: &AtomicBool,
        attempts: &AtomicU64,
    ) -> Option<([u8; 32], [u8; 20])> {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device disappeared");
        let queue = device.newCommandQueue().expect("failed to create a command queue");
        let pipeline = build_pipeline(&device, "mine");

        // Random 24-byte salt prefix; trailing 8 bytes are the GPU counter.
        let mut salt_prefix = [0u8; 24];
        rand::rng().fill_bytes(&mut salt_prefix);
        let mut cbytes = [0u8; 8];
        rand::rng().fill_bytes(&mut cbytes);
        let mut counter = u64::from_le_bytes(cbytes);

        let block = stage1_block(factory, &salt_prefix, code_hash);
        let mut params = Params::new(&block, value, mask);

        // Two hit buffers so the CPU can process dispatch N while N+1 runs.
        let hit_buffers = [
            SharedBuffer::new(&device, HITS_LEN),
            SharedBuffer::new(&device, HITS_LEN),
        ];
        for hb in &hit_buffers {
            unsafe { (hb.as_mut_ptr() as *mut u32).write_unaligned(0) };
        }

        let tg_width = pipeline.maxTotalThreadsPerThreadgroup().min(256);
        let mut inflight: Option<InFlight> = None;

        loop {
            let buf_idx = inflight.as_ref().map_or(0, |f| 1 - f.buf_idx);
            params.set_base_counter(counter);
            counter = counter.wrapping_add(batch as u64);

            let cb = dispatch(&queue, &pipeline, &params, &hit_buffers[buf_idx], batch, tg_width);

            if let Some(prev) = inflight.take() {
                if let Some(found) = collect(
                    &prev, &hit_buffers, batch, &salt_prefix, factory, code_hash, value, mask,
                    attempts, stop,
                ) {
                    cb.waitUntilCompleted();
                    return Some(found);
                }
                if stop.load(Ordering::Relaxed) {
                    cb.waitUntilCompleted();
                    return None;
                }
            }
            inflight = Some(InFlight { cb, buf_idx });
        }
    }

    /// Waits for a finished dispatch, verifies its hits on the CPU, and bumps
    /// the attempt counter. Returns a genuine match if one is found.
    #[allow(clippy::too_many_arguments)]
    fn collect(
        prev: &InFlight,
        hit_buffers: &[SharedBuffer; 2],
        batch: usize,
        salt_prefix: &[u8; 24],
        factory: &[u8; 20],
        code_hash: &[u8; 32],
        value: &[u8; 20],
        mask: &[u8; 20],
        attempts: &AtomicU64,
        stop: &AtomicBool,
    ) -> Option<([u8; 32], [u8; 20])> {
        prev.cb.waitUntilCompleted();
        let hits = hit_buffers[prev.buf_idx].drain_hits();
        attempts.fetch_add(batch as u64, Ordering::Relaxed);

        for c in hits {
            let mut salt = [0u8; 32];
            salt[..24].copy_from_slice(salt_prefix);
            salt[24..32].copy_from_slice(&c.to_be_bytes());
            let addr = create3_address(factory, &salt, code_hash);
            if mask_match(&addr, value, mask) {
                stop.store(true, Ordering::Relaxed);
                return Some((salt, addr));
            }
        }
        None
    }

    fn build_pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        function: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let source = NSString::from_str(include_str!("kernel.metal"));
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .expect("Metal kernel failed to compile");
        let func = library
            .newFunctionWithName(&NSString::from_str(function))
            .expect("kernel function not found");
        device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("failed to create the compute pipeline")
    }

    fn dispatch(
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        params: &Params,
        hits: &SharedBuffer,
        batch: usize,
        tg_width: usize,
    ) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        let cb = queue.commandBuffer().expect("failed to create a command buffer");
        let enc = cb.computeCommandEncoder().expect("failed to create an encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            let ptr = std::ptr::NonNull::new(params.blob.as_ptr() as *mut std::ffi::c_void).unwrap();
            enc.setBytes_length_atIndex(ptr, PARAMS_LEN, 0);
            enc.setBuffer_offset_atIndex(Some(&hits.buf), 0, 1);
        }
        let grid = MTLSize { width: batch, height: 1, depth: 1 };
        let tg = MTLSize { width: tg_width, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cb.commit();
        cb
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use create3_miner::{parse_address, DEFAULT_PROXY_CODE_HASH};
        use std::ffi::c_void;
        use std::ptr::NonNull;

        fn device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
            MTLCreateSystemDefaultDevice()
        }

        /// Runs a single-buffer kernel dispatch (params at index 0, output at
        /// index 1) over `n` threads and blocks until it finishes.
        fn run_kernel(
            device: &ProtocolObject<dyn MTLDevice>,
            kernel: &str,
            params: &Params,
            out: &SharedBuffer,
            n: usize,
        ) {
            let queue = device.newCommandQueue().unwrap();
            let pipeline = build_pipeline(device, kernel);
            let cb = queue.commandBuffer().unwrap();
            let enc = cb.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pipeline);
            unsafe {
                let ptr = NonNull::new(params.blob.as_ptr() as *mut c_void).unwrap();
                enc.setBytes_length_atIndex(ptr, PARAMS_LEN, 0);
                enc.setBuffer_offset_atIndex(Some(&out.buf), 0, 1);
            }
            let tg = pipeline.maxTotalThreadsPerThreadgroup().min(256);
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize { width: n, height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
        }

        /// Extracts the 20-byte address from final-state words 1..3 (the same
        /// hash-byte-12..32 slice the miner matches on).
        fn addr_from_words(w1: u64, w2: u64, w3: u64) -> [u8; 20] {
            let mut addr = [0u8; 20];
            addr[0..4].copy_from_slice(&w1.to_le_bytes()[4..8]);
            addr[4..12].copy_from_slice(&w2.to_le_bytes());
            addr[12..20].copy_from_slice(&w3.to_le_bytes());
            addr
        }

        #[test]
        fn interleave_roundtrip() {
            fn deinterleave(lo: u32, hi: u32) -> u64 {
                let mut v = 0u64;
                for i in 0..32 {
                    v |= (((lo >> i) & 1) as u64) << (2 * i);
                    v |= (((hi >> i) & 1) as u64) << (2 * i + 1);
                }
                v
            }
            for x in [0u64, 1, 0xffff_ffff_ffff_ffff, 0x0123_4567_89ab_cdef, 0x8000_0000_0000_0001] {
                let (lo, hi) = interleave64(x);
                assert_eq!(deinterleave(lo, hi), x, "roundtrip {x:#x}");
            }
        }

        #[test]
        fn metal_derivation_matches_scalar() {
            let Some(device) = device() else {
                eprintln!("no Metal device; skipping");
                return;
            };
            let factory = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
            let prefix = [0x11u8; 24];
            let n = 512usize;
            let base = 0x0102_0304_0500_0000u64;

            for code_hash in [DEFAULT_PROXY_CODE_HASH, [0xABu8; 32]] {
                let block = stage1_block(&factory, &prefix, &code_hash);
                let mut params = Params::new(&block, &[0u8; 20], &[0u8; 20]);
                params.set_base_counter(base);

                let out = SharedBuffer::new(&device, n * 3 * 8);
                run_kernel(&device, "debug_addresses", &params, &out, n);

                let ptr = out.as_mut_ptr() as *const u64;
                for i in 0..n {
                    let c = base + i as u64;
                    let (w1, w2, w3) = unsafe {
                        (
                            ptr.add(3 * i).read_unaligned(),
                            ptr.add(3 * i + 1).read_unaligned(),
                            ptr.add(3 * i + 2).read_unaligned(),
                        )
                    };
                    let got = addr_from_words(w1, w2, w3);

                    let mut salt = [0u8; 32];
                    salt[..24].copy_from_slice(&prefix);
                    salt[24..32].copy_from_slice(&c.to_be_bytes());
                    let want = create3_address(&factory, &salt, &code_hash);
                    assert_eq!(got, want, "counter {c} code_hash {code_hash:02x?}");
                }
            }
        }

        #[test]
        fn metal_hit_path_reports_counter() {
            let Some(device) = device() else {
                eprintln!("no Metal device; skipping");
                return;
            };
            let factory = parse_address("0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf").unwrap();
            let prefix = [0x22u8; 24];
            let code_hash = DEFAULT_PROXY_CODE_HASH;
            let base = 0xdead_0000u64;
            let n = 4096usize;
            let target_counter = base + 1234;

            // Exact-match pattern for the address at target_counter.
            let mut salt = [0u8; 32];
            salt[..24].copy_from_slice(&prefix);
            salt[24..32].copy_from_slice(&target_counter.to_be_bytes());
            let target = create3_address(&factory, &salt, &code_hash);

            let block = stage1_block(&factory, &prefix, &code_hash);
            let mut params = Params::new(&block, &target, &[0xffu8; 20]);
            params.set_base_counter(base);

            let hits = SharedBuffer::new(&device, HITS_LEN);
            unsafe { (hits.as_mut_ptr() as *mut u32).write_unaligned(0) };
            run_kernel(&device, "mine", &params, &hits, n);

            let reported = hits.drain_hits();
            assert!(
                reported.contains(&target_counter),
                "expected counter {target_counter} in hits {reported:?}"
            );
            // The planted address is the only exact match in the range.
            assert_eq!(reported.len(), 1, "unexpected extra hits: {reported:?}");
        }
    }
}
