use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use create3_miner::{
    hex_encode_addr, leading_match, parse_cli, run_search, MatchMode, MiningContext,
};

fn main() {
    let config = parse_cli("create3-miner");
    let factory = config.factory;
    let code_hash = config.code_hash;
    let mode = config.mode.clone();

    run_search(&config, move |stop: &AtomicBool, attempts: &AtomicU64| {
        let mut ctx = MiningContext::new(&factory, &code_hash);
        let mut local: u64 = 0;
        const BATCH: u64 = 8192;

        loop {
            for _ in 0..BATCH {
                let addr = ctx.next_addr();
                local += 1;

                let matched = match &mode {
                    MatchMode::Leading(nibbles) => leading_match(&addr, nibbles),
                    MatchMode::Regex(re) => {
                        hex_encode_addr(&addr, &mut ctx.hex_buf);
                        // hex_buf is always valid ASCII
                        let hex_str = unsafe { std::str::from_utf8_unchecked(&ctx.hex_buf) };
                        re.is_match(hex_str)
                    }
                };

                if matched {
                    attempts.fetch_add(local, Ordering::Relaxed);
                    stop.store(true, Ordering::Relaxed);
                    return Some((ctx.salt, addr));
                }
            }
            attempts.fetch_add(BATCH, Ordering::Relaxed);
            local = 0;
            if stop.load(Ordering::Relaxed) {
                return None;
            }
        }
    });
}
