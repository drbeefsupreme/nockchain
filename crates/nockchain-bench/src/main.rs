//! Simple CLI for testing smaps parsing
//!
//! Usage: cargo run -p nockchain-bench -- <pid>
//!        cargo run -p nockchain-bench -- self

use nockchain_bench::sampler::buckets::{sample_process, AttributionConfig};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let pid = if args.len() < 2 {
        eprintln!("Usage: {} <pid|self>", args[0]);
        eprintln!("  <pid>  - Process ID to sample");
        eprintln!("  self   - Sample this process");
        std::process::exit(1);
    } else if args[1] == "self" {
        std::process::id() as i32
    } else {
        args[1].parse().unwrap_or_else(|_| {
            eprintln!("Invalid PID: {}", args[1]);
            std::process::exit(1);
        })
    };

    // Optional: specify expected NockStack size
    let nockstack_size = args.get(2).and_then(|s| {
        let bytes: u64 = s.parse().ok()?;
        Some(bytes)
    });

    let config = match nockstack_size {
        Some(size) => AttributionConfig::with_nockstack_size(size),
        None => AttributionConfig::default(),
    };

    println!("Sampling process {} ...\n", pid);

    match sample_process(pid, &config, 0) {
        Ok(attr) => {
            println!("=== Memory Attribution for PID {} ===\n", pid);

            println!("Overall (from /proc/{}/status):", pid);
            println!("  VmRSS:      {:>10.1} MiB", kb_to_mib(attr.vm_rss_kb));
            println!("  VmSize:     {:>10.1} MiB", kb_to_mib(attr.vm_size_kb));
            println!("  RssAnon:    {:>10.1} MiB", kb_to_mib(attr.rss_anon_kb));
            println!("  RssFile:    {:>10.1} MiB", kb_to_mib(attr.rss_file_kb));
            println!("  VmSwap:     {:>10.1} MiB", kb_to_mib(attr.vm_swap_kb));
            println!();

            println!("Buckets (from /proc/{}/smaps):", pid);
            println!("  NockStack:  {:>10.1} MiB mapped, {:>10.1} MiB RSS",
                     kb_to_mib(attr.nockstack_size_kb), kb_to_mib(attr.nockstack_rss_kb));
            println!("  PMA:        {:>10.1} MiB mapped, {:>10.1} MiB RSS (ratio: {:.3})",
                     kb_to_mib(attr.pma_size_kb), kb_to_mib(attr.pma_rss_kb), attr.pma_rss_ratio());
            println!("  Heap/Other: {:>10.1} MiB mapped, {:>10.1} MiB RSS",
                     kb_to_mib(attr.heap_other_size_kb), kb_to_mib(attr.heap_other_rss_kb));
            println!("  SharedLibs: {:>10.1} MiB mapped, {:>10.1} MiB RSS",
                     kb_to_mib(attr.shared_libs_size_kb), kb_to_mib(attr.shared_libs_rss_kb));
            println!("  Stacks:     {:>10.1} MiB mapped, {:>10.1} MiB RSS",
                     kb_to_mib(attr.thread_stacks_size_kb), kb_to_mib(attr.thread_stacks_rss_kb));
            println!();

            println!("Page faults:");
            println!("  Minor: {}", attr.minor_faults);
            println!("  Major: {}", attr.major_faults);
            println!();

            let total_attributed = attr.total_attributed_rss_kb();
            let diff = (attr.vm_rss_kb as i64) - (total_attributed as i64);
            println!("Attribution check:");
            println!("  Total attributed RSS: {:>10.1} MiB", kb_to_mib(total_attributed));
            println!("  VmRSS from status:    {:>10.1} MiB", kb_to_mib(attr.vm_rss_kb));
            println!("  Difference:           {:>+10.1} MiB", diff as f64 / 1024.0);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

fn kb_to_mib(kb: u64) -> f64 {
    kb as f64 / 1024.0
}
