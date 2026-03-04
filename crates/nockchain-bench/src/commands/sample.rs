use nockchain_bench::sampler::buckets::{sample_process, AttributionConfig};

use super::kb_to_mib;

/// Sample a process's memory usage
pub fn cmd_sample(
    pid_str: &str,
    nockstack_size: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let pid = if pid_str == "self" {
        std::process::id() as i32
    } else {
        pid_str
            .parse()
            .map_err(|_| format!("Invalid PID: {}", pid_str))?
    };

    let config = match nockstack_size {
        Some(size) => AttributionConfig::with_nockstack_size(size),
        None => AttributionConfig::default(),
    };

    println!("Sampling process {} ...\n", pid);

    let attr = sample_process(pid, &config, 0)?;

    println!("=== Memory Attribution for PID {} ===\n", pid);

    println!("Overall (from /proc/{}/status):", pid);
    println!("  VmRSS:      {:>10.1} MiB", kb_to_mib(attr.vm_rss_kb));
    println!("  VmSize:     {:>10.1} MiB", kb_to_mib(attr.vm_size_kb));
    println!("  RssAnon:    {:>10.1} MiB", kb_to_mib(attr.rss_anon_kb));
    println!("  RssFile:    {:>10.1} MiB", kb_to_mib(attr.rss_file_kb));
    println!("  VmSwap:     {:>10.1} MiB", kb_to_mib(attr.vm_swap_kb));
    println!();

    println!("Buckets (from /proc/{}/smaps):", pid);
    println!(
        "  NockStack:  {:>10.1} MiB mapped, {:>10.1} MiB RSS",
        kb_to_mib(attr.nockstack_size_kb),
        kb_to_mib(attr.nockstack_rss_kb)
    );
    println!(
        "  Heap/Other: {:>10.1} MiB mapped, {:>10.1} MiB RSS",
        kb_to_mib(attr.heap_other_size_kb),
        kb_to_mib(attr.heap_other_rss_kb)
    );
    println!(
        "  SharedLibs: {:>10.1} MiB mapped, {:>10.1} MiB RSS",
        kb_to_mib(attr.shared_libs_size_kb),
        kb_to_mib(attr.shared_libs_rss_kb)
    );
    println!(
        "  Stacks:     {:>10.1} MiB mapped, {:>10.1} MiB RSS",
        kb_to_mib(attr.thread_stacks_size_kb),
        kb_to_mib(attr.thread_stacks_rss_kb)
    );
    println!();

    println!("Page faults:");
    println!("  Minor: {}", attr.minor_faults);
    println!("  Major: {}", attr.major_faults);
    println!();

    let total_attributed = attr.total_attributed_rss_kb();
    let diff = (attr.vm_rss_kb as i64) - (total_attributed as i64);
    println!("Attribution check:");
    println!(
        "  Total attributed RSS: {:>10.1} MiB",
        kb_to_mib(total_attributed)
    );
    println!(
        "  VmRSS from status:    {:>10.1} MiB",
        kb_to_mib(attr.vm_rss_kb)
    );
    println!(
        "  Difference:           {:>+10.1} MiB",
        diff as f64 / 1024.0
    );

    Ok(())
}
