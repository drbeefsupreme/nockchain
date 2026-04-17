//! Quick experiment binary: force-cold the PMA slab and compare cold vs warm
//! peek behavior for a contiguous height range.

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(
    about = "Force the PMA slab cold via MADV_PAGEOUT, then peek a range twice to compare cold vs warm reads"
)]
struct Args {
    #[arg(long, default_value = "checkpoints/0.chkjam")]
    checkpoint: PathBuf,

    #[arg(long, default_value = "assets/dumb.jam")]
    kernel: PathBuf,

    #[arg(long, default_value_t = 1)]
    start_height: u64,

    #[arg(long, default_value_t = 100)]
    count: u64,

    #[arg(long, default_value_t = false)]
    fsync: bool,

    /// Byte count to write to cgroup v2 memory.reclaim (with swappiness=0).
    /// Defaults to 16 GiB. Pass 0 to skip the cgroup reclaim phase.
    #[arg(long, default_value_t = 16u64 * 1024 * 1024 * 1024)]
    cgroup_reclaim_bytes: u64,
}

#[cfg(all(target_os = "linux", feature = "pma-runtime-compat"))]
#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let cgroup_reclaim = if args.cgroup_reclaim_bytes == 0 {
        None
    } else {
        Some(args.cgroup_reclaim_bytes)
    };
    nockchain_bench::speed_of_light::cold_warm_experiment::run_experiment(
        &args.checkpoint, &args.kernel, args.start_height, args.count, args.fsync, cgroup_reclaim,
    )
    .await?;
    Ok(())
}

#[cfg(not(all(target_os = "linux", feature = "pma-runtime-compat")))]
fn main() {
    eprintln!("cold_warm_experiment requires target_os=linux and --features pma-runtime-compat");
    std::process::exit(2);
}
