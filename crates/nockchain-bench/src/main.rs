//! Nockchain Bench CLI
//!
//! Benchmarking and memory profiling tool for Nockchain.
//!
//! Usage:
//!   nockchain-bench sample <pid|self>           # Sample process memory
//!   nockchain-bench run [OPTIONS]               # Run a mining scenario
//!   nockchain-bench attach <container>          # Attach to existing container
//!   nockchain-bench compare [OPTIONS]           # A/B comparison
//!   nockchain-bench sol extract [OPTIONS]       # Extract blocks to archive
//!   nockchain-bench sol archive slice [OPTIONS] # Slice block ranges from archive
//!   nockchain-bench sol inspect [OPTIONS]       # Inspect mempool snapshots

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Once;
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};
use nockchain_bench::events::{EventCorrelator, LogParser};
use nockchain_bench::output::ParquetWriter;
use nockchain_bench::runner::{DockerRunner, NockchainMode};
use nockchain_bench::sampler::buckets::{sample_process, AttributionConfig};
use nockchain_bench::scenario::{MiningScenario, MiningScenarioConfig};
use nockchain_bench::speed_of_light::checkpoint::checkpoint_event_num;
use nockchain_bench::speed_of_light::{
    build_sweep_cases, checkpoint_durations_ms, extract_fixture_to_paths, find_stale_ranges,
    page_fault_bursts, read_fixture_file, slice_archive_file, summarize_case_runs,
    write_fixture_file_from_paths, ArchiveExtractionPhase, ArchiveReader, BenchConfig, BenchRunner,
    BlockExtractor, CheckpointBuilder, CheckpointConfig, ExtractorConfig, FixtureBuildConfig,
    FixtureBuildPhase, FixtureBuilder, FixtureCheckpointMode, NockStackProfile, SolFixtureManifest,
    SolHeight, SweepRunMetrics, PROOF_VERSION_1_START, PROOF_VERSION_2_START,
};
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

#[derive(Parser)]
#[command(name = "nockchain-bench")]
#[command(about = "Benchmarking and memory profiling tool for Nockchain")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Sample memory usage of a running process
    Sample {
        /// Process ID to sample, or "self" for this process
        pid: String,

        /// Expected NockStack size in bytes (for attribution)
        #[arg(long)]
        nockstack_size: Option<u64>,
    },

    /// Run a mining benchmark scenario
    Run {
        /// Scenario name (used in output files)
        #[arg(short, long, default_value = "benchmark")]
        name: String,

        /// Persistence mode
        #[arg(short, long, value_enum, default_value = "checkpoint")]
        mode: PersistenceMode,

        /// Checkpoint save interval in seconds (only for checkpoint mode)
        #[arg(long, default_value = "120")]
        save_interval: u64,

        /// Duration to run in seconds
        #[arg(short, long, default_value = "300")]
        duration: u64,

        /// Sample interval in seconds
        #[arg(long, default_value = "1")]
        sample_interval: u64,

        /// Docker image to use
        #[arg(long, default_value = "nockchain-local:latest")]
        image: String,

        /// Data directory on host
        #[arg(long, default_value = "/tmp/nockchain-bench")]
        data_dir: PathBuf,

        /// Memory limit (e.g., "16g", "8192m")
        #[arg(long, default_value = "16g")]
        memory_limit: String,

        /// Number of mining threads
        #[arg(long, default_value = "1")]
        threads: u32,

        /// Output directory for results
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Attach to an existing container and collect stats
    Attach {
        /// Container name or ID
        container: String,

        /// Duration to collect stats in seconds
        #[arg(short, long, default_value = "60")]
        duration: u64,

        /// Sample interval in seconds
        #[arg(long, default_value = "1")]
        sample_interval: u64,

        /// Output directory for results
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Run A/B comparison between checkpoint and PMA persist modes
    Compare {
        /// Duration to run each scenario in seconds
        #[arg(short, long, default_value = "300")]
        duration: u64,

        /// Sample interval in seconds
        #[arg(long, default_value = "1")]
        sample_interval: u64,

        /// Checkpoint save interval in seconds
        #[arg(long, default_value = "120")]
        save_interval: u64,

        /// Docker image to use
        #[arg(long, default_value = "nockchain-local:latest")]
        image: String,

        /// Base data directory (scenarios use subdirs)
        #[arg(long, default_value = "/tmp/nockchain-bench")]
        data_dir: PathBuf,

        /// Memory limit (e.g., "16g")
        #[arg(long, default_value = "16g")]
        memory_limit: String,

        /// Number of mining threads
        #[arg(long, default_value = "1")]
        threads: u32,

        /// Output directory for results
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Analyze a container with event correlation
    Analyze {
        /// Container name or ID
        container: String,

        /// Duration to collect stats in seconds
        #[arg(short, long, default_value = "30")]
        duration: u64,

        /// Sample interval in seconds
        #[arg(long, default_value = "1")]
        sample_interval: u64,

        /// Memory spike threshold percentage (show spikes > this)
        #[arg(long, default_value = "5.0")]
        spike_threshold: f64,

        /// Show all events (not just significant ones)
        #[arg(long)]
        all_events: bool,
    },

    /// Speed-of-light benchmark commands
    #[command(subcommand)]
    Sol(SolCommands),
}

#[derive(Subcommand)]
enum SolCommands {
    /// Extract blocks from a checkpoint to an archive file
    Extract {
        /// Number of blocks to extract
        #[arg(short = 'n', long, default_value = "1000")]
        blocks: u64,

        /// Start block height (inclusive)
        #[arg(long, default_value = "0")]
        start_height: u64,

        /// End block height (inclusive). If set, overrides --blocks.
        #[arg(long)]
        end_height: Option<u64>,

        /// Path to checkpoint file
        #[arg(short, long, default_value = "0.chkjam")]
        checkpoint: PathBuf,

        /// Path to kernel jam file
        #[arg(short, long, default_value = "assets/dumb.jam")]
        kernel: PathBuf,

        /// Output archive path (defaults to blocks_<N>.solarch)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Chunk size for range queries
        #[arg(long, default_value = "8")]
        chunk_size: u64,

        /// Include mempool snapshots in the archive
        #[arg(long)]
        include_mempool: bool,

        /// Nock stack profile for kernel initialization
        #[arg(long, value_enum)]
        stack_profile: Option<StackProfileArg>,
    },

    /// Run the speed-of-light benchmark from a unified fixture (`.soltest`)
    Bench {
        /// Path to a unified `.soltest` fixture file (includes checkpoint + archive + kernel)
        #[arg(short, long)]
        fixture: PathBuf,

        /// Number of blocks to benchmark (0 = all in archive)
        #[arg(short = 'n', long, default_value = "0")]
        blocks: u64,

        /// Enable kernel checkpointing mode (true/false)
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        enable_checkpointing: bool,

        /// Skip genesis block (block 0) - not recommended
        #[arg(long)]
        skip_genesis: bool,

        /// Enable process memory timeline profiling during benchmark replay
        #[arg(long)]
        profile_memory: bool,

        /// Memory profile sample interval in milliseconds
        #[arg(long, default_value = "500")]
        profile_interval_ms: u64,

        /// Write benchmark + memory profile JSON to this path
        #[arg(long)]
        profile_output: Option<PathBuf>,

        /// Force checkpoint every N accepted blocks (0 disables)
        #[arg(long, default_value = "0")]
        checkpoint_every_blocks: u64,

        /// Max wait for post-checkpoint RSS recovery in ms
        #[arg(long, default_value = "5000")]
        checkpoint_recovery_timeout_ms: u64,

        /// Recovery threshold as percent above pre-checkpoint baseline RSS
        #[arg(long, default_value = "5.0")]
        checkpoint_recovery_tolerance_pct: f64,

        /// Inferred GC threshold in MiB (RSS drop >= threshold)
        #[arg(long, default_value = "64")]
        gc_drop_threshold_mib: u64,

        /// Minor page-fault delta threshold for burst detection
        #[arg(long, default_value = "50000")]
        page_fault_minor_burst_threshold: u64,

        /// Major page-fault delta threshold for burst detection
        #[arg(long, default_value = "1")]
        page_fault_major_burst_threshold: u64,

        /// Nock stack profile for kernel initialization
        #[arg(long, value_enum)]
        stack_profile: Option<StackProfileArg>,

        /// Enable nock interpreter tracing
        #[arg(long, value_enum)]
        trace: Option<TraceModeArg>,
    },

    /// Build a checkpoint by replaying blocks from an archive
    Checkpoint {
        /// Path to the archive file
        #[arg(short, long, default_value = "blocks_1000.solarch")]
        archive: PathBuf,

        /// Path to kernel jam file
        #[arg(short, long, default_value = "assets/dumb.jam")]
        kernel: PathBuf,

        /// Existing checkpoint to start from (optional)
        #[arg(long)]
        checkpoint: Option<PathBuf>,

        /// Target block height to checkpoint at (inclusive)
        #[arg(long)]
        target_height: Option<u64>,

        /// Cutover to build checkpoint for (v1 or v2)
        #[arg(long, value_enum)]
        cutover: Option<CutoverVersion>,

        /// Start height override (defaults to checkpoint height + 1 if checkpoint provided)
        #[arg(long)]
        start_height: Option<u64>,

        /// Output checkpoint path (single file)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Working directory for checkpoint snapshots
        #[arg(long)]
        work_dir: Option<PathBuf>,

        /// Nock stack profile for kernel initialization
        #[arg(long, value_enum)]
        stack_profile: Option<StackProfileArg>,
    },

    /// Inspect mempool snapshots for stale transactions
    Inspect {
        /// Path to the archive file
        #[arg(short, long, default_value = "blocks_1000.solarch")]
        archive: PathBuf,

        /// Retention threshold in blocks (age >= retain is considered stale)
        #[arg(long, default_value = "20")]
        retain: u64,
    },

    /// Archive utility commands
    #[command(subcommand)]
    Archive(ArchiveCommands),

    /// Sweep PMA candidates/chunk sizes/memory limits and summarize checkpoint behavior
    Sweep {
        /// PMA candidate IDs (comma-separated)
        #[arg(long)]
        candidates: String,

        /// Streaming checkpoint chunk sizes (comma-separated)
        #[arg(long)]
        chunk_sizes: String,

        /// Memory limits, e.g. "8g,12g,16g"
        #[arg(long)]
        memory_limits: String,

        /// Repetitions per matrix cell for variance estimates
        #[arg(long, default_value = "1")]
        repeats: u32,

        /// Duration per run in seconds
        #[arg(long, default_value = "300")]
        duration: u64,

        /// Sample interval in seconds
        #[arg(long, default_value = "1")]
        sample_interval: u64,

        /// Checkpoint save interval in seconds
        #[arg(long, default_value = "120")]
        save_interval: u64,

        /// Docker image to use
        #[arg(long, default_value = "nockchain-local:latest")]
        image: String,

        /// Base directory for run data
        #[arg(long, default_value = "/tmp/nockchain-bench-sweep")]
        data_dir: PathBuf,

        /// Mining threads
        #[arg(long, default_value = "1")]
        threads: u32,

        /// Env var name used to select PMA candidate
        #[arg(long, default_value = "NOCK_PMA_CANDIDATE")]
        candidate_env: String,

        /// Env var name used to set streaming chunk size
        #[arg(long, default_value = "NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE")]
        chunk_env: String,

        /// Optional JSON output path for sweep results
        #[arg(long)]
        output_json: Option<PathBuf>,
    },

    /// Build and inspect unified SOL fixture bundles (`.soltest`)
    #[command(subcommand)]
    Fixture(FixtureCommands),
}

#[derive(Subcommand)]
enum FixtureCommands {
    /// Build a fixture from a source checkpoint and block-height range
    Build {
        /// Source checkpoint path (must include the requested block range)
        #[arg(long)]
        source_checkpoint: PathBuf,

        /// Kernel jam path
        #[arg(short, long, default_value = "assets/dumb.jam")]
        kernel: PathBuf,

        /// Start block height for test archive (inclusive)
        #[arg(long)]
        start_height: u64,

        /// End block height for test archive (inclusive)
        #[arg(long)]
        end_height: u64,

        /// Output fixture path (base path when --checkpoint-mode=both)
        #[arg(short, long)]
        output: PathBuf,

        /// Chunk size (blocks per range fetch)
        #[arg(long, default_value = "8")]
        chunk_size: u64,

        /// Checkpoint mode embedded in the fixture
        #[arg(long, value_enum, default_value = "derived")]
        checkpoint_mode: FixtureCheckpointModeArg,

        /// Nock stack profile for kernel initialization (defaults by checkpoint mode)
        #[arg(long, value_enum)]
        stack_profile: Option<StackProfileArg>,

        /// Include mempool snapshots in the test archive
        #[arg(long)]
        include_mempool: bool,

        /// Working directory for temporary artifacts
        #[arg(long, default_value = ".")]
        work_dir: PathBuf,

        /// Minimum interval between progress log lines
        #[arg(long, default_value = "5")]
        progress_interval_secs: u64,
    },

    /// Inspect fixture metadata and embedded payload sizes
    Inspect {
        /// Fixture path
        #[arg(short, long)]
        fixture: PathBuf,
    },

    /// Package a fixture from existing checkpoint/archive/kernel files
    Pack {
        /// Embedded checkpoint path for the fixture payload
        #[arg(long)]
        checkpoint: PathBuf,

        /// Source checkpoint path recorded in manifest (defaults to --checkpoint)
        #[arg(long)]
        source_checkpoint: Option<PathBuf>,

        /// Test archive path
        #[arg(long)]
        archive: PathBuf,

        /// Kernel jam path
        #[arg(short, long, default_value = "assets/dumb.jam")]
        kernel: PathBuf,

        /// Archive start block height (inclusive)
        #[arg(long)]
        archive_start_height: u64,

        /// Archive end block height (inclusive)
        #[arg(long)]
        archive_end_height: u64,

        /// Embedded checkpoint height for manifest metadata
        #[arg(long)]
        checkpoint_height: Option<u64>,

        /// Chunk size metadata
        #[arg(long, default_value = "8")]
        chunk_size: u64,

        /// Include mempool snapshots metadata
        #[arg(long)]
        include_mempool: bool,

        /// Output fixture path
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[derive(Subcommand)]
enum ArchiveCommands {
    /// Copy a block-height range from one archive into a new archive
    Slice {
        /// Input archive file
        #[arg(short, long, default_value = "blocks_1000.solarch")]
        archive: PathBuf,

        /// Start block height for slice (inclusive)
        #[arg(long)]
        start_height: u64,

        /// End block height for slice (inclusive)
        #[arg(long)]
        end_height: u64,

        /// Output archive file
        #[arg(short, long)]
        output: PathBuf,

        /// Working directory for temporary stream files
        #[arg(long, default_value = ".")]
        work_dir: PathBuf,

        /// Include mempool snapshots in the sliced range
        #[arg(long)]
        include_mempool: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FixtureCheckpointModeArg {
    Derived,
    Full,
    Both,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StackProfileArg {
    Medium,
    Large,
    Huge,
}

impl StackProfileArg {
    fn into_profile(self) -> NockStackProfile {
        match self {
            Self::Medium => NockStackProfile::Medium,
            Self::Large => NockStackProfile::Large,
            Self::Huge => NockStackProfile::Huge,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TraceModeArg {
    Tracing,
}

impl TraceModeArg {
    fn into_trace_mode(self) -> nockapp::kernel::boot::TraceMode {
        match self {
            Self::Tracing => nockapp::kernel::boot::TraceMode::Tracing,
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum PersistenceMode {
    /// Checkpoint mode with periodic saves
    Checkpoint,
    /// PMA persistence mode
    PmaPersist,
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    /// Human-readable text output
    Text,
    /// JSON output
    Json,
    /// Parquet files (requires --output)
    Parquet,
}

#[derive(Clone, Debug, ValueEnum)]
enum CutoverVersion {
    V1,
    V2,
}

fn init_tracing() {
    static TRACING_INIT: Once = Once::new();
    TRACING_INIT.call_once(|| {
        let env_filter =
            EnvFilter::new(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()));
        let reg = tracing_subscriber::registry().with(env_filter);

        if std::env::var("TRACY_DISABLE").is_err() {
            let tracy = tracing_tracy::TracyLayer::default();
            if std::env::var("TRACY_ONLY_NOCKCODE").is_ok() {
                reg.with(tracy.with_filter(filter_fn(|meta| meta.target() == "nockcode")))
                    .init();
            } else {
                reg.with(tracy).init();
            }
        } else {
            reg.init();
        }
    });
}

#[tokio::main]
async fn main() {
    init_tracing();
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Sample {
            pid,
            nockstack_size,
        } => cmd_sample(&pid, nockstack_size),
        Commands::Run {
            name,
            mode,
            save_interval,
            duration,
            sample_interval,
            image,
            data_dir,
            memory_limit,
            threads,
            output,
            format,
        } => {
            cmd_run(
                &name, mode, save_interval, duration, sample_interval, &image, data_dir,
                &memory_limit, threads, output, format,
            )
            .await
        }
        Commands::Attach {
            container,
            duration,
            sample_interval,
            output,
            format,
        } => cmd_attach(&container, duration, sample_interval, output, format).await,
        Commands::Compare {
            duration,
            sample_interval,
            save_interval,
            image,
            data_dir,
            memory_limit,
            threads,
            output,
        } => {
            cmd_compare(
                duration, sample_interval, save_interval, &image, data_dir, &memory_limit, threads,
                output,
            )
            .await
        }
        Commands::Analyze {
            container,
            duration,
            sample_interval,
            spike_threshold,
            all_events,
        } => {
            cmd_analyze(
                &container, duration, sample_interval, spike_threshold, all_events,
            )
            .await
        }
        Commands::Sol(sol_cmd) => match sol_cmd {
            SolCommands::Extract {
                blocks,
                start_height,
                end_height,
                checkpoint,
                kernel,
                output,
                chunk_size,
                include_mempool,
                stack_profile,
            } => {
                let stack_profile = stack_profile
                    .unwrap_or(StackProfileArg::Large)
                    .into_profile();
                cmd_sol_extract(
                    blocks, start_height, end_height, checkpoint, kernel, output, chunk_size,
                    include_mempool, stack_profile,
                )
                .await
            }
            SolCommands::Bench {
                fixture,
                blocks,
                enable_checkpointing,
                skip_genesis,
                profile_memory,
                profile_interval_ms,
                profile_output,
                checkpoint_every_blocks,
                checkpoint_recovery_timeout_ms,
                checkpoint_recovery_tolerance_pct,
                gc_drop_threshold_mib,
                page_fault_minor_burst_threshold,
                page_fault_major_burst_threshold,
                stack_profile,
                trace,
            } => {
                let stack_profile = stack_profile
                    .unwrap_or(StackProfileArg::Medium)
                    .into_profile();
                cmd_sol_bench(
                    fixture, blocks, enable_checkpointing, skip_genesis, profile_memory,
                    profile_interval_ms, profile_output, checkpoint_every_blocks,
                    checkpoint_recovery_timeout_ms, checkpoint_recovery_tolerance_pct,
                    gc_drop_threshold_mib, page_fault_minor_burst_threshold,
                    page_fault_major_burst_threshold, stack_profile, trace,
                )
                .await
            }
            SolCommands::Checkpoint {
                archive,
                kernel,
                checkpoint,
                target_height,
                cutover,
                start_height,
                output,
                work_dir,
                stack_profile,
            } => {
                let stack_profile = stack_profile
                    .unwrap_or(StackProfileArg::Large)
                    .into_profile();
                cmd_sol_checkpoint(
                    archive, kernel, checkpoint, target_height, cutover, start_height, output,
                    work_dir, stack_profile,
                )
                .await
            }
            SolCommands::Inspect { archive, retain } => cmd_sol_inspect(archive, retain),
            SolCommands::Archive(ArchiveCommands::Slice {
                archive,
                start_height,
                end_height,
                output,
                work_dir,
                include_mempool,
            }) => cmd_sol_archive_slice(
                archive, start_height, end_height, output, work_dir, include_mempool,
            ),
            SolCommands::Sweep {
                candidates,
                chunk_sizes,
                memory_limits,
                repeats,
                duration,
                sample_interval,
                save_interval,
                image,
                data_dir,
                threads,
                candidate_env,
                chunk_env,
                output_json,
            } => {
                cmd_sol_sweep(
                    &candidates, &chunk_sizes, &memory_limits, repeats, duration, sample_interval,
                    save_interval, &image, data_dir, threads, &candidate_env, &chunk_env,
                    output_json,
                )
                .await
            }
            SolCommands::Fixture(FixtureCommands::Build {
                source_checkpoint,
                kernel,
                start_height,
                end_height,
                output,
                chunk_size,
                checkpoint_mode,
                stack_profile,
                include_mempool,
                work_dir,
                progress_interval_secs,
            }) => {
                let stack_profile = stack_profile.map(StackProfileArg::into_profile);
                cmd_sol_fixture_build(
                    source_checkpoint, kernel, start_height, end_height, output, chunk_size,
                    checkpoint_mode, include_mempool, work_dir, stack_profile,
                    progress_interval_secs,
                )
                .await
            }
            SolCommands::Fixture(FixtureCommands::Inspect { fixture }) => {
                cmd_sol_fixture_inspect(fixture)
            }
            SolCommands::Fixture(FixtureCommands::Pack {
                checkpoint,
                source_checkpoint,
                archive,
                kernel,
                archive_start_height,
                archive_end_height,
                checkpoint_height,
                chunk_size,
                include_mempool,
                output,
            }) => cmd_sol_fixture_pack(
                checkpoint, source_checkpoint, archive, kernel, archive_start_height,
                archive_end_height, checkpoint_height, chunk_size, include_mempool, output,
            ),
        },
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Sample a process's memory usage
fn cmd_sample(
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
        "  PMA:        {:>10.1} MiB mapped, {:>10.1} MiB RSS (ratio: {:.3})",
        kb_to_mib(attr.pma_size_kb),
        kb_to_mib(attr.pma_rss_kb),
        attr.pma_rss_ratio()
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

/// Run a mining scenario
async fn cmd_run(
    name: &str,
    mode: PersistenceMode,
    save_interval: u64,
    duration: u64,
    sample_interval: u64,
    image: &str,
    data_dir: PathBuf,
    memory_limit: &str,
    threads: u32,
    output: Option<PathBuf>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let nockchain_mode = match mode {
        PersistenceMode::Checkpoint => NockchainMode::Checkpoint {
            save_interval_secs: save_interval,
        },
        PersistenceMode::PmaPersist => NockchainMode::PmaPersist,
    };

    let config = MiningScenarioConfig {
        name: name.to_string(),
        mode: nockchain_mode,
        duration: Duration::from_secs(duration),
        sample_interval: Duration::from_secs(sample_interval),
        image: image.to_string(),
        data_dir,
        memory_limit: Some(memory_limit.to_string()),
        num_threads: threads,
        ..Default::default()
    };

    let scenario = MiningScenario::new(config);

    println!("Running scenario: {}", name);
    println!("Mode: {:?}", mode);
    println!("Duration: {}s", duration);
    println!();

    let result = scenario.run().await?;

    // Output results based on format
    match format {
        OutputFormat::Text => {
            result.print_summary();
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&result)?;
            println!("{}", json);
        }
        OutputFormat::Parquet => {
            let output_dir = output.ok_or("--output is required for parquet format")?;
            std::fs::create_dir_all(&output_dir)?;

            let stats_path = output_dir.join(format!("{}_stats.parquet", name));
            let results_path = output_dir.join(format!("{}_results.parquet", name));

            let writer = ParquetWriter::new();
            writer.write_stats(&stats_path, name, &result.samples)?;
            writer.write_results(&results_path, &[&result])?;

            println!("Results written to:");
            println!("  Stats:   {}", stats_path.display());
            println!("  Summary: {}", results_path.display());
        }
    }

    Ok(())
}

/// Attach to an existing container and collect stats
async fn cmd_attach(
    container: &str,
    duration: u64,
    sample_interval: u64,
    output: Option<PathBuf>,
    format: OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Attaching to container: {}", container);

    let runner = DockerRunner::attach_to_existing(container).await?;

    println!(
        "Collecting stats for {}s at {}s intervals...\n",
        duration, sample_interval
    );

    let samples = runner
        .collect_stats(
            Duration::from_secs(duration),
            Duration::from_secs(sample_interval),
        )
        .await?;

    // Calculate summary stats
    let peak_memory = samples
        .iter()
        .map(|s| s.memory_usage_bytes)
        .max()
        .unwrap_or(0);
    let avg_memory = if samples.is_empty() {
        0
    } else {
        samples.iter().map(|s| s.memory_usage_bytes).sum::<u64>() / samples.len() as u64
    };
    let peak_rss = samples
        .iter()
        .map(|s| s.memory_rss_bytes)
        .max()
        .unwrap_or(0);
    let avg_rss = if samples.is_empty() {
        0
    } else {
        samples.iter().map(|s| s.memory_rss_bytes).sum::<u64>() / samples.len() as u64
    };

    match format {
        OutputFormat::Text => {
            println!("=== Stats for {} ===\n", container);
            println!("Samples collected: {}", samples.len());
            println!();
            println!("Memory Usage:");
            println!("  Peak:    {:>10.1} MiB", bytes_to_mib(peak_memory));
            println!("  Average: {:>10.1} MiB", bytes_to_mib(avg_memory));
            println!();
            println!("RSS:");
            println!("  Peak:    {:>10.1} MiB", bytes_to_mib(peak_rss));
            println!("  Average: {:>10.1} MiB", bytes_to_mib(avg_rss));
            println!();

            // Print time series
            println!("Time series:");
            println!(
                "{:>10} {:>12} {:>12} {:>10}",
                "Time (s)", "Memory (MiB)", "RSS (MiB)", "CPU %"
            );
            println!("{}", "-".repeat(50));
            for sample in &samples {
                println!(
                    "{:>10.1} {:>12.1} {:>12.1} {:>10.1}",
                    sample.timestamp_ms as f64 / 1000.0,
                    bytes_to_mib(sample.memory_usage_bytes),
                    bytes_to_mib(sample.memory_rss_bytes),
                    sample.cpu_percent
                );
            }
        }
        OutputFormat::Json => {
            let output = serde_json::json!({
                "container": container,
                "samples": samples.len(),
                "peak_memory_bytes": peak_memory,
                "avg_memory_bytes": avg_memory,
                "peak_rss_bytes": peak_rss,
                "avg_rss_bytes": avg_rss,
                "time_series": samples,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        OutputFormat::Parquet => {
            let output_dir = output.ok_or("--output is required for parquet format")?;
            std::fs::create_dir_all(&output_dir)?;

            let stats_path = output_dir.join(format!("{}_stats.parquet", container));

            let writer = ParquetWriter::new();
            writer.write_stats(&stats_path, container, &samples)?;

            println!("Stats written to: {}", stats_path.display());
        }
    }

    Ok(())
}

/// Run A/B comparison between checkpoint and PMA persist modes
async fn cmd_compare(
    duration: u64,
    sample_interval: u64,
    save_interval: u64,
    image: &str,
    data_dir: PathBuf,
    memory_limit: &str,
    threads: u32,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== A/B Comparison: Checkpoint vs PMA Persist ===\n");

    // Run checkpoint mode
    let checkpoint_config = MiningScenarioConfig {
        name: "checkpoint".to_string(),
        mode: NockchainMode::Checkpoint {
            save_interval_secs: save_interval,
        },
        duration: Duration::from_secs(duration),
        sample_interval: Duration::from_secs(sample_interval),
        image: image.to_string(),
        data_dir: data_dir.join("checkpoint"),
        memory_limit: Some(memory_limit.to_string()),
        num_threads: threads,
        ..Default::default()
    };

    println!("--- Running Checkpoint Mode ---");
    let checkpoint_scenario = MiningScenario::new(checkpoint_config);
    let checkpoint_result = checkpoint_scenario.run().await?;
    checkpoint_result.print_summary();

    // Clean up between runs
    println!("\nCleaning up...\n");
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Run PMA persist mode
    let pma_config = MiningScenarioConfig {
        name: "pma_persist".to_string(),
        mode: NockchainMode::PmaPersist,
        duration: Duration::from_secs(duration),
        sample_interval: Duration::from_secs(sample_interval),
        image: image.to_string(),
        data_dir: data_dir.join("pma_persist"),
        memory_limit: Some(memory_limit.to_string()),
        num_threads: threads,
        ..Default::default()
    };

    println!("--- Running PMA Persist Mode ---");
    let pma_scenario = MiningScenario::new(pma_config);
    let pma_result = pma_scenario.run().await?;
    pma_result.print_summary();

    // Print comparison
    println!("\n=== Comparison Summary ===\n");
    println!(
        "{:<20} {:>15} {:>15} {:>10}",
        "Metric", "Checkpoint", "PMA Persist", "Diff %"
    );
    println!("{}", "-".repeat(65));

    let print_comparison = |name: &str, checkpoint: f64, pma: f64| {
        let diff_pct = if checkpoint > 0.0 {
            ((pma - checkpoint) / checkpoint) * 100.0
        } else {
            0.0
        };
        println!(
            "{:<20} {:>12.1} MiB {:>12.1} MiB {:>+9.1}%",
            name, checkpoint, pma, diff_pct
        );
    };

    print_comparison(
        "Peak Memory",
        checkpoint_result.peak_memory_mib(),
        pma_result.peak_memory_mib(),
    );
    print_comparison(
        "Avg Memory",
        checkpoint_result.avg_memory_mib(),
        pma_result.avg_memory_mib(),
    );
    print_comparison(
        "Peak RSS",
        checkpoint_result.peak_rss_mib(),
        pma_result.peak_rss_mib(),
    );
    print_comparison(
        "Avg RSS",
        checkpoint_result.avg_rss_mib(),
        pma_result.avg_rss_mib(),
    );

    // Write output if requested
    if let Some(output_dir) = output {
        std::fs::create_dir_all(&output_dir)?;

        let writer = ParquetWriter::new();

        // Write combined stats
        let stats_path = output_dir.join("comparison_stats.parquet");
        writer.write_multi_stats(
            &stats_path,
            &[("checkpoint", &checkpoint_result.samples), ("pma_persist", &pma_result.samples)],
        )?;

        // Write results summary
        let results_path = output_dir.join("comparison_results.parquet");
        writer.write_results(&results_path, &[&checkpoint_result, &pma_result])?;

        // Write JSON summary
        let json_path = output_dir.join("comparison_summary.json");
        let summary = serde_json::json!({
            "checkpoint": {
                "peak_memory_mib": checkpoint_result.peak_memory_mib(),
                "avg_memory_mib": checkpoint_result.avg_memory_mib(),
                "peak_rss_mib": checkpoint_result.peak_rss_mib(),
                "avg_rss_mib": checkpoint_result.avg_rss_mib(),
                "samples": checkpoint_result.sample_count(),
                "success": checkpoint_result.success,
            },
            "pma_persist": {
                "peak_memory_mib": pma_result.peak_memory_mib(),
                "avg_memory_mib": pma_result.avg_memory_mib(),
                "peak_rss_mib": pma_result.peak_rss_mib(),
                "avg_rss_mib": pma_result.avg_rss_mib(),
                "samples": pma_result.sample_count(),
                "success": pma_result.success,
            },
            "comparison": {
                "peak_memory_diff_pct": ((pma_result.peak_memory_mib() - checkpoint_result.peak_memory_mib()) / checkpoint_result.peak_memory_mib()) * 100.0,
                "avg_memory_diff_pct": ((pma_result.avg_memory_mib() - checkpoint_result.avg_memory_mib()) / checkpoint_result.avg_memory_mib()) * 100.0,
            }
        });
        std::fs::write(&json_path, serde_json::to_string_pretty(&summary)?)?;

        println!("\nResults written to:");
        println!("  Stats:   {}", stats_path.display());
        println!("  Summary: {}", results_path.display());
        println!("  JSON:    {}", json_path.display());
    }

    Ok(())
}

/// Analyze a container with event correlation
async fn cmd_analyze(
    container: &str,
    duration: u64,
    sample_interval: u64,
    spike_threshold: f64,
    all_events: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Analyzing container: {} ===\n", container);

    let runner = DockerRunner::attach_to_existing(container).await?;

    // Get initial logs for context
    let initial_logs = runner.get_logs(100).await.unwrap_or_default();

    println!(
        "Collecting stats for {}s at {}s intervals...\n",
        duration, sample_interval
    );

    // Collect stats and logs in parallel
    let samples = runner
        .collect_stats(
            Duration::from_secs(duration),
            Duration::from_secs(sample_interval),
        )
        .await?;

    // Get logs after collection
    let final_logs = runner.get_logs(200).await.unwrap_or_default();

    // Combine and deduplicate logs
    let mut all_logs: Vec<String> = initial_logs;
    for log in final_logs {
        if !all_logs.contains(&log) {
            all_logs.push(log);
        }
    }

    // Parse logs into events
    let mut parser = LogParser::new();
    let events = parser.parse_lines(&all_logs);

    println!("Parsed {} events from logs\n", events.len());

    // Correlate events with samples
    let correlator = EventCorrelator::new().with_window_ms(1000);
    let correlated = correlator.correlate(&samples, &events);

    // Print correlated results
    println!(
        "{:>10} {:>12} {:>12} {:>10}  Events",
        "Time (s)", "Memory (MiB)", "RSS (MiB)", "CPU %"
    );
    println!("{}", "-".repeat(80));

    for sample in &correlated {
        let events_str = if all_events {
            sample
                .events
                .iter()
                .map(|e| e.event_type.label())
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            sample
                .significant_events()
                .iter()
                .map(|e| e.event_type.label())
                .collect::<Vec<_>>()
                .join(", ")
        };

        let events_display = if events_str.is_empty() {
            String::new()
        } else {
            format!("  {}", events_str)
        };

        println!(
            "{:>10.1} {:>12.1} {:>12.1} {:>10.1}{}",
            sample.stats.timestamp_ms as f64 / 1000.0,
            bytes_to_mib(sample.stats.memory_usage_bytes),
            bytes_to_mib(sample.stats.memory_rss_bytes),
            sample.stats.cpu_percent,
            events_display
        );
    }

    // Find and report memory spikes
    let spikes = correlator.find_spikes(&correlated, spike_threshold);

    if !spikes.is_empty() {
        println!(
            "\n=== Memory Spikes (>{:.1}% increase) ===\n",
            spike_threshold
        );
        println!(
            "{:>10} {:>12} {:>10}  Correlated Events",
            "Time (s)", "Memory (MiB)", "Change %"
        );
        println!("{}", "-".repeat(70));

        for (_idx, sample, change_pct) in &spikes {
            let events_str = sample
                .events
                .iter()
                .map(|e| {
                    format!(
                        "{}@{:.1}s",
                        e.event_type.label(),
                        e.timestamp_ms as f64 / 1000.0
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");

            println!(
                "{:>10.1} {:>12.1} {:>+10.1}%  {}",
                sample.stats.timestamp_ms as f64 / 1000.0,
                bytes_to_mib(sample.stats.memory_usage_bytes),
                change_pct,
                if events_str.is_empty() {
                    "(no events)"
                } else {
                    &events_str
                }
            );
        }
    } else {
        println!(
            "\nNo memory spikes detected (threshold: {:.1}%)",
            spike_threshold
        );
    }

    // Event summary
    let significant_count = events.iter().filter(|e| e.is_significant()).count();
    let block_count = events
        .iter()
        .filter(|e| {
            matches!(
                e.event_type,
                nockchain_bench::events::EventType::BlockAccepted { .. }
            )
        })
        .count();

    println!("\n=== Event Summary ===");
    println!("Total events:       {}", events.len());
    println!("Significant events: {}", significant_count);
    println!("Blocks accepted:    {}", block_count);

    Ok(())
}

/// Run speed-of-light benchmark (poke blocks as fast as possible)
async fn cmd_sol_bench(
    fixture: PathBuf,
    blocks: u64,
    enable_checkpointing: bool,
    skip_genesis: bool,
    profile_memory: bool,
    profile_interval_ms: u64,
    profile_output: Option<PathBuf>,
    checkpoint_every_blocks: u64,
    checkpoint_recovery_timeout_ms: u64,
    checkpoint_recovery_tolerance_pct: f64,
    gc_drop_threshold_mib: u64,
    page_fault_minor_burst_threshold: u64,
    page_fault_major_burst_threshold: u64,
    stack_profile: NockStackProfile,
    trace: Option<TraceModeArg>,
) -> Result<(), Box<dyn std::error::Error>> {
    struct TempDirGuard {
        path: PathBuf,
    }
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    if !fixture.exists() {
        return Err(format!("Fixture file not found: {}", fixture.display()).into());
    }
    if !enable_checkpointing && checkpoint_every_blocks > 0 {
        return Err(
            "--checkpoint-every-blocks requires --enable-checkpointing=true (or set cadence to 0)"
                .into(),
        );
    }

    let fixture_temp_dir = std::env::temp_dir().join(format!(
        "nockchain-bench-fixture-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&fixture_temp_dir)?;

    let checkpoint_path = fixture_temp_dir.join("fixture.chkjam");
    let archive_path = fixture_temp_dir.join("fixture.solarch");
    let kernel_path = fixture_temp_dir.join("fixture.jam");
    let manifest =
        extract_fixture_to_paths(&fixture, &checkpoint_path, &archive_path, &kernel_path)?;
    let archive_start_height = manifest.archive_start_height.as_u64();
    let archive_end_height = manifest.archive_end_height.as_u64();
    let fixture_temp_guard = TempDirGuard {
        path: fixture_temp_dir,
    };

    println!("=== Speed-of-Light Benchmark ===\n");
    println!("Fixture: {}", fixture.display());
    println!("Archive: {}", archive_path.display());
    println!("Kernel:  {}", kernel_path.display());
    println!("Checkpoint: {}", checkpoint_path.display());
    println!(
        "Archive range: {}..={}",
        archive_start_height, archive_end_height
    );
    println!(
        "Blocks:  {}",
        if blocks == 0 {
            "all".to_string()
        } else {
            blocks.to_string()
        }
    );
    println!("Checkpoint mode: {}", enable_checkpointing);
    println!(
        "Stack profile: {}",
        format!("{stack_profile:?}").to_lowercase()
    );
    println!(
        "Trace mode: {}",
        trace
            .map(|mode| format!("{mode:?}").to_lowercase())
            .unwrap_or_else(|| "off".to_string())
    );
    println!("Skip genesis: {}", skip_genesis);
    println!("Start height: {}", archive_start_height);
    println!("Profile memory: {}", profile_memory);
    if profile_memory {
        println!("Profile interval: {}ms", profile_interval_ms);
        println!("GC drop threshold: {} MiB", gc_drop_threshold_mib);
        println!(
            "Fault burst thresholds: minor={} major={}",
            page_fault_minor_burst_threshold, page_fault_major_burst_threshold
        );
    }
    if checkpoint_every_blocks > 0 {
        println!(
            "Checkpoint cadence: every {} blocks",
            checkpoint_every_blocks
        );
        println!(
            "Checkpoint recovery: timeout={}ms tolerance={}%",
            checkpoint_recovery_timeout_ms, checkpoint_recovery_tolerance_pct
        );
    }
    if let Some(ref out) = profile_output {
        println!("Profile output: {}", out.display());
    }
    println!();

    // Check files exist
    if !archive_path.exists() {
        return Err(format!("Archive file not found: {}", archive_path.display()).into());
    }
    if !kernel_path.exists() {
        return Err(format!("Kernel file not found: {}", kernel_path.display()).into());
    }
    if !checkpoint_path.exists() {
        return Err(format!("Checkpoint file not found: {}", checkpoint_path.display()).into());
    }

    let config = BenchConfig {
        archive_path: archive_path.to_string_lossy().to_string(),
        kernel_path: kernel_path.to_string_lossy().to_string(),
        block_count: blocks,
        skip_genesis,
        proof_version: None,
        checkpoint_path: Some(checkpoint_path.to_string_lossy().to_string()),
        start_height: Some(SolHeight(archive_start_height)),
        enable_checkpointing,
        profile_memory,
        profile_interval_ms,
        gc_drop_threshold_bytes: gc_drop_threshold_mib.saturating_mul(1024 * 1024),
        page_fault_minor_burst_threshold,
        page_fault_major_burst_threshold,
        checkpoint_every_blocks,
        checkpoint_recovery_timeout_ms,
        checkpoint_recovery_tolerance_pct,
        work_dir: PathBuf::from("."),
        stack_profile,
        trace_mode: trace.map(|mode| mode.into_trace_mode()),
    };

    let mut runner = BenchRunner::new(config);

    println!("Initializing fresh kernel (this may take a few minutes)...");
    let results = runner.run().await?;

    results.print_summary();

    if let Some(path) = profile_output {
        let checkpoint_avg_secs = results
            .avg_checkpoint_time()
            .map(|duration| duration.as_secs_f64());
        let payload = serde_json::json!({
            "blocks_poked": results.blocks_poked,
            "failed_pokes": results.failed_pokes,
            "init_time_secs": results.init_time.as_secs_f64(),
            "total_poke_time_secs": results.total_poke_time.as_secs_f64(),
            "blocks_per_second": results.blocks_per_second(),
            "checkpoint_count": results.checkpoint_count,
            "checkpoint_total_time_secs": results.checkpoint_total_time.as_secs_f64(),
            "checkpoint_avg_time_secs": checkpoint_avg_secs,
            "memory_profile": results.memory_profile,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&payload)?)?;
        println!("Profile JSON written to {}", path.display());
    }

    drop(fixture_temp_guard);
    Ok(())
}

/// Build a checkpoint by replaying archived blocks
async fn cmd_sol_checkpoint(
    archive: PathBuf,
    kernel: PathBuf,
    checkpoint: Option<PathBuf>,
    target_height: Option<u64>,
    cutover: Option<CutoverVersion>,
    start_height: Option<u64>,
    output: Option<PathBuf>,
    work_dir: Option<PathBuf>,
    stack_profile: NockStackProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    let target_height = match (target_height, cutover.as_ref()) {
        (Some(height), None) => height,
        (None, Some(CutoverVersion::V1)) => PROOF_VERSION_1_START.saturating_sub(1),
        (None, Some(CutoverVersion::V2)) => PROOF_VERSION_2_START.saturating_sub(1),
        (Some(_), Some(_)) => {
            return Err("Specify either --target-height or --cutover, not both".into());
        }
        (None, None) => {
            return Err("Specify either --target-height or --cutover".into());
        }
    };

    let output_path = output.unwrap_or_else(|| {
        if let Some(cutover) = cutover {
            match cutover {
                CutoverVersion::V1 => PathBuf::from("checkpoint_at_v1_crossover.chkjam"),
                CutoverVersion::V2 => PathBuf::from("checkpoint_at_v2_crossover.chkjam"),
            }
        } else {
            PathBuf::from(format!("checkpoint_at_height_{}.chkjam", target_height))
        }
    });

    let work_dir = match work_dir {
        Some(dir) => dir,
        None => {
            let mut dir = std::env::temp_dir();
            let suffix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_else(|_| Duration::from_secs(0))
                .as_millis();
            dir.push(format!("nockchain-bench-sol-{}", suffix));
            std::fs::create_dir_all(&dir)?;
            dir
        }
    };

    println!("=== Speed-of-Light Checkpoint Builder ===\n");
    println!("Archive:      {}", archive.display());
    println!("Kernel:       {}", kernel.display());
    println!("Target height: {}", target_height);
    if let Some(ref checkpoint_path) = checkpoint {
        println!("Checkpoint:   {}", checkpoint_path.display());
    }
    if let Some(height) = start_height {
        println!("Start height: {}", height);
    }
    println!("Output:       {}", output_path.display());
    println!("Work dir:     {}", work_dir.display());
    println!(
        "Stack profile: {}",
        format!("{stack_profile:?}").to_lowercase()
    );
    println!();

    if !archive.exists() {
        return Err(format!("Archive file not found: {}", archive.display()).into());
    }
    if !kernel.exists() {
        return Err(format!("Kernel file not found: {}", kernel.display()).into());
    }
    if let Some(ref checkpoint_path) = checkpoint {
        if !checkpoint_path.exists() {
            return Err(format!("Checkpoint file not found: {}", checkpoint_path.display()).into());
        }
    }

    let config = CheckpointConfig {
        archive_path: archive.to_string_lossy().to_string(),
        kernel_path: kernel.to_string_lossy().to_string(),
        checkpoint_path: checkpoint.map(|p| p.to_string_lossy().to_string()),
        start_height: start_height.map(SolHeight),
        target_height: SolHeight(target_height),
        output_path: output_path.clone(),
        work_dir: work_dir.clone(),
        stack_profile,
    };

    let mut builder = CheckpointBuilder::new(config);
    let result = builder.run().await?;

    println!(
        "Checkpoint saved: {} (blocks poked: {})",
        result.output_path.display(),
        result.blocks_poked
    );

    Ok(())
}

/// Extract blocks from checkpoint to archive (speed-of-light)
async fn cmd_sol_extract(
    blocks: u64,
    start_height: u64,
    end_height: Option<u64>,
    checkpoint: PathBuf,
    kernel: PathBuf,
    output: Option<PathBuf>,
    chunk_size: u64,
    include_mempool: bool,
    stack_profile: NockStackProfile,
) -> Result<(), Box<dyn std::error::Error>> {
    if blocks == 0 && end_height.is_none() {
        return Err("--blocks must be > 0 when --end-height is not provided".into());
    }
    if chunk_size == 0 {
        return Err("--chunk-size must be > 0".into());
    }

    let resolved_end_height = if let Some(end) = end_height {
        if start_height > end {
            return Err(format!(
                "Invalid range: start height {} is greater than end height {}",
                start_height, end
            )
            .into());
        }
        end
    } else {
        start_height
            .checked_add(blocks.saturating_sub(1))
            .ok_or("Requested range overflows u64 heights")?
    };
    let target_blocks = resolved_end_height
        .saturating_sub(start_height)
        .saturating_add(1);

    let output_path = output.unwrap_or_else(|| {
        if end_height.is_some() || start_height > 0 {
            PathBuf::from(format!(
                "blocks_{}-{}.solarch",
                start_height, resolved_end_height
            ))
        } else {
            PathBuf::from(format!("blocks_{}.solarch", blocks))
        }
    });

    println!("=== Speed-of-Light Block Extraction ===\n");
    println!("Checkpoint: {}", checkpoint.display());
    println!("Kernel:     {}", kernel.display());
    println!("Range:      {}..={}", start_height, resolved_end_height);
    println!("Blocks:     {}", target_blocks);
    println!("Chunk size: {}", chunk_size);
    println!(
        "Mempool:    {}",
        if include_mempool { "included" } else { "off" }
    );
    println!(
        "Stack profile: {}",
        format!("{stack_profile:?}").to_lowercase()
    );
    println!("Output:     {}", output_path.display());
    println!();

    // Check files exist
    if !checkpoint.exists() {
        return Err(format!("Checkpoint file not found: {}", checkpoint.display()).into());
    }
    if !kernel.exists() {
        return Err(format!("Kernel file not found: {}", kernel.display()).into());
    }

    let config = ExtractorConfig {
        checkpoint_path: checkpoint.to_string_lossy().to_string(),
        kernel_path: kernel.to_string_lossy().to_string(),
        block_count: blocks,
        chunk_size,
        work_dir: PathBuf::from("."),
        include_mempool,
        stack_profile,
    };

    let mut extractor = BlockExtractor::new(config);

    println!("Initializing kernel (this may take a few minutes)...");
    let start = std::sync::Arc::new(std::time::Instant::now());
    let init_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let init_done_for_thread = std::sync::Arc::clone(&init_done);
    let start_for_thread = std::sync::Arc::clone(&start);
    let heartbeat = std::thread::spawn(move || {
        use std::io::Write as _;

        loop {
            let elapsed = start_for_thread.elapsed().as_secs();
            print!("\r  still initializing... {elapsed}s elapsed");
            let _ = std::io::stdout().flush();

            if init_done_for_thread.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }

            std::thread::sleep(Duration::from_secs(1));
        }
    });

    let init_result = extractor.initialize().await;
    init_done.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = heartbeat.join();
    println!();
    init_result?;

    println!(
        "Kernel initialized in {:.1}s\n",
        start.elapsed().as_secs_f64()
    );

    println!("Extracting blocks to archive...");
    let extract_start = std::time::Instant::now();
    let mut next_block_report = 1usize;
    let block_report_step = ((target_blocks / 20).max(1)) as usize;
    let mut next_mempool_report = 1usize;
    extractor
        .extract_range_to_archive_with_progress(
            start_height,
            resolved_end_height,
            &output_path,
            |progress| match progress.phase {
                ArchiveExtractionPhase::Blocks => {
                    if progress.blocks_archived >= next_block_report
                        || progress.blocks_archived >= target_blocks as usize
                    {
                        let pct = if target_blocks > 0 {
                            (progress.blocks_archived as f64 / target_blocks as f64 * 100.0)
                                .min(100.0)
                        } else {
                            100.0
                        };
                        println!(
                            "  blocks: {}/{} ({:.1}%) chunk {}..{} (+{})",
                            progress.blocks_archived,
                            target_blocks,
                            pct,
                            progress.chunk_start.unwrap_or(0),
                            progress.chunk_end.unwrap_or(0),
                            progress.chunk_blocks
                        );
                        next_block_report =
                            progress.blocks_archived.saturating_add(block_report_step);
                    }
                }
                ArchiveExtractionPhase::MempoolReplay => {
                    let total = progress.mempool_snapshots_total.max(1);
                    let step = (total / 20).max(1);
                    if progress.mempool_snapshots_done >= next_mempool_report
                        || progress.mempool_snapshots_done >= total
                    {
                        let pct = (progress.mempool_snapshots_done as f64 / total as f64 * 100.0)
                            .min(100.0);
                        println!(
                            "  mempool: {}/{} snapshots ({:.1}%)",
                            progress.mempool_snapshots_done, total, pct
                        );
                        next_mempool_report = progress.mempool_snapshots_done.saturating_add(step);
                    }
                }
                ArchiveExtractionPhase::Complete => {
                    println!(
                        "  archive write complete (blocks: {}, txs: {})",
                        progress.blocks_archived, progress.txs_archived
                    );
                }
            },
        )
        .await?;
    let extract_time = extract_start.elapsed();

    // Get file size
    let file_size = std::fs::metadata(&output_path)?.len();

    println!("\n=== Extraction Complete ===\n");
    println!("Archive:    {}", output_path.display());
    println!("Size:       {:.2} MiB", file_size as f64 / 1024.0 / 1024.0);
    println!("Time:       {:.1}s", extract_time.as_secs_f64());
    println!(
        "Throughput: {:.1} blocks/s",
        target_blocks as f64 / extract_time.as_secs_f64()
    );

    Ok(())
}

/// Build a unified `.soltest` fixture.
async fn cmd_sol_fixture_build(
    source_checkpoint: PathBuf,
    kernel: PathBuf,
    start_height: u64,
    end_height: u64,
    output: PathBuf,
    chunk_size: u64,
    checkpoint_mode: FixtureCheckpointModeArg,
    include_mempool: bool,
    work_dir: PathBuf,
    stack_profile: Option<NockStackProfile>,
    progress_interval_secs: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Speed-of-Light Fixture Build ===\n");
    println!("Source checkpoint: {}", source_checkpoint.display());
    println!("Kernel:            {}", kernel.display());
    println!("Range:             {}..={}", start_height, end_height);
    println!("Chunk size:        {}", chunk_size);
    println!(
        "Checkpoint mode:   {}",
        fixture_checkpoint_mode_arg_label(checkpoint_mode)
    );
    match checkpoint_mode {
        FixtureCheckpointModeArg::Derived => println!(
            "Stack profile:     {}",
            format!("{:?}", stack_profile.unwrap_or(NockStackProfile::Large)).to_lowercase()
        ),
        FixtureCheckpointModeArg::Full => println!(
            "Stack profile:     {}",
            format!("{:?}", stack_profile.unwrap_or(NockStackProfile::Huge)).to_lowercase()
        ),
        FixtureCheckpointModeArg::Both => match stack_profile {
            Some(profile) => println!(
                "Stack profile:     {} (applied to both fixtures)",
                format!("{profile:?}").to_lowercase()
            ),
            None => println!("Stack profile:     derived=large, full=huge (default)"),
        },
    }
    println!(
        "Mempool:           {}",
        if include_mempool { "included" } else { "off" }
    );
    println!("Progress interval: {}s", progress_interval_secs.max(1));
    match checkpoint_mode {
        FixtureCheckpointModeArg::Both => {
            let derived_output = fixture_output_with_suffix(&output, "derived")?;
            let full_output = fixture_output_with_suffix(&output, "full")?;
            println!("Output fixture:    {} (base name)", output.display());
            println!("Derived output:    {}", derived_output.display());
            println!("Full output:       {}", full_output.display());
        }
        _ => println!("Output fixture:    {}", output.display()),
    }
    println!("Work dir:          {}", work_dir.display());
    println!();

    let targets = fixture_build_targets(checkpoint_mode, &output, stack_profile)?;
    let progress_interval = Duration::from_secs(progress_interval_secs.max(1));
    let mut built = Vec::with_capacity(targets.len());
    for target in targets {
        let mode_label = fixture_checkpoint_mode_label(target.checkpoint_mode);
        println!("\n=== Building {} checkpoint fixture ===\n", mode_label);
        println!("Output fixture:    {}", target.output_path.display());
        println!(
            "Stack profile:     {}",
            format!("{:?}", target.stack_profile).to_lowercase()
        );

        let builder = FixtureBuilder::new(FixtureBuildConfig {
            source_checkpoint_path: source_checkpoint.clone(),
            kernel_path: kernel.clone(),
            start_height: SolHeight(start_height),
            end_height: SolHeight(end_height),
            output_path: target.output_path.clone(),
            work_dir: work_dir.clone(),
            chunk_size,
            include_mempool,
            stack_profile: target.stack_profile,
            checkpoint_mode: target.checkpoint_mode,
        });

        let mut last_phase: Option<FixtureBuildPhase> = None;
        let mut last_emit = std::time::Instant::now()
            .checked_sub(progress_interval)
            .unwrap_or_else(std::time::Instant::now);
        let result = builder
            .run_with_progress(|progress| {
                let phase_changed = last_phase != Some(progress.phase);
                if phase_changed {
                    last_phase = Some(progress.phase);
                    println!(
                        "\n[{}:{:?}] {}",
                        mode_label, progress.phase, progress.message
                    );
                    last_emit = std::time::Instant::now();
                }
                let force_emit = matches!(
                    (progress.blocks_done, progress.blocks_total),
                    (Some(done), Some(total)) if done >= total
                );
                let now = std::time::Instant::now();
                if !force_emit && now.duration_since(last_emit) < progress_interval {
                    return;
                }
                if let (Some(done), Some(total)) = (progress.blocks_done, progress.blocks_total) {
                    if total > 0 {
                        let pct = (done as f64 / total as f64 * 100.0).min(100.0);
                        println!("  [{mode_label}] progress: {done}/{total} ({pct:.1}%)");
                    } else {
                        println!("  [{mode_label}] progress: {done}");
                    }
                } else if !progress.message.is_empty() {
                    println!("  [{mode_label}] {}", progress.message);
                }
                last_emit = now;
            })
            .await?;

        println!("\n[{}] Fixture created:", mode_label);
        println!("  Path:              {}", result.output_path.display());
        println!(
            "  Derived checkpoint: {}",
            result.derived_checkpoint_height.as_u64()
        );
        println!(
            "  Archive range:      {}..={}",
            result.archive_start_height.as_u64(),
            result.archive_end_height.as_u64()
        );
        built.push((target.checkpoint_mode, result.output_path));
    }

    if built.len() > 1 {
        println!("\nBuilt fixtures:");
        for (mode, output_path) in built {
            println!(
                "  {}: {}",
                fixture_checkpoint_mode_label(mode),
                output_path.display()
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct FixtureBuildTarget {
    checkpoint_mode: FixtureCheckpointMode,
    output_path: PathBuf,
    stack_profile: NockStackProfile,
}

fn fixture_build_targets(
    checkpoint_mode: FixtureCheckpointModeArg,
    output: &Path,
    stack_profile: Option<NockStackProfile>,
) -> Result<Vec<FixtureBuildTarget>, Box<dyn std::error::Error>> {
    let targets = match checkpoint_mode {
        FixtureCheckpointModeArg::Derived => vec![FixtureBuildTarget {
            checkpoint_mode: FixtureCheckpointMode::Derived,
            output_path: output.to_path_buf(),
            stack_profile: stack_profile.unwrap_or(NockStackProfile::Large),
        }],
        FixtureCheckpointModeArg::Full => vec![FixtureBuildTarget {
            checkpoint_mode: FixtureCheckpointMode::Full,
            output_path: output.to_path_buf(),
            stack_profile: stack_profile.unwrap_or(NockStackProfile::Huge),
        }],
        FixtureCheckpointModeArg::Both => vec![
            FixtureBuildTarget {
                checkpoint_mode: FixtureCheckpointMode::Derived,
                output_path: fixture_output_with_suffix(output, "derived")?,
                stack_profile: stack_profile.unwrap_or(NockStackProfile::Large),
            },
            FixtureBuildTarget {
                checkpoint_mode: FixtureCheckpointMode::Full,
                output_path: fixture_output_with_suffix(output, "full")?,
                stack_profile: stack_profile.unwrap_or(NockStackProfile::Huge),
            },
        ],
    };
    Ok(targets)
}

fn fixture_output_with_suffix(
    output: &Path,
    suffix: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let stem = output.file_stem().ok_or_else(|| {
        format!(
            "output path must include a file name for --checkpoint-mode both: {}",
            output.display()
        )
    })?;
    let mut file_name = std::ffi::OsString::from(stem);
    file_name.push(".");
    file_name.push(suffix);
    if let Some(ext) = output.extension() {
        file_name.push(".");
        file_name.push(ext);
    }
    let mut output_with_suffix = output.to_path_buf();
    output_with_suffix.set_file_name(file_name);
    Ok(output_with_suffix)
}

fn fixture_checkpoint_mode_label(mode: FixtureCheckpointMode) -> &'static str {
    match mode {
        FixtureCheckpointMode::Derived => "derived",
        FixtureCheckpointMode::Full => "full",
    }
}

fn fixture_checkpoint_mode_arg_label(mode: FixtureCheckpointModeArg) -> &'static str {
    match mode {
        FixtureCheckpointModeArg::Derived => "derived",
        FixtureCheckpointModeArg::Full => "full",
        FixtureCheckpointModeArg::Both => "both",
    }
}

/// Inspect a unified `.soltest` fixture.
fn cmd_sol_fixture_inspect(fixture: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Speed-of-Light Fixture Inspect ===\n");
    println!("Fixture: {}", fixture.display());
    println!();

    if !fixture.exists() {
        return Err(format!("Fixture file not found: {}", fixture.display()).into());
    }

    let data = read_fixture_file(&fixture)?;
    let m = data.manifest;
    println!("Format version:            {}", m.format_version);
    println!("Source checkpoint path:    {}", m.source_checkpoint_path);
    println!(
        "Source checkpoint event:   {}",
        m.source_checkpoint_event_num
    );
    println!(
        "Derived checkpoint height: {} (event {})",
        m.derived_checkpoint_height.as_u64(),
        m.derived_checkpoint_event_num
    );
    println!(
        "Archive range:             {}..={}",
        m.archive_start_height.as_u64(),
        m.archive_end_height.as_u64()
    );
    println!(
        "Mempool snapshots:         {}",
        if m.include_mempool { "on" } else { "off" }
    );
    println!("Chunk size:                {}", m.chunk_size);
    println!("Kernel hash:               {}", m.kernel_hash_hex);
    println!("Checkpoint hash:           {}", m.checkpoint_hash_hex);
    println!("Archive hash:              {}", m.archive_hash_hex);
    println!(
        "Embedded sizes:            checkpoint={} bytes, archive={} bytes, kernel={} bytes",
        data.checkpoint_bytes.len(),
        data.archive_bytes.len(),
        data.kernel_bytes.len()
    );

    Ok(())
}

fn cmd_sol_fixture_pack(
    checkpoint: PathBuf,
    source_checkpoint: Option<PathBuf>,
    archive: PathBuf,
    kernel: PathBuf,
    archive_start_height: u64,
    archive_end_height: u64,
    checkpoint_height: Option<u64>,
    chunk_size: u64,
    include_mempool: bool,
    output: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    if archive_start_height > archive_end_height {
        return Err(format!(
            "Invalid archive range: {}..={} (start > end)",
            archive_start_height, archive_end_height
        )
        .into());
    }
    if chunk_size == 0 {
        return Err("--chunk-size must be > 0".into());
    }
    if !checkpoint.exists() {
        return Err(format!("Checkpoint file not found: {}", checkpoint.display()).into());
    }
    if !archive.exists() {
        return Err(format!("Archive file not found: {}", archive.display()).into());
    }
    if !kernel.exists() {
        return Err(format!("Kernel file not found: {}", kernel.display()).into());
    }
    let source_checkpoint_path = source_checkpoint.unwrap_or_else(|| checkpoint.clone());
    if !source_checkpoint_path.exists() {
        return Err(format!(
            "Source checkpoint file not found: {}",
            source_checkpoint_path.display()
        )
        .into());
    }
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let embedded_checkpoint_height =
        checkpoint_height.unwrap_or_else(|| archive_start_height.saturating_sub(1));
    let source_event_num = checkpoint_event_num(&source_checkpoint_path)?;
    let embedded_event_num = checkpoint_event_num(&checkpoint)?;

    let manifest = SolFixtureManifest {
        format_version: 2,
        source_checkpoint_path: source_checkpoint_path.to_string_lossy().to_string(),
        source_checkpoint_event_num: source_event_num,
        derived_checkpoint_height: SolHeight(embedded_checkpoint_height),
        derived_checkpoint_event_num: embedded_event_num,
        archive_start_height: SolHeight(archive_start_height),
        archive_end_height: SolHeight(archive_end_height),
        include_mempool,
        chunk_size,
        kernel_hash_hex: blake3_hash_hex_for_file(&kernel)?,
        checkpoint_hash_hex: blake3_hash_hex_for_file(&checkpoint)?,
        archive_hash_hex: blake3_hash_hex_for_file(&archive)?,
    };

    write_fixture_file_from_paths(&output, &manifest, &checkpoint, &archive, &kernel)?;

    println!("Fixture packed: {}", output.display());
    println!("Embedded checkpoint: {}", checkpoint.display());
    println!("Source checkpoint:   {}", source_checkpoint_path.display());
    println!(
        "Archive range:       {}..={}",
        archive_start_height, archive_end_height
    );
    println!("Checkpoint height:   {}", embedded_checkpoint_height);
    println!("Chunk size:          {}", chunk_size);
    println!(
        "Mempool snapshots:   {}",
        if include_mempool { "on" } else { "off" }
    );

    Ok(())
}

fn blake3_hash_hex_for_file(path: &std::path::Path) -> Result<String, std::io::Error> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Slice a block range from an archive into a new archive file.
fn cmd_sol_archive_slice(
    archive: PathBuf,
    start_height: u64,
    end_height: u64,
    output: PathBuf,
    work_dir: PathBuf,
    include_mempool: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Speed-of-Light Archive Slice ===\n");
    println!("Input archive:  {}", archive.display());
    println!("Range:          {}..={}", start_height, end_height);
    println!("Output archive: {}", output.display());
    println!("Work dir:       {}", work_dir.display());
    println!(
        "Mempool:        {}",
        if include_mempool {
            "included in range"
        } else {
            "dropped"
        }
    );
    println!();

    if !archive.exists() {
        return Err(format!("Archive file not found: {}", archive.display()).into());
    }

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir_all(&work_dir)?;

    let result = slice_archive_file(
        &archive,
        &output,
        &work_dir,
        SolHeight(start_height),
        SolHeight(end_height),
        include_mempool,
    )?;

    let reader = ArchiveReader::from_file(&output)?;
    println!("Slice complete:");
    println!(
        "  copied range:      {}..={}",
        result.start_height.as_u64(),
        result.end_height.as_u64()
    );
    println!("  copied blocks:     {}", result.block_count);
    println!("  copied mempool:    {}", result.mempool_snapshot_count);
    println!(
        "  output stats:      blocks={} txs={} mempool={}",
        reader.block_count(),
        reader.total_tx_count(),
        reader.mempool_snapshot_count()
    );

    Ok(())
}

/// Inspect mempool snapshots for stale transactions
fn cmd_sol_inspect(archive: PathBuf, retain: u64) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Speed-of-Light Mempool Inspector ===\n");
    println!("Archive: {}", archive.display());
    println!("Retain:  {} blocks", retain);
    println!();

    if !archive.exists() {
        return Err(format!("Archive file not found: {}", archive.display()).into());
    }

    let reader = ArchiveReader::from_file(&archive)?;
    let ranges = find_stale_ranges(&reader, retain)?;

    println!(
        "Snapshots: {} (mempool: {})",
        reader.mempool_snapshot_count(),
        if reader.has_mempool() { "on" } else { "off" }
    );
    println!("Stale ranges: {}", ranges.len());

    for range in ranges {
        let age_end = range
            .end_height
            .as_u64()
            .saturating_sub(range.heard_at.as_u64());
        let span = range
            .end_height
            .as_u64()
            .saturating_sub(range.start_height.as_u64())
            .saturating_add(1);
        println!(
            "tx={} heard_at={} stale_range={}..={} age_end={} span={}",
            range.tx_id.to_base58(),
            range.heard_at.as_u64(),
            range.start_height.as_u64(),
            range.end_height.as_u64(),
            age_end,
            span
        );
    }

    Ok(())
}

async fn cmd_sol_sweep(
    candidates_csv: &str,
    chunk_sizes_csv: &str,
    memory_limits_csv: &str,
    repeats: u32,
    duration: u64,
    sample_interval: u64,
    save_interval: u64,
    image: &str,
    data_dir: PathBuf,
    threads: u32,
    candidate_env: &str,
    chunk_env: &str,
    output_json: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let candidates = parse_csv_strings(candidates_csv);
    let chunk_sizes = parse_csv_u64(chunk_sizes_csv)?;
    let memory_limits = parse_csv_strings(memory_limits_csv);

    if candidates.is_empty() {
        return Err("No PMA candidates provided".into());
    }
    if chunk_sizes.is_empty() {
        return Err("No chunk sizes provided".into());
    }
    if memory_limits.is_empty() {
        return Err("No memory limits provided".into());
    }

    let cases = build_sweep_cases(&candidates, &chunk_sizes, &memory_limits);

    println!("=== Speed-of-Light Sweep ===\n");
    println!("Cases: {}", cases.len());
    println!("Repeats: {}", repeats);
    println!("Duration per run: {}s", duration);
    println!("Sample interval: {}s", sample_interval);
    println!("Save interval: {}s", save_interval);
    println!("Image: {}", image);
    println!("Candidate env: {}", candidate_env);
    println!("Chunk env: {}", chunk_env);
    println!("Base data dir: {}", data_dir.display());
    println!();

    let mut runs = Vec::<SweepRunMetrics>::new();

    for (idx, case) in cases.iter().enumerate() {
        println!(
            "[{}/{}] candidate={} chunk={} memory={}",
            idx + 1,
            cases.len(),
            case.candidate,
            case.chunk_size,
            case.memory_limit
        );

        for run_index in 0..repeats {
            let run_dir = data_dir.join(format!(
                "cand-{}-chunk-{}-mem-{}-run-{}",
                sanitize_case_value(&case.candidate),
                case.chunk_size,
                sanitize_case_value(&case.memory_limit),
                run_index + 1
            ));
            let mut env_vars = HashMap::new();
            env_vars.insert(candidate_env.to_string(), case.candidate.clone());
            env_vars.insert(chunk_env.to_string(), case.chunk_size.to_string());

            let config = MiningScenarioConfig {
                name: format!(
                    "sol-sweep-{}-chunk{}-mem{}-run{}",
                    sanitize_case_value(&case.candidate),
                    case.chunk_size,
                    sanitize_case_value(&case.memory_limit),
                    run_index + 1
                ),
                mode: NockchainMode::Checkpoint {
                    save_interval_secs: save_interval,
                },
                duration: Duration::from_secs(duration),
                sample_interval: Duration::from_secs(sample_interval),
                image: image.to_string(),
                data_dir: run_dir.clone(),
                memory_limit: Some(case.memory_limit.clone()),
                num_threads: threads,
                env_vars,
                ..Default::default()
            };

            let scenario = MiningScenario::new(config);
            let result = scenario.run().await?;

            let mut parser = LogParser::new();
            let events = parser.parse_lines(&result.final_logs);
            let checkpoint_durations = checkpoint_durations_ms(&events);
            let checkpoint_count = checkpoint_durations.len() as u64;
            let checkpoint_avg_duration_s = if checkpoint_durations.is_empty() {
                None
            } else {
                Some(
                    checkpoint_durations.iter().sum::<u64>() as f64
                        / checkpoint_durations.len() as f64
                        / 1000.0,
                )
            };
            let checkpoint_size = latest_checkpoint_size_in_dir(&run_dir)?;
            let checkpoint_mib_per_s = match (checkpoint_size, checkpoint_avg_duration_s) {
                (Some(size_bytes), Some(avg_secs)) if avg_secs > 0.0 => {
                    Some((size_bytes as f64 / 1024.0 / 1024.0) / avg_secs)
                }
                _ => None,
            };

            let (fault_bursts, minor_total, major_total) =
                match page_fault_bursts(&result.samples, 50_000, 1) {
                    Some((bursts, minor, major)) => (Some(bursts), Some(minor), Some(major)),
                    None => (None, None, None),
                };

            runs.push(SweepRunMetrics {
                case: case.clone(),
                run_index,
                peak_rss_mib: result.peak_rss_mib(),
                avg_rss_mib: result.avg_rss_mib(),
                checkpoint_count,
                checkpoint_avg_duration_s,
                checkpoint_mib_per_s,
                page_fault_bursts: fault_bursts,
                minor_faults_delta_total: minor_total,
                major_faults_delta_total: major_total,
            });

            println!(
                "  run {}: peak_rss={:.1} MiB checkpoints={} checkpoint_mib_per_s={}",
                run_index + 1,
                result.peak_rss_mib(),
                checkpoint_count,
                checkpoint_mib_per_s
                    .map(|value| format!("{:.2}", value))
                    .unwrap_or_else(|| "n/a".to_string())
            );
        }
    }

    let mut summaries = Vec::new();
    for case in &cases {
        let case_runs: Vec<SweepRunMetrics> = runs
            .iter()
            .filter(|run| {
                run.case.candidate == case.candidate
                    && run.case.chunk_size == case.chunk_size
                    && run.case.memory_limit == case.memory_limit
            })
            .cloned()
            .collect();
        summaries.push(summarize_case_runs(case, &case_runs));
    }

    println!("\n=== Sweep Summary ===\n");
    println!(
        "{:<16} {:>8} {:>8} {:>10} {:>10} {:>10}",
        "candidate", "chunk", "memory", "peak_rss", "ckpt_mib/s", "rss_stddev"
    );
    println!("{}", "-".repeat(74));
    for summary in &summaries {
        println!(
            "{:<16} {:>8} {:>8} {:>10.1} {:>10} {:>10.2}",
            summary.case.candidate,
            summary.case.chunk_size,
            summary.case.memory_limit,
            summary.peak_rss_mib_mean,
            summary
                .checkpoint_mib_per_s_mean
                .map(|value| format!("{:.2}", value))
                .unwrap_or_else(|| "n/a".to_string()),
            summary.peak_rss_mib_stddev
        );
    }

    if let Some(path) = output_json {
        let payload = serde_json::json!({
            "cases": cases,
            "runs": runs,
            "summaries": summaries,
            "config": {
                "repeats": repeats,
                "duration_secs": duration,
                "sample_interval_secs": sample_interval,
                "save_interval_secs": save_interval,
                "image": image,
                "candidate_env": candidate_env,
                "chunk_env": chunk_env,
                "data_dir": data_dir,
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&payload)?)?;
        println!("\nSweep JSON written to {}", path.display());
    }

    Ok(())
}

fn parse_csv_strings(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn parse_csv_u64(input: &str) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let mut values = Vec::new();
    for token in input
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let parsed = u64::from_str(token)
            .map_err(|e| format!("invalid u64 value '{token}' in list: {e}"))?;
        values.push(parsed);
    }
    Ok(values)
}

fn sanitize_case_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

fn latest_checkpoint_size_in_dir(dir: &std::path::Path) -> Result<Option<u64>, std::io::Error> {
    let mut latest: Option<(std::time::SystemTime, u64)> = None;
    for checkpoint_name in ["0.chkjam", "1.chkjam"] {
        let path = dir.join(checkpoint_name);
        if !path.exists() {
            continue;
        }
        let metadata = std::fs::metadata(path)?;
        let modified = metadata
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let size = metadata.len();
        match latest {
            Some((current_modified, _)) if modified <= current_modified => {}
            _ => latest = Some((modified, size)),
        }
    }
    Ok(latest.map(|(_, size)| size))
}

fn kb_to_mib(kb: u64) -> f64 {
    kb as f64 / 1024.0
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv_strings() {
        let values = parse_csv_strings("alpha, beta ,,gamma");
        assert_eq!(values, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_parse_csv_u64() {
        let values = parse_csv_u64("64,128,256").expect("parse");
        assert_eq!(values, vec![64, 128, 256]);
        assert!(parse_csv_u64("abc").is_err());
    }

    #[test]
    fn test_sanitize_case_value() {
        assert_eq!(sanitize_case_value("V1 Candidate"), "v1-candidate");
        assert_eq!(sanitize_case_value("chunk/64"), "chunk-64");
    }

    #[test]
    fn test_latest_checkpoint_size_in_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path0 = dir.path().join("0.chkjam");
        let path1 = dir.path().join("1.chkjam");
        std::fs::write(&path0, vec![0u8; 10]).expect("write");
        std::thread::sleep(std::time::Duration::from_millis(5));
        std::fs::write(&path1, vec![0u8; 20]).expect("write");
        let size = latest_checkpoint_size_in_dir(dir.path()).expect("size");
        assert_eq!(size, Some(20));
    }

    #[test]
    fn test_fixture_output_with_suffix() {
        let base = PathBuf::from("/tmp/fixture.soltest");
        let derived = fixture_output_with_suffix(&base, "derived").expect("derived output");
        let full = fixture_output_with_suffix(&base, "full").expect("full output");
        assert_eq!(derived, PathBuf::from("/tmp/fixture.derived.soltest"));
        assert_eq!(full, PathBuf::from("/tmp/fixture.full.soltest"));
    }

    #[test]
    fn test_fixture_build_targets_for_both_defaults_and_override() {
        let output = PathBuf::from("bundle.soltest");

        let defaults =
            fixture_build_targets(FixtureCheckpointModeArg::Both, &output, None).expect("targets");
        assert_eq!(defaults.len(), 2);
        assert_eq!(defaults[0].checkpoint_mode, FixtureCheckpointMode::Derived);
        assert_eq!(
            defaults[0].output_path,
            PathBuf::from("bundle.derived.soltest")
        );
        assert_eq!(defaults[0].stack_profile, NockStackProfile::Large);
        assert_eq!(defaults[1].checkpoint_mode, FixtureCheckpointMode::Full);
        assert_eq!(
            defaults[1].output_path,
            PathBuf::from("bundle.full.soltest")
        );
        assert_eq!(defaults[1].stack_profile, NockStackProfile::Huge);

        let overridden = fixture_build_targets(
            FixtureCheckpointModeArg::Both,
            &output,
            Some(NockStackProfile::Medium),
        )
        .expect("targets");
        assert_eq!(overridden[0].stack_profile, NockStackProfile::Medium);
        assert_eq!(overridden[1].stack_profile, NockStackProfile::Medium);
    }
}
