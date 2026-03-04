//! Nockchain Bench CLI
//!
//! Benchmarking and memory profiling tool for Nockchain.
//!
//! Usage:
//!   nockchain-bench sample <pid|self>           # Sample process memory
//!   nockchain-bench run [OPTIONS]               # Run a mining scenario
//!   nockchain-bench attach <container>          # Attach to existing container
//!   nockchain-bench compare [OPTIONS]           # A/B checkpoint comparison
//!   nockchain-bench sol extract [OPTIONS]       # Extract blocks to archive
//!   nockchain-bench sol inspect [OPTIONS]       # Inspect mempool snapshots

mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use commands::{CutoverVersion, OutputFormat};

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

        /// Checkpoint save interval in seconds
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

    /// Run A/B comparison between two checkpoint save intervals
    Compare {
        /// Duration to run each scenario in seconds
        #[arg(short, long, default_value = "300")]
        duration: u64,

        /// Sample interval in seconds
        #[arg(long, default_value = "1")]
        sample_interval: u64,

        /// Baseline checkpoint save interval in seconds
        #[arg(long, default_value = "120")]
        baseline_save_interval: u64,

        /// Candidate checkpoint save interval in seconds
        #[arg(long, default_value = "30")]
        candidate_save_interval: u64,

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

    /// Sweep candidate/chunk-size/memory-limit combinations and summarize checkpoint behavior
    Sweep {
        /// Candidate IDs (comma-separated)
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
    /// Build a fixture from a source archive + kernel
    Build {
        /// Source archive path (must include requested range and bootstrap prefix)
        #[arg(long)]
        archive: PathBuf,

        /// Kernel jam path
        #[arg(short, long, default_value = "assets/dumb.jam")]
        kernel: PathBuf,

        /// Start block height for replay window
        /// (intermediate checkpoint is built at this exact height)
        #[arg(long)]
        start_height: u64,

        /// End block height for replay window (inclusive)
        #[arg(long)]
        end_height: u64,

        /// Output fixture path
        #[arg(short, long)]
        output: PathBuf,

        /// Include mempool snapshots in sliced archive payload
        #[arg(long)]
        include_mempool: bool,

        /// Chunk size metadata to record in manifest
        #[arg(long, default_value = "8")]
        chunk_size: u64,

        /// Working directory for temporary artifacts
        #[arg(long, default_value = ".")]
        work_dir: PathBuf,
    },

    /// Inspect fixture metadata and embedded payload sizes
    Inspect {
        /// Fixture path
        #[arg(short, long)]
        fixture: PathBuf,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Sample {
            pid,
            nockstack_size,
        } => commands::sample::cmd_sample(&pid, nockstack_size),
        Commands::Run {
            name,
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
            commands::mining::cmd_run(
                &name, save_interval, duration, sample_interval, &image, data_dir, &memory_limit,
                threads, output, format,
            )
            .await
        }
        Commands::Attach {
            container,
            duration,
            sample_interval,
            output,
            format,
        } => {
            commands::mining::cmd_attach(&container, duration, sample_interval, output, format)
                .await
        }
        Commands::Compare {
            duration,
            sample_interval,
            baseline_save_interval,
            candidate_save_interval,
            image,
            data_dir,
            memory_limit,
            threads,
            output,
        } => {
            commands::mining::cmd_compare(
                duration, sample_interval, baseline_save_interval, candidate_save_interval, &image,
                data_dir, &memory_limit, threads, output,
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
            commands::mining::cmd_analyze(
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
            } => {
                commands::sol::cmd_sol_extract(
                    blocks, start_height, end_height, checkpoint, kernel, output, chunk_size,
                    include_mempool,
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
            } => {
                commands::sol::cmd_sol_bench(
                    fixture, blocks, enable_checkpointing, skip_genesis, profile_memory,
                    profile_interval_ms, profile_output, checkpoint_every_blocks,
                    checkpoint_recovery_timeout_ms, checkpoint_recovery_tolerance_pct,
                    gc_drop_threshold_mib, page_fault_minor_burst_threshold,
                    page_fault_major_burst_threshold,
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
            } => {
                commands::sol::cmd_sol_checkpoint(
                    archive, kernel, checkpoint, target_height, cutover, start_height, output,
                    work_dir,
                )
                .await
            }
            SolCommands::Inspect { archive, retain } => {
                commands::sol::cmd_sol_inspect(archive, retain)
            }
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
                output_json,
            } => {
                commands::sol::cmd_sol_sweep(
                    &candidates, &chunk_sizes, &memory_limits, repeats, duration, sample_interval,
                    save_interval, &image, data_dir, threads, output_json,
                )
                .await
            }
            SolCommands::Fixture(FixtureCommands::Build {
                archive,
                kernel,
                start_height,
                end_height,
                output,
                include_mempool,
                chunk_size,
                work_dir,
            }) => {
                commands::sol::cmd_sol_fixture_build(
                    archive, kernel, start_height, end_height, output, include_mempool, chunk_size,
                    work_dir,
                )
                .await
            }
            SolCommands::Fixture(FixtureCommands::Inspect { fixture }) => {
                commands::sol::cmd_sol_fixture_inspect(fixture)
            }
        },
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
