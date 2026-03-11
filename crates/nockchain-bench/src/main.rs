//! Nockchain Bench CLI
//!
//! Benchmarking and memory profiling tool for Nockchain.
//!
//! Usage:
//!   nockchain-bench sample <pid|self>           # Sample process memory
//!   nockchain-bench sol extract [OPTIONS]       # Extract blocks to archive
//!   nockchain-bench sol inspect [OPTIONS]       # Inspect mempool snapshots

mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use commands::CutoverVersion;

#[derive(Parser)]
#[command(name = "nockchain-bench")]
#[command(about = "Benchmarking and memory profiling tool for Nockchain")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Debug, ValueEnum, PartialEq, Eq)]
enum BenchWorkDirMode {
    HostBind,
    DockerVolume,
    DockerTmpfs,
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

    /// Speed-of-light benchmark commands
    #[command(
        subcommand,
        after_help = "Use `quick-bench` only for inner loop work and NOT reproducible data."
    )]
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

    /// Run a quick inner-loop benchmark from a unified fixture (`.soltest`); NOT reproducible data
    #[command(name = "quick-bench")]
    QuickBench {
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

    /// Run a trusted native SOL benchmark and emit machine-readable artifacts
    Bench {
        /// Path to a unified `.soltest` fixture file (includes checkpoint + archive + kernel)
        #[arg(short, long)]
        fixture: PathBuf,

        /// Output root directory for trusted run artifacts
        #[arg(short, long)]
        output: PathBuf,

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

        /// Force checkpoint every N accepted blocks (0 disables)
        #[arg(long, default_value = "0")]
        checkpoint_every_blocks: u64,

        /// Logical thread count metadata for this requested case
        #[arg(long, default_value = "1")]
        threads: u32,

        /// Warmup repetitions to persist but exclude from summary statistics
        #[arg(long, default_value = "1")]
        warmup_runs: u32,

        /// Measured repetitions to include in summary statistics
        #[arg(long, default_value = "5")]
        measured_runs: u32,

        /// Cooldown between measured repetitions in seconds
        #[arg(long, default_value = "10")]
        cooldown_secs: u64,

        /// Optional human label for the requested case
        #[arg(long)]
        label: Option<String>,

        /// Run the trusted benchmark inside this Docker image instead of natively
        #[arg(long)]
        image_tag: Option<String>,

        /// Docker memory limit for trusted container execution (for example `16g`)
        #[arg(long)]
        memory_limit: Option<String>,

        /// Explicit Docker work directory mode for trusted container execution
        #[arg(long, value_enum)]
        work_dir_mode: Option<BenchWorkDirMode>,

        /// Optional Docker CPU set (for example `0-3`)
        #[arg(long)]
        cpuset: Option<String>,

        /// Optional Docker CPU quota
        #[arg(long)]
        cpu_quota: Option<i64>,

        /// Optional Docker CPU period
        #[arg(long)]
        cpu_period: Option<i64>,

        /// Allow trusted Docker runs when host/container versions differ
        #[arg(long)]
        allow_version_skew: bool,

        /// Allow trusted artifacts from a non-release build
        #[arg(long)]
        allow_debug_benchmark: bool,
    },

    /// Hidden machine-oriented wrapper for one shared once-run execution
    #[command(hide = true, name = "run-once")]
    RunOnce {
        /// Path to a resolved-case JSON payload
        #[arg(long)]
        resolved_case: PathBuf,

        /// Output directory for this run's artifacts
        #[arg(long)]
        run_dir: PathBuf,

        /// Optional explicit run id (defaults to the run_dir basename)
        #[arg(long)]
        run_id: Option<String>,
    },

    /// Hidden machine-oriented binary identity output
    #[command(hide = true, name = "binary-identity")]
    BinaryIdentity,

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
            SolCommands::QuickBench {
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
                commands::sol::cmd_sol_quick_bench(
                    fixture, blocks, enable_checkpointing, skip_genesis, profile_memory,
                    profile_interval_ms, profile_output, checkpoint_every_blocks,
                    checkpoint_recovery_timeout_ms, checkpoint_recovery_tolerance_pct,
                    gc_drop_threshold_mib, page_fault_minor_burst_threshold,
                    page_fault_major_burst_threshold,
                )
                .await
            }
            SolCommands::Bench {
                fixture,
                output,
                blocks,
                enable_checkpointing,
                skip_genesis,
                profile_memory,
                profile_interval_ms,
                checkpoint_every_blocks,
                threads,
                warmup_runs,
                measured_runs,
                cooldown_secs,
                label,
                image_tag,
                memory_limit,
                work_dir_mode,
                cpuset,
                cpu_quota,
                cpu_period,
                allow_version_skew,
                allow_debug_benchmark,
            } => {
                commands::sol::cmd_sol_bench(
                    fixture, output, blocks, enable_checkpointing, skip_genesis, profile_memory,
                    profile_interval_ms, checkpoint_every_blocks, threads, warmup_runs,
                    measured_runs, cooldown_secs, label, image_tag, memory_limit, work_dir_mode,
                    cpuset, cpu_quota, cpu_period, allow_version_skew, allow_debug_benchmark,
                )
                .await
            }
            SolCommands::RunOnce {
                resolved_case,
                run_dir,
                run_id,
            } => commands::sol::cmd_sol_run_once(resolved_case, run_dir, run_id).await,
            SolCommands::BinaryIdentity => commands::sol::cmd_sol_binary_identity(),
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

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    fn subcommand_names(command: &clap::Command) -> Vec<String> {
        command
            .get_subcommands()
            .filter(|subcommand| !subcommand.is_hide_set())
            .map(|subcommand| subcommand.get_name().to_string())
            .collect()
    }

    fn render_help(mut command: clap::Command) -> String {
        let mut buffer = Vec::new();
        command.write_long_help(&mut buffer).expect("render help");
        String::from_utf8(buffer).expect("utf8 help")
    }

    #[test]
    fn test_phase1_cli_surface() {
        let command = Cli::command();
        let top_level = subcommand_names(&command);

        assert_eq!(top_level, vec!["sample", "sol"]);

        let sol = command
            .get_subcommands()
            .find(|subcommand| subcommand.get_name() == "sol")
            .expect("sol subcommand");

        assert_eq!(
            subcommand_names(sol),
            vec!["extract", "quick-bench", "bench", "checkpoint", "inspect", "fixture"]
        );

        let fixture = sol
            .get_subcommands()
            .find(|subcommand| subcommand.get_name() == "fixture")
            .expect("fixture subcommand");

        assert_eq!(subcommand_names(fixture), vec!["build", "inspect"]);
    }

    #[test]
    fn test_sol_bench_cli_surface() {
        let command = Cli::command();
        let sol = command
            .get_subcommands()
            .find(|subcommand| subcommand.get_name() == "sol")
            .expect("sol subcommand");

        assert!(subcommand_names(sol).contains(&"bench".to_string()));
        assert!(subcommand_names(sol).contains(&"quick-bench".to_string()));
    }

    #[test]
    fn test_sol_help_warns_about_quick_bench() {
        let command = Cli::command();
        let sol = command
            .get_subcommands()
            .find(|subcommand| subcommand.get_name() == "sol")
            .expect("sol subcommand")
            .clone();
        let help = render_help(sol);

        assert!(help.contains("quick-bench"));
        assert!(help.contains("NOT reproducible data"));
    }

    #[test]
    fn test_sol_bench_help_hides_quick_only_flags() {
        let command = Cli::command();
        let bench = command
            .get_subcommands()
            .find(|subcommand| subcommand.get_name() == "sol")
            .expect("sol subcommand")
            .get_subcommands()
            .find(|subcommand| subcommand.get_name() == "bench")
            .expect("bench subcommand")
            .clone();
        let help = render_help(bench);

        assert!(!help.contains("--checkpoint-recovery-timeout-ms"));
        assert!(!help.contains("--gc-drop-threshold-mib"));
        assert!(!help.contains("--page-fault-minor-burst-threshold"));
    }

    #[test]
    fn test_sol_help_hides_internal_run_once() {
        let command = Cli::command();
        let sol = command
            .get_subcommands()
            .find(|subcommand| subcommand.get_name() == "sol")
            .expect("sol subcommand")
            .clone();
        let help = render_help(sol);

        assert!(!help.contains("run-once"));
    }

    #[test]
    fn test_sol_run_once_cli_parses_hidden_command() {
        let cli = Cli::try_parse_from([
            "nockchain-bench", "sol", "run-once", "--resolved-case", "resolved_case.json",
            "--run-dir", "out/run-0",
        ])
        .expect("parse run-once");

        match cli.command {
            Commands::Sol(SolCommands::RunOnce {
                resolved_case,
                run_dir,
                run_id,
            }) => {
                assert_eq!(resolved_case, PathBuf::from("resolved_case.json"));
                assert_eq!(run_dir, PathBuf::from("out/run-0"));
                assert_eq!(run_id, None);
            }
            _ => panic!("expected sol run-once command"),
        }
    }

    #[test]
    fn test_sol_help_hides_internal_binary_identity() {
        let command = Cli::command();
        let sol = command
            .get_subcommands()
            .find(|subcommand| subcommand.get_name() == "sol")
            .expect("sol subcommand")
            .clone();
        let help = render_help(sol);

        assert!(!help.contains("binary-identity"));
    }

    #[test]
    fn test_sol_binary_identity_cli_parses_hidden_command() {
        let cli = Cli::try_parse_from(["nockchain-bench", "sol", "binary-identity"])
            .expect("parse binary-identity");

        match cli.command {
            Commands::Sol(SolCommands::BinaryIdentity) => {}
            _ => panic!("expected sol binary-identity command"),
        }
    }

    #[test]
    fn test_sol_bench_accepts_docker_backend_flags() {
        let cli = Cli::try_parse_from([
            "nockchain-bench", "sol", "bench", "--fixture", "fixture.soltest", "--output", "out",
            "--image-tag", "nockchain-bench:test", "--memory-limit", "2g", "--work-dir-mode",
            "docker-volume", "--cpuset", "0-3", "--cpu-quota", "200000", "--cpu-period", "100000",
            "--allow-version-skew",
        ])
        .expect("parse docker bench");

        match cli.command {
            Commands::Sol(SolCommands::Bench {
                image_tag,
                memory_limit,
                work_dir_mode,
                cpuset,
                cpu_quota,
                cpu_period,
                allow_version_skew,
                ..
            }) => {
                assert_eq!(image_tag.as_deref(), Some("nockchain-bench:test"));
                assert_eq!(memory_limit.as_deref(), Some("2g"));
                assert_eq!(work_dir_mode, Some(BenchWorkDirMode::DockerVolume));
                assert_eq!(cpuset.as_deref(), Some("0-3"));
                assert_eq!(cpu_quota, Some(200000));
                assert_eq!(cpu_period, Some(100000));
                assert!(allow_version_skew);
            }
            _ => panic!("expected sol bench command"),
        }
    }
}
