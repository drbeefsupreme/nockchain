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

use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};

use nockchain_bench::events::{EventCorrelator, LogParser};
use nockchain_bench::output::ParquetWriter;
use nockchain_bench::runner::{DockerRunner, NockchainMode};
use nockchain_bench::sampler::buckets::{sample_process, AttributionConfig};
use nockchain_bench::scenario::{MiningScenario, MiningScenarioConfig};
use nockchain_bench::speed_of_light::{
    BenchConfig, BenchRunner, BlockExtractor, CheckpointBuilder, CheckpointConfig, ExtractorConfig,
    ProofVersion, PROOF_VERSION_1_START, PROOF_VERSION_2_START,
};

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
    },

    /// Run the speed-of-light benchmark (poke blocks as fast as possible)
    Bench {
        /// Path to the archive file
        #[arg(short, long, default_value = "blocks_1000.solarch")]
        archive: PathBuf,

        /// Path to kernel jam file
        #[arg(short, long, default_value = "assets/dumb.jam")]
        kernel: PathBuf,

        /// Number of blocks to benchmark (0 = all in archive)
        #[arg(short = 'n', long, default_value = "0")]
        blocks: u64,

        /// Skip genesis block (block 0) - not recommended
        #[arg(long)]
        skip_genesis: bool,

        /// Filter blocks by proof version (v0, v1, v2)
        #[arg(long, value_enum)]
        proof_version: Option<ProofVersionFilter>,

        /// Load an existing checkpoint before benchmarking
        #[arg(long)]
        checkpoint: Option<PathBuf>,

        /// Start height override (defaults to checkpoint height + 1 if checkpoint provided)
        #[arg(long)]
        start_height: Option<u64>,
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

#[derive(Clone, Debug, ValueEnum)]
enum ProofVersionFilter {
    V0,
    V1,
    V2,
}

impl From<ProofVersionFilter> for ProofVersion {
    fn from(value: ProofVersionFilter) -> Self {
        match value {
            ProofVersionFilter::V0 => ProofVersion::V0,
            ProofVersionFilter::V1 => ProofVersion::V1,
            ProofVersionFilter::V2 => ProofVersion::V2,
        }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Sample { pid, nockstack_size } => cmd_sample(&pid, nockstack_size),
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
                &name,
                mode,
                save_interval,
                duration,
                sample_interval,
                &image,
                data_dir,
                &memory_limit,
                threads,
                output,
                format,
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
                duration,
                sample_interval,
                save_interval,
                &image,
                data_dir,
                &memory_limit,
                threads,
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
        } => cmd_analyze(&container, duration, sample_interval, spike_threshold, all_events).await,
        Commands::Sol(sol_cmd) => match sol_cmd {
            SolCommands::Extract {
                blocks,
                checkpoint,
                kernel,
                output,
                chunk_size,
            } => cmd_sol_extract(blocks, checkpoint, kernel, output, chunk_size).await,
            SolCommands::Bench {
                archive,
                kernel,
                blocks,
                skip_genesis,
                proof_version,
                checkpoint,
                start_height,
            } => {
                cmd_sol_bench(
                    archive,
                    kernel,
                    blocks,
                    skip_genesis,
                    proof_version,
                    checkpoint,
                    start_height,
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
                cmd_sol_checkpoint(
                    archive,
                    kernel,
                    checkpoint,
                    target_height,
                    cutover,
                    start_height,
                    output,
                    work_dir,
                )
                .await
            }
        },
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Sample a process's memory usage
fn cmd_sample(pid_str: &str, nockstack_size: Option<u64>) -> Result<(), Box<dyn std::error::Error>> {
    let pid = if pid_str == "self" {
        std::process::id() as i32
    } else {
        pid_str.parse().map_err(|_| format!("Invalid PID: {}", pid_str))?
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
    println!("  VmRSS from status:    {:>10.1} MiB", kb_to_mib(attr.vm_rss_kb));
    println!("  Difference:           {:>+10.1} MiB", diff as f64 / 1024.0);

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

    println!("Collecting stats for {}s at {}s intervals...\n", duration, sample_interval);

    let samples = runner
        .collect_stats(
            Duration::from_secs(duration),
            Duration::from_secs(sample_interval),
        )
        .await?;

    // Calculate summary stats
    let peak_memory = samples.iter().map(|s| s.memory_usage_bytes).max().unwrap_or(0);
    let avg_memory = if samples.is_empty() {
        0
    } else {
        samples.iter().map(|s| s.memory_usage_bytes).sum::<u64>() / samples.len() as u64
    };
    let peak_rss = samples.iter().map(|s| s.memory_rss_bytes).max().unwrap_or(0);
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
            &[
                ("checkpoint", &checkpoint_result.samples),
                ("pma_persist", &pma_result.samples),
            ],
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

    println!("Collecting stats for {}s at {}s intervals...\n", duration, sample_interval);

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
    println!("{:>10} {:>12} {:>12} {:>10}  Events", "Time (s)", "Memory (MiB)", "RSS (MiB)", "CPU %");
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
        println!("\n=== Memory Spikes (>{:.1}% increase) ===\n", spike_threshold);
        println!("{:>10} {:>12} {:>10}  Correlated Events", "Time (s)", "Memory (MiB)", "Change %");
        println!("{}", "-".repeat(70));

        for (_idx, sample, change_pct) in &spikes {
            let events_str = sample
                .events
                .iter()
                .map(|e| format!("{}@{:.1}s", e.event_type.label(), e.timestamp_ms as f64 / 1000.0))
                .collect::<Vec<_>>()
                .join(", ");

            println!(
                "{:>10.1} {:>12.1} {:>+10.1}%  {}",
                sample.stats.timestamp_ms as f64 / 1000.0,
                bytes_to_mib(sample.stats.memory_usage_bytes),
                change_pct,
                if events_str.is_empty() { "(no events)" } else { &events_str }
            );
        }
    } else {
        println!("\nNo memory spikes detected (threshold: {:.1}%)", spike_threshold);
    }

    // Event summary
    let significant_count = events.iter().filter(|e| e.is_significant()).count();
    let block_count = events.iter().filter(|e| matches!(e.event_type, nockchain_bench::events::EventType::BlockAccepted { .. })).count();

    println!("\n=== Event Summary ===");
    println!("Total events:       {}", events.len());
    println!("Significant events: {}", significant_count);
    println!("Blocks accepted:    {}", block_count);

    Ok(())
}

/// Run speed-of-light benchmark (poke blocks as fast as possible)
async fn cmd_sol_bench(
    archive: PathBuf,
    kernel: PathBuf,
    blocks: u64,
    skip_genesis: bool,
    proof_version: Option<ProofVersionFilter>,
    checkpoint: Option<PathBuf>,
    start_height: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Speed-of-Light Benchmark ===\n");
    println!("Archive: {}", archive.display());
    println!("Kernel:  {}", kernel.display());
    println!("Blocks:  {}", if blocks == 0 { "all".to_string() } else { blocks.to_string() });
    println!("Skip genesis: {}", skip_genesis);
    let proof_version = proof_version.map(ProofVersion::from);
    if let Some(version) = proof_version {
        println!("Proof version: {}", version);
    }
    if let Some(ref checkpoint_path) = checkpoint {
        println!("Checkpoint: {}", checkpoint_path.display());
    }
    if let Some(height) = start_height {
        println!("Start height: {}", height);
    }
    println!();

    // Check files exist
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

    let config = BenchConfig {
        archive_path: archive.to_string_lossy().to_string(),
        kernel_path: kernel.to_string_lossy().to_string(),
        block_count: blocks,
        skip_genesis,
        proof_version,
        checkpoint_path: checkpoint.map(|p| p.to_string_lossy().to_string()),
        start_height,
    };

    let mut runner = BenchRunner::new(config);

    println!("Initializing fresh kernel (this may take a few minutes)...");
    let results = runner.run().await?;

    results.print_summary();

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
        start_height,
        target_height,
        output_path: output_path.clone(),
        work_dir: work_dir.clone(),
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
    checkpoint: PathBuf,
    kernel: PathBuf,
    output: Option<PathBuf>,
    chunk_size: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = output.unwrap_or_else(|| PathBuf::from(format!("blocks_{}.solarch", blocks)));

    println!("=== Speed-of-Light Block Extraction ===\n");
    println!("Checkpoint: {}", checkpoint.display());
    println!("Kernel:     {}", kernel.display());
    println!("Blocks:     {}", blocks);
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
    };

    let mut extractor = BlockExtractor::new(config);

    println!("Initializing kernel (this may take a few minutes)...");
    let start = std::time::Instant::now();
    extractor.initialize().await?;
    println!("Kernel initialized in {:.1}s\n", start.elapsed().as_secs_f64());

    println!("Extracting blocks to archive...");
    let extract_start = std::time::Instant::now();
    extractor.extract_to_archive(blocks, &output_path).await?;
    let extract_time = extract_start.elapsed();

    // Get file size
    let file_size = std::fs::metadata(&output_path)?.len();

    println!("\n=== Extraction Complete ===\n");
    println!("Archive:    {}", output_path.display());
    println!("Size:       {:.2} MiB", file_size as f64 / 1024.0 / 1024.0);
    println!("Time:       {:.1}s", extract_time.as_secs_f64());
    println!(
        "Throughput: {:.1} blocks/s",
        blocks as f64 / extract_time.as_secs_f64()
    );

    Ok(())
}

fn kb_to_mib(kb: u64) -> f64 {
    kb as f64 / 1024.0
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}
