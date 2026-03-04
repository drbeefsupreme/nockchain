use std::path::PathBuf;
use std::time::Duration;

use nockchain_bench::events::{EventCorrelator, LogParser};
use nockchain_bench::output::ParquetWriter;
use nockchain_bench::runner::{DockerRunner, NockchainMode};
use nockchain_bench::scenario::{MiningScenario, MiningScenarioConfig};

use super::{bytes_to_mib, OutputFormat};

/// Run a mining scenario
pub async fn cmd_run(
    name: &str,
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
    let nockchain_mode = NockchainMode::Checkpoint {
        save_interval_secs: save_interval,
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
    println!("Mode: checkpoint");
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
pub async fn cmd_attach(
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

/// Run A/B comparison between two checkpoint save intervals
pub async fn cmd_compare(
    duration: u64,
    sample_interval: u64,
    baseline_save_interval: u64,
    candidate_save_interval: u64,
    image: &str,
    data_dir: PathBuf,
    memory_limit: &str,
    threads: u32,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== A/B Comparison: Checkpoint Interval Variants ===\n");

    // Run baseline checkpoint mode
    let baseline_config = MiningScenarioConfig {
        name: "checkpoint_baseline".to_string(),
        mode: NockchainMode::Checkpoint {
            save_interval_secs: baseline_save_interval,
        },
        duration: Duration::from_secs(duration),
        sample_interval: Duration::from_secs(sample_interval),
        image: image.to_string(),
        data_dir: data_dir.join("checkpoint_baseline"),
        memory_limit: Some(memory_limit.to_string()),
        num_threads: threads,
        ..Default::default()
    };

    println!(
        "--- Running Baseline Checkpoint Mode ({}s) ---",
        baseline_save_interval
    );
    let baseline_scenario = MiningScenario::new(baseline_config);
    let baseline_result = baseline_scenario.run().await?;
    baseline_result.print_summary();

    // Clean up between runs
    println!("\nCleaning up...\n");
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Run candidate checkpoint mode
    let candidate_config = MiningScenarioConfig {
        name: "checkpoint_candidate".to_string(),
        mode: NockchainMode::Checkpoint {
            save_interval_secs: candidate_save_interval,
        },
        duration: Duration::from_secs(duration),
        sample_interval: Duration::from_secs(sample_interval),
        image: image.to_string(),
        data_dir: data_dir.join("checkpoint_candidate"),
        memory_limit: Some(memory_limit.to_string()),
        num_threads: threads,
        ..Default::default()
    };

    println!(
        "--- Running Candidate Checkpoint Mode ({}s) ---",
        candidate_save_interval
    );
    let candidate_scenario = MiningScenario::new(candidate_config);
    let candidate_result = candidate_scenario.run().await?;
    candidate_result.print_summary();

    // Print comparison
    println!("\n=== Comparison Summary ===\n");
    println!(
        "{:<20} {:>15} {:>15} {:>10}",
        "Metric", "Baseline", "Candidate", "Diff %"
    );
    println!("{}", "-".repeat(65));

    let print_comparison = |name: &str, baseline: f64, candidate: f64| {
        let diff_pct = if baseline > 0.0 {
            ((candidate - baseline) / baseline) * 100.0
        } else {
            0.0
        };
        println!(
            "{:<20} {:>12.1} MiB {:>12.1} MiB {:>+9.1}%",
            name, baseline, candidate, diff_pct
        );
    };

    print_comparison(
        "Peak Memory",
        baseline_result.peak_memory_mib(),
        candidate_result.peak_memory_mib(),
    );
    print_comparison(
        "Avg Memory",
        baseline_result.avg_memory_mib(),
        candidate_result.avg_memory_mib(),
    );
    print_comparison(
        "Peak RSS",
        baseline_result.peak_rss_mib(),
        candidate_result.peak_rss_mib(),
    );
    print_comparison(
        "Avg RSS",
        baseline_result.avg_rss_mib(),
        candidate_result.avg_rss_mib(),
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
                ("checkpoint_baseline", &baseline_result.samples),
                ("checkpoint_candidate", &candidate_result.samples),
            ],
        )?;

        // Write results summary
        let results_path = output_dir.join("comparison_results.parquet");
        writer.write_results(&results_path, &[&baseline_result, &candidate_result])?;

        // Write JSON summary
        let json_path = output_dir.join("comparison_summary.json");
        let summary = serde_json::json!({
            "baseline": {
                "peak_memory_mib": baseline_result.peak_memory_mib(),
                "avg_memory_mib": baseline_result.avg_memory_mib(),
                "peak_rss_mib": baseline_result.peak_rss_mib(),
                "avg_rss_mib": baseline_result.avg_rss_mib(),
                "samples": baseline_result.sample_count(),
                "success": baseline_result.success,
                "save_interval_secs": baseline_save_interval,
            },
            "candidate": {
                "peak_memory_mib": candidate_result.peak_memory_mib(),
                "avg_memory_mib": candidate_result.avg_memory_mib(),
                "peak_rss_mib": candidate_result.peak_rss_mib(),
                "avg_rss_mib": candidate_result.avg_rss_mib(),
                "samples": candidate_result.sample_count(),
                "success": candidate_result.success,
                "save_interval_secs": candidate_save_interval,
            },
            "comparison": {
                "peak_memory_diff_pct": ((candidate_result.peak_memory_mib() - baseline_result.peak_memory_mib()) / baseline_result.peak_memory_mib()) * 100.0,
                "avg_memory_diff_pct": ((candidate_result.avg_memory_mib() - baseline_result.avg_memory_mib()) / baseline_result.avg_memory_mib()) * 100.0,
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
pub async fn cmd_analyze(
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
