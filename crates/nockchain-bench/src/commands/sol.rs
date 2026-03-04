use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use nockchain_bench::events::LogParser;
use nockchain_bench::scenario::{MiningScenario, MiningScenarioConfig};
use nockchain_bench::runner::NockchainMode;
use nockchain_bench::speed_of_light::{
    build_sweep_cases, checkpoint_durations_ms, checkpoint_event_num, extract_fixture_to_paths,
    find_stale_ranges, page_fault_bursts, read_fixture_file, slice_archive_file,
    summarize_case_runs, write_fixture_file_from_paths, ArchiveExtractionPhase, SolArchiveReader,
    SolBenchConfig, SolBenchRunner, BlockExtractor, CheckpointBuilder, CheckpointConfig,
    ExtractorConfig, SolFixtureManifest, SolHeight, SweepRunMetrics, PROOF_VERSION_1_START,
    PROOF_VERSION_2_START,
};

use super::{
    all_or_number, blake3_hash_hex_for_file, create_timestamped_subdir, ensure_existing_file,
    included_or_off, on_or_off, print_heading, print_heading_with_leading_newline, CutoverVersion,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchiveFixturePlan {
    checkpoint_target_height: u64,
    archive_start_height: u64,
    archive_end_height: u64,
}

pub fn archive_fixture_plan(start_height: u64, end_height: u64) -> Result<ArchiveFixturePlan, String> {
    if start_height > end_height {
        return Err(format!(
            "start height {} must be <= end height {}",
            start_height, end_height
        ));
    }

    if start_height >= end_height {
        return Err(
            "fixture build requires end height to be greater than start height".to_string(),
        );
    }

    Ok(ArchiveFixturePlan {
        checkpoint_target_height: start_height,
        archive_start_height: start_height.saturating_add(1),
        archive_end_height: end_height,
    })
}

/// Run speed-of-light benchmark (poke blocks as fast as possible)
pub async fn cmd_sol_bench(
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
) -> Result<(), Box<dyn std::error::Error>> {
    struct TempDirGuard {
        path: PathBuf,
    }
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    ensure_existing_file(&fixture, "Fixture")?;
    if !enable_checkpointing && checkpoint_every_blocks > 0 {
        return Err(
            "--checkpoint-every-blocks requires --enable-checkpointing=true (or set cadence to 0)"
                .into(),
        );
    }

    let fixture_temp_dir =
        create_timestamped_subdir(&std::env::temp_dir(), "nockchain-bench-fixture")?;

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

    print_heading("Speed-of-Light Benchmark");
    println!("Fixture: {}", fixture.display());
    println!("Archive: {}", archive_path.display());
    println!("Kernel:  {}", kernel_path.display());
    println!("Checkpoint: {}", checkpoint_path.display());
    println!(
        "Archive range: {}..={}",
        archive_start_height, archive_end_height
    );
    println!("Blocks:  {}", all_or_number(blocks));
    println!("Checkpoint mode: {}", enable_checkpointing);
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
    ensure_existing_file(&archive_path, "Archive")?;
    ensure_existing_file(&kernel_path, "Kernel")?;
    ensure_existing_file(&checkpoint_path, "Checkpoint")?;

    let config = SolBenchConfig {
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
    };

    let mut runner = SolBenchRunner::new(config);

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
pub async fn cmd_sol_checkpoint(
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
        None => create_timestamped_subdir(&std::env::temp_dir(), "nockchain-bench-sol")?,
    };

    print_heading("Speed-of-Light Checkpoint Builder");
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

    ensure_existing_file(&archive, "Archive")?;
    ensure_existing_file(&kernel, "Kernel")?;
    if let Some(ref checkpoint_path) = checkpoint {
        ensure_existing_file(checkpoint_path, "Checkpoint")?;
    }

    let config = CheckpointConfig {
        archive_path: archive.to_string_lossy().to_string(),
        kernel_path: kernel.to_string_lossy().to_string(),
        checkpoint_path: checkpoint.map(|p| p.to_string_lossy().to_string()),
        start_height: start_height.map(SolHeight),
        target_height: SolHeight(target_height),
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
pub async fn cmd_sol_extract(
    blocks: u64,
    start_height: u64,
    end_height: Option<u64>,
    checkpoint: PathBuf,
    kernel: PathBuf,
    output: Option<PathBuf>,
    chunk_size: u64,
    include_mempool: bool,
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

    print_heading("Speed-of-Light Block Extraction");
    println!("Checkpoint: {}", checkpoint.display());
    println!("Kernel:     {}", kernel.display());
    println!("Range:      {}..={}", start_height, resolved_end_height);
    println!("Blocks:     {}", target_blocks);
    println!("Chunk size: {}", chunk_size);
    println!("Mempool:    {}", included_or_off(include_mempool));
    println!("Output:     {}", output_path.display());
    println!();

    // Check files exist
    ensure_existing_file(&checkpoint, "Checkpoint")?;
    ensure_existing_file(&kernel, "Kernel")?;

    let config = ExtractorConfig {
        checkpoint_path: checkpoint.to_string_lossy().to_string(),
        kernel_path: kernel.to_string_lossy().to_string(),
        block_count: blocks,
        chunk_size,
        work_dir: PathBuf::from("."),
        include_mempool,
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

    print_heading_with_leading_newline("Extraction Complete");
    println!("Archive:    {}", output_path.display());
    println!("Size:       {:.2} MiB", file_size as f64 / 1024.0 / 1024.0);
    println!("Time:       {:.1}s", extract_time.as_secs_f64());
    println!(
        "Throughput: {:.1} blocks/s",
        target_blocks as f64 / extract_time.as_secs_f64()
    );

    Ok(())
}

/// Build a `.soltest` fixture directly from an input archive and kernel.
pub async fn cmd_sol_fixture_build(
    archive: PathBuf,
    kernel: PathBuf,
    start_height: u64,
    end_height: u64,
    output: PathBuf,
    include_mempool: bool,
    chunk_size: u64,
    work_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    if chunk_size == 0 {
        return Err("--chunk-size must be greater than 0".into());
    }
    ensure_existing_file(&archive, "Archive")?;
    ensure_existing_file(&kernel, "Kernel")?;

    let plan = archive_fixture_plan(start_height, end_height)
        .map_err(|e| format!("Invalid fixture plan: {e}"))?;

    let archive_reader = SolArchiveReader::from_file(&archive)?;
    let source_min = archive_reader.min_height().as_u64();
    let source_max = archive_reader.max_height().as_u64();
    drop(archive_reader);

    if start_height < source_min || end_height > source_max {
        return Err(format!(
            "Requested range {}..={} is outside source archive range {}..={}",
            start_height, end_height, source_min, source_max
        )
        .into());
    }
    if plan.checkpoint_target_height < source_min || plan.checkpoint_target_height > source_max {
        return Err(format!(
            "Checkpoint target height {} is outside source archive range {}..={}",
            plan.checkpoint_target_height, source_min, source_max
        )
        .into());
    }

    print_heading("Speed-of-Light Fixture Build (Archive Source)");
    println!("Source archive:    {}", archive.display());
    println!("Kernel:            {}", kernel.display());
    println!("Requested range:   {}..={}", start_height, end_height);
    println!(
        "Embedded checkpoint height: {}",
        plan.checkpoint_target_height
    );
    println!(
        "Fixture archive range:      {}..={}",
        plan.archive_start_height, plan.archive_end_height
    );
    println!("Mempool:           {}", included_or_off(include_mempool));
    println!("Chunk size:        {}", chunk_size);
    println!("Output fixture:    {}", output.display());
    println!("Work dir:          {}", work_dir.display());
    println!();

    std::fs::create_dir_all(&work_dir)?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let run_dir = create_timestamped_subdir(&work_dir, "sol-fixture-archive")?;

    let sliced_archive_path = run_dir.join("test.solarch");
    let checkpoint_output_path = run_dir.join("embedded.chkjam");
    let checkpoint_work_dir = run_dir.join("checkpoint-work");
    std::fs::create_dir_all(&checkpoint_work_dir)?;

    println!(
        "Slicing archive to {}..={}...",
        plan.archive_start_height, plan.archive_end_height
    );
    let slice_result = slice_archive_file(
        &archive,
        &sliced_archive_path,
        SolHeight(plan.archive_start_height),
        SolHeight(plan.archive_end_height),
        include_mempool,
    )?;
    println!(
        "  sliced blocks: {} ({}..={})",
        slice_result.block_count,
        slice_result.start_height.as_u64(),
        slice_result.end_height.as_u64()
    );
    if include_mempool {
        println!(
            "  sliced mempool snapshots: {}",
            slice_result.mempool_snapshot_count
        );
    }

    println!(
        "Building checkpoint at height {} from source archive...",
        plan.checkpoint_target_height
    );
    let mut checkpoint_builder = CheckpointBuilder::new(CheckpointConfig {
        archive_path: archive.to_string_lossy().to_string(),
        kernel_path: kernel.to_string_lossy().to_string(),
        checkpoint_path: None,
        start_height: Some(SolHeight::ZERO),
        target_height: SolHeight(plan.checkpoint_target_height),
        output_path: checkpoint_output_path.clone(),
        work_dir: checkpoint_work_dir,
    });
    checkpoint_builder.run().await?;

    let embedded_event_num = checkpoint_event_num(&checkpoint_output_path)?;
    let fixture_manifest = SolFixtureManifest {
        format_version: 2,
        source_archive_path: archive.to_string_lossy().to_string(),
        source_archive_event_num: embedded_event_num,
        derived_checkpoint_height: SolHeight(plan.checkpoint_target_height),
        derived_checkpoint_event_num: embedded_event_num,
        archive_start_height: SolHeight(plan.archive_start_height),
        archive_end_height: SolHeight(plan.archive_end_height),
        include_mempool,
        chunk_size,
        kernel_hash_hex: blake3_hash_hex_for_file(&kernel)?,
        checkpoint_hash_hex: blake3_hash_hex_for_file(&checkpoint_output_path)?,
        archive_hash_hex: blake3_hash_hex_for_file(&sliced_archive_path)?,
    };

    println!("Packaging .soltest fixture...");
    write_fixture_file_from_paths(
        &output, &fixture_manifest, &checkpoint_output_path, &sliced_archive_path, &kernel,
    )?;

    println!("\nFixture created:");
    println!("  Path:              {}", output.display());
    println!(
        "  Embedded checkpoint: {} (event {})",
        plan.checkpoint_target_height, embedded_event_num
    );
    println!(
        "  Archive range:      {}..={}",
        plan.archive_start_height, plan.archive_end_height
    );
    Ok(())
}

/// Inspect a unified `.soltest` fixture.
pub fn cmd_sol_fixture_inspect(fixture: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    print_heading("Speed-of-Light Fixture Inspect");
    println!("Fixture: {}", fixture.display());
    println!();

    ensure_existing_file(&fixture, "Fixture")?;

    let data = read_fixture_file(&fixture)?;
    let m = data.manifest;
    println!("Format version:            {}", m.format_version);
    println!("Source archive path:       {}", m.source_archive_path);
    println!("Source archive event:      {}", m.source_archive_event_num);
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
        on_or_off(m.include_mempool)
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

/// Inspect mempool snapshots for stale transactions
pub fn cmd_sol_inspect(archive: PathBuf, retain: u64) -> Result<(), Box<dyn std::error::Error>> {
    print_heading("Speed-of-Light Mempool Inspector");
    println!("Archive: {}", archive.display());
    println!("Retain:  {} blocks", retain);
    println!();

    ensure_existing_file(&archive, "Archive")?;

    let reader = SolArchiveReader::from_file(&archive)?;
    let ranges = find_stale_ranges(&reader, retain)?;

    println!(
        "Snapshots: {} (mempool: {})",
        reader.mempool_snapshot_count(),
        on_or_off(reader.has_mempool())
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

pub async fn cmd_sol_sweep(
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
    output_json: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let candidates = parse_csv_strings(candidates_csv);
    let chunk_sizes = parse_csv_u64(chunk_sizes_csv)?;
    let memory_limits = parse_csv_strings(memory_limits_csv);

    if candidates.is_empty() {
        return Err("No candidates provided".into());
    }
    if chunk_sizes.is_empty() {
        return Err("No chunk sizes provided".into());
    }
    if memory_limits.is_empty() {
        return Err("No memory limits provided".into());
    }

    let cases = build_sweep_cases(&candidates, &chunk_sizes, &memory_limits);

    print_heading("Speed-of-Light Sweep");
    println!("Cases: {}", cases.len());
    println!("Repeats: {}", repeats);
    println!("Duration per run: {}s", duration);
    println!("Sample interval: {}s", sample_interval);
    println!("Save interval: {}s", save_interval);
    println!("Image: {}", image);
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
                env_vars: HashMap::new(),
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

    print_heading_with_leading_newline("Sweep Summary");
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
                "data_dir": data_dir,
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&payload)?)?;
        println!("\nSweep JSON written to {}", path.display());
    }

    Ok(())
}

pub fn parse_csv_strings(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

pub fn parse_csv_u64(input: &str) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
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

pub fn sanitize_case_value(value: &str) -> String {
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

pub fn latest_checkpoint_size_in_dir(dir: &std::path::Path) -> Result<Option<u64>, std::io::Error> {
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
    fn test_archive_fixture_plan_uses_checkpoint_at_range_start() {
        let plan = archive_fixture_plan(10, 42).expect("fixture plan");
        assert_eq!(plan.checkpoint_target_height, 10);
        assert_eq!(plan.archive_start_height, 11);
        assert_eq!(plan.archive_end_height, 42);
    }

    #[test]
    fn test_archive_fixture_plan_rejects_empty_replay_window() {
        let err = archive_fixture_plan(7, 7).expect_err("requires replay block after checkpoint");
        assert!(err.contains("end height to be greater than start height"));
    }
}
