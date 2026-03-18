use std::path::PathBuf;
use std::time::Duration;

use nockchain_bench::speed_of_light::{
    checkpoint_event_num, current_binary_identity, execute_docker_trusted_run,
    execute_docker_validation, execute_native_cpu_profile, execute_native_trusted_run,
    execute_once, execute_once_with_options, execute_sweep, find_stale_ranges, parse_matrix_value,
    read_fixture_file, resolve_requested_case, run_validation_probe, slice_archive_file,
    write_fixture_file_from_paths, ArchiveExtractionPhase, BlockExtractor, CheckpointBuilder,
    CheckpointConfig, CpuProfilerConfig, CpuProfilerKind, ExecuteOptions, ExecutionRequest,
    ExtractorConfig, HarnessSweepExecutor, RequestedCase, ScheduleMode, SolArchiveReader,
    SolFixtureManifest, SolHeight, SweepRunOptions, Validity, WorkDirMode, PROOF_VERSION_1_START,
    PROOF_VERSION_2_START,
};
use nockchain_bench::speed_of_light::harness::HarnessError;

use super::{
    all_or_number, blake3_hash_hex_for_file, create_timestamped_subdir, ensure_existing_file,
    included_or_off, on_or_off, print_heading, print_heading_with_leading_newline, CutoverVersion,
};
use crate::BenchWorkDirMode;

// Keep extraction/fixture chunking internal-only unless we have a concrete need
// to expose it again.
const INTERNAL_SOL_CHUNK_SIZE: u64 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchiveFixturePlan {
    checkpoint_target_height: u64,
    archive_start_height: u64,
    archive_end_height: u64,
}

pub fn archive_fixture_plan(
    start_height: u64,
    end_height: u64,
) -> Result<ArchiveFixturePlan, String> {
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

fn build_cpu_profiler_config(
    kind: CpuProfilerKind,
    sample_rate_hz: u32,
) -> Result<CpuProfilerConfig, HarnessError> {
    if sample_rate_hz == 0 {
        return Err(HarnessError::InvalidRequestedCase(
            "--cpu-profile-rate must be greater than 0".to_string(),
        ));
    }
    Ok(CpuProfilerConfig {
        kind,
        sample_rate_hz,
    })
}

fn build_requested_case(
    fixture: PathBuf,
    execution: ExecutionRequest,
    blocks: u64,
    enable_checkpointing: bool,
    skip_genesis: bool,
    profile_memory: bool,
    profile_interval_ms: u64,
    checkpoint_every_blocks: u64,
    label: Option<String>,
    threads: u32,
    warmup_runs: u32,
    measured_runs: u32,
    cooldown_secs: u64,
) -> RequestedCase {
    let mut requested = RequestedCase::native(fixture);
    requested.blocks = blocks;
    requested.enable_checkpointing = enable_checkpointing;
    requested.skip_genesis = skip_genesis;
    requested.profile_memory = profile_memory;
    requested.profile_interval_ms = profile_interval_ms;
    requested.checkpoint_every_blocks = checkpoint_every_blocks;
    requested.label = label;
    requested.execution = execution;
    requested.threads = threads;
    requested.warmup_runs = warmup_runs;
    requested.measured_runs = measured_runs;
    requested.cooldown_secs = cooldown_secs;
    requested
}

fn build_execute_options(
    checkpoint_recovery_timeout_ms: u64,
    checkpoint_recovery_tolerance_pct: f64,
    gc_drop_threshold_mib: u64,
    page_fault_minor_burst_threshold: u64,
    page_fault_major_burst_threshold: u64,
) -> ExecuteOptions {
    ExecuteOptions {
        checkpoint_recovery_timeout_ms,
        checkpoint_recovery_tolerance_pct,
        gc_drop_threshold_mib,
        page_fault_minor_burst_threshold,
        page_fault_major_burst_threshold,
    }
}

fn verdict_label(validity: &Validity) -> &'static str {
    match validity {
        Validity::Valid => "Valid",
        Validity::Partial { .. } => "Partial",
        Validity::Invalid { .. } => "Invalid",
    }
}

fn docker_work_dir_mode(mode: BenchWorkDirMode) -> WorkDirMode {
    match mode {
        BenchWorkDirMode::HostBind => WorkDirMode::HostBind,
        BenchWorkDirMode::DockerVolume => WorkDirMode::DockerVolume,
        BenchWorkDirMode::DockerTmpfs => WorkDirMode::DockerTmpfs,
    }
}

/// Run a quick speed-of-light benchmark for inner-loop iteration only.
pub async fn cmd_sol_quick_bench(
    fixture: PathBuf,
    blocks: u64,
    enable_checkpointing: bool,
    skip_genesis: bool,
    profile_memory: bool,
    profile_interval_ms: u64,
    profile_output: Option<PathBuf>,
    cpu_profiler: Option<CpuProfilerKind>,
    cpu_profile_rate: u32,
    cpu_profile_output: Option<PathBuf>,
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

    let requested = build_requested_case(
        fixture.clone(),
        ExecutionRequest::Native,
        blocks,
        enable_checkpointing,
        skip_genesis,
        profile_memory,
        profile_interval_ms,
        checkpoint_every_blocks,
        None,
        1,
        1,
        5,
        0,
    );
    let execute_options = build_execute_options(
        checkpoint_recovery_timeout_ms, checkpoint_recovery_tolerance_pct, gc_drop_threshold_mib,
        page_fault_minor_burst_threshold, page_fault_major_burst_threshold,
    );
    let cpu_profiler = cpu_profiler
        .map(|kind| build_cpu_profiler_config(kind, cpu_profile_rate))
        .transpose()?;
    let resolved = resolve_requested_case(&requested)?;
    let artifact_root = create_timestamped_subdir(&std::env::temp_dir(), "nockchain-bench-bench")?;
    let artifact_guard = TempDirGuard {
        path: artifact_root.clone(),
    };

    print_heading("Speed-of-Light Quick Benchmark");
    println!("Fixture: {}", fixture.display());
    println!(
        "Archive range: {}..={}",
        resolved.fixture_manifest.archive_start_height.as_u64(),
        resolved.fixture_manifest.archive_end_height.as_u64()
    );
    println!("Blocks:  {}", all_or_number(blocks));
    println!("Checkpoint mode: {}", enable_checkpointing);
    println!("Skip genesis: {}", skip_genesis);
    println!(
        "Start height: {}",
        resolved.fixture_manifest.archive_start_height.as_u64()
    );
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
    if let Some(ref out) = cpu_profile_output {
        println!("CPU profile output: {}", out.display());
    }
    println!();

    let completed = execute_once_with_options(
        &resolved,
        "bench",
        &artifact_root.join("runs/bench"),
        &execute_options,
    )
    .await?;
    let results = completed.bench_results.as_ref().ok_or_else(|| {
        completed
            .record
            .error
            .clone()
            .unwrap_or_else(|| "benchmark run failed".to_string())
    })?;

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
            "memory_profile": completed.profile,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&payload)?)?;
        println!("Profile JSON written to {}", path.display());
    }

    if let (Some(config), Some(path)) = (cpu_profiler, cpu_profile_output) {
        std::fs::write(
            artifact_root.join("resolved_case.json"),
            serde_json::to_vec_pretty(&resolved)?,
        )?;
        let artifact = execute_native_cpu_profile(&artifact_root, config).await?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(artifact_root.join(&artifact.output_relative_path), &path)?;
        println!("CPU profile written to {}", path.display());
    }

    drop(artifact_guard);
    Ok(())
}

pub async fn cmd_sol_bench(
    fixture: PathBuf,
    output: PathBuf,
    blocks: u64,
    enable_checkpointing: bool,
    skip_genesis: bool,
    profile_memory: bool,
    profile_interval_ms: u64,
    checkpoint_every_blocks: u64,
    threads: u32,
    warmup_runs: u32,
    measured_runs: u32,
    cooldown_secs: u64,
    label: Option<String>,
    image_tag: Option<String>,
    memory_limit: Option<String>,
    work_dir_mode: Option<BenchWorkDirMode>,
    cpuset: Option<String>,
    cpu_quota: Option<i64>,
    cpu_period: Option<i64>,
    allow_version_skew: bool,
    allow_debug_benchmark: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_existing_file(&fixture, "Fixture")?;

    let execution = match image_tag {
        Some(image_tag) => {
            let memory_limit = memory_limit
                .ok_or("--memory-limit is required when --image-tag selects Docker execution")?;
            let work_dir_mode = work_dir_mode
                .ok_or("--work-dir-mode is required when --image-tag selects Docker execution")?;
            ExecutionRequest::Docker {
                image_tag,
                memory_limit,
                cpuset,
                cpu_quota,
                cpu_period,
                work_dir_mode: docker_work_dir_mode(work_dir_mode),
                allow_version_skew,
            }
        }
        None => ExecutionRequest::Native,
    };

    let requested = build_requested_case(
        fixture.clone(),
        execution,
        blocks,
        enable_checkpointing,
        skip_genesis,
        profile_memory,
        profile_interval_ms,
        checkpoint_every_blocks,
        label,
        threads,
        warmup_runs,
        measured_runs,
        cooldown_secs,
    );

    print_heading("Speed-of-Light Trusted Benchmark");
    println!("Fixture: {}", fixture.display());
    println!("Output:  {}", output.display());
    println!("Blocks:  {}", all_or_number(blocks));
    println!("Threads: {}", threads);
    println!("Warmups: {}", warmup_runs);
    println!("Measured runs: {}", measured_runs);
    println!("Cooldown: {}s", cooldown_secs);
    println!();

    let run = match &requested.execution {
        ExecutionRequest::Native => {
            execute_native_trusted_run(requested, &output, allow_debug_benchmark, None).await?
        }
        ExecutionRequest::Docker { .. } => {
            execute_docker_trusted_run(requested, &output, allow_debug_benchmark, None)
                .await?
                .into()
        }
    };
    println!("Artifact root: {}", output.display());
    println!("Verdict: {}", verdict_label(&run.verdict.validity));
    println!(
        "Measured runs succeeded: {}/{}",
        run.summary.measured_runs_succeeded, run.summary.measured_runs_requested
    );

    if let Some(throughput) = &run.summary.throughput_blocks_per_second {
        println!(
            "Throughput median: {:.2} blocks/s (cv {:.3})",
            throughput.median, throughput.cv
        );
    }

    Ok(())
}

pub async fn cmd_sol_run_once(
    resolved_case: PathBuf,
    run_dir: PathBuf,
    run_id: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_existing_file(&resolved_case, "Resolved case")?;

    let resolved = serde_json::from_slice::<nockchain_bench::speed_of_light::ResolvedCase>(
        &std::fs::read(&resolved_case)?,
    )?;
    let run_id = run_id.unwrap_or_else(|| {
        run_dir
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("run")
            .to_string()
    });

    std::fs::create_dir_all(&run_dir)?;
    std::fs::write(
        run_dir.join(".benchmark.pid"),
        format!("{}\n", std::process::id()),
    )?;
    execute_once(&resolved, &run_id, &run_dir).await?;
    Ok(())
}

pub fn cmd_sol_binary_identity() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        serde_json::to_string_pretty(&current_binary_identity())?
    );
    Ok(())
}

pub async fn cmd_sol_validate(
    fixture: PathBuf,
    output: PathBuf,
    image_tag: String,
    memory_limit: String,
    work_dir_mode: BenchWorkDirMode,
    cpuset: Option<String>,
    cpu_quota: Option<i64>,
    cpu_period: Option<i64>,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_existing_file(&fixture, "Fixture")?;

    let requested = build_requested_case(
        fixture.clone(),
        ExecutionRequest::Docker {
            image_tag,
            memory_limit,
            cpuset,
            cpu_quota,
            cpu_period,
            work_dir_mode: docker_work_dir_mode(work_dir_mode),
            allow_version_skew: false,
        },
        0,
        true,
        false,
        false,
        500,
        0,
        None,
        1,
        0,
        3,
        0,
    );

    print_heading("Speed-of-Light Docker Validation");
    println!("Fixture: {}", fixture.display());
    println!("Output:  {}", output.display());
    println!();

    let validation = execute_docker_validation(requested, &output).await?;
    println!("Validation: {:?}", validation.status);
    println!("From cache: {}", validation.from_cache);
    if let Some(reason) = validation.failure_reason {
        println!("Reason: {reason}");
    }

    Ok(())
}

pub async fn cmd_sol_sweep(
    matrix: PathBuf,
    output: PathBuf,
    allow_multi_axis: bool,
    interleave: bool,
    randomize_order: bool,
    comparison_markdown: bool,
    cpu_profiler: Option<CpuProfilerKind>,
    cpu_profile_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let matrix_value = serde_json::from_slice::<serde_json::Value>(&std::fs::read(&matrix)?)?;
    let parsed_matrix = parse_matrix_value(matrix_value.clone())?;
    let (schedule_mode, random_seed) = resolve_sweep_schedule(interleave, randomize_order)?;
    let cpu_profiler = cpu_profiler
        .map(|kind| build_cpu_profiler_config(kind, cpu_profile_rate))
        .transpose()?;

    print_heading("Speed-of-Light Trusted Sweep");
    println!("Matrix: {}", matrix.display());
    println!("Output: {}", output.display());
    println!("Allow multi-axis: {}", allow_multi_axis);
    println!("Schedule: {:?}", schedule_mode);
    println!("Comparison markdown: {}", comparison_markdown);
    println!();

    let mut executor = HarnessSweepExecutor;
    let result = execute_sweep(
        &matrix_value,
        parsed_matrix,
        &output,
        &SweepRunOptions {
            allow_multi_axis,
            schedule_mode,
            random_seed,
            comparison_markdown,
            allow_debug_benchmark: false,
            cpu_profiler,
        },
        &mut executor,
    )
    .await?;

    println!("Artifact root: {}", output.display());
    println!("Cases: {}", result.comparison.case_count);
    println!("Verdict: {}", verdict_label(&result.verdict.validity));
    if !result.comparison.invariant_violations.is_empty() {
        println!(
            "Invariant violations: {}",
            result.comparison.invariant_violations.len()
        );
    }

    Ok(())
}

fn resolve_sweep_schedule(
    interleave: bool,
    randomize_order: bool,
) -> Result<(ScheduleMode, Option<u64>), Box<dyn std::error::Error>> {
    if interleave && randomize_order {
        return Err("choose at most one of --interleave or --randomize-order".into());
    }

    if interleave {
        return Ok((ScheduleMode::Interleaved, None));
    }

    if randomize_order {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0)
            ^ std::process::id() as u64;
        return Ok((ScheduleMode::Randomized, Some(seed)));
    }

    Ok((ScheduleMode::Sequential, None))
}

pub fn cmd_sol_validate_probe() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        serde_json::to_string_pretty(&run_validation_probe()?)?
    );
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
    include_mempool: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if blocks == 0 && end_height.is_none() {
        return Err("--blocks must be > 0 when --end-height is not provided".into());
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
        chunk_size: INTERNAL_SOL_CHUNK_SIZE,
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
    work_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
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
        chunk_size: INTERNAL_SOL_CHUNK_SIZE,
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_resolve_sweep_schedule_rejects_conflicting_flags() {
        let error =
            resolve_sweep_schedule(true, true).expect_err("interleave and randomize conflict");
        assert!(error.to_string().contains("choose at most one"));
    }

    #[test]
    fn test_resolve_sweep_schedule_randomized_mode_uses_generated_seed() {
        let (mode, seed) = resolve_sweep_schedule(false, true).expect("randomized schedule");
        assert_eq!(mode, ScheduleMode::Randomized);
        assert!(seed.is_some());
    }

    #[test]
    fn test_build_cpu_profiler_config_rejects_zero_sample_rate() {
        let error = build_cpu_profiler_config(CpuProfilerKind::Samply, 0)
            .expect_err("zero sample rate should fail");
        assert!(error.to_string().contains("greater than 0"));
    }
}
