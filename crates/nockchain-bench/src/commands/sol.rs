use std::path::{Path, PathBuf};
use std::time::Duration;

use nockchain_bench::speed_of_light::harness::profiler::ensure_samply_profiled_binary;
use nockchain_bench::speed_of_light::harness::{
    build_samply_record_command, preflight_samply_profiler, run_samply_record_command,
    HarnessError, RequestedOrchestrate,
};
use nockchain_bench::speed_of_light::{
    checkpoint_event_num, current_binary_identity, execute_docker_trusted_run,
    execute_docker_validation, execute_native_cpu_profile_for_resolved_case,
    execute_native_trusted_run, execute_once, execute_once_with_options, execute_sweep,
    find_stale_ranges, parse_matrix_value, read_fixture_file, resolve_requested_case,
    run_validation_probe, slice_archive_file, write_fixture_file_from_paths,
    ArchiveExtractionPhase, BlockExtractor, CheckpointBuildMode, CheckpointBuilder,
    CheckpointConfig, ColdMode, CpuProfilerConfig, CpuProfilerKind, DockerImageSource,
    ExecuteOptions, ExecutionRequest, ExtractorConfig, HarnessSweepExecutor, PeekBenchConfig,
    PeekBenchError, PeekBenchResults, PeekBenchRunner, PeekMode, PeekRangeRequest,
    QuickOrchestrateResults, QuickOrchestrateRunner, RequestedCase, ScheduleMode, SolArchiveReader,
    SolFixtureCheckpointKind, SolFixtureManifest, SolHeight, SweepRunOptions, Validity,
    WorkDirMode, PROOF_VERSION_1_START, PROOF_VERSION_2_START,
};

use super::{
    all_or_number, blake3_hash_hex_for_file, create_timestamped_subdir, ensure_existing_file,
    included_or_off, on_or_off, print_heading, print_heading_with_leading_newline, CutoverVersion,
    TempDirGuard,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixtureCheckpointKind {
    Derived,
    Full,
}

impl FixtureCheckpointKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Derived => "derived",
            Self::Full => "full",
        }
    }
}

pub struct QuickBenchOptions {
    pub fixture: PathBuf,
    pub blocks: u64,
    pub enable_checkpointing: bool,
    pub skip_genesis: bool,
    pub fsync: bool,
    pub profile_memory: bool,
    pub profile_interval_ms: u64,
    pub profile_output: Option<PathBuf>,
    pub cpu_profiler: Option<CpuProfilerKind>,
    pub cpu_profile_rate: u32,
    pub cpu_profile_output: Option<PathBuf>,
    pub checkpoint_every_blocks: u64,
    pub checkpoint_recovery_timeout_ms: u64,
    pub checkpoint_recovery_tolerance_pct: f64,
    pub gc_drop_threshold_mib: u64,
    pub page_fault_minor_burst_threshold: u64,
    pub page_fault_major_burst_threshold: u64,
}

pub struct QuickReadBenchOptions {
    pub checkpoint: PathBuf,
    pub kernel: PathBuf,
    pub start_height: u64,
    pub end_height: Option<u64>,
    pub count: Option<u64>,
    #[cfg(feature = "pma-runtime-compat")]
    pub fsync: bool,
    pub dry_run: bool,
    pub profile_memory: bool,
    pub profile_interval_ms: u64,
    pub profile_output: Option<PathBuf>,
    pub cpu_profiler: Option<CpuProfilerKind>,
    pub cpu_profile_rate: u32,
    pub cpu_profile_output: Option<PathBuf>,
}

pub struct QuickOrchestrateOptions {
    pub plan: PathBuf,
    pub profile_output: Option<PathBuf>,
    pub fsync: bool,
    pub cold_mode: ColdMode,
}

struct QuickReadRunContext {
    checkpoint: PathBuf,
    kernel: PathBuf,
    runner: PeekBenchRunner,
    work_dir_guard: TempDirGuard,
}

impl From<FixtureCheckpointKind> for SolFixtureCheckpointKind {
    fn from(value: FixtureCheckpointKind) -> Self {
        match value {
            FixtureCheckpointKind::Derived => Self::Derived,
            FixtureCheckpointKind::Full => Self::Full,
        }
    }
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

fn build_quick_read_bench_config(
    options: &QuickReadBenchOptions,
    checkpoint: PathBuf,
    kernel: PathBuf,
    work_dir: PathBuf,
) -> Result<PeekBenchConfig, PeekBenchError> {
    Ok(PeekBenchConfig {
        checkpoint_path: checkpoint,
        kernel_path: kernel,
        start_height: options.start_height,
        range: PeekRangeRequest::from_bounds(options.end_height, options.count)?,
        #[cfg(feature = "pma-runtime-compat")]
        fsync: options.fsync,
        #[cfg(not(feature = "pma-runtime-compat"))]
        fsync: nockchain_bench::speed_of_light::DEFAULT_FSYNC_ENABLED,
        dry_run: options.dry_run,
        profile_memory: options.profile_memory,
        profile_interval_ms: options.profile_interval_ms,
        work_dir,
    })
}

fn build_quick_read_cpu_profile_command(
    binary: &Path,
    checkpoint: &Path,
    kernel: &Path,
    start_height: u64,
    end_height: u64,
    dry_run: bool,
    fsync_enabled: Option<bool>,
) -> Vec<String> {
    let mut command = vec![
        binary.to_string_lossy().to_string(),
        "sol".to_string(),
        "quick-read-once".to_string(),
        "--checkpoint".to_string(),
        checkpoint.to_string_lossy().to_string(),
        "--kernel".to_string(),
        kernel.to_string_lossy().to_string(),
        "--start-height".to_string(),
        start_height.to_string(),
        "--end-height".to_string(),
        end_height.to_string(),
    ];

    if let Some(fsync_enabled) = fsync_enabled {
        command.extend([
            "--fsync".to_string(),
            nockchain_bench::speed_of_light::fsync_mode_label(fsync_enabled).to_string(),
        ]);
    }

    if dry_run {
        command.push("--dry-run".to_string());
    }

    command
}

fn build_quick_read_profile_output_payload(
    checkpoint: &Path,
    kernel: &Path,
    results: &PeekBenchResults,
) -> serde_json::Value {
    results.profile_output_value(checkpoint, kernel)
}

fn write_profile_output(
    path: &Path,
    payload: impl AsRef<[u8]>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, payload)?;
    println!("Profile JSON written to {}", path.display());
    Ok(())
}

fn prepare_quick_read_run(
    options: &QuickReadBenchOptions,
    temp_dir_prefix: &str,
) -> Result<QuickReadRunContext, Box<dyn std::error::Error>> {
    ensure_existing_file(&options.checkpoint, "Checkpoint")?;
    ensure_existing_file(&options.kernel, "Kernel")?;

    let checkpoint = std::fs::canonicalize(&options.checkpoint)?;
    let kernel = std::fs::canonicalize(&options.kernel)?;
    let work_dir = create_timestamped_subdir(&std::env::temp_dir(), temp_dir_prefix)?;
    let work_dir_guard = TempDirGuard::new(work_dir.clone());
    let runner = PeekBenchRunner::new(build_quick_read_bench_config(
        options,
        checkpoint.clone(),
        kernel.clone(),
        work_dir,
    )?);

    Ok(QuickReadRunContext {
        checkpoint,
        kernel,
        runner,
        work_dir_guard,
    })
}

fn finalize_quick_orchestrate_output(
    profile_output: Option<&Path>,
    payload: &str,
    step_failure: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(path) = profile_output {
        write_profile_output(path, payload.as_bytes())?;
    }
    if let Some(error) = step_failure {
        return Err(error.to_string().into());
    }
    Ok(())
}

fn finalize_quick_orchestrate_results(
    results: &QuickOrchestrateResults,
    profile_output: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let payload = results.to_compact_json()?;
    finalize_quick_orchestrate_output(profile_output, &payload, results.failure_message())
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
    if let RequestedOrchestrate::GeneratedReplay {
        blocks: replay_blocks,
        skip_genesis: replay_skip_genesis,
        ..
    } = &mut requested.orchestrate
    {
        *replay_blocks = Some(blocks);
        *replay_skip_genesis = skip_genesis;
    }
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
    options: QuickBenchOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let QuickBenchOptions {
        fixture,
        blocks,
        enable_checkpointing,
        skip_genesis,
        fsync,
        profile_memory,
        profile_interval_ms,
        profile_output,
        cpu_profiler,
        cpu_profile_rate,
        cpu_profile_output,
        checkpoint_every_blocks,
        checkpoint_recovery_timeout_ms,
        checkpoint_recovery_tolerance_pct,
        gc_drop_threshold_mib,
        page_fault_minor_burst_threshold,
        page_fault_major_burst_threshold,
    } = options;

    ensure_existing_file(&fixture, "Fixture")?;
    if enable_checkpointing || checkpoint_every_blocks > 0 {
        return Err("quick-bench does not support checkpoint cadence controls; use explicit orchestrate checkpoint-save steps when available".into());
    }

    let mut requested = build_requested_case(
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
    requested.set_fsync_enabled(fsync);
    let execute_options = build_execute_options(
        checkpoint_recovery_timeout_ms, checkpoint_recovery_tolerance_pct, gc_drop_threshold_mib,
        page_fault_minor_burst_threshold, page_fault_major_burst_threshold,
    );
    let cpu_profiler = cpu_profiler
        .map(|kind| build_cpu_profiler_config(kind, cpu_profile_rate))
        .transpose()?;
    let resolved = resolve_requested_case(&requested)?;
    let artifact_root = create_timestamped_subdir(&std::env::temp_dir(), "nockchain-bench-bench")?;
    let artifact_guard = TempDirGuard::new(artifact_root.clone());

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
    #[cfg(feature = "pma-runtime-compat")]
    println!(
        "Fsync: {}",
        nockchain_bench::speed_of_light::fsync_mode_label(fsync)
    );
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
        let resolved_case_path = artifact_root.join("quick_cpu_profile_resolved_case.json");
        std::fs::write(&resolved_case_path, serde_json::to_vec_pretty(&resolved)?)?;
        let artifact = execute_native_cpu_profile_for_resolved_case(
            &artifact_root, &resolved_case_path, config,
        )
        .await?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(artifact_root.join(&artifact.output_relative_path), &path)?;
        println!("CPU profile written to {}", path.display());
    }

    drop(artifact_guard);
    Ok(())
}

pub async fn cmd_sol_quick_read_bench(
    options: QuickReadBenchOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let QuickReadRunContext {
        checkpoint,
        kernel,
        mut runner,
        work_dir_guard,
    } = prepare_quick_read_run(&options, "nockchain-bench-quick-read")?;

    print_heading("Speed-of-Light Quick Read Benchmark");
    println!("Checkpoint: {}", checkpoint.display());
    println!("Kernel:     {}", kernel.display());
    println!("Start height: {}", options.start_height);
    match (options.end_height, options.count) {
        (Some(end_height), None) => println!("End height:   {}", end_height),
        (None, Some(count)) => println!("Count:        {}", count),
        (None, None) => println!("Range end:    tip"),
        (Some(_), Some(_)) => {}
    }
    println!("Dry run:      {}", options.dry_run);
    println!("Profile memory: {}", options.profile_memory);
    if options.profile_memory {
        println!("Profile interval: {}ms", options.profile_interval_ms);
    }
    if let Some(ref out) = options.profile_output {
        println!("Profile output: {}", out.display());
    }
    if let Some(ref out) = options.cpu_profile_output {
        println!("CPU profile output: {}", out.display());
    }
    println!();

    let results = runner.run().await?;
    results.print_summary();

    if let Some(path) = options.profile_output.as_ref() {
        let payload = build_quick_read_profile_output_payload(&checkpoint, &kernel, &results);
        write_profile_output(path, serde_json::to_vec_pretty(&payload)?)?;
    }

    if let (Some(profiler), Some(output_path)) =
        (options.cpu_profiler, options.cpu_profile_output.as_ref())
    {
        let config = build_cpu_profiler_config(profiler, options.cpu_profile_rate)?;
        let current_exe = std::env::current_exe()?;
        let profiled_binary = ensure_samply_profiled_binary(&current_exe)?;
        let profiled_command = build_quick_read_cpu_profile_command(
            &profiled_binary,
            &checkpoint,
            &kernel,
            results.range.start_height,
            results.range.end_height,
            options.dry_run,
            #[cfg(feature = "pma-runtime-compat")]
            Some(options.fsync),
            #[cfg(not(feature = "pma-runtime-compat"))]
            None,
        );

        preflight_samply_profiler().await?;
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let command =
            build_samply_record_command(config.sample_rate_hz, output_path, &profiled_command)?;
        run_samply_record_command(&command, output_path).await?;
        println!("CPU profile written to {}", output_path.display());
    }

    drop(work_dir_guard);
    Ok(())
}

pub async fn cmd_sol_quick_read_once(
    options: QuickReadBenchOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let QuickReadRunContext {
        mut runner,
        work_dir_guard,
        ..
    } = prepare_quick_read_run(&options, "nockchain-bench-quick-read-once")?;
    let results = runner.run().await?;
    results.print_summary();

    drop(work_dir_guard);
    Ok(())
}

pub async fn cmd_sol_quick_orchestrate(
    options: QuickOrchestrateOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_existing_file(&options.plan, "Plan")?;

    let plan = std::fs::canonicalize(&options.plan)?;
    let work_dir =
        create_timestamped_subdir(&std::env::temp_dir(), "nockchain-bench-quick-orchestrate")?;
    let work_dir_guard = TempDirGuard::new(work_dir.clone());

    print_heading("Speed-of-Light Quick Orchestrate");
    println!("Plan:       {}", plan.display());
    println!("Fsync:      {}", on_or_off(options.fsync));
    println!(
        "Cold mode:  {}",
        match options.cold_mode {
            ColdMode::Strict => "strict",
            ColdMode::Soft => "soft",
        }
    );
    if let Some(ref out) = options.profile_output {
        println!("Profile output: {}", out.display());
    }
    println!();

    let runner = QuickOrchestrateRunner::new(plan, work_dir, options.fsync, options.cold_mode);
    let results = runner.run().await?;
    results.print_summary();
    let result = finalize_quick_orchestrate_results(&results, options.profile_output.as_deref());

    drop(work_dir_guard);
    result
}

pub async fn cmd_sol_bench(
    benchmark: String,
    plan: Option<PathBuf>,
    fixture: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    kernel: PathBuf,
    start_height: u64,
    end_height: Option<u64>,
    count: Option<u64>,
    peek_mode: PeekMode,
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
    cv_threshold: Option<f64>,
    label: Option<String>,
    docker_image: Option<String>,
    docker_build_tag: Option<String>,
    memory_limit: Option<String>,
    work_dir_mode: Option<BenchWorkDirMode>,
    cpuset: Option<String>,
    cpu_quota: Option<i64>,
    cpu_period: Option<i64>,
    allow_version_skew: bool,
    allow_degraded_cold: bool,
    allow_debug_benchmark: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if benchmark != "sol-orchestrate" {
        return Err(
            format!("trusted SOL benchmark kind must be sol-orchestrate, got {benchmark}").into(),
        );
    }
    if plan.is_none() && fixture.is_none() && checkpoint.is_none() {
        return Err("trusted sol bench requires --plan, --fixture, or --checkpoint".into());
    }
    if plan.is_some() {
        if start_height != 0
            || end_height.is_some()
            || count.is_some()
            || peek_mode != PeekMode::Warm
        {
            return Err("--plan cannot be combined with trusted read shorthand flags".into());
        }
        if blocks != 0 || skip_genesis {
            return Err("--plan cannot be combined with replay shorthand flags".into());
        }
    }
    if checkpoint.is_some() {
        if blocks != 0 || skip_genesis {
            return Err(
                "--checkpoint read shorthand cannot be combined with --blocks or --skip-genesis"
                    .into(),
            );
        }
    }
    if allow_version_skew && docker_image.is_none() && docker_build_tag.is_none() {
        return Err("--allow-version-skew is only valid for Docker trusted runs".into());
    }
    if let Some(plan) = &plan {
        ensure_existing_file(plan, "Plan")?;
    }
    if let Some(fixture) = &fixture {
        ensure_existing_file(fixture, "Fixture")?;
    }
    if let Some(checkpoint) = &checkpoint {
        ensure_existing_file(checkpoint, "Checkpoint")?;
        ensure_existing_file(&kernel, "Kernel")?;
        if end_height.is_some_and(|end_height| end_height < start_height) {
            return Err("--end-height must be greater than or equal to --start-height".into());
        }
        PeekRangeRequest::from_bounds(end_height, count)?;
    }

    let execution = match docker_image_source(docker_image, docker_build_tag)? {
        Some(image) => {
            let memory_limit = memory_limit
                .ok_or("--memory-limit is required when Docker execution is selected")?;
            let work_dir_mode = work_dir_mode
                .ok_or("--work-dir-mode is required when Docker execution is selected")?;
            ExecutionRequest::Docker {
                image,
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

    let mut requested = build_requested_case(
        fixture.clone().unwrap_or_default(),
        execution,
        blocks,
        enable_checkpointing,
        skip_genesis,
        profile_memory,
        profile_interval_ms,
        checkpoint_every_blocks,
        label.or_else(|| {
            output
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        }),
        threads,
        warmup_runs,
        measured_runs,
        cooldown_secs,
    );
    requested.benchmark = benchmark;
    requested.allow_debug_benchmark = allow_debug_benchmark;
    requested.allow_version_skew = allow_version_skew;
    requested.allow_degraded_cold = allow_degraded_cold;
    requested.cv_threshold = cv_threshold;
    if let Some(plan) = plan.clone() {
        requested.orchestrate = RequestedOrchestrate::PlanFile { plan_path: plan };
    } else if let Some(checkpoint) = checkpoint.clone() {
        requested.orchestrate = RequestedOrchestrate::GeneratedRead {
            checkpoint_path: checkpoint,
            kernel_path: kernel.clone(),
            start_height,
            end_height,
            count,
            peek_mode,
        };
    }

    print_heading("Speed-of-Light Trusted Benchmark");
    match &requested.orchestrate {
        RequestedOrchestrate::PlanFile { plan_path } => {
            println!("Plan:    {}", plan_path.display());
        }
        RequestedOrchestrate::GeneratedReplay { fixture_path, .. } => {
            println!("Fixture: {}", fixture_path.display());
        }
        RequestedOrchestrate::GeneratedRead {
            checkpoint_path,
            kernel_path,
            start_height,
            end_height,
            count,
            peek_mode,
        } => {
            println!("Checkpoint: {}", checkpoint_path.display());
            println!("Kernel:     {}", kernel_path.display());
            println!("Read start: {start_height}");
            println!(
                "Read end:   {}",
                end_height
                    .map(|height| height.to_string())
                    .unwrap_or_else(|| "tip".to_string())
            );
            println!(
                "Read count: {}",
                count
                    .map(|count| count.to_string())
                    .unwrap_or_else(|| "to-tip".to_string())
            );
            println!("Peek mode:  {peek_mode:?}");
        }
    }
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

    if let Some((label, unit, throughput)) = run
        .summary
        .pokes_per_second
        .as_ref()
        .map(|stats| ("Poke throughput", "pokes/s", stats))
        .or_else(|| {
            run.summary
                .peeks_per_second
                .as_ref()
                .map(|stats| ("Peek throughput", "peeks/s", stats))
        })
        .or_else(|| {
            run.summary
                .cold_peeks_per_second
                .as_ref()
                .map(|stats| ("Cold peek throughput", "cold peeks/s", stats))
        })
        .or_else(|| {
            run.summary
                .steps_per_second
                .as_ref()
                .map(|stats| ("Step throughput", "steps/s", stats))
        })
    {
        println!(
            "{label} median: {:.2} {unit} (cv {:.3})",
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
    docker_image: Option<String>,
    docker_build_tag: Option<String>,
    memory_limit: String,
    work_dir_mode: BenchWorkDirMode,
    cpuset: Option<String>,
    cpu_quota: Option<i64>,
    cpu_period: Option<i64>,
) -> Result<(), Box<dyn std::error::Error>> {
    ensure_existing_file(&fixture, "Fixture")?;
    let image = docker_image_source(docker_image, docker_build_tag)?
        .ok_or("--docker-image or --docker-build-tag is required for Docker validation")?;

    let requested = build_requested_case(
        fixture.clone(),
        ExecutionRequest::Docker {
            image,
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

fn docker_image_source(
    docker_image: Option<String>,
    docker_build_tag: Option<String>,
) -> Result<Option<DockerImageSource>, Box<dyn std::error::Error>> {
    match (docker_image, docker_build_tag) {
        (Some(reference), None) => Ok(Some(DockerImageSource::Provided { reference })),
        (None, Some(tag)) => Ok(Some(DockerImageSource::AutoBuild { tag })),
        (None, None) => Ok(None),
        (Some(_), Some(_)) => Err(Box::new(HarnessError::InvalidRequestedCase(
            "--docker-image and --docker-build-tag are mutually exclusive".to_string(),
        ))),
    }
}

pub async fn cmd_sol_sweep(
    matrix: PathBuf,
    output: PathBuf,
    interleave: bool,
    randomize_order: bool,
    comparison_markdown: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let matrix_value = serde_json::from_slice::<serde_json::Value>(&std::fs::read(&matrix)?)?;
    let parsed_matrix = parse_matrix_value(matrix_value.clone())?;
    let (schedule_mode, random_seed) = resolve_sweep_schedule(interleave, randomize_order)?;

    print_heading("Speed-of-Light Trusted Sweep");
    println!("Matrix: {}", matrix.display());
    println!("Output: {}", output.display());
    println!("Schedule: {:?}", schedule_mode);
    println!("Comparison markdown: {}", comparison_markdown);
    println!();

    let mut executor = HarnessSweepExecutor;
    let result = execute_sweep(
        &matrix_value,
        parsed_matrix,
        &output,
        &SweepRunOptions {
            schedule_mode,
            random_seed,
            comparison_markdown,
            allow_debug_benchmark: false,
            cpu_profiler: None,
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
        build_mode: CheckpointBuildMode::Derived,
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

// Derived fixture checkpoints use the bench-local replay path.
// Full fixture checkpoints use a bench-local runtime-shaped bootstrap that
// reproduces the checkpoint-relevant startup pokes and born ordering without
// installing live runtime drivers.
/// Build a `.soltest` fixture directly from an input archive and kernel.
pub async fn cmd_sol_fixture_build(
    archive: PathBuf,
    kernel: PathBuf,
    start_height: u64,
    end_height: u64,
    checkpoint_kind: FixtureCheckpointKind,
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
    println!("Checkpoint kind:   {}", checkpoint_kind.as_str());
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
        build_mode: match checkpoint_kind {
            FixtureCheckpointKind::Derived => CheckpointBuildMode::Derived,
            FixtureCheckpointKind::Full => CheckpointBuildMode::Full,
        },
        start_height: Some(SolHeight::ZERO),
        target_height: SolHeight(plan.checkpoint_target_height),
        output_path: checkpoint_output_path.clone(),
        work_dir: checkpoint_work_dir,
    });
    checkpoint_builder.run().await?;

    let embedded_event_num = checkpoint_event_num(&checkpoint_output_path)?;
    let fixture_manifest = SolFixtureManifest {
        source_archive_path: archive.to_string_lossy().to_string(),
        source_archive_event_num: None,
        checkpoint_kind: checkpoint_kind.into(),
        checkpoint_height: SolHeight(plan.checkpoint_target_height),
        checkpoint_event_num: embedded_event_num,
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
    println!("  Checkpoint kind:   {}", checkpoint_kind.as_str());
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
    print!(
        "{}",
        render_fixture_inspect(
            &data.manifest,
            data.checkpoint_bytes.len(),
            data.archive_bytes.len(),
            data.kernel_bytes.len(),
        )
    );

    Ok(())
}

fn render_fixture_inspect(
    manifest: &SolFixtureManifest,
    checkpoint_size_bytes: usize,
    archive_size_bytes: usize,
    kernel_size_bytes: usize,
) -> String {
    let checkpoint_kind = match manifest.checkpoint_kind {
        SolFixtureCheckpointKind::Derived => "derived",
        SolFixtureCheckpointKind::Full => "full",
    };
    let source_archive_event = manifest
        .source_archive_event_num
        .map(|event_num| event_num.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    format!(
        concat!(
            "Source archive path:       {}\n", "Source archive event:      {}\n",
            "Checkpoint kind:           {}\n", "Embedded checkpoint:       {} (event {})\n",
            "Archive range:             {}..={}\n", "Mempool snapshots:         {}\n",
            "Kernel hash:               {}\n", "Checkpoint hash:           {}\n",
            "Archive hash:              {}\n",
            "Embedded sizes:            checkpoint={} bytes, archive={} bytes, kernel={} bytes\n"
        ),
        manifest.source_archive_path,
        source_archive_event,
        checkpoint_kind,
        manifest.checkpoint_height.as_u64(),
        manifest.checkpoint_event_num,
        manifest.archive_start_height.as_u64(),
        manifest.archive_end_height.as_u64(),
        on_or_off(manifest.include_mempool),
        manifest.kernel_hash_hex,
        manifest.checkpoint_hash_hex,
        manifest.archive_hash_hex,
        checkpoint_size_bytes,
        archive_size_bytes,
        kernel_size_bytes,
    )
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
    use nockchain_bench::speed_of_light::peek_bench::ResolvedPeekRange;
    use nockchain_bench::speed_of_light::LatencySummaryUs;
    use tempfile::tempdir;

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

    #[test]
    fn build_quick_read_cpu_profile_command_uses_hidden_quick_read_once() {
        let command = build_quick_read_cpu_profile_command(
            Path::new("/tmp/nockchain-bench"),
            Path::new("/tmp/0.chkjam"),
            Path::new("/tmp/dumb.jam"),
            11,
            42,
            false,
            #[cfg(feature = "pma-runtime-compat")]
            Some(true),
            #[cfg(not(feature = "pma-runtime-compat"))]
            None,
        );

        let expected = vec![
            "/tmp/nockchain-bench", "sol", "quick-read-once", "--checkpoint", "/tmp/0.chkjam",
            "--kernel", "/tmp/dumb.jam", "--start-height", "11", "--end-height", "42",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        #[cfg(feature = "pma-runtime-compat")]
        let expected = {
            let mut expected = expected;
            expected.extend(["--fsync".to_string(), "on".to_string()]);
            expected
        };

        assert_eq!(command, expected);
    }

    #[test]
    fn quick_read_profile_output_payload_uses_read_metric_names() {
        let payload = build_quick_read_profile_output_payload(
            Path::new("/tmp/0.chkjam"),
            Path::new("/tmp/dumb.jam"),
            &PeekBenchResults {
                range: ResolvedPeekRange {
                    start_height: 11,
                    end_height: 42,
                    tip_height: 100,
                },
                peeks_attempted: 32,
                success_peeks: 30,
                missing_peeks: 1,
                error_peeks: 1,
                init_time_secs: 1.0,
                total_peek_time_secs: 2.0,
                peeks_per_second: 16.0,
                avg_latency_us: Some(20_000),
                latency_summary_us: Some(LatencySummaryUs {
                    min: 10_000,
                    p50: 20_000,
                    p95: 30_000,
                    p99: 35_000,
                    max: 40_000,
                }),
                memory_summary: None,
            },
        );

        assert_eq!(payload["peeks_attempted"], serde_json::json!(32));
        assert_eq!(payload["success_peeks"], serde_json::json!(30));
        assert_eq!(payload["missing_peeks"], serde_json::json!(1));
        assert_eq!(payload["error_peeks"], serde_json::json!(1));
        assert_eq!(payload["failed_peeks"], serde_json::json!(2));
        assert!(payload.get("blocks_poked").is_none());
        assert!(payload.get("failed_pokes").is_none());
    }

    #[test]
    fn quick_read_profile_command_preserves_dry_run_when_requested() {
        let command = build_quick_read_cpu_profile_command(
            Path::new("/tmp/nockchain-bench"),
            Path::new("/tmp/0.chkjam"),
            Path::new("/tmp/dumb.jam"),
            11,
            42,
            true,
            #[cfg(feature = "pma-runtime-compat")]
            Some(true),
            #[cfg(not(feature = "pma-runtime-compat"))]
            None,
        );

        assert!(command.iter().any(|arg| arg == "--dry-run"));
    }

    #[test]
    fn quick_orchestrate_profile_output_writer_uses_compact_json() {
        let temp_dir = tempdir().expect("temp dir");
        let output = temp_dir.path().join("quick-orchestrate.json");
        let payload = "{\"boot\":{\"checkpoint\":\"/tmp/0.chkjam\",\"kernel\":\"/tmp/dumb.jam\",\"fsync\":\"on\",\"init_time_secs\":1.0},\"steps\":[{\"label\":\"cold-prep\",\"type\":\"force_cold\",\"outcome\":\"ok\",\"duration_ms\":0.5,\"cold_verified\":false,\"degraded_reason\":\"macos_unsupported\"},{\"label\":\"peek-one\",\"type\":\"peek_height_cold\",\"height\":7,\"outcome\":\"success\",\"duration_ms\":1.5,\"minflt_delta\":12,\"majflt_delta\":0,\"cold_verified\":true,\"residency_pages_after\":0,\"residency_total_pages\":128,\"cold_attempts\":1}]}";

        write_profile_output(&output, payload.as_bytes()).expect("write profile output");

        let written = std::fs::read_to_string(&output).expect("read profile output");
        assert_eq!(written, payload);
        assert!(!written.contains('\n'));
    }

    #[test]
    fn quick_read_profile_output_writer_preserves_pretty_json() {
        let temp_dir = tempdir().expect("temp dir");
        let output = temp_dir.path().join("quick-read.json");
        let payload = serde_json::to_vec_pretty(&serde_json::json!({
            "boot": {
                "checkpoint": "/tmp/0.chkjam",
                "kernel": "/tmp/dumb.jam"
            },
            "peeks_attempted": 1
        }))
        .expect("pretty payload");

        write_profile_output(&output, &payload).expect("write profile output");

        let written = std::fs::read_to_string(&output).expect("read profile output");
        assert!(written.contains("\n  \"boot\""));
        assert!(written.contains("\n  \"peeks_attempted\""));
    }

    #[test]
    fn quick_orchestrate_step_failure_writes_profile_output_before_erroring() {
        let temp_dir = tempdir().expect("temp dir");
        let output = temp_dir.path().join("quick-orchestrate.json");
        let payload = "{\"boot\":{\"checkpoint\":\"/tmp/0.chkjam\",\"kernel\":\"/tmp/dumb.jam\",\"fsync\":\"on\",\"init_time_secs\":1.0},\"steps\":[{\"label\":\"peek-one\",\"type\":\"peek_height\",\"height\":7,\"outcome\":\"success\",\"duration_ms\":1.5},{\"label\":\"cold-prep\",\"type\":\"force_cold\",\"outcome\":\"error\",\"duration_ms\":0.2,\"error\":\"missing block\"}]}";

        let error = finalize_quick_orchestrate_output(
            Some(output.as_path()),
            payload,
            Some("missing block"),
        )
        .expect_err("step failure should bubble a command error");
        let written = std::fs::read_to_string(&output).expect("read profile output");

        assert_eq!(written, payload);
        assert!(error.to_string().contains("missing block"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn quick_orchestrate_pre_run_failures_do_not_write_profile_output() {
        let temp_dir = tempdir().expect("temp dir");
        let output = temp_dir.path().join("quick-orchestrate.json");

        let error = cmd_sol_quick_orchestrate(QuickOrchestrateOptions {
            plan: temp_dir.path().join("missing-plan.json"),
            profile_output: Some(output.clone()),
            fsync: true,
            cold_mode: ColdMode::Strict,
        })
        .await
        .expect_err("missing plan should fail before boot");

        assert!(!output.exists());
        assert!(error.to_string().contains("Plan"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn quick_orchestrate_malformed_plan_does_not_write_profile_output() {
        let temp_dir = tempdir().expect("temp dir");
        let plan = temp_dir.path().join("plan.json");
        let output = temp_dir.path().join("quick-orchestrate.json");
        std::fs::write(&plan, "{ not valid json").expect("write malformed plan");

        let error = cmd_sol_quick_orchestrate(QuickOrchestrateOptions {
            plan,
            profile_output: Some(output.clone()),
            fsync: true,
            cold_mode: ColdMode::Strict,
        })
        .await
        .expect_err("malformed plan should fail before boot");

        assert!(!output.exists());
        assert!(error
            .to_string()
            .contains("failed to parse quick-orchestrate plan"));
    }

    #[cfg(not(feature = "pma-runtime-compat"))]
    #[test]
    fn quick_read_profile_command_accepts_fsync_without_emitting_it_when_pma_is_disabled() {
        let command = build_quick_read_cpu_profile_command(
            Path::new("/tmp/nockchain-bench"),
            Path::new("/tmp/0.chkjam"),
            Path::new("/tmp/dumb.jam"),
            11,
            42,
            false,
            None,
        );

        assert!(!command.iter().any(|arg| arg == "--fsync"));
    }

    #[cfg(feature = "pma-runtime-compat")]
    #[test]
    fn quick_read_profile_command_forwards_fsync_under_pma() {
        let off_command = build_quick_read_cpu_profile_command(
            Path::new("/tmp/nockchain-bench"),
            Path::new("/tmp/0.chkjam"),
            Path::new("/tmp/dumb.jam"),
            11,
            42,
            false,
            Some(false),
        );
        let on_command = build_quick_read_cpu_profile_command(
            Path::new("/tmp/nockchain-bench"),
            Path::new("/tmp/0.chkjam"),
            Path::new("/tmp/dumb.jam"),
            11,
            42,
            false,
            Some(true),
        );

        assert!(off_command
            .windows(2)
            .any(|args| args == ["--fsync", "off"]));
        assert!(on_command.windows(2).any(|args| args == ["--fsync", "on"]));
    }

    #[test]
    fn fixture_inspect_renders_unknown_source_archive_event() {
        let rendered = render_fixture_inspect(
            &SolFixtureManifest {
                source_archive_path: "archive.solarch".to_string(),
                source_archive_event_num: None,
                checkpoint_kind: SolFixtureCheckpointKind::Full,
                checkpoint_height: SolHeight(10),
                checkpoint_event_num: 14,
                archive_start_height: SolHeight(11),
                archive_end_height: SolHeight(12),
                include_mempool: false,
                chunk_size: 1024,
                kernel_hash_hex: "k".repeat(64),
                checkpoint_hash_hex: "c".repeat(64),
                archive_hash_hex: "a".repeat(64),
            },
            1,
            2,
            3,
        );

        assert!(rendered.contains("Source archive event:      unknown"));
        assert!(rendered.contains("Checkpoint kind:           full"));
    }
}
