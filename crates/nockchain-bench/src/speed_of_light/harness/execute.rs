use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::artifacts::write_run_artifacts;
use super::case::{ExecutionConfig, ResolvedCase};
use super::{create_temp_dir, CpuProfilerKind, HarnessError};
use crate::speed_of_light::bench::{SolBenchConfig, SolBenchResults, SolBenchRunner};
use crate::speed_of_light::fixture::extract_fixture_to_paths;
use crate::speed_of_light::profiling::MemoryProfile;

#[derive(Debug, Clone, PartialEq)]
pub struct ExecuteOptions {
    pub checkpoint_recovery_timeout_ms: u64,
    pub checkpoint_recovery_tolerance_pct: f64,
    pub gc_drop_threshold_mib: u64,
    pub page_fault_minor_burst_threshold: u64,
    pub page_fault_major_burst_threshold: u64,
}

impl Default for ExecuteOptions {
    fn default() -> Self {
        Self::from(&ExecutionConfig::default())
    }
}

impl From<&ExecutionConfig> for ExecuteOptions {
    fn from(value: &ExecutionConfig) -> Self {
        Self {
            checkpoint_recovery_timeout_ms: value.checkpoint_recovery_timeout_ms,
            checkpoint_recovery_tolerance_pct: value.checkpoint_recovery_tolerance_pct_bps as f64
                / 100.0,
            gc_drop_threshold_mib: value.gc_drop_threshold_mib,
            page_fault_minor_burst_threshold: value.page_fault_minor_burst_threshold,
            page_fault_major_burst_threshold: value.page_fault_major_burst_threshold,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockTimingRecord {
    pub height: u64,
    pub duration_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: String,
    pub success: bool,
    pub error: Option<String>,
    pub blocks_poked: u64,
    pub failed_pokes: u64,
    pub init_time_secs: f64,
    pub total_replay_time_secs: f64,
    pub throughput_blocks_per_second: f64,
    pub average_block_time_ms: f64,
    pub checkpoint_count: u64,
    pub checkpoint_total_time_secs: f64,
    pub average_checkpoint_time_secs: f64,
    pub peak_process_rss_bytes: Option<f64>,
    pub minor_faults_total: Option<f64>,
    pub major_faults_total: Option<f64>,
}

pub struct CompletedRun {
    pub record: RunRecord,
    pub block_timings: Vec<BlockTimingRecord>,
    pub profile: Option<MemoryProfile>,
    pub bench_results: Option<SolBenchResults>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuProfileExecutionKind {
    Native,
    DockerInContainer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuProfileArtifact {
    pub profiler_kind: CpuProfilerKind,
    pub sample_rate_hz: u32,
    pub execution_kind: CpuProfileExecutionKind,
    pub profiled_command: Vec<String>,
    pub output_relative_path: PathBuf,
    pub symbol_dir_relative_path: PathBuf,
    pub symbol_binary_relative_path: PathBuf,
}

pub fn cpu_profile_output_relative_path(profiler_kind: CpuProfilerKind) -> PathBuf {
    match profiler_kind {
        CpuProfilerKind::Samply => PathBuf::from("profiles/samply-profile.json.gz"),
    }
}

pub async fn execute_once(
    resolved: &ResolvedCase,
    run_id: &str,
    run_dir: &Path,
) -> Result<CompletedRun, HarnessError> {
    execute_once_with_options(
        resolved,
        run_id,
        run_dir,
        &ExecuteOptions::from(&resolved.execution_config),
    )
    .await
}

pub async fn execute_once_with_options(
    resolved: &ResolvedCase,
    run_id: &str,
    run_dir: &Path,
    options: &ExecuteOptions,
) -> Result<CompletedRun, HarnessError> {
    let run = match run_benchmark_once(resolved, options).await {
        Ok(results) => completed_run_from_results(run_id, results),
        Err(error) => CompletedRun {
            record: RunRecord {
                run_id: run_id.to_string(),
                success: false,
                error: Some(error.to_string()),
                blocks_poked: 0,
                failed_pokes: 0,
                init_time_secs: 0.0,
                total_replay_time_secs: 0.0,
                throughput_blocks_per_second: 0.0,
                average_block_time_ms: 0.0,
                checkpoint_count: 0,
                checkpoint_total_time_secs: 0.0,
                average_checkpoint_time_secs: 0.0,
                peak_process_rss_bytes: None,
                minor_faults_total: None,
                major_faults_total: None,
            },
            block_timings: Vec::new(),
            profile: None,
            bench_results: None,
        },
    };

    write_run_artifacts(run_dir, &run)?;
    Ok(run)
}

async fn run_benchmark_once(
    resolved: &ResolvedCase,
    options: &ExecuteOptions,
) -> Result<SolBenchResults, HarnessError> {
    struct TempDirGuard {
        path: PathBuf,
    }

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    let temp_dir = create_temp_dir("nockchain-bench-harness")?;
    let _temp_dir_guard = TempDirGuard {
        path: temp_dir.clone(),
    };

    let checkpoint_path = temp_dir.join("fixture.chkjam");
    let archive_path = temp_dir.join("fixture.solarch");
    let kernel_path = temp_dir.join("fixture.jam");
    let work_dir = temp_dir.join("checkpoint-work");
    std::fs::create_dir_all(&work_dir)?;

    extract_fixture_to_paths(
        &resolved.absolute_fixture_path, &checkpoint_path, &archive_path, &kernel_path,
    )?;

    let config = SolBenchConfig {
        archive_path: archive_path.to_string_lossy().to_string(),
        kernel_path: kernel_path.to_string_lossy().to_string(),
        block_count: resolved.requested.blocks,
        skip_genesis: resolved.requested.skip_genesis,
        proof_version: None,
        checkpoint_path: Some(checkpoint_path.to_string_lossy().to_string()),
        start_height: Some(resolved.fixture_manifest.archive_start_height),
        enable_checkpointing: resolved.requested.enable_checkpointing,
        fsync: {
            #[cfg(feature = "pma-runtime-compat")]
            {
                resolved.requested.fsync
            }
            #[cfg(not(feature = "pma-runtime-compat"))]
            {
                true
            }
        },
        profile_memory: resolved.requested.profile_memory,
        profile_interval_ms: resolved.requested.profile_interval_ms,
        gc_drop_threshold_bytes: options.gc_drop_threshold_mib.saturating_mul(1024 * 1024),
        page_fault_minor_burst_threshold: options.page_fault_minor_burst_threshold,
        page_fault_major_burst_threshold: options.page_fault_major_burst_threshold,
        checkpoint_every_blocks: resolved.requested.checkpoint_every_blocks,
        checkpoint_recovery_timeout_ms: options.checkpoint_recovery_timeout_ms,
        checkpoint_recovery_tolerance_pct: options.checkpoint_recovery_tolerance_pct,
        work_dir,
    };

    let mut runner = SolBenchRunner::new(config);
    Ok(runner.run().await?)
}

fn completed_run_from_results(run_id: &str, results: SolBenchResults) -> CompletedRun {
    let block_timings = results
        .block_timings
        .iter()
        .map(|(height, duration)| BlockTimingRecord {
            height: height.as_u64(),
            duration_ms: duration.as_secs_f64() * 1000.0,
        })
        .collect();
    let profile = results.memory_profile.clone();

    CompletedRun {
        record: RunRecord {
            run_id: run_id.to_string(),
            success: true,
            error: None,
            blocks_poked: results.blocks_poked,
            failed_pokes: results.failed_pokes,
            init_time_secs: results.init_time.as_secs_f64(),
            total_replay_time_secs: results.total_poke_time.as_secs_f64(),
            throughput_blocks_per_second: results.blocks_per_second(),
            average_block_time_ms: results.avg_block_time().as_secs_f64() * 1000.0,
            checkpoint_count: results.checkpoint_count,
            checkpoint_total_time_secs: results.checkpoint_total_time.as_secs_f64(),
            average_checkpoint_time_secs: results
                .avg_checkpoint_time()
                .map(|duration| duration.as_secs_f64())
                .unwrap_or(0.0),
            peak_process_rss_bytes: profile.as_ref().and_then(|profile| {
                profile
                    .samples
                    .iter()
                    .map(|sample| sample.vm_rss_kb.saturating_mul(1024) as f64)
                    .max_by(|left, right| left.total_cmp(right))
            }),
            minor_faults_total: profile.as_ref().and_then(total_minor_faults),
            major_faults_total: profile.as_ref().and_then(total_major_faults),
        },
        block_timings,
        profile,
        bench_results: Some(results),
    }
}

fn total_minor_faults(profile: &MemoryProfile) -> Option<f64> {
    let first = profile.samples.first()?;
    let last = profile.samples.last()?;
    Some(last.minor_faults.saturating_sub(first.minor_faults) as f64)
}

fn total_major_faults(profile: &MemoryProfile) -> Option<f64> {
    let first = profile.samples.first()?;
    let last = profile.samples.last()?;
    Some(last.major_faults.saturating_sub(first.major_faults) as f64)
}
