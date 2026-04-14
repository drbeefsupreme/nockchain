use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nockapp::nockapp::save::SaveableCheckpoint;
use nockapp::nockapp::NockApp;
use nockapp::noun::slab::NounSlab;
use nockapp::utils::make_tas;
use nockvm::noun::{D, T};
use serde::Serialize;
use thiserror::Error;

use super::checkpoint::{load_checkpoint, CheckpointLoadError};
use super::harness::DEFAULT_FSYNC_ENABLED;
use super::kernel_utils::{
    init_nockapp, peek_heaviest_chain_or_block, KernelInitError, PeekChainError,
};
use crate::sampler::smaps::SmapsParser;

const PROGRESS_HEIGHT_INTERVAL: u64 = 100;
const PROGRESS_TIME_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Debug, Error)]
pub enum PeekBenchError {
    #[error("start height {start_height} is past tip height {tip_height}")]
    StartHeightPastTip { start_height: u64, tip_height: u64 },

    #[error("end height {end_height} is past tip height {tip_height}")]
    EndHeightPastTip { end_height: u64, tip_height: u64 },

    #[error("count {count} from start height {start_height} runs past tip height {tip_height}")]
    CountPastTip {
        start_height: u64,
        count: u64,
        tip_height: u64,
    },

    #[error("count {count} from start height {start_height} overflows u64 height space")]
    CountOverflows { start_height: u64, count: u64 },

    #[error("end height {end_height} must be >= start height {start_height}")]
    EndBeforeStart { start_height: u64, end_height: u64 },

    #[error("end height and count are mutually exclusive")]
    ConflictingBounds,

    #[error("count must be at least 1")]
    InvalidCountZero,

    #[error("failed to load checkpoint: {0}")]
    CheckpointLoad(#[from] CheckpointLoadError),

    #[error("failed to initialize checkpoint-backed kernel: {0}")]
    KernelInit(#[from] KernelInitError),

    #[error("failed to resolve heaviest chain: {0}")]
    PeekChain(#[from] PeekChainError),

    #[error("heaviest chain tip is unavailable after boot")]
    HeaviestChainUnavailable,

    #[error("memory sampling failed: {0}")]
    MemorySample(String),
}

#[derive(Debug, Clone)]
pub struct PeekBenchConfig {
    pub checkpoint_path: PathBuf,
    pub kernel_path: PathBuf,
    pub start_height: u64,
    pub range: PeekRangeRequest,
    pub fsync: bool,
    pub dry_run: bool,
    pub profile_memory: bool,
    pub profile_interval_ms: u64,
    pub work_dir: PathBuf,
}

impl Default for PeekBenchConfig {
    fn default() -> Self {
        Self {
            checkpoint_path: PathBuf::from("0.chkjam"),
            kernel_path: PathBuf::from("assets/dumb.jam"),
            start_height: 0,
            range: PeekRangeRequest::ToTip,
            fsync: DEFAULT_FSYNC_ENABLED,
            dry_run: false,
            profile_memory: false,
            profile_interval_ms: 500,
            work_dir: PathBuf::from("."),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeekRangeRequest {
    ToTip,
    EndHeight(u64),
    Count(u64),
}

impl PeekRangeRequest {
    pub fn from_bounds(
        end_height: Option<u64>,
        count: Option<u64>,
    ) -> Result<Self, PeekBenchError> {
        match (end_height, count) {
            (Some(end_height), None) => Ok(Self::EndHeight(end_height)),
            (None, Some(0)) => Err(PeekBenchError::InvalidCountZero),
            (None, Some(count)) => Ok(Self::Count(count)),
            (None, None) => Ok(Self::ToTip),
            (Some(_), Some(_)) => Err(PeekBenchError::ConflictingBounds),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedPeekRange {
    pub start_height: u64,
    pub end_height: u64,
    pub tip_height: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LatencySummaryUs {
    pub min: u64,
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
    pub max: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PeekBenchResults {
    pub range: ResolvedPeekRange,
    pub peeks_attempted: u64,
    pub success_peeks: u64,
    pub missing_peeks: u64,
    pub error_peeks: u64,
    pub init_time_secs: f64,
    pub total_peek_time_secs: f64,
    pub peeks_per_second: f64,
    pub avg_latency_us: Option<u64>,
    pub latency_summary_us: Option<LatencySummaryUs>,
    pub memory_summary: Option<PeekMemorySummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PeekMemorySummary {
    pub setup_peak_rss_bytes: u64,
    pub measurement_start_rss_bytes: u64,
    pub measurement_end_rss_bytes: u64,
    pub measurement_peak_rss_bytes: u64,
    pub measurement_p95_rss_bytes: u64,
    pub measurement_minor_faults_delta: Option<u64>,
    pub measurement_major_faults_delta: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct ReadMemorySample {
    timestamp_ms: u64,
    rss_bytes: u64,
    minor_faults: Option<u64>,
    major_faults: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct DryRunProfileOutput<'a> {
    dry_run: bool,
    checkpoint_path: &'a str,
    kernel_path: &'a str,
    resolved_start_height: u64,
    resolved_end_height: u64,
    resolved_peek_count: u64,
    tip_height: u64,
    init_time_secs: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_summary: Option<SetupMemorySummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct NormalProfileOutput<'a> {
    dry_run: bool,
    checkpoint_path: &'a str,
    kernel_path: &'a str,
    start_height: u64,
    end_height: u64,
    tip_height: u64,
    peeks_attempted: u64,
    success_peeks: u64,
    missing_peeks: u64,
    error_peeks: u64,
    failed_peeks: u64,
    init_time_secs: f64,
    total_peek_time_secs: f64,
    peeks_per_second: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_latency_us: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    latency_summary_us: Option<&'a LatencySummaryUs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_summary: Option<&'a PeekMemorySummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct SetupMemorySummary {
    setup_peak_rss_bytes: u64,
}

impl From<&PeekMemorySummary> for SetupMemorySummary {
    fn from(value: &PeekMemorySummary) -> Self {
        Self {
            setup_peak_rss_bytes: value.setup_peak_rss_bytes,
        }
    }
}

struct ReadMemorySampler {
    pid: i32,
    samples: Arc<Mutex<Vec<ReadMemorySample>>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Result<(), String>>>,
}

impl ReadMemorySampler {
    fn start(started_at: Instant, interval_ms: u64) -> Result<Self, PeekBenchError> {
        let pid = std::process::id() as i32;
        let samples = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_samples = Arc::clone(&samples);
        let thread_stop = Arc::clone(&stop);
        let sleep_interval = Duration::from_millis(interval_ms.max(1));
        let handle = std::thread::spawn(move || -> Result<(), String> {
            loop {
                let timestamp_ms = elapsed_ms_since(started_at);
                push_background_sample(&thread_samples, read_memory_sample(pid, timestamp_ms))
                    .map_err(|err| err.to_string())?;

                if thread_stop.load(Ordering::Relaxed) {
                    break;
                }

                std::thread::sleep(sleep_interval);
            }

            Ok(())
        });

        let sampler = Self {
            pid,
            samples,
            stop,
            handle: Some(handle),
        };
        sampler.sample_now(0)?;
        Ok(sampler)
    }

    fn sample_now(&self, timestamp_ms: u64) -> Result<(), PeekBenchError> {
        let sample = read_memory_sample(self.pid, timestamp_ms)?;
        if let Some(sample) = sample {
            push_memory_sample(&self.samples, sample)?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<ReadMemorySample>, PeekBenchError> {
        self.stop.store(true, Ordering::Relaxed);

        if let Some(handle) = self.handle.take() {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => return Err(PeekBenchError::MemorySample(err)),
                Err(_) => {
                    return Err(PeekBenchError::MemorySample(
                        "memory sampler thread panicked".to_string(),
                    ));
                }
            }
        }

        let mut samples = self
            .samples
            .lock()
            .map_err(|_| PeekBenchError::MemorySample("memory sampler mutex poisoned".to_string()))?
            .clone();
        samples.sort_unstable_by_key(|sample| sample.timestamp_ms);
        Ok(samples)
    }
}

impl Drop for ReadMemorySampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

pub struct PeekBenchRunner {
    config: PeekBenchConfig,
}

impl PeekBenchResults {
    pub fn is_dry_run(&self) -> bool {
        self.peeks_attempted == 0
    }

    pub fn print_summary(&self) {
        if self.is_dry_run() {
            println!("Dry run:         yes");
            println!("Init time:       {:.2}s", self.init_time_secs);
            if let Some(memory_summary) = &self.memory_summary {
                println!(
                    "Setup peak RSS:  {} MiB",
                    bytes_to_mib(memory_summary.setup_peak_rss_bytes)
                );
            }
            return;
        }

        println!("Peeks attempted: {}", self.peeks_attempted);
        println!("Success peeks:   {}", self.success_peeks);
        println!("Missing peeks:   {}", self.missing_peeks);
        println!("Error peeks:     {}", self.error_peeks);
        println!("Init time:       {:.2}s", self.init_time_secs);
        println!("Total peek time: {:.2}s", self.total_peek_time_secs);

        match (self.avg_latency_us, self.latency_summary_us) {
            (Some(avg), Some(summary)) => {
                println!(
                    "Latency:         avg {:.2} ms, min {:.2} ms, p50 {:.2} ms, p95 {:.2} ms, p99 {:.2} ms, max {:.2} ms",
                    micros_to_ms(avg),
                    micros_to_ms(summary.min),
                    micros_to_ms(summary.p50),
                    micros_to_ms(summary.p95),
                    micros_to_ms(summary.p99),
                    micros_to_ms(summary.max),
                );
            }
            _ => println!("Latency:         unavailable"),
        }

        println!("Throughput:      {:.2} peeks/s", self.peeks_per_second);

        if let Some(memory_summary) = &self.memory_summary {
            println!(
                "Setup peak RSS:  {} MiB",
                bytes_to_mib(memory_summary.setup_peak_rss_bytes)
            );
            println!(
                "Measure RSS:     start {} MiB, end {} MiB, peak {} MiB, p95 {} MiB",
                bytes_to_mib(memory_summary.measurement_start_rss_bytes),
                bytes_to_mib(memory_summary.measurement_end_rss_bytes),
                bytes_to_mib(memory_summary.measurement_peak_rss_bytes),
                bytes_to_mib(memory_summary.measurement_p95_rss_bytes),
            );
            println!(
                "Fault deltas:    minor {}, major {}",
                display_optional_counter(memory_summary.measurement_minor_faults_delta),
                display_optional_counter(memory_summary.measurement_major_faults_delta),
            );
        }
    }

    pub fn profile_output_value(
        &self,
        checkpoint_path: &Path,
        kernel_path: &Path,
    ) -> serde_json::Value {
        let checkpoint_path = checkpoint_path.to_string_lossy();
        let kernel_path = kernel_path.to_string_lossy();

        if self.is_dry_run() {
            return build_dry_run_profile_output(
                &checkpoint_path,
                &kernel_path,
                self.range,
                self.init_time_secs,
                self.memory_summary.as_ref().map(SetupMemorySummary::from),
            );
        }

        build_normal_profile_output(&checkpoint_path, &kernel_path, self)
    }
}

impl PeekBenchRunner {
    pub fn new(config: PeekBenchConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &PeekBenchConfig {
        &self.config
    }

    pub async fn run(&mut self) -> Result<PeekBenchResults, PeekBenchError> {
        let run_started_at = Instant::now();
        let memory_sampler = if self.config.profile_memory {
            Some(ReadMemorySampler::start(
                run_started_at, self.config.profile_interval_ms,
            )?)
        } else {
            None
        };

        let checkpoint = load_checkpoint(&self.config.checkpoint_path)?;
        let checkpoint = SaveableCheckpoint {
            ker_hash: checkpoint.ker_hash,
            event_num: checkpoint.event_num,
            state: checkpoint.state,
            cold: checkpoint.cold,
        };

        let mut nockapp = init_nockapp(
            Path::new(&self.config.kernel_path),
            Some(checkpoint),
            &self.config.work_dir,
            false,
            self.config.fsync,
        )
        .await?;

        let tip = peek_heaviest_chain_or_block(&mut nockapp)
            .await?
            .ok_or(PeekBenchError::HeaviestChainUnavailable)?;
        let tip_height = tip.0 .0 .0;
        let resolved = resolve_range(self.config.start_height, self.config.range, tip_height)?;

        let init_time_secs = run_started_at.elapsed().as_secs_f64();
        let setup_end_ms = elapsed_ms_since(run_started_at);

        if let Some(sampler) = memory_sampler.as_ref() {
            handle_boundary_memory_sample_result(sampler.sample_now(setup_end_ms), "setup end")?;
        }

        println!(
            "Resolved quick-read range: {}..={} (tip {})",
            resolved.start_height, resolved.end_height, resolved.tip_height
        );

        if self.config.dry_run {
            println!("Dry run requested; setup completed without executing peeks.");
            let memory_summary = finish_setup_only_memory_sampling(memory_sampler, setup_end_ms)?;

            return Ok(PeekBenchResults {
                range: resolved,
                peeks_attempted: 0,
                success_peeks: 0,
                missing_peeks: 0,
                error_peeks: 0,
                init_time_secs,
                total_peek_time_secs: 0.0,
                peeks_per_second: 0.0,
                avg_latency_us: None,
                latency_summary_us: None,
                memory_summary,
            });
        }

        let measurement_start_ms = elapsed_ms_since(run_started_at);
        if let Some(sampler) = memory_sampler.as_ref() {
            handle_boundary_memory_sample_result(
                sampler.sample_now(measurement_start_ms),
                "measurement start",
            )?;
        }

        let total_peeks = peek_count(resolved);
        let peek_started_at = Instant::now();
        let mut last_progress_at = Instant::now();
        let mut peek_samples = Vec::with_capacity(total_peeks as usize);
        let mut success_peeks = 0u64;
        let mut missing_peeks = 0u64;
        let mut error_peeks = 0u64;

        for height in resolved.start_height..=resolved.end_height {
            let sample = peek_height(&mut nockapp, height).await;
            match sample.kind {
                PeekSampleKind::Success => success_peeks += 1,
                PeekSampleKind::Missing => missing_peeks += 1,
                PeekSampleKind::Error => error_peeks += 1,
            }
            peek_samples.push(sample);

            let peeks_attempted = success_peeks + missing_peeks + error_peeks;
            if should_print_progress(peeks_attempted, total_peeks, last_progress_at.elapsed()) {
                println!(
                    "Peek progress: {}/{} heights through {} (success {}, missing {}, error {})",
                    peeks_attempted, total_peeks, height, success_peeks, missing_peeks, error_peeks
                );
                last_progress_at = Instant::now();
            }
        }

        let total_peek_time_secs = peek_started_at.elapsed().as_secs_f64();
        let measurement_end_ms = elapsed_ms_since(run_started_at);
        if let Some(sampler) = memory_sampler.as_ref() {
            handle_boundary_memory_sample_result(
                sampler.sample_now(measurement_end_ms),
                "measurement end",
            )?;
        }

        let memory_summary = finish_measurement_memory_sampling(
            memory_sampler, setup_end_ms, measurement_start_ms, measurement_end_ms,
        )?;
        let peeks_attempted = success_peeks + missing_peeks + error_peeks;
        let avg_latency_us = average_latency_us(&peek_samples);
        let latency_summary_us = summarize_latency_us(&peek_samples);
        let peeks_per_second = if total_peek_time_secs > 0.0 {
            peeks_attempted as f64 / total_peek_time_secs
        } else {
            0.0
        };

        Ok(PeekBenchResults {
            range: resolved,
            peeks_attempted,
            success_peeks,
            missing_peeks,
            error_peeks,
            init_time_secs,
            total_peek_time_secs,
            peeks_per_second,
            avg_latency_us,
            latency_summary_us,
            memory_summary,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeekSampleKind {
    Success,
    Missing,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PeekSample {
    latency_us: u64,
    pub(crate) kind: PeekSampleKind,
}

impl PeekSample {
    pub(crate) fn latency_us(&self) -> u64 {
        self.latency_us
    }
}

pub(crate) async fn peek_height_result(
    nockapp: &mut NockApp,
    height: u64,
) -> Result<PeekSample, nockapp::nockapp::NockAppError> {
    let mut slab = NounSlab::new();
    let tag = make_tas(&mut slab, "heavy-n").as_noun();
    let request = T(&mut slab, &[tag, D(height), D(0)]);
    slab.set_root(request);

    let started_at = Instant::now();
    let result = nockapp.peek_handle(slab).await?;
    let latency_us = duration_to_micros(started_at.elapsed());

    let kind = match result {
        Some(response) => {
            drop(response);
            PeekSampleKind::Success
        }
        None => PeekSampleKind::Missing,
    };

    Ok(PeekSample { latency_us, kind })
}

async fn peek_height(nockapp: &mut NockApp, height: u64) -> PeekSample {
    match peek_height_result(nockapp, height).await {
        Ok(sample) => sample,
        Err(_error) => PeekSample {
            latency_us: 0,
            kind: PeekSampleKind::Error,
        },
    }
}

fn resolve_range(
    start_height: u64,
    range: PeekRangeRequest,
    tip_height: u64,
) -> Result<ResolvedPeekRange, PeekBenchError> {
    if start_height > tip_height {
        return Err(PeekBenchError::StartHeightPastTip {
            start_height,
            tip_height,
        });
    }

    let end_height = match range {
        PeekRangeRequest::EndHeight(end_height) => {
            if end_height < start_height {
                return Err(PeekBenchError::EndBeforeStart {
                    start_height,
                    end_height,
                });
            }
            if end_height > tip_height {
                return Err(PeekBenchError::EndHeightPastTip {
                    end_height,
                    tip_height,
                });
            }
            end_height
        }
        PeekRangeRequest::Count(0) => return Err(PeekBenchError::InvalidCountZero),
        PeekRangeRequest::Count(count) => {
            let resolved_end =
                start_height
                    .checked_add(count - 1)
                    .ok_or(PeekBenchError::CountOverflows {
                        start_height,
                        count,
                    })?;
            if resolved_end > tip_height {
                return Err(PeekBenchError::CountPastTip {
                    start_height,
                    count,
                    tip_height,
                });
            }
            resolved_end
        }
        PeekRangeRequest::ToTip => tip_height,
    };

    Ok(ResolvedPeekRange {
        start_height,
        end_height,
        tip_height,
    })
}

fn summarize_latency_us(samples: &[PeekSample]) -> Option<LatencySummaryUs> {
    let values = non_error_latencies_us(samples);
    if values.is_empty() {
        return None;
    }

    Some(LatencySummaryUs {
        min: *values.iter().min()?,
        p50: percentile_u64(&values, 0.50)?,
        p95: percentile_u64(&values, 0.95)?,
        p99: percentile_u64(&values, 0.99)?,
        max: *values.iter().max()?,
    })
}

fn average_latency_us(samples: &[PeekSample]) -> Option<u64> {
    let values = non_error_latencies_us(samples);
    if values.is_empty() {
        return None;
    }

    let total: u128 = values.iter().map(|value| *value as u128).sum();
    let count = values.len() as u128;
    Some(((total + (count / 2)) / count) as u64)
}

fn non_error_latencies_us(samples: &[PeekSample]) -> Vec<u64> {
    samples
        .iter()
        .filter(|sample| !matches!(sample.kind, PeekSampleKind::Error))
        .map(|sample| sample.latency_us)
        .collect()
}

fn percentile_u64(values: &[u64], p: f64) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f64 * p.clamp(0.0, 1.0)).round() as usize;
    sorted.get(idx).copied()
}

fn build_memory_summary(
    setup_samples: &[ReadMemorySample],
    measurement_samples: &[ReadMemorySample],
) -> Option<PeekMemorySummary> {
    let setup_peak_rss_bytes = setup_samples.iter().map(|sample| sample.rss_bytes).max()?;
    let measurement_start = measurement_samples.first()?;
    let measurement_end = measurement_samples.last()?;
    let measurement_peak_rss_bytes = measurement_samples
        .iter()
        .map(|sample| sample.rss_bytes)
        .max()?;
    let measurement_rss_bytes: Vec<u64> = measurement_samples
        .iter()
        .map(|sample| sample.rss_bytes)
        .collect();
    let measurement_p95_rss_bytes = percentile_u64(&measurement_rss_bytes, 0.95)?;

    Some(PeekMemorySummary {
        setup_peak_rss_bytes,
        measurement_start_rss_bytes: measurement_start.rss_bytes,
        measurement_end_rss_bytes: measurement_end.rss_bytes,
        measurement_peak_rss_bytes,
        measurement_p95_rss_bytes,
        measurement_minor_faults_delta: optional_fault_delta(
            measurement_start.minor_faults, measurement_end.minor_faults,
        ),
        measurement_major_faults_delta: optional_fault_delta(
            measurement_start.major_faults, measurement_end.major_faults,
        ),
    })
}

fn build_setup_only_memory_summary(
    setup_samples: &[ReadMemorySample],
) -> Option<PeekMemorySummary> {
    let setup_peak_rss_bytes = setup_samples.iter().map(|sample| sample.rss_bytes).max()?;
    let setup_end = setup_samples.last()?;

    Some(PeekMemorySummary {
        setup_peak_rss_bytes,
        measurement_start_rss_bytes: setup_end.rss_bytes,
        measurement_end_rss_bytes: setup_end.rss_bytes,
        measurement_peak_rss_bytes: setup_end.rss_bytes,
        measurement_p95_rss_bytes: setup_end.rss_bytes,
        measurement_minor_faults_delta: None,
        measurement_major_faults_delta: None,
    })
}

fn optional_fault_delta(start: Option<u64>, end: Option<u64>) -> Option<u64> {
    Some(end?.saturating_sub(start?))
}

fn build_dry_run_profile_output(
    checkpoint_path: &str,
    kernel_path: &str,
    range: ResolvedPeekRange,
    init_time_secs: f64,
    memory_summary: Option<SetupMemorySummary>,
) -> serde_json::Value {
    serde_json::to_value(DryRunProfileOutput {
        dry_run: true,
        checkpoint_path,
        kernel_path,
        resolved_start_height: range.start_height,
        resolved_end_height: range.end_height,
        resolved_peek_count: peek_count(range),
        tip_height: range.tip_height,
        init_time_secs,
        memory_summary,
    })
    .expect("serialize dry-run profile output")
}

fn build_normal_profile_output(
    checkpoint_path: &str,
    kernel_path: &str,
    results: &PeekBenchResults,
) -> serde_json::Value {
    serde_json::to_value(NormalProfileOutput {
        dry_run: false,
        checkpoint_path,
        kernel_path,
        start_height: results.range.start_height,
        end_height: results.range.end_height,
        tip_height: results.range.tip_height,
        peeks_attempted: results.peeks_attempted,
        success_peeks: results.success_peeks,
        missing_peeks: results.missing_peeks,
        error_peeks: results.error_peeks,
        failed_peeks: results.missing_peeks + results.error_peeks,
        init_time_secs: results.init_time_secs,
        total_peek_time_secs: results.total_peek_time_secs,
        peeks_per_second: results.peeks_per_second,
        avg_latency_us: results.avg_latency_us,
        latency_summary_us: results.latency_summary_us.as_ref(),
        memory_summary: results.memory_summary.as_ref(),
    })
    .expect("serialize normal profile output")
}

fn finish_setup_only_memory_sampling(
    sampler: Option<ReadMemorySampler>,
    setup_end_ms: u64,
) -> Result<Option<PeekMemorySummary>, PeekBenchError> {
    let Some(sampler) = sampler else {
        return Ok(None);
    };
    let samples = sampler.finish()?;
    let setup_samples = collect_setup_samples(&samples, setup_end_ms);
    Ok(build_setup_only_memory_summary(&setup_samples))
}

fn finish_measurement_memory_sampling(
    sampler: Option<ReadMemorySampler>,
    setup_end_ms: u64,
    measurement_start_ms: u64,
    measurement_end_ms: u64,
) -> Result<Option<PeekMemorySummary>, PeekBenchError> {
    let Some(sampler) = sampler else {
        return Ok(None);
    };
    let samples = sampler.finish()?;
    let setup_samples = collect_setup_samples(&samples, setup_end_ms);
    let measurement_samples =
        collect_measurement_samples(&samples, measurement_start_ms, measurement_end_ms);
    Ok(build_memory_summary(&setup_samples, &measurement_samples))
}

fn collect_setup_samples(samples: &[ReadMemorySample], setup_end_ms: u64) -> Vec<ReadMemorySample> {
    samples
        .iter()
        .copied()
        .filter(|sample| sample.timestamp_ms <= setup_end_ms)
        .collect()
}

fn collect_measurement_samples(
    samples: &[ReadMemorySample],
    measurement_start_ms: u64,
    measurement_end_ms: u64,
) -> Vec<ReadMemorySample> {
    samples
        .iter()
        .copied()
        .filter(|sample| {
            sample.timestamp_ms >= measurement_start_ms && sample.timestamp_ms <= measurement_end_ms
        })
        .collect()
}

fn read_memory_sample(
    pid: i32,
    timestamp_ms: u64,
) -> Result<Option<ReadMemorySample>, PeekBenchError> {
    let parser = SmapsParser::new(pid);
    let status = match parser.parse_status() {
        Ok(status) => status,
        Err(_) => return Ok(None),
    };
    let page_faults = parser.parse_stat().ok();

    Ok(Some(ReadMemorySample {
        timestamp_ms,
        rss_bytes: status.vm_rss_kb.saturating_mul(1024),
        minor_faults: page_faults.map(|(minor_faults, _)| minor_faults),
        major_faults: page_faults.map(|(_, major_faults)| major_faults),
    }))
}

fn push_memory_sample(
    sink: &Arc<Mutex<Vec<ReadMemorySample>>>,
    sample: ReadMemorySample,
) -> Result<(), PeekBenchError> {
    sink.lock()
        .map_err(|_| PeekBenchError::MemorySample("memory sampler mutex poisoned".to_string()))?
        .push(sample);
    Ok(())
}

fn push_background_sample(
    sink: &Arc<Mutex<Vec<ReadMemorySample>>>,
    sample: Result<Option<ReadMemorySample>, PeekBenchError>,
) -> Result<(), PeekBenchError> {
    match sample {
        Ok(Some(sample)) => push_memory_sample(sink, sample),
        Ok(None) => Ok(()),
        Err(PeekBenchError::MemorySample(_)) => Ok(()),
        Err(error) => Err(error),
    }
}

fn handle_boundary_memory_sample_result(
    result: Result<(), PeekBenchError>,
    phase: &str,
) -> Result<(), PeekBenchError> {
    match result {
        Ok(()) => Ok(()),
        Err(PeekBenchError::MemorySample(error)) => {
            eprintln!("Warning: memory sample unavailable during {phase}: {error}");
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn bytes_to_mib(bytes: u64) -> u64 {
    bytes / (1024 * 1024)
}

fn micros_to_ms(micros: u64) -> f64 {
    micros as f64 / 1000.0
}

fn display_optional_counter(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn duration_to_micros(duration: Duration) -> u64 {
    duration
        .as_micros()
        .min(u64::MAX as u128)
        .try_into()
        .expect("duration micros capped to u64")
}

fn elapsed_ms_since(started_at: Instant) -> u64 {
    started_at
        .elapsed()
        .as_millis()
        .min(u64::MAX as u128)
        .try_into()
        .expect("elapsed millis capped to u64")
}

fn peek_count(range: ResolvedPeekRange) -> u64 {
    range.end_height.saturating_sub(range.start_height) + 1
}

fn should_print_progress(
    peeks_attempted: u64,
    total_peeks: u64,
    since_last_progress: Duration,
) -> bool {
    peeks_attempted == total_peeks
        || peeks_attempted % PROGRESS_HEIGHT_INTERVAL == 0
        || since_last_progress >= PROGRESS_TIME_INTERVAL
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{
        build_dry_run_profile_output, build_memory_summary, build_normal_profile_output,
        handle_boundary_memory_sample_result, peek_height_result, push_background_sample,
        read_memory_sample, resolve_range, summarize_latency_us, LatencySummaryUs, PeekBenchError,
        PeekBenchResults, PeekRangeRequest, PeekSample, PeekSampleKind, ReadMemorySample,
        ResolvedPeekRange,
    };

    #[test]
    fn resolve_range_uses_tip_when_no_end_or_count_is_provided() {
        let resolved = resolve_range(3, PeekRangeRequest::ToTip, 10).expect("resolve range");
        assert_eq!(
            resolved,
            ResolvedPeekRange {
                start_height: 3,
                end_height: 10,
                tip_height: 10,
            }
        );
    }

    #[test]
    fn resolve_range_turns_count_into_inclusive_end_height() {
        let resolved = resolve_range(3, PeekRangeRequest::Count(4), 10).expect("resolve range");
        assert_eq!(
            resolved,
            ResolvedPeekRange {
                start_height: 3,
                end_height: 6,
                tip_height: 10,
            }
        );
    }

    #[test]
    fn resolve_range_rejects_start_height_past_tip() {
        let error = resolve_range(11, PeekRangeRequest::ToTip, 10).expect_err("start past tip");
        assert!(matches!(error, PeekBenchError::StartHeightPastTip { .. }));
    }

    #[test]
    fn resolve_range_rejects_explicit_end_height_past_tip() {
        let error =
            resolve_range(3, PeekRangeRequest::EndHeight(11), 10).expect_err("end past tip");
        assert!(matches!(error, PeekBenchError::EndHeightPastTip { .. }));
    }

    #[test]
    fn resolve_range_rejects_count_that_runs_past_tip() {
        let error = resolve_range(8, PeekRangeRequest::Count(4), 10).expect_err("count past tip");
        assert!(matches!(error, PeekBenchError::CountPastTip { .. }));
    }

    #[test]
    fn resolve_range_rejects_end_height_before_start() {
        let error =
            resolve_range(8, PeekRangeRequest::EndHeight(7), 10).expect_err("end before start");
        assert!(matches!(error, PeekBenchError::EndBeforeStart { .. }));
    }

    #[test]
    fn resolve_range_rejects_count_zero() {
        let error = resolve_range(8, PeekRangeRequest::Count(0), 10).expect_err("count zero");
        assert!(matches!(error, PeekBenchError::InvalidCountZero));
    }

    #[test]
    fn resolve_range_rejects_count_overflow() {
        let error = resolve_range(u64::MAX, PeekRangeRequest::Count(2), u64::MAX)
            .expect_err("count overflow");
        assert!(matches!(error, PeekBenchError::CountOverflows { .. }));
    }

    #[test]
    fn resolve_range_rejects_conflicting_bounds() {
        let error =
            PeekRangeRequest::from_bounds(Some(8), Some(2)).expect_err("conflicting bounds");
        assert!(matches!(error, PeekBenchError::ConflictingBounds));
    }

    #[test]
    fn latency_summary_excludes_error_samples() {
        let summary = summarize_latency_us(&[
            PeekSample {
                latency_us: 100,
                kind: PeekSampleKind::Success,
            },
            PeekSample {
                latency_us: 200,
                kind: PeekSampleKind::Missing,
            },
            PeekSample {
                latency_us: 9_999,
                kind: PeekSampleKind::Error,
            },
            PeekSample {
                latency_us: 300,
                kind: PeekSampleKind::Success,
            },
        ])
        .expect("latency summary");

        assert_eq!(summary.min, 100);
        assert_eq!(summary.p50, 200);
        assert_eq!(summary.p95, 300);
        assert_eq!(summary.p99, 300);
        assert_eq!(summary.max, 300);
    }

    #[test]
    fn peek_height_result_helper_exists_for_runtime_error_preserving_callers() {
        let _ = peek_height_result;
    }

    #[test]
    fn peek_sample_exposes_latency_for_orchestrator_json_shaping() {
        let _latency: fn(&PeekSample) -> u64 = PeekSample::latency_us;
    }

    #[test]
    fn dry_run_json_uses_setup_only_shape() {
        let payload = build_dry_run_profile_output(
            "/tmp/checkpoint.chkjam",
            "/tmp/kernel.jam",
            ResolvedPeekRange {
                start_height: 3,
                end_height: 6,
                tip_height: 10,
            },
            1.25,
            None,
        );

        assert_eq!(payload["dry_run"], serde_json::json!(true));
        assert_eq!(payload["resolved_start_height"], serde_json::json!(3));
        assert_eq!(payload["resolved_end_height"], serde_json::json!(6));
        assert_eq!(payload["resolved_peek_count"], serde_json::json!(4));
        assert_eq!(payload["tip_height"], serde_json::json!(10));
        assert!(payload.get("peeks_attempted").is_none());
        assert!(payload.get("latency_summary_us").is_none());
    }

    #[test]
    fn dry_run_json_memory_summary_omits_measurement_fields() {
        let payload = build_dry_run_profile_output(
            "/tmp/checkpoint.chkjam",
            "/tmp/kernel.jam",
            ResolvedPeekRange {
                start_height: 3,
                end_height: 6,
                tip_height: 10,
            },
            1.25,
            Some(super::SetupMemorySummary {
                setup_peak_rss_bytes: 150 * 1024,
            }),
        );

        assert_eq!(
            payload["memory_summary"]["setup_peak_rss_bytes"],
            serde_json::json!(150 * 1024)
        );
        assert!(payload["memory_summary"]
            .get("measurement_peak_rss_bytes")
            .is_none());
    }

    #[test]
    fn normal_run_json_uses_read_specific_metric_names() {
        let payload = build_normal_profile_output(
            "/tmp/checkpoint.chkjam",
            "/tmp/kernel.jam",
            &PeekBenchResults {
                range: ResolvedPeekRange {
                    start_height: 3,
                    end_height: 6,
                    tip_height: 10,
                },
                peeks_attempted: 4,
                success_peeks: 2,
                missing_peeks: 1,
                error_peeks: 1,
                init_time_secs: 1.0,
                total_peek_time_secs: 0.4,
                peeks_per_second: 10.0,
                avg_latency_us: Some(200),
                latency_summary_us: Some(LatencySummaryUs {
                    min: 100,
                    p50: 200,
                    p95: 300,
                    p99: 300,
                    max: 300,
                }),
                memory_summary: None,
            },
        );

        assert_eq!(payload["peeks_attempted"], serde_json::json!(4));
        assert_eq!(payload["success_peeks"], serde_json::json!(2));
        assert_eq!(payload["missing_peeks"], serde_json::json!(1));
        assert_eq!(payload["error_peeks"], serde_json::json!(1));
        assert_eq!(payload["failed_peeks"], serde_json::json!(2));
        assert!(payload.get("blocks_poked").is_none());
        assert!(payload.get("failed_pokes").is_none());
        assert!(payload.get("latency_summary_us").is_some());
    }

    #[test]
    fn memory_summary_preserves_null_fault_deltas_when_counters_are_unavailable() {
        let setup = vec![ReadMemorySample {
            timestamp_ms: 0,
            rss_bytes: 100 * 1024,
            minor_faults: None,
            major_faults: None,
        }];
        let measurement = vec![
            ReadMemorySample {
                timestamp_ms: 10,
                rss_bytes: 120 * 1024,
                minor_faults: None,
                major_faults: None,
            },
            ReadMemorySample {
                timestamp_ms: 20,
                rss_bytes: 150 * 1024,
                minor_faults: None,
                major_faults: None,
            },
            ReadMemorySample {
                timestamp_ms: 30,
                rss_bytes: 130 * 1024,
                minor_faults: None,
                major_faults: None,
            },
        ];

        let summary = build_memory_summary(&setup, &measurement).expect("memory summary");
        let payload = serde_json::to_value(&summary).expect("serialize memory summary");

        assert_eq!(
            payload["measurement_minor_faults_delta"],
            serde_json::Value::Null
        );
        assert_eq!(
            payload["measurement_major_faults_delta"],
            serde_json::Value::Null
        );
        assert_eq!(
            payload["measurement_peak_rss_bytes"],
            serde_json::json!(150 * 1024)
        );
    }

    #[test]
    fn memory_summary_uses_fault_deltas_only_when_both_endpoints_exist() {
        let setup = vec![ReadMemorySample {
            timestamp_ms: 0,
            rss_bytes: 100 * 1024,
            minor_faults: Some(10),
            major_faults: Some(2),
        }];
        let measurement = vec![
            ReadMemorySample {
                timestamp_ms: 10,
                rss_bytes: 120 * 1024,
                minor_faults: Some(11),
                major_faults: Some(3),
            },
            ReadMemorySample {
                timestamp_ms: 20,
                rss_bytes: 180 * 1024,
                minor_faults: Some(16),
                major_faults: Some(5),
            },
            ReadMemorySample {
                timestamp_ms: 30,
                rss_bytes: 140 * 1024,
                minor_faults: Some(18),
                major_faults: Some(6),
            },
        ];

        let summary = build_memory_summary(&setup, &measurement).expect("memory summary");

        assert_eq!(summary.measurement_start_rss_bytes, 120 * 1024);
        assert_eq!(summary.measurement_end_rss_bytes, 140 * 1024);
        assert_eq!(summary.measurement_peak_rss_bytes, 180 * 1024);
        assert_eq!(summary.measurement_p95_rss_bytes, 180 * 1024);
        assert_eq!(summary.measurement_minor_faults_delta, Some(7));
        assert_eq!(summary.measurement_major_faults_delta, Some(3));
    }

    #[test]
    fn background_sampler_skips_memory_read_failures() {
        let sink = Arc::new(Mutex::new(Vec::new()));

        push_background_sample(
            &sink,
            Err(PeekBenchError::MemorySample(
                "transient /proc read failure".to_string(),
            )),
        )
        .expect("background sampler should ignore proc read failures");

        let samples = sink.lock().expect("sink lock");
        assert!(samples.is_empty());
    }

    #[test]
    fn foreground_sample_failure_does_not_abort_run() {
        handle_boundary_memory_sample_result(
            Err(PeekBenchError::MemorySample(
                "transient /proc read failure".to_string(),
            )),
            "setup end",
        )
        .expect("boundary memory sample failures should be non-fatal");

        assert!(build_memory_summary(&[], &[]).is_none());
    }

    #[test]
    fn read_memory_sample_returns_none_when_status_unavailable() {
        let sample = read_memory_sample(-1, 0).expect("unavailable status should not error");
        assert!(sample.is_none());
    }
}
