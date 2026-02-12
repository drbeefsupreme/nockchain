//! Speed-of-light benchmark runner
//!
//! Pokes archived blocks into a fresh kernel as fast as possible
//! to measure maximum throughput.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use nockapp::nockapp::save::SaveableCheckpoint;
use nockapp::nockapp::wire::WireRepr;
use nockapp::nockapp::NockApp;
use thiserror::Error;
use tokio::time::sleep;
use tracing::info;

use super::archive::{ArchiveFilter, ArchiveReader};
use super::checkpoint::{load_checkpoint, CheckpointLoadError};
use super::kernel_utils::{init_nockapp, peek_heaviest_chain, KernelInitError, PeekChainError};
use super::poke::build_poke_slab_from_jam;
use super::profiling::{
    build_scorecard, find_recovery_ms, infer_gc_events, infer_page_fault_bursts, summarize_phases,
    CheckpointProfile, MemoryProfile, PhaseKind, PhaseWindow, ProcessMemoryProfiler,
};
use super::start_height::{resolve_start_height, StartHeightError};
use super::types::{ProofVersion, SolHeight};

#[derive(Debug, Error)]
pub enum BenchError {
    #[error("Archive error: {0}")]
    Archive(#[from] super::archive::ArchiveError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Kernel load error: {0}")]
    KernelLoad(String),

    #[error("Checkpoint load error: {0}")]
    Checkpoint(#[from] CheckpointLoadError),

    #[error("Cue error: {0}")]
    Cue(String),

    #[error("Poke error: {0}")]
    Poke(String),

    #[error("Noun decode error: {0}")]
    NounDecode(#[from] noun_serde::NounDecodeError),

    #[error("Start height error: {0}")]
    StartHeight(#[from] StartHeightError),

    #[error("Checkpoint chain height unavailable; pass --start-height explicitly")]
    CheckpointHeightUnavailable,

    #[error("NockApp error: {0}")]
    NockApp(#[from] nockapp::nockapp::NockAppError),

    #[error("Kernel init error: {0}")]
    KernelInit(#[from] KernelInitError),

    #[error("Chain height peek error: {0}")]
    ChainPeek(#[from] PeekChainError),

    #[error("Memory sampling error: {0}")]
    MemorySample(String),
}

/// Configuration for the benchmark
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Path to the archive file
    pub archive_path: String,
    /// Path to the kernel jam file
    pub kernel_path: String,
    /// Number of blocks to benchmark (0 = all)
    pub block_count: u64,
    /// Whether to skip genesis block (block 0)
    pub skip_genesis: bool,
    /// Optional proof version filter
    pub proof_version: Option<ProofVersion>,
    /// Optional starting checkpoint to load before benchmarking
    pub checkpoint_path: Option<String>,
    /// Optional start height override
    pub start_height: Option<SolHeight>,
    /// Enable memory timeline profiling during benchmark.
    pub profile_memory: bool,
    /// Sampling interval for memory profile.
    pub profile_interval_ms: u64,
    /// Inferred GC threshold based on RSS drops.
    pub gc_drop_threshold_bytes: u64,
    /// Detect page-fault bursts by minor fault delta.
    pub page_fault_minor_burst_threshold: u64,
    /// Detect page-fault bursts by major fault delta.
    pub page_fault_major_burst_threshold: u64,
    /// Force checkpoint every N accepted blocks (0 disables).
    pub checkpoint_every_blocks: u64,
    /// Max time to wait for RSS recovery after checkpoint.
    pub checkpoint_recovery_timeout_ms: u64,
    /// Recovery condition: RSS <= baseline * (1 + pct/100).
    pub checkpoint_recovery_tolerance_pct: f64,
    /// Working directory for generated checkpoint files.
    pub work_dir: PathBuf,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            archive_path: "blocks.solarch".to_string(),
            kernel_path: "assets/dumb.jam".to_string(),
            block_count: 0,
            skip_genesis: false, // Genesis is required for chain validation
            proof_version: None,
            checkpoint_path: None,
            start_height: None,
            profile_memory: false,
            profile_interval_ms: 500,
            gc_drop_threshold_bytes: 64 * 1024 * 1024, // 64 MiB
            page_fault_minor_burst_threshold: 50_000,
            page_fault_major_burst_threshold: 1,
            checkpoint_every_blocks: 0,
            checkpoint_recovery_timeout_ms: 5_000,
            checkpoint_recovery_tolerance_pct: 5.0,
            work_dir: PathBuf::from("."),
        }
    }
}

/// Results from a benchmark run
#[derive(Debug, Clone)]
pub struct BenchResults {
    /// Total number of blocks poked
    pub blocks_poked: u64,
    /// Total time for all pokes
    pub total_poke_time: Duration,
    /// Time for kernel initialization
    pub init_time: Duration,
    /// Individual block timings (height, duration)
    pub block_timings: Vec<(SolHeight, Duration)>,
    /// Number of failed pokes
    pub failed_pokes: u64,
    /// Number of checkpoint saves performed during benchmark replay.
    pub checkpoint_count: u64,
    /// Total time spent in checkpoint saves during benchmark replay.
    pub checkpoint_total_time: Duration,
    /// Optional memory timeline profile and derived scorecard.
    pub memory_profile: Option<MemoryProfile>,
}

impl BenchResults {
    /// Blocks per second
    pub fn blocks_per_second(&self) -> f64 {
        if self.total_poke_time.as_secs_f64() > 0.0 {
            self.blocks_poked as f64 / self.total_poke_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Average time per block
    pub fn avg_block_time(&self) -> Duration {
        if self.blocks_poked > 0 {
            self.total_poke_time / self.blocks_poked as u32
        } else {
            Duration::ZERO
        }
    }

    pub fn avg_checkpoint_time(&self) -> Option<Duration> {
        if self.checkpoint_count == 0 {
            None
        } else {
            Some(self.checkpoint_total_time / self.checkpoint_count as u32)
        }
    }

    /// Print a summary of the results
    pub fn print_summary(&self) {
        println!("\n=== Benchmark Results ===\n");
        println!("Blocks poked:    {}", self.blocks_poked);
        println!("Failed pokes:    {}", self.failed_pokes);
        println!("Init time:       {:.2}s", self.init_time.as_secs_f64());
        println!(
            "Total poke time: {:.2}s",
            self.total_poke_time.as_secs_f64()
        );
        println!(
            "Avg per block:   {:.2}ms",
            self.avg_block_time().as_secs_f64() * 1000.0
        );
        println!("Throughput:      {:.2} blocks/s", self.blocks_per_second());
        println!("Checkpoints:     {}", self.checkpoint_count);
        if let Some(avg) = self.avg_checkpoint_time() {
            println!("Avg checkpoint:  {:.2}s", avg.as_secs_f64());
        }

        if let Some(profile) = &self.memory_profile {
            println!("\n=== Memory Profile ===\n");
            println!("Samples:         {}", profile.samples.len());
            println!("GC events:       {}", profile.gc_events.len());
            println!("Fault bursts:    {}", profile.page_fault_bursts.len());
            println!("Peak RSS:        {:.2} MiB", profile.scorecard.peak_rss_mib);
            println!("P95 RSS:         {:.2} MiB", profile.scorecard.p95_rss_mib);
            if let Some(value) = profile.scorecard.checkpoint_peak_rss_mib {
                println!("Ckpt peak RSS:   {:.2} MiB", value);
            }
            if let Some(value) = profile.scorecard.checkpoint_seconds_per_gib {
                println!("Ckpt sec/GiB:    {:.2}", value);
            }
            if let Some(value) = profile.scorecard.gc_pause_p95_ms {
                println!("GC pause p95:    {:.1} ms", value);
            }
            println!(
                "GC / 1k blocks:  {:.2}",
                profile.scorecard.gc_events_per_1k_blocks
            );
        }
    }
}

/// Speed-of-light benchmark runner
pub struct BenchRunner {
    config: BenchConfig,
    nockapp: Option<NockApp>,
}

impl BenchRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchConfig) -> Self {
        Self {
            config,
            nockapp: None,
        }
    }

    /// Initialize a fresh kernel (no checkpoint state)
    pub async fn initialize(&mut self) -> Result<(), BenchError> {
        info!(kernel = %self.config.kernel_path, "Initializing fresh kernel for benchmark");

        let checkpoint = if let Some(path) = &self.config.checkpoint_path {
            let loaded = load_checkpoint(path)?;
            Some(SaveableCheckpoint {
                ker_hash: loaded.ker_hash,
                event_num: loaded.event_num,
                state: loaded.state,
                cold: loaded.cold,
            })
        } else {
            None
        };

        let nockapp = init_nockapp(
            std::path::Path::new(&self.config.kernel_path),
            checkpoint,
            &self.config.work_dir,
            self.config.checkpoint_every_blocks > 0,
            false,
        )
        .await?;

        info!("Fresh kernel initialized");
        self.nockapp = Some(nockapp);
        Ok(())
    }

    /// Run the benchmark
    pub async fn run(&mut self) -> Result<BenchResults, BenchError> {
        // Load archive
        info!(archive = %self.config.archive_path, "Loading archive");
        let archive_bytes = std::fs::read(&self.config.archive_path)?;
        let reader = ArchiveReader::from_bytes(archive_bytes)?;
        let metadata = reader.metadata();

        info!(
            blocks = metadata.block_count,
            min_height = metadata.min_height.as_u64(),
            max_height = metadata.max_height.as_u64(),
            "Archive loaded"
        );

        let run_start = Instant::now();
        let mut profiler = if self.config.profile_memory {
            Some(ProcessMemoryProfiler::new(self.config.profile_interval_ms))
        } else {
            None
        };
        if let Some(profiler) = profiler.as_mut() {
            profiler
                .sample_now(0)
                .map_err(|e| BenchError::MemorySample(e.to_string()))?;
        }

        let mut phase_windows = Vec::new();
        let mut checkpoint_profiles = Vec::new();
        let mut checkpoint_count = 0u64;
        let mut checkpoint_total_time = Duration::ZERO;

        // Initialize kernel
        let init_start = Instant::now();
        self.initialize().await?;
        let init_time = init_start.elapsed();
        let init_end_ms = run_start.elapsed().as_millis() as u64;
        phase_windows.push(PhaseWindow::new(PhaseKind::Init, 0, init_end_ms));
        if let Some(profiler) = profiler.as_mut() {
            profiler
                .sample_now(init_end_ms)
                .map_err(|e| BenchError::MemorySample(e.to_string()))?;
        }

        let nockapp = self.nockapp.as_mut().ok_or(BenchError::KernelLoad(
            "NockApp not initialized".to_string(),
        ))?;

        let checkpoint_height = if self.config.checkpoint_path.is_some() {
            let height = peek_heaviest_chain(nockapp).await?;
            height
                .map(|(height, _)| SolHeight(height.0 .0))
                .ok_or(BenchError::CheckpointHeightUnavailable)
                .map(Some)?
        } else {
            None
        };

        let start_height = resolve_start_height(self.config.start_height, checkpoint_height)?;

        let block_limit = if self.config.block_count > 0 {
            Some(self.config.block_count)
        } else {
            None
        };

        info!(
            skip_genesis = self.config.skip_genesis,
            proof_version = self.config.proof_version.map(|v| v.as_str()),
            start_height = start_height.as_u64(),
            profile_memory = self.config.profile_memory,
            checkpoint_every_blocks = self.config.checkpoint_every_blocks,
            "Starting benchmark"
        );

        let replay_start_ms = run_start.elapsed().as_millis() as u64;
        let mut block_timings = Vec::new();
        let mut blocks_poked = 0u64;
        let mut failed_pokes = 0u64;
        let poke_start = Instant::now();

        // Wire for the poke
        let wire = WireRepr {
            source: "bench",
            version: 1,
            tags: vec![],
        };

        let filter = ArchiveFilter {
            proof_version: self.config.proof_version,
            start_height: Some(start_height),
            end_height: None,
        };

        for (entry, jam_bytes) in reader.iter_filtered(filter) {
            if self.config.skip_genesis && entry.height == SolHeight::ZERO {
                continue;
            }
            if let Some(limit) = block_limit {
                if blocks_poked >= limit {
                    break;
                }
            }

            if let Some(profiler) = profiler.as_mut() {
                let now_ms = run_start.elapsed().as_millis() as u64;
                profiler
                    .maybe_sample(now_ms)
                    .map_err(|e| BenchError::MemorySample(e.to_string()))?;
            }

            let block_start = Instant::now();

            let poke_slab = match build_poke_slab_from_jam(jam_bytes) {
                Ok(slab) => slab,
                Err(e) => {
                    info!(
                        height = entry.height.as_u64(),
                        error = %e,
                        "Failed to build poke slab"
                    );
                    failed_pokes += 1;
                    continue;
                }
            };

            // Poke!
            match nockapp.poke(wire.clone(), poke_slab).await {
                Ok(_effects) => {
                    let block_time = block_start.elapsed();
                    block_timings.push((entry.height, block_time));
                    blocks_poked += 1;

                    if let Some(profiler) = profiler.as_mut() {
                        let now_ms = run_start.elapsed().as_millis() as u64;
                        profiler
                            .maybe_sample(now_ms)
                            .map_err(|e| BenchError::MemorySample(e.to_string()))?;
                    }

                    if self.config.checkpoint_every_blocks > 0
                        && blocks_poked % self.config.checkpoint_every_blocks == 0
                    {
                        let checkpoint_start_ms = run_start.elapsed().as_millis() as u64;
                        let before_size = latest_checkpoint_size(&self.config.work_dir)?;
                        let pre_rss = profiler
                            .as_ref()
                            .and_then(|profiler| profiler.latest_rss_bytes())
                            .unwrap_or(0);
                        if let Some(profiler) = profiler.as_mut() {
                            profiler
                                .sample_now(checkpoint_start_ms)
                                .map_err(|e| BenchError::MemorySample(e.to_string()))?;
                        }

                        let checkpoint_start = Instant::now();
                        nockapp.save_blocking().await?;
                        let checkpoint_duration = checkpoint_start.elapsed();
                        checkpoint_total_time += checkpoint_duration;
                        checkpoint_count += 1;

                        let checkpoint_end_ms = run_start.elapsed().as_millis() as u64;
                        if let Some(profiler) = profiler.as_mut() {
                            profiler
                                .sample_now(checkpoint_end_ms)
                                .map_err(|e| BenchError::MemorySample(e.to_string()))?;
                        }
                        let after_size = latest_checkpoint_size(&self.config.work_dir)?;
                        let checkpoint_size_bytes =
                            estimate_checkpoint_size(before_size, after_size);

                        let mut recovery_ms = None;
                        if self.config.profile_memory {
                            let recovery_deadline_ms = checkpoint_end_ms
                                .saturating_add(self.config.checkpoint_recovery_timeout_ms);
                            loop {
                                let now_ms = run_start.elapsed().as_millis() as u64;
                                if now_ms > recovery_deadline_ms {
                                    break;
                                }
                                sleep(Duration::from_millis(
                                    self.config.profile_interval_ms.min(250).max(1),
                                ))
                                .await;
                                let now_ms = run_start.elapsed().as_millis() as u64;
                                if let Some(profiler) = profiler.as_mut() {
                                    profiler
                                        .sample_now(now_ms)
                                        .map_err(|e| BenchError::MemorySample(e.to_string()))?;
                                    recovery_ms = find_recovery_ms(
                                        profiler.samples(),
                                        checkpoint_end_ms,
                                        pre_rss,
                                        self.config.checkpoint_recovery_tolerance_pct,
                                    );
                                    if recovery_ms.is_some() {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            }
                        }

                        let post_rss = profiler
                            .as_ref()
                            .and_then(|profiler| profiler.latest_rss_bytes())
                            .unwrap_or(0);
                        let peak_rss = profiler
                            .as_ref()
                            .and_then(|profiler| {
                                profiler.peak_rss_between(checkpoint_start_ms, checkpoint_end_ms)
                            })
                            .unwrap_or(post_rss.max(pre_rss));

                        checkpoint_profiles.push(CheckpointProfile {
                            start_ms: checkpoint_start_ms,
                            end_ms: checkpoint_end_ms,
                            duration_ms: checkpoint_duration.as_millis() as u64,
                            pre_checkpoint_rss_bytes: pre_rss,
                            post_checkpoint_rss_bytes: post_rss,
                            peak_rss_bytes: peak_rss,
                            recovery_ms,
                            checkpoint_size_bytes,
                        });
                        phase_windows.push(PhaseWindow::new(
                            PhaseKind::Checkpoint,
                            checkpoint_start_ms,
                            checkpoint_end_ms,
                        ));
                    }

                    if blocks_poked % 100 == 0 {
                        info!(
                            blocks = blocks_poked,
                            height = entry.height.as_u64(),
                            elapsed_ms = poke_start.elapsed().as_millis(),
                            "Progress"
                        );
                    }
                }
                Err(e) => {
                    info!(height = entry.height.as_u64(), error = ?e, "Poke failed");
                    failed_pokes += 1;
                }
            }
        }

        let total_poke_time = poke_start.elapsed();
        let replay_end_ms = run_start.elapsed().as_millis() as u64;
        phase_windows.push(PhaseWindow::new(
            PhaseKind::Replay,
            replay_start_ms,
            replay_end_ms,
        ));

        let memory_profile = if let Some(mut profiler) = profiler {
            profiler
                .sample_now(replay_end_ms)
                .map_err(|e| BenchError::MemorySample(e.to_string()))?;

            let mut samples = profiler.into_samples();
            samples.sort_by_key(|sample| sample.timestamp_ms);

            let gc_events = infer_gc_events(&samples, self.config.gc_drop_threshold_bytes);
            for event in &gc_events {
                phase_windows.push(PhaseWindow::new(
                    PhaseKind::Gc,
                    event.start_ms,
                    event.end_ms,
                ));
            }
            phase_windows.sort_by_key(|window| (window.start_ms, window.end_ms));

            let phase_summaries = summarize_phases(&samples, &phase_windows);
            let page_fault_bursts = infer_page_fault_bursts(
                &samples, self.config.page_fault_minor_burst_threshold,
                self.config.page_fault_major_burst_threshold,
            );
            let scorecard = build_scorecard(
                &samples, &checkpoint_profiles, &gc_events, &page_fault_bursts, blocks_poked,
                failed_pokes, total_poke_time,
            );

            Some(MemoryProfile {
                interval_ms: self.config.profile_interval_ms,
                samples,
                phase_windows,
                phase_summaries,
                checkpoint_profiles: checkpoint_profiles.clone(),
                gc_events,
                page_fault_bursts,
                scorecard,
            })
        } else {
            None
        };

        Ok(BenchResults {
            blocks_poked,
            total_poke_time,
            init_time,
            block_timings,
            failed_pokes,
            checkpoint_count,
            checkpoint_total_time,
            memory_profile,
        })
    }
}

fn latest_checkpoint_size(work_dir: &Path) -> Result<Option<u64>, std::io::Error> {
    let mut latest: Option<(SystemTime, u64)> = None;
    for name in ["0.chkjam", "1.chkjam"] {
        let path = work_dir.join(name);
        if !path.exists() {
            continue;
        }
        let metadata = std::fs::metadata(path)?;
        let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let size = metadata.len();
        match latest {
            Some((current_modified, _)) if modified <= current_modified => {}
            _ => latest = Some((modified, size)),
        }
    }
    Ok(latest.map(|(_, size)| size))
}

fn estimate_checkpoint_size(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    match (before, after) {
        (_, None) => None,
        (None, Some(after_size)) => Some(after_size),
        (Some(before_size), Some(after_size)) => {
            let diff = after_size.saturating_sub(before_size);
            if diff > 0 {
                Some(diff)
            } else {
                Some(after_size)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_config_default_profile_values() {
        let config = BenchConfig::default();
        assert!(!config.profile_memory);
        assert_eq!(config.profile_interval_ms, 500);
        assert_eq!(config.gc_drop_threshold_bytes, 64 * 1024 * 1024);
        assert_eq!(config.checkpoint_every_blocks, 0);
    }

    #[test]
    fn test_estimate_checkpoint_size_prefers_delta() {
        assert_eq!(estimate_checkpoint_size(Some(100), Some(160)), Some(60));
    }

    #[test]
    fn test_estimate_checkpoint_size_falls_back_to_after() {
        assert_eq!(estimate_checkpoint_size(Some(160), Some(160)), Some(160));
        assert_eq!(estimate_checkpoint_size(None, Some(160)), Some(160));
        assert_eq!(estimate_checkpoint_size(None, None), None);
    }
}
