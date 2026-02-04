//! Speed-of-light benchmark runner
//!
//! Pokes archived blocks into a fresh kernel as fast as possible
//! to measure maximum throughput.

use std::time::{Duration, Instant};

use bytes::Bytes;
use nockapp::nockapp::wire::WireRepr;
use nockapp::nockapp::NockApp;
use nockapp::noun::slab::NounSlab;
use nockapp::nockapp::save::SaveableCheckpoint;
use nockvm::noun::NounAllocator;
use thiserror::Error;
use tracing::info;

use super::archive::{ArchiveFilter, ArchiveReader};
use super::checkpoint::{load_checkpoint, CheckpointLoadError};
use super::kernel_utils::{init_nockapp, peek_heaviest_chain, KernelInitError, PeekChainError};
use super::poke::{extract_page_from_entry, make_heard_block_cause};
use super::start_height::{resolve_start_height, StartHeightError};
use super::types::ProofVersion;

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
    pub start_height: Option<u64>,
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
    pub block_timings: Vec<(u64, Duration)>,
    /// Number of failed pokes
    pub failed_pokes: u64,
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

    /// Print a summary of the results
    pub fn print_summary(&self) {
        println!("\n=== Benchmark Results ===\n");
        println!("Blocks poked:    {}", self.blocks_poked);
        println!("Failed pokes:    {}", self.failed_pokes);
        println!("Init time:       {:.2}s", self.init_time.as_secs_f64());
        println!("Total poke time: {:.2}s", self.total_poke_time.as_secs_f64());
        println!("Avg per block:   {:.2}ms", self.avg_block_time().as_secs_f64() * 1000.0);
        println!("Throughput:      {:.2} blocks/s", self.blocks_per_second());
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

        let work_dir = std::path::PathBuf::from(".");
        let nockapp = init_nockapp(
            std::path::Path::new(&self.config.kernel_path),
            checkpoint,
            &work_dir,
            false,
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
            min_height = metadata.min_height,
            max_height = metadata.max_height,
            "Archive loaded"
        );

        // Initialize kernel
        let init_start = Instant::now();
        self.initialize().await?;
        let init_time = init_start.elapsed();

        let nockapp = self.nockapp.as_mut().ok_or(BenchError::KernelLoad(
            "NockApp not initialized".to_string(),
        ))?;

        let checkpoint_height = if self.config.checkpoint_path.is_some() {
            let height = peek_heaviest_chain(nockapp).await?;
            height
                .map(|(height, _)| height.0 .0)
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
            start_height,
            "Starting benchmark"
        );

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
            if self.config.skip_genesis && entry.height == 0 {
                continue;
            }
            if let Some(limit) = block_limit {
                if blocks_poked >= limit {
                    break;
                }
            }

            let block_start = Instant::now();

            // Cue the block entry
            let mut entry_slab: NounSlab = NounSlab::new();
            let entry_noun = match entry_slab.cue_into(Bytes::from(jam_bytes.to_vec())) {
                Ok(noun) => noun,
                Err(e) => {
                    info!(height = entry.height, error = ?e, "Failed to cue block");
                    failed_pokes += 1;
                    continue;
                }
            };

            // Extract the page from the entry
            let page = match extract_page_from_entry(entry_noun, &entry_slab) {
                Ok(p) => p,
                Err(e) => {
                    info!(height = entry.height, error = %e, "Failed to extract page");
                    failed_pokes += 1;
                    continue;
                }
            };

            // Build the poke cause in a new slab
            let mut poke_slab: NounSlab = NounSlab::new();
            let space = entry_slab.noun_space();
            let page_copy = poke_slab.copy_into(page, &space);
            let cause = make_heard_block_cause(page_copy, &mut poke_slab);
            poke_slab.set_root(cause);

            // Poke!
            match nockapp.poke(wire.clone(), poke_slab).await {
                Ok(_effects) => {
                    let block_time = block_start.elapsed();
                    block_timings.push((entry.height, block_time));
                    blocks_poked += 1;

                    if blocks_poked % 100 == 0 {
                        info!(
                            blocks = blocks_poked,
                            height = entry.height,
                            elapsed_ms = poke_start.elapsed().as_millis(),
                            "Progress"
                        );
                    }
                }
                Err(e) => {
                    info!(height = entry.height, error = ?e, "Poke failed");
                    failed_pokes += 1;
                }
            }
        }

        let total_poke_time = poke_start.elapsed();

        Ok(BenchResults {
            blocks_poked,
            total_poke_time,
            init_time,
            block_timings,
            failed_pokes,
        })
    }
}
