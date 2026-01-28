//! Block extraction from a running kernel via peek

use std::path::PathBuf;

use nockapp::kernel::boot::TraceOpts;
use nockapp::kernel::form::Kernel;
use nockapp::nockapp::NockApp;
use nockapp::nockapp::save::SaveableCheckpoint;
use nockapp::noun::slab::NounSlab;
use nockchain_types::tx_engine::common::{BlockHeight, Hash};
use nockvm::noun::{NounAllocator, SIG};
use noun_serde::NounDecode;
use thiserror::Error;
use tracing::{debug, info};
use zkvm_jetpack::hot::produce_prover_hot_state;

use super::cache::SpeedOfLightCache;
use super::checkpoint::{load_checkpoint, CheckpointLoadError};
use super::types::{BlockData, BlockRangeEntryNoun};

#[derive(Debug, Error)]
pub enum ExtractorError {
    #[error("Checkpoint load error: {0}")]
    CheckpointLoad(#[from] CheckpointLoadError),

    #[error("Kernel load error: {0}")]
    KernelLoad(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Peek failed")]
    PeekFailed,

    #[error("Peek returned no data")]
    PeekReturnedNoData,

    #[error("Noun decode error: {0}")]
    NounDecode(#[from] noun_serde::NounDecodeError),

    #[error("NockApp error: {0}")]
    NockApp(#[from] nockapp::nockapp::NockAppError),
}

/// Configuration for block extraction
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// Path to the checkpoint file
    pub checkpoint_path: String,
    /// Path to the kernel jam file
    pub kernel_path: String,
    /// Number of blocks to extract (starting from genesis)
    pub block_count: u64,
    /// Chunk size for range queries
    pub chunk_size: u64,
    /// Working directory for NockApp (for any temp files)
    pub work_dir: PathBuf,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            checkpoint_path: "0.chkjam".to_string(),
            kernel_path: "assets/dumb.jam".to_string(),
            block_count: 1000,
            chunk_size: 8,
            work_dir: PathBuf::from("."),
        }
    }
}

/// Extracts blocks from a checkpoint using kernel peek operations
pub struct BlockExtractor {
    config: ExtractorConfig,
    nockapp: Option<NockApp>,
}

impl BlockExtractor {
    /// Create a new extractor with the given configuration
    pub fn new(config: ExtractorConfig) -> Self {
        Self {
            config,
            nockapp: None,
        }
    }

    /// Initialize the NockApp from checkpoint and kernel files
    pub async fn initialize(&mut self) -> Result<(), ExtractorError> {
        println!("[DEBUG initialize] Starting initialization");
        info!(
            checkpoint = %self.config.checkpoint_path,
            kernel = %self.config.kernel_path,
            "Initializing block extractor"
        );

        // Load checkpoint
        println!("[DEBUG initialize] Loading checkpoint from {}", self.config.checkpoint_path);
        let loaded = load_checkpoint(&self.config.checkpoint_path)?;
        println!("[DEBUG initialize] Checkpoint loaded, event_num={}", loaded.event_num);
        info!(
            event_num = loaded.event_num,
            "Loaded checkpoint"
        );

        // Load kernel bytes
        println!("[DEBUG initialize] Loading kernel from {}", self.config.kernel_path);
        let kernel_bytes = std::fs::read(&self.config.kernel_path)?;
        println!("[DEBUG initialize] Kernel loaded, size={} bytes", kernel_bytes.len());
        info!(
            kernel_size = kernel_bytes.len(),
            "Loaded kernel jam"
        );

        // Create SaveableCheckpoint from loaded data
        println!("[DEBUG initialize] Creating SaveableCheckpoint");
        let checkpoint = SaveableCheckpoint {
            ker_hash: loaded.ker_hash,
            event_num: loaded.event_num,
            state: loaded.state,
            cold: loaded.cold,
        };

        // Create NockApp with kernel loader closure
        println!("[DEBUG initialize] Creating NockApp");
        let work_dir = self.config.work_dir.clone();

        // Get the prover hot state (jets) - this is critical for performance!
        let hot_state = produce_prover_hot_state();
        println!("[DEBUG initialize] Got {} hot state entries (jets)", hot_state.len());

        let nockapp = NockApp::new(
            |existing_checkpoint| {
                println!("[DEBUG initialize] Kernel loader closure called");
                // Use the checkpoint we loaded, ignoring any existing checkpoint from disk
                let checkpoint_to_use = existing_checkpoint.or(Some(checkpoint));
                async move {
                    println!("[DEBUG initialize] Calling Kernel::load_with_hot_state_medium (16GB stack)");
                    let result = Kernel::load_with_hot_state_medium(
                        &kernel_bytes,
                        checkpoint_to_use,
                        &hot_state,
                        vec![],
                        TraceOpts::default(),
                        None, // No PMA for now
                    )
                    .await;
                    println!("[DEBUG initialize] Kernel::load_with_hot_state_medium completed");
                    result
                }
            },
            &work_dir,
            None, // No save interval - we're read-only
            false, // Disable checkpointing
        )
        .await
        .map_err(|e| ExtractorError::KernelLoad(e.to_string()))?;

        println!("[DEBUG initialize] NockApp::new completed");
        info!("NockApp initialized successfully");
        self.nockapp = Some(nockapp);
        Ok(())
    }

    /// Get the current chain tip height
    pub async fn get_chain_height(&mut self) -> Result<(u64, Hash), ExtractorError> {
        let nockapp = self.nockapp.as_mut().ok_or(ExtractorError::KernelLoad(
            "NockApp not initialized".to_string(),
        ))?;

        let mut path_slab = NounSlab::new();
        let tag = nockapp::utils::make_tas(&mut path_slab, "heaviest-chain").as_noun();
        let path_noun = nockvm::noun::T(&mut path_slab, &[tag, SIG]);
        path_slab.set_root(path_noun);

        let result = nockapp.peek(path_slab).await?;

        let result_noun = unsafe { result.root() };
        let space = result.noun_space();

        let opt: Option<Option<(BlockHeight, Hash)>> = NounDecode::from_noun(&result_noun, &space)?;

        let (height, hash) = opt.flatten().ok_or(ExtractorError::PeekReturnedNoData)?;

        Ok((height.0 .0, hash))
    }

    /// Extract blocks in a range and return as BlockData
    pub async fn extract_blocks_range(
        &mut self,
        start: u64,
        end: u64,
    ) -> Result<Vec<BlockData>, ExtractorError> {
        let nockapp = self.nockapp.as_mut().ok_or(ExtractorError::KernelLoad(
            "NockApp not initialized".to_string(),
        ))?;

        debug!(start, end, "Extracting block range");

        let mut path_slab = NounSlab::new();
        let tag = nockapp::utils::make_tas(&mut path_slab, "heaviest-chain-blocks-range").as_noun();
        let start_noun = nockvm::noun::D(start);
        let end_noun = nockvm::noun::D(end);
        let path_noun = nockvm::noun::T(&mut path_slab, &[tag, start_noun, end_noun, SIG]);
        path_slab.set_root(path_noun);

        let result = nockapp.peek(path_slab).await?;

        let result_noun = unsafe { result.root() };
        let space = result.noun_space();

        // Decode Option<Option<Vec<BlockRangeEntryNoun>>>
        let opt: Option<Option<Vec<BlockRangeEntryNoun>>> =
            NounDecode::from_noun(&result_noun, &space)?;
        let entries = opt.flatten().ok_or(ExtractorError::PeekReturnedNoData)?;

        let mut blocks = Vec::with_capacity(entries.len());
        for entry in entries {
            let block = entry.into_block_data(&space)?;
            blocks.push(block);
        }

        debug!(
            start,
            end,
            block_count = blocks.len(),
            "Extracted block range"
        );

        Ok(blocks)
    }

    /// Extract the first N blocks into a cache
    /// If chain height is available, uses that as an upper bound.
    /// Otherwise, extracts until empty results or count is reached.
    pub async fn extract_to_cache(&mut self, count: u64) -> Result<SpeedOfLightCache, ExtractorError> {
        info!(count, "Extracting blocks to cache");

        // Try to get chain height, but don't fail if unavailable
        let end_height = match self.get_chain_height().await {
            Ok((chain_height, _)) => {
                info!(chain_height, "Chain height available");
                count.saturating_sub(1).min(chain_height)
            }
            Err(ExtractorError::PeekReturnedNoData) => {
                info!("Chain height unavailable, will extract until empty results");
                count.saturating_sub(1)
            }
            Err(e) => return Err(e),
        };

        let mut cache = SpeedOfLightCache::new();
        let mut current = 0u64;

        while current <= end_height {
            let chunk_end = (current + self.config.chunk_size - 1).min(end_height);

            match self.extract_blocks_range(current, chunk_end).await {
                Ok(blocks) => {
                    if blocks.is_empty() {
                        info!(current, "No more blocks available, stopping extraction");
                        break;
                    }
                    let block_count = blocks.len();
                    cache.insert_blocks(blocks);

                    info!(
                        start = current,
                        end = chunk_end,
                        blocks = block_count,
                        total_blocks = cache.block_count(),
                        total_txs = cache.transaction_count(),
                        "Inserted block chunk"
                    );
                }
                Err(ExtractorError::PeekReturnedNoData) => {
                    info!(current, "No more blocks available, stopping extraction");
                    break;
                }
                Err(e) => return Err(e),
            }

            current = chunk_end + 1;
        }

        info!(stats = %cache.stats(), "Extraction complete");
        Ok(cache)
    }

    /// Run the full extraction pipeline
    pub async fn run(&mut self) -> Result<SpeedOfLightCache, ExtractorError> {
        self.initialize().await?;
        self.extract_to_cache(self.config.block_count).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speed_of_light::checkpoint::load_checkpoint;
    use std::sync::Arc;
    use tokio::sync::{Mutex, OnceCell};

    // Path helpers - tests run from crate root, so we need to go up to repo root
    fn checkpoint_path() -> String {
        std::env::var("SOL_CHECKPOINT_PATH")
            .unwrap_or_else(|_| "../../0.chkjam".to_string())
    }

    fn kernel_path() -> String {
        std::env::var("SOL_KERNEL_PATH")
            .unwrap_or_else(|_| "../../assets/dumb.jam".to_string())
    }

    // Shared extractor for integration tests - avoids reinitializing for each test
    static SHARED_EXTRACTOR: OnceCell<Arc<Mutex<BlockExtractor>>> = OnceCell::const_new();

    async fn get_shared_extractor() -> Arc<Mutex<BlockExtractor>> {
        SHARED_EXTRACTOR
            .get_or_init(|| async {
                println!("=== Initializing shared BlockExtractor ===");
                let config = ExtractorConfig {
                    checkpoint_path: checkpoint_path(),
                    kernel_path: kernel_path(),
                    block_count: 1000,
                    chunk_size: 8,
                    work_dir: PathBuf::from("."),
                };
                let mut extractor = BlockExtractor::new(config);
                extractor.initialize().await.expect("should initialize NockApp");
                println!("=== Shared BlockExtractor ready ===");
                Arc::new(Mutex::new(extractor))
            })
            .await
            .clone()
    }

    // ==================== QUICK TESTS ====================
    // These tests don't require kernel initialization and run fast

    /// Test that we can load and parse a checkpoint file
    #[test]
    fn test_load_checkpoint_standalone() {
        let path = checkpoint_path();
        println!("Loading checkpoint from: {}", path);

        let loaded = load_checkpoint(&path).expect("should load checkpoint");

        println!("Checkpoint loaded successfully:");
        println!("  event_num: {}", loaded.event_num);
        println!("  ker_hash: {:?}", loaded.ker_hash);

        assert!(loaded.event_num > 0, "event_num should be > 0 for a real checkpoint");
    }

    /// Test ExtractorConfig defaults
    #[test]
    fn test_extractor_config_defaults() {
        let config = ExtractorConfig::default();
        assert_eq!(config.block_count, 1000);
        assert_eq!(config.chunk_size, 8);
    }

    /// Test BlockExtractor can be created without initialization
    #[test]
    fn test_extractor_creation() {
        let config = ExtractorConfig {
            checkpoint_path: checkpoint_path(),
            kernel_path: kernel_path(),
            block_count: 100,
            chunk_size: 8,
            work_dir: PathBuf::from("."),
        };
        let extractor = BlockExtractor::new(config);
        assert!(extractor.nockapp.is_none(), "nockapp should be None before initialize");
    }

    // ==================== INTEGRATION TESTS ====================
    // These tests require full kernel initialization (~2-3 minutes)
    // Run with: cargo test --release -p nockchain-bench integration -- --ignored --test-threads=1
    // NOTE: Must use --test-threads=1 to share the extractor instance

    /// Full integration test: Initialize extractor with kernel and checkpoint
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_01_extractor_initializes() {
        let extractor = get_shared_extractor().await;
        let guard = extractor.lock().await;
        assert!(guard.nockapp.is_some(), "nockapp should be Some after initialize");
        println!("Extractor initialized successfully");
    }

    /// Full integration test: Get chain height via peek
    /// NOTE: The heaviest-chain peek may return no data if the checkpoint
    /// was created before the chain tip was calculated/cached. This is expected
    /// behavior - blocks can still be extracted via the range peek.
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_02_peek_chain_height() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        println!("[TEST 02] About to call get_chain_height()");
        match guard.get_chain_height().await {
            Ok((height, hash)) => {
                println!("[TEST 02] Chain height: {}", height);
                println!("[TEST 02] Tip hash: {}", hash.to_base58());
                assert!(height > 0, "chain height should be > 0 for a real checkpoint");
            }
            Err(ExtractorError::PeekReturnedNoData) => {
                // This is expected for some checkpoints - the heaviest chain tip
                // isn't always cached in the checkpoint state
                println!("[TEST 02] Chain height not available in checkpoint (expected for some states)");
                println!("[TEST 02] Blocks can still be extracted via extract_blocks_range()");
            }
            Err(e) => {
                println!("[TEST 02] get_chain_height failed with unexpected error: {:?}", e);
                panic!("unexpected error: {:?}", e);
            }
        }
    }

    /// Full integration test: Extract a single block
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_03_extract_single_block() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        let blocks = guard.extract_blocks_range(0, 0).await.expect("should extract block 0");

        assert_eq!(blocks.len(), 1, "should get exactly 1 block");
        let block = &blocks[0];

        println!("Block 0:");
        println!("  height: {}", block.height);
        println!("  block_id: {}", block.block_id.to_base58());
        println!("  parent_id: {}", block.parent_id.to_base58());
        println!("  timestamp: {}", block.timestamp);
        println!("  tx_count: {}", block.transactions.len());

        assert_eq!(block.height, 0, "first block should have height 0");
    }

    /// Full integration test: Extract a range of blocks
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_04_extract_block_range() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        let blocks = guard.extract_blocks_range(0, 7).await.expect("should extract blocks 0-7");

        assert_eq!(blocks.len(), 8, "should get 8 blocks");

        for (i, block) in blocks.iter().enumerate() {
            println!("Block {}: height={}, txs={}", i, block.height, block.transactions.len());
            assert_eq!(block.height, i as u64, "block height should match index");
        }
    }
}
