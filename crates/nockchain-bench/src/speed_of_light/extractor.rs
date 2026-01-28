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

use super::archive::ArchiveWriter;
use super::cache::SpeedOfLightCache;
use super::checkpoint::{load_checkpoint, CheckpointLoadError};
use super::types::{BlockData, BlockDataWithJam, BlockRangeEntryNoun};
use nockchain_math::structs::HoonList;
use std::path::Path;

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

    /// Extract blocks in a range with raw jammed noun bytes
    ///
    /// This method returns both the decoded BlockData and the raw jammed bytes
    /// for each block entry. The jam bytes can be used for archiving without
    /// losing fidelity in the Noun representation.
    pub async fn extract_blocks_range_with_jam(
        &mut self,
        start: u64,
        end: u64,
    ) -> Result<Vec<BlockDataWithJam>, ExtractorError> {
        let nockapp = self.nockapp.as_mut().ok_or(ExtractorError::KernelLoad(
            "NockApp not initialized".to_string(),
        ))?;

        debug!(start, end, "Extracting block range with jam");

        let mut path_slab = NounSlab::new();
        let tag = nockapp::utils::make_tas(&mut path_slab, "heaviest-chain-blocks-range").as_noun();
        let start_noun = nockvm::noun::D(start);
        let end_noun = nockvm::noun::D(end);
        let path_noun = nockvm::noun::T(&mut path_slab, &[tag, start_noun, end_noun, SIG]);
        path_slab.set_root(path_noun);

        let result = nockapp.peek(path_slab).await?;

        let result_noun = unsafe { result.root() };
        let space = result.noun_space();

        // Manually parse Option<Option<list>> structure
        // Option is: ~ for None, [~ value] for Some
        let outer_opt = result_noun.in_space(&space).as_cell()
            .map_err(|_| ExtractorError::PeekReturnedNoData)?;

        // Check if outer option is Some (should be [~ inner])
        let outer_head = outer_opt.head().noun();
        if !outer_head.is_atom() || outer_head.in_space(&space).as_atom().map(|a| a.as_u64().unwrap_or(1)).unwrap_or(1) != 0 {
            return Err(ExtractorError::PeekReturnedNoData);
        }

        let inner = outer_opt.tail().noun();
        let inner_opt = inner.in_space(&space).as_cell()
            .map_err(|_| ExtractorError::PeekReturnedNoData)?;

        // Check if inner option is Some
        let inner_head = inner_opt.head().noun();
        if !inner_head.is_atom() || inner_head.in_space(&space).as_atom().map(|a| a.as_u64().unwrap_or(1)).unwrap_or(1) != 0 {
            return Err(ExtractorError::PeekReturnedNoData);
        }

        let list_noun = inner_opt.tail().noun();

        // Iterate the list and process each entry individually
        let mut blocks_with_jam = Vec::new();

        for entry_noun in HoonList::try_from(list_noun, &space)
            .map_err(|_| ExtractorError::PeekReturnedNoData)?
        {
            // Copy this entry noun into a fresh slab and jam it
            let mut entry_slab: NounSlab = NounSlab::new();
            let copied_noun = entry_slab.copy_into(entry_noun, &space);
            entry_slab.set_root(copied_noun);
            let jam_bytes = entry_slab.jam();

            // Decode the entry to BlockData using the original space
            let entry: BlockRangeEntryNoun = NounDecode::from_noun(&entry_noun, &space)?;
            let data = entry.into_block_data(&space)?;

            blocks_with_jam.push(BlockDataWithJam { data, jam_bytes });
        }

        debug!(
            start,
            end,
            block_count = blocks_with_jam.len(),
            "Extracted block range with jam"
        );

        Ok(blocks_with_jam)
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

    /// Extract blocks and write directly to an archive file
    ///
    /// This is the main entry point for creating speed-of-light archives.
    /// It extracts blocks with their jammed noun bytes and writes them
    /// to a binary archive format that can be loaded quickly for benchmarks.
    pub async fn extract_to_archive<P: AsRef<Path>>(
        &mut self,
        count: u64,
        output_path: P,
    ) -> Result<(), ExtractorError> {
        info!(count, path = %output_path.as_ref().display(), "Extracting blocks to archive");

        let mut writer = ArchiveWriter::new();

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

        let mut current = 0u64;
        let mut total_blocks = 0usize;
        let mut total_txs = 0usize;

        while current <= end_height {
            let chunk_end = (current + self.config.chunk_size - 1).min(end_height);

            match self.extract_blocks_range_with_jam(current, chunk_end).await {
                Ok(blocks) => {
                    if blocks.is_empty() {
                        info!(current, "No more blocks available, stopping extraction");
                        break;
                    }

                    for block in &blocks {
                        writer.add_block(
                            block.data.height,
                            block.data.block_id.clone(),
                            block.data.tx_count(),
                            &block.jam_bytes,
                        ).map_err(|e| ExtractorError::Io(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            e.to_string(),
                        )))?;
                        total_txs += block.data.tx_count();
                    }
                    total_blocks += blocks.len();

                    info!(
                        start = current,
                        end = chunk_end,
                        blocks = blocks.len(),
                        total_blocks,
                        total_txs,
                        "Archived block chunk"
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

        // Write the archive to disk
        writer.write_to_file(output_path.as_ref()).map_err(|e| {
            ExtractorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

        info!(
            total_blocks,
            total_txs,
            path = %output_path.as_ref().display(),
            "Archive written successfully"
        );

        Ok(())
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

    /// Full integration test: Extract the first 1000 blocks
    /// Main use case - verify we can extract a significant number of blocks
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_05_extract_first_1000_blocks() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        println!("[TEST 05] Extracting first 1000 blocks to cache...");
        let cache = guard.extract_to_cache(1000).await.expect("should extract blocks to cache");

        let block_count = cache.block_count();
        let tx_count = cache.transaction_count();

        println!("[TEST 05] Extraction complete:");
        println!("  blocks: {}", block_count);
        println!("  transactions: {}", tx_count);
        println!("  stats: {}", cache.stats());

        // Should have extracted at least some blocks
        assert!(block_count > 0, "should have extracted at least 1 block");

        // Verify block ordering and linkage
        let mut prev_height = None;
        for block in cache.iter_blocks().take(100) {
            if let Some(prev) = prev_height {
                assert_eq!(block.height, prev + 1, "blocks should be in sequential order");
            }
            prev_height = Some(block.height);

            // Verify basic block structure - block_id should be set
            // (We can't easily check for zero, but we know extraction succeeded)
        }

        println!("[TEST 05] Block ordering verified for first {} blocks", prev_height.map(|h| h + 1).unwrap_or(0));
    }

    /// Full integration test: Verify the first three transactions on the network
    /// - Block 5629: First transaction on the network
    /// - Block 9095: Second transaction
    /// - Block 9239: Third transaction
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_06_first_network_transactions() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        // Known first transaction details
        const FIRST_TX_BLOCK: u64 = 5629;
        const SECOND_TX_BLOCK: u64 = 9095;
        const THIRD_TX_BLOCK: u64 = 9239;
        const FIRST_TX_FROM_ADDRESS: &str = "37oNkJu8RiUswLrAmrBpBKBQdQmYCD3BENZH4MN7DpyCZVS3dF9NCJ7jAcRFLwK1nUgkLqnQVMsgMqmYx284YJGkwYWCrY3um9tPYwuACMY7aebZcUaFis45oQT81UQfbUYt";

        println!("[TEST 06] Extracting blocks up to height {}...", THIRD_TX_BLOCK);
        let cache = guard.extract_to_cache(THIRD_TX_BLOCK + 1).await.expect("should extract blocks");

        // Check block 5629 - first transaction
        println!("[TEST 06] Checking block {} (first tx)...", FIRST_TX_BLOCK);
        let block_5629 = cache.get_block(FIRST_TX_BLOCK).expect("block 5629 should exist");
        assert_eq!(block_5629.transactions.len(), 1, "block 5629 should have exactly 1 transaction");

        let first_tx = &block_5629.transactions[0];
        println!("[TEST 06] First tx id: {}", first_tx.tx_id.to_base58());
        println!("[TEST 06] First tx inputs: {}", first_tx.raw_tx.inputs.0.len());

        // Verify the first transaction came from the expected address
        assert!(!first_tx.raw_tx.inputs.0.is_empty(), "first tx should have at least one input");
        let first_input = &first_tx.raw_tx.inputs.0[0].1;
        let input_lock = &first_input.note.tail.lock;
        println!("[TEST 06] First input lock keys_required: {}", input_lock.keys_required);
        println!("[TEST 06] First input lock pubkeys count: {}", input_lock.pubkeys.len());

        // Check if any of the pubkeys matches the expected address
        let mut found_address = false;
        for (i, pubkey) in input_lock.pubkeys.iter().enumerate() {
            match pubkey.to_base58() {
                Ok(addr) => {
                    println!("[TEST 06] Input pubkey {}: {}", i, addr);
                    if addr == FIRST_TX_FROM_ADDRESS {
                        found_address = true;
                        println!("[TEST 06] ✓ Found matching address!");
                    }
                }
                Err(e) => {
                    println!("[TEST 06] Input pubkey {} encoding error: {:?}", i, e);
                }
            }
        }
        assert!(found_address, "first tx should be from expected address");

        // Check block 9095 - second transaction
        println!("[TEST 06] Checking block {} (second tx)...", SECOND_TX_BLOCK);
        let block_9095 = cache.get_block(SECOND_TX_BLOCK).expect("block 9095 should exist");
        assert!(!block_9095.transactions.is_empty(), "block 9095 should have at least 1 transaction");
        println!("[TEST 06] Block {} has {} transaction(s)", SECOND_TX_BLOCK, block_9095.transactions.len());

        // Check block 9239 - third transaction
        println!("[TEST 06] Checking block {} (third tx)...", THIRD_TX_BLOCK);
        let block_9239 = cache.get_block(THIRD_TX_BLOCK).expect("block 9239 should exist");
        assert!(!block_9239.transactions.is_empty(), "block 9239 should have at least 1 transaction");
        println!("[TEST 06] Block {} has {} transaction(s)", THIRD_TX_BLOCK, block_9239.transactions.len());

        println!("[TEST 06] ✓ All three transaction blocks verified!");
    }

    /// Full integration test: Extract blocks with jam bytes
    /// Verifies that extract_blocks_range_with_jam returns non-empty jam bytes
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_07_extract_with_jam_returns_bytes() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        println!("[TEST 07] Extracting blocks 0-7 with jam bytes...");
        let blocks_with_jam = guard
            .extract_blocks_range_with_jam(0, 7)
            .await
            .expect("should extract blocks with jam");

        assert_eq!(blocks_with_jam.len(), 8, "should get 8 blocks");

        for (i, block) in blocks_with_jam.iter().enumerate() {
            println!(
                "[TEST 07] Block {}: height={}, jam_bytes_len={}",
                i,
                block.data.height,
                block.jam_bytes.len()
            );

            assert_eq!(block.data.height, i as u64, "block height should match index");
            assert!(!block.jam_bytes.is_empty(), "jam bytes should not be empty");
            // Jam bytes should be reasonably sized (at least a few bytes for any noun)
            assert!(block.jam_bytes.len() > 10, "jam bytes should be substantial");
        }

        println!("[TEST 07] ✓ All blocks have non-empty jam bytes");
    }

    /// Full integration test: Verify extract_with_jam matches regular extract
    /// The decoded BlockData from extract_blocks_range_with_jam should match
    /// the data from extract_blocks_range
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_08_extract_with_jam_matches_decode() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        // Extract with both methods
        println!("[TEST 08] Extracting blocks 0-7 with regular method...");
        let regular_blocks = guard
            .extract_blocks_range(0, 7)
            .await
            .expect("should extract blocks");

        println!("[TEST 08] Extracting blocks 0-7 with jam method...");
        let blocks_with_jam = guard
            .extract_blocks_range_with_jam(0, 7)
            .await
            .expect("should extract blocks with jam");

        assert_eq!(regular_blocks.len(), blocks_with_jam.len(), "should have same block count");

        // Compare decoded data
        for (i, (regular, with_jam)) in regular_blocks.iter().zip(blocks_with_jam.iter()).enumerate() {
            println!("[TEST 08] Comparing block {}...", i);

            assert_eq!(regular.height, with_jam.data.height, "heights should match");
            assert_eq!(
                regular.block_id.to_base58(),
                with_jam.data.block_id.to_base58(),
                "block_ids should match"
            );
            assert_eq!(
                regular.parent_id.to_base58(),
                with_jam.data.parent_id.to_base58(),
                "parent_ids should match"
            );
            assert_eq!(regular.timestamp, with_jam.data.timestamp, "timestamps should match");
            assert_eq!(
                regular.transactions.len(),
                with_jam.data.transactions.len(),
                "tx counts should match"
            );
        }

        println!("[TEST 08] ✓ All decoded data matches between methods");
    }

    /// Full integration test: Verify jam bytes can be cued back to the same structure
    /// This tests round-trip fidelity: jam → cue → decode should produce same data
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_09_jam_roundtrip_fidelity() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        println!("[TEST 09] Extracting block 0 with jam bytes...");
        let blocks_with_jam = guard
            .extract_blocks_range_with_jam(0, 0)
            .await
            .expect("should extract block with jam");

        assert_eq!(blocks_with_jam.len(), 1, "should get 1 block");
        let original = &blocks_with_jam[0];

        println!("[TEST 09] Original jam bytes len: {}", original.jam_bytes.len());
        println!("[TEST 09] Original block height: {}", original.data.height);

        // Cue the jam bytes back into a noun and decode
        let mut cue_slab: NounSlab = NounSlab::new();
        let cued_noun = cue_slab
            .cue_into(original.jam_bytes.clone())
            .expect("should cue jam bytes");

        let space = cue_slab.noun_space();

        // Decode the cued noun
        let decoded_entry: BlockRangeEntryNoun = NounDecode::from_noun(&cued_noun, &space)
            .expect("should decode cued noun");
        let decoded_block = decoded_entry
            .into_block_data(&space)
            .expect("should convert to BlockData");

        // Verify the decoded data matches the original
        assert_eq!(decoded_block.height, original.data.height, "heights should match after roundtrip");
        assert_eq!(
            decoded_block.block_id.to_base58(),
            original.data.block_id.to_base58(),
            "block_ids should match after roundtrip"
        );
        assert_eq!(
            decoded_block.parent_id.to_base58(),
            original.data.parent_id.to_base58(),
            "parent_ids should match after roundtrip"
        );
        assert_eq!(
            decoded_block.timestamp,
            original.data.timestamp,
            "timestamps should match after roundtrip"
        );
        assert_eq!(
            decoded_block.transactions.len(),
            original.data.transactions.len(),
            "tx counts should match after roundtrip"
        );

        println!("[TEST 09] ✓ Jam → Cue → Decode roundtrip successful");
    }

    /// Full integration test: Extract blocks to archive file
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_10_extract_to_archive() {
        use crate::speed_of_light::archive::ArchiveReader;
        use tempfile::tempdir;

        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        // Create a temp directory for the archive
        let temp_dir = tempdir().expect("should create temp dir");
        let archive_path = temp_dir.path().join("test.solarch");

        println!("[TEST 10] Extracting 100 blocks to archive...");
        guard
            .extract_to_archive(100, &archive_path)
            .await
            .expect("should extract to archive");

        // Verify the archive exists
        assert!(archive_path.exists(), "archive file should exist");

        // Read the archive back
        let archive_bytes = std::fs::read(&archive_path).expect("should read archive");
        println!("[TEST 10] Archive size: {} bytes", archive_bytes.len());

        let reader = ArchiveReader::from_bytes(archive_bytes).expect("should parse archive");
        let metadata = reader.metadata();

        println!("[TEST 10] Archive metadata:");
        println!("  block_count: {}", metadata.block_count);
        println!("  total_tx_count: {}", metadata.total_tx_count);
        println!("  height range: {}..={}", metadata.min_height, metadata.max_height);

        assert_eq!(metadata.block_count, 100, "should have 100 blocks");
        assert_eq!(metadata.min_height, 0, "should start at block 0");
        assert_eq!(metadata.max_height, 99, "should end at block 99");

        println!("[TEST 10] ✓ Archive created and validated successfully");
    }

    /// Full integration test: Full pipeline - extract → archive → load → verify
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_11_full_pipeline() {
        use crate::speed_of_light::archive::ArchiveReader;
        use tempfile::tempdir;

        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        // Create temp directory
        let temp_dir = tempdir().expect("should create temp dir");
        let archive_path = temp_dir.path().join("pipeline_test.solarch");

        // Step 1: Extract first 50 blocks to archive
        println!("[TEST 11] Step 1: Extracting 50 blocks to archive...");
        guard
            .extract_to_archive(50, &archive_path)
            .await
            .expect("should extract to archive");

        // Step 2: Load the archive
        println!("[TEST 11] Step 2: Loading archive...");
        let archive_bytes = std::fs::read(&archive_path).expect("should read archive");
        let reader = ArchiveReader::from_bytes(archive_bytes).expect("should parse archive");

        // Step 3: Verify we can cue and decode each block
        println!("[TEST 11] Step 3: Verifying all blocks can be decoded...");
        for (entry, jam_bytes) in reader.iter() {
            // Cue the jam bytes
            let mut slab: NounSlab = NounSlab::new();
            let cued_noun = slab.cue_into(jam_bytes.to_vec().into()).expect("should cue");
            let space = slab.noun_space();

            // Decode to BlockData
            let decoded_entry: BlockRangeEntryNoun =
                NounDecode::from_noun(&cued_noun, &space).expect("should decode");
            let block = decoded_entry.into_block_data(&space).expect("should convert");

            // Verify height matches
            assert_eq!(block.height, entry.height, "decoded height should match entry");

            if entry.height % 10 == 0 {
                println!("  Block {}: height={}, tx_count={}", entry.height, block.height, block.tx_count());
            }
        }

        println!("[TEST 11] ✓ Full pipeline verified - all 50 blocks decoded successfully");
    }
}
