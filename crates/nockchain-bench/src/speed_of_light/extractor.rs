//! Block extraction from a running kernel via peek

use std::path::{Path, PathBuf};

use bytes::Bytes;
use nockapp::nockapp::save::SaveableCheckpoint;
use nockapp::nockapp::wire::WireRepr;
use nockapp::nockapp::NockApp;
use nockapp::noun::slab::NounSlab;
use nockchain_math::noun_ext::NounMathExt;
use nockchain_math::structs::{HoonList, HoonMapIter};
use nockchain_types::tx_engine::common::Hash;
use nockvm::noun::{Noun, SIG};
use noun_serde::NounDecode;
use thiserror::Error;
use tracing::{debug, info};

use super::archive::{MempoolTxEntry, SolArchiveReader, SolArchiveWriter};
use super::checkpoint::{load_checkpoint, CheckpointLoadError};
use super::kernel_utils::{
    init_nockapp, peek_heaviest_chain, sol_replay_wire, KernelInitError, PeekChainError,
};
use super::poke::build_poke_slab_from_jam;
use super::types::{summarize_archive_entry, ArchiveBlockSummary, SolHeight};

#[derive(Debug, Clone)]
struct ArchiveBlockWithJam {
    summary: ArchiveBlockSummary,
    jam_bytes: Bytes,
}

/// Phase of archive extraction progress reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveExtractionPhase {
    /// Extracting block jam blobs and writing archive entries.
    Blocks,
    /// Replaying archived blocks to capture mempool snapshots.
    MempoolReplay,
    /// Finished writing the archive file.
    Complete,
}

/// Progress update emitted during archive extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchiveExtractionProgress {
    /// Current extraction phase.
    pub phase: ArchiveExtractionPhase,
    /// Number of blocks archived so far.
    pub blocks_archived: usize,
    /// Requested block target for extraction.
    pub target_blocks: u64,
    /// Number of transactions archived so far.
    pub txs_archived: usize,
    /// Inclusive start height for the current chunk (blocks phase).
    pub chunk_start: Option<u64>,
    /// Inclusive end height for the current chunk (blocks phase).
    pub chunk_end: Option<u64>,
    /// Number of blocks in the current chunk (blocks phase).
    pub chunk_blocks: usize,
    /// Number of mempool snapshots captured so far (mempool phase).
    pub mempool_snapshots_done: usize,
    /// Total mempool snapshots expected (mempool phase).
    pub mempool_snapshots_total: usize,
}

#[derive(Debug, Error)]
pub enum ExtractorError {
    #[error("Archive error: {0}")]
    Archive(#[from] super::archive::ArchiveError),

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

    #[error("Entry decode error: {0}")]
    EntryDecode(String),

    #[error("Noun decode error: {0}")]
    NounDecode(#[from] noun_serde::NounDecodeError),

    #[error("NockApp error: {0}")]
    NockApp(#[from] nockapp::nockapp::NockAppError),

    #[error("Kernel init error: {0}")]
    KernelInit(#[from] KernelInitError),

    #[error("Chain height peek error: {0}")]
    ChainPeek(#[from] PeekChainError),

    #[error("Invalid extraction range: start={start} end={end}")]
    InvalidRange { start: u64, end: u64 },

    #[error("Requested start height {start} exceeds chain tip {tip}")]
    StartAboveChainTip { start: u64, tip: u64 },
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
    /// Whether to include mempool snapshots in the archive
    pub include_mempool: bool,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            checkpoint_path: "checkpoint_1000.chkjam".to_string(),
            kernel_path: "assets/dumb.jam".to_string(),
            block_count: 1000,
            chunk_size: 8,
            work_dir: PathBuf::from("."),
            include_mempool: false,
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
        info!(
            checkpoint = %self.config.checkpoint_path,
            kernel = %self.config.kernel_path,
            "Initializing block extractor"
        );

        // Load checkpoint
        let loaded = load_checkpoint(&self.config.checkpoint_path)?;
        info!(event_num = loaded.event_num, "Loaded checkpoint");

        // Create SaveableCheckpoint from loaded data
        let checkpoint = SaveableCheckpoint {
            ker_hash: loaded.ker_hash,
            event_num: loaded.event_num,
            state: loaded.state,
            cold: loaded.cold,
        };

        let work_dir = self.config.work_dir.clone();

        let nockapp = init_nockapp(
            std::path::Path::new(&self.config.kernel_path),
            Some(checkpoint),
            &work_dir,
            true,
        )
        .await?;

        info!("NockApp initialized successfully");
        self.nockapp = Some(nockapp);
        Ok(())
    }

    /// Get the current chain tip height
    pub async fn get_chain_height(&mut self) -> Result<(u64, Hash), ExtractorError> {
        let nockapp = self.nockapp_mut()?;

        let (height, hash) = peek_heaviest_chain(nockapp)
            .await?
            .ok_or(ExtractorError::PeekReturnedNoData)?;

        Ok((height.0 .0, hash))
    }

    async fn poke_block_jam_bytes(
        &mut self,
        jam_bytes: &[u8],
        wire: &WireRepr,
    ) -> Result<(), ExtractorError> {
        let nockapp = self.nockapp_mut()?;

        let poke_slab = build_poke_slab_from_jam(jam_bytes).map_err(ExtractorError::EntryDecode)?;

        nockapp.poke(wire.clone(), poke_slab).await?;
        Ok(())
    }

    async fn peek_raw_transactions(&mut self) -> Result<Vec<MempoolTxEntry>, ExtractorError> {
        let nockapp = self.nockapp_mut()?;

        let mut path_slab = NounSlab::new();
        let tag = nockapp::utils::make_tas(&mut path_slab, "raw-transactions").as_noun();
        let path_noun = nockvm::noun::T(&mut path_slab, &[tag, SIG]);
        path_slab.set_root(path_noun);

        let result = nockapp.peek(path_slab).await?;
        let result_noun = unsafe { result.root() };

        let map_noun = match decode_unit_unit(*result_noun) {
            Some(noun) => noun,
            None => return Ok(Vec::new()),
        };

        if let Ok(atom) = map_noun.as_atom() {
            if atom.as_u64().unwrap_or(1) == 0 {
                return Ok(Vec::new());
            }
        }

        let mut entries = Vec::new();
        for entry in HoonMapIter::from(map_noun) {
            let [key, value] = match entry.uncell() {
                Ok(kv) => kv,
                Err(_) => continue,
            };

            let tx_id = Hash::from_noun(&key)?;
            let value_cell = value
                .as_cell()
                .map_err(|_| ExtractorError::EntryDecode("raw-tx entry not a cell".to_string()))?;
            let heard_at_noun = value_cell.tail();
            let heard_at = u64::from_noun(&heard_at_noun)?;

            entries.push(MempoolTxEntry {
                tx_id,
                heard_at: SolHeight(heard_at),
            });
        }

        Ok(entries)
    }

    async fn populate_mempool_snapshots_with_progress<F>(
        &mut self,
        writer: &mut SolArchiveWriter,
        mut on_progress: F,
    ) -> Result<(), ExtractorError>
    where
        F: FnMut(usize, usize, SolHeight),
    {
        let reader = SolArchiveReader::from_bytes(writer.to_bytes()?)?;
        let total = reader.metadata().block_count as usize;

        let wire = sol_replay_wire();

        for (idx, (entry, jam_bytes)) in reader.iter().enumerate() {
            self.poke_block_jam_bytes(jam_bytes, &wire).await?;
            let snapshot = self.peek_raw_transactions().await?;
            writer.add_mempool_snapshot(entry.height, &snapshot)?;
            on_progress(idx + 1, total, entry.height);
        }

        Ok(())
    }

    /// Extract blocks for archive writing without decoding historical page/tx shapes.
    async fn extract_archive_blocks_range_with_jam(
        &mut self,
        start: u64,
        end: u64,
    ) -> Result<Vec<ArchiveBlockWithJam>, ExtractorError> {
        let nockapp = self.nockapp_mut()?;

        debug!(start, end, "Extracting archive block range with jam");

        let mut path_slab = NounSlab::new();
        let tag = nockapp::utils::make_tas(&mut path_slab, "heaviest-chain-blocks-range").as_noun();
        let start_noun = nockvm::noun::D(start);
        let end_noun = nockvm::noun::D(end);
        let path_noun = nockvm::noun::T(&mut path_slab, &[tag, start_noun, end_noun, SIG]);
        path_slab.set_root(path_noun);

        let result = nockapp.peek(path_slab).await?;
        let result_noun = unsafe { result.root() };

        let outer_opt = result_noun
            .as_cell()
            .map_err(|_| ExtractorError::PeekReturnedNoData)?;
        let outer_head = outer_opt.head();
        if !outer_head.is_atom() || u64::from_noun(&outer_head).ok().unwrap_or(1) != 0 {
            return Err(ExtractorError::PeekReturnedNoData);
        }

        let inner = outer_opt.tail();
        let inner_opt = inner
            .as_cell()
            .map_err(|_| ExtractorError::PeekReturnedNoData)?;
        let inner_head = inner_opt.head();
        if !inner_head.is_atom() || u64::from_noun(&inner_head).ok().unwrap_or(1) != 0 {
            return Err(ExtractorError::PeekReturnedNoData);
        }

        let list_noun = inner_opt.tail();
        let mut blocks_with_jam = Vec::new();

        for entry_noun in
            HoonList::try_from(list_noun).map_err(|_| ExtractorError::PeekReturnedNoData)?
        {
            let summary = summarize_archive_entry(entry_noun).map_err(|e| {
                ExtractorError::EntryDecode(format!(
                    "range {start}..={end}: failed to summarize archive block-range entry noun: {e}"
                ))
            })?;

            let mut entry_slab: NounSlab = NounSlab::new();
            let copied_noun = entry_slab.copy_into(entry_noun);
            entry_slab.set_root(copied_noun);
            let jam_bytes = entry_slab.jam();

            blocks_with_jam.push(ArchiveBlockWithJam { summary, jam_bytes });
        }

        debug!(
            start,
            end,
            block_count = blocks_with_jam.len(),
            "Extracted archive block range with jam"
        );

        Ok(blocks_with_jam)
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
        self.extract_to_archive_with_progress(count, output_path, |_| {})
            .await
    }

    /// Extract an inclusive block-height range directly to an archive file.
    pub async fn extract_range_to_archive<P: AsRef<Path>>(
        &mut self,
        start_height: u64,
        end_height: u64,
        output_path: P,
    ) -> Result<(), ExtractorError> {
        self.extract_range_to_archive_with_progress(start_height, end_height, output_path, |_| {})
            .await
    }

    /// Extract blocks and write directly to an archive file with progress callbacks.
    pub async fn extract_to_archive_with_progress<P, F>(
        &mut self,
        count: u64,
        output_path: P,
        on_progress: F,
    ) -> Result<(), ExtractorError>
    where
        P: AsRef<Path>,
        F: FnMut(ArchiveExtractionProgress),
    {
        if count == 0 {
            return Err(ExtractorError::InvalidRange { start: 0, end: 0 });
        }
        let end_height = count.saturating_sub(1);
        self.extract_range_to_archive_with_progress(0, end_height, output_path, on_progress)
            .await
    }

    /// Extract an inclusive block-height range directly to an archive file with progress callbacks.
    pub async fn extract_range_to_archive_with_progress<P, F>(
        &mut self,
        start_height: u64,
        end_height: u64,
        output_path: P,
        mut on_progress: F,
    ) -> Result<(), ExtractorError>
    where
        P: AsRef<Path>,
        F: FnMut(ArchiveExtractionProgress),
    {
        if start_height > end_height {
            return Err(ExtractorError::InvalidRange {
                start: start_height,
                end: end_height,
            });
        }

        info!(
            start_height,
            end_height,
            path = %output_path.as_ref().display(),
            "Extracting block range to archive"
        );

        let mut writer = SolArchiveWriter::new();
        let requested_target_blocks = end_height.saturating_sub(start_height).saturating_add(1);

        // Try to get chain height. If available, cap the end to the chain tip.
        let effective_end_height = match self.get_chain_height().await {
            Ok((chain_height, _)) => {
                info!(chain_height, "Chain height available");
                if start_height > chain_height {
                    return Err(ExtractorError::StartAboveChainTip {
                        start: start_height,
                        tip: chain_height,
                    });
                }
                end_height.min(chain_height)
            }
            Err(ExtractorError::PeekReturnedNoData) => {
                info!("Chain height unavailable, will extract until empty results");
                end_height
            }
            Err(e) => return Err(e),
        };

        let target_blocks = effective_end_height
            .saturating_sub(start_height)
            .saturating_add(1);
        let mut current = start_height;
        let mut total_blocks = 0usize;
        let mut total_txs = 0usize;

        while current <= effective_end_height {
            let chunk_end = (current + self.config.chunk_size - 1).min(effective_end_height);

            match self
                .extract_archive_blocks_range_with_jam(current, chunk_end)
                .await
            {
                Ok(blocks) => {
                    if blocks.is_empty() {
                        info!(current, "No more blocks available, stopping extraction");
                        break;
                    }

                    for block in &blocks {
                        writer
                            .add_block(
                                block.summary.height,
                                block.summary.block_id.clone(),
                                block.summary.tx_count,
                                block.summary.proof_version,
                                &block.jam_bytes,
                            )
                            .map_err(|e| {
                                ExtractorError::Io(std::io::Error::new(
                                    std::io::ErrorKind::Other,
                                    e.to_string(),
                                ))
                            })?;
                        total_txs += block.summary.tx_count;
                    }
                    total_blocks += blocks.len();
                    on_progress(ArchiveExtractionProgress::blocks(
                        total_blocks,
                        target_blocks,
                        total_txs,
                        current,
                        chunk_end,
                        blocks.len(),
                    ));

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

        if self.config.include_mempool {
            info!("Replaying blocks to capture mempool snapshots");
            self.populate_mempool_snapshots_with_progress(&mut writer, |done, total, _height| {
                on_progress(ArchiveExtractionProgress::mempool(
                    total_blocks, target_blocks, total_txs, done, total,
                ));
            })
            .await?;
        }

        // Write the archive to disk
        writer.write_to_file(output_path.as_ref()).map_err(|e| {
            ExtractorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;
        on_progress(ArchiveExtractionProgress::complete(
            total_blocks, target_blocks, total_txs,
        ));

        if total_blocks == 0 && requested_target_blocks > 0 {
            return Err(ExtractorError::StartAboveChainTip {
                start: start_height,
                tip: effective_end_height,
            });
        }

        info!(
            total_blocks,
            total_txs,
            path = %output_path.as_ref().display(),
            "Archive written successfully"
        );

        Ok(())
    }

    fn nockapp_mut(&mut self) -> Result<&mut NockApp, ExtractorError> {
        self.nockapp
            .as_mut()
            .ok_or_else(|| ExtractorError::KernelLoad("NockApp not initialized".to_string()))
    }
}

impl ArchiveExtractionProgress {
    fn blocks(
        blocks_archived: usize,
        target_blocks: u64,
        txs_archived: usize,
        chunk_start: u64,
        chunk_end: u64,
        chunk_blocks: usize,
    ) -> Self {
        Self {
            phase: ArchiveExtractionPhase::Blocks,
            blocks_archived,
            target_blocks,
            txs_archived,
            chunk_start: Some(chunk_start),
            chunk_end: Some(chunk_end),
            chunk_blocks,
            mempool_snapshots_done: 0,
            mempool_snapshots_total: 0,
        }
    }

    fn mempool(
        blocks_archived: usize,
        target_blocks: u64,
        txs_archived: usize,
        mempool_snapshots_done: usize,
        mempool_snapshots_total: usize,
    ) -> Self {
        Self {
            phase: ArchiveExtractionPhase::MempoolReplay,
            blocks_archived,
            target_blocks,
            txs_archived,
            chunk_start: None,
            chunk_end: None,
            chunk_blocks: 0,
            mempool_snapshots_done,
            mempool_snapshots_total,
        }
    }

    fn complete(blocks_archived: usize, target_blocks: u64, txs_archived: usize) -> Self {
        Self {
            phase: ArchiveExtractionPhase::Complete,
            blocks_archived,
            target_blocks,
            txs_archived,
            chunk_start: None,
            chunk_end: None,
            chunk_blocks: 0,
            mempool_snapshots_done: 0,
            mempool_snapshots_total: 0,
        }
    }
}

fn decode_unit(noun: Noun) -> Option<Noun> {
    if let Ok(atom) = noun.as_atom() {
        if atom.as_u64().ok()? == 0 {
            return None;
        }
    }

    let cell = noun.as_cell().ok()?;
    let head = cell.head();
    let head_atom = head.as_atom().ok()?;
    if head_atom.as_u64().ok()? != 0 {
        return None;
    }

    Some(cell.tail())
}

fn decode_unit_unit(noun: Noun) -> Option<Noun> {
    let inner = decode_unit(noun)?;
    decode_unit(inner)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::tempdir;
    use tokio::sync::{Mutex, OnceCell};

    use super::*;
    use crate::speed_of_light::archive::SolArchiveReader;

    // Path helpers - tests run from crate root, so we need to go up to repo root
    fn checkpoint_path() -> String {
        std::env::var("SOL_CHECKPOINT_PATH")
            .unwrap_or_else(|_| "../../checkpoint_1000.chkjam".to_string())
    }

    fn kernel_path() -> String {
        std::env::var("SOL_KERNEL_PATH").unwrap_or_else(|_| "../../assets/dumb.jam".to_string())
    }

    // Shared extractor for integration tests - avoids reinitializing for each test
    static SHARED_EXTRACTOR: OnceCell<Arc<Mutex<BlockExtractor>>> = OnceCell::const_new();

    async fn initialized_extractor(include_mempool: bool) -> BlockExtractor {
        let config = ExtractorConfig {
            checkpoint_path: checkpoint_path(),
            kernel_path: kernel_path(),
            block_count: 1000,
            chunk_size: 8,
            work_dir: PathBuf::from("."),
            include_mempool,
        };
        let mut extractor = BlockExtractor::new(config);
        extractor
            .initialize()
            .await
            .expect("should initialize NockApp");
        extractor
    }

    async fn get_shared_extractor() -> Arc<Mutex<BlockExtractor>> {
        SHARED_EXTRACTOR
            .get_or_init(|| async {
                println!("=== Initializing shared BlockExtractor ===");
                let extractor = initialized_extractor(false).await;
                println!("=== Shared BlockExtractor ready ===");
                Arc::new(Mutex::new(extractor))
            })
            .await
            .clone()
    }

    // ==================== QUICK TESTS ====================
    // These tests don't require kernel initialization and run fast

    /// Test ExtractorConfig defaults
    #[test]
    fn test_extractor_config_defaults() {
        let config = ExtractorConfig::default();
        assert_eq!(config.block_count, 1000);
        assert_eq!(config.chunk_size, 8);
        assert!(!config.include_mempool);
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
            include_mempool: false,
        };
        let extractor = BlockExtractor::new(config);
        assert!(
            extractor.nockapp.is_none(),
            "nockapp should be None before initialize"
        );
    }

    #[tokio::test]
    async fn test_extract_to_archive_rejects_zero_count() {
        let mut extractor = BlockExtractor::new(ExtractorConfig::default());
        let temp_dir = tempdir().expect("should create temp dir");
        let err = extractor
            .extract_to_archive(0, temp_dir.path().join("empty.solarch"))
            .await
            .expect_err("zero-count archive extraction should fail");
        assert!(matches!(
            err,
            ExtractorError::InvalidRange { start: 0, end: 0 }
        ));
    }

    #[tokio::test]
    async fn test_extract_range_to_archive_rejects_invalid_range() {
        let mut extractor = BlockExtractor::new(ExtractorConfig::default());
        let temp_dir = tempdir().expect("should create temp dir");
        let err = extractor
            .extract_range_to_archive(8, 7, temp_dir.path().join("invalid.solarch"))
            .await
            .expect_err("descending archive extraction range should fail");
        assert!(matches!(
            err,
            ExtractorError::InvalidRange { start: 8, end: 7 }
        ));
    }

    // ==================== INTEGRATION TESTS ====================
    // These tests require full kernel initialization.
    // Run with: cargo test -p nockchain-bench integration_test_ -- --ignored --test-threads=1

    /// Full integration test: Initialize extractor with kernel and checkpoint
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_01_extractor_initializes() {
        let extractor = get_shared_extractor().await;
        let guard = extractor.lock().await;
        assert!(
            guard.nockapp.is_some(),
            "nockapp should be Some after initialize"
        );
        println!("Extractor initialized successfully");
    }

    /// Full integration test: Get chain height via peek.
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
                assert!(
                    height > 0,
                    "chain height should be > 0 for a real checkpoint"
                );
            }
            Err(ExtractorError::PeekReturnedNoData) => {
                println!(
                    "[TEST 02] Chain height not available in checkpoint (expected for some states)"
                );
                println!("[TEST 02] Archive extraction can still proceed via range peek");
            }
            Err(e) => {
                println!(
                    "[TEST 02] get_chain_height failed with unexpected error: {:?}",
                    e
                );
                panic!("unexpected error: {:?}", e);
            }
        }
    }

    /// Full integration test: Extract blocks to archive file.
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_03_extract_to_archive() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        // Create a temp directory for the archive
        let temp_dir = tempdir().expect("should create temp dir");
        let archive_path = temp_dir.path().join("test.solarch");

        println!("[TEST 03] Extracting 100 blocks to archive...");
        guard
            .extract_to_archive(100, &archive_path)
            .await
            .expect("should extract to archive");

        // Verify the archive exists
        assert!(archive_path.exists(), "archive file should exist");

        // Read the archive back
        let archive_bytes = std::fs::read(&archive_path).expect("should read archive");
        println!("[TEST 03] Archive size: {} bytes", archive_bytes.len());

        let reader = SolArchiveReader::from_bytes(archive_bytes).expect("should parse archive");
        let metadata = reader.metadata();

        println!("[TEST 03] Archive metadata:");
        println!("  block_count: {}", metadata.block_count);
        println!("  total_tx_count: {}", metadata.total_tx_count);
        println!(
            "  height range: {}..={}",
            metadata.min_height.as_u64(),
            metadata.max_height.as_u64()
        );

        assert_eq!(metadata.block_count, 100, "should have 100 blocks");
        assert_eq!(metadata.min_height, SolHeight(0), "should start at block 0");
        assert_eq!(metadata.max_height, SolHeight(99), "should end at block 99");

        println!("[TEST 03] ✓ Archive created and validated successfully");
    }

    /// Archive regression test for the first two historical chunks.
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_04_full_pipeline_archive_roundtrip() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        // Create temp directory
        let temp_dir = tempdir().expect("should create temp dir");
        let archive_path = temp_dir.path().join("pipeline_test.solarch");
        let mut progress = Vec::new();

        println!("[TEST 04] Extracting blocks 0-15 to archive...");
        guard
            .extract_range_to_archive_with_progress(0, 15, &archive_path, |update| {
                progress.push(update);
            })
            .await
            .expect("should extract to archive");

        let block_progress: Vec<_> = progress
            .iter()
            .copied()
            .filter(|update| update.phase == ArchiveExtractionPhase::Blocks)
            .collect();
        assert_eq!(
            block_progress.len(),
            2,
            "0..15 should archive in two chunks"
        );
        assert_eq!(block_progress[0].chunk_start, Some(0));
        assert_eq!(block_progress[0].chunk_end, Some(7));
        assert_eq!(block_progress[1].chunk_start, Some(8));
        assert_eq!(block_progress[1].chunk_end, Some(15));

        println!("[TEST 04] Loading archive...");
        let archive_bytes = std::fs::read(&archive_path).expect("should read archive");
        let reader = SolArchiveReader::from_bytes(archive_bytes).expect("should parse archive");

        assert_eq!(
            reader.block_count(),
            16,
            "archive should include blocks 0..15"
        );
        assert_eq!(reader.min_height(), SolHeight(0));
        assert_eq!(reader.max_height(), SolHeight(15));

        for expected_height in 0..=15 {
            let entry = reader
                .get_entry_by_height(SolHeight(expected_height))
                .expect("archive entry should exist");
            let jam_bytes = reader
                .get_jam_by_height(SolHeight(expected_height))
                .expect("jam bytes should exist");
            assert_eq!(entry.height, SolHeight(expected_height));
            assert!(!jam_bytes.is_empty(), "jam bytes should not be empty");
        }

        assert!(progress
            .iter()
            .any(|update| update.phase == ArchiveExtractionPhase::Complete));

        println!("[TEST 04] ✓ Archive roundtrip verified for blocks 0-15");
    }

    /// Archive regression test for optional mempool snapshot capture.
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_05_archive_extract_with_mempool_snapshots() {
        let mut extractor = initialized_extractor(true).await;
        let temp_dir = tempdir().expect("should create temp dir");
        let archive_path = temp_dir.path().join("mempool.solarch");
        let mut progress = Vec::new();

        println!("[TEST 05] Extracting blocks 0-15 to archive with mempool snapshots...");
        extractor
            .extract_range_to_archive_with_progress(0, 15, &archive_path, |update| {
                progress.push(update);
            })
            .await
            .expect("should extract archive with mempool snapshots");

        let archive_bytes = std::fs::read(&archive_path).expect("should read archive");
        let reader = SolArchiveReader::from_bytes(archive_bytes).expect("should parse archive");
        let metadata = reader.metadata();

        assert!(
            metadata.has_mempool,
            "archive should record mempool snapshots"
        );
        assert_eq!(metadata.mempool_snapshot_count, 16);
        assert_eq!(metadata.mempool_min_height, Some(SolHeight(0)));
        assert_eq!(metadata.mempool_max_height, Some(SolHeight(15)));
        assert_eq!(
            reader.mempool_snapshot_count(),
            16,
            "reader should expose one snapshot per archived block"
        );
        assert!(progress
            .iter()
            .any(|update| update.phase == ArchiveExtractionPhase::MempoolReplay));
        assert!(progress
            .iter()
            .any(|update| update.phase == ArchiveExtractionPhase::Complete));

        println!("[TEST 05] ✓ Archive mempool replay verified for blocks 0-15");
    }
}
