//! Block extraction from a running kernel via peek

use std::path::{Path, PathBuf};

use nockapp::nockapp::save::SaveableCheckpoint;
use nockapp::nockapp::wire::WireRepr;
use nockapp::nockapp::NockApp;
use nockapp::noun::slab::NounSlab;
use nockchain_math::belt::Belt;
use nockchain_math::noun_ext::NounMathExt;
use nockchain_math::structs::{HoonList, HoonMapIter};
use nockchain_types::tx_engine::common::Hash;
use nockvm::noun::{Noun, NounAllocator, NounSpace, SIG};
use noun_serde::NounDecode;
use thiserror::Error;
use tracing::{debug, info};

use super::archive::{ArchiveReader, ArchiveWriter, MempoolTxEntry};
use super::cache::SpeedOfLightCache;
use super::checkpoint::{load_checkpoint, CheckpointLoadError};
use super::kernel_utils::{
    init_nockapp, peek_heaviest_chain, sol_replay_wire, KernelInitError, PeekChainError,
};
use super::poke::build_poke_slab_from_jam;
use super::types::{BlockData, BlockDataWithJam, BlockRangeEntryNoun, ProofVersion, SolHeight};

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

fn zero_hash() -> Hash {
    Hash([Belt(0); 5])
}

fn synthetic_block_data(height: u64) -> BlockData {
    let zero = zero_hash();
    BlockData {
        height: SolHeight(height),
        block_id: zero.clone(),
        parent_id: zero,
        timestamp: 0,
        transactions: Vec::new(),
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
            false,
            true,
        )
        .await?;

        info!("NockApp initialized successfully");
        self.nockapp = Some(nockapp);
        Ok(())
    }

    /// Get the current chain tip height
    pub async fn get_chain_height(&mut self) -> Result<(u64, Hash), ExtractorError> {
        let nockapp = self.nockapp.as_mut().ok_or(ExtractorError::KernelLoad(
            "NockApp not initialized".to_string(),
        ))?;

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
        let nockapp = self.nockapp.as_mut().ok_or(ExtractorError::KernelLoad(
            "NockApp not initialized".to_string(),
        ))?;

        let poke_slab = build_poke_slab_from_jam(jam_bytes).map_err(ExtractorError::EntryDecode)?;

        nockapp.poke(wire.clone(), poke_slab).await?;
        Ok(())
    }

    async fn peek_raw_transactions(&mut self) -> Result<Vec<MempoolTxEntry>, ExtractorError> {
        let nockapp = self.nockapp.as_mut().ok_or(ExtractorError::KernelLoad(
            "NockApp not initialized".to_string(),
        ))?;

        let mut path_slab = NounSlab::new();
        let tag = nockapp::utils::make_tas(&mut path_slab, "raw-transactions").as_noun();
        let path_noun = nockvm::noun::T(&mut path_slab, &[tag, SIG]);
        path_slab.set_root(path_noun);

        let result = nockapp.peek(path_slab).await?;
        let result_noun = unsafe { result.root() };
        let space = result.noun_space();

        let map_noun = match decode_unit_unit(*result_noun, &space) {
            Some(noun) => noun,
            None => return Ok(Vec::new()),
        };

        if let Ok(atom) = map_noun.in_space(&space).as_atom() {
            if atom.as_u64().unwrap_or(1) == 0 {
                return Ok(Vec::new());
            }
        }

        let mut entries = Vec::new();
        for entry in HoonMapIter::new(map_noun, &space) {
            let [key, value] = match entry.uncell(&space) {
                Ok(kv) => kv,
                Err(_) => continue,
            };

            let tx_id = Hash::from_noun(&key, &space)?;
            let value_cell = value
                .in_space(&space)
                .as_cell()
                .map_err(|_| ExtractorError::EntryDecode("raw-tx entry not a cell".to_string()))?;
            let heard_at_noun = value_cell.tail().noun();
            let heard_at = u64::from_noun(&heard_at_noun, &space)?;

            entries.push(MempoolTxEntry {
                tx_id,
                heard_at: SolHeight(heard_at),
            });
        }

        Ok(entries)
    }

    async fn populate_mempool_snapshots_with_progress<F>(
        &mut self,
        writer: &mut ArchiveWriter,
        mut on_progress: F,
    ) -> Result<(), ExtractorError>
    where
        F: FnMut(usize, usize, SolHeight),
    {
        let reader = ArchiveReader::from_bytes(writer.to_bytes()?)?;
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
        let outer_opt = result_noun
            .in_space(&space)
            .as_cell()
            .map_err(|_| ExtractorError::PeekReturnedNoData)?;

        // Check if outer option is Some (should be [~ inner])
        let outer_head = outer_opt.head().noun();
        if !outer_head.is_atom()
            || outer_head
                .in_space(&space)
                .as_atom()
                .map(|a| a.as_u64().unwrap_or(1))
                .unwrap_or(1)
                != 0
        {
            return Err(ExtractorError::PeekReturnedNoData);
        }

        let inner = outer_opt.tail().noun();
        let inner_opt = inner
            .in_space(&space)
            .as_cell()
            .map_err(|_| ExtractorError::PeekReturnedNoData)?;

        // Check if inner option is Some
        let inner_head = inner_opt.head().noun();
        if !inner_head.is_atom()
            || inner_head
                .in_space(&space)
                .as_atom()
                .map(|a| a.as_u64().unwrap_or(1))
                .unwrap_or(1)
                != 0
        {
            return Err(ExtractorError::PeekReturnedNoData);
        }

        let list_noun = inner_opt.tail().noun();

        // Iterate the list and process each entry individually
        let mut blocks_with_jam = Vec::new();
        let mut decode_fallbacks = 0usize;

        for (idx, entry_noun) in
            HoonList::try_from(list_noun, &space).map_err(|_| ExtractorError::PeekReturnedNoData)?
                .into_iter()
                .enumerate()
        {
            // Copy this entry noun into a fresh slab and jam it
            let mut entry_slab: NounSlab = NounSlab::new();
            let copied_noun = entry_slab.copy_into(entry_noun, &space);
            entry_slab.set_root(copied_noun);
            let jam_bytes = entry_slab.jam();

            // Decode the entry to BlockData using the original space
            let data = match NounDecode::from_noun(&entry_noun, &space)
                .and_then(|entry: BlockRangeEntryNoun| entry.into_block_data(&space))
            {
                Ok(data) => data,
                Err(_) => {
                    decode_fallbacks += 1;
                    synthetic_block_data(start.saturating_add(idx as u64))
                }
            };

            blocks_with_jam.push(BlockDataWithJam { data, jam_bytes });
        }

        if decode_fallbacks > 0 {
            info!(
                start,
                end,
                decode_fallbacks,
                "Used synthetic metadata for undecodable block entries"
            );
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
    pub async fn extract_to_cache(
        &mut self,
        count: u64,
    ) -> Result<SpeedOfLightCache, ExtractorError> {
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

        let mut writer = ArchiveWriter::new();
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

            match self.extract_blocks_range_with_jam(current, chunk_end).await {
                Ok(blocks) => {
                    if blocks.is_empty() {
                        info!(current, "No more blocks available, stopping extraction");
                        break;
                    }

                    for block in &blocks {
                        writer
                            .add_block(
                                block.data.height,
                                block.data.block_id.clone(),
                                block.data.tx_count(),
                                ProofVersion::for_height(block.data.height),
                                &block.jam_bytes,
                            )
                            .map_err(|e| {
                                ExtractorError::Io(std::io::Error::new(
                                    std::io::ErrorKind::Other,
                                    e.to_string(),
                                ))
                            })?;
                        total_txs += block.data.tx_count();
                    }
                    total_blocks += blocks.len();
                    on_progress(ArchiveExtractionProgress {
                        phase: ArchiveExtractionPhase::Blocks,
                        blocks_archived: total_blocks,
                        target_blocks,
                        txs_archived: total_txs,
                        chunk_start: Some(current),
                        chunk_end: Some(chunk_end),
                        chunk_blocks: blocks.len(),
                        mempool_snapshots_done: 0,
                        mempool_snapshots_total: 0,
                    });

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
                on_progress(ArchiveExtractionProgress {
                    phase: ArchiveExtractionPhase::MempoolReplay,
                    blocks_archived: total_blocks,
                    target_blocks,
                    txs_archived: total_txs,
                    chunk_start: None,
                    chunk_end: None,
                    chunk_blocks: 0,
                    mempool_snapshots_done: done,
                    mempool_snapshots_total: total,
                });
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
        on_progress(ArchiveExtractionProgress {
            phase: ArchiveExtractionPhase::Complete,
            blocks_archived: total_blocks,
            target_blocks,
            txs_archived: total_txs,
            chunk_start: None,
            chunk_end: None,
            chunk_blocks: 0,
            mempool_snapshots_done: 0,
            mempool_snapshots_total: 0,
        });

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

    /// Run the full extraction pipeline
    pub async fn run(&mut self) -> Result<SpeedOfLightCache, ExtractorError> {
        self.initialize().await?;
        self.extract_to_cache(self.config.block_count).await
    }
}

fn decode_unit(noun: Noun, space: &NounSpace) -> Option<Noun> {
    if let Ok(atom) = noun.in_space(space).as_atom() {
        if atom.as_u64().ok()? == 0 {
            return None;
        }
    }

    let cell = noun.in_space(space).as_cell().ok()?;
    let head = cell.head().noun();
    let head_atom = head.in_space(space).as_atom().ok()?;
    if head_atom.as_u64().ok()? != 0 {
        return None;
    }

    Some(cell.tail().noun())
}

fn decode_unit_unit(noun: Noun, space: &NounSpace) -> Option<Noun> {
    let inner = decode_unit(noun, space)?;
    decode_unit(inner, space)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::{Mutex, OnceCell};

    use super::*;
    use crate::speed_of_light::checkpoint::load_checkpoint;

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
                    include_mempool: false,
                };
                let mut extractor = BlockExtractor::new(config);
                extractor
                    .initialize()
                    .await
                    .expect("should initialize NockApp");
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

        assert!(
            loaded.event_num > 0,
            "event_num should be > 0 for a real checkpoint"
        );
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
            include_mempool: false,
        };
        let extractor = BlockExtractor::new(config);
        assert!(
            extractor.nockapp.is_none(),
            "nockapp should be None before initialize"
        );
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
        assert!(
            guard.nockapp.is_some(),
            "nockapp should be Some after initialize"
        );
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
                assert!(
                    height > 0,
                    "chain height should be > 0 for a real checkpoint"
                );
            }
            Err(ExtractorError::PeekReturnedNoData) => {
                // This is expected for some checkpoints - the heaviest chain tip
                // isn't always cached in the checkpoint state
                println!(
                    "[TEST 02] Chain height not available in checkpoint (expected for some states)"
                );
                println!("[TEST 02] Blocks can still be extracted via extract_blocks_range()");
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

    /// Full integration test: Extract a single block
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_03_extract_single_block() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        let blocks = guard
            .extract_blocks_range(0, 0)
            .await
            .expect("should extract block 0");

        assert_eq!(blocks.len(), 1, "should get exactly 1 block");
        let block = &blocks[0];

        println!("Block 0:");
        println!("  height: {}", block.height.as_u64());
        println!("  block_id: {}", block.block_id.to_base58());
        println!("  parent_id: {}", block.parent_id.to_base58());
        println!("  timestamp: {}", block.timestamp);
        println!("  tx_count: {}", block.transactions.len());

        assert_eq!(
            block.height,
            SolHeight(0),
            "first block should have height 0"
        );
    }

    /// Full integration test: Extract a range of blocks
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_04_extract_block_range() {
        let extractor = get_shared_extractor().await;
        let mut guard = extractor.lock().await;

        let blocks = guard
            .extract_blocks_range(0, 7)
            .await
            .expect("should extract blocks 0-7");

        assert_eq!(blocks.len(), 8, "should get 8 blocks");

        for (i, block) in blocks.iter().enumerate() {
            println!(
                "Block {}: height={}, txs={}",
                i,
                block.height.as_u64(),
                block.transactions.len()
            );
            assert_eq!(
                block.height,
                SolHeight(i as u64),
                "block height should match index"
            );
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
        let cache = guard
            .extract_to_cache(1000)
            .await
            .expect("should extract blocks to cache");

        let block_count = cache.block_count();
        let tx_count = cache.transaction_count();

        println!("[TEST 05] Extraction complete:");
        println!("  blocks: {}", block_count);
        println!("  transactions: {}", tx_count);
        println!("  stats: {}", cache.stats());

        // Should have extracted at least some blocks
        assert!(block_count > 0, "should have extracted at least 1 block");

        // Verify block ordering and linkage
        let mut prev_height: Option<SolHeight> = None;
        for block in cache.iter_blocks().take(100) {
            if let Some(prev) = prev_height {
                assert_eq!(
                    block.height,
                    prev.saturating_add(1),
                    "blocks should be in sequential order"
                );
            }
            prev_height = Some(block.height);

            // Verify basic block structure - block_id should be set
            // (We can't easily check for zero, but we know extraction succeeded)
        }

        println!(
            "[TEST 05] Block ordering verified for first {} blocks",
            prev_height.map(|h| h.as_u64() + 1).unwrap_or(0)
        );
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

        println!(
            "[TEST 06] Extracting blocks up to height {}...",
            THIRD_TX_BLOCK
        );
        let cache = guard
            .extract_to_cache(THIRD_TX_BLOCK + 1)
            .await
            .expect("should extract blocks");

        // Check block 5629 - first transaction
        println!("[TEST 06] Checking block {} (first tx)...", FIRST_TX_BLOCK);
        let block_5629 = cache
            .get_block(SolHeight(FIRST_TX_BLOCK))
            .expect("block 5629 should exist");
        assert_eq!(
            block_5629.transactions.len(),
            1,
            "block 5629 should have exactly 1 transaction"
        );

        let first_tx = &block_5629.transactions[0];
        println!("[TEST 06] First tx id: {}", first_tx.tx_id.to_base58());
        println!(
            "[TEST 06] First tx inputs: {}",
            first_tx.raw_tx.inputs.0.len()
        );

        // Verify the first transaction came from the expected address
        assert!(
            !first_tx.raw_tx.inputs.0.is_empty(),
            "first tx should have at least one input"
        );
        let first_input = &first_tx.raw_tx.inputs.0[0].1;
        let input_lock = &first_input.note.tail.lock;
        println!(
            "[TEST 06] First input lock keys_required: {}",
            input_lock.keys_required
        );
        println!(
            "[TEST 06] First input lock pubkeys count: {}",
            input_lock.pubkeys.len()
        );

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
        println!(
            "[TEST 06] Checking block {} (second tx)...",
            SECOND_TX_BLOCK
        );
        let block_9095 = cache
            .get_block(SolHeight(SECOND_TX_BLOCK))
            .expect("block 9095 should exist");
        assert!(
            !block_9095.transactions.is_empty(),
            "block 9095 should have at least 1 transaction"
        );
        println!(
            "[TEST 06] Block {} has {} transaction(s)",
            SECOND_TX_BLOCK,
            block_9095.transactions.len()
        );

        // Check block 9239 - third transaction
        println!("[TEST 06] Checking block {} (third tx)...", THIRD_TX_BLOCK);
        let block_9239 = cache
            .get_block(SolHeight(THIRD_TX_BLOCK))
            .expect("block 9239 should exist");
        assert!(
            !block_9239.transactions.is_empty(),
            "block 9239 should have at least 1 transaction"
        );
        println!(
            "[TEST 06] Block {} has {} transaction(s)",
            THIRD_TX_BLOCK,
            block_9239.transactions.len()
        );

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
                block.data.height.as_u64(),
                block.jam_bytes.len()
            );

            assert_eq!(
                block.data.height,
                SolHeight(i as u64),
                "block height should match index"
            );
            assert!(!block.jam_bytes.is_empty(), "jam bytes should not be empty");
            // Jam bytes should be reasonably sized (at least a few bytes for any noun)
            assert!(
                block.jam_bytes.len() > 10,
                "jam bytes should be substantial"
            );
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

        assert_eq!(
            regular_blocks.len(),
            blocks_with_jam.len(),
            "should have same block count"
        );

        // Compare decoded data
        for (i, (regular, with_jam)) in regular_blocks
            .iter()
            .zip(blocks_with_jam.iter())
            .enumerate()
        {
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
            assert_eq!(
                regular.timestamp, with_jam.data.timestamp,
                "timestamps should match"
            );
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

        println!(
            "[TEST 09] Original jam bytes len: {}",
            original.jam_bytes.len()
        );
        println!(
            "[TEST 09] Original block height: {}",
            original.data.height.as_u64()
        );

        // Cue the jam bytes back into a noun and decode
        let mut cue_slab: NounSlab = NounSlab::new();
        let cued_noun = cue_slab
            .cue_into(original.jam_bytes.clone())
            .expect("should cue jam bytes");

        let space = cue_slab.noun_space();

        // Decode the cued noun
        let decoded_entry: BlockRangeEntryNoun =
            NounDecode::from_noun(&cued_noun, &space).expect("should decode cued noun");
        let decoded_block = decoded_entry
            .into_block_data(&space)
            .expect("should convert to BlockData");

        // Verify the decoded data matches the original
        assert_eq!(
            decoded_block.height, original.data.height,
            "heights should match after roundtrip"
        );
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
            decoded_block.timestamp, original.data.timestamp,
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
        use tempfile::tempdir;

        use crate::speed_of_light::archive::ArchiveReader;

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
        println!(
            "  height range: {}..={}",
            metadata.min_height.as_u64(),
            metadata.max_height.as_u64()
        );

        assert_eq!(metadata.block_count, 100, "should have 100 blocks");
        assert_eq!(metadata.min_height, SolHeight(0), "should start at block 0");
        assert_eq!(metadata.max_height, SolHeight(99), "should end at block 99");

        println!("[TEST 10] ✓ Archive created and validated successfully");
    }

    /// Full integration test: Full pipeline - extract → archive → load → verify
    #[tokio::test]
    #[ignore = "Requires checkpoint - run with --ignored --test-threads=1"]
    async fn integration_test_11_full_pipeline() {
        use tempfile::tempdir;

        use crate::speed_of_light::archive::ArchiveReader;

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
            let cued_noun = slab
                .cue_into(jam_bytes.to_vec().into())
                .expect("should cue");
            let space = slab.noun_space();

            // Decode to BlockData
            let decoded_entry: BlockRangeEntryNoun =
                NounDecode::from_noun(&cued_noun, &space).expect("should decode");
            let block = decoded_entry
                .into_block_data(&space)
                .expect("should convert");

            // Verify height matches
            assert_eq!(
                block.height, entry.height,
                "decoded height should match entry"
            );

            if entry.height.as_u64() % 10 == 0 {
                println!(
                    "  Block {}: height={}, tx_count={}",
                    entry.height.as_u64(),
                    block.height.as_u64(),
                    block.tx_count()
                );
            }
        }

        println!("[TEST 11] ✓ Full pipeline verified - all 50 blocks decoded successfully");
    }
}
