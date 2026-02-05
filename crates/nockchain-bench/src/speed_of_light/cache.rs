//! Speed-of-light benchmark cache
//!
//! Stores extracted blocks and transactions in a Rust data structure
//! for fast access during benchmarking.

use std::collections::{BTreeMap, HashMap};

use nockchain_types::tx_engine::common::Hash;

use super::types::{BlockData, SolHeight, TransactionData};

/// Cache for speed-of-light benchmark data
///
/// Stores blocks and transactions extracted from a checkpoint,
/// ready for use in throughput benchmarks.
#[derive(Debug, Default)]
pub struct SpeedOfLightCache {
    /// Blocks indexed by height
    blocks: BTreeMap<SolHeight, BlockData>,

    /// Transaction lookup: tx_id → block height
    tx_index: HashMap<Hash, SolHeight>,

    /// Maximum block height in cache
    max_height: SolHeight,

    /// Minimum block height in cache
    min_height: SolHeight,

    /// Total transaction count
    tx_count: usize,
}

impl SpeedOfLightCache {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self {
            blocks: BTreeMap::new(),
            tx_index: HashMap::new(),
            max_height: SolHeight::ZERO,
            min_height: SolHeight::MAX,
            tx_count: 0,
        }
    }

    /// Insert a block into the cache
    pub fn insert_block(&mut self, block: BlockData) {
        let height = block.height;

        // Update tx index
        for tx in &block.transactions {
            self.tx_index.insert(tx.tx_id.clone(), height);
        }
        self.tx_count += block.transactions.len();

        // Update height bounds
        if height > self.max_height {
            self.max_height = height;
        }
        if height < self.min_height {
            self.min_height = height;
        }

        self.blocks.insert(height, block);
    }

    /// Insert multiple blocks
    pub fn insert_blocks(&mut self, blocks: impl IntoIterator<Item = BlockData>) {
        for block in blocks {
            self.insert_block(block);
        }
    }

    /// Get a block by height
    pub fn get_block(&self, height: SolHeight) -> Option<&BlockData> {
        self.blocks.get(&height)
    }

    /// Get the block containing a transaction
    pub fn get_block_for_tx(&self, tx_id: &Hash) -> Option<&BlockData> {
        let height = self.tx_index.get(tx_id)?;
        self.blocks.get(height)
    }

    /// Get a transaction by ID
    pub fn get_transaction(&self, tx_id: &Hash) -> Option<&TransactionData> {
        let block = self.get_block_for_tx(tx_id)?;
        block.transactions.iter().find(|tx| &tx.tx_id == tx_id)
    }

    /// Iterate over all blocks in ascending height order
    pub fn iter_blocks(&self) -> impl Iterator<Item = &BlockData> {
        self.blocks.values()
    }

    /// Iterate over blocks in a height range (inclusive)
    pub fn iter_blocks_range(
        &self,
        start: SolHeight,
        end: SolHeight,
    ) -> impl Iterator<Item = &BlockData> {
        self.blocks.range(start..=end).map(|(_, b)| b)
    }

    /// Number of blocks in cache
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Total number of transactions across all blocks
    pub fn transaction_count(&self) -> usize {
        self.tx_count
    }

    /// Maximum block height in cache
    pub fn max_height(&self) -> SolHeight {
        self.max_height
    }

    /// Minimum block height in cache
    pub fn min_height(&self) -> SolHeight {
        if self.blocks.is_empty() {
            SolHeight::ZERO
        } else {
            self.min_height
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let total_outputs: usize = self
            .blocks
            .values()
            .flat_map(|b| &b.transactions)
            .map(|tx| tx.outputs.len())
            .sum();

        CacheStats {
            block_count: self.block_count(),
            transaction_count: self.tx_count,
            output_count: total_outputs,
            min_height: self.min_height(),
            max_height: self.max_height,
        }
    }

    /// Clear all data from the cache
    pub fn clear(&mut self) {
        self.blocks.clear();
        self.tx_index.clear();
        self.max_height = SolHeight::ZERO;
        self.min_height = SolHeight::MAX;
        self.tx_count = 0;
    }
}

/// Statistics about the cache contents
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub block_count: usize,
    pub transaction_count: usize,
    pub output_count: usize,
    pub min_height: SolHeight,
    pub max_height: SolHeight,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SpeedOfLightCache: {} blocks (heights {}..={}), {} txs, {} outputs",
            self.block_count,
            self.min_height.as_u64(),
            self.max_height.as_u64(),
            self.transaction_count,
            self.output_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nockchain_math::belt::Belt;

    fn dummy_hash(v: u64) -> Hash {
        Hash([Belt(v), Belt(v + 1), Belt(v + 2), Belt(v + 3), Belt(v + 4)])
    }

    fn dummy_block(height: u64, tx_count: usize) -> BlockData {
        let transactions = (0..tx_count)
            .map(|i| TransactionData {
                tx_id: dummy_hash(height * 1000 + i as u64),
                version: 0,
                raw_tx: nockchain_types::tx_engine::v0::RawTx {
                    id: dummy_hash(height * 1000 + i as u64), // TxId is just a Hash alias
                    inputs: nockchain_types::tx_engine::v0::Inputs(vec![]),
                    timelock_range: nockchain_types::tx_engine::common::TimelockRangeAbsolute::none(),
                    total_fees: nockchain_types::tx_engine::common::Nicks(0),
                },
                total_size: 100,
                outputs: vec![],
            })
            .collect();

        BlockData {
            height: SolHeight(height),
            block_id: dummy_hash(height),
            parent_id: dummy_hash(height.saturating_sub(1)),
            timestamp: 1234567890 + height,
            transactions,
        }
    }

    #[test]
    fn test_cache_insert_and_lookup() {
        let mut cache = SpeedOfLightCache::new();

        cache.insert_block(dummy_block(0, 2));
        cache.insert_block(dummy_block(1, 3));
        cache.insert_block(dummy_block(2, 1));

        assert_eq!(cache.block_count(), 3);
        assert_eq!(cache.transaction_count(), 6);
        assert_eq!(cache.min_height(), SolHeight(0));
        assert_eq!(cache.max_height(), SolHeight(2));

        // Lookup by height
        let block = cache.get_block(SolHeight(1)).unwrap();
        assert_eq!(block.height, SolHeight(1));
        assert_eq!(block.transactions.len(), 3);

        // Lookup by tx
        let tx_id = dummy_hash(1000); // First tx in block 1
        let block = cache.get_block_for_tx(&tx_id).unwrap();
        assert_eq!(block.height, SolHeight(1));
    }

    #[test]
    fn test_cache_range_iteration() {
        let mut cache = SpeedOfLightCache::new();

        for i in 0..10 {
            cache.insert_block(dummy_block(i, 1));
        }

        let range: Vec<_> = cache.iter_blocks_range(SolHeight(3), SolHeight(7)).collect();
        assert_eq!(range.len(), 5);
        assert_eq!(range[0].height, SolHeight(3));
        assert_eq!(range[4].height, SolHeight(7));
    }

    /// Test cache block lookup by height
    #[test]
    fn test_cache_block_lookup() {
        let mut cache = SpeedOfLightCache::new();

        // Insert blocks with known data
        cache.insert_block(dummy_block(0, 1));
        cache.insert_block(dummy_block(5, 2));
        cache.insert_block(dummy_block(10, 3));

        // Verify get_block returns correct block for each height
        let block_0 = cache.get_block(SolHeight(0)).expect("block 0 should exist");
        assert_eq!(block_0.height, SolHeight(0));
        assert_eq!(block_0.transactions.len(), 1);

        let block_5 = cache.get_block(SolHeight(5)).expect("block 5 should exist");
        assert_eq!(block_5.height, SolHeight(5));
        assert_eq!(block_5.transactions.len(), 2);

        let block_10 = cache.get_block(SolHeight(10)).expect("block 10 should exist");
        assert_eq!(block_10.height, SolHeight(10));
        assert_eq!(block_10.transactions.len(), 3);

        // Verify missing blocks return None
        assert!(cache.get_block(SolHeight(1)).is_none(), "block 1 should not exist");
        assert!(cache.get_block(SolHeight(100)).is_none(), "block 100 should not exist");
    }

    /// Test cache transaction lookup by tx_id
    #[test]
    fn test_cache_tx_lookup() {
        let mut cache = SpeedOfLightCache::new();

        // Insert blocks with transactions
        cache.insert_block(dummy_block(0, 2)); // tx_ids: 0, 1
        cache.insert_block(dummy_block(1, 3)); // tx_ids: 1000, 1001, 1002
        cache.insert_block(dummy_block(2, 1)); // tx_ids: 2000

        // Look up transaction from block 0
        let tx_id_0 = dummy_hash(0);
        let tx_0 = cache.get_transaction(&tx_id_0).expect("tx should exist");
        assert_eq!(tx_0.tx_id, tx_id_0);

        // Look up transaction from block 1
        let tx_id_1001 = dummy_hash(1001);
        let tx_1001 = cache.get_transaction(&tx_id_1001).expect("tx should exist");
        assert_eq!(tx_1001.tx_id, tx_id_1001);

        // Look up transaction from block 2
        let tx_id_2000 = dummy_hash(2000);
        let tx_2000 = cache.get_transaction(&tx_id_2000).expect("tx should exist");
        assert_eq!(tx_2000.tx_id, tx_id_2000);

        // Verify missing transaction returns None
        let missing_tx_id = dummy_hash(9999);
        assert!(cache.get_transaction(&missing_tx_id).is_none(), "missing tx should return None");

        // Verify get_block_for_tx returns the correct block
        let block_for_tx = cache.get_block_for_tx(&tx_id_1001).expect("block should exist");
        assert_eq!(block_for_tx.height, SolHeight(1));
    }

    /// Test cache stats are accurate
    #[test]
    fn test_cache_stats() {
        let mut cache = SpeedOfLightCache::new();

        // Empty cache stats
        let stats = cache.stats();
        assert_eq!(stats.block_count, 0);
        assert_eq!(stats.transaction_count, 0);
        assert_eq!(stats.output_count, 0);

        // Insert blocks and verify stats update
        cache.insert_block(dummy_block(5, 2));
        cache.insert_block(dummy_block(10, 3));
        cache.insert_block(dummy_block(15, 1));

        let stats = cache.stats();
        assert_eq!(stats.block_count, 3, "should have 3 blocks");
        assert_eq!(stats.transaction_count, 6, "should have 6 transactions (2+3+1)");
        assert_eq!(stats.min_height, SolHeight(5), "min height should be 5");
        assert_eq!(stats.max_height, SolHeight(15), "max height should be 15");

        // Verify individual accessors match stats
        assert_eq!(cache.block_count(), stats.block_count);
        assert_eq!(cache.transaction_count(), stats.transaction_count);
        assert_eq!(cache.min_height(), stats.min_height);
        assert_eq!(cache.max_height(), stats.max_height);

        // Test stats display format
        let display = format!("{}", stats);
        assert!(display.contains("3 blocks"), "display should show block count");
        assert!(display.contains("5..=15"), "display should show height range");
        assert!(display.contains("6 txs"), "display should show tx count");
    }

    /// Test cache iteration in correct order
    #[test]
    fn test_cache_iteration() {
        let mut cache = SpeedOfLightCache::new();

        // Insert blocks out of order
        cache.insert_block(dummy_block(5, 1));
        cache.insert_block(dummy_block(2, 1));
        cache.insert_block(dummy_block(8, 1));
        cache.insert_block(dummy_block(0, 1));
        cache.insert_block(dummy_block(3, 1));

        // iter_blocks() should return blocks in ascending height order
        let all_blocks: Vec<_> = cache.iter_blocks().collect();
        assert_eq!(all_blocks.len(), 5);
        assert_eq!(all_blocks[0].height, SolHeight(0));
        assert_eq!(all_blocks[1].height, SolHeight(2));
        assert_eq!(all_blocks[2].height, SolHeight(3));
        assert_eq!(all_blocks[3].height, SolHeight(5));
        assert_eq!(all_blocks[4].height, SolHeight(8));

        // Verify heights are strictly ascending
        for window in all_blocks.windows(2) {
            assert!(window[0].height < window[1].height, "blocks should be in ascending order");
        }

        // iter_blocks_range() should return blocks in range, in order
        let range_blocks: Vec<_> = cache.iter_blocks_range(SolHeight(2), SolHeight(5)).collect();
        assert_eq!(range_blocks.len(), 3, "should have 3 blocks in range 2..=5");
        assert_eq!(range_blocks[0].height, SolHeight(2));
        assert_eq!(range_blocks[1].height, SolHeight(3));
        assert_eq!(range_blocks[2].height, SolHeight(5));

        // Empty range returns empty iterator
        let empty_range: Vec<_> = cache.iter_blocks_range(SolHeight(100), SolHeight(200)).collect();
        assert!(empty_range.is_empty(), "range with no blocks should be empty");
    }
}
