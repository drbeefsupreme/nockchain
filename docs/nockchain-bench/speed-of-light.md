---
tags:
  - pma
  - benchmarking
  - speed-of-light
date: 2026-01-26
---

# Speed of Light Benchmark

## Definition

The term "speed of light" comes from NVIDIA profiling tools. It refers to the **baseline theoretical level of throughput** - performance when not limited by external factors like the network.

> If we weren't limited by network, just had all the blocks ready to go, hallucinated or not, how quickly could we poke blocks into the serf as fast as possible?

## Current Findings

- There's a **10x difference** between getting blocks over the network vs being fed them as fast as possible
- Chris wrote a speed of light benchmark (location TBD)

## Testing Approach

- Try speed of light tests with different memory limits
- Would be easier to test memory pressure with **batch sync** (poke with a batch of blocks)
- Need to hallucinate a variety of types of blocks

## Getting Test Data

Get chain history out of a live checkpoint and extract the blocks to feed into the serf for building a new speed of light checkpoint.

### Using the Block Explorer

Per [[2026-01-23 Chris|Chris's suggestion]], the block explorer can peek the state out:

- **File:** `crates/nockapp-grpc/src/services/public_nockchain/v2/block_explorer.rs`
- Creates a Rust in-memory cache of all blockchain state
- Makes it very fast to query
- Types have custom `From` and `TryFrom` implementations for noun conversion

### Key Modules

| Module | Purpose |
|--------|---------|
| `block_explorer.rs` | Cache and peek logic for blockchain state |
| `cache.rs` | Balance caching with pagination |
| `noun-serde` | Noun serialization/deserialization traits |
| `nockchain-types` | Shared types (`NoteV0`, `RawTx`, `Hash`, etc.) |

### Checkpoint Approach

Using real checkpoint data is better than hallucinating:

1. Get a checkpoint jam
2. Test if non-chaff jam/cue can read it (since it was made with chaff jam/cue)
3. See if it can get far enough to make a checkpoint
4. Kill it and analyze

## Related

- [[2026-01-23 Chris]] - Original discussion of speed of light benchmark
- [[Overview]] - nockchain-bench overview

## Parameters / Toggles (Chris's Requirements)

Ideally the speed-of-light test should have configurable parameters for:

1. **Proof version cutover height** - Toggle for when it cuts over the proof version at X height
2. **Transaction weight ranges** - Configurable tx weight ranges mapped to block height ranges
3. **Dead transaction simulation** - Block height range maps for transactions that sit in the mempool for ~20 blocks and then _should_ get dropped
   - Use case: Repro Auri's issue and verify if GC is capturing those dead txs

## Test Plan

### 1. Smoke Tests (basic functionality)

| Test | Description |
|------|-------------|
| `test_load_checkpoint` | Verify we can load and cue `0.chkjam` without errors |
| `test_boot_nockapp_from_checkpoint` | Verify NockApp boots successfully with checkpoint + `dumb.jam` |
| `test_peek_chain_height` | Verify we can peek `/heaviest-chain` and get a valid height > 0 |

### 2. Extraction Tests

| Test                               | Description                                                               |
| ---------------------------------- | ------------------------------------------------------------------------- |
| `test_extract_single_block`        | Extract block 0, verify expected structure (height=0, has block_id, etc.) |
| `test_extract_block_range`         | Extract blocks 0-7 (one chunk), verify we get 8 blocks in order           |
| `test_extract_first_1000_blocks`   | Main use case - extract 1000 blocks, verify count and basic integrity     |
| `test_extract_beyond_chain_height` | Request 10000 blocks when chain only has N, verify we get N blocks        |

### 3. Cache Tests

| Test | Description |
|------|-------------|
| `test_cache_block_lookup` | Insert blocks, verify `get_block(height)` returns correct block |
| `test_cache_tx_lookup` | Insert blocks with txs, verify `get_transaction(tx_id)` works |
| `test_cache_stats` | Verify stats are accurate (block count, tx count, height range) |
| `test_cache_iteration` | Verify `iter_blocks()` and `iter_blocks_range()` return correct order |

### 4. Data Integrity Tests

| Test                              | Description                                                          |
| --------------------------------- | -------------------------------------------------------------------- |
| `test_block_fields_populated`     | Verify extracted blocks have non-zero block_id, parent_id, timestamp |
| `test_parent_chain_linkage`       | Verify block N's parent_id == block N-1's block_id                   |
| `test_transaction_structure`      | Verify transactions have tx_id, raw_tx, outputs decoded correctly    |
| `test_genesis_block_special_case` | Block 0 may have special properties (no parent, coinbase only?)      |

### 5. Round-Trip Fidelity Tests (Chris's concern)

| Test | Description |
|------|-------------|
| `test_rawtx_roundtrip` | Decode a RawTx from checkpoint, encode it back, verify noun equality |
| `test_note_roundtrip` | Same for NoteV0 |
| `test_block_txs_roundtrip` | For each tx in extracted blocks, verify decode→encode→decode produces same data |

### 6. Performance Baselines

| Test | Description |
|------|-------------|
| `test_extraction_timing` | Measure time to extract 100/1000 blocks, print results (no assertions) |
| `test_memory_usage` | Track peak memory during extraction (if instrumentable) |                              

## Progress Log

### 2026-01-27

- Added Chris's parameter requirements
- Investigated round-trip fidelity for Noun → Rust → Noun conversions
- Key finding: z-map/z-set types (Lock, Balance, Inputs, Seeds) use Vec in Rust but tree structure in Hoon
- Existing tests in `nockchain-types` verify isomorphic round-trip for `RawTx`, `Balance`, `NoteV0`
- Block explorer's `TxV0` is decode-only (no `NounEncode`) - would need to use `nockchain_types::v0::RawTx` for re-encoding
- **Implemented speed_of_light module in nockchain-bench:**
  - `mod.rs` - Module exports
  - `types.rs` - BlockData, TransactionData, TxOutput structs + noun decoding (replicates block_explorer.rs types)
  - `checkpoint.rs` - Load and cue checkpoint files
  - `cache.rs` - SpeedOfLightCache for storing extracted blocks/txs
  - `extractor.rs` - BlockExtractor using NockApp to peek blocks from kernel
- Uses NockApp (not Kernel directly) for public peek API
- Peek paths: `/heaviest-chain ~` and `/heaviest-chain-blocks-range/[start]/[end] ~`

#### Key Finding: Jets Required for Performance

**Problem:** Initially, kernel initialization took 15+ minutes without completing.

**Root cause:** Missing jets (hot state). Without jets, the Nock interpreter runs pure Nock for all operations, which is extremely slow.

**Solution:**
1. Added `zkvm-jetpack` dependency
2. Used `produce_prover_hot_state()` to get 82 jet entries
3. Used `Kernel::load_with_hot_state_medium` (16GB stack size)

Result: Initialization now takes ~2.4 minutes (142 seconds).

#### Heaviest-Chain Peek Limitation

The `/heaviest-chain ~` peek may return `Some(None)` for some checkpoints, indicating the chain tip data isn't cached. This is expected - blocks can still be extracted via `/heaviest-chain-blocks-range` without knowing the tip height.

The `extract_to_cache()` method now gracefully handles this by extracting until empty results.

#### Running Tests

```bash
# Quick tests (~2 minutes, mostly checkpoint cue time)
cargo test --release -p nockchain-bench -- --nocapture

# Full integration tests (~2.4 minutes for init + tests)
# MUST use --test-threads=1 to share extractor instance
cargo test --release -p nockchain-bench -- --ignored --test-threads=1 --nocapture
```

#### Environment Variables

```bash
# Override checkpoint and kernel paths
SOL_CHECKPOINT_PATH=path/to/0.chkjam
SOL_KERNEL_PATH=path/to/dumb.jam
```

### 2026-01-28

#### Morning: Fixed Kernel Initialization

- Fixed kernel initialization: Added jets via `produce_prover_hot_state()` from `zkvm-jetpack`
- Changed stack size to Medium (16GB) via `Kernel::load_with_hot_state_medium`
- Initialization time reduced from 15+ minutes (incomplete) to ~2.4 minutes
- Made `extract_to_cache()` robust to missing chain height - extracts until empty results

#### Afternoon: Comprehensive Test Suite Complete

**Integration Tests (6 total, all passing):**

| Test | Description | Status |
|------|-------------|--------|
| `integration_test_01_extractor_initializes` | NockApp boots with checkpoint + kernel | ✅ |
| `integration_test_02_peek_chain_height` | Gracefully handles missing chain tip | ✅ |
| `integration_test_03_extract_single_block` | Extract block 0, verify structure | ✅ |
| `integration_test_04_extract_block_range` | Extract blocks 0-7, verify order | ✅ |
| `integration_test_05_extract_first_1000_blocks` | Extract 1000 blocks to cache | ✅ |
| `integration_test_06_first_network_transactions` | Verify first 3 txs on network | ✅ |

**Cache Tests (6 total, all passing):**

| Test | Description | Status |
|------|-------------|--------|
| `test_cache_block_lookup` | `get_block(height)` returns correct block | ✅ |
| `test_cache_tx_lookup` | `get_transaction(tx_id)` works correctly | ✅ |
| `test_cache_stats` | Stats accurate (count, height range) | ✅ |
| `test_cache_iteration` | `iter_blocks()` returns ascending order | ✅ |
| `test_cache_insert_and_lookup` | Basic insert/lookup operations | ✅ |
| `test_cache_range_iteration` | `iter_blocks_range()` returns subset | ✅ |

#### First Network Transactions Verified

Found and verified the first three transactions on the network:

| Block | Transaction ID | From Address |
|-------|---------------|--------------|
| 5629 | `3srZpNCmbcu5V3BAahWS8wSApju6JfxByLPPuxUBZB3TMLhHw8tHMtv` | `37oNkJu8RiUswLrAmrBpBKBQdQmYCD3BENZH4MN7DpyCZVS3dF9NCJ7jAcRFLwK1nUgkLqnQVMsgMqmYx284YJGkwYWCrY3um9tPYwuACMY7aebZcUaFis45oQT81UQfbUYt` |
| 9095 | (1 tx) | TBD |
| 9239 | (1 tx) | TBD |

**Key insight:** Blocks 0-5628 have no transactions (empty mining blocks). First real tx at block 5629.

#### Performance Benchmarks

| Operation | Time |
|-----------|------|
| Kernel initialization | ~2.4 minutes (142s) |
| Extract 1000 blocks | ~13 seconds (after init) |
| Extract 9240 blocks | ~4.4 minutes (263s total including init) |

#### Ready for Speed of Light Testing

The extraction infrastructure is now complete:
- Can load any checkpoint and extract blocks/transactions
- Cache provides fast lookup by height or tx_id
- Transaction data includes full inputs with lock pubkeys (addresses)
- All types properly decoded from Nock nouns

**Next steps:**
1. ~~Design the actual speed-of-light benchmark (poke blocks as fast as possible)~~ ✅
2. Implement block injection into a fresh kernel
3. Measure throughput under various conditions

#### Evening: Archive Format Implementation Complete

Implemented Option C (Hybrid): Bincode Metadata + Raw Jammed Nouns

**Archive Format:**
```
[8 bytes: metadata_len][bincode metadata][jam blob 1][jam blob 2]...[jam blob N]
```

**Key Types:**
- `ArchiveMetadata` - Header with block count, height range, and per-block entries
- `BlockEntry` - Per-block metadata: height, block_id, tx_count, jam_offset, jam_size
- `ArchiveWriter` - Builds archive in memory, writes to file
- `ArchiveReader` - Loads archive, provides iteration and random access

**New Integration Tests (5 total, all passing):**

| Test | Description | Status |
|------|-------------|--------|
| `integration_test_07_extract_with_jam_returns_bytes` | Extract with jam bytes (~117KB/block) | ✅ |
| `integration_test_08_extract_with_jam_matches_decode` | Decoded data matches regular extraction | ✅ |
| `integration_test_09_jam_roundtrip_fidelity` | Jam → Cue → Decode preserves data | ✅ |
| `integration_test_10_extract_to_archive` | Extract 100 blocks to archive (~11.7MB) | ✅ |
| `integration_test_11_full_pipeline` | Extract → Archive → Load → Decode | ✅ |

**Archive Tests (15 total, all passing):**

| Test | Description | Status |
|------|-------------|--------|
| `test_archive_metadata_creation` | Empty metadata initialization | ✅ |
| `test_archive_metadata_add_blocks` | Adding blocks updates metadata | ✅ |
| `test_archive_version_check` | Version validation | ✅ |
| `test_block_entry_roundtrip` | BlockEntry bincode roundtrip | ✅ |
| `test_archive_writer_empty` | Empty archive creation | ✅ |
| `test_archive_writer_single_block` | Single block write | ✅ |
| `test_archive_writer_multiple_blocks` | Multiple blocks write | ✅ |
| `test_archive_writer_file_structure` | File format verification | ✅ |
| `test_archive_reader_roundtrip` | Write → Read roundtrip | ✅ |
| `test_archive_reader_get_jam_by_height` | Random access by height | ✅ |
| `test_archive_reader_get_jam_by_index` | Random access by index | ✅ |
| `test_archive_reader_missing_height` | Error for missing height | ✅ |
| `test_archive_reader_iterate` | Full iteration | ✅ |
| `test_archive_reader_iterate_range` | Range iteration | ✅ |
| `test_archive_reader_invalid_data` | Invalid data handling | ✅ |

**Key Implementation Details:**

1. **Block extraction with jam bytes**: New `extract_blocks_range_with_jam()` method uses `HoonList` to manually iterate the noun list, copies each entry to a fresh slab, and jams it individually.

2. **Archive writing**: `extract_to_archive()` method extracts blocks and writes directly to archive file, combining metadata and jam blobs.

3. **Round-trip fidelity**: Jam → Cue → Decode produces identical BlockData, confirming no data loss in the archive format.

**Archive Size Metrics:**
- ~117KB per block (empty mining blocks)
- 100 blocks: ~11.7 MB
- Estimated full chain (9240 blocks): ~1.08 GB

**Ready for Benchmark Implementation:**
- Archive format preserves exact noun structure
- Can load archive quickly (no kernel initialization needed)
- Cue individual blocks on-demand for injection

#### Late Evening: CLI and Benchmark Runner Complete

**New CLI Commands:**

```bash
# Extract blocks from checkpoint to archive
nockchain-bench sol extract -n 1000 -c 0.chkjam -k assets/dumb.jam

# Run speed-of-light benchmark
nockchain-bench sol bench -a blocks_1000.solarch -k assets/dumb.jam
```

**`sol extract` options:**
- `-n, --blocks` - Number of blocks to extract (default: 1000)
- `-c, --checkpoint` - Path to checkpoint file (default: 0.chkjam)
- `-k, --kernel` - Path to kernel jam (default: assets/dumb.jam)
- `-o, --output` - Output archive path (default: blocks_<N>.solarch)

**`sol bench` options:**
- `-a, --archive` - Path to archive file (default: blocks_1000.solarch)
- `-k, --kernel` - Path to kernel jam (default: assets/dumb.jam)
- `-n, --blocks` - Number of blocks to benchmark, 0 = all (default: 0)
- `--skip-genesis` - Skip genesis block (not recommended)

**Benchmark Runner Implementation:**

New module `speed_of_light/bench.rs` with:
- `BenchConfig` - Configuration for benchmark runs
- `BenchResults` - Results including throughput, timings, failures
- `BenchRunner` - Main benchmark executor

**How the benchmark works:**
1. Load archive file (fast - just reads bytes)
2. Boot fresh kernel (no checkpoint - starts from genesis state)
3. For each block in archive:
   - Cue the jammed block entry noun
   - Extract the `page` noun from `[height [block_id [page txs]]]`
   - Construct poke: `[%fact [%heard-block page]]`
   - Poke into kernel with wire `{source: "bench", version: 1}`
   - Measure time
4. Report throughput statistics

**Poke Format Discovery:**
- Blocks are poked as `%heard-block` facts
- Full structure: `[%fact [%heard-block page:dt]]`
- Wire source can be "miner", "sys", "libp2p", or "bench"
- Kernel validates: digest, PoW, parent exists, duplicate check

**Files Added/Modified:**
- `speed_of_light/bench.rs` - New benchmark runner module
- `speed_of_light/mod.rs` - Added bench module exports
- `main.rs` - Added `sol extract` and `sol bench` CLI commands

**Next Steps:**
1. ~~Implement block injection into a fresh kernel~~ ✅
2. Run benchmark and analyze results
3. Tune based on validation failures (if any)
4. Add memory profiling during benchmark

### 2026-01-26

- Started investigating block explorer architecture
- Key peek paths identified:
  - `/heaviest-chain ~` - get tip height and hash
  - `/heaviest-chain-blocks-range/[start]/[end] ~` - get block range with transactions
