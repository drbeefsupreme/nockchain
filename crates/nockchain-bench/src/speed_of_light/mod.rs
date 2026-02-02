//! Speed-of-Light Benchmark Module
//!
//! Extracts blockchain data from a checkpoint to measure maximum possible
//! throughput when not limited by network latency.
//!
//! # Overview
//!
//! The "speed of light" benchmark measures how fast we can poke blocks into
//! the serf when blocks are pre-fetched and ready, eliminating network overhead.
//!
//! This module provides:
//! - Checkpoint loading and cue'ing
//! - Block extraction via kernel peek
//! - A Rust cache for storing extracted blocks and transactions
//! - Archive format for persisting extracted data to disk
//! - Benchmark runner for injection testing

pub mod archive;
pub mod bench;
pub mod cache;
pub mod checkpoint;
pub mod checkpoint_builder;
pub mod extractor;
pub mod mempool_inspector;
pub mod poke;
pub mod start_height;
pub mod types;

pub use archive::{
    ArchiveFilter, ArchiveMetadata, ArchiveReader, ArchiveWriter, BlockEntry, MempoolSnapshotEntry,
    MempoolTxEntry,
};
pub use bench::{BenchConfig, BenchResults, BenchRunner};
pub use cache::SpeedOfLightCache;
pub use checkpoint::load_checkpoint;
pub use checkpoint_builder::{CheckpointBuildError, CheckpointBuilder, CheckpointConfig, CheckpointResult};
pub use extractor::{BlockExtractor, ExtractorConfig};
pub use mempool_inspector::{find_stale_ranges, InspectorError, StaleTxRange};
pub use start_height::{resolve_start_height, StartHeightError};
pub use types::{
    BlockData, BlockDataWithJam, ProofVersion, TransactionData, PROOF_VERSION_1_START,
    PROOF_VERSION_2_START,
};
