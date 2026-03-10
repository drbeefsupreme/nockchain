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
pub mod fixture;
pub mod harness;
pub mod kernel_utils;
pub mod mempool_inspector;
pub mod poke;
pub mod profiling;
pub mod start_height;
pub mod types;

pub use archive::{
    slice_archive_file, ArchiveFilter, ArchiveMetadata, SolArchiveReader, ArchiveSliceResult,
    SolArchiveWriter, BlockEntry, ByteOffset, ByteSize, MempoolSnapshotEntry, MempoolTxEntry,
};
pub use bench::{SolBenchConfig, SolBenchResults, SolBenchRunner};
pub use cache::SpeedOfLightCache;
pub use checkpoint::{checkpoint_event_num, load_checkpoint};
pub use checkpoint_builder::{
    CheckpointBuildError, CheckpointBuilder, CheckpointConfig, CheckpointResult,
};
pub use extractor::{
    ArchiveExtractionPhase, ArchiveExtractionProgress, BlockExtractor, ExtractorConfig,
};
pub use fixture::{
    extract_fixture_to_paths, read_fixture_file, write_fixture_file, write_fixture_file_from_paths,
    FixtureError, SolFixtureFile, SolFixtureManifest,
};
pub use harness::{
    capture_native_provenance, evaluate_verdict, execute_native_trusted_run,
    resolve_requested_case, ExecutionRequest, RequestedCase, ResolvedCase, RunFailure, RunMetrics,
    RunSummary, RunSummaryInput, Validity, ValueStats, Verdict,
};
pub use harness::docker::{
    connect_docker, parse_memory_limit, parse_proc_stat_faults, ContainerStats, HarnessDockerError,
};
pub use mempool_inspector::{find_stale_ranges, InspectorError, StaleTxRange};
pub use profiling::{
    build_scorecard, find_recovery_ms, infer_gc_events, infer_page_fault_bursts, summarize_phases,
    CheckpointProfile, GcEvent, MemoryProfile, PageFaultBurst, PhaseKind, PhaseSummary,
    PhaseWindow, ProcessMemoryProfiler, SolScorecard,
};
pub use start_height::{resolve_start_height, StartHeightError};
pub use types::{
    BlockData, BlockDataWithJam, ProofVersion, SolHeight, TransactionData, PROOF_VERSION_1_START,
    PROOF_VERSION_2_START,
};

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::harness::{
        evaluate_verdict, ExecutionRequest, RequestedCase, RunFailure, RunSummaryInput, Validity,
    };

    #[test]
    fn harness_summary_uses_phase1_defaults() {
        let requested = RequestedCase::native(PathBuf::from("fixture.soltest"));
        assert_eq!(requested.execution, ExecutionRequest::Native);
        assert_eq!(requested.warmup_runs, 1);
        assert_eq!(requested.measured_runs, 5);
        assert_eq!(requested.cooldown_secs, 10);
    }

    #[test]
    fn harness_summary_marks_failed_measured_runs_partial() {
        let verdict = evaluate_verdict(&RunSummaryInput {
            measured_run_count: 5,
            run_failures: vec![RunFailure {
                run_id: "run-2".to_string(),
                reason: "poke failed".to_string(),
            }],
            throughput_cv: Some(0.02),
            release_build: true,
            allow_debug_benchmark: false,
        });

        match verdict.validity {
            Validity::Partial { reasons } => {
                assert!(reasons.iter().any(|reason| reason.contains("run-2")));
            }
            other => panic!("expected partial verdict, got {other:?}"),
        }
    }

    #[test]
    fn harness_summary_marks_high_cv_partial() {
        let verdict = evaluate_verdict(&RunSummaryInput {
            measured_run_count: 5,
            run_failures: Vec::new(),
            throughput_cv: Some(0.25),
            release_build: true,
            allow_debug_benchmark: false,
        });

        match verdict.validity {
            Validity::Partial { reasons } => {
                assert!(reasons.iter().any(|reason| reason.contains("throughput CV")));
            }
            other => panic!("expected partial verdict, got {other:?}"),
        }
    }

    #[test]
    fn harness_summary_rejects_debug_trusted_runs_by_default() {
        let verdict = evaluate_verdict(&RunSummaryInput {
            measured_run_count: 5,
            run_failures: Vec::new(),
            throughput_cv: Some(0.02),
            release_build: false,
            allow_debug_benchmark: false,
        });

        match verdict.validity {
            Validity::Invalid { reasons } => {
                assert!(reasons.iter().any(|reason| reason.contains("release")));
            }
            other => panic!("expected invalid verdict, got {other:?}"),
        }
    }
}
