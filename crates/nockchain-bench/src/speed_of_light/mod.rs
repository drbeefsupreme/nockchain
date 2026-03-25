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
//! - Archive extraction via checkpoint peek
//! - Archive format for persisting extracted data to disk
//! - Benchmark runner for injection testing

pub mod archive;
pub mod bench;
pub mod checkpoint;
pub mod checkpoint_builder;
pub mod extractor;
pub mod fixture;
pub mod harness;
pub mod kernel_utils;
pub mod mempool_inspector;
mod noun_compat;
pub mod poke;
pub mod profiling;
mod runtime_compat;
pub mod start_height;
pub mod types;

pub use archive::{
    ArchiveFilter, ArchiveMetadata, ArchiveSliceResult, BlockEntry, ByteOffset, ByteSize,
    MempoolSnapshotEntry, MempoolTxEntry, SolArchiveReader, SolArchiveWriter, slice_archive_file,
};
pub use bench::{SolBenchConfig, SolBenchResults, SolBenchRunner};
pub use checkpoint::{checkpoint_event_num, load_checkpoint};
pub use checkpoint_builder::{
    CheckpointBuildError, CheckpointBuildMode, CheckpointBuilder, CheckpointConfig,
    CheckpointResult,
};
pub use extractor::{
    ArchiveExtractionPhase, ArchiveExtractionProgress, BlockExtractor, ExtractorConfig,
};
pub use fixture::{
    FixtureError, SolFixtureCheckpointKind, SolFixtureFile, SolFixtureManifest,
    extract_fixture_to_paths, read_fixture_file, write_fixture_file, write_fixture_file_from_paths,
};
pub use harness::docker::{
    ContainerStats, DockerRunPlan, HarnessDockerError, connect_docker, execute_docker_validation,
    parse_memory_limit, parse_proc_stat_faults,
};
pub use harness::{
    AxisValue, CpuProfileArtifact, CpuProfileExecutionKind, CpuProfilerConfig, CpuProfilerKind,
    DockerImageSource, DockerImageVariant, DockerResolvedConfig, ExecuteOptions, ExecutionConfig,
    ExecutionRequest, ExpandedCase, HarnessSweepExecutor, RequestedCase, ResolvedCase,
    ResolvedDockerImage, RunFailure, RunMetrics, RunSummary, RunSummaryInput, ScheduleMode,
    SweepComparison, SweepMatrix, SweepMatrixFile, SweepResult, SweepRunOptions, SweepSchedule,
    Validity, ValueStats, Verdict, WorkDirMode, capture_native_provenance,
    cpu_profile_output_relative_path, current_binary_identity, evaluate_validation_probe,
    evaluate_verdict, execute_docker_trusted_run, execute_native_cpu_profile,
    execute_native_trusted_run, execute_once, execute_once_with_options, execute_sweep,
    expand_matrix, parse_matrix_value, resolve_requested_case, run_validation_probe,
};
pub use mempool_inspector::{InspectorError, StaleTxRange, find_stale_ranges};
pub use profiling::{
    CheckpointProfile, GcEvent, MemoryProfile, PageFaultBurst, PhaseKind, PhaseSummary,
    PhaseWindow, ProcessMemoryProfiler, SolScorecard, build_scorecard, find_recovery_ms,
    infer_gc_events, infer_page_fault_bursts, summarize_phases,
};
pub use start_height::{StartHeightError, resolve_start_height};
pub use types::{PROOF_VERSION_1_START, PROOF_VERSION_2_START, ProofVersion, SolHeight};

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::harness::{
        ExecutionRequest, RequestedCase, RunFailure, RunSummaryInput, Validity, evaluate_verdict,
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
    fn harness_requested_case_stays_spec_authoritative() {
        let requested = RequestedCase::native(PathBuf::from("fixture.soltest"));
        let value = serde_json::to_value(&requested).expect("serialize requested case");
        let object = value.as_object().expect("requested case object");

        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut expected = vec![
            "benchmark", "label", "fixture_path", "blocks", "skip_genesis", "enable_checkpointing",
            "checkpoint_every_blocks", "profile_memory", "profile_interval_ms", "execution",
            "threads", "warmup_runs", "measured_runs", "cooldown_secs",
        ];
        expected.sort_unstable();
        assert_eq!(keys, expected);

        assert_eq!(
            object.get("execution"),
            Some(&Value::String("Native".to_string()))
        );
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
            invalid_reasons: Vec::new(),
            partial_reasons: Vec::new(),
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
            invalid_reasons: Vec::new(),
            partial_reasons: Vec::new(),
        });

        match verdict.validity {
            Validity::Partial { reasons } => {
                assert!(
                    reasons
                        .iter()
                        .any(|reason| reason.contains("throughput CV"))
                );
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
            invalid_reasons: Vec::new(),
            partial_reasons: Vec::new(),
        });

        match verdict.validity {
            Validity::Invalid { reasons } => {
                assert!(reasons.iter().any(|reason| reason.contains("release")));
            }
            other => panic!("expected invalid verdict, got {other:?}"),
        }
    }

    #[test]
    fn harness_summary_rejects_debug_override_as_partial() {
        let verdict = evaluate_verdict(&RunSummaryInput {
            measured_run_count: 5,
            run_failures: Vec::new(),
            throughput_cv: Some(0.02),
            release_build: false,
            allow_debug_benchmark: true,
            invalid_reasons: Vec::new(),
            partial_reasons: Vec::new(),
        });

        match verdict.validity {
            Validity::Partial { reasons } => {
                assert!(reasons.iter().any(|reason| reason.contains("debug build")));
            }
            other => panic!("expected partial verdict, got {other:?}"),
        }
    }
}
