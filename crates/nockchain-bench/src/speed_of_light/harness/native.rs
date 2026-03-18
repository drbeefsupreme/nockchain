use std::path::Path;

use futures::FutureExt;

use super::artifacts::{write_cpu_profile_artifact, write_verdict};
use super::case::{RequestedCase, ResolvedCase};
use super::execute::{cpu_profile_output_relative_path, execute_once, CpuProfileExecutionKind};
use super::orchestrate::{execute_trusted_run, prepare_output_root, TrustedBackend, TrustedRunResult};
use super::profiler::{
    build_run_once_command, CpuProfilerLaunchRequest, CpuProfilerLauncher,
    SystemCpuProfilerLauncher,
};
use super::provenance::{BackendRuntimeFacts, Provenance};
use super::summary::{RunSummary, Validity, Verdict};
use super::{CpuProfilerConfig, HarnessError};

#[derive(Debug)]
pub struct NativeRunResult {
    pub resolved: ResolvedCase,
    pub provenance: Provenance,
    pub summary: RunSummary,
    pub verdict: Verdict,
}

impl From<TrustedRunResult> for NativeRunResult {
    fn from(value: TrustedRunResult) -> Self {
        Self {
            resolved: value.resolved,
            provenance: value.provenance,
            summary: value.summary,
            verdict: value.verdict,
        }
    }
}

pub async fn execute_native_trusted_run(
    requested: RequestedCase,
    output_root: &Path,
    allow_debug_benchmark: bool,
    cpu_profiler: Option<CpuProfilerConfig>,
) -> Result<NativeRunResult, HarnessError> {
    execute_native_trusted_run_with_backend_and_profiler(
        NativeBackend, SystemCpuProfilerLauncher, requested, output_root, allow_debug_benchmark,
        cpu_profiler,
    )
    .await
}

pub async fn execute_native_cpu_profile(
    output_root: &Path,
    cpu_profiler: CpuProfilerConfig,
) -> Result<super::execute::CpuProfileArtifact, HarnessError> {
    let request = build_native_profiler_request(output_root, cpu_profiler)?;
    let mut launcher = SystemCpuProfilerLauncher;
    launcher.preflight(&request).await?;
    launcher.launch(&request).await
}

#[cfg(test)]
async fn execute_native_trusted_run_with_backend<B: TrustedBackend>(
    backend: B,
    requested: RequestedCase,
    output_root: &Path,
    allow_debug_benchmark: bool,
) -> Result<NativeRunResult, HarnessError> {
    execute_native_trusted_run_with_backend_and_profiler(
        backend, SystemCpuProfilerLauncher, requested, output_root, allow_debug_benchmark, None,
    )
    .await
}

async fn execute_native_trusted_run_with_backend_and_profiler<
    B: TrustedBackend,
    P: CpuProfilerLauncher,
>(
    backend: B,
    profiler_launcher: P,
    requested: RequestedCase,
    output_root: &Path,
    allow_debug_benchmark: bool,
    cpu_profiler: Option<CpuProfilerConfig>,
) -> Result<NativeRunResult, HarnessError> {
    execute_native_trusted_run_with_backend_and_profiling_hooks(
        backend, profiler_launcher, build_native_profiler_request, write_cpu_profile_artifact,
        requested, output_root, allow_debug_benchmark, cpu_profiler,
    )
    .await
}

async fn execute_native_trusted_run_with_backend_and_profiling_hooks<
    B: TrustedBackend,
    P: CpuProfilerLauncher,
    R,
    W,
>(
    backend: B,
    mut profiler_launcher: P,
    mut request_builder: R,
    mut artifact_writer: W,
    requested: RequestedCase,
    output_root: &Path,
    allow_debug_benchmark: bool,
    cpu_profiler: Option<CpuProfilerConfig>,
) -> Result<NativeRunResult, HarnessError>
where
    R: FnMut(&Path, CpuProfilerConfig) -> Result<CpuProfilerLaunchRequest, HarnessError>,
    W: FnMut(&Path, &super::execute::CpuProfileArtifact) -> Result<(), HarnessError>,
{
    if cpu_profiler.is_some() {
        prepare_output_root(output_root)?;
    }

    let profiling_request = if let Some(config) = cpu_profiler {
        let request = match request_builder(output_root, config) {
            Ok(request) => request,
            Err(error) => {
                invalidate_verdict_for_cpu_profiling_failure(output_root, &error)?;
                return Err(error);
            }
        };
        if let Err(error) = profiler_launcher.preflight(&request).await {
            invalidate_verdict_for_cpu_profiling_failure(output_root, &error)?;
            return Err(error);
        }
        Some(request)
    } else {
        None
    };

    let run = execute_trusted_run(backend, requested, output_root, allow_debug_benchmark).await?;
    if let Some(request) = profiling_request {
        let profiling_result = async {
            let artifact = profiler_launcher.launch(&request).await?;
            artifact_writer(output_root, &artifact)
        }
        .await;

        if let Err(error) = profiling_result {
            invalidate_verdict_for_cpu_profiling_failure(output_root, &error)?;
            return Err(error);
        }
    }
    Ok(run.into())
}

fn build_native_profiler_request(
    output_root: &Path,
    config: CpuProfilerConfig,
) -> Result<CpuProfilerLaunchRequest, HarnessError> {
    let current_binary = std::env::current_exe()?;
    let resolved_case_path = output_root.join("resolved_case.json");
    let profile_run_dir = output_root.join("profile-run");
    let output_relative_path = cpu_profile_output_relative_path(config.kind);
    let profiled_command = build_run_once_command(
        &path_string(&current_binary),
        &path_string(&resolved_case_path),
        &path_string(&profile_run_dir),
        "profile",
    );

    Ok(CpuProfilerLaunchRequest {
        profiler_kind: config.kind,
        sample_rate_hz: config.sample_rate_hz,
        execution_kind: CpuProfileExecutionKind::Native,
        case_root: output_root.to_path_buf(),
        output_relative_path,
        profiled_run_dir: profile_run_dir,
        profiled_command,
    })
}

fn invalidate_verdict_for_cpu_profiling_failure(
    output_root: &Path,
    error: &HarnessError,
) -> Result<(), HarnessError> {
    std::fs::create_dir_all(output_root)?;
    write_verdict(
        output_root,
        &Verdict {
            validity: Validity::Invalid {
                reasons: vec![format!("cpu profiling failed: {error}")],
            },
        },
    )
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

struct NativeBackend;

impl TrustedBackend for NativeBackend {
    fn execute_run<'a>(
        &'a mut self,
        resolved: &'a ResolvedCase,
        run_id: &'a str,
        run_dir: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<super::execute::CompletedRun, HarnessError>> {
        execute_once(resolved, run_id, run_dir).boxed()
    }

    fn prepare<'a>(
        &'a mut self,
        _resolved: &'a ResolvedCase,
        _output_root: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<(), HarnessError>> {
        async { Ok(()) }.boxed()
    }

    fn capture_runtime_facts(&self) -> Result<BackendRuntimeFacts, HarnessError> {
        Ok(BackendRuntimeFacts::Native)
    }

    fn capture_raw_evidence<'a>(
        &'a self,
        _raw_dir: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<(), HarnessError>> {
        async { Ok(()) }.boxed()
    }

    fn cleanup<'a>(&'a mut self) -> futures::future::BoxFuture<'a, Result<(), HarnessError>> {
        async { Ok(()) }.boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    use futures::FutureExt;
    use tempfile::tempdir;

    use super::{
        build_native_profiler_request, execute_native_trusted_run_with_backend,
        execute_native_trusted_run_with_backend_and_profiler,
        execute_native_trusted_run_with_backend_and_profiling_hooks, NativeRunResult,
    };
    use crate::speed_of_light::fixture::{write_fixture_file, SolFixtureFile, SolFixtureManifest};
    use crate::speed_of_light::harness::artifacts::{
        read_cpu_profile_artifact, write_cpu_profile_artifact, write_run_artifacts,
    };
    use crate::speed_of_light::harness::case::{
        BinaryIdentity, ExecutionConfig, RequestedCase, ResolvedCase,
    };
    use crate::speed_of_light::harness::execute::{
        cpu_profile_output_relative_path, BlockTimingRecord, CompletedRun, CpuProfileArtifact,
        CpuProfileExecutionKind, RunRecord,
    };
    use crate::speed_of_light::harness::orchestrate::{
        prepare_output_root, TrustedBackend, TrustedRunResult,
    };
    use crate::speed_of_light::harness::provenance::{
        BackendRuntimeFacts, HostIdentity, Provenance,
    };
    use crate::speed_of_light::harness::summary::{RunSummary, Validity, Verdict};
    use crate::speed_of_light::harness::{
        CpuProfilerConfig, CpuProfilerKind, CpuProfilerLaunchRequest, CpuProfilerLauncher,
        HarnessError, SCHEMA_VERSION,
    };
    use crate::speed_of_light::types::SolHeight;

    #[test]
    fn native_run_rejects_non_empty_output_root() {
        let tempdir = tempdir().expect("tempdir");
        std::fs::write(tempdir.path().join("stale.txt"), "stale").expect("stale file");

        let error = prepare_output_root(tempdir.path()).expect_err("should reject stale output");
        assert!(error
            .to_string()
            .contains("already exists and is not empty"));
    }

    #[test]
    fn native_run_allows_empty_output_root() {
        let tempdir = tempdir().expect("tempdir");
        prepare_output_root(tempdir.path()).expect("empty dir should be allowed");
    }

    #[test]
    fn native_run_result_converts_from_trusted_run_result() {
        let requested = RequestedCase::native(PathBuf::from("fixture.soltest"));
        let resolved = ResolvedCase {
            schema_version: SCHEMA_VERSION.to_string(),
            requested: requested.clone(),
            absolute_fixture_path: PathBuf::from("/tmp/fixture.soltest"),
            fixture_sha256_hex: "abc".to_string(),
            fixture_manifest: SolFixtureManifest {
                format_version: 3,
                source_archive_path: "archive.solarch".to_string(),
                source_archive_event_num: Some(1),
                checkpoint_kind: crate::speed_of_light::SolFixtureCheckpointKind::Derived,
                checkpoint_height: SolHeight(1),
                checkpoint_event_num: 1,
                archive_start_height: SolHeight(2),
                archive_end_height: SolHeight(3),
                include_mempool: false,
                chunk_size: 8,
                kernel_hash_hex: "kernel".to_string(),
                checkpoint_hash_hex: "checkpoint".to_string(),
                archive_hash_hex: "archive".to_string(),
            },
            execution_config: ExecutionConfig::default(),
            binary: BinaryIdentity {
                version: "0.1.0".to_string(),
                build_profile: "release".to_string(),
                git_commit: None,
            },
            docker: None,
        };
        let trusted = TrustedRunResult {
            resolved: resolved.clone(),
            provenance: Provenance {
                schema_version: SCHEMA_VERSION.to_string(),
                capture_timestamp_ms: 1,
                host: HostIdentity {
                    hostname: Some("host".to_string()),
                    os: "linux".to_string(),
                    arch: "x86_64".to_string(),
                    kernel: None,
                    cpu_count: 4,
                    total_memory_bytes: None,
                    cpu_model: None,
                },
                git: None,
                backend: BackendRuntimeFacts::Native,
                binary: resolved.binary.clone(),
                fixture_path: resolved.absolute_fixture_path.clone(),
                fixture_sha256_hex: resolved.fixture_sha256_hex.clone(),
                fixture_manifest: resolved.fixture_manifest.clone(),
            },
            summary: RunSummary {
                measured_runs_requested: 3,
                measured_runs_succeeded: 3,
                failed_runs: Vec::new(),
                throughput_blocks_per_second: None,
                init_time_secs: None,
                total_replay_time_secs: None,
                average_block_time_ms: None,
                failed_pokes: None,
                checkpoint_count: None,
                average_checkpoint_time_secs: None,
                peak_process_rss_bytes: None,
                minor_faults_total: None,
                major_faults_total: None,
            },
            verdict: Verdict {
                validity: Validity::Valid,
            },
        };

        let native = NativeRunResult::from(trusted);

        assert_eq!(native.resolved, resolved);
        assert_eq!(native.provenance.backend, BackendRuntimeFacts::Native);
        assert_eq!(native.verdict.validity, Validity::Valid);
    }

    #[tokio::test]
    async fn native_trusted_run_preserves_artifact_semantics_after_refactor() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let output_root = tempdir.path().join("out");
        let backend = FakeNativeBackend::successful();
        let events = backend.shared_events();

        let result =
            execute_native_trusted_run_with_backend(backend, requested, &output_root, false)
                .await
                .expect("native trusted run result");

        assert_eq!(
            events.lock().expect("events").clone(),
            vec![
                "prepare", "runtime-facts", "raw-evidence", "warmup-0", "run-0", "run-1", "run-2",
                "cleanup",
            ]
        );
        assert_eq!(result.provenance.backend, BackendRuntimeFacts::Native);
        assert_eq!(
            result.provenance.binary.git_commit,
            result.resolved.binary.git_commit
        );
        assert_eq!(result.summary.measured_runs_requested, 3);
        assert_eq!(result.summary.measured_runs_succeeded, 3);
        assert_eq!(result.verdict.validity, Validity::Valid);

        let root_entries = sorted_relative_paths(&output_root);
        assert_eq!(
            root_entries,
            vec![
                "provenance.json", "raw", "raw/host_env.json", "requested_case.json",
                "resolved_case.json", "runs", "runs/run-0", "runs/run-0/block_timings.ndjson",
                "runs/run-0/result.json", "runs/run-0/stderr.log", "runs/run-0/stdout.log",
                "runs/run-1", "runs/run-1/block_timings.ndjson", "runs/run-1/result.json",
                "runs/run-1/stderr.log", "runs/run-1/stdout.log", "runs/run-2",
                "runs/run-2/block_timings.ndjson", "runs/run-2/result.json",
                "runs/run-2/stderr.log", "runs/run-2/stdout.log", "runs/warmup-0",
                "runs/warmup-0/block_timings.ndjson", "runs/warmup-0/result.json",
                "runs/warmup-0/stderr.log", "runs/warmup-0/stdout.log", "schema_version.txt",
                "summary.json", "verdict.json",
            ]
        );

        assert_eq!(
            normalized_json(&output_root.join("requested_case.json")),
            serde_json::json!({
                "benchmark": "sol-replay",
                "blocks": 0,
                "checkpoint_every_blocks": 0,
                "cooldown_secs": 0,
                "enable_checkpointing": true,
                "execution": "Native",
                "fixture_path": tempdir.path().join("fixture.soltest"),
                "label": null,
                "measured_runs": 3,
                "profile_interval_ms": 500,
                "profile_memory": false,
                "skip_genesis": false,
                "threads": 1,
                "warmup_runs": 1,
            })
        );
        assert_eq!(
            normalized_json(&output_root.join("resolved_case.json")),
            serde_json::json!({
                "absolute_fixture_path": tempdir.path().join("fixture.soltest"),
                "binary": {
                    "build_profile": "release",
                    "git_commit": "<normalized>",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "execution_config": {
                    "checkpoint_recovery_timeout_ms": 5_000,
                    "checkpoint_recovery_tolerance_pct_bps": 500,
                    "gc_drop_threshold_mib": 64,
                    "page_fault_major_burst_threshold": 1,
                    "page_fault_minor_burst_threshold": 50_000,
                },
                "fixture_manifest": {
                    "archive_end_height": 3,
                    "archive_hash_hex": "archive",
                    "archive_start_height": 2,
                    "checkpoint_hash_hex": "checkpoint",
                    "checkpoint_event_num": 1,
                    "checkpoint_height": 1,
                    "checkpoint_kind": "derived",
                    "chunk_size": 8,
                    "format_version": 3,
                    "include_mempool": false,
                    "kernel_hash_hex": "kernel",
                    "source_archive_event_num": 1,
                    "source_archive_path": "archive.solarch",
                },
                "fixture_sha256_hex": "<normalized>",
                "requested": {
                    "benchmark": "sol-replay",
                    "blocks": 0,
                    "checkpoint_every_blocks": 0,
                    "cooldown_secs": 0,
                    "enable_checkpointing": true,
                    "execution": "Native",
                    "fixture_path": tempdir.path().join("fixture.soltest"),
                    "label": null,
                    "measured_runs": 3,
                    "profile_interval_ms": 500,
                    "profile_memory": false,
                    "skip_genesis": false,
                    "threads": 1,
                    "warmup_runs": 1,
                },
                "schema_version": SCHEMA_VERSION,
            })
        );
        assert_eq!(
            normalized_json(&output_root.join("summary.json")),
            serde_json::json!({
                "average_block_time_ms": uniform_stats_json(100.0),
                "average_checkpoint_time_secs": uniform_stats_json(0.5),
                "checkpoint_count": uniform_stats_json(1.0),
                "failed_pokes": uniform_stats_json(0.0),
                "failed_runs": [],
                "init_time_secs": uniform_stats_json(1.0),
                "major_faults_total": uniform_stats_json(0.0),
                "measured_runs_requested": 3,
                "measured_runs_succeeded": 3,
                "minor_faults_total": uniform_stats_json(10.0),
                "peak_process_rss_bytes": uniform_stats_json(128.0),
                "throughput_blocks_per_second": uniform_stats_json(10.0),
                "total_replay_time_secs": uniform_stats_json(2.0)
            })
        );
        assert_eq!(
            normalized_json(&output_root.join("verdict.json")),
            serde_json::json!({
                "validity": "Valid"
            })
        );
        assert_eq!(
            normalized_json(&output_root.join("provenance.json")),
            serde_json::json!({
                "backend": "Native",
                "binary": {
                    "build_profile": "release",
                    "git_commit": "<normalized>",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "capture_timestamp_ms": "<normalized>",
                "fixture_manifest": {
                    "archive_end_height": 3,
                    "archive_hash_hex": "archive",
                    "archive_start_height": 2,
                    "checkpoint_hash_hex": "checkpoint",
                    "checkpoint_event_num": 1,
                    "checkpoint_height": 1,
                    "checkpoint_kind": "derived",
                    "chunk_size": 8,
                    "format_version": 3,
                    "include_mempool": false,
                    "kernel_hash_hex": "kernel",
                    "source_archive_event_num": 1,
                    "source_archive_path": "archive.solarch",
                },
                "fixture_path": tempdir.path().join("fixture.soltest"),
                "fixture_sha256_hex": "<normalized>",
                "git": "<normalized>",
                "host": "<normalized>",
                "schema_version": SCHEMA_VERSION,
            })
        );
    }

    #[tokio::test]
    async fn native_trusted_run_writes_cpu_profile_artifacts() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let output_root = tempdir.path().join("out");
        let backend = FakeNativeBackend::successful();
        let events = backend.shared_events();
        let profiler = FakeCpuProfilerLauncher::new(events.clone());

        let result = execute_native_trusted_run_with_backend_and_profiler(
            backend,
            profiler,
            requested,
            &output_root,
            false,
            Some(CpuProfilerConfig {
                kind: CpuProfilerKind::Samply,
                sample_rate_hz: 1_000,
            }),
        )
        .await
        .expect("native trusted run result");

        assert_eq!(
            events.lock().expect("events").clone(),
            vec![
                "prepare", "runtime-facts", "raw-evidence", "warmup-0", "run-0", "run-1", "run-2",
                "cleanup", "profile",
            ]
        );
        assert_eq!(result.summary.measured_runs_requested, 3);
        assert_eq!(result.summary.measured_runs_succeeded, 3);
        assert_eq!(result.verdict.validity, Validity::Valid);

        let artifact = read_cpu_profile_artifact(&output_root).expect("cpu profile artifact");
        assert_eq!(artifact.profiler_kind, CpuProfilerKind::Samply);
        assert_eq!(artifact.sample_rate_hz, 1_000);
        assert_eq!(artifact.execution_kind, CpuProfileExecutionKind::Native);
        assert_eq!(
            artifact.output_relative_path,
            cpu_profile_output_relative_path(CpuProfilerKind::Samply)
        );
        assert!(artifact
            .profiled_command
            .iter()
            .any(|arg| arg == "run-once"));
        assert!(output_root.join("cpu_profile.json").exists());
        assert!(output_root.join("profiles/samply-profile.json.gz").exists());
        assert!(output_root.join("profile-run/result.json").exists());
    }

    #[tokio::test]
    async fn native_trusted_run_marks_verdict_invalid_when_cpu_profiling_fails() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let output_root = tempdir.path().join("out");
        let backend = FakeNativeBackend::successful();

        let error = execute_native_trusted_run_with_backend_and_profiler(
            backend,
            FailingCpuProfilerLauncher,
            requested,
            &output_root,
            false,
            Some(CpuProfilerConfig {
                kind: CpuProfilerKind::Samply,
                sample_rate_hz: 1_000,
            }),
        )
        .await
        .expect_err("profiling failure should fail the case");

        assert!(error.to_string().contains("samply"));
        let verdict = normalized_json(&output_root.join("verdict.json"));
        assert_eq!(
            verdict,
            serde_json::json!({
                "validity": {
                    "Invalid": {
                        "reasons": [format!("cpu profiling failed: {error}")]
                    }
                }
            })
        );
    }

    #[tokio::test]
    async fn native_trusted_run_marks_verdict_invalid_when_cpu_profile_request_build_fails() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let output_root = tempdir.path().join("out");
        let backend = FakeNativeBackend::successful();

        let error = execute_native_trusted_run_with_backend_and_profiling_hooks(
            backend,
            FakeCpuProfilerLauncher::new(Arc::new(Mutex::new(Vec::new()))),
            |_output_root: &Path,
             _config: CpuProfilerConfig|
             -> Result<CpuProfilerLaunchRequest, HarnessError> {
                Err(HarnessError::CommandFailure(
                    "request build failed".to_string(),
                ))
            },
            write_cpu_profile_artifact,
            requested,
            &output_root,
            false,
            Some(CpuProfilerConfig {
                kind: CpuProfilerKind::Samply,
                sample_rate_hz: 1_000,
            }),
        )
        .await
        .expect_err("request build failure should fail the case");

        assert!(error.to_string().contains("request build failed"));
        let verdict = normalized_json(&output_root.join("verdict.json"));
        assert_eq!(
            verdict,
            serde_json::json!({
                "validity": {
                    "Invalid": {
                        "reasons": [format!("cpu profiling failed: {error}")]
                    }
                }
            })
        );
    }

    #[tokio::test]
    async fn native_trusted_run_preflight_failure_rejects_stale_output_root() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let output_root = tempdir.path().join("out");
        std::fs::create_dir_all(&output_root).expect("output root");
        std::fs::write(output_root.join("stale.txt"), "stale").expect("stale file");
        let backend = FakeNativeBackend::successful();
        let events = backend.shared_events();

        let error = execute_native_trusted_run_with_backend_and_profiling_hooks(
            backend,
            PreflightFailingCpuProfilerLauncher,
            build_native_profiler_request,
            write_cpu_profile_artifact,
            requested,
            &output_root,
            false,
            Some(CpuProfilerConfig {
                kind: CpuProfilerKind::Samply,
                sample_rate_hz: 1_000,
            }),
        )
        .await
        .expect_err("stale output root should be rejected before profiling preflight");

        assert!(error
            .to_string()
            .contains("already exists and is not empty"));
        assert!(events.lock().expect("events").is_empty());
        assert!(output_root.join("stale.txt").exists());
        assert!(!output_root.join("verdict.json").exists());
    }

    #[tokio::test]
    async fn native_trusted_run_preflight_failure_stops_before_trusted_runs() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let output_root = tempdir.path().join("out");
        let backend = FakeNativeBackend::successful();
        let events = backend.shared_events();

        let error = execute_native_trusted_run_with_backend_and_profiling_hooks(
            backend,
            PreflightFailingCpuProfilerLauncher,
            build_native_profiler_request,
            write_cpu_profile_artifact,
            requested,
            &output_root,
            false,
            Some(CpuProfilerConfig {
                kind: CpuProfilerKind::Samply,
                sample_rate_hz: 1_000,
            }),
        )
        .await
        .expect_err("preflight failure should fail the case before trusted runs");

        assert!(error.to_string().contains("preflight"));
        assert!(events.lock().expect("events").is_empty());
        let verdict = normalized_json(&output_root.join("verdict.json"));
        assert_eq!(
            verdict,
            serde_json::json!({
                "validity": {
                    "Invalid": {
                        "reasons": [format!("cpu profiling failed: {error}")]
                    }
                }
            })
        );
    }

    #[tokio::test]
    async fn native_trusted_run_marks_verdict_invalid_when_cpu_profile_artifact_write_fails() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let output_root = tempdir.path().join("out");
        let backend = FakeNativeBackend::successful();
        let events = backend.shared_events();
        let profiler = FakeCpuProfilerLauncher::new(events);

        let error = execute_native_trusted_run_with_backend_and_profiling_hooks(
            backend,
            profiler,
            build_native_profiler_request,
            |_output_root: &Path, _artifact: &CpuProfileArtifact| -> Result<(), HarnessError> {
                Err(HarnessError::CommandFailure(
                    "persisting cpu profile artifact failed".to_string(),
                ))
            },
            requested,
            &output_root,
            false,
            Some(CpuProfilerConfig {
                kind: CpuProfilerKind::Samply,
                sample_rate_hz: 1_000,
            }),
        )
        .await
        .expect_err("artifact write failure should fail the case");

        assert!(error
            .to_string()
            .contains("persisting cpu profile artifact failed"));
        let verdict = normalized_json(&output_root.join("verdict.json"));
        assert_eq!(
            verdict,
            serde_json::json!({
                "validity": {
                    "Invalid": {
                        "reasons": [format!("cpu profiling failed: {error}")]
                    }
                }
            })
        );
    }

    struct FakeNativeBackend {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl FakeNativeBackend {
        fn successful() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn shared_events(&self) -> Arc<Mutex<Vec<String>>> {
            Arc::clone(&self.events)
        }
    }

    impl TrustedBackend for FakeNativeBackend {
        fn execute_run<'a>(
            &'a mut self,
            _resolved: &'a ResolvedCase,
            run_id: &'a str,
            run_dir: &'a Path,
        ) -> futures::future::BoxFuture<
            'a,
            Result<CompletedRun, crate::speed_of_light::harness::HarnessError>,
        > {
            self.events.lock().expect("events").push(run_id.to_string());
            let run_dir = run_dir.to_path_buf();
            async move {
                let completed = completed_run(run_id);
                write_run_artifacts(&run_dir, &completed).expect("run artifacts");
                Ok(completed)
            }
            .boxed()
        }

        fn prepare<'a>(
            &'a mut self,
            _resolved: &'a ResolvedCase,
            _output_root: &'a Path,
        ) -> futures::future::BoxFuture<'a, Result<(), crate::speed_of_light::harness::HarnessError>>
        {
            self.events
                .lock()
                .expect("events")
                .push("prepare".to_string());
            async { Ok(()) }.boxed()
        }

        fn capture_runtime_facts(
            &self,
        ) -> Result<BackendRuntimeFacts, crate::speed_of_light::harness::HarnessError> {
            self.events
                .lock()
                .expect("events")
                .push("runtime-facts".to_string());
            Ok(BackendRuntimeFacts::Native)
        }

        fn capture_raw_evidence<'a>(
            &'a self,
            _raw_dir: &'a Path,
        ) -> futures::future::BoxFuture<'a, Result<(), crate::speed_of_light::harness::HarnessError>>
        {
            self.events
                .lock()
                .expect("events")
                .push("raw-evidence".to_string());
            async { Ok(()) }.boxed()
        }

        fn cleanup<'a>(
            &'a mut self,
        ) -> futures::future::BoxFuture<'a, Result<(), crate::speed_of_light::harness::HarnessError>>
        {
            self.events
                .lock()
                .expect("events")
                .push("cleanup".to_string());
            async { Ok(()) }.boxed()
        }
    }

    struct FakeCpuProfilerLauncher {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl FakeCpuProfilerLauncher {
        fn new(events: Arc<Mutex<Vec<String>>>) -> Self {
            Self { events }
        }
    }

    impl CpuProfilerLauncher for FakeCpuProfilerLauncher {
        fn launch<'a>(
            &'a mut self,
            request: &'a CpuProfilerLaunchRequest,
        ) -> futures::future::BoxFuture<'a, Result<CpuProfileArtifact, HarnessError>> {
            self.events
                .lock()
                .expect("events")
                .push("profile".to_string());

            async move {
                let output_path = request.case_root.join(&request.output_relative_path);
                if let Some(parent) = output_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&output_path, "profile")?;

                write_run_artifacts(&request.profiled_run_dir, &completed_run("profile"))?;

                Ok(request.artifact())
            }
            .boxed()
        }
    }

    struct FailingCpuProfilerLauncher;

    impl CpuProfilerLauncher for FailingCpuProfilerLauncher {
        fn launch<'a>(
            &'a mut self,
            _request: &'a CpuProfilerLaunchRequest,
        ) -> futures::future::BoxFuture<'a, Result<CpuProfileArtifact, HarnessError>> {
            async {
                Err(HarnessError::CommandFailure(
                    "samply is not installed or not on PATH".to_string(),
                ))
            }
            .boxed()
        }
    }

    struct PreflightFailingCpuProfilerLauncher;

    impl CpuProfilerLauncher for PreflightFailingCpuProfilerLauncher {
        fn preflight<'a>(
            &'a self,
            _request: &'a CpuProfilerLaunchRequest,
        ) -> futures::future::BoxFuture<'a, Result<(), HarnessError>> {
            async {
                Err(HarnessError::CommandFailure(
                    "preflight failed: samply is not installed".to_string(),
                ))
            }
            .boxed()
        }

        fn launch<'a>(
            &'a mut self,
            _request: &'a CpuProfilerLaunchRequest,
        ) -> futures::future::BoxFuture<'a, Result<CpuProfileArtifact, HarnessError>> {
            async { panic!("launch should not run after preflight failure") }.boxed()
        }
    }

    fn completed_run(run_id: &str) -> CompletedRun {
        CompletedRun {
            record: RunRecord {
                run_id: run_id.to_string(),
                success: true,
                error: None,
                blocks_poked: 1,
                failed_pokes: 0,
                init_time_secs: 1.0,
                total_replay_time_secs: 2.0,
                throughput_blocks_per_second: 10.0,
                average_block_time_ms: 100.0,
                checkpoint_count: 1,
                checkpoint_total_time_secs: 0.5,
                average_checkpoint_time_secs: 0.5,
                peak_process_rss_bytes: Some(128.0),
                minor_faults_total: Some(10.0),
                major_faults_total: Some(0.0),
            },
            block_timings: vec![BlockTimingRecord {
                height: 2,
                duration_ms: 10.0,
            }],
            profile: None,
            bench_results: None,
        }
    }

    fn write_requested_case(root: &Path) -> RequestedCase {
        let fixture_path = root.join("fixture.soltest");
        write_fixture_file(&fixture_path, &fixture_file()).expect("fixture");

        let mut requested = RequestedCase::native(PathBuf::from(&fixture_path));
        requested.warmup_runs = 1;
        requested.measured_runs = 3;
        requested.cooldown_secs = 0;
        requested
    }

    fn fixture_file() -> SolFixtureFile {
        SolFixtureFile {
            manifest: SolFixtureManifest {
                format_version: 3,
                source_archive_path: "archive.solarch".to_string(),
                source_archive_event_num: Some(1),
                checkpoint_kind: crate::speed_of_light::SolFixtureCheckpointKind::Derived,
                checkpoint_height: SolHeight(1),
                checkpoint_event_num: 1,
                archive_start_height: SolHeight(2),
                archive_end_height: SolHeight(3),
                include_mempool: false,
                chunk_size: 8,
                kernel_hash_hex: "kernel".to_string(),
                checkpoint_hash_hex: "checkpoint".to_string(),
                archive_hash_hex: "archive".to_string(),
            },
            checkpoint_bytes: vec![1, 2, 3],
            archive_bytes: vec![4, 5, 6],
            kernel_bytes: vec![7, 8, 9],
        }
    }

    fn sorted_relative_paths(root: &Path) -> Vec<String> {
        fn visit(root: &Path, dir: &Path, entries: &mut Vec<String>) {
            let mut children: Vec<_> = std::fs::read_dir(dir)
                .expect("read dir")
                .map(|entry| entry.expect("entry").path())
                .collect();
            children.sort();
            for path in children {
                let relative = path
                    .strip_prefix(root)
                    .expect("relative path")
                    .to_string_lossy()
                    .to_string();
                entries.push(relative);
                if path.is_dir() {
                    visit(root, &path, entries);
                }
            }
        }

        let mut entries = Vec::new();
        visit(root, root, &mut entries);
        entries
    }

    fn normalized_json(path: &Path) -> serde_json::Value {
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).expect("read json")).expect("json");

        if path.ends_with("resolved_case.json") || path.ends_with("provenance.json") {
            if let Some(object) = value.as_object_mut() {
                object.insert(
                    "fixture_sha256_hex".to_string(),
                    serde_json::Value::String("<normalized>".to_string()),
                );
                if let Some(binary) = object
                    .get_mut("binary")
                    .and_then(serde_json::Value::as_object_mut)
                {
                    binary.insert(
                        "git_commit".to_string(),
                        serde_json::Value::String("<normalized>".to_string()),
                    );
                }

                if path.ends_with("provenance.json") {
                    object.insert(
                        "capture_timestamp_ms".to_string(),
                        serde_json::Value::String("<normalized>".to_string()),
                    );
                    object.insert(
                        "host".to_string(),
                        serde_json::Value::String("<normalized>".to_string()),
                    );
                    object.insert(
                        "git".to_string(),
                        serde_json::Value::String("<normalized>".to_string()),
                    );
                }
            }
        }

        value
    }

    fn uniform_stats_json(value: f64) -> serde_json::Value {
        serde_json::json!({
            "cv": 0.0,
            "mad": 0.0,
            "max": value,
            "median": value,
            "min": value,
            "stddev": 0.0,
            "values": [value, value, value]
        })
    }
}
