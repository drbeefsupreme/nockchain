use std::path::Path;
use std::time::Duration;

use futures::FutureExt;

use super::artifacts::{refresh_run_trace_artifacts, write_run_artifacts};
use super::case::{RequestedCase, ResolvedCase};
use super::orchestrate::{execute_trusted_run, TrustedBackend, TrustedRunResult};
use super::provenance::{BackendRuntimeFacts, Provenance};
use super::summary::{RunSummary, Verdict};
use super::tracy_capture::{
    ensure_tracy_capture_available, start_native_tracy_capture, TracyEndpoint,
};
use super::HarnessError;
use crate::speed_of_light::InvocationTracingConfig;

const NATIVE_TRACY_CAPTURE_EXIT_TIMEOUT: Duration = Duration::from_secs(5);
const NATIVE_TRACY_STARTUP_GRACE_PERIOD: Duration = Duration::from_millis(500);

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
    tracing: InvocationTracingConfig,
    output_root: &Path,
    allow_debug_benchmark: bool,
) -> Result<NativeRunResult, HarnessError> {
    execute_native_trusted_run_with_backend(
        NativeBackend, requested, tracing, output_root, allow_debug_benchmark,
    )
    .await
}

async fn execute_native_trusted_run_with_backend<B: TrustedBackend>(
    backend: B,
    requested: RequestedCase,
    tracing: InvocationTracingConfig,
    output_root: &Path,
    allow_debug_benchmark: bool,
) -> Result<NativeRunResult, HarnessError> {
    execute_trusted_run(
        backend, requested, tracing, output_root, allow_debug_benchmark,
    )
    .await
    .map(NativeRunResult::from)
}

struct NativeBackend;

impl TrustedBackend for NativeBackend {
    fn execute_run<'a>(
        &'a mut self,
        resolved: &'a ResolvedCase,
        tracing: &'a InvocationTracingConfig,
        run_id: &'a str,
        run_dir: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<super::execute::CompletedRun, HarnessError>> {
        async move {
            let options = super::execute::ExecuteOptions::from(&resolved.execution_config);
            let completed = if tracing.tracy != crate::speed_of_light::TracyMode::Off {
                let resolved_owned = resolved.clone();
                let tracing_owned = tracing.clone();
                let run_id_owned = run_id.to_string();
                let run_dir_owned = run_dir.to_path_buf();
                let options_owned = options.clone();
                let endpoint = TracyEndpoint::native();
                let mut tracy_capture = match start_native_tracy_capture(
                    &run_dir.join("tracy_capture.tracy"),
                    &endpoint,
                ) {
                    Ok(capture) => capture,
                    Err(error) => {
                        let completed = failed_completed_run(run_id, error.to_string());
                        write_run_artifacts(run_dir, &completed)?;
                        let _ = refresh_run_trace_artifacts(run_dir)?;
                        return Ok(completed);
                    }
                };
                if let Err(error) = tracy_capture.ensure_started() {
                    let completed = failed_completed_run(run_id, error.to_string());
                    write_run_artifacts(run_dir, &completed)?;
                    let _ = refresh_run_trace_artifacts(run_dir)?;
                    return Ok(completed);
                }
                tokio::time::sleep(NATIVE_TRACY_STARTUP_GRACE_PERIOD).await;

                let bench = tokio::spawn(async move {
                    super::execute::execute_once_with_options(
                        &resolved_owned,
                        &tracing_owned,
                        &run_id_owned,
                        &run_dir_owned,
                        &options_owned,
                    )
                    .await
                });

                let mut completed = bench.await.map_err(|error| {
                    HarnessError::CommandFailure(format!(
                        "native run task join failed: {error}"
                    ))
                })??;

                if let Err(error) =
                    tracy_capture.wait_for_natural_exit(NATIVE_TRACY_CAPTURE_EXIT_TIMEOUT)
                {
                    completed.record.success = false;
                    completed.record.error = Some(error.to_string());
                }
                completed
            } else {
                super::execute::execute_once_with_options(resolved, tracing, run_id, run_dir, &options)
                    .await?
            };

            write_run_artifacts(run_dir, &completed)?;
            let _ = refresh_run_trace_artifacts(run_dir)?;
            Ok(completed)
        }
        .boxed()
    }

    fn prepare<'a>(
        &'a mut self,
        _resolved: &'a ResolvedCase,
        tracing: &'a InvocationTracingConfig,
        _output_root: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<(), HarnessError>> {
        async move {
            if tracing.tracy != crate::speed_of_light::TracyMode::Off {
                ensure_tracy_capture_available()?;
            }
            Ok(())
        }
        .boxed()
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

fn failed_completed_run(run_id: &str, error: String) -> super::execute::CompletedRun {
    super::execute::CompletedRun {
        record: super::execute::RunRecord {
            run_id: run_id.to_string(),
            success: false,
            error: Some(error),
            blocks_poked: 0,
            failed_pokes: 0,
            init_time_secs: 0.0,
            total_replay_time_secs: 0.0,
            throughput_blocks_per_second: 0.0,
            average_block_time_ms: 0.0,
            checkpoint_count: 0,
            checkpoint_total_time_secs: 0.0,
            average_checkpoint_time_secs: 0.0,
            peak_process_rss_bytes: None,
            minor_faults_total: None,
            major_faults_total: None,
        },
        block_timings: Vec::new(),
        profile: None,
        bench_results: None,
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    use futures::FutureExt;
    use tempfile::tempdir;

    use super::{execute_native_trusted_run_with_backend, NativeRunResult};
    use crate::speed_of_light::fixture::{write_fixture_file, SolFixtureFile, SolFixtureManifest};
    use crate::speed_of_light::harness::artifacts::write_run_artifacts;
    use crate::speed_of_light::harness::case::{
        BinaryIdentity, ExecutionConfig, RequestedCase, ResolvedCase,
    };
    use crate::speed_of_light::harness::execute::{BlockTimingRecord, CompletedRun, RunRecord};
    use crate::speed_of_light::harness::orchestrate::{
        prepare_output_root, TrustedBackend, TrustedRunResult,
    };
    use crate::speed_of_light::harness::provenance::{
        BackendRuntimeFacts, HostIdentity, Provenance,
    };
    use crate::speed_of_light::harness::summary::{RunSummary, Validity, Verdict};
    use crate::speed_of_light::harness::SCHEMA_VERSION;
    use crate::speed_of_light::types::SolHeight;
    use crate::speed_of_light::InvocationTracingConfig;

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
                format_version: 2,
                source_archive_path: "archive.solarch".to_string(),
                source_archive_event_num: 1,
                derived_checkpoint_height: SolHeight(1),
                derived_checkpoint_event_num: 1,
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
                backend: Some(BackendRuntimeFacts::Native),
                binary: resolved.binary.clone(),
                fixture_path: resolved.absolute_fixture_path.clone(),
                fixture_sha256_hex: resolved.fixture_sha256_hex.clone(),
                fixture_manifest: resolved.fixture_manifest.clone(),
                tracing: InvocationTracingConfig::default().provenance(),
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
        assert_eq!(native.provenance.backend, Some(BackendRuntimeFacts::Native));
        assert_eq!(native.verdict.validity, Validity::Valid);
    }

    #[tokio::test]
    async fn native_trusted_run_preserves_artifact_semantics_after_refactor() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let output_root = tempdir.path().join("out");
        let backend = FakeNativeBackend::successful();
        let events = backend.shared_events();

        let result = execute_native_trusted_run_with_backend(
            backend,
            requested,
            InvocationTracingConfig::default(),
            &output_root,
            false,
        )
        .await
        .expect("native trusted run result");

        assert_eq!(
            events.lock().expect("events").clone(),
            vec![
                "prepare", "runtime-facts", "raw-evidence", "warmup-0", "run-0", "run-1", "run-2",
                "cleanup",
            ]
        );
        assert_eq!(result.provenance.backend, Some(BackendRuntimeFacts::Native));
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
                "runs/warmup-0/stderr.log", "runs/warmup-0/stdout.log", "runtime_config.json",
                "schema_version.txt", "summary.json", "verdict.json",
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
                    "chunk_size": 8,
                    "derived_checkpoint_event_num": 1,
                    "derived_checkpoint_height": 1,
                    "format_version": 2,
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
                    "chunk_size": 8,
                    "derived_checkpoint_event_num": 1,
                    "derived_checkpoint_height": 1,
                    "format_version": 2,
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
                "tracing": {
                    "demangling_enabled": true,
                    "nock_tracing": false,
                    "tracy_compiled": true,
                    "tracy_mode": "off",
                },
            })
        );
    }

    #[tokio::test]
    async fn native_trusted_run_persists_requested_trace_artifacts() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let output_root = tempdir.path().join("out");
        let backend = FakeNativeBackend::successful();

        execute_native_trusted_run_with_backend(
            backend,
            requested,
            InvocationTracingConfig {
                nock_tracing: true,
                nock_tracing_keyword_filter: Some("foo".to_string()),
                nock_tracing_interval_filter: Some(8),
                tracy: crate::speed_of_light::TracyMode::Nockcode,
            },
            &output_root,
            false,
        )
        .await
        .expect("native trusted run result");

        for run_id in ["warmup-0", "run-0", "run-1", "run-2"] {
            let run_dir = output_root.join("runs").join(run_id);
            assert!(
                run_dir.join("trace_artifacts.json").exists(),
                "missing trace_artifacts.json for {run_id}"
            );
            assert!(
                run_dir.join("nock_trace.ndjson").exists(),
                "missing nock_trace.ndjson for {run_id}"
            );
            assert!(
                run_dir.join("nock_trace_meta.json").exists(),
                "missing nock_trace_meta.json for {run_id}"
            );
            assert!(
                run_dir.join("tracy_capture.tracy").exists(),
                "missing tracy_capture.tracy for {run_id}"
            );
        }
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
            tracing: &'a InvocationTracingConfig,
            run_id: &'a str,
            run_dir: &'a Path,
        ) -> futures::future::BoxFuture<
            'a,
            Result<CompletedRun, crate::speed_of_light::harness::HarnessError>,
        > {
            self.events.lock().expect("events").push(run_id.to_string());
            let run_dir = run_dir.to_path_buf();
            let tracing = tracing.clone();
            async move {
                let completed = completed_run(run_id);
                write_requested_trace_artifacts(&run_dir, &tracing).expect("trace artifacts");
                write_run_artifacts(&run_dir, &completed).expect("run artifacts");
                Ok(completed)
            }
            .boxed()
        }

        fn prepare<'a>(
            &'a mut self,
            _resolved: &'a ResolvedCase,
            _tracing: &'a InvocationTracingConfig,
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

    fn write_requested_trace_artifacts(
        run_dir: &Path,
        tracing: &InvocationTracingConfig,
    ) -> Result<(), crate::speed_of_light::harness::HarnessError> {
        if let Some(paths) = tracing.nock_trace_paths_for_run(run_dir) {
            std::fs::create_dir_all(run_dir)?;
            std::fs::write(paths.ndjson_path, "{\"path\":\"/fake\"}\n")?;
            std::fs::write(paths.metadata_path, "{\"format\":\"nock-trace-v1\"}\n")?;
        }

        if tracing.tracy != crate::speed_of_light::TracyMode::Off {
            std::fs::create_dir_all(run_dir)?;
            std::fs::write(run_dir.join("tracy_capture.tracy"), b"fake tracy capture")?;
        }

        Ok(())
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
                format_version: 2,
                source_archive_path: "archive.solarch".to_string(),
                source_archive_event_num: 1,
                derived_checkpoint_height: SolHeight(1),
                derived_checkpoint_event_num: 1,
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
