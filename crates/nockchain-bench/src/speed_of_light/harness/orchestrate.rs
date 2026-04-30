use std::path::Path;
use std::time::Duration;

use sha2::{Digest, Sha256};
use tokio::time::sleep;

use super::artifacts::{
    write_host_env, write_json, write_provenance, write_requested_case, write_resolved_case,
    write_schema_version, write_summary, write_verdict,
};
use super::case::{ExecutionRequest, RequestedCase, RequestedOrchestrate};
use super::execute::CompletedRun;
use super::provenance::{build_provenance, capture_host_env, BackendRuntimeFacts, Provenance};
use super::summary::{
    evaluate_verdict, summarize_runs, RunFailure, RunMetrics, RunSummary, RunSummaryInput, Verdict,
};
use super::validate::BackendValidationOutcome;
use super::{
    is_release_build, resolve_requested_case, HarnessError, ResolvedCase,
    DEFAULT_THROUGHPUT_CV_THRESHOLD,
};
use crate::speed_of_light::kernel_utils::{
    init_checkpoint_backed_nockapp, peek_heaviest_chain_or_block,
};
use crate::speed_of_light::{
    build_generated_read_plan, build_generated_replay_plan, load_plan_input, normalize_plan,
    GeneratedReadOptions, GeneratedReplayOptions, PeekRangeRequest, TrustedStep,
};

#[derive(Debug)]
pub struct TrustedRunResult {
    pub resolved: ResolvedCase,
    pub provenance: Provenance,
    pub summary: RunSummary,
    pub verdict: Verdict,
}

pub trait TrustedBackend {
    fn prepare<'a>(
        &'a mut self,
        resolved: &'a mut ResolvedCase,
        output_root: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<(), HarnessError>>;

    fn capture_runtime_facts(&self) -> Result<BackendRuntimeFacts, HarnessError>;

    fn validation_outcome(&self) -> BackendValidationOutcome {
        BackendValidationOutcome::default()
    }

    fn execute_run<'a>(
        &'a mut self,
        resolved: &'a ResolvedCase,
        run_id: &'a str,
        run_dir: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<CompletedRun, HarnessError>>;

    fn capture_raw_evidence<'a>(
        &'a self,
        raw_dir: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<(), HarnessError>>;

    fn cleanup<'a>(&'a mut self) -> futures::future::BoxFuture<'a, Result<(), HarnessError>>;
}

pub async fn execute_trusted_run<B: TrustedBackend>(
    mut backend: B,
    requested: RequestedCase,
    output_root: &Path,
    allow_debug_benchmark: bool,
) -> Result<TrustedRunResult, HarnessError> {
    prepare_output_root(output_root)?;
    std::fs::create_dir_all(output_root)?;
    let mut resolved = resolve_requested_case(&requested)?;
    resolve_trusted_plan_artifact(&requested, &mut resolved, output_root).await?;
    let runs_root = output_root.join("runs");
    let raw_dir = output_root.join("raw");
    std::fs::create_dir_all(&runs_root)?;
    std::fs::create_dir_all(&raw_dir)?;
    if let Err(error) = backend.prepare(&mut resolved, output_root).await {
        return fail_after_prepare(&mut backend, &raw_dir, error).await;
    }
    let runtime_facts_result = backend.capture_runtime_facts();
    let runtime_facts = fail_with_cleanup(&mut backend, runtime_facts_result).await?;
    let validation_outcome = backend.validation_outcome();
    let provenance = build_provenance(
        &resolved,
        runtime_facts,
        validation_outcome.pma_replay_proven(),
    );

    fail_with_cleanup(
        &mut backend,
        write_trusted_run_scaffold(output_root, &requested, &resolved, &provenance),
    )
    .await?;
    let raw_evidence_result = backend.capture_raw_evidence(&raw_dir).await;
    fail_with_cleanup(&mut backend, raw_evidence_result).await?;

    let release_build = is_release_build();
    let (invalid_reasons, partial_reasons) = trusted_policy_reasons(
        &resolved, &provenance, &validation_outcome, allow_debug_benchmark,
    );
    if !invalid_reasons.is_empty() {
        let summary = summarize_runs(&[], &[], requested.measured_runs);
        let verdict = evaluate_verdict(&RunSummaryInput {
            measured_run_count: requested.measured_runs,
            run_failures: Vec::new(),
            throughput_cv: None,
            cv_threshold: requested
                .cv_threshold
                .unwrap_or(DEFAULT_THROUGHPUT_CV_THRESHOLD),
            release_build,
            allow_debug_benchmark,
            invalid_reasons,
            partial_reasons,
        });
        fail_with_cleanup(&mut backend, write_summary(output_root, &summary)).await?;
        fail_with_cleanup(&mut backend, write_verdict(output_root, &verdict)).await?;
        backend.cleanup().await?;
        return Ok(TrustedRunResult {
            resolved,
            provenance,
            summary,
            verdict,
        });
    }

    for index in 0..requested.warmup_runs {
        let run_id = format!("warmup-{index}");
        let run_dir = runs_root.join(&run_id);
        let warmup_result = backend.execute_run(&resolved, &run_id, &run_dir).await;
        fail_with_cleanup(&mut backend, warmup_result).await?;
    }

    let mut run_failures = Vec::new();
    let mut run_metrics = Vec::new();
    for index in 0..requested.measured_runs {
        let run_id = format!("run-{index}");
        let run_dir = runs_root.join(&run_id);
        let run_result = backend.execute_run(&resolved, &run_id, &run_dir).await;
        let completed = fail_with_cleanup(&mut backend, run_result).await?;
        if completed.record.success {
            run_metrics.push(completed_run_into_metrics(&completed));
        } else {
            run_failures.push(RunFailure {
                run_id,
                reason: completed
                    .record
                    .error
                    .clone()
                    .unwrap_or_else(|| "run failed".to_string()),
            });
        }

        if index + 1 < requested.measured_runs && requested.cooldown_secs > 0 {
            sleep(Duration::from_secs(requested.cooldown_secs)).await;
        }
    }

    let run_metrics: Vec<_> = run_metrics.into_iter().flatten().collect();
    let summary = summarize_runs(&run_metrics, &run_failures, requested.measured_runs);
    let verdict = evaluate_verdict(&RunSummaryInput {
        measured_run_count: requested.measured_runs,
        run_failures: run_failures.clone(),
        throughput_cv: primary_cv(&summary),
        cv_threshold: requested
            .cv_threshold
            .unwrap_or(DEFAULT_THROUGHPUT_CV_THRESHOLD),
        release_build,
        allow_debug_benchmark,
        invalid_reasons: Vec::new(),
        partial_reasons,
    });

    fail_with_cleanup(&mut backend, write_summary(output_root, &summary)).await?;
    fail_with_cleanup(&mut backend, write_verdict(output_root, &verdict)).await?;

    backend.cleanup().await?;

    Ok(TrustedRunResult {
        resolved,
        provenance,
        summary,
        verdict,
    })
}

fn primary_cv(summary: &RunSummary) -> Option<f64> {
    summary
        .pokes_per_second
        .as_ref()
        .or(summary.peeks_per_second.as_ref())
        .or(summary.cold_peeks_per_second.as_ref())
        .or(summary.steps_per_second.as_ref())
        .or(summary.throughput_blocks_per_second.as_ref())
        .map(|stats| stats.cv)
}

async fn resolve_trusted_plan_artifact(
    requested: &RequestedCase,
    resolved: &mut ResolvedCase,
    output_root: &Path,
) -> Result<(), HarnessError> {
    let trusted_plan = match &requested.orchestrate {
        RequestedOrchestrate::GeneratedReplay {
            fixture_path,
            blocks,
            skip_genesis,
        } => {
            let generated = build_generated_replay_plan(&GeneratedReplayOptions {
                fixture_path: fixture_path.clone(),
                output_root: output_root.to_path_buf(),
                blocks: *blocks,
                skip_genesis: *skip_genesis,
                enable_checkpointing: requested.enable_checkpointing,
                checkpoint_every_blocks: if requested.checkpoint_every_blocks > 0 {
                    Some(requested.checkpoint_every_blocks)
                } else {
                    None
                },
            })
            .map_err(|error| HarnessError::InvalidRequestedCase(error.to_string()))?;
            normalize_plan(generated.plan_input)
                .map_err(|error| HarnessError::InvalidRequestedCase(error.to_string()))?
        }
        RequestedOrchestrate::PlanFile { plan_path } => {
            let source_plan_path = canonicalize_source_path(plan_path)?;
            resolved.orchestrate.source_plan_sha256_hex =
                Some(sha256_hex_for_file(&source_plan_path)?);
            resolved.orchestrate.source_plan_path = Some(source_plan_path);
            let input = load_plan_input(plan_path)
                .map_err(|error| HarnessError::InvalidRequestedCase(error.to_string()))?;
            normalize_plan(input)
                .map_err(|error| HarnessError::InvalidRequestedCase(error.to_string()))?
        }
        RequestedOrchestrate::GeneratedRead {
            checkpoint_path,
            kernel_path,
            start_height,
            end_height,
            count,
            peek_mode,
        } => {
            let work_dir = output_root.join("input/read-tip-work");
            std::fs::create_dir_all(&work_dir)?;
            let mut nockapp = init_checkpoint_backed_nockapp(
                checkpoint_path,
                kernel_path,
                &work_dir,
                requested.fsync_enabled(),
            )
            .await
            .map_err(|error| HarnessError::InvalidRequestedCase(error.to_string()))?;
            let tip = peek_heaviest_chain_or_block(&mut nockapp)
                .await
                .map_err(|error| HarnessError::InvalidRequestedCase(error.to_string()))?
                .ok_or_else(|| {
                    HarnessError::InvalidRequestedCase(
                        "heaviest chain tip is unavailable after boot".to_string(),
                    )
                })?;
            let generated = build_generated_read_plan(&GeneratedReadOptions {
                checkpoint_path: checkpoint_path.clone(),
                kernel_path: kernel_path.clone(),
                start_height: *start_height,
                range: PeekRangeRequest::from_bounds(*end_height, *count)
                    .map_err(|error| HarnessError::InvalidRequestedCase(error.to_string()))?,
                peek_mode: *peek_mode,
                tip_height: tip.0 .0 .0,
            })
            .map_err(|error| HarnessError::InvalidRequestedCase(error.to_string()))?;
            resolved.orchestrate.read_range_resolution =
                Some(generated.read_range_resolution.clone());
            normalize_plan(generated.plan_input)
                .map_err(|error| HarnessError::InvalidRequestedCase(error.to_string()))?
        }
    };

    write_json(output_root.join("trusted_plan.json"), &trusted_plan)?;
    resolved.orchestrate.normalized_plan_sha256_hex =
        Some(trusted_plan.normalized_plan_sha256_hex.clone());
    resolved.orchestrate.inputs = trusted_plan.inputs.clone();
    resolved.orchestrate.step_count = trusted_plan.steps.len();
    resolved.orchestrate.step_signature_sha256_hex =
        Some(trusted_plan.step_signature_sha256_hex.clone());
    resolved.orchestrate.contains_cold_steps = trusted_plan.steps.iter().any(|step| {
        matches!(
            step,
            TrustedStep::ForceCold { .. } | TrustedStep::PeekHeightCold { .. }
        )
    });
    Ok(())
}

fn canonicalize_source_path(path: &Path) -> Result<std::path::PathBuf, HarnessError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    Ok(absolute.canonicalize()?)
}

fn sha256_hex_for_file(path: &Path) -> Result<String, HarnessError> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

async fn fail_after_prepare<B: TrustedBackend>(
    backend: &mut B,
    raw_dir: &Path,
    error: HarnessError,
) -> Result<TrustedRunResult, HarnessError> {
    let _ = backend.capture_raw_evidence(raw_dir).await;
    let _ = backend.cleanup().await;
    Err(error)
}

async fn fail_with_cleanup<B: TrustedBackend, T>(
    backend: &mut B,
    result: Result<T, HarnessError>,
) -> Result<T, HarnessError> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            let _ = backend.cleanup().await;
            Err(error)
        }
    }
}

fn write_trusted_run_scaffold(
    output_root: &Path,
    requested: &RequestedCase,
    resolved: &ResolvedCase,
    provenance: &Provenance,
) -> Result<(), HarnessError> {
    write_schema_version(output_root)?;
    write_requested_case(output_root, requested)?;
    write_resolved_case(output_root, resolved)?;
    write_provenance(output_root, provenance)?;
    write_host_env(output_root, &capture_host_env())?;
    Ok(())
}

fn trusted_policy_reasons(
    resolved: &ResolvedCase,
    provenance: &Provenance,
    _validation_outcome: &BackendValidationOutcome,
    allow_debug_benchmark: bool,
) -> (Vec<String>, Vec<String>) {
    let mut invalid_reasons = Vec::new();
    let mut partial_reasons = Vec::new();

    if let BackendRuntimeFacts::Docker {
        host_binary,
        container_binary,
        ..
    } = &provenance.backend
    {
        #[cfg(feature = "pma-runtime-compat")]
        if !_validation_outcome.pma_replay_proven() {
            invalid_reasons.push("trusted Docker PMA run did not prove PMA replay".to_string());
        }

        if !is_trusted_release_profile(&container_binary.build_profile) {
            let reason = format!(
                "trusted Docker runs require a release build unless --allow-debug-benchmark is set (container build profile: {})",
                container_binary.build_profile
            );
            if allow_debug_benchmark {
                partial_reasons.push(reason);
            } else {
                invalid_reasons.push(reason);
            }
        }

        if let Some(reason) = version_skew_reason(host_binary, container_binary) {
            let allow_version_skew = matches!(
                &resolved.requested.execution,
                ExecutionRequest::Docker {
                    allow_version_skew: true,
                    ..
                }
            );
            if allow_version_skew {
                partial_reasons.push(format!("{reason} under --allow-version-skew override"));
            } else {
                invalid_reasons.push(reason);
            }
        }
    }

    if resolved.requested.allow_degraded_cold {
        partial_reasons.push(
            "cold verification degradation override was enabled with --allow-degraded-cold"
                .to_string(),
        );
    }

    (invalid_reasons, partial_reasons)
}

fn is_trusted_release_profile(build_profile: &str) -> bool {
    matches!(build_profile, "release" | "bytehound")
}

fn version_skew_reason(
    host_binary: &crate::speed_of_light::harness::BinaryIdentity,
    container_binary: &crate::speed_of_light::harness::BinaryIdentity,
) -> Option<String> {
    if host_binary.version != container_binary.version {
        return Some(format!(
            "host/container version skew detected: host={} container={}",
            host_binary.version, container_binary.version
        ));
    }

    if host_binary.git_commit != container_binary.git_commit {
        return Some(format!(
            "host/container git commit skew detected: host={:?} container={:?}",
            host_binary.git_commit, container_binary.git_commit
        ));
    }

    None
}

pub(crate) fn prepare_output_root(output_root: &Path) -> Result<(), HarnessError> {
    if !output_root.exists() {
        return Ok(());
    }

    let mut entries = std::fs::read_dir(output_root)?;
    if entries.next().is_some() {
        return Err(HarnessError::InvalidRequestedCase(format!(
            "output directory {} already exists and is not empty",
            output_root.display()
        )));
    }

    Ok(())
}

fn completed_run_into_metrics(completed: &CompletedRun) -> Option<RunMetrics> {
    if let Some(record) = &completed.trusted_orchestrate_record {
        return trusted_run_record_into_metrics(record);
    }
    run_record_into_metrics(&completed.record)
}

fn trusted_run_record_into_metrics(
    record: &crate::speed_of_light::orchestrate_execute::RunRecord,
) -> Option<RunMetrics> {
    if !record.success {
        return None;
    }

    Some(RunMetrics {
        throughput_blocks_per_second: record.throughput.pokes_per_second.unwrap_or(0.0),
        steps_per_second: record.throughput.steps_per_second,
        pokes_per_second: record.throughput.pokes_per_second,
        peeks_per_second: record.throughput.peeks_per_second,
        cold_peeks_per_second: record.throughput.cold_peeks_per_second,
        init_time_secs: 0.0,
        total_replay_time_secs: record.timing.total_step_time_secs,
        average_block_time_ms: if record.counts.poke_archive_block > 0 {
            record.timing.total_poke_time_secs * 1000.0 / record.counts.poke_archive_block as f64
        } else {
            0.0
        },
        failed_pokes: record.counts.errors as f64,
        checkpoint_count: 0.0,
        average_checkpoint_time_secs: 0.0,
        peak_process_rss_bytes: None,
        minor_faults_total: None,
        major_faults_total: None,
    })
}

fn run_record_into_metrics(record: &super::execute::RunRecord) -> Option<RunMetrics> {
    if !record.success {
        return None;
    }

    Some(RunMetrics {
        throughput_blocks_per_second: record.throughput_blocks_per_second,
        steps_per_second: None,
        pokes_per_second: Some(record.throughput_blocks_per_second),
        peeks_per_second: None,
        cold_peeks_per_second: None,
        init_time_secs: record.init_time_secs,
        total_replay_time_secs: record.total_replay_time_secs,
        average_block_time_ms: record.average_block_time_ms,
        failed_pokes: record.failed_pokes as f64,
        checkpoint_count: record.checkpoint_count as f64,
        average_checkpoint_time_secs: record.average_checkpoint_time_secs,
        peak_process_rss_bytes: record.peak_process_rss_bytes,
        minor_faults_total: record.minor_faults_total,
        major_faults_total: record.major_faults_total,
    })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    use futures::FutureExt;
    use tempfile::tempdir;

    use super::{execute_trusted_run, is_trusted_release_profile, TrustedBackend};
    use crate::speed_of_light::harness::artifacts::write_run_artifacts;
    use crate::speed_of_light::harness::docker_image::DockerImageSource;
    use crate::speed_of_light::harness::execute::{BlockTimingRecord, CompletedRun, RunRecord};
    use crate::speed_of_light::harness::provenance::BackendRuntimeFacts;
    use crate::speed_of_light::harness::validate::BackendValidationOutcome;
    use crate::speed_of_light::harness::{RequestedCase, RequestedOrchestrate};

    #[tokio::test]
    async fn orchestrator_captures_runtime_facts_before_measured_runs() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let backend = FakeBackend::successful();
        let events = backend.shared_events();

        let result =
            execute_trusted_run(backend, requested, &tempdir.path().join("out"), false).await;

        assert!(result.is_ok(), "orchestrator should succeed: {result:?}");
        assert_eq!(
            events.lock().expect("events").clone(),
            vec![
                "prepare", "setup", "raw-evidence", "warmup-0", "run-0", "run-1", "run-2",
                "cleanup",
            ]
        );
    }

    #[tokio::test]
    async fn orchestrator_marks_failed_measured_runs_partial() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let backend = FakeBackend::with_failure("run-1", "synthetic failure");

        let result = execute_trusted_run(backend, requested, &tempdir.path().join("out"), false)
            .await
            .expect("orchestrator result");

        assert_eq!(result.summary.measured_runs_succeeded, 2);
        match result.verdict.validity {
            crate::speed_of_light::harness::Validity::Partial { reasons } => {
                assert!(reasons.iter().any(|reason| reason.contains("run-1")));
            }
            other => panic!("expected partial verdict, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn orchestrator_writes_expected_artifact_tree() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let output_root = tempdir.path().join("out");
        let backend = FakeBackend::successful();

        execute_trusted_run(backend, requested, &output_root, false)
            .await
            .expect("orchestrator result");

        assert!(output_root.join("schema_version.txt").exists());
        assert!(output_root.join("requested_case.json").exists());
        assert!(output_root.join("resolved_case.json").exists());
        assert!(output_root.join("provenance.json").exists());
        assert!(output_root.join("raw/host_env.json").exists());
        assert!(output_root.join("raw/backend.txt").exists());
        assert!(output_root.join("runs/warmup-0/result.json").exists());
        assert!(output_root.join("runs/run-0/result.json").exists());
        assert!(output_root.join("runs/run-1/result.json").exists());
        assert!(output_root.join("runs/run-2/result.json").exists());
        assert!(output_root.join("summary.json").exists());
        assert!(output_root.join("verdict.json").exists());
    }

    #[tokio::test]
    async fn orchestrator_cleans_up_when_runtime_facts_fail() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_requested_case(tempdir.path());
        let mut backend = FakeBackend::successful();
        backend.fail_runtime_facts = true;
        let events = backend.shared_events();

        let error = execute_trusted_run(backend, requested, &tempdir.path().join("out"), false)
            .await
            .expect_err("runtime facts should fail");

        assert!(error.to_string().contains("runtime facts"));
        assert_eq!(
            events.lock().expect("events").clone(),
            vec!["prepare", "setup", "cleanup"]
        );
    }

    #[tokio::test]
    async fn orchestrator_preserves_invalid_artifacts_for_docker_version_skew() {
        let tempdir = tempdir().expect("tempdir");
        let output_root = tempdir.path().join("out");
        let requested = write_docker_requested_case(tempdir.path(), false);
        let mut backend = FakeBackend::successful();
        backend.runtime_facts = docker_runtime_facts("0.1.1", "release", "container");

        let result = execute_trusted_run(backend, requested, &output_root, false)
            .await
            .expect("invalid run should still produce artifacts");

        assert!(output_root.join("requested_case.json").exists());
        assert!(output_root.join("resolved_case.json").exists());
        assert!(output_root.join("provenance.json").exists());
        assert!(output_root.join("summary.json").exists());
        assert!(output_root.join("verdict.json").exists());
        match result.verdict.validity {
            crate::speed_of_light::harness::Validity::Invalid { reasons } => {
                assert!(reasons.iter().any(|reason| reason.contains("version skew")));
            }
            other => panic!("expected invalid verdict, got {other:?}"),
        }
        assert_eq!(result.summary.measured_runs_succeeded, 0);
    }

    #[tokio::test]
    async fn orchestrator_rejects_debug_container_build_without_override() {
        let tempdir = tempdir().expect("tempdir");
        let requested = write_docker_requested_case(tempdir.path(), false);
        let mut backend = FakeBackend::successful();
        backend.runtime_facts = docker_runtime_facts("0.1.0", "debug", "host");

        let result = execute_trusted_run(backend, requested, &tempdir.path().join("out"), false)
            .await
            .expect("debug container should produce invalid verdict");

        match result.verdict.validity {
            crate::speed_of_light::harness::Validity::Invalid { reasons } => {
                assert!(reasons
                    .iter()
                    .any(|reason| reason.contains("release build")));
            }
            other => panic!("expected invalid verdict, got {other:?}"),
        }
        assert_eq!(result.summary.measured_runs_succeeded, 0);
    }

    #[cfg(not(feature = "pma-runtime-compat"))]
    #[tokio::test]
    async fn docker_trusted_run_keeps_pma_fields_omitted_without_feature_even_when_proven() {
        let tempdir = tempdir().expect("tempdir");
        let output_root = tempdir.path().join("out");
        let requested = write_docker_requested_case(tempdir.path(), false);
        let mut backend = FakeBackend::successful();
        backend.runtime_facts = docker_runtime_facts("0.1.0", "release", "host");
        backend.validation_outcome = BackendValidationOutcome::new(true);

        let result = execute_trusted_run(backend, requested, &output_root, false)
            .await
            .expect("docker run result");

        assert_eq!(result.provenance.runtime_flavor, None);
        assert_eq!(result.provenance.boot_source, None);
        assert_eq!(result.provenance.boot_event_num, None);
        assert_eq!(result.provenance.pma_work_dir_mode, None);
    }

    #[cfg(feature = "pma-runtime-compat")]
    #[tokio::test]
    async fn docker_trusted_run_writes_additive_pma_provenance_under_feature() {
        let tempdir = tempdir().expect("tempdir");
        let output_root = tempdir.path().join("out");
        let requested = write_docker_requested_case(tempdir.path(), false);
        let mut backend = FakeBackend::successful();
        backend.runtime_facts = docker_runtime_facts("0.1.0", "release", "host");
        backend.validation_outcome = BackendValidationOutcome::new(true);

        let result = execute_trusted_run(backend, requested, &output_root, false)
            .await
            .expect("pma docker run result");

        assert_eq!(result.provenance.runtime_flavor.as_deref(), Some("pma"));
        assert_eq!(result.provenance.boot_source.as_deref(), Some("checkpoint"));
        assert_eq!(result.provenance.boot_event_num, Some(1));
        assert_eq!(
            result.provenance.pma_work_dir_mode.as_deref(),
            Some("docker_tmpfs")
        );

        let provenance = serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(output_root.join("provenance.json")).expect("provenance"),
        )
        .expect("provenance json");
        assert_eq!(
            provenance.get("runtime_flavor"),
            Some(&serde_json::json!("pma"))
        );
        assert_eq!(
            provenance.get("boot_source"),
            Some(&serde_json::json!("checkpoint"))
        );
        assert_eq!(
            provenance.get("boot_event_num"),
            Some(&serde_json::json!(1))
        );
        assert_eq!(
            provenance.get("pma_work_dir_mode"),
            Some(&serde_json::json!("docker_tmpfs"))
        );
    }

    #[cfg(feature = "pma-runtime-compat")]
    #[tokio::test]
    async fn docker_trusted_run_preserves_version_skew_policy_under_pma_feature() {
        let tempdir = tempdir().expect("tempdir");

        let mut skewed_backend = FakeBackend::successful();
        skewed_backend.runtime_facts = docker_runtime_facts("0.1.1", "release", "container");
        skewed_backend.validation_outcome = BackendValidationOutcome::new(true);

        let invalid = execute_trusted_run(
            skewed_backend,
            write_docker_requested_case(tempdir.path(), false),
            &tempdir.path().join("invalid-out"),
            false,
        )
        .await
        .expect("invalid skew result");

        match invalid.verdict.validity {
            crate::speed_of_light::harness::Validity::Invalid { reasons } => {
                assert!(reasons.iter().any(|reason| reason.contains("version skew")));
            }
            other => panic!("expected invalid verdict, got {other:?}"),
        }

        let mut allowed_backend = FakeBackend::successful();
        allowed_backend.runtime_facts = docker_runtime_facts("0.1.1", "release", "container");
        allowed_backend.validation_outcome = BackendValidationOutcome::new(true);

        let partial = execute_trusted_run(
            allowed_backend,
            write_docker_requested_case(tempdir.path(), true),
            &tempdir.path().join("partial-out"),
            false,
        )
        .await
        .expect("partial skew result");

        match partial.verdict.validity {
            crate::speed_of_light::harness::Validity::Partial { reasons } => {
                assert!(reasons
                    .iter()
                    .any(|reason| reason.contains("--allow-version-skew")));
            }
            other => panic!("expected partial verdict, got {other:?}"),
        }

        let mut unproven_backend = FakeBackend::successful();
        unproven_backend.runtime_facts = docker_runtime_facts("0.1.0", "release", "host");
        unproven_backend.validation_outcome = BackendValidationOutcome::default();

        let unproven = execute_trusted_run(
            unproven_backend,
            write_docker_requested_case(tempdir.path(), false),
            &tempdir.path().join("unproven-out"),
            false,
        )
        .await
        .expect("unproven result");

        match unproven.verdict.validity {
            crate::speed_of_light::harness::Validity::Invalid { reasons } => {
                assert!(reasons
                    .iter()
                    .any(|reason| reason.contains("did not prove PMA replay")));
            }
            other => panic!("expected invalid verdict, got {other:?}"),
        }
    }

    #[test]
    fn trusted_release_profiles_include_bytehound() {
        assert!(is_trusted_release_profile("release"));
        assert!(is_trusted_release_profile("bytehound"));
        assert!(!is_trusted_release_profile("debug"));
    }

    struct FakeBackend {
        events: Arc<Mutex<Vec<String>>>,
        failed_run_id: Option<String>,
        failure_message: Option<String>,
        fail_runtime_facts: bool,
        runtime_facts: BackendRuntimeFacts,
        validation_outcome: BackendValidationOutcome,
    }

    impl FakeBackend {
        fn successful() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
                failed_run_id: None,
                failure_message: None,
                fail_runtime_facts: false,
                runtime_facts: BackendRuntimeFacts::Native,
                validation_outcome: BackendValidationOutcome::default(),
            }
        }

        fn with_failure(run_id: &str, message: &str) -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
                failed_run_id: Some(run_id.to_string()),
                failure_message: Some(message.to_string()),
                fail_runtime_facts: false,
                runtime_facts: BackendRuntimeFacts::Native,
                validation_outcome: BackendValidationOutcome::default(),
            }
        }

        fn shared_events(&self) -> Arc<Mutex<Vec<String>>> {
            Arc::clone(&self.events)
        }
    }

    impl TrustedBackend for FakeBackend {
        fn execute_run<'a>(
            &'a mut self,
            _resolved: &'a crate::speed_of_light::harness::ResolvedCase,
            run_id: &'a str,
            run_dir: &'a Path,
        ) -> futures::future::BoxFuture<
            'a,
            Result<CompletedRun, crate::speed_of_light::harness::HarnessError>,
        > {
            self.events.lock().expect("events").push(run_id.to_string());

            let should_fail = self.failed_run_id.as_deref() == Some(run_id);
            let failure_message = self.failure_message.clone();
            let run_dir = run_dir.to_path_buf();

            async move {
                let completed = CompletedRun {
                    record: RunRecord {
                        run_id: run_id.to_string(),
                        success: !should_fail,
                        error: should_fail.then(|| {
                            failure_message.unwrap_or_else(|| "synthetic failure".to_string())
                        }),
                        blocks_poked: (!should_fail) as u64,
                        failed_pokes: should_fail as u64,
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
                    trusted_orchestrate_record: None,
                    block_timings: vec![BlockTimingRecord {
                        height: 2,
                        duration_ms: 10.0,
                    }],
                    profile: None,
                    bench_results: None,
                };
                write_run_artifacts(&run_dir, &completed).expect("run artifacts");
                Ok(completed)
            }
            .boxed()
        }

        fn prepare<'a>(
            &'a mut self,
            _resolved: &'a mut crate::speed_of_light::harness::ResolvedCase,
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
                .push("setup".to_string());
            if self.fail_runtime_facts {
                return Err(
                    crate::speed_of_light::harness::HarnessError::InvalidRequestedCase(
                        "runtime facts failed".to_string(),
                    ),
                );
            }
            Ok(self.runtime_facts.clone())
        }

        fn validation_outcome(&self) -> BackendValidationOutcome {
            self.validation_outcome
        }

        fn capture_raw_evidence<'a>(
            &'a self,
            raw_dir: &'a Path,
        ) -> futures::future::BoxFuture<'a, Result<(), crate::speed_of_light::harness::HarnessError>>
        {
            self.events
                .lock()
                .expect("events")
                .push("raw-evidence".to_string());
            let raw_dir = raw_dir.to_path_buf();
            async move {
                std::fs::create_dir_all(&raw_dir)?;
                std::fs::write(raw_dir.join("backend.txt"), "backend")?;
                Ok(())
            }
            .boxed()
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

    fn docker_runtime_facts(
        container_version: &str,
        container_build_profile: &str,
        container_commit: &str,
    ) -> BackendRuntimeFacts {
        BackendRuntimeFacts::Docker {
            host_binary: crate::speed_of_light::harness::BinaryIdentity {
                version: "0.1.0".to_string(),
                build_profile: "release".to_string(),
                git_commit: Some("host".to_string()),
            },
            container_binary: crate::speed_of_light::harness::BinaryIdentity {
                version: container_version.to_string(),
                build_profile: container_build_profile.to_string(),
                git_commit: Some(container_commit.to_string()),
            },
            image_source: DockerImageSource::AutoBuild {
                tag: "nockchain-bench:test".to_string(),
            },
            requested_image_ref: "nockchain-bench:test".to_string(),
            resolved_image_ref: "sha256:test".to_string(),
            image_digest: "sha256:test".to_string(),
            container_id: "abc".to_string(),
            docker_engine_version: "29.1.3".to_string(),
            docker_context: "default".to_string(),
            cgroup_version: "2".to_string(),
            storage_driver: "overlayfs".to_string(),
            realized_memory_max: 1024,
            realized_memory_current: 512,
            realized_cpuset: Some("0-3".to_string()),
            realized_cpu_max: Some("max 100000".to_string()),
        }
    }

    fn write_requested_case(root: &Path) -> RequestedCase {
        let mut requested = RequestedCase::native(PathBuf::new());
        requested.orchestrate = RequestedOrchestrate::PlanFile {
            plan_path: write_test_plan(root),
        };
        requested.warmup_runs = 1;
        requested.measured_runs = 3;
        requested.cooldown_secs = 0;
        requested
    }

    fn write_docker_requested_case(root: &Path, allow_version_skew: bool) -> RequestedCase {
        let mut requested = write_requested_case(root);
        requested.execution = crate::speed_of_light::harness::ExecutionRequest::Docker {
            image: DockerImageSource::AutoBuild {
                tag: "nockchain-bench:test".to_string(),
            },
            memory_limit: "1g".to_string(),
            cpuset: Some("0-3".to_string()),
            cpu_quota: None,
            cpu_period: None,
            work_dir_mode: crate::speed_of_light::harness::WorkDirMode::DockerTmpfs,
            allow_version_skew,
        };
        requested
    }

    fn write_test_plan(root: &Path) -> PathBuf {
        let checkpoint_path = root.join("checkpoint.chkjam");
        let kernel_path = root.join("kernel.jam");
        std::fs::write(&checkpoint_path, [1, 2, 3]).expect("checkpoint");
        std::fs::write(&kernel_path, [4, 5, 6]).expect("kernel");
        let plan_path = root.join("trusted-input-plan.json");
        let plan = serde_json::json!({
            "schema_version": crate::speed_of_light::TRUSTED_PLAN_SCHEMA_VERSION,
            "checkpoint": checkpoint_path,
            "kernel": kernel_path,
            "steps": [{ "type": "peek_height", "height": 1 }]
        });
        std::fs::write(
            &plan_path,
            serde_json::to_vec_pretty(&plan).expect("plan json"),
        )
        .expect("plan");
        plan_path
    }
}
