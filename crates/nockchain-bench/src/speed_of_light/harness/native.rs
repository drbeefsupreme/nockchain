use std::path::Path;
use std::time::Duration;

use tokio::time::sleep;

use super::artifacts::{
    write_host_env, write_provenance, write_requested_case, write_resolved_case,
    write_schema_version, write_summary, write_verdict,
};
use super::case::{resolve_requested_case, RequestedCase, ResolvedCase};
use super::execute::execute_once;
use super::provenance::{capture_host_env, capture_native_provenance, Provenance};
use super::summary::{evaluate_verdict, summarize_runs, RunFailure, RunSummary, RunSummaryInput, Verdict};
use super::{HarnessError, is_release_build};

pub struct NativeRunResult {
    pub resolved: ResolvedCase,
    pub provenance: Provenance,
    pub summary: RunSummary,
    pub verdict: Verdict,
}

pub async fn execute_native_trusted_run(
    requested: RequestedCase,
    output_root: &Path,
    allow_debug_benchmark: bool,
) -> Result<NativeRunResult, HarnessError> {
    prepare_output_root(output_root)?;
    let resolved = resolve_requested_case(&requested)?;
    let provenance = capture_native_provenance(&resolved);

    write_schema_version(output_root)?;
    write_requested_case(output_root, &requested)?;
    write_resolved_case(output_root, &resolved)?;
    write_provenance(output_root, &provenance)?;
    write_host_env(output_root, &capture_host_env())?;

    let release_build = is_release_build();
    if !release_build && !allow_debug_benchmark {
        let summary = summarize_runs(&[], &[], requested.measured_runs);
        let verdict = evaluate_verdict(&RunSummaryInput {
            measured_run_count: requested.measured_runs,
            run_failures: Vec::new(),
            throughput_cv: None,
            release_build,
            allow_debug_benchmark,
        });
        write_summary(output_root, &summary)?;
        write_verdict(output_root, &verdict)?;
        return Err(HarnessError::InvalidRequestedCase(
            "trusted runs require a release build unless --allow-debug-benchmark is set"
                .to_string(),
        ));
    }

    let runs_root = output_root.join("runs");
    std::fs::create_dir_all(&runs_root)?;

    for index in 0..requested.warmup_runs {
        let run_id = format!("warmup-{index}");
        let run_dir = runs_root.join(&run_id);
        let _ = execute_once(&resolved, &run_id, &run_dir).await?;
    }

    let mut run_failures = Vec::new();
    let mut run_metrics = Vec::new();
    for index in 0..requested.measured_runs {
        let run_id = format!("run-{index}");
        let run_dir = runs_root.join(&run_id);
        let completed = execute_once(&resolved, &run_id, &run_dir).await?;
        if completed.record.success {
            run_metrics.push(completed.record.clone().into_metrics());
        } else {
            run_failures.push(RunFailure {
                run_id,
                reason: completed
                    .record
                    .error
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
        throughput_cv: summary
            .throughput_blocks_per_second
            .as_ref()
            .map(|throughput| throughput.cv),
        release_build,
        allow_debug_benchmark,
    });

    write_summary(output_root, &summary)?;
    write_verdict(output_root, &verdict)?;

    Ok(NativeRunResult {
        resolved,
        provenance,
        summary,
        verdict,
    })
}

fn prepare_output_root(output_root: &Path) -> Result<(), HarnessError> {
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

trait IntoMetrics {
    fn into_metrics(self) -> Option<super::summary::RunMetrics>;
}

impl IntoMetrics for super::execute::RunRecord {
    fn into_metrics(self) -> Option<super::summary::RunMetrics> {
        if !self.success {
            return None;
        }
        Some(super::summary::RunMetrics {
            throughput_blocks_per_second: self.throughput_blocks_per_second,
            init_time_secs: self.init_time_secs,
            total_replay_time_secs: self.total_replay_time_secs,
            average_block_time_ms: self.average_block_time_ms,
            failed_pokes: self.failed_pokes as f64,
            checkpoint_count: self.checkpoint_count as f64,
            average_checkpoint_time_secs: self.average_checkpoint_time_secs,
            peak_process_rss_bytes: self.peak_process_rss_bytes,
            minor_faults_total: self.minor_faults_total,
            major_faults_total: self.major_faults_total,
        })
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::prepare_output_root;

    #[test]
    fn native_run_rejects_non_empty_output_root() {
        let tempdir = tempdir().expect("tempdir");
        std::fs::write(tempdir.path().join("stale.txt"), "stale").expect("stale file");

        let error = prepare_output_root(tempdir.path()).expect_err("should reject stale output");
        assert!(error.to_string().contains("already exists and is not empty"));
    }

    #[test]
    fn native_run_allows_empty_output_root() {
        let tempdir = tempdir().expect("tempdir");
        prepare_output_root(tempdir.path()).expect("empty dir should be allowed");
    }
}
