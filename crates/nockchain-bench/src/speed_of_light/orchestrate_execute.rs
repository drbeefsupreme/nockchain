use std::fs::File;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::orchestrate_plan::{TrustedPlan, TrustedStep};
use super::orchestrator::{
    ColdMode, QuickOrchestratePlan, QuickOrchestrateRunner, QuickOrchestrateStep,
};

pub const RUN_RESULT_SCHEMA_VERSION: &str = "run-result/v1";
pub const STEP_RESULT_SCHEMA_VERSION: &str = "step-result/v1";
pub const COLD_EVIDENCE_SCHEMA_VERSION: &str = "cold-evidence/v1";

#[derive(Debug, Error)]
pub enum OrchestrateExecuteError {
    #[error("invalid throughput denominator for {metric}")]
    InvalidThroughputDenominator { metric: &'static str },
    #[error("unverified cold step {step_id} is not allowed without --allow-degraded-cold")]
    UnverifiedColdStrict { step_id: String },
    #[error("unrecognized degraded cold reason {reason:?} for step {step_id}")]
    UnknownDegradedColdReason {
        step_id: String,
        reason: Option<String>,
    },
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("trusted plan references missing input id {0}")]
    MissingInput(String),
    #[error("orchestrate pre-run failure: {0}")]
    PreRun(#[from] super::orchestrator::PreRunError),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunRecord {
    pub schema_version: String,
    pub benchmark: String,
    pub run_id: String,
    pub success: bool,
    pub error: Option<String>,
    pub counts: RunCounts,
    pub timing: RunTiming,
    pub throughput: RunThroughput,
    pub final_tip: Option<FinalTip>,
    pub failed_step_index: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunCounts {
    pub steps_executed: u64,
    pub poke_archive_block: u64,
    pub peek_height: u64,
    pub force_cold: u64,
    pub peek_height_cold: u64,
    pub missing: u64,
    pub errors: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RunTiming {
    pub total_step_time_secs: f64,
    pub total_poke_time_secs: f64,
    pub total_peek_time_secs: f64,
    pub total_cold_force_time_secs: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RunThroughput {
    pub steps_per_second: Option<f64>,
    pub pokes_per_second: Option<f64>,
    pub peeks_per_second: Option<f64>,
    pub cold_peeks_per_second: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FinalTip {
    pub height: u64,
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StepResultRow {
    pub schema_version: String,
    pub run_id: String,
    pub step_index: usize,
    pub step_id: String,
    pub label: String,
    #[serde(rename = "type")]
    pub step_type: String,
    pub outcome: String,
    pub duration_ms: f64,
    pub height: Option<u64>,
    pub input_id: Option<String>,
    pub minflt_delta: Option<u64>,
    pub majflt_delta: Option<u64>,
    pub cold_evidence_id: Option<String>,
    pub trusted_metric_valid: Option<bool>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColdEvidenceRow {
    pub schema_version: String,
    pub evidence_id: String,
    pub run_id: String,
    pub step_index: usize,
    pub step_id: String,
    pub step_type: String,
    pub cold_target: String,
    pub tolerance_pages: Option<u64>,
    pub cold_attempts: u32,
    pub cold_verified: bool,
    pub cold_force_duration_ms: f64,
    pub degraded_reason: Option<String>,
    pub error: Option<String>,
    pub peek_completed: Option<bool>,
    pub peek_outcome: Option<String>,
    pub residency_pages_before: Option<u64>,
    pub residency_pages_after: Option<u64>,
    pub residency_total_pages: Option<u64>,
    pub page_size_bytes: Option<u64>,
    pub reclaim: ColdReclaimEvidence,
    pub vmas: Vec<ColdVmaEvidence>,
    pub operations: ColdOperationsEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ColdReclaimEvidence {
    pub method: Option<String>,
    pub requested_bytes: Option<u64>,
    pub reclaimed_bytes: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ColdVmaEvidence {
    pub start: Option<String>,
    pub end: Option<String>,
    pub permissions: Option<String>,
    pub path: Option<String>,
    pub resident_pages_before: Option<u64>,
    pub resident_pages_after: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ColdOperationsEvidence {
    pub mincore_available: bool,
    pub memory_reclaim_available: bool,
    pub swappiness_writable: bool,
    pub fsync_attempted: bool,
    pub msync_attempted: bool,
}

#[derive(Debug, Clone)]
pub struct SyntheticStepMeasurement {
    pub step: TrustedStep,
    pub outcome: StepOutcomeKind,
    pub duration_ms: f64,
    pub minflt_delta: Option<u64>,
    pub majflt_delta: Option<u64>,
    pub cold_force_duration_ms: Option<f64>,
    pub cold_verified: Option<bool>,
    pub cold_attempts: Option<u32>,
    pub residency_pages_after: Option<u64>,
    pub residency_total_pages: Option<u64>,
    pub degraded_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcomeKind {
    Ok,
    Success,
    Missing,
    Error,
}

impl StepOutcomeKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Success => "success",
            Self::Missing => "missing",
            Self::Error => "error",
        }
    }

    fn is_error(self) -> bool {
        matches!(self, Self::Error)
    }
}

pub fn build_run_record_from_measurements(
    run_id: &str,
    measurements: &[SyntheticStepMeasurement],
    final_tip: Option<FinalTip>,
) -> Result<(RunRecord, Vec<StepResultRow>, Vec<ColdEvidenceRow>), OrchestrateExecuteError> {
    build_run_record_from_measurements_with_policy(run_id, measurements, final_tip, false)
}

pub fn build_run_record_from_measurements_with_policy(
    run_id: &str,
    measurements: &[SyntheticStepMeasurement],
    final_tip: Option<FinalTip>,
    allow_degraded_cold: bool,
) -> Result<(RunRecord, Vec<StepResultRow>, Vec<ColdEvidenceRow>), OrchestrateExecuteError> {
    let mut counts = RunCounts::default();
    let mut timing = RunTiming::default();
    let mut step_rows = Vec::new();
    let mut cold_rows = Vec::new();
    let mut fail_fast_step_index = None;

    for measurement in measurements {
        if fail_fast_step_index.is_some() {
            break;
        }

        let descriptor = StepDescriptor::from(&measurement.step);
        counts.steps_executed += 1;
        timing.total_step_time_secs += measurement.duration_ms / 1000.0;
        match descriptor.step_type {
            "poke_archive_block" => {
                counts.poke_archive_block += 1;
                timing.total_poke_time_secs += measurement.duration_ms / 1000.0;
            }
            "peek_height" => {
                counts.peek_height += 1;
                timing.total_peek_time_secs += measurement.duration_ms / 1000.0;
            }
            "force_cold" => {
                counts.force_cold += 1;
                timing.total_cold_force_time_secs +=
                    measurement.cold_force_duration_ms.unwrap_or(0.0) / 1000.0;
            }
            "peek_height_cold" => {
                counts.peek_height_cold += 1;
                timing.total_cold_force_time_secs +=
                    measurement.cold_force_duration_ms.unwrap_or(0.0) / 1000.0;
            }
            _ => {}
        }
        if matches!(measurement.outcome, StepOutcomeKind::Missing) {
            counts.missing += 1;
        }
        if measurement.outcome.is_error() {
            counts.errors += 1;
            fail_fast_step_index = Some(descriptor.step_index);
        }

        let cold_evidence_id = if matches!(descriptor.step_type, "force_cold" | "peek_height_cold")
        {
            let cold_verified = measurement.cold_verified.unwrap_or(false);
            if !cold_verified {
                validate_degraded_cold_reason(
                    &descriptor.step_id,
                    measurement.degraded_reason.as_deref(),
                    allow_degraded_cold,
                )?;
            }
            let evidence_id = format!("{run_id}-{}", descriptor.step_id);
            cold_rows.push(ColdEvidenceRow {
                schema_version: COLD_EVIDENCE_SCHEMA_VERSION.to_string(),
                evidence_id: evidence_id.clone(),
                run_id: run_id.to_string(),
                step_index: descriptor.step_index,
                step_id: descriptor.step_id.clone(),
                step_type: descriptor.step_type.to_string(),
                cold_target: "pma_replay_nockstack".to_string(),
                tolerance_pages: None,
                cold_attempts: measurement.cold_attempts.unwrap_or(1),
                cold_verified,
                cold_force_duration_ms: measurement
                    .cold_force_duration_ms
                    .unwrap_or(measurement.duration_ms),
                degraded_reason: measurement.degraded_reason.clone(),
                error: None,
                peek_completed: if descriptor.step_type == "peek_height_cold" {
                    Some(!measurement.outcome.is_error())
                } else {
                    None
                },
                peek_outcome: if descriptor.step_type == "peek_height_cold" {
                    Some(measurement.outcome.as_str().to_string())
                } else {
                    None
                },
                residency_pages_before: None,
                residency_pages_after: measurement.residency_pages_after,
                residency_total_pages: measurement.residency_total_pages,
                page_size_bytes: None,
                reclaim: ColdReclaimEvidence::default(),
                vmas: Vec::new(),
                operations: ColdOperationsEvidence::default(),
            });
            Some(evidence_id)
        } else {
            None
        };

        step_rows.push(StepResultRow {
            schema_version: STEP_RESULT_SCHEMA_VERSION.to_string(),
            run_id: run_id.to_string(),
            step_index: descriptor.step_index,
            step_id: descriptor.step_id,
            label: descriptor.label,
            step_type: descriptor.step_type.to_string(),
            outcome: measurement.outcome.as_str().to_string(),
            duration_ms: measurement.duration_ms,
            height: descriptor.height,
            input_id: descriptor.input_id,
            minflt_delta: measurement.minflt_delta,
            majflt_delta: measurement.majflt_delta,
            cold_evidence_id,
            trusted_metric_valid: Some(!measurement.outcome.is_error()),
            error: None,
        });
    }

    let throughput = RunThroughput {
        steps_per_second: throughput(
            "steps_per_second", counts.steps_executed, timing.total_step_time_secs,
        )?,
        pokes_per_second: throughput(
            "pokes_per_second", counts.poke_archive_block, timing.total_poke_time_secs,
        )?,
        peeks_per_second: throughput(
            "peeks_per_second", counts.peek_height, timing.total_peek_time_secs,
        )?,
        cold_peeks_per_second: throughput(
            "cold_peeks_per_second", counts.peek_height_cold, timing.total_cold_force_time_secs,
        )?,
    };

    let success = counts.errors == 0
        || (allow_degraded_cold
            && counts.errors as usize == cold_rows.iter().filter(|row| !row.cold_verified).count()
            && cold_rows
                .iter()
                .filter(|row| !row.cold_verified)
                .all(|row| {
                    row.degraded_reason
                        .as_deref()
                        .is_some_and(is_allowed_degraded_cold_reason)
                }));
    Ok((
        RunRecord {
            schema_version: RUN_RESULT_SCHEMA_VERSION.to_string(),
            benchmark: "sol-orchestrate".to_string(),
            run_id: run_id.to_string(),
            success,
            error: if success {
                None
            } else {
                Some("step failed".to_string())
            },
            counts,
            timing,
            throughput,
            final_tip,
            failed_step_index: fail_fast_step_index,
        },
        step_rows,
        cold_rows,
    ))
}

pub fn write_run_artifacts(
    run_dir: &Path,
    record: &RunRecord,
    steps: &[StepResultRow],
    cold_evidence: &[ColdEvidenceRow],
) -> Result<(), OrchestrateExecuteError> {
    std::fs::create_dir_all(run_dir)?;
    std::fs::write(
        run_dir.join("result.json"),
        serde_json::to_vec_pretty(record)?,
    )?;
    write_ndjson(&run_dir.join("steps.ndjson"), steps)?;
    write_ndjson(&run_dir.join("cold_evidence.ndjson"), cold_evidence)?;
    std::fs::write(run_dir.join("stdout.log"), b"")?;
    std::fs::write(run_dir.join("stderr.log"), b"")?;
    Ok(())
}

pub async fn execute_trusted_plan_once(
    plan: &TrustedPlan,
    run_id: &str,
    run_dir: &Path,
    work_dir: &Path,
    fsync: bool,
    allow_degraded_cold: bool,
) -> Result<RunRecord, OrchestrateExecuteError> {
    let quick_plan = quick_plan_from_trusted(plan)?;
    std::fs::create_dir_all(work_dir)?;
    std::fs::create_dir_all(run_dir)?;
    let plan_path = run_dir.join("trusted-plan-run-input.json");
    std::fs::write(&plan_path, serde_json::to_vec_pretty(&quick_plan)?)?;

    let runner = QuickOrchestrateRunner::new(
        plan_path,
        work_dir.to_path_buf(),
        fsync,
        if allow_degraded_cold {
            ColdMode::Soft
        } else {
            ColdMode::Strict
        },
    );
    let results = runner.run().await?;
    let measurements = measurements_from_quick_results(plan, &results);
    let final_tip = results.final_tip_parts().map(|(height, hash)| FinalTip {
        height,
        hash: hash.to_string(),
    });
    let (record, steps, cold) = build_run_record_from_measurements_with_policy(
        run_id, &measurements, final_tip, allow_degraded_cold,
    )?;
    write_run_artifacts(run_dir, &record, &steps, &cold)?;
    Ok(record)
}

fn quick_plan_from_trusted(
    plan: &TrustedPlan,
) -> Result<QuickOrchestratePlan, OrchestrateExecuteError> {
    let input_path = |input_id: &str| {
        plan.inputs
            .iter()
            .find(|input| input.input_id == input_id)
            .map(|input| {
                input
                    .container_path
                    .clone()
                    .unwrap_or_else(|| input.absolute_path.clone())
            })
            .ok_or_else(|| OrchestrateExecuteError::MissingInput(input_id.to_string()))
    };

    let checkpoint = input_path(&plan.boot.checkpoint_input_id)?;
    let kernel = input_path(&plan.boot.kernel_input_id)?;
    let mut steps = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        steps.push(match step {
            TrustedStep::PokeArchiveBlock {
                archive_input_id,
                height,
                label,
                ..
            } => QuickOrchestrateStep::PokeArchiveBlock {
                archive: input_path(archive_input_id)?,
                height: *height,
                label: Some(label.clone()),
            },
            TrustedStep::PeekHeight { height, label, .. } => QuickOrchestrateStep::PeekHeight {
                height: *height,
                label: Some(label.clone()),
            },
            TrustedStep::ForceCold {
                label,
                tolerance_pages,
                max_attempts,
                ..
            } => QuickOrchestrateStep::ForceCold {
                label: Some(label.clone()),
                tolerance_pages: *tolerance_pages,
                max_attempts: *max_attempts,
            },
            TrustedStep::PeekHeightCold {
                height,
                label,
                tolerance_pages,
                max_attempts,
                ..
            } => QuickOrchestrateStep::PeekHeightCold {
                height: *height,
                label: Some(label.clone()),
                tolerance_pages: *tolerance_pages,
                max_attempts: *max_attempts,
            },
        });
    }
    Ok(QuickOrchestratePlan {
        checkpoint,
        kernel,
        steps,
    })
}

fn measurements_from_quick_results(
    plan: &TrustedPlan,
    results: &super::orchestrator::QuickOrchestrateResults,
) -> Vec<SyntheticStepMeasurement> {
    results
        .steps()
        .iter()
        .enumerate()
        .filter_map(|(index, quick_step)| {
            let step = plan.steps.get(index)?.clone();
            Some(SyntheticStepMeasurement {
                step,
                outcome: match quick_step.outcome_str() {
                    "ok" => StepOutcomeKind::Ok,
                    "success" => StepOutcomeKind::Success,
                    "missing" => StepOutcomeKind::Missing,
                    _ => StepOutcomeKind::Error,
                },
                duration_ms: quick_step.duration_ms_value(),
                minflt_delta: quick_step.minflt_delta(),
                majflt_delta: quick_step.majflt_delta(),
                cold_force_duration_ms: None,
                cold_verified: quick_step.cold_verified(),
                cold_attempts: quick_step.cold_attempts(),
                residency_pages_after: quick_step.residency_pages_after(),
                residency_total_pages: quick_step.residency_total_pages(),
                degraded_reason: quick_step.degraded_reason().map(str::to_string),
            })
        })
        .collect()
}

fn write_ndjson<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), OrchestrateExecuteError> {
    let mut file = File::create(path)?;
    for row in rows {
        serde_json::to_writer(&mut file, row)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn throughput(
    metric: &'static str,
    numerator: u64,
    denominator_secs: f64,
) -> Result<Option<f64>, OrchestrateExecuteError> {
    if numerator == 0 {
        return Ok(None);
    }
    if denominator_secs <= 0.0 || !denominator_secs.is_finite() {
        return Err(OrchestrateExecuteError::InvalidThroughputDenominator { metric });
    }
    Ok(Some(numerator as f64 / denominator_secs))
}

fn validate_degraded_cold_reason(
    step_id: &str,
    reason: Option<&str>,
    allow_degraded_cold: bool,
) -> Result<(), OrchestrateExecuteError> {
    if !allow_degraded_cold {
        return Err(OrchestrateExecuteError::UnverifiedColdStrict {
            step_id: step_id.to_string(),
        });
    }
    if matches!(
        reason,
        Some(
            "mincore_unavailable"
                | "memory_reclaim_eagain"
                | "partial_pageout"
                | "swappiness_unwritable"
        )
    ) {
        return Ok(());
    }
    Err(OrchestrateExecuteError::UnknownDegradedColdReason {
        step_id: step_id.to_string(),
        reason: reason.map(str::to_string),
    })
}

pub fn is_allowed_degraded_cold_reason(reason: &str) -> bool {
    matches!(
        reason,
        "mincore_unavailable"
            | "memory_reclaim_eagain"
            | "partial_pageout"
            | "swappiness_unwritable"
    )
}

struct StepDescriptor {
    step_index: usize,
    step_id: String,
    label: String,
    step_type: &'static str,
    height: Option<u64>,
    input_id: Option<String>,
}

impl From<&TrustedStep> for StepDescriptor {
    fn from(step: &TrustedStep) -> Self {
        match step {
            TrustedStep::PokeArchiveBlock {
                step_index,
                step_id,
                label,
                archive_input_id,
                height,
            } => Self {
                step_index: *step_index,
                step_id: step_id.clone(),
                label: label.clone(),
                step_type: "poke_archive_block",
                height: Some(*height),
                input_id: Some(archive_input_id.clone()),
            },
            TrustedStep::PeekHeight {
                step_index,
                step_id,
                label,
                height,
            } => Self {
                step_index: *step_index,
                step_id: step_id.clone(),
                label: label.clone(),
                step_type: "peek_height",
                height: Some(*height),
                input_id: None,
            },
            TrustedStep::ForceCold {
                step_index,
                step_id,
                label,
                ..
            } => Self {
                step_index: *step_index,
                step_id: step_id.clone(),
                label: label.clone(),
                step_type: "force_cold",
                height: None,
                input_id: None,
            },
            TrustedStep::PeekHeightCold {
                step_index,
                step_id,
                label,
                height,
                ..
            } => Self {
                step_index: *step_index,
                step_id: step_id.clone(),
                label: label.clone(),
                step_type: "peek_height_cold",
                height: Some(*height),
                input_id: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::json;
    use tempfile::tempdir;

    use super::super::orchestrate_plan::{normalize_plan, OrchestratePlanInput};
    use super::*;

    fn trusted_steps(value: serde_json::Value) -> Vec<TrustedStep> {
        let tempdir = tempdir().expect("tempdir");
        let input: OrchestratePlanInput =
            serde_json::from_value(materialize_paths(value, tempdir.path())).expect("input");
        normalize_plan(input).expect("trusted plan").steps
    }

    fn materialize_paths(mut value: serde_json::Value, root: &Path) -> serde_json::Value {
        match &mut value {
            serde_json::Value::Object(map) => {
                for (key, child) in map.iter_mut() {
                    if matches!(key.as_str(), "checkpoint" | "kernel" | "archive") {
                        if let Some(path) = child.as_str() {
                            let materialized = root.join(path);
                            if let Some(parent) = materialized.parent() {
                                std::fs::create_dir_all(parent).expect("create parent");
                            }
                            std::fs::write(&materialized, path.as_bytes()).expect("write input");
                            *child = serde_json::Value::String(
                                materialized.to_string_lossy().to_string(),
                            );
                        }
                    } else {
                        *child = materialize_paths(child.take(), root);
                    }
                }
            }
            serde_json::Value::Array(items) => {
                for child in items {
                    *child = materialize_paths(child.take(), root);
                }
            }
            _ => {}
        }
        value
    }

    #[test]
    fn orchestrate_execute_computes_normative_throughput_formulas() {
        let steps = trusted_steps(json!({
            "checkpoint": "checkpoint.chkjam",
            "kernel": "kernel.jam",
            "steps": [
                { "type": "poke_archive_block", "archive": "archive.solarch", "height": 1 },
                { "type": "peek_height", "height": 2 },
                { "type": "peek_height_cold", "height": 3 }
            ]
        }));
        let measurements = vec![
            SyntheticStepMeasurement {
                step: steps[0].clone(),
                outcome: StepOutcomeKind::Ok,
                duration_ms: 100.0,
                minflt_delta: Some(1),
                majflt_delta: Some(0),
                cold_force_duration_ms: None,
                cold_verified: None,
                cold_attempts: None,
                residency_pages_after: None,
                residency_total_pages: None,
                degraded_reason: None,
            },
            SyntheticStepMeasurement {
                step: steps[1].clone(),
                outcome: StepOutcomeKind::Success,
                duration_ms: 200.0,
                minflt_delta: None,
                majflt_delta: None,
                cold_force_duration_ms: None,
                cold_verified: None,
                cold_attempts: None,
                residency_pages_after: None,
                residency_total_pages: None,
                degraded_reason: None,
            },
            SyntheticStepMeasurement {
                step: steps[2].clone(),
                outcome: StepOutcomeKind::Success,
                duration_ms: 300.0,
                minflt_delta: None,
                majflt_delta: None,
                cold_force_duration_ms: Some(400.0),
                cold_verified: Some(true),
                cold_attempts: None,
                residency_pages_after: None,
                residency_total_pages: None,
                degraded_reason: None,
            },
        ];

        let (record, rows, cold) =
            build_run_record_from_measurements("run-0", &measurements, None).expect("record");

        assert_eq!(record.counts.steps_executed, 3);
        assert_close(record.throughput.steps_per_second, 5.0);
        assert_eq!(record.throughput.pokes_per_second, Some(10.0));
        assert_close(record.throughput.peeks_per_second, 5.0);
        assert_eq!(record.throughput.cold_peeks_per_second, Some(2.5));
        assert_eq!(rows[1].minflt_delta, None);
        assert_eq!(rows[1].majflt_delta, None);
        assert_eq!(cold[0].cold_force_duration_ms, 400.0);
    }

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("throughput");
        assert!(
            (actual - expected).abs() < 0.000_000_001,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn orchestrate_execute_serializes_null_throughput_for_zero_numerator() {
        let steps = trusted_steps(json!({
            "checkpoint": "checkpoint.chkjam",
            "kernel": "kernel.jam",
            "steps": [{ "type": "peek_height", "height": 2 }]
        }));
        let measurements = vec![SyntheticStepMeasurement {
            step: steps[0].clone(),
            outcome: StepOutcomeKind::Missing,
            duration_ms: 200.0,
            minflt_delta: None,
            majflt_delta: None,
            cold_force_duration_ms: None,
            cold_verified: None,
            cold_attempts: None,
            residency_pages_after: None,
            residency_total_pages: None,
            degraded_reason: None,
        }];

        let (record, _, _) =
            build_run_record_from_measurements("run-0", &measurements, None).expect("record");
        let value = serde_json::to_value(record).expect("json");

        assert_eq!(
            value["throughput"]["pokes_per_second"],
            serde_json::Value::Null
        );
        assert_eq!(value["counts"]["missing"], json!(1));
    }

    #[test]
    fn orchestrate_execute_rejects_nonzero_count_with_zero_denominator() {
        let steps = trusted_steps(json!({
            "checkpoint": "checkpoint.chkjam",
            "kernel": "kernel.jam",
            "steps": [{ "type": "peek_height", "height": 2 }]
        }));
        let measurements = vec![SyntheticStepMeasurement {
            step: steps[0].clone(),
            outcome: StepOutcomeKind::Success,
            duration_ms: 0.0,
            minflt_delta: None,
            majflt_delta: None,
            cold_force_duration_ms: None,
            cold_verified: None,
            cold_attempts: None,
            residency_pages_after: None,
            residency_total_pages: None,
            degraded_reason: None,
        }];

        assert!(matches!(
            build_run_record_from_measurements("run-0", &measurements, None),
            Err(OrchestrateExecuteError::InvalidThroughputDenominator {
                metric: "steps_per_second"
            })
        ));
    }

    #[test]
    fn orchestrate_execute_records_fail_fast_index_and_final_tip() {
        let steps = trusted_steps(json!({
            "checkpoint": "checkpoint.chkjam",
            "kernel": "kernel.jam",
            "steps": [
                { "type": "peek_height", "height": 2 },
                { "type": "peek_height", "height": 3 }
            ]
        }));
        let measurements = vec![
            SyntheticStepMeasurement {
                step: steps[0].clone(),
                outcome: StepOutcomeKind::Error,
                duration_ms: 10.0,
                minflt_delta: None,
                majflt_delta: None,
                cold_force_duration_ms: None,
                cold_verified: None,
                cold_attempts: None,
                residency_pages_after: None,
                residency_total_pages: None,
                degraded_reason: None,
            },
            SyntheticStepMeasurement {
                step: steps[1].clone(),
                outcome: StepOutcomeKind::Success,
                duration_ms: 10.0,
                minflt_delta: None,
                majflt_delta: None,
                cold_force_duration_ms: None,
                cold_verified: None,
                cold_attempts: None,
                residency_pages_after: None,
                residency_total_pages: None,
                degraded_reason: None,
            },
        ];

        let (record, rows, _) = build_run_record_from_measurements(
            "run-0",
            &measurements,
            Some(FinalTip {
                height: 42,
                hash: "tip-hash".to_string(),
            }),
        )
        .expect("record");

        assert!(!record.success);
        assert_eq!(record.failed_step_index, Some(0));
        assert_eq!(record.counts.steps_executed, 1);
        assert_eq!(rows.len(), 1);
        assert_eq!(
            record.final_tip,
            Some(FinalTip {
                height: 42,
                hash: "tip-hash".to_string()
            })
        );
    }

    #[test]
    fn orchestrate_execute_rejects_unverified_cold_in_strict_mode() {
        let steps = trusted_steps(json!({
            "checkpoint": "checkpoint.chkjam",
            "kernel": "kernel.jam",
            "steps": [{ "type": "force_cold" }]
        }));
        let measurements = vec![SyntheticStepMeasurement {
            step: steps[0].clone(),
            outcome: StepOutcomeKind::Error,
            duration_ms: 10.0,
            minflt_delta: None,
            majflt_delta: None,
            cold_force_duration_ms: Some(10.0),
            cold_verified: Some(false),
            cold_attempts: None,
            residency_pages_after: None,
            residency_total_pages: None,
            degraded_reason: Some("memory_reclaim_eagain".to_string()),
        }];

        assert!(matches!(
            build_run_record_from_measurements("run-0", &measurements, None),
            Err(OrchestrateExecuteError::UnverifiedColdStrict { .. })
        ));
    }

    #[test]
    fn orchestrate_execute_allows_enumerated_degraded_cold_reasons() {
        for reason in [
            "mincore_unavailable", "memory_reclaim_eagain", "partial_pageout",
            "swappiness_unwritable",
        ] {
            let steps = trusted_steps(json!({
                "checkpoint": "checkpoint.chkjam",
                "kernel": "kernel.jam",
                "steps": [{ "type": "force_cold" }, { "type": "peek_height", "height": 1 }]
            }));
            let measurements = vec![
                SyntheticStepMeasurement {
                    step: steps[0].clone(),
                    outcome: StepOutcomeKind::Error,
                    duration_ms: 10.0,
                    minflt_delta: None,
                    majflt_delta: None,
                    cold_force_duration_ms: Some(10.0),
                    cold_verified: Some(false),
                    cold_attempts: None,
                    residency_pages_after: None,
                    residency_total_pages: None,
                    degraded_reason: Some(reason.to_string()),
                },
                SyntheticStepMeasurement {
                    step: steps[1].clone(),
                    outcome: StepOutcomeKind::Success,
                    duration_ms: 10.0,
                    minflt_delta: None,
                    majflt_delta: None,
                    cold_force_duration_ms: None,
                    cold_verified: None,
                    cold_attempts: None,
                    residency_pages_after: None,
                    residency_total_pages: None,
                    degraded_reason: None,
                },
            ];

            let (_record, _steps, cold) =
                build_run_record_from_measurements_with_policy("run-0", &measurements, None, true)
                    .expect("degraded cold allowed");
            assert_eq!(cold[0].cold_verified, false);
            assert_eq!(cold[0].degraded_reason.as_deref(), Some(reason));
            assert_eq!(cold[0].peek_completed, None);
            assert_eq!(cold[0].peek_outcome, None);
        }
    }

    #[test]
    fn orchestrate_execute_rejects_unknown_degraded_cold_reason() {
        let steps = trusted_steps(json!({
            "checkpoint": "checkpoint.chkjam",
            "kernel": "kernel.jam",
            "steps": [{ "type": "peek_height_cold", "height": 1 }]
        }));
        let measurements = vec![SyntheticStepMeasurement {
            step: steps[0].clone(),
            outcome: StepOutcomeKind::Error,
            duration_ms: 10.0,
            minflt_delta: None,
            majflt_delta: None,
            cold_force_duration_ms: Some(10.0),
            cold_verified: Some(false),
            cold_attempts: None,
            residency_pages_after: None,
            residency_total_pages: None,
            degraded_reason: Some("unknown".to_string()),
        }];

        assert!(matches!(
            build_run_record_from_measurements_with_policy("run-0", &measurements, None, true),
            Err(OrchestrateExecuteError::UnknownDegradedColdReason { .. })
        ));
    }

    #[test]
    fn orchestrate_execute_cold_evidence_distinguishes_force_and_peek_companions() {
        let steps = trusted_steps(json!({
            "checkpoint": "checkpoint.chkjam",
            "kernel": "kernel.jam",
            "steps": [
                { "type": "force_cold" },
                { "type": "peek_height_cold", "height": 1 }
            ]
        }));
        let measurements = vec![
            SyntheticStepMeasurement {
                step: steps[0].clone(),
                outcome: StepOutcomeKind::Ok,
                duration_ms: 11.0,
                minflt_delta: None,
                majflt_delta: None,
                cold_force_duration_ms: Some(11.0),
                cold_verified: Some(true),
                cold_attempts: None,
                residency_pages_after: None,
                residency_total_pages: None,
                degraded_reason: None,
            },
            SyntheticStepMeasurement {
                step: steps[1].clone(),
                outcome: StepOutcomeKind::Success,
                duration_ms: 7.0,
                minflt_delta: None,
                majflt_delta: None,
                cold_force_duration_ms: Some(13.0),
                cold_verified: Some(true),
                cold_attempts: None,
                residency_pages_after: None,
                residency_total_pages: None,
                degraded_reason: None,
            },
        ];

        let (_record, _steps, cold) =
            build_run_record_from_measurements("run-0", &measurements, None).expect("record");
        assert_eq!(cold[0].cold_force_duration_ms, 11.0);
        assert_eq!(cold[0].peek_completed, None);
        assert_eq!(cold[0].peek_outcome, None);
        assert_eq!(cold[1].cold_force_duration_ms, 13.0);
        assert_eq!(cold[1].peek_completed, Some(true));
        assert_eq!(cold[1].peek_outcome.as_deref(), Some("success"));
    }
}
