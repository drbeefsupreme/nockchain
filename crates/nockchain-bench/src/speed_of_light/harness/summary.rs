use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{
    is_release_build, DEFAULT_THROUGHPUT_CV_THRESHOLD, SUMMARY_SCHEMA_VERSION,
    VERDICT_SCHEMA_VERSION,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueStats {
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub mad: f64,
    pub stddev: f64,
    pub cv: f64,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMetrics {
    pub throughput_blocks_per_second: f64,
    pub steps_per_second: Option<f64>,
    pub pokes_per_second: Option<f64>,
    pub peeks_per_second: Option<f64>,
    pub cold_peeks_per_second: Option<f64>,
    pub init_time_secs: f64,
    pub total_replay_time_secs: f64,
    pub average_block_time_ms: f64,
    pub failed_pokes: f64,
    pub checkpoint_count: f64,
    pub average_checkpoint_time_secs: f64,
    pub peak_process_rss_bytes: Option<f64>,
    pub minor_faults_total: Option<f64>,
    pub major_faults_total: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunFailure {
    pub run_id: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    #[serde(default = "summary_schema_version")]
    pub schema_version: String,
    pub measured_runs_requested: u32,
    pub measured_runs_succeeded: usize,
    pub failed_runs: Vec<RunFailure>,
    #[serde(default)]
    pub aggregate: BTreeMap<String, ValueStats>,
    #[serde(default)]
    pub by_step_type: BTreeMap<String, BTreeMap<String, ValueStats>>,
    #[serde(default)]
    pub steps: Vec<StepSummary>,
    pub throughput_blocks_per_second: Option<ValueStats>,
    #[serde(default)]
    pub steps_per_second: Option<ValueStats>,
    #[serde(default)]
    pub pokes_per_second: Option<ValueStats>,
    #[serde(default)]
    pub peeks_per_second: Option<ValueStats>,
    #[serde(default)]
    pub cold_peeks_per_second: Option<ValueStats>,
    pub init_time_secs: Option<ValueStats>,
    pub total_replay_time_secs: Option<ValueStats>,
    pub average_block_time_ms: Option<ValueStats>,
    pub failed_pokes: Option<ValueStats>,
    pub checkpoint_count: Option<ValueStats>,
    pub average_checkpoint_time_secs: Option<ValueStats>,
    pub peak_process_rss_bytes: Option<ValueStats>,
    pub minor_faults_total: Option<ValueStats>,
    pub major_faults_total: Option<ValueStats>,
}

#[derive(Debug, Clone)]
pub struct RunSummaryInput {
    pub measured_run_count: u32,
    pub run_failures: Vec<RunFailure>,
    pub throughput_cv: Option<f64>,
    pub cv_threshold: f64,
    pub release_build: bool,
    pub allow_debug_benchmark: bool,
    pub invalid_reasons: Vec<String>,
    pub partial_reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepSummary {
    pub step_id: String,
    #[serde(rename = "type")]
    pub step_type: String,
    pub duration_ms: Option<ValueStats>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Validity {
    Valid,
    Partial { reasons: Vec<String> },
    Invalid { reasons: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Verdict {
    #[serde(default = "verdict_schema_version")]
    pub schema_version: String,
    pub validity: Validity,
}

fn summary_schema_version() -> String {
    SUMMARY_SCHEMA_VERSION.to_string()
}

fn verdict_schema_version() -> String {
    VERDICT_SCHEMA_VERSION.to_string()
}

pub fn summarize_runs(
    metrics: &[RunMetrics],
    failed_runs: &[RunFailure],
    measured_runs_requested: u32,
) -> RunSummary {
    RunSummary {
        schema_version: SUMMARY_SCHEMA_VERSION.to_string(),
        measured_runs_requested,
        measured_runs_succeeded: metrics.len(),
        failed_runs: failed_runs.to_vec(),
        aggregate: aggregate_metrics(metrics),
        by_step_type: BTreeMap::new(),
        steps: Vec::new(),
        throughput_blocks_per_second: stats(
            metrics.iter().map(|run| run.throughput_blocks_per_second),
        ),
        steps_per_second: stats_option(metrics.iter().map(|run| run.steps_per_second)),
        pokes_per_second: stats_option(metrics.iter().map(|run| run.pokes_per_second)),
        peeks_per_second: stats_option(metrics.iter().map(|run| run.peeks_per_second)),
        cold_peeks_per_second: stats_option(metrics.iter().map(|run| run.cold_peeks_per_second)),
        init_time_secs: stats(metrics.iter().map(|run| run.init_time_secs)),
        total_replay_time_secs: stats(metrics.iter().map(|run| run.total_replay_time_secs)),
        average_block_time_ms: stats(metrics.iter().map(|run| run.average_block_time_ms)),
        failed_pokes: stats(metrics.iter().map(|run| run.failed_pokes)),
        checkpoint_count: stats(metrics.iter().map(|run| run.checkpoint_count)),
        average_checkpoint_time_secs: stats(
            metrics.iter().map(|run| run.average_checkpoint_time_secs),
        ),
        peak_process_rss_bytes: stats_option(metrics.iter().map(|run| run.peak_process_rss_bytes)),
        minor_faults_total: stats_option(metrics.iter().map(|run| run.minor_faults_total)),
        major_faults_total: stats_option(metrics.iter().map(|run| run.major_faults_total)),
    }
}

pub fn evaluate_verdict(input: &RunSummaryInput) -> Verdict {
    let mut invalid_reasons = input.invalid_reasons.clone();
    if input.measured_run_count > 0 && input.run_failures.len() == input.measured_run_count as usize
    {
        invalid_reasons.push("all measured runs failed".to_string());
    }
    if !input.release_build && !input.allow_debug_benchmark {
        invalid_reasons.push(
            "trusted runs require a release build unless --allow-debug-benchmark is set"
                .to_string(),
        );
    }

    if !invalid_reasons.is_empty() {
        return Verdict {
            schema_version: VERDICT_SCHEMA_VERSION.to_string(),
            validity: Validity::Invalid {
                reasons: invalid_reasons,
            },
        };
    }

    let mut partial_reasons = input.partial_reasons.clone();
    if !input.release_build && input.allow_debug_benchmark {
        partial_reasons.push("debug build used under --allow-debug-benchmark override".to_string());
    }

    for failure in &input.run_failures {
        partial_reasons.push(format!(
            "measured run {} failed: {}",
            failure.run_id, failure.reason
        ));
    }

    if let Some(cv) = input.throughput_cv {
        if cv > input.cv_threshold {
            partial_reasons.push(format!(
                "throughput CV {:.3} exceeded threshold {:.2}",
                cv, input.cv_threshold
            ));
        }
    }

    if partial_reasons.is_empty() {
        Verdict {
            schema_version: VERDICT_SCHEMA_VERSION.to_string(),
            validity: Validity::Valid,
        }
    } else {
        Verdict {
            schema_version: VERDICT_SCHEMA_VERSION.to_string(),
            validity: Validity::Partial {
                reasons: partial_reasons,
            },
        }
    }
}

fn aggregate_metrics(metrics: &[RunMetrics]) -> BTreeMap<String, ValueStats> {
    let mut aggregate = BTreeMap::new();
    if let Some(value) = stats(metrics.iter().map(|run| run.throughput_blocks_per_second)) {
        aggregate.insert("throughput_blocks_per_second".to_string(), value);
    }
    if let Some(value) = stats_option(metrics.iter().map(|run| run.steps_per_second)) {
        aggregate.insert("steps_per_second".to_string(), value);
    }
    if let Some(value) = stats_option(metrics.iter().map(|run| run.pokes_per_second)) {
        aggregate.insert("pokes_per_second".to_string(), value);
    }
    if let Some(value) = stats_option(metrics.iter().map(|run| run.peeks_per_second)) {
        aggregate.insert("peeks_per_second".to_string(), value);
    }
    if let Some(value) = stats_option(metrics.iter().map(|run| run.cold_peeks_per_second)) {
        aggregate.insert("cold_peeks_per_second".to_string(), value);
    }
    if let Some(value) = stats(metrics.iter().map(|run| run.init_time_secs)) {
        aggregate.insert("init_time_secs".to_string(), value);
    }
    if let Some(value) = stats(metrics.iter().map(|run| run.total_replay_time_secs)) {
        aggregate.insert("total_replay_time_secs".to_string(), value);
    }
    aggregate
}

pub fn current_release_build_verdict(
    measured_run_count: u32,
    run_failures: Vec<RunFailure>,
    throughput_cv: Option<f64>,
    allow_debug_benchmark: bool,
) -> Verdict {
    evaluate_verdict(&RunSummaryInput {
        measured_run_count,
        run_failures,
        throughput_cv,
        cv_threshold: DEFAULT_THROUGHPUT_CV_THRESHOLD,
        release_build: is_release_build(),
        allow_debug_benchmark,
        invalid_reasons: Vec::new(),
        partial_reasons: Vec::new(),
    })
}

fn stats(values: impl Iterator<Item = f64>) -> Option<ValueStats> {
    let values: Vec<f64> = values.collect();
    if values.is_empty() {
        return None;
    }
    Some(compute_stats(values))
}

fn stats_option(values: impl Iterator<Item = Option<f64>>) -> Option<ValueStats> {
    let values: Vec<f64> = values.flatten().collect();
    if values.is_empty() {
        return None;
    }
    Some(compute_stats(values))
}

fn compute_stats(mut values: Vec<f64>) -> ValueStats {
    values.sort_by(|left, right| left.total_cmp(right));
    let mean = values.iter().copied().sum::<f64>() / values.len() as f64;
    let median_value = median(&values);
    let deviations: Vec<f64> = values.iter().map(|value| (value - mean).powi(2)).collect();
    let stddev = (deviations.iter().sum::<f64>() / values.len() as f64).sqrt();
    let mut mad_values: Vec<f64> = values
        .iter()
        .map(|value| (value - median_value).abs())
        .collect();
    mad_values.sort_by(|left, right| left.total_cmp(right));
    let mad = median(&mad_values);

    ValueStats {
        median: median_value,
        min: *values.first().unwrap_or(&0.0),
        max: *values.last().unwrap_or(&0.0),
        mad,
        stddev,
        cv: if mean.abs() > f64::EPSILON {
            stddev / mean.abs()
        } else {
            0.0
        },
        values,
    }
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_summary_computes_spread_metrics() {
        let summary = summarize_runs(
            &[
                RunMetrics {
                    throughput_blocks_per_second: 10.0,
                    steps_per_second: None,
                    pokes_per_second: Some(10.0),
                    peeks_per_second: None,
                    cold_peeks_per_second: None,
                    init_time_secs: 1.0,
                    total_replay_time_secs: 2.0,
                    average_block_time_ms: 100.0,
                    failed_pokes: 0.0,
                    checkpoint_count: 1.0,
                    average_checkpoint_time_secs: 0.5,
                    peak_process_rss_bytes: Some(100.0),
                    minor_faults_total: Some(10.0),
                    major_faults_total: Some(1.0),
                },
                RunMetrics {
                    throughput_blocks_per_second: 14.0,
                    steps_per_second: None,
                    pokes_per_second: Some(14.0),
                    peeks_per_second: None,
                    cold_peeks_per_second: None,
                    init_time_secs: 3.0,
                    total_replay_time_secs: 4.0,
                    average_block_time_ms: 140.0,
                    failed_pokes: 1.0,
                    checkpoint_count: 2.0,
                    average_checkpoint_time_secs: 0.8,
                    peak_process_rss_bytes: Some(200.0),
                    minor_faults_total: Some(30.0),
                    major_faults_total: Some(2.0),
                },
                RunMetrics {
                    throughput_blocks_per_second: 18.0,
                    steps_per_second: None,
                    pokes_per_second: Some(18.0),
                    peeks_per_second: None,
                    cold_peeks_per_second: None,
                    init_time_secs: 5.0,
                    total_replay_time_secs: 6.0,
                    average_block_time_ms: 180.0,
                    failed_pokes: 0.0,
                    checkpoint_count: 3.0,
                    average_checkpoint_time_secs: 1.1,
                    peak_process_rss_bytes: Some(300.0),
                    minor_faults_total: Some(50.0),
                    major_faults_total: Some(3.0),
                },
            ],
            &[],
            3,
        );

        let throughput = summary
            .throughput_blocks_per_second
            .expect("throughput stats");
        assert_eq!(throughput.median, 14.0);
        assert_eq!(throughput.min, 10.0);
        assert_eq!(throughput.max, 18.0);
        assert!((throughput.mad - 4.0).abs() < 1e-9);
        assert!(throughput.stddev > 0.0);
        assert!(throughput.cv > 0.0);
    }
}
