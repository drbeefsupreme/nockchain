use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use thiserror::Error;

use super::ingest::parse_combined_summary_tsv;
use super::model::{
    CanonicalMetric, ComparisonConfig, ComparisonReport, ComparisonResult, ComparisonVerdict,
    MetricDirection,
};
use super::stats::{bootstrap_median_ci, mad, median, ConfidenceInterval};

#[derive(Debug, Error)]
pub enum ComparisonError {
    #[error("failed to read {path}: {source}")]
    IoError {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse TSV: {0}")]
    ParseError(String),
    #[error("no metrics found in comparison data")]
    NoMetrics,
}

/// Classify a candidate vs baseline based on confidence interval overlap and metric direction.
pub fn classify_verdict(
    candidate_ci: ConfidenceInterval,
    baseline_ci: ConfidenceInterval,
    direction: MetricDirection,
    actual_samples: usize,
    min_samples: usize,
) -> ComparisonVerdict {
    if actual_samples < min_samples {
        return ComparisonVerdict::Inconclusive;
    }

    // Check CI overlap
    let overlaps = candidate_ci.low <= baseline_ci.high && baseline_ci.low <= candidate_ci.high;

    if overlaps {
        return ComparisonVerdict::NoSignificantChange;
    }

    // No overlap — determine direction
    let candidate_above = candidate_ci.low > baseline_ci.high;

    match direction {
        MetricDirection::Higher => {
            if candidate_above {
                ComparisonVerdict::Improvement
            } else {
                ComparisonVerdict::Regression
            }
        }
        MetricDirection::Lower => {
            if candidate_above {
                ComparisonVerdict::Regression
            } else {
                ComparisonVerdict::Improvement
            }
        }
    }
}

/// Compare a single metric between candidate and baseline value sets.
pub fn compare_metric(
    metric: CanonicalMetric,
    candidate_values: &[f64],
    baseline_values: &[f64],
    config: &ComparisonConfig,
) -> ComparisonResult {
    let metric_name = format!("{:?}", metric);

    // Check for per-metric overrides
    let effective_threshold = config
        .metric_overrides
        .get(&metric_name)
        .and_then(|o| o.significance_threshold)
        .unwrap_or(config.significance_threshold);

    let effective_min_samples = config
        .metric_overrides
        .get(&metric_name)
        .and_then(|o| o.min_samples)
        .unwrap_or(config.min_samples);

    let mut baseline_vals = baseline_values.to_vec();
    let baseline_med = median(&mut baseline_vals).unwrap_or(0.0);
    let baseline_mad_val = mad(baseline_values, baseline_med).unwrap_or(0.0);

    let mut candidate_vals = candidate_values.to_vec();
    let candidate_med = median(&mut candidate_vals).unwrap_or(0.0);

    let confidence = 1.0 - effective_threshold;

    // Compute bootstrap CIs
    let candidate_ci = bootstrap_median_ci(
        candidate_values,
        config.bootstrap_iterations,
        effective_threshold,
        config.bootstrap_seed,
    )
    .unwrap_or(ConfidenceInterval {
        low: candidate_med,
        high: candidate_med,
    });

    let baseline_ci = bootstrap_median_ci(
        baseline_values,
        config.bootstrap_iterations,
        effective_threshold,
        config.bootstrap_seed.wrapping_add(1),
    )
    .unwrap_or(ConfidenceInterval {
        low: baseline_med,
        high: baseline_med,
    });

    // Compute deltas
    let delta_abs = candidate_med - baseline_med;
    let delta_pct = if baseline_med.abs() > f64::EPSILON {
        (candidate_med - baseline_med) / baseline_med * 100.0
    } else {
        0.0
    };

    let direction = metric.metric_direction();
    let verdict = classify_verdict(
        candidate_ci,
        baseline_ci,
        direction,
        baseline_values.len(),
        effective_min_samples,
    );

    let reason = match verdict {
        ComparisonVerdict::Improvement => format!(
            "Candidate CI [{:.4}, {:.4}] does not overlap baseline CI [{:.4}, {:.4}]; direction {:?} favors candidate ({:+.1}%)",
            candidate_ci.low, candidate_ci.high, baseline_ci.low, baseline_ci.high, direction, delta_pct
        ),
        ComparisonVerdict::Regression => format!(
            "Candidate CI [{:.4}, {:.4}] does not overlap baseline CI [{:.4}, {:.4}]; direction {:?} indicates regression ({:+.1}%)",
            candidate_ci.low, candidate_ci.high, baseline_ci.low, baseline_ci.high, direction, delta_pct
        ),
        ComparisonVerdict::NoSignificantChange => format!(
            "Candidate CI [{:.4}, {:.4}] overlaps baseline CI [{:.4}, {:.4}]; no significant change ({:+.1}%)",
            candidate_ci.low, candidate_ci.high, baseline_ci.low, baseline_ci.high, delta_pct
        ),
        ComparisonVerdict::Inconclusive => format!(
            "Insufficient baseline samples ({} < {}); cannot determine significance",
            baseline_values.len(), effective_min_samples
        ),
    };

    ComparisonResult {
        metric,
        verdict,
        candidate_value: candidate_med,
        baseline_median: baseline_med,
        baseline_mad: baseline_mad_val,
        delta_pct,
        delta_abs,
        confidence,
        baseline_samples: baseline_values.len(),
        reason,
    }
}

/// Run a full comparison of candidate vs baseline TSV files.
pub fn run_comparison(
    candidate_tsv: &Path,
    baseline_tsv: &Path,
    config: &ComparisonConfig,
) -> Result<ComparisonReport, ComparisonError> {
    let candidate_rows =
        parse_combined_summary_tsv(candidate_tsv).map_err(|e| ComparisonError::ParseError(e.to_string()))?;
    let baseline_rows =
        parse_combined_summary_tsv(baseline_tsv).map_err(|e| ComparisonError::ParseError(e.to_string()))?;

    if candidate_rows.is_empty() && baseline_rows.is_empty() {
        return Err(ComparisonError::NoMetrics);
    }

    // Collect values per metric
    let mut candidate_by_metric: BTreeMap<CanonicalMetric, Vec<f64>> = BTreeMap::new();
    let mut baseline_by_metric: BTreeMap<CanonicalMetric, Vec<f64>> = BTreeMap::new();

    for row in &candidate_rows {
        for metric in CanonicalMetric::all() {
            candidate_by_metric
                .entry(*metric)
                .or_default()
                .push(row.metric_value(*metric));
        }
    }

    for row in &baseline_rows {
        for metric in CanonicalMetric::all() {
            baseline_by_metric
                .entry(*metric)
                .or_default()
                .push(row.metric_value(*metric));
        }
    }

    let mut results = Vec::new();
    for metric in CanonicalMetric::all() {
        let candidate_vals = candidate_by_metric.get(metric).map(|v| v.as_slice()).unwrap_or(&[]);
        let baseline_vals = baseline_by_metric.get(metric).map(|v| v.as_slice()).unwrap_or(&[]);

        if candidate_vals.is_empty() && baseline_vals.is_empty() {
            continue;
        }

        results.push(compare_metric(*metric, candidate_vals, baseline_vals, config));
    }

    if results.is_empty() {
        return Err(ComparisonError::NoMetrics);
    }

    // Overall verdict = worst across all metrics
    let overall_verdict = results
        .iter()
        .map(|r| r.verdict)
        .max_by_key(|v| v.severity_rank())
        .unwrap_or(ComparisonVerdict::Inconclusive);

    let baseline_total_samples = baseline_rows.len();

    Ok(ComparisonReport {
        results,
        overall_verdict,
        candidate_source: candidate_tsv.display().to_string(),
        baseline_source: baseline_tsv.display().to_string(),
        baseline_total_samples,
        significance_threshold: config.significance_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_cis_yield_no_significant_change() {
        let candidate_ci = ConfidenceInterval { low: 9.0, high: 11.0 };
        let baseline_ci = ConfidenceInterval { low: 10.0, high: 12.0 };
        let verdict = classify_verdict(candidate_ci, baseline_ci, MetricDirection::Higher, 5, 3);
        assert_eq!(verdict, ComparisonVerdict::NoSignificantChange);
    }

    #[test]
    fn non_overlapping_higher_above_is_improvement() {
        let candidate_ci = ConfidenceInterval { low: 13.0, high: 15.0 };
        let baseline_ci = ConfidenceInterval { low: 10.0, high: 12.0 };
        let verdict = classify_verdict(candidate_ci, baseline_ci, MetricDirection::Higher, 5, 3);
        assert_eq!(verdict, ComparisonVerdict::Improvement);
    }

    #[test]
    fn non_overlapping_higher_below_is_regression() {
        let candidate_ci = ConfidenceInterval { low: 7.0, high: 9.0 };
        let baseline_ci = ConfidenceInterval { low: 10.0, high: 12.0 };
        let verdict = classify_verdict(candidate_ci, baseline_ci, MetricDirection::Higher, 5, 3);
        assert_eq!(verdict, ComparisonVerdict::Regression);
    }

    #[test]
    fn non_overlapping_lower_below_is_improvement() {
        let candidate_ci = ConfidenceInterval { low: 7.0, high: 9.0 };
        let baseline_ci = ConfidenceInterval { low: 10.0, high: 12.0 };
        let verdict = classify_verdict(candidate_ci, baseline_ci, MetricDirection::Lower, 5, 3);
        assert_eq!(verdict, ComparisonVerdict::Improvement);
    }

    #[test]
    fn insufficient_samples_is_inconclusive() {
        let candidate_ci = ConfidenceInterval { low: 13.0, high: 15.0 };
        let baseline_ci = ConfidenceInterval { low: 10.0, high: 12.0 };
        let verdict = classify_verdict(candidate_ci, baseline_ci, MetricDirection::Higher, 2, 3);
        assert_eq!(verdict, ComparisonVerdict::Inconclusive);
    }
}
