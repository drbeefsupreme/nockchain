use std::path::{Path, PathBuf};

use thiserror::Error;

use super::ingest::CombinedSummaryRow;
use super::model::{GuardContract, GuardMetricResult, GuardVerdict};
use super::stats::{bootstrap_median_ci, mad, median};

#[derive(Debug, Error)]
pub enum ContractError {
    #[error("failed to read contract {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse contract {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

pub fn load_contract(path: &Path) -> Result<GuardContract, ContractError> {
    let content = std::fs::read_to_string(path).map_err(|source| ContractError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    toml::from_str(&content).map_err(|source| ContractError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContractEvaluation {
    pub verdict: GuardVerdict,
    pub metric_results: Vec<GuardMetricResult>,
    pub baseline_samples: usize,
    pub insufficient_baseline_reason: Option<String>,
}

pub fn evaluate_contract(
    candidate: &CombinedSummaryRow,
    baseline_rows: &[&CombinedSummaryRow],
    contract: &GuardContract,
) -> ContractEvaluation {
    if baseline_rows.len() < contract.baseline.min_samples {
        return ContractEvaluation {
            verdict: GuardVerdict::InsufficientBaseline,
            metric_results: Vec::new(),
            baseline_samples: baseline_rows.len(),
            insufficient_baseline_reason: Some(format!(
                "need at least {} baseline rows, got {}",
                contract.baseline.min_samples,
                baseline_rows.len()
            )),
        };
    }

    let mut metric_results = Vec::new();
    let mut fail_weight = 0.0f64;
    let mut warn_weight = 0.0f64;

    for rule in contract.rules.values() {
        let candidate_value = candidate.metric_value(rule.metric);
        let mut baseline_values: Vec<f64> = baseline_rows
            .iter()
            .map(|row| row.metric_value(rule.metric))
            .collect();
        let baseline_median = median(&mut baseline_values);
        let baseline_mad = baseline_median.and_then(|m| mad(&baseline_values, m));
        let ci = bootstrap_median_ci(&baseline_values, 500, 0.05, 42);
        let delta_pct = baseline_median.and_then(|m| {
            if m.abs() <= f64::EPSILON {
                None
            } else {
                Some(((candidate_value - m) / m) * 100.0)
            }
        });

        let mut passed = true;
        let mut reasons = Vec::new();

        if let Some(floor_pct) = rule.floor_pct_of_baseline {
            if let Some(m) = baseline_median {
                let floor = m * floor_pct;
                if candidate_value < floor {
                    passed = false;
                    reasons.push(format!(
                        "candidate {:.4} < floor {:.4} ({:.1}% of baseline)",
                        candidate_value,
                        floor,
                        floor_pct * 100.0
                    ));
                }
            }
        }
        if let Some(ceiling_pct) = rule.ceiling_pct_of_baseline {
            if let Some(m) = baseline_median {
                let ceiling = m * ceiling_pct;
                if candidate_value > ceiling {
                    passed = false;
                    reasons.push(format!(
                        "candidate {:.4} > ceiling {:.4} ({:.1}% of baseline)",
                        candidate_value,
                        ceiling,
                        ceiling_pct * 100.0
                    ));
                }
            }
        }
        if let Some(absolute_floor) = rule.absolute_floor {
            if candidate_value < absolute_floor {
                passed = false;
                reasons.push(format!(
                    "candidate {:.4} < absolute floor {:.4}",
                    candidate_value, absolute_floor
                ));
            }
        }
        if let Some(absolute_ceiling) = rule.absolute_ceiling {
            if candidate_value > absolute_ceiling {
                passed = false;
                reasons.push(format!(
                    "candidate {:.4} > absolute ceiling {:.4}",
                    candidate_value, absolute_ceiling
                ));
            }
        }

        if !passed {
            if matches!(rule.severity, super::model::Severity::Fail) {
                fail_weight += rule.weight.max(0.0);
            } else {
                warn_weight += rule.weight.max(0.0);
            }
        }

        let reason = if reasons.is_empty() {
            if let Some(ci) = ci {
                format!(
                    "within contract (baseline median CI [{:.4}, {:.4}])",
                    ci.low, ci.high
                )
            } else {
                "within contract".to_string()
            }
        } else {
            reasons.join("; ")
        };

        metric_results.push(GuardMetricResult {
            metric: rule.metric,
            candidate_value,
            baseline_median,
            baseline_mad,
            delta_pct,
            severity: rule.severity,
            passed,
            reason,
        });
    }

    let verdict = if fail_weight > 0.0 {
        GuardVerdict::Fail
    } else if warn_weight > 0.0 {
        GuardVerdict::Warn
    } else {
        GuardVerdict::Pass
    };

    ContractEvaluation {
        verdict,
        metric_results,
        baseline_samples: baseline_rows.len(),
        insufficient_baseline_reason: None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{evaluate_contract, GuardContract};
    use crate::speed_of_light::guard::ingest::CombinedSummaryRow;

    fn row(
        throughput: f64,
        init_s: f64,
        peak_rss_mib: f64,
        major_faults_delta: f64,
    ) -> CombinedSummaryRow {
        CombinedSummaryRow {
            pass: 1,
            env: "native".to_string(),
            branch: "master".to_string(),
            fixture: "v0".to_string(),
            throughput_blocks_s: throughput,
            init_time_s: init_s,
            total_poke_time_s: 10.0,
            avg_per_block_ms: 100.0,
            peak_rss_mib,
            p95_rss_mib: peak_rss_mib - 2.0,
            minor_faults_delta: 50_000.0,
            major_faults_delta,
            checkpoints: 0.0,
            failed_pokes: 0.0,
            exit_status: 0,
            raw: HashMap::new(),
        }
    }

    fn contract() -> GuardContract {
        let toml = r#"
        [metadata]
        name = "test"
        version = "1"

        [baseline]
        window_runs = 20
        max_age_days = 30
        min_samples = 3

        [rules.throughput]
        metric = "throughput_blocks_s"
        floor_pct_of_baseline = 0.95
        severity = "fail"
        weight = 1.0

        [rules.init]
        metric = "init_time_s"
        ceiling_pct_of_baseline = 1.10
        severity = "warn"
        weight = 0.5
        "#;
        toml::from_str(toml).expect("contract parse")
    }

    #[test]
    fn evaluates_pass_warn_fail() {
        let c = contract();
        let b1 = row(10.0, 0.10, 800.0, 1.0);
        let b2 = row(10.2, 0.11, 790.0, 1.0);
        let b3 = row(9.9, 0.10, 795.0, 1.0);
        let baseline = vec![&b1, &b2, &b3];

        let pass = row(10.1, 0.10, 801.0, 1.0);
        let pass_eval = evaluate_contract(&pass, &baseline, &c);
        assert!(matches!(
            pass_eval.verdict,
            crate::speed_of_light::guard::GuardVerdict::Pass
        ));

        let warn = row(10.0, 0.14, 801.0, 1.0);
        let warn_eval = evaluate_contract(&warn, &baseline, &c);
        assert!(matches!(
            warn_eval.verdict,
            crate::speed_of_light::guard::GuardVerdict::Warn
        ));

        let fail = row(8.0, 0.12, 801.0, 1.0);
        let fail_eval = evaluate_contract(&fail, &baseline, &c);
        assert!(matches!(
            fail_eval.verdict,
            crate::speed_of_light::guard::GuardVerdict::Fail
        ));
    }

    #[test]
    fn reports_insufficient_baseline() {
        let c = contract();
        let only = row(10.0, 0.10, 800.0, 1.0);
        let baseline = vec![&only];
        let candidate = row(10.0, 0.10, 800.0, 1.0);
        let eval = evaluate_contract(&candidate, &baseline, &c);
        assert!(matches!(
            eval.verdict,
            crate::speed_of_light::guard::GuardVerdict::InsufficientBaseline
        ));
        assert!(eval.insufficient_baseline_reason.is_some());
    }
}
