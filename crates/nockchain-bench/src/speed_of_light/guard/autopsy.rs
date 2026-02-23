use std::collections::HashMap;
use std::path::Path;

use super::ingest::ProfileMetrics;
use super::model::{AutopsyHint, GuardMetricResult};

pub fn rank_metric_failures(metrics: &[GuardMetricResult]) -> Vec<&GuardMetricResult> {
    let mut failed: Vec<&GuardMetricResult> = metrics.iter().filter(|m| !m.passed).collect();
    failed.sort_by(|a, b| {
        let ad = a.delta_pct.unwrap_or_default().abs();
        let bd = b.delta_pct.unwrap_or_default().abs();
        bd.total_cmp(&ad)
    });
    failed
}

pub fn build_basic_hints(metrics: &[GuardMetricResult]) -> Vec<AutopsyHint> {
    rank_metric_failures(metrics)
        .into_iter()
        .take(3)
        .map(|m| AutopsyHint {
            summary: format!("{} regression: {}", metric_label(m), m.reason),
            suspects: Vec::new(),
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct SymbolShift {
    pub symbol: String,
    pub candidate_samples: u64,
    pub baseline_samples: f64,
    pub delta_pct: f64,
}

pub fn parse_folded_symbol_totals(path: &Path) -> Result<HashMap<String, u64>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    let mut symbols: HashMap<String, u64> = HashMap::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((stack, count_str)) = trimmed.rsplit_once(' ') else {
            continue;
        };
        let count = count_str.parse::<u64>().unwrap_or(0);
        let symbol = stack
            .split(';')
            .next_back()
            .unwrap_or("unknown")
            .trim()
            .to_string();
        if symbol.is_empty() {
            continue;
        }
        *symbols.entry(symbol).or_default() += count;
    }
    Ok(symbols)
}

pub fn detect_stack_shifts(
    candidate: &HashMap<String, u64>,
    baseline: &[HashMap<String, u64>],
    limit: usize,
) -> Vec<SymbolShift> {
    if baseline.is_empty() {
        return Vec::new();
    }
    let mut rows = Vec::new();
    for (symbol, &candidate_samples) in candidate {
        let mut total_baseline = 0.0;
        for b in baseline {
            total_baseline += *b.get(symbol).unwrap_or(&0) as f64;
        }
        let baseline_avg = total_baseline / baseline.len() as f64;
        if baseline_avg <= 0.0 {
            continue;
        }
        let delta_pct = ((candidate_samples as f64 - baseline_avg) / baseline_avg) * 100.0;
        rows.push(SymbolShift {
            symbol: symbol.clone(),
            candidate_samples,
            baseline_samples: baseline_avg,
            delta_pct,
        });
    }
    rows.sort_by(|a, b| b.delta_pct.total_cmp(&a.delta_pct));
    rows.into_iter().take(limit).collect()
}

pub fn detect_profile_anomalies(profile: &ProfileMetrics) -> Vec<AutopsyHint> {
    let mut out = Vec::new();
    if profile.samples.is_empty() {
        return out;
    }
    if profile.checkpoint_count > 0 {
        out.push(AutopsyHint {
            summary: format!(
                "Unexpected checkpoint activity during guard run (count={})",
                profile.checkpoint_count
            ),
            suspects: Vec::new(),
        });
    }
    let peak_rss_mib = profile.peak_rss_mib();
    if peak_rss_mib > 0.0 {
        out.push(AutopsyHint {
            summary: format!("Peak RSS observed at {:.1} MiB", peak_rss_mib),
            suspects: Vec::new(),
        });
    }
    let major_delta = profile.major_faults_delta();
    if major_delta > 0 {
        out.push(AutopsyHint {
            summary: format!("Major page faults increased by {}", major_delta),
            suspects: vec!["mmap/fault paths".to_string()],
        });
    }
    out
}

fn metric_label(metric: &GuardMetricResult) -> &'static str {
    match metric.metric {
        super::model::CanonicalMetric::ThroughputBlocksS => "throughput_blocks_s",
        super::model::CanonicalMetric::InitTimeS => "init_time_s",
        super::model::CanonicalMetric::TotalPokeTimeS => "total_poke_time_s",
        super::model::CanonicalMetric::AvgPerBlockMs => "avg_per_block_ms",
        super::model::CanonicalMetric::PeakRssMib => "peak_rss_mib",
        super::model::CanonicalMetric::P95RssMib => "p95_rss_mib",
        super::model::CanonicalMetric::MinorFaultsDelta => "minor_faults_delta",
        super::model::CanonicalMetric::MajorFaultsDelta => "major_faults_delta",
        super::model::CanonicalMetric::Checkpoints => "checkpoints",
        super::model::CanonicalMetric::FailedPokes => "failed_pokes",
        super::model::CanonicalMetric::ExitStatus => "exit_status",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::{detect_profile_anomalies, detect_stack_shifts, parse_folded_symbol_totals};
    use crate::speed_of_light::guard::ingest::parse_profile_metrics;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("guard")
            .join(name)
    }

    #[test]
    fn parses_folded_symbols() {
        let symbols =
            parse_folded_symbol_totals(&fixture_path("perf_regress.folded")).expect("folded");
        assert!(symbols.get("fault_handler").copied().unwrap_or(0) > 0);
    }

    #[test]
    fn computes_stack_shift_deltas() {
        let mut candidate = HashMap::new();
        candidate.insert("malloc".to_string(), 100);
        let mut base = HashMap::new();
        base.insert("malloc".to_string(), 40);
        let shifts = detect_stack_shifts(&candidate, &[base], 5);
        assert_eq!(shifts.len(), 1);
        assert!(shifts[0].delta_pct > 100.0);
    }

    #[test]
    fn reports_profile_anomalies() {
        let profile =
            parse_profile_metrics(&fixture_path("profile_regress.json")).expect("profile");
        let hints = detect_profile_anomalies(&profile);
        assert!(!hints.is_empty());
    }
}
