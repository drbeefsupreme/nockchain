use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalMetric {
    ThroughputBlocksS,
    InitTimeS,
    TotalPokeTimeS,
    AvgPerBlockMs,
    PeakRssMib,
    P95RssMib,
    MinorFaultsDelta,
    MajorFaultsDelta,
    Checkpoints,
    FailedPokes,
    ExitStatus,
}

/// Direction indicating whether higher or lower values are better for a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    /// Higher values are better (e.g., throughput).
    Higher,
    /// Lower values are better (e.g., latency, memory).
    Lower,
}

impl CanonicalMetric {
    /// Returns the direction indicating whether higher or lower values are better.
    pub fn metric_direction(&self) -> MetricDirection {
        match self {
            CanonicalMetric::ThroughputBlocksS => MetricDirection::Higher,
            CanonicalMetric::InitTimeS
            | CanonicalMetric::TotalPokeTimeS
            | CanonicalMetric::AvgPerBlockMs
            | CanonicalMetric::PeakRssMib
            | CanonicalMetric::P95RssMib
            | CanonicalMetric::MinorFaultsDelta
            | CanonicalMetric::MajorFaultsDelta
            | CanonicalMetric::Checkpoints
            | CanonicalMetric::FailedPokes
            | CanonicalMetric::ExitStatus => MetricDirection::Lower,
        }
    }

    /// All known canonical metrics.
    pub fn all() -> &'static [CanonicalMetric] {
        &[
            CanonicalMetric::ThroughputBlocksS,
            CanonicalMetric::InitTimeS,
            CanonicalMetric::TotalPokeTimeS,
            CanonicalMetric::AvgPerBlockMs,
            CanonicalMetric::PeakRssMib,
            CanonicalMetric::P95RssMib,
            CanonicalMetric::MinorFaultsDelta,
            CanonicalMetric::MajorFaultsDelta,
            CanonicalMetric::Checkpoints,
            CanonicalMetric::FailedPokes,
            CanonicalMetric::ExitStatus,
        ]
    }
}

/// Four-way classification of a candidate metric vs baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonVerdict {
    Improvement,
    Regression,
    NoSignificantChange,
    Inconclusive,
}

impl ComparisonVerdict {
    /// Severity ordering for computing overall verdict (higher = worse).
    pub fn severity_rank(&self) -> u8 {
        match self {
            ComparisonVerdict::Improvement => 0,
            ComparisonVerdict::NoSignificantChange => 1,
            ComparisonVerdict::Inconclusive => 2,
            ComparisonVerdict::Regression => 3,
        }
    }
}

/// Result of comparing a single metric between candidate and baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub metric: CanonicalMetric,
    pub verdict: ComparisonVerdict,
    pub candidate_value: f64,
    pub baseline_median: f64,
    pub baseline_mad: f64,
    pub delta_pct: f64,
    pub delta_abs: f64,
    pub confidence: f64,
    pub baseline_samples: usize,
    pub reason: String,
}

/// Full comparison report across all metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub results: Vec<ComparisonResult>,
    pub overall_verdict: ComparisonVerdict,
    pub candidate_source: String,
    pub baseline_source: String,
    pub baseline_total_samples: usize,
    pub significance_threshold: f64,
}

/// Per-metric threshold overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricOverride {
    pub significance_threshold: Option<f64>,
    pub min_samples: Option<usize>,
}

/// Configuration for comparison runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonConfig {
    pub significance_threshold: f64,
    pub min_samples: usize,
    pub bootstrap_iterations: usize,
    pub bootstrap_seed: u64,
    #[serde(default)]
    pub metric_overrides: BTreeMap<String, MetricOverride>,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            significance_threshold: 0.10,
            min_samples: 3,
            bootstrap_iterations: 500,
            bootstrap_seed: 42,
            metric_overrides: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Warn,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardVerdict {
    Pass,
    Warn,
    Fail,
    InsufficientBaseline,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselineKey {
    pub env: String,
    pub fixture: String,
    pub branch: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselinePolicy {
    pub window_runs: usize,
    pub max_age_days: u32,
    pub min_samples: usize,
}

impl Default for BaselinePolicy {
    fn default() -> Self {
        Self {
            window_runs: 20,
            max_age_days: 30,
            min_samples: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractRule {
    pub metric: CanonicalMetric,
    pub floor_pct_of_baseline: Option<f64>,
    pub ceiling_pct_of_baseline: Option<f64>,
    pub absolute_floor: Option<f64>,
    pub absolute_ceiling: Option<f64>,
    #[serde(default = "default_severity")]
    pub severity: Severity,
    #[serde(default = "default_weight")]
    pub weight: f64,
}

fn default_severity() -> Severity {
    Severity::Fail
}

fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardContract {
    pub metadata: ContractMetadata,
    #[serde(default)]
    pub baseline: BaselinePolicy,
    #[serde(default)]
    pub rules: BTreeMap<String, ContractRule>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractMetadata {
    pub name: String,
    #[serde(default)]
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardMetricResult {
    pub metric: CanonicalMetric,
    pub candidate_value: f64,
    pub baseline_median: Option<f64>,
    pub baseline_mad: Option<f64>,
    pub delta_pct: Option<f64>,
    pub severity: Severity,
    pub passed: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutopsyHint {
    pub summary: String,
    #[serde(default)]
    pub suspects: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportContext {
    pub run_id: String,
    pub env: String,
    pub fixture: String,
    pub branch: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardReport {
    pub context: ReportContext,
    pub verdict: GuardVerdict,
    pub baseline_key: BaselineKey,
    pub baseline_samples: usize,
    pub metrics: Vec<GuardMetricResult>,
    #[serde(default)]
    pub autopsy: Vec<AutopsyHint>,
}
