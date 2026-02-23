use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
