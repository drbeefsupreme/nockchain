use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::model::{BaselineKey, CanonicalMetric};

#[derive(Debug, Error)]
pub enum IngestError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid tsv row: {0}")]
    InvalidTsvRow(String),
    #[error("missing required tsv column: {0}")]
    MissingColumn(String),
    #[error("failed to parse json {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CombinedSummaryRow {
    pub pass: u32,
    pub env: String,
    pub branch: String,
    pub fixture: String,
    pub throughput_blocks_s: f64,
    pub init_time_s: f64,
    pub total_poke_time_s: f64,
    pub avg_per_block_ms: f64,
    pub peak_rss_mib: f64,
    pub p95_rss_mib: f64,
    pub minor_faults_delta: f64,
    pub major_faults_delta: f64,
    pub checkpoints: f64,
    pub failed_pokes: f64,
    pub exit_status: i32,
    pub raw: HashMap<String, String>,
}

impl CombinedSummaryRow {
    pub fn baseline_key(&self) -> BaselineKey {
        BaselineKey {
            env: self.env.clone(),
            fixture: self.fixture.clone(),
            branch: Some(self.branch.clone()),
        }
    }

    pub fn metric_value(&self, metric: CanonicalMetric) -> f64 {
        match metric {
            CanonicalMetric::ThroughputBlocksS => self.throughput_blocks_s,
            CanonicalMetric::InitTimeS => self.init_time_s,
            CanonicalMetric::TotalPokeTimeS => self.total_poke_time_s,
            CanonicalMetric::AvgPerBlockMs => self.avg_per_block_ms,
            CanonicalMetric::PeakRssMib => self.peak_rss_mib,
            CanonicalMetric::P95RssMib => self.p95_rss_mib,
            CanonicalMetric::MinorFaultsDelta => self.minor_faults_delta,
            CanonicalMetric::MajorFaultsDelta => self.major_faults_delta,
            CanonicalMetric::Checkpoints => self.checkpoints,
            CanonicalMetric::FailedPokes => self.failed_pokes,
            CanonicalMetric::ExitStatus => self.exit_status as f64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunsManifest {
    pub updated_on: Option<String>,
    pub latest_run_id: Option<String>,
    #[serde(default)]
    pub runs: Vec<RunsManifestEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunsManifestEntry {
    pub run_id: String,
    pub date: Option<String>,
}

pub fn parse_combined_summary_tsv(path: &Path) -> Result<Vec<CombinedSummaryRow>, IngestError> {
    let content = std::fs::read_to_string(path).map_err(|source| IngestError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut lines = content.lines();
    let header = lines
        .next()
        .ok_or_else(|| IngestError::InvalidTsvRow("missing header".to_string()))?;
    let cols: Vec<&str> = header.split('\t').collect();
    let mut out = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let values: Vec<&str> = line.split('\t').collect();
        if values.len() != cols.len() {
            return Err(IngestError::InvalidTsvRow(line.to_string()));
        }
        let raw: HashMap<String, String> = cols
            .iter()
            .zip(values.iter())
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        out.push(CombinedSummaryRow {
            pass: parse_u32(&raw, "pass")?,
            env: get_string(&raw, "env")?,
            branch: get_string(&raw, "branch")?,
            fixture: get_string(&raw, "fixture")?,
            throughput_blocks_s: parse_f64_default(&raw, "throughput_blocks_s"),
            init_time_s: parse_f64_default(&raw, "init_time_s"),
            total_poke_time_s: parse_f64_default(&raw, "total_poke_time_s"),
            avg_per_block_ms: parse_f64_default(&raw, "avg_per_block_ms"),
            peak_rss_mib: parse_f64_default(&raw, "peak_rss_mib"),
            p95_rss_mib: parse_f64_default(&raw, "p95_rss_mib"),
            minor_faults_delta: parse_f64_default(&raw, "minor_faults_delta"),
            major_faults_delta: parse_f64_default(&raw, "major_faults_delta"),
            checkpoints: parse_f64_default(&raw, "checkpoints"),
            failed_pokes: parse_f64_default(&raw, "failed_pokes"),
            exit_status: parse_i32_default(&raw, "exit_status"),
            raw,
        });
    }
    Ok(out)
}

pub fn parse_runs_manifest(path: &Path) -> Result<RunsManifest, IngestError> {
    let bytes = std::fs::read(path).map_err(|source| IngestError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| IngestError::Json {
        path: path.to_path_buf(),
        source,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileSample {
    pub rss_kb: u64,
    pub minor_faults: u64,
    pub major_faults: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileMetrics {
    pub checkpoint_count: u64,
    pub samples: Vec<ProfileSample>,
}

impl ProfileMetrics {
    pub fn peak_rss_mib(&self) -> f64 {
        self.samples
            .iter()
            .map(|s| s.rss_kb)
            .max()
            .map(|kb| kb as f64 / 1024.0)
            .unwrap_or(0.0)
    }

    pub fn minor_faults_delta(&self) -> u64 {
        faults_delta(&self.samples, |s| s.minor_faults)
    }

    pub fn major_faults_delta(&self) -> u64 {
        faults_delta(&self.samples, |s| s.major_faults)
    }
}

pub fn parse_profile_metrics(path: &Path) -> Result<ProfileMetrics, IngestError> {
    let bytes = std::fs::read(path).map_err(|source| IngestError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|source| IngestError::Json {
            path: path.to_path_buf(),
            source,
        })?;

    let checkpoint_count = v
        .get("checkpoint_count")
        .and_then(|n| n.as_u64())
        .unwrap_or(0);

    let mut samples = Vec::new();
    if let Some(arr) = v
        .get("memory_profile")
        .and_then(|mp| mp.get("samples"))
        .and_then(|s| s.as_array())
    {
        for sample in arr {
            samples.push(ProfileSample {
                rss_kb: sample.get("rss_kb").and_then(|n| n.as_u64()).unwrap_or(0),
                minor_faults: sample
                    .get("minor_faults")
                    .and_then(|n| n.as_u64())
                    .unwrap_or(0),
                major_faults: sample
                    .get("major_faults")
                    .and_then(|n| n.as_u64())
                    .unwrap_or(0),
            });
        }
    }

    Ok(ProfileMetrics {
        checkpoint_count,
        samples,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactPaths {
    pub perf_summary_json: Option<PathBuf>,
    pub perf_folded: Option<PathBuf>,
    pub profile_json: Option<PathBuf>,
}

pub fn resolve_row_artifacts(
    run_root: &Path,
    row: &CombinedSummaryRow,
) -> Result<ArtifactPaths, IngestError> {
    let perf_summary_json = row
        .raw
        .get("perf_summary_json")
        .filter(|v| !v.is_empty())
        .map(|v| run_root.join(v));
    let perf_folded = row
        .raw
        .get("perf_folded")
        .filter(|v| !v.is_empty())
        .map(|v| run_root.join(v));
    let profile_json = row
        .raw
        .get("profile_path")
        .filter(|v| !v.is_empty())
        .map(|v| run_root.join(v));
    Ok(ArtifactPaths {
        perf_summary_json,
        perf_folded,
        profile_json,
    })
}

fn get_string(raw: &HashMap<String, String>, key: &str) -> Result<String, IngestError> {
    raw.get(key)
        .cloned()
        .ok_or_else(|| IngestError::MissingColumn(key.to_string()))
}

fn parse_u32(raw: &HashMap<String, String>, key: &str) -> Result<u32, IngestError> {
    let value = get_string(raw, key)?;
    value.parse::<u32>().map_err(|_| {
        IngestError::InvalidTsvRow(format!("invalid u32 value for {}: {}", key, value))
    })
}

fn parse_f64_default(raw: &HashMap<String, String>, key: &str) -> f64 {
    raw.get(key)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_default()
}

fn parse_i32_default(raw: &HashMap<String, String>, key: &str) -> i32 {
    raw.get(key)
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or_default()
}

fn faults_delta(samples: &[ProfileSample], pick: impl Fn(&ProfileSample) -> u64) -> u64 {
    match (samples.first(), samples.last()) {
        (Some(first), Some(last)) => pick(last).saturating_sub(pick(first)),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        parse_combined_summary_tsv, parse_profile_metrics, parse_runs_manifest, IngestError,
    };

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("guard")
            .join(name)
    }

    #[test]
    fn parses_combined_summary_fixture() {
        let rows = parse_combined_summary_tsv(&fixture_path("combined_summary.tsv")).expect("rows");
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].env, "native");
        assert!(rows[0].throughput_blocks_s > 10.0);
    }

    #[test]
    fn errors_on_malformed_tsv_row() {
        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        std::fs::write(tmp.path(), "a\tb\n1\n").expect("write");
        let err = parse_combined_summary_tsv(tmp.path()).expect_err("should fail");
        assert!(matches!(err, IngestError::InvalidTsvRow(_)));
    }

    #[test]
    fn parses_runs_manifest_fixture() {
        let manifest = parse_runs_manifest(&fixture_path("runs-manifest.json")).expect("manifest");
        assert!(!manifest.runs.is_empty());
    }

    #[test]
    fn parses_profile_metrics_fixture() {
        let metrics =
            parse_profile_metrics(&fixture_path("profile_regress.json")).expect("profile");
        assert!(metrics.peak_rss_mib() > 900.0);
        assert!(metrics.major_faults_delta() >= 4);
    }
}
