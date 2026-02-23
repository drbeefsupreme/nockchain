use std::path::{Path, PathBuf};

use thiserror::Error;

use super::model::{GuardMetricResult, GuardReport, GuardVerdict};

#[derive(Debug, Error)]
pub enum ReportError {
    #[error("failed to write report {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to serialize report json: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn write_json(path: &Path, report: &GuardReport) -> Result<(), ReportError> {
    let content = serde_json::to_string_pretty(report)?;
    std::fs::write(path, content).map_err(|source| ReportError::Io {
        path: path.to_path_buf(),
        source,
    })
}

pub fn write_markdown(path: &Path, report: &GuardReport) -> Result<(), ReportError> {
    let content = render_markdown(report);
    std::fs::write(path, content).map_err(|source| ReportError::Io {
        path: path.to_path_buf(),
        source,
    })
}

pub fn render_markdown(report: &GuardReport) -> String {
    let mut lines = Vec::new();
    lines.push("# SOL Guard Report".to_string());
    lines.push(String::new());
    lines.push(format!(
        "- Run: `{}` (`{}` / `{}` / `{}`)",
        report.context.run_id, report.context.env, report.context.branch, report.context.fixture
    ));
    lines.push(format!("- Verdict: `{}`", verdict_label(report.verdict)));
    lines.push(format!("- Baseline samples: `{}`", report.baseline_samples));
    lines.push(String::new());
    lines.push("## Metrics".to_string());
    lines.push(String::new());
    lines.push(
        "| metric | candidate | baseline median | delta % | severity | passed | reason |"
            .to_string(),
    );
    lines.push("|---|---:|---:|---:|---|---|---|".to_string());
    for metric in &report.metrics {
        lines.push(render_metric_row(metric));
    }
    lines.push(String::new());
    lines.push("## Autopsy".to_string());
    lines.push(String::new());
    if report.autopsy.is_empty() {
        lines.push("- none".to_string());
    } else {
        for hint in &report.autopsy {
            lines.push(format!("- {}", hint.summary));
        }
    }
    lines.join("\n") + "\n"
}

fn verdict_label(verdict: GuardVerdict) -> &'static str {
    match verdict {
        GuardVerdict::Pass => "pass",
        GuardVerdict::Warn => "warn",
        GuardVerdict::Fail => "fail",
        GuardVerdict::InsufficientBaseline => "insufficient_baseline",
    }
}

fn render_metric_row(metric: &GuardMetricResult) -> String {
    format!(
        "| {:?} | {:.4} | {} | {} | {:?} | {} | {} |",
        metric.metric,
        metric.candidate_value,
        metric
            .baseline_median
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "-".to_string()),
        metric
            .delta_pct
            .map(|v| format!("{:.2}", v))
            .unwrap_or_else(|| "-".to_string()),
        metric.severity,
        if metric.passed { "yes" } else { "no" },
        metric.reason
    )
}
