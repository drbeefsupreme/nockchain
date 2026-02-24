use std::path::{Path, PathBuf};

use thiserror::Error;

use super::model::{ComparisonReport, ComparisonResult, ComparisonVerdict};

#[derive(Debug, Error)]
pub enum ComparisonReportError {
    #[error("failed to write report {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to serialize report json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Render a two-tier GitHub-flavored Markdown comparison report.
pub fn render_comparison_markdown(report: &ComparisonReport) -> String {
    let mut lines = Vec::new();

    lines.push("## SOL Benchmark Regression Report".to_string());
    lines.push(String::new());
    lines.push(format!(
        "**Overall verdict:** {}",
        verdict_label(report.overall_verdict)
    ));
    lines.push(String::new());

    // Compact summary table
    lines.push("| Benchmark | Verdict | Effect | Confidence |".to_string());
    lines.push("|-----------|---------|--------|------------|".to_string());

    for result in &report.results {
        lines.push(format!(
            "| {:?} | {} | {:+.1}% | {} |",
            result.metric,
            verdict_icon(result.verdict),
            result.delta_pct,
            if result.verdict == ComparisonVerdict::Inconclusive {
                "\u{2014}".to_string()
            } else {
                format!("{:.0}%", result.confidence * 100.0)
            },
        ));
    }

    lines.push(String::new());

    // Expandable detail section
    lines.push("<details><summary>Per-benchmark statistical detail</summary>".to_string());
    lines.push(String::new());

    for result in &report.results {
        render_detail_section(&mut lines, result);
    }

    lines.push("</details>".to_string());
    lines.push(String::new());

    lines.push("> Advisory only \u{2014} this check does not block merge.".to_string());
    lines.push(format!(
        "> Baseline: {} ({} samples)",
        report.baseline_source, report.baseline_total_samples
    ));

    lines.join("\n") + "\n"
}

fn render_detail_section(lines: &mut Vec<String>, result: &ComparisonResult) {
    lines.push(format!("### {:?}", result.metric));
    lines.push(String::new());
    lines.push(format!("- **Candidate value:** {:.4}", result.candidate_value));
    lines.push(format!("- **Baseline median:** {:.4}", result.baseline_median));
    lines.push(format!("- **Baseline MAD:** {:.4}", result.baseline_mad));
    lines.push(format!("- **Baseline samples:** {}", result.baseline_samples));
    lines.push(format!("- **Delta (absolute):** {:+.4}", result.delta_abs));
    lines.push(format!("- **Delta (percent):** {:+.1}%", result.delta_pct));
    lines.push(format!(
        "- **Confidence:** {:.0}%",
        result.confidence * 100.0
    ));
    lines.push(format!(
        "- **Verdict:** {}",
        verdict_icon(result.verdict)
    ));
    lines.push(format!("- **Reason:** {}", result.reason));
    lines.push(String::new());
}

fn verdict_icon(verdict: ComparisonVerdict) -> &'static str {
    match verdict {
        ComparisonVerdict::Improvement => "\u{1f7e2} improvement",
        ComparisonVerdict::Regression => "\u{1f534} regression",
        ComparisonVerdict::NoSignificantChange => "\u{2705} no change",
        ComparisonVerdict::Inconclusive => "\u{2753} inconclusive",
    }
}

fn verdict_label(verdict: ComparisonVerdict) -> &'static str {
    match verdict {
        ComparisonVerdict::Improvement => "Improvement",
        ComparisonVerdict::Regression => "Regression",
        ComparisonVerdict::NoSignificantChange => "No Significant Change",
        ComparisonVerdict::Inconclusive => "Inconclusive",
    }
}

/// Serialize the report as pretty-printed JSON.
pub fn render_comparison_json(report: &ComparisonReport) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}

/// Write the Markdown comparison report to a file.
pub fn write_comparison_markdown(
    path: &Path,
    report: &ComparisonReport,
) -> Result<(), ComparisonReportError> {
    let content = render_comparison_markdown(report);
    std::fs::write(path, content).map_err(|source| ComparisonReportError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Write the JSON comparison report to a file.
pub fn write_comparison_json(
    path: &Path,
    report: &ComparisonReport,
) -> Result<(), ComparisonReportError> {
    let content = render_comparison_json(report)?;
    std::fs::write(path, content).map_err(|source| ComparisonReportError::Io {
        path: path.to_path_buf(),
        source,
    })
}
