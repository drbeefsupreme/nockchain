use std::path::PathBuf;

use nockchain_bench::speed_of_light::guard::{
    run_comparison, render_comparison_markdown, ComparisonConfig, ComparisonVerdict, MetricDirection,
    CanonicalMetric,
};
use nockchain_bench::speed_of_light::guard::compare_report::render_comparison_json;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("guard")
        .join(name)
}

#[test]
fn test_self_comparison_no_significant_change() {
    let tsv = fixture_path("combined_summary.tsv");
    let config = ComparisonConfig::default();
    let report = run_comparison(&tsv, &tsv, &config).expect("comparison should succeed");

    for result in &report.results {
        assert!(
            result.verdict == ComparisonVerdict::NoSignificantChange
                || result.verdict == ComparisonVerdict::Inconclusive,
            "Self-comparison for {:?} should be NoSignificantChange or Inconclusive, got {:?}: {}",
            result.metric,
            result.verdict,
            result.reason,
        );
    }
}

#[test]
fn test_comparison_produces_json_output() {
    let tsv = fixture_path("combined_summary.tsv");
    let config = ComparisonConfig::default();
    let report = run_comparison(&tsv, &tsv, &config).expect("comparison should succeed");
    let json_str = render_comparison_json(&report).expect("JSON rendering should succeed");

    // Verify it parses back as valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("should be valid JSON");
    assert!(
        parsed.get("overall_verdict").is_some(),
        "JSON should contain overall_verdict field"
    );
    assert!(
        parsed.get("results").is_some(),
        "JSON should contain results field"
    );
}

#[test]
fn test_comparison_produces_markdown_output() {
    let tsv = fixture_path("combined_summary.tsv");
    let config = ComparisonConfig::default();
    let report = run_comparison(&tsv, &tsv, &config).expect("comparison should succeed");
    let md = render_comparison_markdown(&report);

    assert!(
        md.contains("## SOL Benchmark Regression Report"),
        "Markdown should contain report title"
    );
    assert!(
        md.contains("| Benchmark | Verdict | Effect | Confidence |"),
        "Markdown should contain summary table header"
    );
    assert!(
        md.contains("<details>"),
        "Markdown should contain expandable details section"
    );
    assert!(
        md.contains("Advisory only"),
        "Markdown should contain advisory footer"
    );
}

#[test]
fn test_inconclusive_with_insufficient_samples() {
    let tsv = fixture_path("combined_summary.tsv");
    let config = ComparisonConfig {
        min_samples: 100, // fixture has only 3 rows
        ..ComparisonConfig::default()
    };
    let report = run_comparison(&tsv, &tsv, &config).expect("comparison should succeed");

    let has_inconclusive = report
        .results
        .iter()
        .any(|r| r.verdict == ComparisonVerdict::Inconclusive);
    assert!(
        has_inconclusive,
        "With min_samples=100, at least one metric should be Inconclusive"
    );
}

#[test]
fn test_metric_direction_correctness() {
    assert_eq!(
        CanonicalMetric::ThroughputBlocksS.metric_direction(),
        MetricDirection::Higher,
        "ThroughputBlocksS should be Higher (higher is better)"
    );
    assert_eq!(
        CanonicalMetric::PeakRssMib.metric_direction(),
        MetricDirection::Lower,
        "PeakRssMib should be Lower (lower is better)"
    );
    assert_eq!(
        CanonicalMetric::InitTimeS.metric_direction(),
        MetricDirection::Lower,
        "InitTimeS should be Lower (lower is better)"
    );
    assert_eq!(
        CanonicalMetric::AvgPerBlockMs.metric_direction(),
        MetricDirection::Lower,
        "AvgPerBlockMs should be Lower (lower is better)"
    );
}
