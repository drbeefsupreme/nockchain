pub mod autopsy;
pub mod baseline;
pub mod contract;
pub mod ingest;
pub mod model;
pub mod report;
pub mod stats;

pub use autopsy::{
    build_basic_hints, detect_profile_anomalies, detect_stack_shifts, parse_folded_symbol_totals,
    rank_metric_failures, SymbolShift,
};
pub use baseline::{
    has_sufficient_baseline, select_baseline_rows, select_baseline_rows_with_fallback,
};
pub use contract::{evaluate_contract, load_contract, ContractError, ContractEvaluation};
pub use ingest::{
    parse_combined_summary_tsv, parse_profile_metrics, parse_runs_manifest, resolve_row_artifacts,
    ArtifactPaths, CombinedSummaryRow, IngestError, ProfileMetrics, ProfileSample, RunsManifest,
    RunsManifestEntry,
};
pub use model::{
    AutopsyHint, BaselineKey, BaselinePolicy, CanonicalMetric, GuardContract, GuardMetricResult,
    GuardReport, GuardVerdict, ReportContext, Severity,
};
pub use report::{render_markdown, write_json, write_markdown, ReportError};
pub use stats::{bootstrap_median_ci, mad, median, ConfidenceInterval};

pub const EXIT_PASS: i32 = 0;
pub const EXIT_REGRESSION: i32 = 2;
pub const EXIT_INSUFFICIENT_BASELINE: i32 = 3;
pub const EXIT_CONFIG_ERROR: i32 = 4;
