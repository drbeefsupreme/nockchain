pub mod artifacts;
pub mod case;
pub mod docker;
pub mod execute;
pub mod native;
pub mod provenance;
pub mod summary;

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub use case::{
    resolve_requested_case, BinaryIdentity, ExecutionConfig, ExecutionRequest, RequestedCase,
    ResolvedCase,
};
pub use execute::{
    execute_once, execute_once_with_options, BlockTimingRecord, CompletedRun, ExecuteOptions,
    RunRecord,
};
pub use native::execute_native_trusted_run;
pub use provenance::{
    capture_host_env, capture_native_provenance, BackendRuntimeFacts, GitIdentity, HostEnvSnapshot,
    HostIdentity, Provenance,
};
pub use summary::{
    evaluate_verdict, summarize_runs, RunFailure, RunMetrics, RunSummary, RunSummaryInput,
    Validity, ValueStats, Verdict,
};
use thiserror::Error;

pub const SCHEMA_VERSION: &str = "1";
pub const DEFAULT_THROUGHPUT_CV_THRESHOLD: f64 = 0.10;

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Fixture error: {0}")]
    Fixture(#[from] crate::speed_of_light::fixture::FixtureError),

    #[error("Bench error: {0}")]
    Bench(#[from] crate::speed_of_light::bench::BenchError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    InvalidRequestedCase(String),
}

pub fn is_release_build() -> bool {
    !cfg!(debug_assertions)
}

pub fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

pub fn create_temp_dir(prefix: &str) -> Result<PathBuf, HarnessError> {
    let path = std::env::temp_dir().join(format!(
        "{prefix}-{}-{}",
        std::process::id(),
        unix_timestamp_ms()
    ));
    std::fs::create_dir_all(&path)?;
    Ok(path)
}
