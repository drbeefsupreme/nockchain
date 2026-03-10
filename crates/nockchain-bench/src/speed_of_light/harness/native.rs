use std::path::Path;

use futures::FutureExt;

use super::case::{RequestedCase, ResolvedCase};
use super::execute::execute_once;
use super::orchestrate::{execute_trusted_run, TrustedBackend, TrustedRunResult};
use super::provenance::{BackendRuntimeFacts, Provenance};
use super::summary::{RunSummary, Verdict};
use super::HarnessError;

#[derive(Debug)]
pub struct NativeRunResult {
    pub resolved: ResolvedCase,
    pub provenance: Provenance,
    pub summary: RunSummary,
    pub verdict: Verdict,
}

impl From<TrustedRunResult> for NativeRunResult {
    fn from(value: TrustedRunResult) -> Self {
        Self {
            resolved: value.resolved,
            provenance: value.provenance,
            summary: value.summary,
            verdict: value.verdict,
        }
    }
}

pub async fn execute_native_trusted_run(
    requested: RequestedCase,
    output_root: &Path,
    allow_debug_benchmark: bool,
) -> Result<NativeRunResult, HarnessError> {
    execute_trusted_run(NativeBackend, requested, output_root, allow_debug_benchmark)
        .await
        .map(NativeRunResult::from)
}

struct NativeBackend;

impl TrustedBackend for NativeBackend {
    fn execute_run<'a>(
        &'a mut self,
        resolved: &'a ResolvedCase,
        run_id: &'a str,
        run_dir: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<super::execute::CompletedRun, HarnessError>> {
        execute_once(resolved, run_id, run_dir).boxed()
    }

    fn prepare<'a>(
        &'a mut self,
        _resolved: &'a ResolvedCase,
        _output_root: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<(), HarnessError>> {
        async { Ok(()) }.boxed()
    }

    fn capture_runtime_facts(&self) -> Result<BackendRuntimeFacts, HarnessError> {
        Ok(BackendRuntimeFacts::Native)
    }

    fn capture_raw_evidence<'a>(
        &'a self,
        _raw_dir: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<(), HarnessError>> {
        async { Ok(()) }.boxed()
    }

    fn cleanup<'a>(&'a mut self) -> futures::future::BoxFuture<'a, Result<(), HarnessError>> {
        async { Ok(()) }.boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::tempdir;

    use super::NativeRunResult;
    use crate::speed_of_light::fixture::SolFixtureManifest;
    use crate::speed_of_light::harness::case::{BinaryIdentity, ExecutionConfig, RequestedCase, ResolvedCase};
    use crate::speed_of_light::harness::orchestrate::prepare_output_root;
    use crate::speed_of_light::harness::orchestrate::TrustedRunResult;
    use crate::speed_of_light::harness::provenance::{
        BackendRuntimeFacts, HostIdentity, Provenance,
    };
    use crate::speed_of_light::harness::summary::{RunSummary, Validity, Verdict};
    use crate::speed_of_light::harness::SCHEMA_VERSION;
    use crate::speed_of_light::types::SolHeight;

    #[test]
    fn native_run_rejects_non_empty_output_root() {
        let tempdir = tempdir().expect("tempdir");
        std::fs::write(tempdir.path().join("stale.txt"), "stale").expect("stale file");

        let error = prepare_output_root(tempdir.path()).expect_err("should reject stale output");
        assert!(error
            .to_string()
            .contains("already exists and is not empty"));
    }

    #[test]
    fn native_run_allows_empty_output_root() {
        let tempdir = tempdir().expect("tempdir");
        prepare_output_root(tempdir.path()).expect("empty dir should be allowed");
    }

    #[test]
    fn native_run_result_converts_from_trusted_run_result() {
        let requested = RequestedCase::native(PathBuf::from("fixture.soltest"));
        let resolved = ResolvedCase {
            schema_version: SCHEMA_VERSION.to_string(),
            requested: requested.clone(),
            absolute_fixture_path: PathBuf::from("/tmp/fixture.soltest"),
            fixture_sha256_hex: "abc".to_string(),
            fixture_manifest: SolFixtureManifest {
                format_version: 2,
                source_archive_path: "archive.solarch".to_string(),
                source_archive_event_num: 1,
                derived_checkpoint_height: SolHeight(1),
                derived_checkpoint_event_num: 1,
                archive_start_height: SolHeight(2),
                archive_end_height: SolHeight(3),
                include_mempool: false,
                chunk_size: 8,
                kernel_hash_hex: "kernel".to_string(),
                checkpoint_hash_hex: "checkpoint".to_string(),
                archive_hash_hex: "archive".to_string(),
            },
            execution_config: ExecutionConfig::default(),
            binary: BinaryIdentity {
                version: "0.1.0".to_string(),
                build_profile: "release".to_string(),
                git_commit: None,
            },
        };
        let trusted = TrustedRunResult {
            resolved: resolved.clone(),
            provenance: Provenance {
                schema_version: SCHEMA_VERSION.to_string(),
                capture_timestamp_ms: 1,
                host: HostIdentity {
                    hostname: Some("host".to_string()),
                    os: "linux".to_string(),
                    arch: "x86_64".to_string(),
                    kernel: None,
                    cpu_count: 4,
                    total_memory_bytes: None,
                    cpu_model: None,
                },
                git: None,
                backend: BackendRuntimeFacts::Native,
                binary: resolved.binary.clone(),
                fixture_path: resolved.absolute_fixture_path.clone(),
                fixture_sha256_hex: resolved.fixture_sha256_hex.clone(),
                fixture_manifest: resolved.fixture_manifest.clone(),
            },
            summary: RunSummary {
                measured_runs_requested: 3,
                measured_runs_succeeded: 3,
                failed_runs: Vec::new(),
                throughput_blocks_per_second: None,
                init_time_secs: None,
                total_replay_time_secs: None,
                average_block_time_ms: None,
                failed_pokes: None,
                checkpoint_count: None,
                average_checkpoint_time_secs: None,
                peak_process_rss_bytes: None,
                minor_faults_total: None,
                major_faults_total: None,
            },
            verdict: Verdict {
                validity: Validity::Valid,
            },
        };

        let native = NativeRunResult::from(trusted);

        assert_eq!(native.resolved, resolved);
        assert_eq!(native.provenance.backend, BackendRuntimeFacts::Native);
        assert_eq!(native.verdict.validity, Validity::Valid);
    }
}
