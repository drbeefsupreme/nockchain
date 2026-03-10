use std::io::Write;
use std::path::Path;

use serde::Serialize;

use super::case::{RequestedCase, ResolvedCase};
use super::execute::CompletedRun;
use super::provenance::{HostEnvSnapshot, Provenance};
use super::summary::{RunSummary, Verdict};
use super::{HarnessError, SCHEMA_VERSION};

pub fn write_schema_version(root: &Path) -> Result<(), HarnessError> {
    std::fs::create_dir_all(root)?;
    std::fs::write(
        root.join("schema_version.txt"),
        format!("{SCHEMA_VERSION}\n"),
    )?;
    Ok(())
}

pub fn write_requested_case(root: &Path, requested: &RequestedCase) -> Result<(), HarnessError> {
    write_json(root.join("requested_case.json"), requested)
}

pub fn write_resolved_case(root: &Path, resolved: &ResolvedCase) -> Result<(), HarnessError> {
    write_json(root.join("resolved_case.json"), resolved)
}

pub fn write_provenance(root: &Path, provenance: &Provenance) -> Result<(), HarnessError> {
    write_json(root.join("provenance.json"), provenance)
}

pub fn write_host_env(root: &Path, host_env: &HostEnvSnapshot) -> Result<(), HarnessError> {
    let raw_dir = root.join("raw");
    std::fs::create_dir_all(&raw_dir)?;
    write_json(raw_dir.join("host_env.json"), host_env)
}

pub fn write_summary(root: &Path, summary: &RunSummary) -> Result<(), HarnessError> {
    write_json(root.join("summary.json"), summary)
}

pub fn write_verdict(root: &Path, verdict: &Verdict) -> Result<(), HarnessError> {
    write_json(root.join("verdict.json"), verdict)
}

pub fn write_run_artifacts(run_dir: &Path, run: &CompletedRun) -> Result<(), HarnessError> {
    std::fs::create_dir_all(run_dir)?;
    write_json(run_dir.join("result.json"), &run.record)?;

    if let Some(profile) = &run.profile {
        write_json(run_dir.join("profile.json"), profile)?;
    }

    let mut timings = std::fs::File::create(run_dir.join("block_timings.ndjson"))?;
    for timing in &run.block_timings {
        serde_json::to_writer(&mut timings, timing)?;
        timings.write_all(b"\n")?;
    }

    std::fs::write(run_dir.join("stdout.log"), "")?;
    let stderr = run
        .record
        .error
        .as_deref()
        .map(|error| format!("{error}\n"))
        .unwrap_or_default();
    std::fs::write(run_dir.join("stderr.log"), stderr)?;

    Ok(())
}

fn write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<(), HarnessError> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::tempdir;

    use super::*;
    use crate::speed_of_light::fixture::SolFixtureManifest;
    use crate::speed_of_light::harness::case::{BinaryIdentity, ExecutionConfig};
    use crate::speed_of_light::harness::execute::{BlockTimingRecord, RunRecord};
    use crate::speed_of_light::harness::provenance::BackendRuntimeFacts;
    use crate::speed_of_light::harness::summary::Validity;
    use crate::speed_of_light::types::SolHeight;

    #[test]
    fn harness_artifacts_write_expected_run_files() {
        let tempdir = tempdir().expect("tempdir");
        let run_dir = tempdir.path().join("runs/run-0");
        let completed = CompletedRun {
            record: RunRecord {
                run_id: "run-0".to_string(),
                success: true,
                error: None,
                blocks_poked: 10,
                failed_pokes: 0,
                init_time_secs: 1.0,
                total_replay_time_secs: 2.0,
                throughput_blocks_per_second: 5.0,
                average_block_time_ms: 200.0,
                checkpoint_count: 1,
                checkpoint_total_time_secs: 0.5,
                average_checkpoint_time_secs: 0.5,
                peak_process_rss_bytes: Some(123.0),
                minor_faults_total: Some(10.0),
                major_faults_total: Some(1.0),
            },
            block_timings: vec![BlockTimingRecord {
                height: 42,
                duration_ms: 10.0,
            }],
            profile: None,
            bench_results: None,
        };

        write_run_artifacts(&run_dir, &completed).expect("write artifacts");

        assert!(run_dir.join("result.json").exists());
        assert!(run_dir.join("block_timings.ndjson").exists());
        assert!(run_dir.join("stdout.log").exists());
        assert!(run_dir.join("stderr.log").exists());
    }

    #[test]
    fn harness_artifacts_write_root_files() {
        let tempdir = tempdir().expect("tempdir");
        let root = tempdir.path();
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
        let provenance = Provenance {
            schema_version: SCHEMA_VERSION.to_string(),
            capture_timestamp_ms: 1,
            host: super::super::provenance::HostIdentity {
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
        };
        let host_env = HostEnvSnapshot {
            current_dir: Some(PathBuf::from("/tmp")),
            shell: Some("/bin/zsh".to_string()),
            user: Some("tester".to_string()),
            hostname_env: Some("host".to_string()),
            rust_log: None,
        };
        let summary = RunSummary {
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
        };
        let verdict = Verdict {
            validity: Validity::Valid,
        };

        write_schema_version(root).expect("schema version");
        write_requested_case(root, &requested).expect("requested");
        write_resolved_case(root, &resolved).expect("resolved");
        let resolved_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("resolved_case.json")).expect("read"))
                .expect("resolved json");
        let resolved_object = resolved_json.as_object().expect("resolved case object");
        let execution_config = resolved_object
            .get("execution_config")
            .and_then(serde_json::Value::as_object)
            .expect("execution_config object");
        assert_eq!(
            execution_config.get("checkpoint_recovery_timeout_ms"),
            Some(&serde_json::Value::from(5_000))
        );
        assert_eq!(
            execution_config.get("checkpoint_recovery_tolerance_pct_bps"),
            Some(&serde_json::Value::from(500))
        );
        assert_eq!(
            execution_config.get("gc_drop_threshold_mib"),
            Some(&serde_json::Value::from(64))
        );
        write_provenance(root, &provenance).expect("provenance");
        write_host_env(root, &host_env).expect("host env");
        write_summary(root, &summary).expect("summary");
        write_verdict(root, &verdict).expect("verdict");

        assert!(root.join("schema_version.txt").exists());
        assert!(root.join("requested_case.json").exists());
        assert!(root.join("resolved_case.json").exists());
        assert!(root.join("provenance.json").exists());
        assert!(root.join("raw/host_env.json").exists());
        assert!(root.join("summary.json").exists());
        assert!(root.join("verdict.json").exists());
    }
}
