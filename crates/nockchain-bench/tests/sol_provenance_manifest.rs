//! Contract tests for provenance manifest validation.
//!
//! Verifies that RunProvenance validation accepts valid manifests and
//! rejects manifests with missing or invalid required fields.

use nockchain_bench::speed_of_light::guard::provenance::{
    validate_manifest, write_manifest, EnvironmentInfo, RunProvenance, ToolVersions,
};

fn fixture_path() -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/guard/run-manifest.json");
    path
}

fn load_fixture() -> RunProvenance {
    let content = std::fs::read_to_string(fixture_path()).expect("fixture must exist");
    serde_json::from_str(&content).expect("fixture must be valid JSON")
}

fn valid_provenance() -> RunProvenance {
    RunProvenance {
        schema_version: "1".to_string(),
        timestamp: "2026-02-24T15:30:00Z".to_string(),
        git_commit: "abc1234def5678abc1234def5678abc1234def56".to_string(),
        git_branch: "master".to_string(),
        benchmark_config: serde_json::json!({"profile": "full", "passes": 5}),
        config_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .to_string(),
        environment: EnvironmentInfo {
            os: "Linux 6.17.0-14-generic x86_64".to_string(),
            kernel: "6.17.0-14-generic".to_string(),
            cpu_model: "AMD EPYC 7763".to_string(),
            cpu_cores: 16,
            cpu_frequency_mhz: Some(2450),
            ram_bytes: 34359738368,
            active_cgroups: None,
        },
        tool_versions: ToolVersions {
            rustc: "rustc 1.82.0".to_string(),
            cargo: "cargo 1.82.0".to_string(),
            nockchain_bench: "0.1.0".to_string(),
        },
    }
}

#[test]
fn test_valid_manifest_passes_validation() {
    let manifest = load_fixture();
    assert!(
        validate_manifest(&manifest).is_ok(),
        "Valid fixture manifest should pass validation"
    );
}

#[test]
fn test_missing_git_commit_fails() {
    let mut m = valid_provenance();
    m.git_commit = String::new();
    let err = validate_manifest(&m).unwrap_err();
    assert!(
        err.iter().any(|e| e.contains("git_commit")),
        "Should report git_commit error, got: {:?}",
        err
    );
}

#[test]
fn test_short_git_commit_fails() {
    let mut m = valid_provenance();
    m.git_commit = "abc123".to_string();
    let err = validate_manifest(&m).unwrap_err();
    assert!(
        err.iter().any(|e| e.contains("git_commit")),
        "Should report git_commit error for short hash"
    );
}

#[test]
fn test_missing_branch_fails() {
    let mut m = valid_provenance();
    m.git_branch = String::new();
    let err = validate_manifest(&m).unwrap_err();
    assert!(
        err.iter().any(|e| e.contains("git_branch")),
        "Should report git_branch error"
    );
}

#[test]
fn test_zero_cpu_cores_fails() {
    let mut m = valid_provenance();
    m.environment.cpu_cores = 0;
    let err = validate_manifest(&m).unwrap_err();
    assert!(
        err.iter().any(|e| e.contains("cpu_cores")),
        "Should report cpu_cores error"
    );
}

#[test]
fn test_invalid_config_sha_fails() {
    let mut m = valid_provenance();
    m.config_sha256 = "not-a-hash".to_string();
    let err = validate_manifest(&m).unwrap_err();
    assert!(
        err.iter().any(|e| e.contains("config_sha256")),
        "Should report config_sha256 error"
    );
}

#[test]
fn test_write_manifest_creates_file() {
    let m = valid_provenance();
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("manifest.json");

    write_manifest(&m, &path).expect("should write valid manifest");

    assert!(path.exists(), "Manifest file should exist after write");

    // Read back and verify roundtrip
    let content = std::fs::read_to_string(&path).expect("read back");
    let loaded: RunProvenance = serde_json::from_str(&content).expect("parse JSON");
    assert_eq!(loaded.git_commit, m.git_commit);
    assert_eq!(loaded.schema_version, "1");
}

#[test]
fn test_write_manifest_rejects_invalid() {
    let mut m = valid_provenance();
    m.git_commit = String::new(); // Make invalid

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("manifest.json");

    let result = write_manifest(&m, &path);
    assert!(result.is_err(), "Should reject invalid manifest");
    assert!(!path.exists(), "File should NOT be created for invalid manifest");
}
