//! Run provenance model, validation, and manifest writing.
//!
//! Defines the canonical schema for benchmark run provenance metadata.
//! The manifest captures git commit, branch, resolved config, environment
//! fingerprint, and tool versions so each run is traceable and comparable.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Complete provenance record for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunProvenance {
    /// Schema version for forward compatibility ("1").
    pub schema_version: String,
    /// ISO 8601 timestamp of the run start.
    pub timestamp: String,
    /// Full 40-character git commit SHA.
    pub git_commit: String,
    /// Git branch name.
    pub git_branch: String,
    /// Full resolved benchmark configuration (embedded for inspection).
    pub benchmark_config: serde_json::Value,
    /// SHA-256 hex digest of the raw config file content.
    pub config_sha256: String,
    /// Environment fingerprint.
    pub environment: EnvironmentInfo,
    /// Tool version information.
    pub tool_versions: ToolVersions,
}

/// Hardware and OS environment fingerprint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    /// OS identification (e.g., "Linux 6.17.0-14-generic x86_64").
    pub os: String,
    /// Kernel version string.
    pub kernel: String,
    /// CPU model name.
    pub cpu_model: String,
    /// Number of CPU cores.
    pub cpu_cores: u32,
    /// CPU frequency in MHz (may not be available on all systems).
    pub cpu_frequency_mhz: Option<u64>,
    /// Total RAM in bytes.
    pub ram_bytes: u64,
    /// Active cgroup limits (if applicable).
    pub active_cgroups: Option<String>,
}

/// Tool versions used during the benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolVersions {
    /// rustc version string.
    pub rustc: String,
    /// cargo version string.
    pub cargo: String,
    /// nockchain-bench version.
    pub nockchain_bench: String,
}

/// Validate a provenance manifest strictly.
///
/// All required fields must be present and well-formed. Returns `Ok(())` if
/// all checks pass, or `Err(Vec<String>)` with descriptions of every failure.
pub fn validate_manifest(provenance: &RunProvenance) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Schema version
    if provenance.schema_version != "1" {
        errors.push(format!(
            "schema_version must be \"1\", got \"{}\"",
            provenance.schema_version
        ));
    }

    // Git commit: exactly 40 hex characters
    if provenance.git_commit.len() != 40
        || !provenance.git_commit.chars().all(|c| c.is_ascii_hexdigit())
    {
        errors.push(format!(
            "git_commit must be exactly 40 hex characters, got \"{}\" (len={})",
            provenance.git_commit,
            provenance.git_commit.len()
        ));
    }

    // Git branch: non-empty
    if provenance.git_branch.is_empty() {
        errors.push("git_branch must not be empty".to_string());
    }

    // Timestamp: non-empty and contains "T" (basic ISO 8601 check)
    if provenance.timestamp.is_empty() || !provenance.timestamp.contains('T') {
        errors.push(format!(
            "timestamp must be non-empty ISO 8601 (contain 'T'), got \"{}\"",
            provenance.timestamp
        ));
    }

    // Config SHA-256: exactly 64 hex characters
    if provenance.config_sha256.len() != 64
        || !provenance.config_sha256.chars().all(|c| c.is_ascii_hexdigit())
    {
        errors.push(format!(
            "config_sha256 must be exactly 64 hex characters, got \"{}\" (len={})",
            provenance.config_sha256,
            provenance.config_sha256.len()
        ));
    }

    // Environment: os non-empty
    if provenance.environment.os.is_empty() {
        errors.push("environment.os must not be empty".to_string());
    }

    // Environment: cpu_cores > 0
    if provenance.environment.cpu_cores == 0 {
        errors.push("environment.cpu_cores must be > 0".to_string());
    }

    // Environment: ram_bytes > 0
    if provenance.environment.ram_bytes == 0 {
        errors.push("environment.ram_bytes must be > 0".to_string());
    }

    // Tool versions: rustc non-empty
    if provenance.tool_versions.rustc.is_empty() {
        errors.push("tool_versions.rustc must not be empty".to_string());
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Write a validated provenance manifest as pretty-printed JSON.
///
/// Validates the manifest first -- rejects incomplete artifacts.
/// Returns an error if validation fails or the file cannot be written.
pub fn write_manifest(
    provenance: &RunProvenance,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Validate first -- no incomplete artifacts
    validate_manifest(provenance).map_err(|errs| {
        format!(
            "Manifest validation failed ({} errors): {}",
            errs.len(),
            errs.join("; ")
        )
    })?;

    let json = serde_json::to_string_pretty(provenance)?;
    std::fs::write(path, json)?;
    Ok(())
}
