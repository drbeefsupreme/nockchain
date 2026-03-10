use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{is_release_build, HarnessError, SCHEMA_VERSION};
use crate::speed_of_light::fixture::{read_fixture_file, SolFixtureManifest};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionRequest {
    Native,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestedCase {
    pub benchmark: String,
    pub label: Option<String>,
    pub fixture_path: PathBuf,
    pub blocks: u64,
    pub skip_genesis: bool,
    pub enable_checkpointing: bool,
    pub checkpoint_every_blocks: u64,
    pub profile_memory: bool,
    pub profile_interval_ms: u64,
    pub execution: ExecutionRequest,
    pub threads: u32,
    pub warmup_runs: u32,
    pub measured_runs: u32,
    pub cooldown_secs: u64,
}

impl RequestedCase {
    pub fn native(fixture_path: PathBuf) -> Self {
        Self {
            benchmark: "sol-replay".to_string(),
            label: None,
            fixture_path,
            blocks: 0,
            skip_genesis: false,
            enable_checkpointing: true,
            checkpoint_every_blocks: 0,
            profile_memory: false,
            profile_interval_ms: 500,
            execution: ExecutionRequest::Native,
            threads: 1,
            warmup_runs: 1,
            measured_runs: 5,
            cooldown_secs: 10,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryIdentity {
    pub version: String,
    pub build_profile: String,
    pub git_commit: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub checkpoint_recovery_timeout_ms: u64,
    pub checkpoint_recovery_tolerance_pct_bps: u64,
    pub gc_drop_threshold_mib: u64,
    pub page_fault_minor_burst_threshold: u64,
    pub page_fault_major_burst_threshold: u64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            checkpoint_recovery_timeout_ms: 5_000,
            checkpoint_recovery_tolerance_pct_bps: 500,
            gc_drop_threshold_mib: 64,
            page_fault_minor_burst_threshold: 50_000,
            page_fault_major_burst_threshold: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedCase {
    pub schema_version: String,
    pub requested: RequestedCase,
    pub absolute_fixture_path: PathBuf,
    pub fixture_sha256_hex: String,
    pub fixture_manifest: SolFixtureManifest,
    pub execution_config: ExecutionConfig,
    pub binary: BinaryIdentity,
}

pub fn resolve_requested_case(requested: &RequestedCase) -> Result<ResolvedCase, HarnessError> {
    validate_requested_case(requested)?;

    let absolute_fixture_path = canonicalize_path(&requested.fixture_path)?;
    let fixture = read_fixture_file(&absolute_fixture_path)?;
    let fixture_sha256_hex = sha256_hex_for_file(&absolute_fixture_path)?;

    Ok(ResolvedCase {
        schema_version: SCHEMA_VERSION.to_string(),
        requested: requested.clone(),
        absolute_fixture_path,
        fixture_sha256_hex,
        fixture_manifest: fixture.manifest,
        execution_config: ExecutionConfig::default(),
        binary: BinaryIdentity {
            version: env!("CARGO_PKG_VERSION").to_string(),
            build_profile: if is_release_build() {
                "release".to_string()
            } else {
                "debug".to_string()
            },
            git_commit: git_head_commit(),
        },
    })
}

fn validate_requested_case(requested: &RequestedCase) -> Result<(), HarnessError> {
    if requested.measured_runs < 3 {
        return Err(HarnessError::InvalidRequestedCase(
            "trusted runs require at least 3 measured runs".to_string(),
        ));
    }

    if !requested.enable_checkpointing && requested.checkpoint_every_blocks > 0 {
        return Err(HarnessError::InvalidRequestedCase(
            "--checkpoint-every-blocks requires checkpointing to be enabled".to_string(),
        ));
    }

    if requested.threads == 0 {
        return Err(HarnessError::InvalidRequestedCase(
            "--threads must be at least 1".to_string(),
        ));
    }

    Ok(())
}

fn canonicalize_path(path: &Path) -> Result<PathBuf, HarnessError> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn sha256_hex_for_file(path: &Path) -> Result<String, HarnessError> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn git_head_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let commit = String::from_utf8(output.stdout).ok()?;
    let commit = commit.trim();
    if commit.is_empty() {
        None
    } else {
        Some(commit.to_string())
    }
}
