use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::docker::parse_memory_limit;
use super::{is_release_build, HarnessError, SCHEMA_VERSION};
use crate::speed_of_light::fixture::{read_fixture_file, SolFixtureManifest};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkDirMode {
    HostBind,
    DockerVolume,
    DockerTmpfs,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionRequest {
    Native,
    Docker {
        image_tag: String,
        memory_limit: String,
        cpuset: Option<String>,
        cpu_quota: Option<i64>,
        cpu_period: Option<i64>,
        work_dir_mode: WorkDirMode,
        allow_version_skew: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DockerResolvedConfig {
    pub image_tag: String,
    pub requested_memory_limit_bytes: u64,
    pub cpuset: Option<String>,
    pub cpu_quota: Option<i64>,
    pub cpu_period: Option<i64>,
    pub work_dir_mode: WorkDirMode,
    pub allow_version_skew: bool,
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

pub fn current_binary_identity() -> BinaryIdentity {
    BinaryIdentity {
        version: env!("CARGO_PKG_VERSION").to_string(),
        build_profile: if is_release_build() {
            "release".to_string()
        } else {
            "debug".to_string()
        },
        git_commit: option_env!("NOCKCHAIN_BENCH_GIT_COMMIT")
            .map(str::trim)
            .filter(|commit| !commit.is_empty())
            .map(str::to_string),
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docker: Option<DockerResolvedConfig>,
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
        binary: current_binary_identity(),
        docker: resolve_docker_execution(&requested.execution)?,
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

    validate_execution_request(&requested.execution)?;

    Ok(())
}

fn validate_execution_request(execution: &ExecutionRequest) -> Result<(), HarnessError> {
    let ExecutionRequest::Docker {
        image_tag,
        memory_limit,
        cpuset,
        cpu_quota,
        cpu_period,
        ..
    } = execution
    else {
        return Ok(());
    };

    if image_tag.trim().is_empty() {
        return Err(HarnessError::InvalidRequestedCase(
            "Docker execution requires a non-empty image tag".to_string(),
        ));
    }

    if parse_memory_limit(memory_limit) <= 0 {
        return Err(HarnessError::InvalidRequestedCase(
            "Docker execution requires a positive memory limit".to_string(),
        ));
    }

    if cpuset
        .as_ref()
        .is_some_and(|cpuset| cpuset.trim().is_empty())
    {
        return Err(HarnessError::InvalidRequestedCase(
            "Docker execution requires a non-empty cpuset when provided".to_string(),
        ));
    }

    if cpu_quota.is_some_and(|value| value <= 0) {
        return Err(HarnessError::InvalidRequestedCase(
            "Docker execution requires a positive cpu_quota when provided".to_string(),
        ));
    }

    if cpu_period.is_some_and(|value| value <= 0) {
        return Err(HarnessError::InvalidRequestedCase(
            "Docker execution requires a positive cpu_period when provided".to_string(),
        ));
    }

    Ok(())
}

fn resolve_docker_execution(
    execution: &ExecutionRequest,
) -> Result<Option<DockerResolvedConfig>, HarnessError> {
    let ExecutionRequest::Docker {
        image_tag,
        memory_limit,
        cpuset,
        cpu_quota,
        cpu_period,
        work_dir_mode,
        allow_version_skew,
    } = execution
    else {
        return Ok(None);
    };

    Ok(Some(DockerResolvedConfig {
        image_tag: image_tag.clone(),
        requested_memory_limit_bytes: parse_memory_limit(memory_limit) as u64,
        cpuset: cpuset.clone(),
        cpu_quota: *cpu_quota,
        cpu_period: *cpu_period,
        work_dir_mode: work_dir_mode.clone(),
        allow_version_skew: *allow_version_skew,
    }))
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::tempdir;

    use super::{
        current_binary_identity, resolve_requested_case, ExecutionRequest, RequestedCase,
        WorkDirMode,
    };
    use crate::speed_of_light::fixture::{write_fixture_file, SolFixtureFile, SolFixtureManifest};
    use crate::speed_of_light::types::SolHeight;

    #[test]
    fn resolve_requested_case_parses_docker_execution() {
        let tempdir = tempdir().expect("tempdir");
        let fixture_path = tempdir.path().join("fixture.soltest");
        write_fixture_file(
            &fixture_path,
            &SolFixtureFile {
                manifest: SolFixtureManifest {
                    source_archive_path: "archive.solarch".to_string(),
                    source_archive_event_num: Some(1),
                    checkpoint_kind: crate::speed_of_light::SolFixtureCheckpointKind::Derived,
                    checkpoint_height: SolHeight(1),
                    checkpoint_event_num: 1,
                    archive_start_height: SolHeight(2),
                    archive_end_height: SolHeight(3),
                    include_mempool: false,
                    chunk_size: 8,
                    kernel_hash_hex: "kernel".to_string(),
                    checkpoint_hash_hex: "checkpoint".to_string(),
                    archive_hash_hex: "archive".to_string(),
                },
                checkpoint_bytes: vec![1, 2, 3],
                archive_bytes: vec![4, 5, 6],
                kernel_bytes: vec![7, 8, 9],
            },
        )
        .expect("write fixture");

        let requested = RequestedCase {
            execution: ExecutionRequest::Docker {
                image_tag: "nockchain-bench:test".to_string(),
                memory_limit: "2g".to_string(),
                cpuset: Some("0-3".to_string()),
                cpu_quota: Some(200_000),
                cpu_period: Some(100_000),
                work_dir_mode: WorkDirMode::DockerVolume,
                allow_version_skew: true,
            },
            measured_runs: 3,
            cooldown_secs: 0,
            warmup_runs: 1,
            fixture_path: PathBuf::from(&fixture_path),
            ..RequestedCase::native(PathBuf::from(&fixture_path))
        };

        let resolved = resolve_requested_case(&requested).expect("resolve requested case");

        let docker = resolved.docker.expect("docker execution details");
        assert_eq!(docker.image_tag, "nockchain-bench:test");
        assert_eq!(docker.requested_memory_limit_bytes, 2 * 1024 * 1024 * 1024);
        assert_eq!(docker.work_dir_mode, WorkDirMode::DockerVolume);
        assert!(docker.allow_version_skew);
    }

    #[test]
    fn current_binary_identity_uses_compiled_git_commit() {
        let identity = current_binary_identity();
        assert_eq!(identity.version, env!("CARGO_PKG_VERSION"));
        assert!(!identity.build_profile.is_empty());
    }

    #[test]
    fn resolve_requested_case_rejects_invalid_docker_memory_limit() {
        let requested = RequestedCase {
            execution: ExecutionRequest::Docker {
                image_tag: "nockchain-bench:test".to_string(),
                memory_limit: "0".to_string(),
                cpuset: None,
                cpu_quota: None,
                cpu_period: None,
                work_dir_mode: WorkDirMode::HostBind,
                allow_version_skew: false,
            },
            measured_runs: 3,
            cooldown_secs: 0,
            ..RequestedCase::native(PathBuf::from("fixture.soltest"))
        };

        let error = resolve_requested_case(&requested).expect_err("invalid memory limit");
        assert!(error.to_string().contains("memory limit"));
    }
}
