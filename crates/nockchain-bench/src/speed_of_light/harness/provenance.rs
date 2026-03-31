use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};

use super::case::{BinaryIdentity, ResolvedCase};
use super::docker_image::DockerImageSource;
use super::{read_trimmed_file, unix_timestamp_ms};
use crate::speed_of_light::fixture::SolFixtureManifest;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostIdentity {
    pub hostname: Option<String>,
    pub os: String,
    pub arch: String,
    pub kernel: Option<String>,
    pub cpu_count: usize,
    pub total_memory_bytes: Option<u64>,
    pub cpu_model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitIdentity {
    pub commit: Option<String>,
    pub branch: Option<String>,
    pub commit_date: Option<String>,
    pub dirty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostEnvSnapshot {
    pub current_dir: Option<PathBuf>,
    pub shell: Option<String>,
    pub user: Option<String>,
    pub hostname_env: Option<String>,
    pub rust_log: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendRuntimeFacts {
    Native,
    Docker {
        host_binary: BinaryIdentity,
        container_binary: BinaryIdentity,
        image_source: DockerImageSource,
        requested_image_ref: String,
        resolved_image_ref: String,
        image_digest: String,
        container_id: String,
        docker_engine_version: String,
        docker_context: String,
        cgroup_version: String,
        storage_driver: String,
        realized_memory_max: u64,
        realized_memory_current: u64,
        realized_cpuset: Option<String>,
        realized_cpu_max: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub schema_version: String,
    pub capture_timestamp_ms: u128,
    pub host: HostIdentity,
    pub git: Option<GitIdentity>,
    pub backend: BackendRuntimeFacts,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_flavor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boot_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boot_event_num: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pma_work_dir_mode: Option<String>,
    pub binary: BinaryIdentity,
    pub fixture_path: PathBuf,
    pub fixture_sha256_hex: String,
    pub fixture_manifest: SolFixtureManifest,
}

pub fn build_provenance(resolved: &ResolvedCase, backend: BackendRuntimeFacts) -> Provenance {
    let (runtime_flavor, boot_source, boot_event_num, pma_work_dir_mode) =
        phase2_pma_provenance_fields(resolved, &backend);
    Provenance {
        schema_version: resolved.schema_version.clone(),
        capture_timestamp_ms: unix_timestamp_ms(),
        host: capture_host_identity(),
        git: capture_git_identity(),
        backend,
        runtime_flavor,
        boot_source,
        boot_event_num,
        pma_work_dir_mode,
        binary: resolved.binary.clone(),
        fixture_path: resolved.absolute_fixture_path.clone(),
        fixture_sha256_hex: resolved.fixture_sha256_hex.clone(),
        fixture_manifest: resolved.fixture_manifest.clone(),
    }
}

pub fn capture_native_provenance(resolved: &ResolvedCase) -> Provenance {
    build_provenance(resolved, BackendRuntimeFacts::Native)
}

pub fn capture_host_env() -> HostEnvSnapshot {
    HostEnvSnapshot {
        current_dir: std::env::current_dir().ok(),
        shell: std::env::var("SHELL").ok(),
        user: std::env::var("USER").ok(),
        hostname_env: std::env::var("HOSTNAME").ok(),
        rust_log: std::env::var("RUST_LOG").ok(),
    }
}

#[cfg(feature = "pma-runtime-compat")]
fn phase2_pma_provenance_fields(
    resolved: &ResolvedCase,
    backend: &BackendRuntimeFacts,
) -> (Option<String>, Option<String>, Option<u64>, Option<String>) {
    if matches!(backend, BackendRuntimeFacts::Native) {
        (
            Some("pma".to_string()),
            Some("checkpoint".to_string()),
            Some(resolved.fixture_manifest.checkpoint_event_num),
            None,
        )
    } else {
        (None, None, None, None)
    }
}

#[cfg(not(feature = "pma-runtime-compat"))]
fn phase2_pma_provenance_fields(
    _resolved: &ResolvedCase,
    _backend: &BackendRuntimeFacts,
) -> (Option<String>, Option<String>, Option<u64>, Option<String>) {
    (None, None, None, None)
}

fn capture_host_identity() -> HostIdentity {
    HostIdentity {
        hostname: read_trimmed_file("/proc/sys/kernel/hostname")
            .or_else(|| std::env::var("HOSTNAME").ok()),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        kernel: read_trimmed_file("/proc/sys/kernel/osrelease"),
        cpu_count: std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1),
        total_memory_bytes: read_total_memory_bytes(),
        cpu_model: read_cpu_model(),
    }
}

fn capture_git_identity() -> Option<GitIdentity> {
    let commit = git_stdout(["rev-parse", "HEAD"]);
    let branch = git_stdout(["rev-parse", "--abbrev-ref", "HEAD"]);
    let commit_date = git_stdout(["log", "-1", "--format=%cI", "HEAD"]);
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .map(|output| !String::from_utf8_lossy(&output.stdout).trim().is_empty())
        .unwrap_or(false);

    if commit.is_none() && branch.is_none() {
        None
    } else {
        Some(GitIdentity {
            commit,
            branch,
            commit_date,
            dirty,
        })
    }
}

fn git_stdout<const N: usize>(args: [&str; N]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn read_total_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

fn read_cpu_model() -> Option<String> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in cpuinfo.lines() {
        if let Some(model) = line
            .split(':')
            .nth(1)
            .filter(|_| line.starts_with("model name"))
        {
            return Some(model.trim().to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{BackendRuntimeFacts, build_provenance};
    use crate::speed_of_light::fixture::SolFixtureManifest;
    use crate::speed_of_light::harness::SCHEMA_VERSION;
    use crate::speed_of_light::harness::case::{
        BinaryIdentity, ExecutionConfig, RequestedCase, ResolvedCase,
    };
    use crate::speed_of_light::types::SolHeight;

    fn test_resolved_case() -> ResolvedCase {
        let requested = RequestedCase::native(PathBuf::from("fixture.soltest"));
        ResolvedCase {
            schema_version: SCHEMA_VERSION.to_string(),
            requested,
            absolute_fixture_path: PathBuf::from("/tmp/fixture.soltest"),
            fixture_sha256_hex: "fixture-sha".to_string(),
            fixture_manifest: SolFixtureManifest {
                source_archive_path: "archive.solarch".to_string(),
                source_archive_event_num: Some(12_000),
                checkpoint_kind: crate::speed_of_light::SolFixtureCheckpointKind::Derived,
                checkpoint_height: SolHeight(11_999),
                checkpoint_event_num: 12_000,
                archive_start_height: SolHeight(12_000),
                archive_end_height: SolHeight(12_099),
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
            docker: None,
        }
    }

    #[test]
    fn build_provenance_omits_optional_pma_fields_without_feature() {
        let provenance = build_provenance(&test_resolved_case(), BackendRuntimeFacts::Native);
        assert_eq!(provenance.backend, BackendRuntimeFacts::Native);
        assert_eq!(provenance.runtime_flavor, None);
        assert_eq!(provenance.boot_source, None);
        assert_eq!(provenance.boot_event_num, None);
        assert_eq!(provenance.pma_work_dir_mode, None);

        let json = serde_json::to_value(&provenance).expect("serialize provenance");
        let object = json.as_object().expect("provenance object");
        assert!(!object.contains_key("runtime_flavor"));
        assert!(!object.contains_key("boot_source"));
        assert!(!object.contains_key("boot_event_num"));
        assert!(!object.contains_key("pma_work_dir_mode"));
    }

    #[cfg(feature = "pma-runtime-compat")]
    #[test]
    fn build_provenance_populates_pma_replay_fields_under_feature() {
        let resolved = test_resolved_case();
        let provenance = build_provenance(&resolved, BackendRuntimeFacts::Native);
        assert_eq!(provenance.backend, BackendRuntimeFacts::Native);
        assert_eq!(provenance.runtime_flavor.as_deref(), Some("pma"));
        assert_eq!(provenance.boot_source.as_deref(), Some("checkpoint"));
        assert_eq!(
            provenance.boot_event_num,
            Some(resolved.fixture_manifest.checkpoint_event_num)
        );
        assert_eq!(provenance.pma_work_dir_mode, None);

        let json = serde_json::to_value(&provenance).expect("serialize provenance");
        assert_eq!(json.get("backend"), Some(&serde_json::json!("Native")));
        assert_eq!(json.get("runtime_flavor"), Some(&serde_json::json!("pma")));
        assert_eq!(json.get("boot_source"), Some(&serde_json::json!("checkpoint")));
        assert_eq!(
            json.get("boot_event_num"),
            Some(&serde_json::json!(resolved.fixture_manifest.checkpoint_event_num))
        );
        assert!(json.get("pma_work_dir_mode").is_none());
    }
}
