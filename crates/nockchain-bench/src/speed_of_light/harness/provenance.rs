use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};

use super::case::{BinaryIdentity, ResolvedCase};
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
        image_tag: String,
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
    pub binary: BinaryIdentity,
    pub fixture_path: PathBuf,
    pub fixture_sha256_hex: String,
    pub fixture_manifest: SolFixtureManifest,
}

pub fn build_provenance(resolved: &ResolvedCase, backend: BackendRuntimeFacts) -> Provenance {
    Provenance {
        schema_version: resolved.schema_version.clone(),
        capture_timestamp_ms: unix_timestamp_ms(),
        host: capture_host_identity(),
        git: capture_git_identity(),
        backend,
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
