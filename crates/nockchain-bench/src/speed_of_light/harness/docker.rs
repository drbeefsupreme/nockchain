use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use bollard::container::Stats;
use bollard::Docker;
use futures::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use super::artifacts::read_run_artifacts;
use super::case::{BinaryIdentity, ExecutionRequest, RequestedCase, ResolvedCase, WorkDirMode};
use super::orchestrate::{execute_trusted_run, TrustedBackend, TrustedRunResult};
use super::provenance::BackendRuntimeFacts;
use super::{unix_timestamp_ms, HarnessError};

#[derive(Debug, Error)]
pub enum HarnessDockerError {
    #[error("Docker API error: {0}")]
    Api(#[from] bollard::errors::Error),

    #[error("Docker not available: {0}")]
    NotAvailable(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContainerStats {
    pub timestamp_ms: u64,
    pub memory_usage_bytes: u64,
    pub memory_limit_bytes: u64,
    pub memory_percent: f64,
    pub memory_cache_bytes: u64,
    pub memory_rss_bytes: u64,
    pub cpu_percent: f64,
    pub minor_faults: Option<u64>,
    pub major_faults: Option<u64>,
}

impl ContainerStats {
    pub fn from_docker_stats(stats: &Stats, start_time: Instant) -> Self {
        use bollard::container::MemoryStatsStats;

        let memory_usage = stats.memory_stats.usage.unwrap_or(0);
        let memory_limit = stats.memory_stats.limit.unwrap_or(0);
        let (memory_cache, memory_rss) = stats
            .memory_stats
            .stats
            .as_ref()
            .map(|memory_stats| match memory_stats {
                MemoryStatsStats::V1(v1) => (v1.cache, v1.rss),
                MemoryStatsStats::V2(v2) => (v2.file, v2.anon),
            })
            .unwrap_or((0, memory_usage));

        let memory_percent = if memory_limit > 0 {
            (memory_usage as f64 / memory_limit as f64) * 100.0
        } else {
            0.0
        };

        Self {
            timestamp_ms: start_time.elapsed().as_millis() as u64,
            memory_usage_bytes: memory_usage,
            memory_limit_bytes: memory_limit,
            memory_percent,
            memory_cache_bytes: memory_cache,
            memory_rss_bytes: memory_rss,
            cpu_percent: calculate_cpu_percent(stats),
            minor_faults: None,
            major_faults: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DockerRunPlan {
    pub program: String,
    pub args: Vec<String>,
}

impl DockerRunPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn for_run(
        container_name: &str,
        image_tag: &str,
        fixture_path: &str,
        output_root: &str,
        input_root: &str,
        host_work_dir: Option<&str>,
        memory_limit: &str,
        cpuset: Option<&str>,
        cpu_quota: Option<i64>,
        cpu_period: Option<i64>,
        work_dir_mode: WorkDirMode,
        run_id: &str,
    ) -> Self {
        let mut args = vec![
            "run".to_string(),
            "--rm".to_string(),
            "--name".to_string(),
            container_name.to_string(),
            "--memory=".to_string() + memory_limit,
            "-v".to_string(),
            format!("{fixture_path}:/bench/fixture.soltest:ro"),
            "-v".to_string(),
            format!("{output_root}:/bench/output"),
            "-v".to_string(),
            format!("{input_root}:/bench/input:ro"),
        ];

        if let Some(cpuset) = cpuset {
            args.push(format!("--cpuset-cpus={cpuset}"));
        }
        if let Some(cpu_quota) = cpu_quota {
            args.push(format!("--cpu-quota={cpu_quota}"));
        }
        if let Some(cpu_period) = cpu_period {
            args.push(format!("--cpu-period={cpu_period}"));
        }

        match work_dir_mode {
            WorkDirMode::HostBind => {
                if let Some(host_work_dir) = host_work_dir {
                    args.push("-v".to_string());
                    args.push(format!("{host_work_dir}:/bench/work"));
                }
            }
            WorkDirMode::DockerVolume => {
                args.push("--mount".to_string());
                args.push(format!(
                    "type=volume,src={container_name}-work,dst=/bench/work"
                ));
            }
            WorkDirMode::DockerTmpfs => {
                args.push("--tmpfs".to_string());
                args.push("/bench/work".to_string());
            }
        }

        args.extend([
            image_tag.to_string(),
            "sol".to_string(),
            "run-once".to_string(),
            "--resolved-case".to_string(),
            "/bench/input/resolved_case.json".to_string(),
            "--run-dir".to_string(),
            format!("/bench/output/runs/{run_id}"),
            "--run-id".to_string(),
            run_id.to_string(),
        ]);

        Self {
            program: "docker".to_string(),
            args,
        }
    }
}

#[derive(Debug, Clone)]
struct DockerExecutionConfig {
    image_tag: String,
    memory_limit: String,
    cpuset: Option<String>,
    cpu_quota: Option<i64>,
    cpu_period: Option<i64>,
    work_dir_mode: WorkDirMode,
    allow_version_skew: bool,
}

#[derive(Debug, Clone)]
struct DockerBackendState {
    container_name: String,
    container_id: String,
    image_digest: String,
    output_root: PathBuf,
    volume_name: Option<String>,
    host_binary: BinaryIdentity,
}

#[derive(Debug, Clone)]
struct DockerBackend {
    execution: DockerExecutionConfig,
    state: Option<DockerBackendState>,
}

impl DockerBackend {
    fn from_requested(requested: &RequestedCase) -> Result<Self, HarnessError> {
        let ExecutionRequest::Docker {
            image_tag,
            memory_limit,
            cpuset,
            cpu_quota,
            cpu_period,
            work_dir_mode,
            allow_version_skew,
        } = &requested.execution
        else {
            return Err(HarnessError::InvalidRequestedCase(
                "Docker backend requires ExecutionRequest::Docker".to_string(),
            ));
        };

        Ok(Self {
            execution: DockerExecutionConfig {
                image_tag: image_tag.clone(),
                memory_limit: memory_limit.clone(),
                cpuset: cpuset.clone(),
                cpu_quota: *cpu_quota,
                cpu_period: *cpu_period,
                work_dir_mode: work_dir_mode.clone(),
                allow_version_skew: *allow_version_skew,
            },
            state: None,
        })
    }
}

pub async fn execute_docker_trusted_run(
    requested: RequestedCase,
    output_root: &Path,
    allow_debug_benchmark: bool,
) -> Result<TrustedRunResult, HarnessError> {
    let backend = DockerBackend::from_requested(&requested)?;
    execute_trusted_run(backend, requested, output_root, allow_debug_benchmark).await
}

impl TrustedBackend for DockerBackend {
    fn prepare<'a>(
        &'a mut self,
        resolved: &'a ResolvedCase,
        output_root: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<(), HarnessError>> {
        async move {
            connect_docker().await?;

            let output_root = canonicalize_existing_dir(output_root)?;
            let input_root = output_root.join("input");
            std::fs::create_dir_all(&input_root)?;

            let container_resolved = containerize_resolved_case(resolved);
            std::fs::write(
                input_root.join("resolved_case.json"),
                serde_json::to_vec_pretty(&container_resolved)?,
            )?;

            let container_name = format!(
                "nockchain-bench-{}-{}",
                std::process::id(),
                unix_timestamp_ms()
            );
            let volume_name = match self.execution.work_dir_mode {
                WorkDirMode::DockerVolume => Some(format!("{container_name}-work")),
                _ => None,
            };
            let host_work_dir = match self.execution.work_dir_mode {
                WorkDirMode::HostBind => {
                    let path = output_root.join("work");
                    std::fs::create_dir_all(&path)?;
                    Some(path)
                }
                _ => None,
            };

            if let Some(volume_name) = &volume_name {
                let _ = docker_stdout(["volume", "create", volume_name.as_str()])?;
            }

            let create_args = docker_create_args(
                &container_name,
                &self.execution,
                &resolved.absolute_fixture_path,
                &output_root,
                &input_root,
                host_work_dir.as_deref(),
                volume_name.as_deref(),
            );
            let container_id = docker_stdout_vec(create_args)?.trim().to_string();
            let _ = docker_stdout(["start", container_name.as_str()])?;
            let image_digest = resolve_image_digest(&self.execution.image_tag)?;

            self.state = Some(DockerBackendState {
                container_name,
                container_id,
                image_digest,
                output_root,
                volume_name,
                host_binary: resolved.binary.clone(),
            });

            Ok(())
        }
        .boxed()
    }

    fn capture_runtime_facts(&self) -> Result<BackendRuntimeFacts, HarnessError> {
        let state = self.state.as_ref().ok_or_else(|| {
            HarnessError::InvalidRequestedCase("Docker backend not prepared".to_string())
        })?;
        let info = docker_info_json()?;
        let container_binary = inspect_container_binary(&state.container_name)?;

        if !self.execution.allow_version_skew {
            verify_version_skew(&state.host_binary, &container_binary)?;
        }

        Ok(BackendRuntimeFacts::Docker {
            host_binary: state.host_binary.clone(),
            container_binary,
            image_tag: self.execution.image_tag.clone(),
            image_digest: state.image_digest.clone(),
            container_id: state.container_id.clone(),
            docker_engine_version: docker_engine_version(&info),
            docker_context: docker_context()?,
            cgroup_version: info
                .get("CgroupVersion")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            storage_driver: info
                .get("Driver")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            realized_memory_max: read_cgroup_u64(&state.container_name, "/sys/fs/cgroup/memory.max")?,
            realized_memory_current: read_cgroup_u64(
                &state.container_name,
                "/sys/fs/cgroup/memory.current",
            )?,
            realized_cpuset: read_optional_container_file(
                &state.container_name,
                "/sys/fs/cgroup/cpuset.cpus.effective",
            )
            .or_else(|_| {
                read_optional_container_file(&state.container_name, "/sys/fs/cgroup/cpuset.cpus")
            })
            .ok()
            .flatten(),
            realized_cpu_max: read_optional_container_file(
                &state.container_name,
                "/sys/fs/cgroup/cpu.max",
            )
            .ok()
            .flatten(),
        })
    }

    fn execute_run<'a>(
        &'a mut self,
        _resolved: &'a ResolvedCase,
        run_id: &'a str,
        run_dir: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<super::execute::CompletedRun, HarnessError>> {
        async move {
            let state = self.state.as_ref().ok_or_else(|| {
                HarnessError::InvalidRequestedCase("Docker backend not prepared".to_string())
            })?;
            let run_dir = canonicalize_run_dir_parent(run_dir)?;
            let relative_run_dir = run_dir
                .strip_prefix(&state.output_root)
                .unwrap_or_else(|_| Path::new(""));
            let container_run_dir = Path::new("/bench/output").join(relative_run_dir);
            let args = vec![
                "exec".to_string(),
                state.container_name.clone(),
                "nockchain-bench".to_string(),
                "sol".to_string(),
                "run-once".to_string(),
                "--resolved-case".to_string(),
                "/bench/input/resolved_case.json".to_string(),
                "--run-dir".to_string(),
                container_run_dir.to_string_lossy().to_string(),
                "--run-id".to_string(),
                run_id.to_string(),
            ];
            let _ = docker_stdout_vec(args)?;
            read_run_artifacts(&run_dir)
        }
        .boxed()
    }

    fn capture_raw_evidence<'a>(
        &'a self,
        raw_dir: &'a Path,
    ) -> futures::future::BoxFuture<'a, Result<(), HarnessError>> {
        async move {
            let state = self.state.as_ref().ok_or_else(|| {
                HarnessError::InvalidRequestedCase("Docker backend not prepared".to_string())
            })?;
            std::fs::create_dir_all(raw_dir)?;
            std::fs::write(
                raw_dir.join("docker_inspect.json"),
                docker_stdout(["inspect", state.container_name.as_str()])?,
            )?;
            std::fs::write(raw_dir.join("docker_info.json"), docker_info_json_string()?)?;
            std::fs::write(
                raw_dir.join("container_env.json"),
                serde_json::to_vec_pretty(&read_container_env(&state.container_name)?)?,
            )?;
            Ok(())
        }
        .boxed()
    }

    fn cleanup<'a>(&'a mut self) -> futures::future::BoxFuture<'a, Result<(), HarnessError>> {
        async move {
            if let Some(state) = self.state.take() {
                let _ = docker_stdout(["rm", "-f", state.container_name.as_str()]);
                if let Some(volume_name) = state.volume_name.as_deref() {
                    let _ = docker_stdout(["volume", "rm", "-f", volume_name]);
                }
            }
            Ok(())
        }
        .boxed()
    }
}

fn containerize_resolved_case(resolved: &ResolvedCase) -> ResolvedCase {
    let mut container_resolved = resolved.clone();
    container_resolved.absolute_fixture_path = PathBuf::from("/bench/fixture.soltest");
    container_resolved.requested.fixture_path = PathBuf::from("/bench/fixture.soltest");
    container_resolved
}

fn docker_create_args(
    container_name: &str,
    execution: &DockerExecutionConfig,
    fixture_path: &Path,
    output_root: &Path,
    input_root: &Path,
    host_work_dir: Option<&Path>,
    volume_name: Option<&str>,
) -> Vec<String> {
    let mut args = vec![
        "create".to_string(),
        "--name".to_string(),
        container_name.to_string(),
        "--entrypoint".to_string(),
        "sleep".to_string(),
        format!("--memory={}", execution.memory_limit),
        "-v".to_string(),
        format!("{}:/bench/fixture.soltest:ro", fixture_path.display()),
        "-v".to_string(),
        format!("{}:/bench/output", output_root.display()),
        "-v".to_string(),
        format!("{}:/bench/input:ro", input_root.display()),
    ];

    if let Some(cpuset) = &execution.cpuset {
        args.push(format!("--cpuset-cpus={cpuset}"));
    }
    if let Some(cpu_quota) = execution.cpu_quota {
        args.push(format!("--cpu-quota={cpu_quota}"));
    }
    if let Some(cpu_period) = execution.cpu_period {
        args.push(format!("--cpu-period={cpu_period}"));
    }

    match execution.work_dir_mode {
        WorkDirMode::HostBind => {
            if let Some(host_work_dir) = host_work_dir {
                args.push("-v".to_string());
                args.push(format!("{}:/bench/work", host_work_dir.display()));
            }
        }
        WorkDirMode::DockerVolume => {
            if let Some(volume_name) = volume_name {
                args.push("--mount".to_string());
                args.push(format!("type=volume,src={volume_name},dst=/bench/work"));
            }
        }
        WorkDirMode::DockerTmpfs => {
            args.push("--tmpfs".to_string());
            args.push("/bench/work".to_string());
        }
    }

    args.extend([
        execution.image_tag.clone(),
        "infinity".to_string(),
    ]);
    args
}

fn docker_stdout<const N: usize>(args: [&str; N]) -> Result<String, HarnessError> {
    docker_stdout_vec(args.into_iter().map(str::to_string).collect())
}

fn docker_stdout_vec(args: Vec<String>) -> Result<String, HarnessError> {
    let output = Command::new("docker").args(&args).output()?;
    if !output.status.success() {
        return Err(HarnessError::CommandFailure(format!(
            "docker {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn docker_info_json() -> Result<Value, HarnessError> {
    serde_json::from_str(&docker_info_json_string()?).map_err(HarnessError::from)
}

fn docker_info_json_string() -> Result<String, HarnessError> {
    docker_stdout(["info", "--format", "{{json .}}"])
}

fn docker_engine_version(info: &Value) -> String {
    info.get("ServerVersion")
        .or_else(|| info.get("Version"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string()
}

fn docker_context() -> Result<String, HarnessError> {
    docker_stdout(["context", "show"])
}

fn resolve_image_digest(image_tag: &str) -> Result<String, HarnessError> {
    let output = docker_stdout([
        "image",
        "inspect",
        image_tag,
        "--format",
        "{{index .RepoDigests 0}}",
    ])?;
    Ok(output
        .split('@')
        .nth(1)
        .unwrap_or(output.as_str())
        .to_string())
}

fn inspect_container_binary(container_name: &str) -> Result<BinaryIdentity, HarnessError> {
    let version_text =
        docker_stdout(["exec", container_name, "nockchain-bench", "--version"])?;
    let version = version_text
        .split_whitespace()
        .last()
        .unwrap_or(version_text.as_str())
        .to_string();
    Ok(BinaryIdentity {
        version,
        build_profile: "release".to_string(),
        git_commit: None,
    })
}

fn verify_version_skew(
    host_binary: &BinaryIdentity,
    container_binary: &BinaryIdentity,
) -> Result<(), HarnessError> {
    if host_binary.version != container_binary.version {
        return Err(HarnessError::InvalidRequestedCase(format!(
            "host/container version skew detected: host={} container={}",
            host_binary.version, container_binary.version
        )));
    }

    if let (Some(host_commit), Some(container_commit)) =
        (&host_binary.git_commit, &container_binary.git_commit)
    {
        if host_commit != container_commit {
            return Err(HarnessError::InvalidRequestedCase(format!(
                "host/container git commit skew detected: host={} container={}",
                host_commit, container_commit
            )));
        }
    }

    Ok(())
}

fn read_cgroup_u64(container_name: &str, path: &str) -> Result<u64, HarnessError> {
    let value = docker_stdout(["exec", container_name, "cat", path])?;
    parse_cgroup_numeric(&value).ok_or_else(|| {
        HarnessError::CommandFailure(format!("failed to parse cgroup value `{value}` from {path}"))
    })
}

fn read_optional_container_file(
    container_name: &str,
    path: &str,
) -> Result<Option<String>, HarnessError> {
    let output = Command::new("docker")
        .args(["exec", container_name, "cat", path])
        .output()?;
    if !output.status.success() {
        return Err(HarnessError::CommandFailure(format!(
            "docker exec {container_name} cat {path} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        Ok(None)
    } else {
        Ok(Some(text))
    }
}

fn read_container_env(container_name: &str) -> Result<BTreeMap<String, String>, HarnessError> {
    let output = docker_stdout(["exec", container_name, "env"])?;
    Ok(output
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once('=')?;
            Some((key.to_string(), value.to_string()))
        })
        .collect())
}

fn parse_cgroup_numeric(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("max") || value.is_empty() {
        return Some(0);
    }
    value.parse::<u64>().ok()
}

fn canonicalize_existing_dir(path: &Path) -> Result<PathBuf, HarnessError> {
    std::fs::canonicalize(path).map_err(HarnessError::from)
}

fn canonicalize_run_dir_parent(run_dir: &Path) -> Result<PathBuf, HarnessError> {
    std::fs::create_dir_all(run_dir)?;
    std::fs::canonicalize(run_dir).map_err(HarnessError::from)
}

pub async fn connect_docker() -> Result<Docker, HarnessDockerError> {
    let home = std::env::var("HOME").unwrap_or_default();
    let socket_paths = [
        "/var/run/docker.sock".to_string(),
        format!("{home}/.docker/desktop/docker.sock"),
        format!("{home}/.docker/run/docker.sock"),
    ];

    if let Ok(docker) = Docker::connect_with_local_defaults() {
        if docker.ping().await.is_ok() {
            return Ok(docker);
        }
    }

    for socket_path in socket_paths {
        if !Path::new(&socket_path).exists() {
            continue;
        }
        if let Ok(docker) =
            Docker::connect_with_unix(&socket_path, 120, bollard::API_DEFAULT_VERSION)
        {
            if docker.ping().await.is_ok() {
                return Ok(docker);
            }
        }
    }

    Err(HarnessDockerError::NotAvailable(
        "Cannot connect to Docker. Tried: default, /var/run/docker.sock, ~/.docker/desktop/docker.sock, ~/.docker/run/docker.sock"
            .to_string(),
    ))
}

pub fn parse_proc_stat_faults(stat: &str) -> Option<(u64, u64)> {
    let stat = stat.trim();
    if stat.is_empty() {
        return None;
    }

    let stat_after_comm = stat.rfind(')').map(|index| &stat[index + 1..]).unwrap_or(stat);
    let fields: Vec<&str> = stat_after_comm.split_whitespace().collect();
    let minflt = fields.get(7).and_then(|value| value.parse::<u64>().ok())?;
    let majflt = fields.get(9).and_then(|value| value.parse::<u64>().ok())?;
    Some((minflt, majflt))
}

pub fn parse_memory_limit(value: &str) -> i64 {
    let value = value.trim().to_lowercase();

    if let Some(num) = value.strip_suffix('g') {
        num.parse::<i64>().unwrap_or(0) * 1024 * 1024 * 1024
    } else if let Some(num) = value.strip_suffix('m') {
        num.parse::<i64>().unwrap_or(0) * 1024 * 1024
    } else if let Some(num) = value.strip_suffix('k') {
        num.parse::<i64>().unwrap_or(0) * 1024
    } else {
        value.parse::<i64>().unwrap_or(0)
    }
}

pub fn calculate_cpu_percent(stats: &Stats) -> f64 {
    let cpu_delta = stats.cpu_stats.cpu_usage.total_usage as i64
        - stats.precpu_stats.cpu_usage.total_usage as i64;
    let system_delta = stats.cpu_stats.system_cpu_usage.unwrap_or(0) as i64
        - stats.precpu_stats.system_cpu_usage.unwrap_or(0) as i64;
    let num_cpus = stats.cpu_stats.online_cpus.unwrap_or(1) as f64;

    if system_delta > 0 && cpu_delta > 0 {
        (cpu_delta as f64 / system_delta as f64) * num_cpus * 100.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_memory_limit() {
        assert_eq!(parse_memory_limit("16g"), 16 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_limit("512m"), 512 * 1024 * 1024);
        assert_eq!(parse_memory_limit("1024k"), 1024 * 1024);
        assert_eq!(parse_memory_limit("1073741824"), 1073741824);
        assert_eq!(parse_memory_limit("16G"), 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_proc_stat_faults() {
        let stat = "1 (nockchain) S 0 0 0 0 0 0 123 0 4 0 0 0 0 0 0 0 0 0 0 0 0";
        let parsed = parse_proc_stat_faults(stat).expect("expected parse");
        assert_eq!(parsed.0, 123);
        assert_eq!(parsed.1, 4);
        assert!(parse_proc_stat_faults("").is_none());
    }

    #[test]
    fn docker_run_once_command_mounts_fixture_output_and_limits() {
        let plan = DockerRunPlan::for_run(
            "bench-harness-test",
            "nockchain-bench:test",
            "/host/fixture.soltest",
            "/host/output",
            "/host/input",
            Some("/host/work"),
            "2g",
            Some("0-3"),
            Some(200_000),
            Some(100_000),
            WorkDirMode::HostBind,
            "run-0",
        );

        assert_eq!(plan.program, "docker");
        assert!(plan.args.iter().any(|arg| arg == "--memory=2g"));
        assert!(plan.args.iter().any(|arg| arg == "--cpuset-cpus=0-3"));
        assert!(plan.args.iter().any(|arg| arg == "--cpu-quota=200000"));
        assert!(plan.args.iter().any(|arg| arg == "--cpu-period=100000"));
        assert!(plan
            .args
            .iter()
            .any(|arg| arg == "/host/fixture.soltest:/bench/fixture.soltest:ro"));
        assert!(plan
            .args
            .iter()
            .any(|arg| arg == "/host/output:/bench/output"));
        assert!(plan
            .args
            .iter()
            .any(|arg| arg == "/host/input:/bench/input:ro"));
        assert!(plan
            .args
            .iter()
            .any(|arg| arg == "/host/work:/bench/work"));
        assert!(plan.args.ends_with(&[
            "nockchain-bench:test".to_string(),
            "sol".to_string(),
            "run-once".to_string(),
            "--resolved-case".to_string(),
            "/bench/input/resolved_case.json".to_string(),
            "--run-dir".to_string(),
            "/bench/output/runs/run-0".to_string(),
            "--run-id".to_string(),
            "run-0".to_string(),
        ]));
    }
}
