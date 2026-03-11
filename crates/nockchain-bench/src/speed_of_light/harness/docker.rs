use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use bollard::container::Stats;
use bollard::Docker;
use futures::{FutureExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::watch;
use tokio::task::JoinError;
use tokio::time::sleep;

use super::artifacts::{read_run_artifacts, write_container_samples};
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
            allow_version_skew: _,
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
            realized_memory_max: read_realized_memory_max(&state.container_name)?,
            realized_memory_current: read_realized_memory_current(&state.container_name)?,
            realized_cpuset: read_optional_container_file(
                &state.container_name, "/sys/fs/cgroup/cpuset.cpus.effective",
            )
            .or_else(|_| {
                read_optional_container_file(&state.container_name, "/sys/fs/cgroup/cpuset.cpus")
            })
            .or_else(|_| {
                read_optional_container_file(
                    &state.container_name,
                    "/sys/fs/cgroup/cpuset/cpuset.cpus",
                )
            })
            .ok()
            .flatten(),
            realized_cpu_max: read_realized_cpu_max(&state.container_name)?,
        })
    }

    fn execute_run<'a>(
        &'a mut self,
        resolved: &'a ResolvedCase,
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
            let should_capture_samples = run_id.starts_with("run-");

            if !should_capture_samples {
                let _ = docker_stdout_vec(args)?;
                return read_run_artifacts(&run_dir);
            }

            let sample_interval_ms = resolved.requested.profile_interval_ms.max(1);
            let (stop_tx, stop_rx) = watch::channel(false);
            let container_name = state.container_name.clone();
            let run_dir_for_sampler = run_dir.clone();
            let sampler = tokio::spawn(async move {
                collect_container_samples_until_stopped(
                    container_name,
                    run_dir_for_sampler,
                    Duration::from_millis(sample_interval_ms),
                    stop_rx,
                )
                .await
            });

            let command = tokio::task::spawn_blocking(move || docker_stdout_vec(args)).await;
            let _ = stop_tx.send(true);
            let samples = collect_sampler_output(sampler.await)?;
            let _ = std::fs::remove_file(run_dir.join(".benchmark.pid"));
            write_container_samples(&run_dir, &samples)?;

            match command {
                Ok(Ok(_)) => read_run_artifacts(&run_dir),
                Ok(Err(error)) => Err(error),
                Err(error) => Err(HarnessError::CommandFailure(format!(
                    "docker run-once task join failed: {error}"
                ))),
            }
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

    args.extend([execution.image_tag.clone(), "infinity".to_string()]);
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
    let output =
        docker_stdout(["image", "inspect", image_tag, "--format", "{{index .RepoDigests 0}}"])?;
    Ok(output
        .split('@')
        .nth(1)
        .unwrap_or(output.as_str())
        .to_string())
}

fn inspect_container_binary(container_name: &str) -> Result<BinaryIdentity, HarnessError> {
    let payload = docker_stdout([
        "exec",
        container_name,
        "nockchain-bench",
        "sol",
        "binary-identity",
    ])?;
    parse_binary_identity_json(&payload)
}

fn parse_binary_identity_json(payload: &str) -> Result<BinaryIdentity, HarnessError> {
    serde_json::from_str(payload).map_err(HarnessError::from)
}

fn collect_sampler_output(
    result: Result<Result<Vec<ContainerStats>, HarnessError>, JoinError>,
) -> Result<Vec<ContainerStats>, HarnessError> {
    match result {
        Ok(Ok(samples)) => Ok(samples),
        Ok(Err(error)) => Err(error),
        Err(error) => Err(HarnessError::CommandFailure(format!(
            "docker stats sampler join failed: {error}"
        ))),
    }
}

async fn collect_container_samples_until_stopped(
    container_name: String,
    run_dir: PathBuf,
    interval: Duration,
    mut stop_rx: watch::Receiver<bool>,
) -> Result<Vec<ContainerStats>, HarnessError> {
    let docker = connect_docker().await?;
    let start_time = Instant::now();
    let mut samples = Vec::new();
    let mut benchmark_pid =
        wait_for_benchmark_pid(&run_dir, Duration::from_millis(250), &mut stop_rx).await;

    loop {
        if *stop_rx.borrow() {
            break;
        }

        if benchmark_pid.is_none() {
            benchmark_pid = read_benchmark_pid(&run_dir);
        }
        samples.push(
            read_container_sample(&docker, &container_name, benchmark_pid, start_time).await?,
        );

        tokio::select! {
            changed = stop_rx.changed() => {
                match changed {
                    Ok(_) if *stop_rx.borrow() => break,
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            _ = sleep(interval) => {}
        }
    }

    Ok(samples)
}

async fn read_container_sample(
    docker: &Docker,
    container_name: &str,
    benchmark_pid: Option<u32>,
    start_time: Instant,
) -> Result<ContainerStats, HarnessError> {
    let mut stats_stream = docker.stats(
        container_name,
        Some(bollard::container::StatsOptions {
            stream: false,
            one_shot: true,
        }),
    );
    let stats = stats_stream
        .next()
        .await
        .ok_or_else(|| {
            HarnessError::CommandFailure(format!(
                "docker stats returned no sample for container {container_name}"
            ))
        })?
        .map_err(HarnessDockerError::from)
        .map_err(HarnessError::from)?;
    let mut sample = ContainerStats::from_docker_stats(&stats, start_time);
    sample.memory_limit_bytes =
        resolve_sample_memory_limit_bytes(sample.memory_limit_bytes, read_realized_memory_max(container_name).ok());
    sample.memory_percent = if sample.memory_limit_bytes > 0 {
        (sample.memory_usage_bytes as f64 / sample.memory_limit_bytes as f64) * 100.0
    } else {
        0.0
    };
    if let Some(proc_stat_path) = benchmark_proc_stat_path(benchmark_pid) {
        if let Ok(proc_stat) =
            docker_stdout(["exec", container_name, "cat", proc_stat_path.as_str()])
        {
            if let Some((minor_faults, major_faults)) = parse_proc_stat_faults(&proc_stat) {
                sample.minor_faults = Some(minor_faults);
                sample.major_faults = Some(major_faults);
            }
        }
    }
    Ok(sample)
}

fn resolve_sample_memory_limit_bytes(stats_limit: u64, realized_limit: Option<u64>) -> u64 {
    realized_limit.filter(|limit| *limit > 0).unwrap_or(stats_limit)
}

fn read_benchmark_pid(run_dir: &Path) -> Option<u32> {
    let pid = std::fs::read_to_string(run_dir.join(".benchmark.pid")).ok()?;
    pid.trim().parse::<u32>().ok()
}

fn benchmark_proc_stat_path(pid: Option<u32>) -> Option<String> {
    pid.map(|pid| format!("/proc/{pid}/stat"))
}

async fn wait_for_benchmark_pid(
    run_dir: &Path,
    max_wait: Duration,
    stop_rx: &mut watch::Receiver<bool>,
) -> Option<u32> {
    let start = Instant::now();
    loop {
        if let Some(pid) = read_benchmark_pid(run_dir) {
            return Some(pid);
        }
        if *stop_rx.borrow() || start.elapsed() >= max_wait {
            return None;
        }
        tokio::select! {
            changed = stop_rx.changed() => {
                match changed {
                    Ok(_) if *stop_rx.borrow() => return None,
                    Ok(_) => {}
                    Err(_) => return None,
                }
            }
            _ = sleep(Duration::from_millis(10)) => {}
        }
    }
}

#[cfg(test)]
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

    if host_binary.git_commit != container_binary.git_commit {
        return Err(HarnessError::InvalidRequestedCase(format!(
            "host/container git commit skew detected: host={:?} container={:?}",
            host_binary.git_commit, container_binary.git_commit
        )));
    }

    Ok(())
}

fn read_realized_memory_max(container_name: &str) -> Result<u64, HarnessError> {
    read_cgroup_u64_any(
        container_name,
        &["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.limit_in_bytes"],
    )
}

fn read_realized_memory_current(container_name: &str) -> Result<u64, HarnessError> {
    read_cgroup_u64_any(
        container_name,
        &[
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory.usage_in_bytes",
        ],
    )
}

fn read_realized_cpu_max(container_name: &str) -> Result<Option<String>, HarnessError> {
    if let Some(cpu_max) = read_optional_container_file(container_name, "/sys/fs/cgroup/cpu.max")
        .ok()
        .flatten()
    {
        return Ok(Some(cpu_max));
    }

    for (quota_path, period_path) in cgroup_v1_cpu_paths() {
        let quota = read_optional_container_file(container_name, quota_path)
            .ok()
            .flatten();
        let period = read_optional_container_file(container_name, period_path)
            .ok()
            .flatten();
        if let Some(cpu_max) = format_cpu_max_from_v1(quota.as_deref(), period.as_deref()) {
            return Ok(Some(cpu_max));
        }
    }
    Ok(None)
}

fn cgroup_v1_cpu_paths() -> [(&'static str, &'static str); 2] {
    [
        (
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        ),
        (
            "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
            "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
        ),
    ]
}

fn format_cpu_max_from_v1(quota: Option<&str>, period: Option<&str>) -> Option<String> {
    let quota = quota?.trim();
    let period = period?.trim();
    if quota.is_empty() || period.is_empty() {
        return None;
    }
    let quota = if quota == "-1" { "max" } else { quota };
    Some(format!("{quota} {period}"))
}

fn read_cgroup_u64_any(container_name: &str, paths: &[&str]) -> Result<u64, HarnessError> {
    let mut last_error = None;
    for path in paths {
        match read_cgroup_u64(container_name, path) {
            Ok(value) => return Ok(value),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        HarnessError::CommandFailure(format!(
            "failed to read any cgroup value from {}",
            paths.join(", ")
        ))
    }))
}

fn read_cgroup_u64(container_name: &str, path: &str) -> Result<u64, HarnessError> {
    let value = docker_stdout(["exec", container_name, "cat", path])?;
    parse_cgroup_numeric(&value).ok_or_else(|| {
        HarnessError::CommandFailure(format!(
            "failed to parse cgroup value `{value}` from {path}"
        ))
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

    let stat_after_comm = stat
        .rfind(')')
        .map(|index| &stat[index + 1..])
        .unwrap_or(stat);
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
    use tempfile::tempdir;
    use tokio::sync::watch;
    use tokio::time::Duration;

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
        assert!(plan.args.iter().any(|arg| arg == "/host/work:/bench/work"));
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

    #[test]
    fn docker_run_artifact_semantics_include_container_samples() {
        let tempdir = tempdir().expect("tempdir");
        let run_dir = tempdir.path().join("runs/run-0");
        let completed = super::super::execute::CompletedRun {
            record: super::super::execute::RunRecord {
                run_id: "run-0".to_string(),
                success: true,
                error: None,
                blocks_poked: 100,
                failed_pokes: 0,
                init_time_secs: 1.0,
                total_replay_time_secs: 2.0,
                throughput_blocks_per_second: 50.0,
                average_block_time_ms: 20.0,
                checkpoint_count: 0,
                checkpoint_total_time_secs: 0.0,
                average_checkpoint_time_secs: 0.0,
                peak_process_rss_bytes: Some(123.0),
                minor_faults_total: Some(4.0),
                major_faults_total: Some(0.0),
            },
            block_timings: vec![super::super::execute::BlockTimingRecord {
                height: 1,
                duration_ms: 20.0,
            }],
            profile: None,
            bench_results: None,
        };
        let samples = vec![ContainerStats {
            timestamp_ms: 25,
            memory_usage_bytes: 1024,
            memory_limit_bytes: 2048,
            memory_percent: 50.0,
            memory_cache_bytes: 128,
            memory_rss_bytes: 768,
            cpu_percent: 90.0,
            minor_faults: Some(9),
            major_faults: Some(1),
        }];

        super::super::artifacts::write_run_artifacts(&run_dir, &completed)
            .expect("write run artifacts");
        super::super::artifacts::write_container_samples(&run_dir, &samples)
            .expect("write container samples");

        assert!(run_dir.join("result.json").exists());
        assert!(run_dir.join("block_timings.ndjson").exists());
        assert!(run_dir.join("container_samples.ndjson").exists());
        assert!(run_dir.join("stdout.log").exists());
        assert!(run_dir.join("stderr.log").exists());
    }

    #[test]
    fn parse_container_binary_identity_json_preserves_commit() {
        let identity = parse_binary_identity_json(
            r#"{
                "version":"0.1.0",
                "build_profile":"release",
                "git_commit":"abc123"
            }"#,
        )
        .expect("parse binary identity");

        assert_eq!(identity.version, "0.1.0");
        assert_eq!(identity.build_profile, "release");
        assert_eq!(identity.git_commit.as_deref(), Some("abc123"));
    }

    #[test]
    fn verify_version_skew_rejects_commit_mismatch() {
        let host = BinaryIdentity {
            version: "0.1.0".to_string(),
            build_profile: "release".to_string(),
            git_commit: Some("host-commit".to_string()),
        };
        let container = BinaryIdentity {
            version: "0.1.0".to_string(),
            build_profile: "release".to_string(),
            git_commit: Some("container-commit".to_string()),
        };

        let error = verify_version_skew(&host, &container).expect_err("commit mismatch");
        assert!(error
            .to_string()
            .contains("host/container git commit skew detected"));
    }

    #[test]
    fn docker_sample_prefers_realized_memory_limit() {
        let limit = resolve_sample_memory_limit_bytes(8_210_616_320, Some(8_589_934_592));
        assert_eq!(limit, 8_589_934_592);
    }

    #[test]
    fn benchmark_proc_stat_path_uses_recorded_pid() {
        assert_eq!(
            benchmark_proc_stat_path(Some(4321)),
            Some("/proc/4321/stat".to_string())
        );
        assert_eq!(benchmark_proc_stat_path(None), None);
    }

    #[test]
    fn cpu_max_falls_back_to_v1_quota_period() {
        assert_eq!(
            format_cpu_max_from_v1(Some("200000"), Some("100000")),
            Some("200000 100000".to_string())
        );
        assert_eq!(
            format_cpu_max_from_v1(Some("-1"), Some("100000")),
            Some("max 100000".to_string())
        );
    }

    #[test]
    fn cgroup_v1_cpu_paths_include_cpuacct_layout() {
        let paths = cgroup_v1_cpu_paths();
        assert_eq!(
            paths[1],
            (
                "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
                "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
            )
        );
    }

    #[tokio::test]
    async fn wait_for_benchmark_pid_observes_late_pid_file() {
        let tempdir = tempdir().expect("tempdir");
        let run_dir = tempdir.path().to_path_buf();
        let (stop_tx, mut stop_rx) = watch::channel(false);
        let writer_dir = run_dir.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(20)).await;
            std::fs::write(writer_dir.join(".benchmark.pid"), "4321\n").expect("pid file");
            let _ = stop_tx.send(false);
        });

        let pid = wait_for_benchmark_pid(&run_dir, Duration::from_millis(200), &mut stop_rx).await;
        assert_eq!(pid, Some(4321));
    }
}
