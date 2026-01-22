//! Docker-based runner for Nockchain
//!
//! Uses bollard to manage Docker containers and collect stats.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use bollard::container::{
    Config, CreateContainerOptions, LogsOptions, RemoveContainerOptions, StartContainerOptions,
    Stats, StatsOptions, StopContainerOptions,
};
use bollard::models::{HostConfig, Mount, MountTypeEnum};
use bollard::Docker;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DockerError {
    #[error("Docker API error: {0}")]
    Api(#[from] bollard::errors::Error),

    #[error("Container {name} not found")]
    ContainerNotFound { name: String },

    #[error("Container {name} failed to start: {reason}")]
    StartFailed { name: String, reason: String },

    #[error("Timeout waiting for container")]
    Timeout,

    #[error("Docker not available: {0}")]
    NotAvailable(String),
}

/// Nockchain run mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NockchainMode {
    /// Standard checkpointing mode
    Checkpoint {
        /// Checkpoint save interval in seconds
        save_interval_secs: u64,
    },
    /// PMA persistence mode (no checkpoints)
    PmaPersist,
}

impl Default for NockchainMode {
    fn default() -> Self {
        Self::Checkpoint {
            save_interval_secs: 120,
        }
    }
}

/// Configuration for the Docker runner
#[derive(Debug, Clone)]
pub struct DockerRunnerConfig {
    /// Docker image to use
    pub image: String,

    /// Container name prefix
    pub container_name: String,

    /// Host directory to mount as data directory
    pub data_dir: String,

    /// Memory limit for the container (e.g., "16g")
    pub memory_limit: Option<String>,

    /// Nockchain run mode
    pub mode: NockchainMode,

    /// Enable mining
    pub mine: bool,

    /// Mining public key hash (required if mining)
    pub mining_pkh: Option<String>,

    /// Use fakenet (local testing)
    pub fakenet: bool,

    /// Number of mining threads (0 = auto)
    pub num_threads: u32,

    /// Enable fast sync
    pub fast_sync: bool,

    /// Additional environment variables
    pub env_vars: HashMap<String, String>,

    /// UDP port for P2P networking (default: 30000)
    pub bind_port: u16,
}

impl Default for DockerRunnerConfig {
    fn default() -> Self {
        Self {
            image: "nockchain-local:latest".to_string(),
            container_name: "nockchain-bench".to_string(),
            data_dir: "/tmp/nockchain-bench-data".to_string(),
            memory_limit: Some("16g".to_string()),
            mode: NockchainMode::default(),
            mine: false,
            mining_pkh: None,
            fakenet: true,
            num_threads: 1,
            fast_sync: true,
            env_vars: HashMap::new(),
            bind_port: 30000,
        }
    }
}

impl DockerRunnerConfig {
    /// Create a config for checkpoint mode benchmarking
    pub fn checkpoint_mode(save_interval_secs: u64) -> Self {
        Self {
            mode: NockchainMode::Checkpoint { save_interval_secs },
            ..Default::default()
        }
    }

    /// Create a config for PMA persist mode benchmarking
    pub fn pma_persist_mode() -> Self {
        Self {
            mode: NockchainMode::PmaPersist,
            ..Default::default()
        }
    }

    /// Set the data directory
    pub fn with_data_dir(mut self, data_dir: impl Into<String>) -> Self {
        self.data_dir = data_dir.into();
        self
    }

    /// Set the container name
    pub fn with_container_name(mut self, name: impl Into<String>) -> Self {
        self.container_name = name.into();
        self
    }

    /// Enable mining with fakenet
    pub fn with_fakenet_mining(mut self) -> Self {
        self.mine = true;
        self.fakenet = true;
        // Use a test PKH for fakenet
        self.mining_pkh = Some("11111111111111111111111111111111111".to_string());
        self
    }

    /// Set memory limit
    pub fn with_memory_limit(mut self, limit: impl Into<String>) -> Self {
        self.memory_limit = Some(limit.into());
        self
    }

    /// Build the command line arguments for nockchain
    fn build_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        // Mode-specific args
        match &self.mode {
            NockchainMode::Checkpoint { save_interval_secs } => {
                args.push("--save-interval".to_string());
                args.push(save_interval_secs.to_string());
            }
            NockchainMode::PmaPersist => {
                args.push("--pma-persist".to_string());
            }
        }

        // Common args
        args.push("--data-dir".to_string());
        args.push("/data/.data.nockchain".to_string());

        args.push("--num-threads".to_string());
        args.push(self.num_threads.to_string());

        if self.fast_sync {
            args.push("--fast-sync".to_string());
        }

        if self.fakenet {
            args.push("--fakenet".to_string());
        }

        if self.mine {
            args.push("--mine".to_string());
            if let Some(pkh) = &self.mining_pkh {
                args.push("--mining-pkh".to_string());
                args.push(pkh.clone());
            }
        }

        args.push("--bind".to_string());
        args.push(format!("/ip4/0.0.0.0/udp/{}/quic-v1", self.bind_port));

        // First run needs --new
        args.push("--new".to_string());

        args
    }

    /// Build environment variables
    fn build_env(&self) -> Vec<String> {
        // Enable trace logging for save operations to capture checkpoint events
        let mut env = vec![
            "RUST_LOG=info,nockchain=info,nockapp::nockapp::save=debug,nockapp::nockapp::mod=debug,nockapp::nockapp::actors::save=debug".to_string(),
            "MINIMAL_LOG_FORMAT=true".to_string(),
            "TRACY_NO_INVARIANT_CHECK=1".to_string(),
        ];

        if matches!(self.mode, NockchainMode::PmaPersist) {
            env.push("NOCK_PMA_PERSIST=1".to_string());
        }

        for (key, value) in &self.env_vars {
            env.push(format!("{}={}", key, value));
        }

        env
    }
}

/// Memory statistics from a container
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContainerStats {
    /// Timestamp when this sample was taken (milliseconds since start)
    pub timestamp_ms: u64,

    /// Memory usage in bytes
    pub memory_usage_bytes: u64,

    /// Memory limit in bytes
    pub memory_limit_bytes: u64,

    /// Memory usage percentage
    pub memory_percent: f64,

    /// Cache memory in bytes (can be reclaimed)
    pub memory_cache_bytes: u64,

    /// RSS (Resident Set Size) in bytes
    pub memory_rss_bytes: u64,

    /// CPU usage percentage (0-100 * num_cpus)
    pub cpu_percent: f64,
}

impl ContainerStats {
    /// Parse stats from Docker stats response
    fn from_docker_stats(stats: &Stats, start_time: Instant) -> Self {
        use bollard::container::MemoryStatsStats;

        let memory_usage = stats
            .memory_stats
            .usage
            .unwrap_or(0);

        let memory_limit = stats
            .memory_stats
            .limit
            .unwrap_or(0);

        // Extract cache and RSS from the stats enum (V1 or V2 cgroups)
        let (memory_cache, memory_rss) = stats
            .memory_stats
            .stats
            .as_ref()
            .map(|s| match s {
                MemoryStatsStats::V1(v1) => (v1.cache, v1.rss),
                MemoryStatsStats::V2(v2) => (v2.file, v2.anon),
            })
            .unwrap_or((0, memory_usage)); // Fall back to usage if stats not available

        let memory_percent = if memory_limit > 0 {
            (memory_usage as f64 / memory_limit as f64) * 100.0
        } else {
            0.0
        };

        // CPU percentage calculation
        let cpu_percent = calculate_cpu_percent(stats);

        Self {
            timestamp_ms: start_time.elapsed().as_millis() as u64,
            memory_usage_bytes: memory_usage,
            memory_limit_bytes: memory_limit,
            memory_percent,
            memory_cache_bytes: memory_cache,
            memory_rss_bytes: memory_rss,
            cpu_percent,
        }
    }
}

/// Calculate CPU percentage from Docker stats
fn calculate_cpu_percent(stats: &Stats) -> f64 {
    let cpu_delta = stats.cpu_stats.cpu_usage.total_usage as i64
        - stats.precpu_stats.cpu_usage.total_usage as i64;

    let system_delta = stats.cpu_stats.system_cpu_usage.unwrap_or(0) as i64
        - stats.precpu_stats.system_cpu_usage.unwrap_or(0) as i64;

    let num_cpus = stats
        .cpu_stats
        .online_cpus
        .unwrap_or(1) as f64;

    if system_delta > 0 && cpu_delta > 0 {
        (cpu_delta as f64 / system_delta as f64) * num_cpus * 100.0
    } else {
        0.0
    }
}

/// Docker-based runner for Nockchain
pub struct DockerRunner {
    docker: Docker,
    config: DockerRunnerConfig,
    container_id: Option<String>,
    start_time: Option<Instant>,
}

impl DockerRunner {
    /// Create a new Docker runner
    pub async fn new(config: DockerRunnerConfig) -> Result<Self, DockerError> {
        let docker = Self::connect_docker().await?;

        Ok(Self {
            docker,
            config,
            container_id: None,
            start_time: None,
        })
    }

    /// Connect to Docker, trying multiple socket locations
    async fn connect_docker() -> Result<Docker, DockerError> {
        // Try standard locations first
        let socket_paths = [
            "/var/run/docker.sock",
            &format!("{}/.docker/desktop/docker.sock", std::env::var("HOME").unwrap_or_default()),
            &format!("{}/.docker/run/docker.sock", std::env::var("HOME").unwrap_or_default()),
        ];

        // First try the default (uses DOCKER_HOST env var if set)
        if let Ok(docker) = Docker::connect_with_local_defaults() {
            if docker.ping().await.is_ok() {
                return Ok(docker);
            }
        }

        // Try each socket path
        for path in &socket_paths {
            if std::path::Path::new(path).exists() {
                if let Ok(docker) = Docker::connect_with_unix(path, 120, bollard::API_DEFAULT_VERSION) {
                    if docker.ping().await.is_ok() {
                        return Ok(docker);
                    }
                }
            }
        }

        Err(DockerError::NotAvailable(
            "Cannot connect to Docker. Tried: default, /var/run/docker.sock, ~/.docker/desktop/docker.sock".to_string()
        ))
    }

    /// Check if Docker is available
    pub async fn is_available() -> bool {
        Self::connect_docker().await.is_ok()
    }

    /// Start the container
    pub async fn start(&mut self) -> Result<(), DockerError> {
        // Ensure data directory exists
        if let Err(e) = std::fs::create_dir_all(&self.config.data_dir) {
            return Err(DockerError::StartFailed {
                name: self.config.container_name.clone(),
                reason: format!("Failed to create data directory '{}': {}", self.config.data_dir, e),
            });
        }

        // Verify the directory actually exists
        if !std::path::Path::new(&self.config.data_dir).exists() {
            return Err(DockerError::StartFailed {
                name: self.config.container_name.clone(),
                reason: format!("Data directory '{}' does not exist after creation attempt", self.config.data_dir),
            });
        }

        // Remove existing container if any
        self.remove_if_exists().await?;

        // Parse memory limit
        let memory_bytes = self.config.memory_limit.as_ref().map(|s| parse_memory_limit(s));

        // Create host config with mount
        let host_config = HostConfig {
            memory: memory_bytes,
            network_mode: Some("host".to_string()),
            mounts: Some(vec![Mount {
                target: Some("/data/.data.nockchain".to_string()),
                source: Some(self.config.data_dir.clone()),
                typ: Some(MountTypeEnum::BIND),
                ..Default::default()
            }]),
            ..Default::default()
        };

        // Create container config
        let container_config = Config {
            image: Some(self.config.image.clone()),
            cmd: Some(self.config.build_args()),
            env: Some(self.config.build_env()),
            host_config: Some(host_config),
            ..Default::default()
        };

        // Create container
        let create_options = CreateContainerOptions {
            name: &self.config.container_name,
            platform: None,
        };

        let response = self
            .docker
            .create_container(Some(create_options), container_config)
            .await?;

        self.container_id = Some(response.id.clone());

        // Start container
        self.docker
            .start_container(&response.id, None::<StartContainerOptions<String>>)
            .await?;

        self.start_time = Some(Instant::now());

        Ok(())
    }

    /// Stop the container
    pub async fn stop(&mut self) -> Result<(), DockerError> {
        if let Some(ref id) = self.container_id {
            self.docker
                .stop_container(id, Some(StopContainerOptions { t: 10 }))
                .await?;
        }
        Ok(())
    }

    /// Remove the container
    pub async fn remove(&mut self) -> Result<(), DockerError> {
        if let Some(ref id) = self.container_id {
            self.docker
                .remove_container(
                    id,
                    Some(RemoveContainerOptions {
                        force: true,
                        ..Default::default()
                    }),
                )
                .await?;
            self.container_id = None;
        }
        Ok(())
    }

    /// Remove container if it exists
    async fn remove_if_exists(&self) -> Result<(), DockerError> {
        // Try to remove by name
        let _ = self
            .docker
            .remove_container(
                &self.config.container_name,
                Some(RemoveContainerOptions {
                    force: true,
                    ..Default::default()
                }),
            )
            .await;
        Ok(())
    }

    /// Check if the container is running
    pub async fn is_running(&self) -> Result<bool, DockerError> {
        if let Some(ref id) = self.container_id {
            let info = self.docker.inspect_container(id, None).await?;
            Ok(info
                .state
                .and_then(|s| s.running)
                .unwrap_or(false))
        } else {
            Ok(false)
        }
    }

    /// Get current container stats
    pub async fn get_stats(&self) -> Result<ContainerStats, DockerError> {
        let id = self
            .container_id
            .as_ref()
            .ok_or_else(|| DockerError::ContainerNotFound {
                name: self.config.container_name.clone(),
            })?;

        let start_time = self.start_time.unwrap_or_else(Instant::now);

        let mut stats_stream = self.docker.stats(
            id,
            Some(StatsOptions {
                stream: false,
                one_shot: true,
            }),
        );

        if let Some(stats_result) = stats_stream.next().await {
            let stats = stats_result?;
            Ok(ContainerStats::from_docker_stats(&stats, start_time))
        } else {
            Err(DockerError::ContainerNotFound {
                name: self.config.container_name.clone(),
            })
        }
    }

    /// Collect stats for a duration at the specified interval
    pub async fn collect_stats(
        &self,
        duration: Duration,
        interval: Duration,
    ) -> Result<Vec<ContainerStats>, DockerError> {
        let id = self
            .container_id
            .as_ref()
            .ok_or_else(|| DockerError::ContainerNotFound {
                name: self.config.container_name.clone(),
            })?;

        let start_time = self.start_time.unwrap_or_else(Instant::now);
        let collection_start = Instant::now();
        let mut samples = Vec::new();

        while collection_start.elapsed() < duration {
            // Check if still running
            if !self.is_running().await? {
                break;
            }

            // Get stats
            let mut stats_stream = self.docker.stats(
                id,
                Some(StatsOptions {
                    stream: false,
                    one_shot: true,
                }),
            );

            if let Some(stats_result) = stats_stream.next().await {
                let stats = stats_result?;
                samples.push(ContainerStats::from_docker_stats(&stats, start_time));
            }

            // Wait for next interval
            tokio::time::sleep(interval).await;
        }

        Ok(samples)
    }

    /// Get container logs
    pub async fn get_logs(&self, tail: usize) -> Result<Vec<String>, DockerError> {
        let id = self
            .container_id
            .as_ref()
            .ok_or_else(|| DockerError::ContainerNotFound {
                name: self.config.container_name.clone(),
            })?;

        let options = LogsOptions::<String> {
            stdout: true,
            stderr: true,
            tail: tail.to_string(),
            ..Default::default()
        };

        let mut logs_stream = self.docker.logs(id, Some(options));
        let mut logs = Vec::new();

        while let Some(log_result) = logs_stream.next().await {
            match log_result {
                Ok(output) => {
                    logs.push(output.to_string().trim().to_string());
                }
                Err(e) => {
                    return Err(DockerError::Api(e));
                }
            }
        }

        Ok(logs)
    }

    /// Wait for container to be ready (logs contain expected message)
    pub async fn wait_for_ready(
        &self,
        ready_message: &str,
        timeout: Duration,
    ) -> Result<(), DockerError> {
        let start = Instant::now();

        while start.elapsed() < timeout {
            let logs = self.get_logs(50).await?;

            for line in &logs {
                if line.contains(ready_message) {
                    return Ok(());
                }
            }

            // Check if container is still running
            if !self.is_running().await? {
                let logs = self.get_logs(20).await?;
                return Err(DockerError::StartFailed {
                    name: self.config.container_name.clone(),
                    reason: logs.join("\n"),
                });
            }

            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        Err(DockerError::Timeout)
    }

    /// Get the container ID
    pub fn container_id(&self) -> Option<&str> {
        self.container_id.as_deref()
    }

    /// Get the config
    pub fn config(&self) -> &DockerRunnerConfig {
        &self.config
    }

    /// Attach to an existing container by name
    pub async fn attach_to_existing(container_name: &str) -> Result<Self, DockerError> {
        let docker = Self::connect_docker().await?;

        // Check if container exists and is running
        let info = docker
            .inspect_container(container_name, None)
            .await
            .map_err(|_| DockerError::ContainerNotFound {
                name: container_name.to_string(),
            })?;

        let is_running = info
            .state
            .as_ref()
            .and_then(|s| s.running)
            .unwrap_or(false);

        if !is_running {
            return Err(DockerError::ContainerNotFound {
                name: format!("{} (not running)", container_name),
            });
        }

        let container_id = info.id.clone();

        Ok(Self {
            docker,
            config: DockerRunnerConfig {
                container_name: container_name.to_string(),
                ..Default::default()
            },
            container_id,
            start_time: Some(Instant::now()),
        })
    }
}

impl Drop for DockerRunner {
    fn drop(&mut self) {
        // Note: We can't do async cleanup in Drop, so the container may be left running.
        // Users should call stop() and remove() explicitly, or use the cleanup methods.
    }
}

/// Parse memory limit string (e.g., "16g", "512m", "1024") to bytes
fn parse_memory_limit(s: &str) -> i64 {
    let s = s.trim().to_lowercase();

    if let Some(num) = s.strip_suffix('g') {
        num.parse::<i64>().unwrap_or(0) * 1024 * 1024 * 1024
    } else if let Some(num) = s.strip_suffix('m') {
        num.parse::<i64>().unwrap_or(0) * 1024 * 1024
    } else if let Some(num) = s.strip_suffix('k') {
        num.parse::<i64>().unwrap_or(0) * 1024
    } else {
        s.parse::<i64>().unwrap_or(0)
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
    fn test_config_build_args_checkpoint() {
        let config = DockerRunnerConfig::checkpoint_mode(60);
        let args = config.build_args();

        assert!(args.contains(&"--save-interval".to_string()));
        assert!(args.contains(&"60".to_string()));
        assert!(!args.contains(&"--pma-persist".to_string()));
    }

    #[test]
    fn test_config_build_args_pma_persist() {
        let config = DockerRunnerConfig::pma_persist_mode();
        let args = config.build_args();

        assert!(args.contains(&"--pma-persist".to_string()));
        assert!(!args.contains(&"--save-interval".to_string()));
    }

    #[test]
    fn test_config_build_args_mining() {
        let config = DockerRunnerConfig::default().with_fakenet_mining();
        let args = config.build_args();

        assert!(args.contains(&"--mine".to_string()));
        assert!(args.contains(&"--fakenet".to_string()));
        assert!(args.contains(&"--mining-pkh".to_string()));
    }

    #[test]
    fn test_config_build_env() {
        let config = DockerRunnerConfig::pma_persist_mode();
        let env = config.build_env();

        assert!(env.iter().any(|e| e.contains("RUST_LOG")));
        assert!(env.iter().any(|e| e == "NOCK_PMA_PERSIST=1"));
        assert!(env.iter().any(|e| e == "TRACY_NO_INVARIANT_CHECK=1"));
    }

    // Integration tests that require Docker
    #[tokio::test]
    #[ignore] // Run with: cargo test -p nockchain-bench -- --ignored
    async fn test_docker_available() {
        let available = DockerRunner::is_available().await;
        println!("Docker available: {}", available);
        assert!(available, "Docker must be available for this test");
    }

    #[tokio::test]
    #[ignore] // Run with: cargo test -p nockchain-bench -- --ignored
    async fn test_docker_runner_start_stop() {
        // This test requires:
        // 1. Docker to be running
        // 2. The nockchain-local:latest image to exist

        let config = DockerRunnerConfig::default()
            .with_container_name("nockchain-bench-test")
            .with_data_dir("/tmp/nockchain-bench-test")
            .with_fakenet_mining();

        let mut runner = match DockerRunner::new(config).await {
            Ok(r) => r,
            Err(e) => {
                println!("Skipping test - Docker not available: {}", e);
                return;
            }
        };

        // Start container
        println!("Starting container...");
        if let Err(e) = runner.start().await {
            println!("Failed to start container: {}", e);
            // Clean up
            let _ = runner.remove().await;
            panic!("Failed to start: {}", e);
        }

        // Wait a bit for it to initialize
        tokio::time::sleep(Duration::from_secs(5)).await;

        // Check if running
        let running = runner.is_running().await.unwrap_or(false);
        println!("Container running: {}", running);

        if running {
            // Get stats
            match runner.get_stats().await {
                Ok(stats) => {
                    println!("Memory usage: {} MB", stats.memory_usage_bytes / 1024 / 1024);
                    println!("Memory limit: {} MB", stats.memory_limit_bytes / 1024 / 1024);
                    println!("CPU percent: {:.2}%", stats.cpu_percent);
                }
                Err(e) => println!("Failed to get stats: {}", e),
            }

            // Get logs
            match runner.get_logs(10).await {
                Ok(logs) => {
                    println!("Recent logs:");
                    for log in logs {
                        println!("  {}", log);
                    }
                }
                Err(e) => println!("Failed to get logs: {}", e),
            }
        }

        // Stop and remove
        println!("Stopping container...");
        let _ = runner.stop().await;
        let _ = runner.remove().await;

        // Clean up data directory
        let _ = std::fs::remove_dir_all("/tmp/nockchain-bench-test");

        println!("Test completed");
    }

    #[tokio::test]
    #[ignore] // Run with: cargo test -p nockchain-bench -- --ignored test_attach_existing
    async fn test_attach_existing_container() {
        // This test requires a running container named "nockchain-exp"
        // Start one with: docker run -d --name nockchain-exp ...

        let runner = match DockerRunner::attach_to_existing("nockchain-exp").await {
            Ok(r) => r,
            Err(e) => {
                println!("Skipping test - no running nockchain-exp container: {}", e);
                println!("Start one with: docker run -d --name nockchain-exp ...");
                return;
            }
        };

        println!("Attached to container: {:?}", runner.container_id());

        // Get stats
        match runner.get_stats().await {
            Ok(stats) => {
                println!("\n=== Container Stats ===");
                println!("Memory usage:  {:>10.1} MiB", stats.memory_usage_bytes as f64 / 1024.0 / 1024.0);
                println!("Memory limit:  {:>10.1} MiB", stats.memory_limit_bytes as f64 / 1024.0 / 1024.0);
                println!("Memory %:      {:>10.1}%", stats.memory_percent);
                println!("Memory cache:  {:>10.1} MiB", stats.memory_cache_bytes as f64 / 1024.0 / 1024.0);
                println!("Memory RSS:    {:>10.1} MiB", stats.memory_rss_bytes as f64 / 1024.0 / 1024.0);
                println!("CPU %:         {:>10.1}%", stats.cpu_percent);

                // Verify we got real data
                assert!(stats.memory_usage_bytes > 0, "Expected non-zero memory usage");
                assert!(stats.memory_limit_bytes > 0, "Expected non-zero memory limit");
            }
            Err(e) => panic!("Failed to get stats: {}", e),
        }

        // Get logs
        match runner.get_logs(5).await {
            Ok(logs) => {
                println!("\n=== Recent Logs ===");
                for log in &logs {
                    println!("  {}", log);
                }
                assert!(!logs.is_empty(), "Expected some log output");
            }
            Err(e) => println!("Warning: Failed to get logs: {}", e),
        }

        println!("\nTest completed successfully!");
    }

    #[tokio::test]
    #[ignore] // Run with: cargo test -p nockchain-bench -- --ignored test_collect_stats
    async fn test_collect_stats_over_time() {
        // This test requires a running container named "nockchain-exp"

        let runner = match DockerRunner::attach_to_existing("nockchain-exp").await {
            Ok(r) => r,
            Err(e) => {
                println!("Skipping test - no running nockchain-exp container: {}", e);
                return;
            }
        };

        println!("Collecting stats for 10 seconds at 1-second intervals...\n");

        let samples = runner
            .collect_stats(Duration::from_secs(10), Duration::from_secs(1))
            .await
            .expect("Failed to collect stats");

        println!("Collected {} samples:\n", samples.len());
        println!("{:>10} {:>12} {:>12} {:>10}", "Time (ms)", "Memory (MiB)", "Cache (MiB)", "CPU %");
        println!("{}", "-".repeat(50));

        for sample in &samples {
            println!(
                "{:>10} {:>12.1} {:>12.1} {:>10.1}",
                sample.timestamp_ms,
                sample.memory_usage_bytes as f64 / 1024.0 / 1024.0,
                sample.memory_cache_bytes as f64 / 1024.0 / 1024.0,
                sample.cpu_percent
            );
        }

        assert!(samples.len() >= 5, "Expected at least 5 samples");
        println!("\nTest completed successfully!");
    }
}
