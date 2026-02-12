//! Mining scenario for benchmarking Nockchain memory usage during mining
//!
//! This scenario runs a Nockchain node in mining mode and collects memory
//! statistics over time, allowing comparison between different configurations.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::runner::{ContainerStats, DockerError, DockerRunner, DockerRunnerConfig, NockchainMode};

#[derive(Debug, Error)]
pub enum ScenarioError {
    #[error("Docker error: {0}")]
    Docker(#[from] DockerError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Scenario failed: {0}")]
    Failed(String),

    #[error("Container exited unexpectedly: {0}")]
    ContainerExited(String),
}

/// Configuration for a mining scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningScenarioConfig {
    /// Human-readable name for this scenario run
    pub name: String,

    /// Nockchain mode (Checkpoint or PmaPersist)
    pub mode: NockchainMode,

    /// Duration to run the mining scenario
    pub duration: Duration,

    /// Interval between stat samples
    pub sample_interval: Duration,

    /// Docker image to use
    pub image: String,

    /// Host directory for data (will be created if needed)
    pub data_dir: PathBuf,

    /// Memory limit for the container (e.g., "16g")
    pub memory_limit: Option<String>,

    /// Number of mining threads (0 = auto)
    pub num_threads: u32,

    /// Use fakenet (local testing mode)
    pub fakenet: bool,

    /// Mining public key hash (required for real mining, auto-generated for fakenet)
    pub mining_pkh: Option<String>,

    /// Message to wait for in logs before starting measurement
    pub ready_message: Option<String>,

    /// Timeout waiting for ready message
    pub ready_timeout: Duration,

    /// Extra environment variables passed into the container.
    pub env_vars: HashMap<String, String>,
}

impl Default for MiningScenarioConfig {
    fn default() -> Self {
        Self {
            name: "mining-benchmark".to_string(),
            mode: NockchainMode::default(),
            duration: Duration::from_secs(300), // 5 minutes
            sample_interval: Duration::from_secs(1),
            image: "nockchain-local:latest".to_string(),
            data_dir: PathBuf::from("/tmp/nockchain-bench"),
            memory_limit: Some("16g".to_string()),
            num_threads: 1,
            fakenet: true,
            mining_pkh: None,
            ready_message: Some("heaviest".to_string()), // First mined block
            ready_timeout: Duration::from_secs(120),
            env_vars: HashMap::new(),
        }
    }
}

impl MiningScenarioConfig {
    /// Create a checkpoint mode scenario
    pub fn checkpoint_mode(name: impl Into<String>, save_interval_secs: u64) -> Self {
        Self {
            name: name.into(),
            mode: NockchainMode::Checkpoint { save_interval_secs },
            ..Default::default()
        }
    }

    /// Create a PMA persist mode scenario
    pub fn pma_persist_mode(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            mode: NockchainMode::PmaPersist,
            ..Default::default()
        }
    }

    /// Set the duration
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Set the sample interval
    pub fn with_sample_interval(mut self, interval: Duration) -> Self {
        self.sample_interval = interval;
        self
    }

    /// Set the data directory
    pub fn with_data_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.data_dir = path.into();
        self
    }

    /// Set the memory limit
    pub fn with_memory_limit(mut self, limit: impl Into<String>) -> Self {
        self.memory_limit = Some(limit.into());
        self
    }

    /// Set number of threads
    pub fn with_num_threads(mut self, threads: u32) -> Self {
        self.num_threads = threads;
        self
    }

    /// Set the Docker image
    pub fn with_image(mut self, image: impl Into<String>) -> Self {
        self.image = image.into();
        self
    }

    /// Add an environment variable for the scenario container.
    pub fn with_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Convert to DockerRunnerConfig
    fn to_docker_config(&self) -> DockerRunnerConfig {
        let container_name = format!("nockchain-bench-{}", sanitize_name(&self.name));

        DockerRunnerConfig {
            image: self.image.clone(),
            container_name,
            data_dir: self.data_dir.to_string_lossy().to_string(),
            memory_limit: self.memory_limit.clone(),
            mode: self.mode.clone(),
            mine: true,
            mining_pkh: self.mining_pkh.clone().or_else(|| {
                if self.fakenet {
                    // Use a test PKH for fakenet
                    Some("11111111111111111111111111111111111".to_string())
                } else {
                    None
                }
            }),
            fakenet: self.fakenet,
            num_threads: self.num_threads,
            fast_sync: true,
            env_vars: self.env_vars.clone(),
            bind_port: 30000,
        }
    }
}

/// Results from running a mining scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningResult {
    /// Configuration used for this run
    pub config: MiningScenarioConfig,

    /// When the scenario started (Unix timestamp)
    pub start_time_unix: u64,

    /// Total duration of the run
    pub actual_duration: Duration,

    /// Time-series of container stats
    pub samples: Vec<ContainerStats>,

    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,

    /// Average memory usage in bytes
    pub avg_memory_bytes: u64,

    /// Peak RSS in bytes
    pub peak_rss_bytes: u64,

    /// Average RSS in bytes
    pub avg_rss_bytes: u64,

    /// Final container logs (last N lines)
    pub final_logs: Vec<String>,

    /// Whether the scenario completed successfully
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,
}

impl MiningResult {
    /// Create a new result from collected samples
    fn from_samples(
        config: MiningScenarioConfig,
        samples: Vec<ContainerStats>,
        start_time: Instant,
        logs: Vec<String>,
        success: bool,
        error: Option<String>,
    ) -> Self {
        let actual_duration = start_time.elapsed();

        let (peak_memory, avg_memory, peak_rss, avg_rss) = if samples.is_empty() {
            (0, 0, 0, 0)
        } else {
            let peak_memory = samples
                .iter()
                .map(|s| s.memory_usage_bytes)
                .max()
                .unwrap_or(0);
            let avg_memory =
                samples.iter().map(|s| s.memory_usage_bytes).sum::<u64>() / samples.len() as u64;
            let peak_rss = samples
                .iter()
                .map(|s| s.memory_rss_bytes)
                .max()
                .unwrap_or(0);
            let avg_rss =
                samples.iter().map(|s| s.memory_rss_bytes).sum::<u64>() / samples.len() as u64;
            (peak_memory, avg_memory, peak_rss, avg_rss)
        };

        Self {
            config,
            start_time_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            actual_duration,
            samples,
            peak_memory_bytes: peak_memory,
            avg_memory_bytes: avg_memory,
            peak_rss_bytes: peak_rss,
            avg_rss_bytes: avg_rss,
            final_logs: logs,
            success,
            error,
        }
    }

    /// Get memory usage as MiB
    pub fn peak_memory_mib(&self) -> f64 {
        self.peak_memory_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get average memory as MiB
    pub fn avg_memory_mib(&self) -> f64 {
        self.avg_memory_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get peak RSS as MiB
    pub fn peak_rss_mib(&self) -> f64 {
        self.peak_rss_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get average RSS as MiB
    pub fn avg_rss_mib(&self) -> f64 {
        self.avg_rss_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get the number of samples collected
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Print a summary of the results
    pub fn print_summary(&self) {
        println!("\n=== Mining Scenario Results: {} ===\n", self.config.name);

        let mode_str = match &self.config.mode {
            NockchainMode::Checkpoint { save_interval_secs } => {
                format!("Checkpoint ({}s interval)", save_interval_secs)
            }
            NockchainMode::PmaPersist => "PMA Persist".to_string(),
        };

        println!("Mode:            {}", mode_str);
        println!(
            "Duration:        {:.1}s",
            self.actual_duration.as_secs_f64()
        );
        println!("Samples:         {}", self.samples.len());
        println!(
            "Status:          {}",
            if self.success { "Success" } else { "Failed" }
        );

        if let Some(ref err) = self.error {
            println!("Error:           {}", err);
        }

        println!();
        println!("Memory Usage:");
        println!("  Peak:          {:>10.1} MiB", self.peak_memory_mib());
        println!("  Average:       {:>10.1} MiB", self.avg_memory_mib());
        println!();
        println!("RSS:");
        println!("  Peak:          {:>10.1} MiB", self.peak_rss_mib());
        println!("  Average:       {:>10.1} MiB", self.avg_rss_mib());
    }
}

/// Mining scenario runner
pub struct MiningScenario {
    config: MiningScenarioConfig,
}

impl MiningScenario {
    /// Create a new mining scenario
    pub fn new(config: MiningScenarioConfig) -> Self {
        Self { config }
    }

    /// Create a checkpoint mode scenario
    pub fn checkpoint(name: impl Into<String>, save_interval_secs: u64) -> Self {
        Self::new(MiningScenarioConfig::checkpoint_mode(
            name, save_interval_secs,
        ))
    }

    /// Create a PMA persist mode scenario
    pub fn pma_persist(name: impl Into<String>) -> Self {
        Self::new(MiningScenarioConfig::pma_persist_mode(name))
    }

    /// Get the configuration
    pub fn config(&self) -> &MiningScenarioConfig {
        &self.config
    }

    /// Get a mutable reference to the configuration
    pub fn config_mut(&mut self) -> &mut MiningScenarioConfig {
        &mut self.config
    }

    /// Run the scenario and collect results
    pub async fn run(&self) -> Result<MiningResult, ScenarioError> {
        let start_time = Instant::now();

        // Ensure data directory exists and is empty
        self.prepare_data_dir()?;

        // Create Docker runner
        let docker_config = self.config.to_docker_config();
        let mut runner = DockerRunner::new(docker_config).await?;

        println!("Starting container for scenario: {}", self.config.name);

        // Start the container
        runner.start().await?;

        // Wait for ready message if specified
        if let Some(ref ready_msg) = self.config.ready_message {
            println!("Waiting for ready message: '{}'...", ready_msg);
            match runner
                .wait_for_ready(ready_msg, self.config.ready_timeout)
                .await
            {
                Ok(()) => println!("Container ready!"),
                Err(e) => {
                    let logs = runner.get_logs(20).await.unwrap_or_default();
                    runner.stop().await.ok();
                    runner.remove().await.ok();
                    return Ok(MiningResult::from_samples(
                        self.config.clone(),
                        vec![],
                        start_time,
                        logs,
                        false,
                        Some(format!("Failed waiting for ready: {}", e)),
                    ));
                }
            }
        }

        println!("Collecting stats for {:?}...", self.config.duration);

        // Collect stats
        let samples = match runner
            .collect_stats(self.config.duration, self.config.sample_interval)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                let logs = runner.get_logs(20).await.unwrap_or_default();
                runner.stop().await.ok();
                runner.remove().await.ok();
                return Ok(MiningResult::from_samples(
                    self.config.clone(),
                    vec![],
                    start_time,
                    logs,
                    false,
                    Some(format!("Failed collecting stats: {}", e)),
                ));
            }
        };

        // Get final logs
        let logs = runner.get_logs(50).await.unwrap_or_default();

        // Stop and remove container
        println!("Stopping container...");
        runner.stop().await.ok();
        runner.remove().await.ok();

        Ok(MiningResult::from_samples(
            self.config.clone(),
            samples,
            start_time,
            logs,
            true,
            None,
        ))
    }

    /// Run the scenario against an existing container (for testing)
    pub async fn run_against_existing(
        &self,
        container_name: &str,
    ) -> Result<MiningResult, ScenarioError> {
        let start_time = Instant::now();

        // Attach to existing container
        let runner = DockerRunner::attach_to_existing(container_name).await?;

        println!("Attached to existing container: {}", container_name);
        println!("Collecting stats for {:?}...", self.config.duration);

        // Collect stats
        let samples = runner
            .collect_stats(self.config.duration, self.config.sample_interval)
            .await?;

        // Get logs
        let logs = runner.get_logs(50).await.unwrap_or_default();

        // Don't stop the container since we didn't start it

        Ok(MiningResult::from_samples(
            self.config.clone(),
            samples,
            start_time,
            logs,
            true,
            None,
        ))
    }

    /// Prepare the data directory (create if needed, optionally clean)
    fn prepare_data_dir(&self) -> Result<(), ScenarioError> {
        let path = &self.config.data_dir;

        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Create the directory if it doesn't exist
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }

        Ok(())
    }
}

/// Sanitize a name for use in container names
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_config_default() {
        let config = MiningScenarioConfig::default();
        assert_eq!(config.duration, Duration::from_secs(300));
        assert_eq!(config.sample_interval, Duration::from_secs(1));
        assert!(config.fakenet);
        assert!(matches!(config.mode, NockchainMode::Checkpoint { .. }));
    }

    #[test]
    fn test_scenario_config_checkpoint() {
        let config = MiningScenarioConfig::checkpoint_mode("test-checkpoint", 60);
        assert_eq!(config.name, "test-checkpoint");
        assert!(matches!(
            config.mode,
            NockchainMode::Checkpoint {
                save_interval_secs: 60
            }
        ));
    }

    #[test]
    fn test_scenario_config_pma_persist() {
        let config = MiningScenarioConfig::pma_persist_mode("test-pma");
        assert_eq!(config.name, "test-pma");
        assert!(matches!(config.mode, NockchainMode::PmaPersist));
    }

    #[test]
    fn test_scenario_config_builder() {
        let config = MiningScenarioConfig::checkpoint_mode("test", 120)
            .with_duration(Duration::from_secs(60))
            .with_sample_interval(Duration::from_millis(500))
            .with_memory_limit("8g")
            .with_num_threads(2);

        assert_eq!(config.duration, Duration::from_secs(60));
        assert_eq!(config.sample_interval, Duration::from_millis(500));
        assert_eq!(config.memory_limit, Some("8g".to_string()));
        assert_eq!(config.num_threads, 2);
    }

    #[test]
    fn test_to_docker_config() {
        let config = MiningScenarioConfig::pma_persist_mode("my-test")
            .with_memory_limit("4g")
            .with_num_threads(2);

        let docker_config = config.to_docker_config();

        assert_eq!(docker_config.container_name, "nockchain-bench-my-test");
        assert!(matches!(docker_config.mode, NockchainMode::PmaPersist));
        assert!(docker_config.mine);
        assert!(docker_config.fakenet);
        assert_eq!(docker_config.memory_limit, Some("4g".to_string()));
        assert_eq!(docker_config.num_threads, 2);
    }

    #[test]
    fn test_sanitize_name() {
        assert_eq!(sanitize_name("My Test"), "my-test");
        assert_eq!(sanitize_name("test_123"), "test_123");
        assert_eq!(sanitize_name("TEST!@#"), "test---");
    }

    #[test]
    fn test_mining_result_calculations() {
        let samples = vec![
            ContainerStats {
                timestamp_ms: 0,
                memory_usage_bytes: 1000 * 1024 * 1024, // 1000 MiB
                memory_limit_bytes: 16000 * 1024 * 1024,
                memory_percent: 6.25,
                memory_cache_bytes: 100 * 1024 * 1024,
                memory_rss_bytes: 900 * 1024 * 1024, // 900 MiB
                cpu_percent: 100.0,
                minor_faults: None,
                major_faults: None,
            },
            ContainerStats {
                timestamp_ms: 1000,
                memory_usage_bytes: 2000 * 1024 * 1024, // 2000 MiB
                memory_limit_bytes: 16000 * 1024 * 1024,
                memory_percent: 12.5,
                memory_cache_bytes: 200 * 1024 * 1024,
                memory_rss_bytes: 1800 * 1024 * 1024, // 1800 MiB
                cpu_percent: 100.0,
                minor_faults: None,
                major_faults: None,
            },
            ContainerStats {
                timestamp_ms: 2000,
                memory_usage_bytes: 1500 * 1024 * 1024, // 1500 MiB
                memory_limit_bytes: 16000 * 1024 * 1024,
                memory_percent: 9.375,
                memory_cache_bytes: 150 * 1024 * 1024,
                memory_rss_bytes: 1350 * 1024 * 1024, // 1350 MiB
                cpu_percent: 100.0,
                minor_faults: None,
                major_faults: None,
            },
        ];

        let result = MiningResult::from_samples(
            MiningScenarioConfig::default(),
            samples,
            Instant::now(),
            vec![],
            true,
            None,
        );

        assert_eq!(result.peak_memory_mib(), 2000.0);
        assert_eq!(result.avg_memory_mib(), 1500.0); // (1000 + 2000 + 1500) / 3
        assert_eq!(result.peak_rss_mib(), 1800.0);
        assert_eq!(result.avg_rss_mib(), 1350.0); // (900 + 1800 + 1350) / 3
        assert_eq!(result.sample_count(), 3);
        assert!(result.success);
    }

    #[test]
    fn test_mining_result_empty_samples() {
        let result = MiningResult::from_samples(
            MiningScenarioConfig::default(),
            vec![],
            Instant::now(),
            vec![],
            false,
            Some("test error".to_string()),
        );

        assert_eq!(result.peak_memory_bytes, 0);
        assert_eq!(result.avg_memory_bytes, 0);
        assert_eq!(result.sample_count(), 0);
        assert!(!result.success);
        assert_eq!(result.error, Some("test error".to_string()));
    }

    // Integration test - requires Docker
    #[tokio::test]
    #[ignore]
    async fn test_run_against_existing() {
        let scenario = MiningScenario::new(
            MiningScenarioConfig::default()
                .with_duration(Duration::from_secs(5))
                .with_sample_interval(Duration::from_secs(1)),
        );

        match scenario.run_against_existing("nockchain-exp").await {
            Ok(result) => {
                result.print_summary();
                assert!(result.success);
                assert!(result.sample_count() >= 3);
                assert!(result.peak_memory_bytes > 0);
            }
            Err(e) => {
                println!("Skipping test - no running container: {}", e);
            }
        }
    }
}
