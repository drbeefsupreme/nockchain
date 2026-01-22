//! Test runner - orchestrates running benchmarks
//!
//! Manages the execution of benchmark tests, collecting data and coordinating
//! between the GUI and nockchain-bench library.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender};
use tokio::runtime::Runtime;
use uuid::Uuid;

use nockchain_bench::runner::{ContainerStats, DockerRunner, DockerRunnerConfig, NockchainMode};

use crate::config::{ContainerConfig, MetricType, PersistenceMode, TestConfig};
use crate::storage::{DataSample, TestEvent, TestResult, TestStatus};

/// Message from the runner to the GUI
#[derive(Debug, Clone)]
pub enum RunnerMessage {
    /// Test started
    Started {
        test_id: Uuid,
    },

    /// Sample collected
    Sample {
        test_id: Uuid,
        container_id: Uuid,
        sample: DataSample,
    },

    /// Event occurred
    Event {
        test_id: Uuid,
        event: TestEvent,
    },

    /// Log line from container
    Log {
        test_id: Uuid,
        container_id: Uuid,
        line: String,
        is_error: bool,
    },

    /// Progress update
    Progress {
        test_id: Uuid,
        elapsed_secs: f64,
        total_secs: f64,
        sample_count: usize,
    },

    /// Test completed
    Completed {
        test_id: Uuid,
        result: TestResult,
    },

    /// Test failed
    Failed {
        test_id: Uuid,
        error: String,
    },

    /// Docker availability status
    DockerAvailable(bool),
}

/// Command to the runner
#[derive(Debug)]
pub enum RunnerCommand {
    /// Start a test with a given ID
    Start { test_id: Uuid, config: TestConfig },

    /// Stop a running test
    Stop(Uuid),

    /// Check Docker availability
    CheckDocker,
}

/// Runner state for a single test
#[allow(dead_code)]
struct RunningTest {
    config: TestConfig,
    result: TestResult,
    start_time: Instant,
    cancelled: bool,
}

/// Test runner that executes benchmarks
pub struct TestRunner {
    /// Channel to send messages to the GUI
    tx: Sender<RunnerMessage>,

    /// Channel to receive commands from the GUI
    cmd_rx: Receiver<RunnerCommand>,

    /// Tokio runtime for async operations
    runtime: Runtime,

    /// Currently running tests
    running: Arc<Mutex<HashMap<Uuid, RunningTest>>>,
}

impl TestRunner {
    /// Create a new test runner and return the message receiver and command sender
    pub fn new() -> (Self, Receiver<RunnerMessage>, Sender<RunnerCommand>) {
        let (tx, rx) = bounded(1000);
        let (cmd_tx, cmd_rx) = bounded(100);

        let runtime = Runtime::new().expect("Failed to create Tokio runtime");

        let runner = Self {
            tx,
            cmd_rx,
            runtime,
            running: Arc::new(Mutex::new(HashMap::new())),
        };

        (runner, rx, cmd_tx)
    }

    /// Run the event loop (blocking)
    pub fn run(&self) {
        loop {
            match self.cmd_rx.recv() {
                Ok(cmd) => match cmd {
                    RunnerCommand::Start { test_id, config } => {
                        self.start_test(test_id, config);
                    }
                    RunnerCommand::Stop(id) => {
                        self.stop_test(id);
                    }
                    RunnerCommand::CheckDocker => {
                        self.check_docker();
                    }
                },
                Err(_) => {
                    // Channel closed, exit
                    break;
                }
            }
        }
    }

    /// Check if Docker is available
    fn check_docker(&self) {
        let tx = self.tx.clone();
        self.runtime.spawn(async move {
            let available = DockerRunner::is_available().await;
            let _ = tx.send(RunnerMessage::DockerAvailable(available));
        });
    }

    /// Start a test with the given ID
    fn start_test(&self, test_id: Uuid, config: TestConfig) {
        let tx = self.tx.clone();
        let running = self.running.clone();

        // Create the test result
        let result = TestResult::new(config.clone());

        // Register the running test
        {
            let mut r = running.lock().unwrap();
            r.insert(
                test_id,
                RunningTest {
                    config: config.clone(),
                    result,
                    start_time: Instant::now(),
                    cancelled: false,
                },
            );
        }

        // Notify start
        let _ = tx.send(RunnerMessage::Started { test_id });

        // Run the test in the async runtime
        let config_clone = config.clone();
        self.runtime.spawn(async move {
            match run_test_async(test_id, config_clone, tx.clone(), running.clone()).await {
                Ok(result) => {
                    let _ = tx.send(RunnerMessage::Completed { test_id, result });
                }
                Err(e) => {
                    let _ = tx.send(RunnerMessage::Failed {
                        test_id,
                        error: e.to_string(),
                    });
                }
            }

            // Clean up
            let mut r = running.lock().unwrap();
            r.remove(&test_id);
        });
    }

    /// Stop a running test
    fn stop_test(&self, test_id: Uuid) {
        let mut r = self.running.lock().unwrap();
        if let Some(test) = r.get_mut(&test_id) {
            test.cancelled = true;
        }
    }
}

/// Run a test asynchronously
async fn run_test_async(
    test_id: Uuid,
    config: TestConfig,
    tx: Sender<RunnerMessage>,
    running: Arc<Mutex<HashMap<Uuid, RunningTest>>>,
) -> Result<TestResult, Box<dyn std::error::Error + Send + Sync>> {
    let mut result = TestResult::new(config.clone());
    let start_time = Instant::now();
    let duration = Duration::from_secs(config.duration_secs);
    let sample_interval = Duration::from_millis(config.sample_interval_ms);

    // Start containers
    let mut runners: Vec<(Uuid, DockerRunner)> = Vec::new();

    for container_config in &config.containers {
        let docker_config = convert_container_config(container_config);

        // Create the data directory for the bind mount
        let data_dir_path = &docker_config.data_dir;
        let _ = tx.send(RunnerMessage::Log {
            test_id,
            container_id: container_config.id,
            line: format!("Creating data directory: {}", data_dir_path),
            is_error: false,
        });

        if let Err(e) = std::fs::create_dir_all(data_dir_path) {
            result.fail(format!(
                "Failed to create data directory '{}': {}",
                data_dir_path, e
            ));
            return Ok(result);
        }

        // Verify it exists
        if !std::path::Path::new(data_dir_path).exists() {
            result.fail(format!(
                "Data directory '{}' does not exist after creation",
                data_dir_path
            ));
            return Ok(result);
        }

        let _ = tx.send(RunnerMessage::Log {
            test_id,
            container_id: container_config.id,
            line: format!("Data directory created successfully: {}", data_dir_path),
            is_error: false,
        });

        match DockerRunner::new(docker_config).await {
            Ok(mut runner) => {
                // Start the container
                if let Err(e) = runner.start().await {
                    // Clean up any started containers
                    for (_, mut r) in runners {
                        let _ = r.stop().await;
                        let _ = r.remove().await;
                    }
                    result.fail(format!("Failed to start {}: {}", container_config.name, e));
                    return Ok(result);
                }

                let _ = tx.send(RunnerMessage::Log {
                    test_id,
                    container_id: container_config.id,
                    line: format!("Container {} started", container_config.name),
                    is_error: false,
                });

                runners.push((container_config.id, runner));
            }
            Err(e) => {
                result.fail(format!(
                    "Failed to create runner for {}: {}",
                    container_config.name, e
                ));
                return Ok(result);
            }
        }
    }

    // Wait for containers to be ready (simple delay for now)
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Collection loop
    let mut sample_count = 0;

    while start_time.elapsed() < duration {
        // Check if cancelled
        {
            let r = running.lock().unwrap();
            if let Some(test) = r.get(&test_id) {
                if test.cancelled {
                    result.cancel();
                    break;
                }
            }
        }

        let timestamp_ms = start_time.elapsed().as_millis() as u64;

        // Collect stats from each container
        for (container_id, runner) in &runners {
            match runner.get_stats().await {
                Ok(stats) => {
                    let sample = convert_stats_to_sample(*container_id, timestamp_ms, &stats, &config.metrics);
                    result.add_sample(sample.clone());

                    let _ = tx.send(RunnerMessage::Sample {
                        test_id,
                        container_id: *container_id,
                        sample,
                    });
                }
                Err(e) => {
                    let _ = tx.send(RunnerMessage::Log {
                        test_id,
                        container_id: *container_id,
                        line: format!("Failed to get stats: {}", e),
                        is_error: true,
                    });
                }
            }
        }

        sample_count += 1;

        // Send progress
        let _ = tx.send(RunnerMessage::Progress {
            test_id,
            elapsed_secs: start_time.elapsed().as_secs_f64(),
            total_secs: duration.as_secs_f64(),
            sample_count,
        });

        // Wait for next sample
        tokio::time::sleep(sample_interval).await;
    }

    // Collect final logs
    for (container_id, runner) in &runners {
        if let Ok(logs) = runner.get_logs(100).await {
            result.add_logs(*container_id, logs.clone());

            for line in logs {
                let _ = tx.send(RunnerMessage::Log {
                    test_id,
                    container_id: *container_id,
                    line,
                    is_error: false,
                });
            }
        }
    }

    // Stop containers
    for (container_id, mut runner) in runners {
        let _ = tx.send(RunnerMessage::Log {
            test_id,
            container_id,
            line: "Stopping container...".to_string(),
            is_error: false,
        });

        let _ = runner.stop().await;
        let _ = runner.remove().await;
    }

    if result.status == TestStatus::Running {
        result.complete();
    }

    Ok(result)
}

/// Get the base directory for benchmark data
/// Uses ~/.nockchain-bench-data since /tmp may not be shared with Docker Desktop
fn get_bench_data_dir() -> String {
    if let Some(home) = dirs::home_dir() {
        home.join(".nockchain-bench-data").to_string_lossy().to_string()
    } else {
        // Fallback to /tmp if home dir not available
        "/tmp/nockchain-bench-data".to_string()
    }
}

/// Convert our ContainerConfig to nockchain-bench's DockerRunnerConfig
fn convert_container_config(config: &ContainerConfig) -> DockerRunnerConfig {
    let mode = match config.persistence_mode {
        PersistenceMode::Checkpoint => NockchainMode::Checkpoint {
            save_interval_secs: config.checkpoint_interval_secs,
        },
        PersistenceMode::PmaPersist => NockchainMode::PmaPersist,
    };

    let env_vars: HashMap<String, String> = config
        .env_vars
        .iter()
        .cloned()
        .collect();

    // Use home directory instead of /tmp for Docker Desktop compatibility
    let base_dir = get_bench_data_dir();
    let data_dir = format!("{}/{}", base_dir, config.id);

    DockerRunnerConfig {
        image: config.image.clone(),
        container_name: format!("bench-{}-{}", sanitize_name(&config.name), &config.id.to_string()[..8]),
        data_dir,
        memory_limit: Some(config.memory_limit.clone()),
        mode,
        mine: config.enable_mining,
        mining_pkh: if config.use_fakenet {
            Some("11111111111111111111111111111111111".to_string())
        } else {
            None
        },
        fakenet: config.use_fakenet,
        num_threads: config.num_threads,
        fast_sync: config.enable_fast_sync,
        env_vars,
    }
}

/// Convert Docker stats to our DataSample
fn convert_stats_to_sample(
    container_id: Uuid,
    timestamp_ms: u64,
    stats: &ContainerStats,
    metrics: &[MetricType],
) -> DataSample {
    let mut sample = DataSample::new(timestamp_ms, container_id);

    for metric in metrics {
        let value = match metric {
            MetricType::ContainerMemory => stats.memory_usage_bytes as f64 / 1024.0,
            MetricType::ContainerRss => stats.memory_rss_bytes as f64 / 1024.0,
            MetricType::ContainerCache => stats.memory_cache_bytes as f64 / 1024.0,
            MetricType::CpuPercent => stats.cpu_percent,
            // For now, we only have Docker stats; proc-based metrics would need pid access
            _ => continue,
        };

        sample.values.insert(*metric, value);
    }

    sample
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

/// Handle for interacting with a runner from the GUI
pub struct RunnerHandle {
    /// Command sender
    cmd_tx: Sender<RunnerCommand>,

    /// Message receiver
    msg_rx: Receiver<RunnerMessage>,

    /// Pending messages (for non-blocking polling)
    pending: Vec<RunnerMessage>,
}

impl RunnerHandle {
    /// Create a new runner handle
    pub fn new(cmd_tx: Sender<RunnerCommand>, msg_rx: Receiver<RunnerMessage>) -> Self {
        Self {
            cmd_tx,
            msg_rx,
            pending: Vec::new(),
        }
    }

    /// Start a test with the given ID
    pub fn start_test(&self, test_id: Uuid, config: TestConfig) -> Result<(), crossbeam_channel::SendError<RunnerCommand>> {
        self.cmd_tx.send(RunnerCommand::Start { test_id, config })
    }

    /// Stop a test
    pub fn stop_test(&self, test_id: Uuid) -> Result<(), crossbeam_channel::SendError<RunnerCommand>> {
        self.cmd_tx.send(RunnerCommand::Stop(test_id))
    }

    /// Check Docker availability
    pub fn check_docker(&self) -> Result<(), crossbeam_channel::SendError<RunnerCommand>> {
        self.cmd_tx.send(RunnerCommand::CheckDocker)
    }

    /// Poll for messages (non-blocking)
    pub fn poll(&mut self) -> Vec<RunnerMessage> {
        let mut messages = std::mem::take(&mut self.pending);

        while let Ok(msg) = self.msg_rx.try_recv() {
            messages.push(msg);
        }

        messages
    }

    /// Wait for a message (blocking with timeout)
    pub fn recv_timeout(&self, timeout: Duration) -> Option<RunnerMessage> {
        self.msg_rx.recv_timeout(timeout).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_name() {
        assert_eq!(sanitize_name("My Container"), "my-container");
        assert_eq!(sanitize_name("Test_123"), "test_123");
        assert_eq!(sanitize_name("a!@#b"), "a---b");
    }

    #[test]
    fn test_convert_container_config() {
        let config = ContainerConfig::checkpoint("Test", 60);
        let docker_config = convert_container_config(&config);

        assert!(docker_config.container_name.starts_with("bench-test-"));
        assert!(matches!(
            docker_config.mode,
            NockchainMode::Checkpoint { save_interval_secs: 60 }
        ));
    }

    #[test]
    fn test_convert_container_config_pma() {
        let config = ContainerConfig::pma_persist("Test PMA");
        let docker_config = convert_container_config(&config);

        assert!(matches!(docker_config.mode, NockchainMode::PmaPersist));
    }

    #[test]
    fn test_runner_message_types() {
        // Just verify the enum variants compile correctly
        let _msg = RunnerMessage::Started {
            test_id: Uuid::new_v4(),
        };
        let _msg = RunnerMessage::Progress {
            test_id: Uuid::new_v4(),
            elapsed_secs: 10.0,
            total_secs: 100.0,
            sample_count: 5,
        };
        let _msg = RunnerMessage::DockerAvailable(true);
    }

    #[test]
    fn test_runner_handle() {
        let (tx, rx) = bounded(10);
        let (cmd_tx, _cmd_rx) = bounded(10);

        let mut handle = RunnerHandle::new(cmd_tx, rx);

        // Send some messages
        tx.send(RunnerMessage::DockerAvailable(true)).unwrap();
        tx.send(RunnerMessage::Started {
            test_id: Uuid::new_v4(),
        })
        .unwrap();

        // Poll for them
        let messages = handle.poll();
        assert_eq!(messages.len(), 2);
    }
}
