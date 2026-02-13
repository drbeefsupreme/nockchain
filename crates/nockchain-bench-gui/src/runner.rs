//! Test runner - orchestrates running benchmarks
//!
//! Manages the execution of benchmark tests, collecting data and coordinating
//! between the GUI and nockchain-bench library.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender};
use nockchain_bench::events::log_parser::LogParser;
use nockchain_bench::runner::{ContainerStats, DockerRunner, DockerRunnerConfig, NockchainMode};
use nockchain_bench::sampler::buckets::MemoryAttribution;
use nockchain_bench::scenario::{MiningScenario, MiningScenarioConfig};
use nockchain_bench::speed_of_light::{
    build_sweep_cases, checkpoint_durations_ms, page_fault_bursts, read_fixture_file,
    summarize_case_runs, BenchConfig as SolBenchConfig, BenchRunner, FixtureBuildConfig,
    FixtureBuildPhase, FixtureBuildProgress, FixtureBuilder, SolHeight, SweepRunMetrics,
};
use tokio::runtime::Runtime;
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::config::{
    BenchmarkMode, ContainerConfig, MetricType, PersistenceMode, SolFixtureOptions, TestConfig,
};
use crate::storage::{
    DataSample, SolBenchResult, SolSweepResult, TestEvent, TestResult, TestStatus,
};

/// Message from the runner to the GUI
#[derive(Debug, Clone)]
pub enum RunnerMessage {
    /// Test started
    Started { test_id: Uuid },

    /// Sample collected
    Sample {
        test_id: Uuid,
        container_id: Uuid,
        sample: DataSample,
    },

    /// Event occurred
    Event { test_id: Uuid, event: TestEvent },

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
    Completed { test_id: Uuid, result: TestResult },

    /// Test failed
    Failed { test_id: Uuid, error: String },

    /// Docker availability status
    DockerAvailable(bool),

    /// SOL fixture creation started
    SolFixtureStarted { job_id: Uuid },

    /// SOL fixture creator log line
    SolFixtureLog {
        job_id: Uuid,
        line: String,
        is_error: bool,
    },

    /// SOL fixture build progress
    SolFixtureProgress {
        job_id: Uuid,
        phase: FixtureBuildPhase,
        blocks_done: Option<u64>,
        blocks_total: Option<u64>,
        message: String,
    },

    /// SOL fixture creation completed
    SolFixtureCompleted {
        job_id: Uuid,
        output_path: PathBuf,
        derived_checkpoint_height: u64,
        archive_start_height: u64,
        archive_end_height: u64,
    },

    /// SOL fixture creation failed
    SolFixtureFailed { job_id: Uuid, error: String },
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

    /// Build a unified SOL fixture (.soltest)
    CreateSolFixture {
        job_id: Uuid,
        options: SolFixtureOptions,
    },
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
                    RunnerCommand::CreateSolFixture { job_id, options } => {
                        self.create_sol_fixture(job_id, options);
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

    /// Start a SOL fixture creation job
    fn create_sol_fixture(&self, job_id: Uuid, options: SolFixtureOptions) {
        let tx = self.tx.clone();
        let _ = tx.send(RunnerMessage::SolFixtureStarted { job_id });

        self.runtime.spawn(async move {
            match run_sol_fixture_build_async(job_id, options, tx.clone()).await {
                Ok((output_path, derived_height, archive_start, archive_end)) => {
                    let _ = tx.send(RunnerMessage::SolFixtureCompleted {
                        job_id,
                        output_path,
                        derived_checkpoint_height: derived_height,
                        archive_start_height: archive_start,
                        archive_end_height: archive_end,
                    });
                }
                Err(error) => {
                    let _ = tx.send(RunnerMessage::SolFixtureFailed {
                        job_id,
                        error: error.to_string(),
                    });
                }
            }
        });
    }
}

/// Run a test asynchronously
async fn run_test_async(
    test_id: Uuid,
    config: TestConfig,
    tx: Sender<RunnerMessage>,
    running: Arc<Mutex<HashMap<Uuid, RunningTest>>>,
) -> Result<TestResult, Box<dyn std::error::Error + Send + Sync>> {
    match config.benchmark_mode {
        BenchmarkMode::Container => run_container_test_async(test_id, config, tx, running).await,
        BenchmarkMode::SpeedOfLightBench => {
            run_sol_bench_test_async(test_id, config, tx, running).await
        }
        BenchmarkMode::SpeedOfLightSweep => {
            run_sol_sweep_test_async(test_id, config, tx, running).await
        }
    }
}

async fn run_container_test_async(
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

    for (idx, container_config) in config.containers.iter().enumerate() {
        // Each container gets a unique port to avoid conflicts with host networking
        let docker_config = convert_container_config(container_config, idx as u16);

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
    // Track how many log lines we've already sent per container to avoid duplicates
    let mut log_cursors: HashMap<Uuid, usize> = HashMap::new();
    // Log parsers for each container to extract events
    let mut log_parsers: HashMap<Uuid, LogParser> = HashMap::new();
    for (container_id, _) in &runners {
        log_cursors.insert(*container_id, 0);
        log_parsers.insert(*container_id, LogParser::new());
    }

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

        // Collect stats and stream logs from each container
        for (container_id, runner) in &runners {
            // Get stats
            match runner.get_stats().await {
                Ok(stats) => {
                    let sample = convert_stats_to_sample(
                        *container_id, timestamp_ms, &stats, &config.metrics,
                    );
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

            // Stream new logs (fetch recent logs and send only new ones)
            if let Ok(logs) = runner.get_logs(200).await {
                let cursor = log_cursors.get(container_id).copied().unwrap_or(0);
                // If we have more logs than before, send the new ones
                if logs.len() > cursor {
                    for line in logs.iter().skip(cursor) {
                        let _ = tx.send(RunnerMessage::Log {
                            test_id,
                            container_id: *container_id,
                            line: line.clone(),
                            is_error: false,
                        });

                        // Parse the log line for events
                        if let Some(parser) = log_parsers.get_mut(container_id) {
                            if let Some(log_event) = parser.parse_line(line) {
                                if log_event.is_significant() {
                                    let event = TestEvent::new(
                                        timestamp_ms, // Use sample timestamp for consistency
                                        *container_id,
                                        log_event.event_type.label(),
                                        log_event.raw_line.clone(),
                                    )
                                    .significant();
                                    result.add_event(event.clone());
                                    let _ = tx.send(RunnerMessage::Event { test_id, event });
                                }
                            }
                        }
                    }
                    log_cursors.insert(*container_id, logs.len());
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

    // Collect any remaining logs at the end
    for (container_id, runner) in &runners {
        if let Ok(logs) = runner.get_logs(200).await {
            let cursor = log_cursors.get(container_id).copied().unwrap_or(0);
            // Send any new logs we haven't sent yet
            if logs.len() > cursor {
                for line in logs.iter().skip(cursor) {
                    let _ = tx.send(RunnerMessage::Log {
                        test_id,
                        container_id: *container_id,
                        line: line.clone(),
                        is_error: false,
                    });
                }
            }
            // Save all logs to result
            result.add_logs(*container_id, logs);
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

async fn run_sol_bench_test_async(
    test_id: Uuid,
    config: TestConfig,
    tx: Sender<RunnerMessage>,
    running: Arc<Mutex<HashMap<Uuid, RunningTest>>>,
) -> Result<TestResult, Box<dyn std::error::Error + Send + Sync>> {
    let mut result = TestResult::new(config.clone());
    let container_id = config
        .containers
        .first()
        .map(|container| container.id)
        .unwrap_or_else(Uuid::nil);
    let bench_start = Instant::now();

    let _ = tx.send(RunnerMessage::Log {
        test_id,
        container_id,
        line: "Starting SOL bench run".to_string(),
        is_error: false,
    });
    let _ = tx.send(RunnerMessage::Progress {
        test_id,
        elapsed_secs: 0.0,
        total_secs: 0.0,
        sample_count: 0,
    });

    // Check cancellation before startup
    if is_test_cancelled(test_id, &running) {
        result.cancel();
        return Ok(result);
    }

    let options = &config.sol_bench;
    struct TempDirGuard {
        path: PathBuf,
    }
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    let fixture_path = options
        .fixture_path
        .as_ref()
        .map(|path| path.trim())
        .filter(|path| !path.is_empty())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "SOL bench requires a .soltest fixture path",
            )
        })?;

    let fixture = read_fixture_file(fixture_path)?;
    let fixture_temp_dir = std::env::temp_dir().join(format!(
        "nockchain-bench-gui-fixture-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&fixture_temp_dir)?;

    let checkpoint_file = fixture_temp_dir.join("fixture.chkjam");
    let archive_file = fixture_temp_dir.join("fixture.solarch");
    let kernel_file = fixture_temp_dir.join("fixture.jam");
    std::fs::write(&checkpoint_file, &fixture.checkpoint_bytes)?;
    std::fs::write(&archive_file, &fixture.archive_bytes)?;
    std::fs::write(&kernel_file, &fixture.kernel_bytes)?;

    let archive_path = archive_file.to_string_lossy().to_string();
    let kernel_path = kernel_file.to_string_lossy().to_string();
    let checkpoint_path = Some(checkpoint_file.to_string_lossy().to_string());
    let start_height = Some(fixture.manifest.archive_start_height.as_u64());

    let _ = tx.send(RunnerMessage::Log {
        test_id,
        container_id,
        line: format!("Using SOL fixture: {}", fixture_path),
        is_error: false,
    });
    let _ = tx.send(RunnerMessage::Log {
        test_id,
        container_id,
        line: format!(
            "Fixture archive range: {}..={}",
            fixture.manifest.archive_start_height.as_u64(),
            fixture.manifest.archive_end_height.as_u64()
        ),
        is_error: false,
    });

    let fixture_temp_guard = TempDirGuard {
        path: fixture_temp_dir,
    };

    let mut runner = BenchRunner::new(SolBenchConfig {
        archive_path,
        kernel_path,
        block_count: options.block_count,
        skip_genesis: options.skip_genesis,
        proof_version: None,
        checkpoint_path,
        start_height: start_height.map(SolHeight),
        profile_memory: options.profile_memory,
        profile_interval_ms: options.profile_interval_ms,
        gc_drop_threshold_bytes: options.gc_drop_threshold_mib.saturating_mul(1024 * 1024),
        page_fault_minor_burst_threshold: options.page_fault_minor_burst_threshold,
        page_fault_major_burst_threshold: options.page_fault_major_burst_threshold,
        checkpoint_every_blocks: options.checkpoint_every_blocks,
        checkpoint_recovery_timeout_ms: options.checkpoint_recovery_timeout_ms,
        checkpoint_recovery_tolerance_pct: options.checkpoint_recovery_tolerance_pct,
        work_dir: PathBuf::from(&options.work_dir),
    });

    let (heartbeat_stop_tx, heartbeat_stop_rx) = oneshot::channel::<()>();
    let heartbeat_tx = tx.clone();
    let heartbeat_running = running.clone();
    let heartbeat_test_id = test_id;
    let heartbeat_container_id = container_id;
    let heartbeat = tokio::spawn(async move {
        let mut stop_rx = heartbeat_stop_rx;
        let mut cancellation_noted = false;
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                _ = tokio::time::sleep(Duration::from_secs(1)) => {
                    let _ = heartbeat_tx.send(RunnerMessage::Progress {
                        test_id: heartbeat_test_id,
                        elapsed_secs: bench_start.elapsed().as_secs_f64(),
                        total_secs: 0.0,
                        sample_count: 0,
                    });
                    if !cancellation_noted && is_test_cancelled(heartbeat_test_id, &heartbeat_running) {
                        cancellation_noted = true;
                        let _ = heartbeat_tx.send(RunnerMessage::Log {
                            test_id: heartbeat_test_id,
                            container_id: heartbeat_container_id,
                            line: "Stop requested. SOL bench will stop after replay completes.".to_string(),
                            is_error: false,
                        });
                    }
                }
            }
        }
    });

    let bench = runner
        .run()
        .await
        .map_err(|e| std::io::Error::other(e.to_string()));
    let _ = heartbeat_stop_tx.send(());
    let _ = heartbeat.await;
    let bench = bench?;

    if let Some(profile) = bench.memory_profile.clone() {
        for attribution in &profile.samples {
            let sample = convert_memory_attribution_to_sample(
                container_id, attribution.timestamp_ms, attribution, &config.metrics,
            );
            result.add_sample(sample.clone());
            let _ = tx.send(RunnerMessage::Sample {
                test_id,
                container_id,
                sample,
            });
        }

        for (idx, checkpoint) in profile.checkpoint_profiles.iter().enumerate() {
            let start = TestEvent::new(
                checkpoint.start_ms,
                container_id,
                "checkpoint-start",
                format!("Checkpoint {} started", idx + 1),
            )
            .significant();
            result.add_event(start.clone());
            let _ = tx.send(RunnerMessage::Event {
                test_id,
                event: start,
            });

            let done = TestEvent::new(
                checkpoint.end_ms,
                container_id,
                "checkpoint-done",
                format!(
                    "Checkpoint {} completed in {}ms",
                    idx + 1,
                    checkpoint.duration_ms
                ),
            )
            .significant();
            result.add_event(done.clone());
            let _ = tx.send(RunnerMessage::Event {
                test_id,
                event: done,
            });
        }

        for gc in &profile.gc_events {
            let event = TestEvent::new(
                gc.end_ms,
                container_id,
                "gc-inferred",
                format!("Inferred GC reclaimed {} bytes", gc.reclaimed_bytes),
            )
            .significant();
            result.add_event(event.clone());
            let _ = tx.send(RunnerMessage::Event { test_id, event });
        }

        for burst in &profile.page_fault_bursts {
            let event = TestEvent::new(
                burst.end_ms,
                container_id,
                "page-fault-burst",
                format!(
                    "Fault burst minor={} major={}",
                    burst.minor_faults_delta, burst.major_faults_delta
                ),
            )
            .significant();
            result.add_event(event.clone());
            let _ = tx.send(RunnerMessage::Event { test_id, event });
        }
    }

    let checkpoint_avg_time_secs = if bench.checkpoint_count == 0 {
        None
    } else {
        Some(bench.checkpoint_total_time.as_secs_f64() / bench.checkpoint_count as f64)
    };

    result.sol_bench = Some(SolBenchResult {
        blocks_poked: bench.blocks_poked,
        failed_pokes: bench.failed_pokes,
        init_time_secs: bench.init_time.as_secs_f64(),
        total_poke_time_secs: bench.total_poke_time.as_secs_f64(),
        blocks_per_second: bench.blocks_per_second(),
        checkpoint_count: bench.checkpoint_count,
        checkpoint_total_time_secs: bench.checkpoint_total_time.as_secs_f64(),
        checkpoint_avg_time_secs,
        memory_profile: bench.memory_profile.clone(),
    });

    let _ = tx.send(RunnerMessage::Progress {
        test_id,
        elapsed_secs: bench_start.elapsed().as_secs_f64(),
        total_secs: 0.0,
        sample_count: result.samples.len(),
    });

    if let Some(path) = &options.profile_output {
        if let Some(payload) = &result.sol_bench {
            std::fs::write(path, serde_json::to_string_pretty(payload)?)?;
        }
    }

    if is_test_cancelled(test_id, &running) {
        let _ = tx.send(RunnerMessage::Log {
            test_id,
            container_id,
            line: "SOL bench cancellation acknowledged".to_string(),
            is_error: false,
        });
        result.cancel();
    } else if result.status == TestStatus::Running {
        result.complete();
    }

    drop(fixture_temp_guard);
    Ok(result)
}

async fn run_sol_fixture_build_async(
    job_id: Uuid,
    options: SolFixtureOptions,
    tx: Sender<RunnerMessage>,
) -> Result<(PathBuf, u64, u64, u64), Box<dyn std::error::Error + Send + Sync>> {
    let source_checkpoint = PathBuf::from(&options.source_checkpoint_path);
    let kernel = PathBuf::from(&options.kernel_path);
    let output_fixture = PathBuf::from(&options.output_fixture);
    let work_dir = PathBuf::from(&options.work_dir);

    let _ = tx.send(RunnerMessage::SolFixtureLog {
        job_id,
        line: format!("Source checkpoint: {}", source_checkpoint.display()),
        is_error: false,
    });
    let _ = tx.send(RunnerMessage::SolFixtureLog {
        job_id,
        line: format!("Kernel: {}", kernel.display()),
        is_error: false,
    });
    let _ = tx.send(RunnerMessage::SolFixtureLog {
        job_id,
        line: format!("Range: {}..={}", options.start_height, options.end_height),
        is_error: false,
    });
    let _ = tx.send(RunnerMessage::SolFixtureLog {
        job_id,
        line: format!("Blocks per fetch: {}", options.chunk_size),
        is_error: false,
    });
    let _ = tx.send(RunnerMessage::SolFixtureLog {
        job_id,
        line: format!(
            "Mempool snapshots: {}",
            if options.include_mempool {
                "enabled"
            } else {
                "disabled"
            }
        ),
        is_error: false,
    });
    let _ = tx.send(RunnerMessage::SolFixtureLog {
        job_id,
        line: format!("Output fixture: {}", output_fixture.display()),
        is_error: false,
    });

    if !source_checkpoint.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Source checkpoint file not found: {}",
                source_checkpoint.display()
            ),
        )
        .into());
    }
    if !kernel.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Kernel file not found: {}", kernel.display()),
        )
        .into());
    }
    std::fs::create_dir_all(&work_dir)?;

    let progress_tx = tx.clone();
    let builder = FixtureBuilder::new(FixtureBuildConfig {
        source_checkpoint_path: source_checkpoint,
        kernel_path: kernel,
        start_height: SolHeight(options.start_height),
        end_height: SolHeight(options.end_height),
        output_path: output_fixture.clone(),
        work_dir,
        chunk_size: options.chunk_size,
        include_mempool: options.include_mempool,
    });

    let result = builder
        .run_with_progress(|progress: FixtureBuildProgress| {
            let _ = progress_tx.send(RunnerMessage::SolFixtureProgress {
                job_id,
                phase: progress.phase,
                blocks_done: progress.blocks_done,
                blocks_total: progress.blocks_total,
                message: progress.message,
            });
        })
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    let _ = tx.send(RunnerMessage::SolFixtureProgress {
        job_id,
        phase: FixtureBuildPhase::Packaging,
        blocks_done: None,
        blocks_total: None,
        message: "Fixture build complete".to_string(),
    });

    Ok((
        result.output_path,
        result.derived_checkpoint_height.as_u64(),
        result.archive_start_height.as_u64(),
        result.archive_end_height.as_u64(),
    ))
}

async fn run_sol_sweep_test_async(
    test_id: Uuid,
    config: TestConfig,
    tx: Sender<RunnerMessage>,
    running: Arc<Mutex<HashMap<Uuid, RunningTest>>>,
) -> Result<TestResult, Box<dyn std::error::Error + Send + Sync>> {
    let mut result = TestResult::new(config.clone());
    let options = &config.sol_sweep;
    let data_root = resolve_sol_sweep_data_root(&options.data_dir);
    if data_root.to_string_lossy() != options.data_dir {
        let _ = tx.send(RunnerMessage::Log {
            test_id,
            container_id: Uuid::nil(),
            line: format!(
                "SOL sweep data dir '{}' is legacy /tmp path; using '{}' for Docker compatibility",
                options.data_dir,
                data_root.display()
            ),
            is_error: false,
        });
    }
    std::fs::create_dir_all(&data_root)?;
    let candidates = options.candidates();
    let chunk_sizes = options.chunk_sizes().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("invalid chunk sizes: {e}"),
        )
    })?;
    let memory_limits = options.memory_limits();
    let cases = build_sweep_cases(&candidates, &chunk_sizes, &memory_limits);
    let total_runs = cases.len().saturating_mul(options.repeats as usize);
    let mut runs_completed = 0usize;
    let sweep_start = Instant::now();
    let estimated_total_secs = (total_runs as f64) * (options.duration_secs as f64);

    let _ = tx.send(RunnerMessage::Log {
        test_id,
        container_id: Uuid::nil(),
        line: format!("Starting SOL sweep with {} case(s)", cases.len()),
        is_error: false,
    });
    let _ = tx.send(RunnerMessage::Progress {
        test_id,
        elapsed_secs: 0.0,
        total_secs: estimated_total_secs,
        sample_count: 0,
    });

    let mut runs = Vec::<SweepRunMetrics>::new();
    let mut cancel_requested = false;

    'matrix: for (case_index, case) in cases.iter().enumerate() {
        for run_index in 0..options.repeats {
            if is_test_cancelled(test_id, &running) {
                cancel_requested = true;
                break 'matrix;
            }

            let run_dir = data_root.join(format!(
                "cand-{}-chunk-{}-mem-{}-run-{}",
                sanitize_case_value(&case.candidate),
                case.chunk_size,
                sanitize_case_value(&case.memory_limit),
                run_index + 1
            ));
            std::fs::create_dir_all(&run_dir)?;

            let mut env_vars = HashMap::new();
            env_vars.insert(options.candidate_env.clone(), case.candidate.clone());
            env_vars.insert(options.chunk_env.clone(), case.chunk_size.to_string());

            let scenario = MiningScenario::new(MiningScenarioConfig {
                name: format!(
                    "sol-sweep-{}-chunk{}-mem{}-run{}",
                    sanitize_case_value(&case.candidate),
                    case.chunk_size,
                    sanitize_case_value(&case.memory_limit),
                    run_index + 1
                ),
                mode: NockchainMode::Checkpoint {
                    save_interval_secs: options.save_interval_secs,
                },
                duration: Duration::from_secs(options.duration_secs),
                sample_interval: Duration::from_secs(options.sample_interval_secs),
                image: options.image.clone(),
                data_dir: run_dir.clone(),
                memory_limit: Some(case.memory_limit.clone()),
                num_threads: options.threads,
                env_vars,
                ..Default::default()
            });

            let _ = tx.send(RunnerMessage::Log {
                test_id,
                container_id: Uuid::nil(),
                line: format!(
                    "Case {}/{} run {}/{}: candidate={} chunk={} memory={}",
                    case_index + 1,
                    cases.len(),
                    run_index + 1,
                    options.repeats,
                    case.candidate,
                    case.chunk_size,
                    case.memory_limit
                ),
                is_error: false,
            });
            let _ = tx.send(RunnerMessage::Progress {
                test_id,
                elapsed_secs: sweep_start.elapsed().as_secs_f64(),
                total_secs: estimated_total_secs,
                sample_count: runs_completed,
            });

            let scenario_result = scenario
                .run()
                .await
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let mut parser = LogParser::new();
            let events = parser.parse_lines(&scenario_result.final_logs);
            let checkpoint_durations = checkpoint_durations_ms(&events);
            let checkpoint_count = checkpoint_durations.len() as u64;
            let checkpoint_avg_duration_s = if checkpoint_durations.is_empty() {
                None
            } else {
                Some(
                    checkpoint_durations.iter().sum::<u64>() as f64
                        / checkpoint_durations.len() as f64
                        / 1000.0,
                )
            };
            let checkpoint_size = latest_checkpoint_size_in_dir(&run_dir)?;
            let checkpoint_mib_per_s = match (checkpoint_size, checkpoint_avg_duration_s) {
                (Some(size_bytes), Some(avg_secs)) if avg_secs > 0.0 => {
                    Some((size_bytes as f64 / 1024.0 / 1024.0) / avg_secs)
                }
                _ => None,
            };
            let (fault_bursts, minor_total, major_total) =
                match page_fault_bursts(&scenario_result.samples, 50_000, 1) {
                    Some((bursts, minor, major)) => (Some(bursts), Some(minor), Some(major)),
                    None => (None, None, None),
                };

            runs.push(SweepRunMetrics {
                case: case.clone(),
                run_index,
                peak_rss_mib: scenario_result.peak_rss_mib(),
                avg_rss_mib: scenario_result.avg_rss_mib(),
                checkpoint_count,
                checkpoint_avg_duration_s,
                checkpoint_mib_per_s,
                page_fault_bursts: fault_bursts,
                minor_faults_delta_total: minor_total,
                major_faults_delta_total: major_total,
            });
            runs_completed += 1;
            let _ = tx.send(RunnerMessage::Progress {
                test_id,
                elapsed_secs: sweep_start.elapsed().as_secs_f64(),
                total_secs: estimated_total_secs,
                sample_count: runs_completed,
            });

            if is_test_cancelled(test_id, &running) {
                cancel_requested = true;
                break 'matrix;
            }
        }
    }

    let mut summaries = Vec::new();
    for case in &cases {
        let case_runs: Vec<SweepRunMetrics> = runs
            .iter()
            .filter(|run| {
                run.case.candidate == case.candidate
                    && run.case.chunk_size == case.chunk_size
                    && run.case.memory_limit == case.memory_limit
            })
            .cloned()
            .collect();
        if case_runs.is_empty() {
            continue;
        }
        summaries.push(summarize_case_runs(case, &case_runs));
    }

    result.sol_sweep = Some(SolSweepResult {
        runs: runs.clone(),
        summaries: summaries.clone(),
    });

    if let Some(path) = &options.output_json {
        let payload = serde_json::json!({
            "cases": cases,
            "runs": runs,
            "summaries": summaries,
            "config": options,
        });
        std::fs::write(path, serde_json::to_string_pretty(&payload)?)?;
    }

    if cancel_requested {
        let _ = tx.send(RunnerMessage::Log {
            test_id,
            container_id: Uuid::nil(),
            line: format!("SOL sweep cancelled after {runs_completed}/{total_runs} runs"),
            is_error: false,
        });
        result.cancel();
    } else if result.status == TestStatus::Running {
        result.complete();
    }

    Ok(result)
}

fn is_test_cancelled(test_id: Uuid, running: &Arc<Mutex<HashMap<Uuid, RunningTest>>>) -> bool {
    let tests = running.lock().unwrap();
    tests
        .get(&test_id)
        .map(|test| test.cancelled)
        .unwrap_or(false)
}

fn latest_checkpoint_size_in_dir(dir: &PathBuf) -> Result<Option<u64>, std::io::Error> {
    let mut latest: Option<(std::time::SystemTime, u64)> = None;
    for name in ["0.chkjam", "1.chkjam"] {
        let path = dir.join(name);
        if !path.exists() {
            continue;
        }
        let metadata = std::fs::metadata(path)?;
        let modified = metadata
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        let size = metadata.len();
        match latest {
            Some((current_modified, _)) if modified <= current_modified => {}
            _ => latest = Some((modified, size)),
        }
    }
    Ok(latest.map(|(_, size)| size))
}

/// Get the base directory for benchmark data
/// Uses ~/.nockchain-bench-data since /tmp may not be shared with Docker Desktop
fn get_bench_data_dir() -> String {
    if let Some(home) = dirs::home_dir() {
        home.join(".nockchain-bench-data")
            .to_string_lossy()
            .to_string()
    } else {
        // Fallback to /tmp if home dir not available
        "/tmp/nockchain-bench-data".to_string()
    }
}

/// Resolve SOL sweep root directory to a Docker-compatible host path.
///
/// Historically this used `/tmp/nockchain-bench-sweep`, but Docker Desktop setups
/// often cannot bind-mount from `/tmp`. We transparently remap that legacy default
/// to `~/.nockchain-bench-data/sweep`.
fn resolve_sol_sweep_data_root(configured: &str) -> PathBuf {
    let trimmed = configured.trim();
    if trimmed == "/tmp/nockchain-bench-sweep" {
        return PathBuf::from(get_bench_data_dir()).join("sweep");
    }
    PathBuf::from(trimmed)
}

/// Convert our ContainerConfig to nockchain-bench's DockerRunnerConfig
/// The `port_offset` is added to base port 30000 to give each container a unique port
fn convert_container_config(config: &ContainerConfig, port_offset: u16) -> DockerRunnerConfig {
    let mode = match config.persistence_mode {
        PersistenceMode::Checkpoint => NockchainMode::Checkpoint {
            save_interval_secs: config.checkpoint_interval_secs,
        },
        PersistenceMode::PmaPersist => NockchainMode::PmaPersist,
    };

    let env_vars: HashMap<String, String> = config.env_vars.iter().cloned().collect();

    // Use home directory instead of /tmp for Docker Desktop compatibility
    let base_dir = get_bench_data_dir();
    let data_dir = format!("{}/{}", base_dir, config.id);

    DockerRunnerConfig {
        image: config.image.clone(),
        container_name: format!(
            "bench-{}-{}",
            sanitize_name(&config.name),
            &config.id.to_string()[..8]
        ),
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
        bind_port: 30000 + port_offset,
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
            MetricType::MinorFaults => stats.minor_faults.map(|value| value as f64).unwrap_or(0.0),
            MetricType::MajorFaults => stats.major_faults.map(|value| value as f64).unwrap_or(0.0),
            // For now, we only have Docker stats; proc-based metrics would need pid access
            _ => continue,
        };

        sample.values.insert(*metric, value);
    }

    sample
}

/// Convert a SOL memory-attribution sample to GUI sample metrics
fn convert_memory_attribution_to_sample(
    container_id: Uuid,
    timestamp_ms: u64,
    attribution: &MemoryAttribution,
    metrics: &[MetricType],
) -> DataSample {
    let mut sample = DataSample::new(timestamp_ms, container_id);

    for metric in metrics {
        let value = match metric {
            MetricType::VmRss => attribution.vm_rss_kb as f64,
            MetricType::VmSize => attribution.vm_size_kb as f64,
            MetricType::RssAnon => attribution.rss_anon_kb as f64,
            MetricType::RssFile => attribution.rss_file_kb as f64,
            MetricType::NockStackRss => attribution.nockstack_rss_kb as f64,
            MetricType::PmaRss => attribution.pma_rss_kb as f64,
            MetricType::PmaSize => attribution.pma_size_kb as f64,
            MetricType::HeapOtherRss => attribution.heap_other_rss_kb as f64,
            MetricType::MinorFaults => attribution.minor_faults as f64,
            MetricType::MajorFaults => attribution.major_faults as f64,
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

fn sanitize_case_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch.to_ascii_lowercase()
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
    pub fn start_test(
        &self,
        test_id: Uuid,
        config: TestConfig,
    ) -> Result<(), crossbeam_channel::SendError<RunnerCommand>> {
        self.cmd_tx.send(RunnerCommand::Start { test_id, config })
    }

    /// Stop a test
    pub fn stop_test(
        &self,
        test_id: Uuid,
    ) -> Result<(), crossbeam_channel::SendError<RunnerCommand>> {
        self.cmd_tx.send(RunnerCommand::Stop(test_id))
    }

    /// Check Docker availability
    pub fn check_docker(&self) -> Result<(), crossbeam_channel::SendError<RunnerCommand>> {
        self.cmd_tx.send(RunnerCommand::CheckDocker)
    }

    /// Start a SOL fixture creation job
    pub fn create_sol_fixture(
        &self,
        job_id: Uuid,
        options: SolFixtureOptions,
    ) -> Result<(), crossbeam_channel::SendError<RunnerCommand>> {
        self.cmd_tx
            .send(RunnerCommand::CreateSolFixture { job_id, options })
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
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

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
        let docker_config = convert_container_config(&config, 0);

        assert!(docker_config.container_name.starts_with("bench-test-"));
        assert!(matches!(
            docker_config.mode,
            NockchainMode::Checkpoint {
                save_interval_secs: 60
            }
        ));
        assert_eq!(docker_config.bind_port, 30000);
    }

    #[test]
    fn test_convert_container_config_pma() {
        let config = ContainerConfig::pma_persist("Test PMA");
        let docker_config = convert_container_config(&config, 1);

        assert!(matches!(docker_config.mode, NockchainMode::PmaPersist));
        assert_eq!(docker_config.bind_port, 30001); // Second container gets different port
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

    #[test]
    fn test_runner_handle_create_sol_fixture_command() {
        let (_msg_tx, msg_rx) = bounded(1);
        let (cmd_tx, cmd_rx) = bounded(1);
        let handle = RunnerHandle::new(cmd_tx, msg_rx);
        let job_id = Uuid::new_v4();

        handle
            .create_sol_fixture(job_id, SolFixtureOptions::default())
            .expect("command should send");

        let cmd = cmd_rx.recv().expect("command should be queued");
        assert!(matches!(
            cmd,
            RunnerCommand::CreateSolFixture { job_id: queued, .. } if queued == job_id
        ));
    }

    #[test]
    fn test_sanitize_case_value() {
        assert_eq!(sanitize_case_value("alpha_beta"), "alpha_beta");
        assert_eq!(sanitize_case_value("A/B C"), "a-b-c");
    }

    #[test]
    fn test_resolve_sol_sweep_data_root_legacy_tmp() {
        let resolved = resolve_sol_sweep_data_root("/tmp/nockchain-bench-sweep");
        assert!(resolved.to_string_lossy().contains(".nockchain-bench-data"));
        assert!(resolved.ends_with("sweep"));
    }

    #[test]
    fn test_convert_stats_to_sample_fault_metrics() {
        let container_id = Uuid::new_v4();
        let stats = ContainerStats {
            timestamp_ms: 100,
            memory_usage_bytes: 2048 * 1024,
            memory_limit_bytes: 0,
            memory_percent: 0.0,
            memory_cache_bytes: 512 * 1024,
            memory_rss_bytes: 1024 * 1024,
            cpu_percent: 42.0,
            minor_faults: Some(123),
            major_faults: Some(4),
        };
        let sample = convert_stats_to_sample(
            container_id,
            100,
            &stats,
            &[
                MetricType::ContainerMemory,
                MetricType::ContainerRss,
                MetricType::MinorFaults,
                MetricType::MajorFaults,
            ],
        );
        assert_eq!(sample.get(MetricType::ContainerMemory), Some(2048.0));
        assert_eq!(sample.get(MetricType::ContainerRss), Some(1024.0));
        assert_eq!(sample.get(MetricType::MinorFaults), Some(123.0));
        assert_eq!(sample.get(MetricType::MajorFaults), Some(4.0));
    }

    #[test]
    fn test_convert_memory_attribution_to_sample() {
        let container_id = Uuid::new_v4();
        let attribution = MemoryAttribution {
            timestamp_ms: 77,
            vm_rss_kb: 1024,
            vm_size_kb: 4096,
            rss_anon_kb: 700,
            rss_file_kb: 324,
            nockstack_rss_kb: 111,
            pma_rss_kb: 222,
            pma_size_kb: 333,
            heap_other_rss_kb: 444,
            minor_faults: 55,
            major_faults: 2,
            ..Default::default()
        };
        let sample = convert_memory_attribution_to_sample(
            container_id,
            77,
            &attribution,
            &[
                MetricType::VmRss,
                MetricType::PmaRss,
                MetricType::NockStackRss,
                MetricType::MinorFaults,
                MetricType::MajorFaults,
            ],
        );
        assert_eq!(sample.get(MetricType::VmRss), Some(1024.0));
        assert_eq!(sample.get(MetricType::PmaRss), Some(222.0));
        assert_eq!(sample.get(MetricType::NockStackRss), Some(111.0));
        assert_eq!(sample.get(MetricType::MinorFaults), Some(55.0));
        assert_eq!(sample.get(MetricType::MajorFaults), Some(2.0));
    }

    #[test]
    fn test_is_test_cancelled() {
        let test_id = Uuid::new_v4();
        let mut map = HashMap::new();
        map.insert(
            test_id,
            RunningTest {
                config: TestConfig::default(),
                result: TestResult::new(TestConfig::default()),
                start_time: Instant::now(),
                cancelled: true,
            },
        );
        let running = Arc::new(Mutex::new(map));
        assert!(is_test_cancelled(test_id, &running));
    }
}
