//! Configuration types for the GUI
//!
//! Defines the data structures for container configuration, test configuration,
//! and metrics selection.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Persistence mode for Nockchain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PersistenceMode {
    /// Standard checkpointing mode with periodic saves
    #[default]
    Checkpoint,
    /// PMA persistence mode (no explicit checkpoints)
    PmaPersist,
}

impl PersistenceMode {
    /// Get a human-readable label for the mode
    pub fn label(&self) -> &'static str {
        match self {
            PersistenceMode::Checkpoint => "Checkpoint",
            PersistenceMode::PmaPersist => "PMA Persist",
        }
    }

    /// Get all available modes
    pub fn all() -> &'static [PersistenceMode] {
        &[PersistenceMode::Checkpoint, PersistenceMode::PmaPersist]
    }
}

/// Benchmark execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BenchmarkMode {
    /// Run live Docker containers and sample stats continuously
    #[default]
    Container,
    /// Run speed-of-light replay benchmark
    SpeedOfLightBench,
    /// Run speed-of-light candidate/chunk-size sweep
    SpeedOfLightSweep,
}

impl BenchmarkMode {
    /// Human-readable label for UI
    pub fn label(&self) -> &'static str {
        match self {
            BenchmarkMode::Container => "Container",
            BenchmarkMode::SpeedOfLightBench => "SOL Bench",
            BenchmarkMode::SpeedOfLightSweep => "SOL Sweep",
        }
    }

    /// Inline help copy shown in the New Test form.
    pub fn inline_help(&self) -> &'static str {
        match self {
            BenchmarkMode::Container => {
                "Run one or more Docker containers and sample live CPU/memory/fault metrics over time."
            }
            BenchmarkMode::SpeedOfLightBench => {
                "Replay one .soltest fixture to profile speed, memory behavior, and checkpoint recovery."
            }
            BenchmarkMode::SpeedOfLightSweep => {
                "Run a matrix of SOL runs across PMA candidate, chunk size, and memory limit settings for side-by-side comparison."
            }
        }
    }

    /// All benchmark modes
    pub fn all() -> &'static [BenchmarkMode] {
        &[BenchmarkMode::Container, BenchmarkMode::SpeedOfLightBench]
    }
}

/// Proof-version filter used by SOL benchmark replays
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolProofVersion {
    V0,
    V1,
    V2,
}

impl SolProofVersion {
    pub fn label(&self) -> &'static str {
        match self {
            SolProofVersion::V0 => "v0",
            SolProofVersion::V1 => "v1",
            SolProofVersion::V2 => "v2",
        }
    }

    pub fn all() -> &'static [SolProofVersion] {
        &[SolProofVersion::V0, SolProofVersion::V1, SolProofVersion::V2]
    }
}

/// Options for speed-of-light benchmark runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolBenchOptions {
    /// Optional unified fixture path (`.soltest`) containing archive+checkpoint+kernel
    pub fixture_path: Option<String>,
    /// Number of blocks to replay (0 = all)
    pub block_count: u64,
    /// Skip genesis block (generally false)
    pub skip_genesis: bool,
    /// Optional proof version filter
    pub proof_version: Option<SolProofVersion>,
    /// Enable memory profile timeline
    pub profile_memory: bool,
    /// Memory profile interval (ms)
    pub profile_interval_ms: u64,
    /// Optional profile output JSON path
    pub profile_output: Option<String>,
    /// Force checkpoint every N accepted blocks (0 = off)
    pub checkpoint_every_blocks: u64,
    /// Max wait for post-checkpoint recovery (ms)
    pub checkpoint_recovery_timeout_ms: u64,
    /// Recovery tolerance percent above baseline RSS
    pub checkpoint_recovery_tolerance_pct: f64,
    /// GC inference threshold in MiB
    pub gc_drop_threshold_mib: u64,
    /// Minor fault burst threshold
    pub page_fault_minor_burst_threshold: u64,
    /// Major fault burst threshold
    pub page_fault_major_burst_threshold: u64,
    /// Working directory for generated checkpoints
    pub work_dir: String,
}

impl Default for SolBenchOptions {
    fn default() -> Self {
        Self {
            fixture_path: None,
            block_count: 0,
            skip_genesis: false,
            proof_version: None,
            profile_memory: true,
            profile_interval_ms: 500,
            profile_output: None,
            checkpoint_every_blocks: 0,
            checkpoint_recovery_timeout_ms: 5000,
            checkpoint_recovery_tolerance_pct: 5.0,
            gc_drop_threshold_mib: 64,
            page_fault_minor_burst_threshold: 50_000,
            page_fault_major_burst_threshold: 1,
            work_dir: ".".to_string(),
        }
    }
}

impl SolBenchOptions {
    /// Validate option values that don't require touching the filesystem
    pub fn validate(&self) -> Result<(), String> {
        let fixture_path = self
            .fixture_path
            .as_ref()
            .map(|path| path.trim())
            .filter(|path| !path.is_empty());
        if fixture_path.is_none() {
            return Err("SOL fixture path cannot be empty".to_string());
        }
        if self.profile_interval_ms == 0 {
            return Err("SOL profile interval must be greater than 0ms".to_string());
        }
        if self.checkpoint_recovery_timeout_ms == 0 {
            return Err("SOL checkpoint recovery timeout must be greater than 0ms".to_string());
        }
        if self.checkpoint_recovery_tolerance_pct < 0.0 {
            return Err("SOL checkpoint recovery tolerance must be >= 0".to_string());
        }
        if self.gc_drop_threshold_mib == 0 {
            return Err("SOL GC drop threshold must be greater than 0 MiB".to_string());
        }
        if self.page_fault_minor_burst_threshold == 0 {
            return Err("SOL minor fault burst threshold must be greater than 0".to_string());
        }
        if self.page_fault_major_burst_threshold == 0 {
            return Err("SOL major fault burst threshold must be greater than 0".to_string());
        }
        if self.work_dir.trim().is_empty() {
            return Err("SOL work directory cannot be empty".to_string());
        }
        Ok(())
    }
}

/// Options for building unified SOL fixtures (`.soltest`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolFixtureOptions {
    /// Source checkpoint path (must contain target range history)
    pub source_checkpoint_path: String,
    /// Path to kernel jam
    pub kernel_path: String,
    /// Start block height for test archive (inclusive)
    pub start_height: u64,
    /// End block height for test archive (inclusive)
    pub end_height: u64,
    /// Output fixture path
    pub output_fixture: String,
    /// Chunk size for extraction fetches
    pub chunk_size: u64,
    /// Include mempool snapshots in test archive
    pub include_mempool: bool,
    /// Working directory for temporary artifacts
    pub work_dir: String,
}

impl Default for SolFixtureOptions {
    fn default() -> Self {
        Self {
            source_checkpoint_path: "0.chkjam".to_string(),
            kernel_path: "assets/dumb.jam".to_string(),
            start_height: 50_000,
            end_height: 60_000,
            output_fixture: "sol_fixture_50000_60000.soltest".to_string(),
            chunk_size: 8,
            include_mempool: false,
            work_dir: ".".to_string(),
        }
    }
}

impl SolFixtureOptions {
    /// Validate option values that don't require touching the filesystem
    pub fn validate(&self) -> Result<(), String> {
        if self.source_checkpoint_path.trim().is_empty() {
            return Err("SOL fixture source checkpoint path cannot be empty".to_string());
        }
        if self.kernel_path.trim().is_empty() {
            return Err("SOL fixture kernel path cannot be empty".to_string());
        }
        if self.start_height == 0 {
            return Err(
                "SOL fixture start height must be greater than 0 (needs start-1 checkpoint)"
                    .to_string(),
            );
        }
        if self.start_height > self.end_height {
            return Err("SOL fixture start height must be <= end height".to_string());
        }
        if self.output_fixture.trim().is_empty() {
            return Err("SOL fixture output path cannot be empty".to_string());
        }
        if self.chunk_size == 0 {
            return Err("SOL fixture chunk size must be greater than 0".to_string());
        }
        if self.work_dir.trim().is_empty() {
            return Err("SOL fixture work directory cannot be empty".to_string());
        }
        Ok(())
    }
}

/// Options for SOL PMA/chunk-size matrix sweeps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolSweepOptions {
    /// Candidate IDs CSV
    pub candidates_csv: String,
    /// Chunk sizes CSV
    pub chunk_sizes_csv: String,
    /// Memory limits CSV
    pub memory_limits_csv: String,
    /// Repetitions per case
    pub repeats: u32,
    /// Duration per run (seconds)
    pub duration_secs: u64,
    /// Sample interval (seconds)
    pub sample_interval_secs: u64,
    /// Checkpoint save interval (seconds)
    pub save_interval_secs: u64,
    /// Docker image
    pub image: String,
    /// Base directory for run data
    pub data_dir: String,
    /// Mining threads
    pub threads: u32,
    /// Candidate selector env var
    pub candidate_env: String,
    /// Chunk-size selector env var
    pub chunk_env: String,
    /// Optional summary JSON output path
    pub output_json: Option<String>,
}

impl Default for SolSweepOptions {
    fn default() -> Self {
        Self {
            candidates_csv: "baseline".to_string(),
            chunk_sizes_csv: "8".to_string(),
            memory_limits_csv: "16g".to_string(),
            repeats: 1,
            duration_secs: 300,
            sample_interval_secs: 1,
            save_interval_secs: 120,
            image: "nockchain-local:latest".to_string(),
            data_dir: default_sol_sweep_data_dir(),
            threads: 1,
            candidate_env: "NOCK_PMA_CANDIDATE".to_string(),
            chunk_env: "NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE".to_string(),
            output_json: None,
        }
    }
}

fn default_sol_sweep_data_dir() -> String {
    if let Some(home) = dirs::home_dir() {
        home.join(".nockchain-bench-data")
            .join("sweep")
            .to_string_lossy()
            .to_string()
    } else {
        "/tmp/nockchain-bench-sweep".to_string()
    }
}

impl SolSweepOptions {
    pub fn candidates(&self) -> Vec<String> {
        parse_csv_strings(&self.candidates_csv)
    }

    pub fn chunk_sizes(&self) -> Result<Vec<u64>, String> {
        parse_csv_u64(&self.chunk_sizes_csv)
    }

    pub fn memory_limits(&self) -> Vec<String> {
        parse_csv_strings(&self.memory_limits_csv)
    }

    pub fn case_count(&self) -> Result<usize, String> {
        let candidates = self.candidates();
        let chunk_sizes = self.chunk_sizes()?;
        let memory_limits = self.memory_limits();
        Ok(candidates.len() * chunk_sizes.len() * memory_limits.len())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.candidates().is_empty() {
            return Err("SOL sweep requires at least one candidate".to_string());
        }
        if self.chunk_sizes()?.is_empty() {
            return Err("SOL sweep requires at least one chunk size".to_string());
        }
        if self.memory_limits().is_empty() {
            return Err("SOL sweep requires at least one memory limit".to_string());
        }
        if self.repeats == 0 {
            return Err("SOL sweep repeats must be greater than 0".to_string());
        }
        if self.duration_secs == 0 {
            return Err("SOL sweep duration must be greater than 0".to_string());
        }
        if self.sample_interval_secs == 0 {
            return Err("SOL sweep sample interval must be greater than 0".to_string());
        }
        if self.save_interval_secs == 0 {
            return Err("SOL sweep save interval must be greater than 0".to_string());
        }
        if self.image.trim().is_empty() {
            return Err("SOL sweep image cannot be empty".to_string());
        }
        if self.data_dir.trim().is_empty() {
            return Err("SOL sweep data directory cannot be empty".to_string());
        }
        if self.candidate_env.trim().is_empty() {
            return Err("SOL sweep candidate env var cannot be empty".to_string());
        }
        if self.chunk_env.trim().is_empty() {
            return Err("SOL sweep chunk env var cannot be empty".to_string());
        }
        Ok(())
    }
}

/// Configuration for a Docker container running Nockchain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerConfig {
    /// Unique identifier for this configuration
    pub id: Uuid,

    /// Human-readable name for this configuration
    pub name: String,

    /// Docker image to use
    pub image: String,

    /// Git branch or commit to use for building (optional)
    pub git_ref: Option<String>,

    /// Persistence mode
    pub persistence_mode: PersistenceMode,

    /// Checkpoint interval in seconds (only used if persistence_mode is Checkpoint)
    pub checkpoint_interval_secs: u64,

    /// Memory limit for the container (e.g., "16g", "8g")
    pub memory_limit: String,

    /// Number of mining threads (0 = auto)
    pub num_threads: u32,

    /// Enable mining
    pub enable_mining: bool,

    /// Use fakenet (local testing mode)
    pub use_fakenet: bool,

    /// Enable fast sync
    pub enable_fast_sync: bool,

    /// Additional environment variables
    pub env_vars: Vec<(String, String)>,

    /// Additional CLI arguments
    pub extra_args: Vec<String>,
}

impl Default for ContainerConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "New Container".to_string(),
            image: "nockchain-local:latest".to_string(),
            git_ref: None,
            persistence_mode: PersistenceMode::Checkpoint,
            checkpoint_interval_secs: 120,
            memory_limit: "16g".to_string(),
            num_threads: 1,
            enable_mining: true,
            use_fakenet: true,
            enable_fast_sync: true,
            env_vars: Vec::new(),
            extra_args: Vec::new(),
        }
    }
}

impl ContainerConfig {
    /// Create a new container config with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Create a checkpoint mode config
    pub fn checkpoint(name: impl Into<String>, interval_secs: u64) -> Self {
        Self {
            name: name.into(),
            persistence_mode: PersistenceMode::Checkpoint,
            checkpoint_interval_secs: interval_secs,
            ..Default::default()
        }
    }

    /// Create a PMA persist mode config
    pub fn pma_persist(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            persistence_mode: PersistenceMode::PmaPersist,
            ..Default::default()
        }
    }

    /// Set the memory limit
    pub fn with_memory_limit(mut self, limit: impl Into<String>) -> Self {
        self.memory_limit = limit.into();
        self
    }

    /// Set the number of threads
    pub fn with_threads(mut self, threads: u32) -> Self {
        self.num_threads = threads;
        self
    }

    /// Enable or disable mining
    pub fn with_mining(mut self, enable: bool) -> Self {
        self.enable_mining = enable;
        self
    }

    /// Get a summary description of this config
    pub fn summary(&self) -> String {
        let mode = match self.persistence_mode {
            PersistenceMode::Checkpoint => {
                format!("Checkpoint ({}s)", self.checkpoint_interval_secs)
            }
            PersistenceMode::PmaPersist => "PMA Persist".to_string(),
        };
        format!(
            "{} | {} | {} RAM | {} threads",
            self.name, mode, self.memory_limit, self.num_threads
        )
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Container name cannot be empty".to_string());
        }
        if self.image.is_empty() {
            return Err("Docker image cannot be empty".to_string());
        }
        if self.memory_limit.is_empty() {
            return Err("Memory limit cannot be empty".to_string());
        }
        // Parse memory limit to validate format
        if !parse_memory_limit(&self.memory_limit).is_some() {
            return Err(format!(
                "Invalid memory limit format: '{}'. Use format like '16g', '8g', '512m'",
                self.memory_limit
            ));
        }
        Ok(())
    }
}

/// Parse a memory limit string to bytes
fn parse_memory_limit(s: &str) -> Option<u64> {
    let s = s.trim().to_lowercase();
    if let Some(num) = s.strip_suffix('g') {
        num.parse::<u64>().ok().map(|n| n * 1024 * 1024 * 1024)
    } else if let Some(num) = s.strip_suffix('m') {
        num.parse::<u64>().ok().map(|n| n * 1024 * 1024)
    } else if let Some(num) = s.strip_suffix('k') {
        num.parse::<u64>().ok().map(|n| n * 1024)
    } else {
        s.parse::<u64>().ok()
    }
}

/// Parse comma-separated strings, trimming and removing empties
fn parse_csv_strings(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

/// Parse comma-separated u64 values
fn parse_csv_u64(input: &str) -> Result<Vec<u64>, String> {
    let mut values = Vec::new();
    for token in input
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let parsed = token
            .parse::<u64>()
            .map_err(|e| format!("Invalid u64 '{token}': {e}"))?;
        values.push(parsed);
    }
    Ok(values)
}

/// Types of metrics that can be tracked during a test
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    /// Total VM RSS (resident set size)
    VmRss,
    /// Total VM size (virtual memory)
    VmSize,
    /// Anonymous RSS (heap + stack + anonymous mappings)
    RssAnon,
    /// File-backed RSS
    RssFile,
    /// NockStack RSS
    NockStackRss,
    /// PMA RSS
    PmaRss,
    /// PMA mapped size
    PmaSize,
    /// Heap and other anonymous memory RSS
    HeapOtherRss,
    /// Container memory usage (from Docker stats)
    ContainerMemory,
    /// Container RSS (from Docker stats)
    ContainerRss,
    /// Container cache memory
    ContainerCache,
    /// CPU usage percentage
    CpuPercent,
    /// Minor page faults
    MinorFaults,
    /// Major page faults
    MajorFaults,
}

impl MetricType {
    /// Get a human-readable label for the metric
    pub fn label(&self) -> &'static str {
        match self {
            MetricType::VmRss => "VM RSS",
            MetricType::VmSize => "VM Size",
            MetricType::RssAnon => "RSS Anon",
            MetricType::RssFile => "RSS File",
            MetricType::NockStackRss => "NockStack RSS",
            MetricType::PmaRss => "PMA RSS",
            MetricType::PmaSize => "PMA Size",
            MetricType::HeapOtherRss => "Heap/Other RSS",
            MetricType::ContainerMemory => "Container Memory",
            MetricType::ContainerRss => "Container RSS",
            MetricType::ContainerCache => "Container Cache",
            MetricType::CpuPercent => "CPU %",
            MetricType::MinorFaults => "Minor Faults",
            MetricType::MajorFaults => "Major Faults",
        }
    }

    /// Get a short description of the metric
    pub fn description(&self) -> &'static str {
        match self {
            MetricType::VmRss => "Total resident set size from /proc/status",
            MetricType::VmSize => "Total virtual memory size from /proc/status",
            MetricType::RssAnon => "Anonymous memory (heap, stack, anon mappings)",
            MetricType::RssFile => "File-backed memory (shared libs, mmap files)",
            MetricType::NockStackRss => "NockStack memory (Nock computation stack)",
            MetricType::PmaRss => "PMA resident memory (persistent arena)",
            MetricType::PmaSize => "PMA total mapped size",
            MetricType::HeapOtherRss => "Heap and other anonymous memory",
            MetricType::ContainerMemory => "Total container memory from Docker stats",
            MetricType::ContainerRss => "Container RSS from Docker stats",
            MetricType::ContainerCache => "Container cache memory (reclaimable)",
            MetricType::CpuPercent => "CPU usage percentage",
            MetricType::MinorFaults => "Minor page faults (no disk I/O)",
            MetricType::MajorFaults => "Major page faults (required disk I/O)",
        }
    }

    /// Get all available metric types
    pub fn all() -> &'static [MetricType] {
        &[
            MetricType::VmRss,
            MetricType::VmSize,
            MetricType::RssAnon,
            MetricType::RssFile,
            MetricType::NockStackRss,
            MetricType::PmaRss,
            MetricType::PmaSize,
            MetricType::HeapOtherRss,
            MetricType::ContainerMemory,
            MetricType::ContainerRss,
            MetricType::ContainerCache,
            MetricType::CpuPercent,
            MetricType::MinorFaults,
            MetricType::MajorFaults,
        ]
    }

    /// Get the default set of metrics to track
    pub fn defaults() -> Vec<MetricType> {
        vec![
            MetricType::VmRss,
            MetricType::ContainerMemory,
            MetricType::ContainerRss,
            MetricType::CpuPercent,
        ]
    }

    /// Default metrics for SOL benchmark timelines
    pub fn sol_defaults() -> Vec<MetricType> {
        vec![
            MetricType::VmRss,
            MetricType::PmaRss,
            MetricType::NockStackRss,
            MetricType::HeapOtherRss,
            MetricType::MinorFaults,
            MetricType::MajorFaults,
        ]
    }

    /// Check if this metric is memory-related (for graphing purposes)
    pub fn is_memory(&self) -> bool {
        !matches!(
            self,
            MetricType::CpuPercent | MetricType::MinorFaults | MetricType::MajorFaults
        )
    }

    /// Get the unit for this metric
    pub fn unit(&self) -> &'static str {
        match self {
            MetricType::CpuPercent => "%",
            MetricType::MinorFaults | MetricType::MajorFaults => "count",
            _ => "KiB",
        }
    }
}

/// Configuration for a benchmark test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    /// Unique identifier for this test
    pub id: Uuid,

    /// Human-readable name for this test
    pub name: String,

    /// Optional description
    pub description: Option<String>,

    /// Benchmark execution mode
    #[serde(default)]
    pub benchmark_mode: BenchmarkMode,

    /// Containers to run in this test
    pub containers: Vec<ContainerConfig>,

    /// Metrics to track
    pub metrics: Vec<MetricType>,

    /// Test duration in seconds
    pub duration_secs: u64,

    /// Sampling interval in milliseconds
    pub sample_interval_ms: u64,

    /// SOL bench mode options
    #[serde(default)]
    pub sol_bench: SolBenchOptions,

    /// SOL sweep mode options
    #[serde(default)]
    pub sol_sweep: SolSweepOptions,

    /// Tags for organizing tests
    pub tags: Vec<String>,

    /// When this config was created
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "New Test".to_string(),
            description: None,
            benchmark_mode: BenchmarkMode::Container,
            containers: Vec::new(),
            metrics: MetricType::defaults(),
            duration_secs: 300, // 5 minutes
            sample_interval_ms: 1000,
            sol_bench: SolBenchOptions::default(),
            sol_sweep: SolSweepOptions::default(),
            tags: Vec::new(),
            created_at: chrono::Utc::now(),
        }
    }
}

impl TestConfig {
    /// Create a new test config with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Add a container to the test
    pub fn with_container(mut self, config: ContainerConfig) -> Self {
        self.containers.push(config);
        self
    }

    /// Set the metrics to track
    pub fn with_metrics(mut self, metrics: Vec<MetricType>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Set the test duration
    pub fn with_duration_secs(mut self, secs: u64) -> Self {
        self.duration_secs = secs;
        self
    }

    /// Set the sampling interval
    pub fn with_sample_interval_ms(mut self, ms: u64) -> Self {
        self.sample_interval_ms = ms;
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Validate the test configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Test name cannot be empty".to_string());
        }
        match self.benchmark_mode {
            BenchmarkMode::Container => {
                if self.containers.is_empty() {
                    return Err("Container mode requires at least one container".to_string());
                }
                if self.metrics.is_empty() {
                    return Err("Container mode must track at least one metric".to_string());
                }
                if self.duration_secs == 0 {
                    return Err("Container mode duration must be greater than 0".to_string());
                }
                if self.sample_interval_ms == 0 {
                    return Err("Container mode sample interval must be greater than 0".to_string());
                }
                for container in &self.containers {
                    container.validate()?;
                }
            }
            BenchmarkMode::SpeedOfLightBench => {
                if self.metrics.is_empty() {
                    return Err("SOL bench must track at least one metric".to_string());
                }
                self.sol_bench.validate()?;
            }
            BenchmarkMode::SpeedOfLightSweep => {
                self.sol_sweep.validate()?;
            }
        }
        Ok(())
    }

    /// Get a summary of this test configuration
    pub fn summary(&self) -> String {
        match self.benchmark_mode {
            BenchmarkMode::Container => {
                let containers = self.containers.len();
                let metrics = self.metrics.len();
                let duration = format_duration(self.duration_secs);
                format!(
                    "{} | {} container(s) | {} metric(s) | {}",
                    self.name, containers, metrics, duration
                )
            }
            BenchmarkMode::SpeedOfLightBench => {
                let source = self
                    .sol_bench
                    .fixture_path
                    .as_ref()
                    .filter(|path| !path.trim().is_empty())
                    .cloned()
                    .unwrap_or_else(|| "<fixture not set>".to_string());
                format!(
                    "{} | SOL bench | {} metric(s) | {}",
                    self.name,
                    self.metrics.len(),
                    source
                )
            }
            BenchmarkMode::SpeedOfLightSweep => {
                let cases = self.sol_sweep.case_count().unwrap_or(0);
                format!(
                    "{} | SOL sweep | {} case(s) | {} repeat(s)",
                    self.name, cases, self.sol_sweep.repeats
                )
            }
        }
    }
}

/// Format a duration in seconds to a human-readable string
fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_config_default() {
        let config = ContainerConfig::default();
        assert_eq!(config.persistence_mode, PersistenceMode::Checkpoint);
        assert_eq!(config.checkpoint_interval_secs, 120);
        assert_eq!(config.memory_limit, "16g");
        assert!(config.enable_mining);
        assert!(config.use_fakenet);
    }

    #[test]
    fn test_container_config_checkpoint() {
        let config = ContainerConfig::checkpoint("test", 60);
        assert_eq!(config.name, "test");
        assert_eq!(config.persistence_mode, PersistenceMode::Checkpoint);
        assert_eq!(config.checkpoint_interval_secs, 60);
    }

    #[test]
    fn test_container_config_pma_persist() {
        let config = ContainerConfig::pma_persist("test");
        assert_eq!(config.name, "test");
        assert_eq!(config.persistence_mode, PersistenceMode::PmaPersist);
    }

    #[test]
    fn test_container_config_validate() {
        let valid = ContainerConfig::default();
        assert!(valid.validate().is_ok());

        let invalid_name = ContainerConfig {
            name: String::new(),
            ..Default::default()
        };
        assert!(invalid_name.validate().is_err());

        let invalid_image = ContainerConfig {
            image: String::new(),
            ..Default::default()
        };
        assert!(invalid_image.validate().is_err());

        let invalid_memory = ContainerConfig {
            memory_limit: "invalid".to_string(),
            ..Default::default()
        };
        assert!(invalid_memory.validate().is_err());
    }

    #[test]
    fn test_parse_memory_limit() {
        assert_eq!(parse_memory_limit("16g"), Some(16 * 1024 * 1024 * 1024));
        assert_eq!(parse_memory_limit("8G"), Some(8 * 1024 * 1024 * 1024));
        assert_eq!(parse_memory_limit("512m"), Some(512 * 1024 * 1024));
        assert_eq!(parse_memory_limit("1024k"), Some(1024 * 1024));
        assert_eq!(parse_memory_limit("1073741824"), Some(1073741824));
        assert_eq!(parse_memory_limit("invalid"), None);
    }

    #[test]
    fn test_test_config_default() {
        let config = TestConfig::default();
        assert!(!config.metrics.is_empty());
        assert_eq!(config.duration_secs, 300);
        assert_eq!(config.sample_interval_ms, 1000);
    }

    #[test]
    fn test_test_config_validate() {
        let mut config = TestConfig::default();
        // No containers - should fail
        assert!(config.validate().is_err());

        // Add a container
        config.containers.push(ContainerConfig::default());
        assert!(config.validate().is_ok());

        // Empty name - should fail
        config.name = String::new();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_metric_type_all() {
        let all = MetricType::all();
        assert!(all.len() >= 10);
        assert!(all.contains(&MetricType::VmRss));
        assert!(all.contains(&MetricType::CpuPercent));
    }

    #[test]
    fn test_metric_type_defaults() {
        let defaults = MetricType::defaults();
        assert!(!defaults.is_empty());
        assert!(defaults.contains(&MetricType::VmRss));
    }

    #[test]
    fn test_metric_type_sol_defaults() {
        let defaults = MetricType::sol_defaults();
        assert!(defaults.contains(&MetricType::VmRss));
        assert!(defaults.contains(&MetricType::PmaRss));
        assert!(defaults.contains(&MetricType::MinorFaults));
    }

    #[test]
    fn test_metric_type_is_memory() {
        assert!(MetricType::VmRss.is_memory());
        assert!(MetricType::PmaRss.is_memory());
        assert!(!MetricType::CpuPercent.is_memory());
        assert!(!MetricType::MinorFaults.is_memory());
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(90), "1m 30s");
        assert_eq!(format_duration(3661), "1h 1m");
    }

    #[test]
    fn test_persistence_mode_label() {
        assert_eq!(PersistenceMode::Checkpoint.label(), "Checkpoint");
        assert_eq!(PersistenceMode::PmaPersist.label(), "PMA Persist");
    }

    #[test]
    fn test_benchmark_mode_label() {
        assert_eq!(BenchmarkMode::Container.label(), "Container");
        assert_eq!(BenchmarkMode::SpeedOfLightBench.label(), "SOL Bench");
        assert_eq!(BenchmarkMode::SpeedOfLightSweep.label(), "SOL Sweep");
    }

    #[test]
    fn test_benchmark_mode_all_excludes_sweep() {
        assert_eq!(
            BenchmarkMode::all(),
            &[BenchmarkMode::Container, BenchmarkMode::SpeedOfLightBench]
        );
        assert!(!BenchmarkMode::all().contains(&BenchmarkMode::SpeedOfLightSweep));
    }

    #[test]
    fn test_benchmark_mode_inline_help() {
        assert!(BenchmarkMode::Container.inline_help().contains("Docker"));
        assert!(BenchmarkMode::SpeedOfLightBench
            .inline_help()
            .contains(".soltest"));
        assert!(BenchmarkMode::SpeedOfLightSweep
            .inline_help()
            .contains("PMA candidate"));
    }

    #[test]
    fn test_sol_bench_options_validate() {
        let valid = SolBenchOptions {
            fixture_path: Some("sample.soltest".to_string()),
            ..Default::default()
        };
        assert!(valid.validate().is_ok());

        let invalid = SolBenchOptions {
            fixture_path: Some("sample.soltest".to_string()),
            profile_interval_ms: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let missing_fixture = SolBenchOptions {
            fixture_path: None,
            ..Default::default()
        };
        assert!(missing_fixture.validate().is_err());
    }

    #[test]
    fn test_sol_fixture_options_validate() {
        let valid = SolFixtureOptions::default();
        assert!(valid.validate().is_ok());

        let invalid = SolFixtureOptions {
            start_height: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid_order = SolFixtureOptions {
            start_height: 10,
            end_height: 9,
            ..Default::default()
        };
        assert!(invalid_order.validate().is_err());
    }

    #[test]
    fn test_sol_sweep_options_validate_and_case_count() {
        let valid = SolSweepOptions {
            candidates_csv: "a,b".to_string(),
            chunk_sizes_csv: "8,16".to_string(),
            memory_limits_csv: "8g,16g".to_string(),
            ..Default::default()
        };
        assert!(valid.validate().is_ok());
        assert_eq!(valid.case_count().expect("case count"), 8);

        let invalid = SolSweepOptions {
            chunk_sizes_csv: "bad".to_string(),
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_test_config_validate_sol_bench_mode() {
        let mut config = TestConfig::default();
        config.benchmark_mode = BenchmarkMode::SpeedOfLightBench;
        config.containers.clear();
        config.metrics = MetricType::sol_defaults();
        config.sol_bench.fixture_path = Some("sample.soltest".to_string());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_test_config_validate_sol_sweep_mode() {
        let mut config = TestConfig::default();
        config.benchmark_mode = BenchmarkMode::SpeedOfLightSweep;
        config.containers.clear();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_parse_csv_helpers() {
        assert_eq!(parse_csv_strings("a, b ,,c"), vec!["a", "b", "c"]);
        assert_eq!(parse_csv_u64("1, 2, 3").expect("u64 csv"), vec![1, 2, 3]);
        assert!(parse_csv_u64("1, nope").is_err());
    }
}
