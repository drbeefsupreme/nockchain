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

    /// Containers to run in this test
    pub containers: Vec<ContainerConfig>,

    /// Metrics to track
    pub metrics: Vec<MetricType>,

    /// Test duration in seconds
    pub duration_secs: u64,

    /// Sampling interval in milliseconds
    pub sample_interval_ms: u64,

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
            containers: Vec::new(),
            metrics: MetricType::defaults(),
            duration_secs: 300, // 5 minutes
            sample_interval_ms: 1000,
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
        if self.containers.is_empty() {
            return Err("Test must have at least one container".to_string());
        }
        if self.metrics.is_empty() {
            return Err("Test must track at least one metric".to_string());
        }
        if self.duration_secs == 0 {
            return Err("Test duration must be greater than 0".to_string());
        }
        if self.sample_interval_ms == 0 {
            return Err("Sample interval must be greater than 0".to_string());
        }
        for container in &self.containers {
            container.validate()?;
        }
        Ok(())
    }

    /// Get a summary of this test configuration
    pub fn summary(&self) -> String {
        let containers = self.containers.len();
        let metrics = self.metrics.len();
        let duration = format_duration(self.duration_secs);
        format!(
            "{} | {} container(s) | {} metric(s) | {}",
            self.name, containers, metrics, duration
        )
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
}
