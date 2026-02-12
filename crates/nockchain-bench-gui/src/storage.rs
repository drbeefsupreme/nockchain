//! Storage for test configurations and results
//!
//! Handles saving and loading test data to disk, including:
//! - Test configurations
//! - Time-series samples
//! - Event logs
//! - Comparison results

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use nockchain_bench::speed_of_light::{MemoryProfile, SweepCaseSummary, SweepRunMetrics};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::config::{MetricType, TestConfig};

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Test not found: {0}")]
    NotFound(Uuid),

    #[error("Storage directory does not exist: {0}")]
    DirectoryNotFound(PathBuf),
}

/// A single data sample collected during a test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSample {
    /// Timestamp in milliseconds since test start
    pub timestamp_ms: u64,

    /// Container ID this sample is from
    pub container_id: Uuid,

    /// Metric values keyed by MetricType
    pub values: HashMap<MetricType, f64>,
}

impl DataSample {
    /// Create a new sample
    pub fn new(timestamp_ms: u64, container_id: Uuid) -> Self {
        Self {
            timestamp_ms,
            container_id,
            values: HashMap::new(),
        }
    }

    /// Add a metric value
    pub fn with_value(mut self, metric: MetricType, value: f64) -> Self {
        self.values.insert(metric, value);
        self
    }

    /// Get a metric value
    pub fn get(&self, metric: MetricType) -> Option<f64> {
        self.values.get(&metric).copied()
    }
}

/// An event that occurred during a test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEvent {
    /// Timestamp in milliseconds since test start
    pub timestamp_ms: u64,

    /// Container ID this event is from
    pub container_id: Uuid,

    /// Event type/label
    pub event_type: String,

    /// Event message
    pub message: String,

    /// Whether this event is significant (worth highlighting on graphs)
    pub is_significant: bool,
}

impl TestEvent {
    /// Create a new event
    pub fn new(
        timestamp_ms: u64,
        container_id: Uuid,
        event_type: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            timestamp_ms,
            container_id,
            event_type: event_type.into(),
            message: message.into(),
            is_significant: false,
        }
    }

    /// Mark this event as significant
    pub fn significant(mut self) -> Self {
        self.is_significant = true;
        self
    }
}

/// Statistics computed from a test result
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TestStatistics {
    /// Peak values for each metric
    pub peak: HashMap<MetricType, f64>,

    /// Average values for each metric
    pub average: HashMap<MetricType, f64>,

    /// Minimum values for each metric
    pub min: HashMap<MetricType, f64>,

    /// P50 (median) values for each metric
    pub p50: HashMap<MetricType, f64>,

    /// P95 values for each metric
    pub p95: HashMap<MetricType, f64>,

    /// P99 values for each metric
    pub p99: HashMap<MetricType, f64>,

    /// Standard deviation for each metric
    pub std_dev: HashMap<MetricType, f64>,
}

impl TestStatistics {
    /// Compute statistics from samples for a specific container
    pub fn compute(samples: &[DataSample], container_id: Uuid) -> Self {
        let mut stats = TestStatistics::default();

        // Get samples for this container
        let container_samples: Vec<&DataSample> = samples
            .iter()
            .filter(|s| s.container_id == container_id)
            .collect();

        if container_samples.is_empty() {
            return stats;
        }

        // Collect all metric types present
        let mut metric_values: HashMap<MetricType, Vec<f64>> = HashMap::new();
        for sample in &container_samples {
            for (metric, value) in &sample.values {
                metric_values.entry(*metric).or_default().push(*value);
            }
        }

        // Compute statistics for each metric
        for (metric, mut values) in metric_values {
            if values.is_empty() {
                continue;
            }

            // Sort for percentiles
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = values.len();
            let sum: f64 = values.iter().sum();
            let avg = sum / n as f64;

            // Peak (max)
            stats.peak.insert(metric, values[n - 1]);

            // Min
            stats.min.insert(metric, values[0]);

            // Average
            stats.average.insert(metric, avg);

            // Percentiles
            stats.p50.insert(metric, percentile(&values, 0.50));
            stats.p95.insert(metric, percentile(&values, 0.95));
            stats.p99.insert(metric, percentile(&values, 0.99));

            // Standard deviation
            let variance = values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / n as f64;
            stats.std_dev.insert(metric, variance.sqrt());
        }

        stats
    }
}

/// Calculate a percentile from sorted values
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Status of a test result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    /// Test completed successfully
    Success,
    /// Test completed with errors
    Failed,
    /// Test is currently running
    Running,
    /// Test was cancelled
    Cancelled,
}

/// Persisted SOL bench result payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolBenchResult {
    pub blocks_poked: u64,
    pub failed_pokes: u64,
    pub init_time_secs: f64,
    pub total_poke_time_secs: f64,
    pub blocks_per_second: f64,
    pub checkpoint_count: u64,
    pub checkpoint_total_time_secs: f64,
    pub checkpoint_avg_time_secs: Option<f64>,
    pub memory_profile: Option<MemoryProfile>,
}

/// Persisted SOL sweep result payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolSweepResult {
    pub runs: Vec<SweepRunMetrics>,
    pub summaries: Vec<SweepCaseSummary>,
}

impl TestStatus {
    pub fn label(&self) -> &'static str {
        match self {
            TestStatus::Success => "Success",
            TestStatus::Failed => "Failed",
            TestStatus::Running => "Running",
            TestStatus::Cancelled => "Cancelled",
        }
    }
}

/// Result of running a test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Unique identifier
    pub id: Uuid,

    /// The test configuration used
    pub config: TestConfig,

    /// When the test started
    pub started_at: DateTime<Utc>,

    /// When the test ended (if completed)
    pub ended_at: Option<DateTime<Utc>>,

    /// Test status
    pub status: TestStatus,

    /// Error message if failed
    pub error: Option<String>,

    /// Time-series samples
    pub samples: Vec<DataSample>,

    /// Events that occurred during the test
    pub events: Vec<TestEvent>,

    /// Computed statistics per container
    pub statistics: HashMap<Uuid, TestStatistics>,

    /// Container logs (last N lines per container)
    pub logs: HashMap<Uuid, Vec<String>>,

    /// SOL benchmark result details (if run in SOL bench mode)
    #[serde(default)]
    pub sol_bench: Option<SolBenchResult>,

    /// SOL sweep result details (if run in SOL sweep mode)
    #[serde(default)]
    pub sol_sweep: Option<SolSweepResult>,
}

impl TestResult {
    /// Create a new test result
    pub fn new(config: TestConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            config,
            started_at: Utc::now(),
            ended_at: None,
            status: TestStatus::Running,
            error: None,
            samples: Vec::new(),
            events: Vec::new(),
            statistics: HashMap::new(),
            logs: HashMap::new(),
            sol_bench: None,
            sol_sweep: None,
        }
    }

    /// Mark the test as completed successfully
    pub fn complete(&mut self) {
        self.ended_at = Some(Utc::now());
        self.status = TestStatus::Success;
        self.compute_statistics();
    }

    /// Mark the test as failed
    pub fn fail(&mut self, error: impl Into<String>) {
        self.ended_at = Some(Utc::now());
        self.status = TestStatus::Failed;
        self.error = Some(error.into());
        self.compute_statistics();
    }

    /// Mark the test as cancelled
    pub fn cancel(&mut self) {
        self.ended_at = Some(Utc::now());
        self.status = TestStatus::Cancelled;
        self.compute_statistics();
    }

    /// Add a sample
    pub fn add_sample(&mut self, sample: DataSample) {
        self.samples.push(sample);
    }

    /// Add an event
    pub fn add_event(&mut self, event: TestEvent) {
        self.events.push(event);
    }

    /// Add logs for a container
    pub fn add_logs(&mut self, container_id: Uuid, logs: Vec<String>) {
        self.logs.insert(container_id, logs);
    }

    /// Compute statistics from samples
    pub fn compute_statistics(&mut self) {
        if self.config.containers.is_empty() {
            let mut seen = std::collections::HashSet::new();
            for sample in &self.samples {
                if seen.insert(sample.container_id) {
                    let stats = TestStatistics::compute(&self.samples, sample.container_id);
                    self.statistics.insert(sample.container_id, stats);
                }
            }
        } else {
            for container in &self.config.containers {
                let stats = TestStatistics::compute(&self.samples, container.id);
                self.statistics.insert(container.id, stats);
            }
        }
    }

    /// Get the duration of the test
    pub fn duration(&self) -> std::time::Duration {
        let end = self.ended_at.unwrap_or_else(Utc::now);
        let duration = end.signed_duration_since(self.started_at);
        std::time::Duration::from_millis(duration.num_milliseconds().max(0) as u64)
    }

    /// Get samples for a specific container
    pub fn samples_for_container(&self, container_id: Uuid) -> Vec<&DataSample> {
        self.samples
            .iter()
            .filter(|s| s.container_id == container_id)
            .collect()
    }

    /// Get events for a specific container
    pub fn events_for_container(&self, container_id: Uuid) -> Vec<&TestEvent> {
        self.events
            .iter()
            .filter(|e| e.container_id == container_id)
            .collect()
    }

    /// Get significant events
    pub fn significant_events(&self) -> Vec<&TestEvent> {
        self.events.iter().filter(|e| e.is_significant).collect()
    }
}

/// Comparison between two test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestComparison {
    /// ID of the baseline result
    pub baseline_id: Uuid,

    /// ID of the comparison result
    pub comparison_id: Uuid,

    /// Baseline statistics (keyed by container name -> metric -> value)
    pub baseline_stats: HashMap<String, TestStatistics>,

    /// Comparison statistics
    pub comparison_stats: HashMap<String, TestStatistics>,

    /// Percentage changes (positive = regression, negative = improvement)
    pub changes: HashMap<String, HashMap<MetricType, f64>>,

    /// Whether this represents a regression
    pub is_regression: bool,

    /// Confidence score (0-1)
    pub confidence: f64,
}

impl TestComparison {
    /// Create a comparison between two test results
    pub fn compare(baseline: &TestResult, comparison: &TestResult) -> Self {
        let mut baseline_stats = HashMap::new();
        let mut comparison_stats = HashMap::new();
        let mut changes = HashMap::new();

        // Compute stats for baseline containers
        for container in &baseline.config.containers {
            let stats = TestStatistics::compute(&baseline.samples, container.id);
            baseline_stats.insert(container.name.clone(), stats);
        }

        // Compute stats for comparison containers
        for container in &comparison.config.containers {
            let stats = TestStatistics::compute(&comparison.samples, container.id);
            comparison_stats.insert(container.name.clone(), stats);
        }

        // Calculate changes for matching containers
        let mut has_regression = false;
        for (name, base_stats) in &baseline_stats {
            if let Some(comp_stats) = comparison_stats.get(name) {
                let mut container_changes = HashMap::new();

                for (metric, base_avg) in &base_stats.average {
                    if let Some(comp_avg) = comp_stats.average.get(metric) {
                        if *base_avg > 0.0 {
                            let change = ((comp_avg - base_avg) / base_avg) * 100.0;
                            container_changes.insert(*metric, change);

                            // Memory increase > 5% is considered a regression
                            if metric.is_memory() && change > 5.0 {
                                has_regression = true;
                            }
                        }
                    }
                }

                changes.insert(name.clone(), container_changes);
            }
        }

        TestComparison {
            baseline_id: baseline.id,
            comparison_id: comparison.id,
            baseline_stats,
            comparison_stats,
            changes,
            is_regression: has_regression,
            confidence: 0.95, // Placeholder - would compute from statistical tests
        }
    }

    /// Get the change percentage for a specific container and metric
    pub fn get_change(&self, container_name: &str, metric: MetricType) -> Option<f64> {
        self.changes
            .get(container_name)
            .and_then(|m| m.get(&metric).copied())
    }
}

/// Storage manager for test results
pub struct TestStorage {
    /// Base directory for storing results
    base_dir: PathBuf,
}

impl TestStorage {
    /// Create a new storage manager
    pub fn new(base_dir: impl Into<PathBuf>) -> Result<Self, StorageError> {
        let base_dir = base_dir.into();

        // Create directories if they don't exist
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(base_dir.join("results"))?;
        fs::create_dir_all(base_dir.join("configs"))?;

        Ok(Self { base_dir })
    }

    /// Get the default storage location
    pub fn default_location() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("nockchain-bench")
    }

    /// Save a test result
    pub fn save_result(&self, result: &TestResult) -> Result<PathBuf, StorageError> {
        let path = self.result_path(result.id);
        let json = serde_json::to_string_pretty(result)?;
        fs::write(&path, json)?;
        Ok(path)
    }

    /// Load a test result by ID
    pub fn load_result(&self, id: Uuid) -> Result<TestResult, StorageError> {
        let path = self.result_path(id);
        if !path.exists() {
            return Err(StorageError::NotFound(id));
        }
        let json = fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Delete a test result
    pub fn delete_result(&self, id: Uuid) -> Result<(), StorageError> {
        let path = self.result_path(id);
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }

    /// List all test results
    pub fn list_results(&self) -> Result<Vec<TestResultSummary>, StorageError> {
        let results_dir = self.base_dir.join("results");
        let mut summaries = Vec::new();

        for entry in fs::read_dir(results_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(json) = fs::read_to_string(&path) {
                    if let Ok(result) = serde_json::from_str::<TestResult>(&json) {
                        summaries.push(TestResultSummary::from(&result));
                    }
                }
            }
        }

        // Sort by date, newest first
        summaries.sort_by(|a, b| b.started_at.cmp(&a.started_at));

        Ok(summaries)
    }

    /// Save a test configuration template
    pub fn save_config(&self, config: &TestConfig) -> Result<PathBuf, StorageError> {
        let path = self.config_path(config.id);
        let json = serde_json::to_string_pretty(config)?;
        fs::write(&path, json)?;
        Ok(path)
    }

    /// Load a test configuration
    pub fn load_config(&self, id: Uuid) -> Result<TestConfig, StorageError> {
        let path = self.config_path(id);
        if !path.exists() {
            return Err(StorageError::NotFound(id));
        }
        let json = fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&json)?)
    }

    /// List all saved test configurations
    pub fn list_configs(&self) -> Result<Vec<TestConfig>, StorageError> {
        let configs_dir = self.base_dir.join("configs");
        let mut configs = Vec::new();

        for entry in fs::read_dir(configs_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(json) = fs::read_to_string(&path) {
                    if let Ok(config) = serde_json::from_str::<TestConfig>(&json) {
                        configs.push(config);
                    }
                }
            }
        }

        // Sort by date, newest first
        configs.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(configs)
    }

    fn result_path(&self, id: Uuid) -> PathBuf {
        self.base_dir.join("results").join(format!("{}.json", id))
    }

    fn config_path(&self, id: Uuid) -> PathBuf {
        self.base_dir.join("configs").join(format!("{}.json", id))
    }
}

/// Summary of a test result (for listing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResultSummary {
    pub id: Uuid,
    pub name: String,
    pub started_at: DateTime<Utc>,
    pub status: TestStatus,
    pub container_count: usize,
    pub sample_count: usize,
    pub duration_secs: f64,
}

impl From<&TestResult> for TestResultSummary {
    fn from(result: &TestResult) -> Self {
        Self {
            id: result.id,
            name: result.config.name.clone(),
            started_at: result.started_at,
            status: result.status,
            container_count: result.config.containers.len(),
            sample_count: result.samples.len(),
            duration_secs: result.duration().as_secs_f64(),
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::config::{BenchmarkMode, ContainerConfig};

    fn make_test_config() -> TestConfig {
        TestConfig::new("Test")
            .with_container(ContainerConfig::default())
            .with_metrics(vec![MetricType::VmRss, MetricType::CpuPercent])
    }

    fn make_test_result() -> TestResult {
        let config = make_test_config();
        let container_id = config.containers[0].id;
        let mut result = TestResult::new(config);

        // Add some samples
        for i in 0..10 {
            let sample = DataSample::new(i * 1000, container_id)
                .with_value(MetricType::VmRss, 1000.0 + i as f64 * 100.0)
                .with_value(MetricType::CpuPercent, 50.0 + i as f64 * 2.0);
            result.add_sample(sample);
        }

        // Add an event
        result.add_event(
            TestEvent::new(5000, container_id, "checkpoint", "Checkpoint saved").significant(),
        );

        result.complete();
        result
    }

    #[test]
    fn test_data_sample() {
        let container_id = Uuid::new_v4();
        let sample = DataSample::new(1000, container_id)
            .with_value(MetricType::VmRss, 12345.0)
            .with_value(MetricType::CpuPercent, 50.0);

        assert_eq!(sample.timestamp_ms, 1000);
        assert_eq!(sample.get(MetricType::VmRss), Some(12345.0));
        assert_eq!(sample.get(MetricType::CpuPercent), Some(50.0));
        assert_eq!(sample.get(MetricType::PmaRss), None);
    }

    #[test]
    fn test_test_statistics_compute() {
        let container_id = Uuid::new_v4();
        let samples: Vec<DataSample> = (0..10)
            .map(|i| {
                DataSample::new(i * 1000, container_id)
                    .with_value(MetricType::VmRss, 100.0 + i as f64 * 10.0)
            })
            .collect();

        let stats = TestStatistics::compute(&samples, container_id);

        // Values: 100, 110, 120, ..., 190
        assert_eq!(stats.min.get(&MetricType::VmRss), Some(&100.0));
        assert_eq!(stats.peak.get(&MetricType::VmRss), Some(&190.0));
        assert_eq!(stats.average.get(&MetricType::VmRss), Some(&145.0));
    }

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 0.5), 3.0);
        assert_eq!(percentile(&values, 1.0), 5.0);
    }

    #[test]
    fn test_test_result_lifecycle() {
        let config = make_test_config();
        let mut result = TestResult::new(config);

        assert_eq!(result.status, TestStatus::Running);
        assert!(result.ended_at.is_none());

        result.complete();

        assert_eq!(result.status, TestStatus::Success);
        assert!(result.ended_at.is_some());
    }

    #[test]
    fn test_test_result_fail() {
        let config = make_test_config();
        let mut result = TestResult::new(config);

        result.fail("Test error");

        assert_eq!(result.status, TestStatus::Failed);
        assert_eq!(result.error, Some("Test error".to_string()));
    }

    #[test]
    fn test_storage_save_load() {
        let dir = tempdir().unwrap();
        let storage = TestStorage::new(dir.path()).unwrap();

        let result = make_test_result();
        let id = result.id;

        // Save
        storage.save_result(&result).unwrap();

        // Load
        let loaded = storage.load_result(id).unwrap();
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.config.name, "Test");
        assert_eq!(loaded.samples.len(), 10);
    }

    #[test]
    fn test_storage_list_results() {
        let dir = tempdir().unwrap();
        let storage = TestStorage::new(dir.path()).unwrap();

        // Save multiple results
        for i in 0..3 {
            let mut config = make_test_config();
            config.name = format!("Test {}", i);
            let mut result = TestResult::new(config);
            result.complete();
            storage.save_result(&result).unwrap();
        }

        let summaries = storage.list_results().unwrap();
        assert_eq!(summaries.len(), 3);
    }

    #[test]
    fn test_storage_delete() {
        let dir = tempdir().unwrap();
        let storage = TestStorage::new(dir.path()).unwrap();

        let result = make_test_result();
        let id = result.id;

        storage.save_result(&result).unwrap();
        assert!(storage.load_result(id).is_ok());

        storage.delete_result(id).unwrap();
        assert!(storage.load_result(id).is_err());
    }

    #[test]
    fn test_storage_config() {
        let dir = tempdir().unwrap();
        let storage = TestStorage::new(dir.path()).unwrap();

        let config = make_test_config();
        let id = config.id;

        storage.save_config(&config).unwrap();

        let loaded = storage.load_config(id).unwrap();
        assert_eq!(loaded.name, "Test");
    }

    #[test]
    fn test_comparison() {
        let config1 = make_test_config();
        let config2 = make_test_config();

        let container_id1 = config1.containers[0].id;
        let container_id2 = config2.containers[0].id;

        let mut result1 = TestResult::new(config1);
        let mut result2 = TestResult::new(config2);

        // Baseline: average 150
        for i in 0..10 {
            result1.add_sample(
                DataSample::new(i * 1000, container_id1)
                    .with_value(MetricType::VmRss, 100.0 + i as f64 * 10.0),
            );
        }

        // Comparison: average 180 (20% increase)
        for i in 0..10 {
            result2.add_sample(
                DataSample::new(i * 1000, container_id2)
                    .with_value(MetricType::VmRss, 130.0 + i as f64 * 10.0),
            );
        }

        result1.complete();
        result2.complete();

        let comparison = TestComparison::compare(&result1, &result2);

        // Both use "New Container" as name
        let change = comparison.get_change("New Container", MetricType::VmRss);
        assert!(change.is_some());

        // ~20% increase
        let change_val = change.unwrap();
        assert!(change_val > 15.0 && change_val < 25.0);
    }

    #[test]
    fn test_compute_statistics_without_config_containers() {
        let mut config = TestConfig::default();
        config.benchmark_mode = BenchmarkMode::SpeedOfLightBench;
        config.containers.clear();

        let mut result = TestResult::new(config);
        let synthetic_container = Uuid::new_v4();
        result.add_sample(
            DataSample::new(0, synthetic_container).with_value(MetricType::VmRss, 1024.0),
        );
        result.add_sample(
            DataSample::new(1000, synthetic_container).with_value(MetricType::VmRss, 2048.0),
        );
        result.complete();

        assert!(result.statistics.contains_key(&synthetic_container));
    }

    #[test]
    fn test_sol_payload_roundtrip() {
        let dir = tempdir().unwrap();
        let storage = TestStorage::new(dir.path()).unwrap();

        let mut result = make_test_result();
        result.sol_bench = Some(SolBenchResult {
            blocks_poked: 1000,
            failed_pokes: 2,
            init_time_secs: 4.2,
            total_poke_time_secs: 10.5,
            blocks_per_second: 95.2,
            checkpoint_count: 3,
            checkpoint_total_time_secs: 1.5,
            checkpoint_avg_time_secs: Some(0.5),
            memory_profile: None,
        });
        result.sol_sweep = Some(SolSweepResult {
            runs: Vec::new(),
            summaries: Vec::new(),
        });

        storage.save_result(&result).unwrap();
        let loaded = storage.load_result(result.id).unwrap();
        assert!(loaded.sol_bench.is_some());
        assert!(loaded.sol_sweep.is_some());
        assert_eq!(loaded.sol_bench.unwrap().blocks_poked, 1000);
    }
}
