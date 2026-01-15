//! Parquet output for benchmark results
//!
//! Writes time-series container stats and scenario results to Parquet files
//! for later analysis with tools like DuckDB, Pandas, or Polars.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Array, StringArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use thiserror::Error;

use crate::runner::ContainerStats;
use crate::scenario::MiningResult;

#[derive(Debug, Error)]
pub enum ParquetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("No data to write")]
    NoData,
}

/// Writer for Parquet output files
pub struct ParquetWriter {
    /// Compression to use (default: Snappy)
    compression: Compression,
}

impl Default for ParquetWriter {
    fn default() -> Self {
        Self {
            compression: Compression::SNAPPY,
        }
    }
}

impl ParquetWriter {
    /// Create a new Parquet writer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set compression algorithm
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Write container stats time-series to a Parquet file
    ///
    /// Schema:
    /// - run_name: String - identifier for this benchmark run
    /// - timestamp_ms: UInt64 - milliseconds since run start
    /// - memory_usage_bytes: UInt64 - total memory usage
    /// - memory_limit_bytes: UInt64 - container memory limit
    /// - memory_percent: Float64 - usage as percentage
    /// - memory_cache_bytes: UInt64 - cache memory (can be reclaimed)
    /// - memory_rss_bytes: UInt64 - resident set size
    /// - cpu_percent: Float64 - CPU usage percentage
    pub fn write_stats<P: AsRef<Path>>(
        &self,
        path: P,
        run_name: &str,
        stats: &[ContainerStats],
    ) -> Result<(), ParquetError> {
        if stats.is_empty() {
            return Err(ParquetError::NoData);
        }

        let schema = Self::stats_schema();
        let batch = Self::stats_to_batch(run_name, stats, &schema)?;

        self.write_batch(path, batch)
    }

    /// Write multiple runs' stats to a single Parquet file
    ///
    /// Useful for A/B comparisons where you want all data in one file.
    pub fn write_multi_stats<P: AsRef<Path>>(
        &self,
        path: P,
        runs: &[(&str, &[ContainerStats])],
    ) -> Result<(), ParquetError> {
        if runs.is_empty() || runs.iter().all(|(_, s)| s.is_empty()) {
            return Err(ParquetError::NoData);
        }

        let schema = Self::stats_schema();
        let mut batches = Vec::new();

        for (name, stats) in runs {
            if !stats.is_empty() {
                batches.push(Self::stats_to_batch(name, stats, &schema)?);
            }
        }

        self.write_batches(path, &batches)
    }

    /// Write mining results summary to a Parquet file
    ///
    /// Schema:
    /// - run_name: String
    /// - mode: String - "checkpoint" or "pma_persist"
    /// - duration_secs: Float64
    /// - sample_count: UInt64
    /// - peak_memory_bytes: UInt64
    /// - avg_memory_bytes: UInt64
    /// - peak_rss_bytes: UInt64
    /// - avg_rss_bytes: UInt64
    /// - success: UInt64 (boolean as 0/1)
    /// - error: String (nullable)
    pub fn write_results<P: AsRef<Path>>(
        &self,
        path: P,
        results: &[&MiningResult],
    ) -> Result<(), ParquetError> {
        if results.is_empty() {
            return Err(ParquetError::NoData);
        }

        let schema = Self::results_schema();
        let batch = Self::results_to_batch(results, &schema)?;

        self.write_batch(path, batch)
    }

    /// Get the schema for stats time-series
    fn stats_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("run_name", DataType::Utf8, false),
            Field::new("timestamp_ms", DataType::UInt64, false),
            Field::new("memory_usage_bytes", DataType::UInt64, false),
            Field::new("memory_limit_bytes", DataType::UInt64, false),
            Field::new("memory_percent", DataType::Float64, false),
            Field::new("memory_cache_bytes", DataType::UInt64, false),
            Field::new("memory_rss_bytes", DataType::UInt64, false),
            Field::new("cpu_percent", DataType::Float64, false),
        ]))
    }

    /// Get the schema for results summary
    fn results_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("run_name", DataType::Utf8, false),
            Field::new("mode", DataType::Utf8, false),
            Field::new("duration_secs", DataType::Float64, false),
            Field::new("sample_count", DataType::UInt64, false),
            Field::new("peak_memory_bytes", DataType::UInt64, false),
            Field::new("avg_memory_bytes", DataType::UInt64, false),
            Field::new("peak_rss_bytes", DataType::UInt64, false),
            Field::new("avg_rss_bytes", DataType::UInt64, false),
            Field::new("success", DataType::UInt64, false),
            Field::new("error", DataType::Utf8, true),
        ]))
    }

    /// Convert stats to Arrow RecordBatch
    fn stats_to_batch(
        run_name: &str,
        stats: &[ContainerStats],
        schema: &Arc<Schema>,
    ) -> Result<RecordBatch, ParquetError> {
        let n = stats.len();

        let run_names: Vec<&str> = vec![run_name; n];
        let timestamps: Vec<u64> = stats.iter().map(|s| s.timestamp_ms).collect();
        let memory_usage: Vec<u64> = stats.iter().map(|s| s.memory_usage_bytes).collect();
        let memory_limit: Vec<u64> = stats.iter().map(|s| s.memory_limit_bytes).collect();
        let memory_percent: Vec<f64> = stats.iter().map(|s| s.memory_percent).collect();
        let memory_cache: Vec<u64> = stats.iter().map(|s| s.memory_cache_bytes).collect();
        let memory_rss: Vec<u64> = stats.iter().map(|s| s.memory_rss_bytes).collect();
        let cpu_percent: Vec<f64> = stats.iter().map(|s| s.cpu_percent).collect();

        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(run_names)),
            Arc::new(UInt64Array::from(timestamps)),
            Arc::new(UInt64Array::from(memory_usage)),
            Arc::new(UInt64Array::from(memory_limit)),
            Arc::new(Float64Array::from(memory_percent)),
            Arc::new(UInt64Array::from(memory_cache)),
            Arc::new(UInt64Array::from(memory_rss)),
            Arc::new(Float64Array::from(cpu_percent)),
        ];

        Ok(RecordBatch::try_new(schema.clone(), columns)?)
    }

    /// Convert results to Arrow RecordBatch
    fn results_to_batch(
        results: &[&MiningResult],
        schema: &Arc<Schema>,
    ) -> Result<RecordBatch, ParquetError> {
        let run_names: Vec<&str> = results.iter().map(|r| r.config.name.as_str()).collect();

        let modes: Vec<String> = results
            .iter()
            .map(|r| match &r.config.mode {
                crate::runner::NockchainMode::Checkpoint { save_interval_secs } => {
                    format!("checkpoint_{}s", save_interval_secs)
                }
                crate::runner::NockchainMode::PmaPersist => "pma_persist".to_string(),
            })
            .collect();
        let mode_refs: Vec<&str> = modes.iter().map(|s| s.as_str()).collect();

        let durations: Vec<f64> = results.iter().map(|r| r.actual_duration.as_secs_f64()).collect();
        let sample_counts: Vec<u64> = results.iter().map(|r| r.samples.len() as u64).collect();
        let peak_memory: Vec<u64> = results.iter().map(|r| r.peak_memory_bytes).collect();
        let avg_memory: Vec<u64> = results.iter().map(|r| r.avg_memory_bytes).collect();
        let peak_rss: Vec<u64> = results.iter().map(|r| r.peak_rss_bytes).collect();
        let avg_rss: Vec<u64> = results.iter().map(|r| r.avg_rss_bytes).collect();
        let success: Vec<u64> = results.iter().map(|r| if r.success { 1 } else { 0 }).collect();
        let errors: Vec<Option<&str>> = results
            .iter()
            .map(|r| r.error.as_deref())
            .collect();

        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(run_names)),
            Arc::new(StringArray::from(mode_refs)),
            Arc::new(Float64Array::from(durations)),
            Arc::new(UInt64Array::from(sample_counts)),
            Arc::new(UInt64Array::from(peak_memory)),
            Arc::new(UInt64Array::from(avg_memory)),
            Arc::new(UInt64Array::from(peak_rss)),
            Arc::new(UInt64Array::from(avg_rss)),
            Arc::new(UInt64Array::from(success)),
            Arc::new(StringArray::from(errors)),
        ];

        Ok(RecordBatch::try_new(schema.clone(), columns)?)
    }

    /// Write a single batch to a file
    fn write_batch<P: AsRef<Path>>(&self, path: P, batch: RecordBatch) -> Result<(), ParquetError> {
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(self.compression)
            .build();

        let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    /// Write multiple batches to a file
    fn write_batches<P: AsRef<Path>>(
        &self,
        path: P,
        batches: &[RecordBatch],
    ) -> Result<(), ParquetError> {
        if batches.is_empty() {
            return Err(ParquetError::NoData);
        }

        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(self.compression)
            .build();

        let mut writer = ArrowWriter::try_new(file, batches[0].schema(), Some(props))?;

        for batch in batches {
            writer.write(batch)?;
        }

        writer.close()?;

        Ok(())
    }
}

/// Convenience function to write stats to a Parquet file
pub fn write_stats_parquet<P: AsRef<Path>>(
    path: P,
    run_name: &str,
    stats: &[ContainerStats],
) -> Result<(), ParquetError> {
    ParquetWriter::new().write_stats(path, run_name, stats)
}

/// Convenience function to write results to a Parquet file
pub fn write_results_parquet<P: AsRef<Path>>(
    path: P,
    results: &[&MiningResult],
) -> Result<(), ParquetError> {
    ParquetWriter::new().write_results(path, results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::NockchainMode;
    use crate::scenario::MiningScenarioConfig;
    use std::time::Duration;
    use tempfile::tempdir;

    fn make_test_stats() -> Vec<ContainerStats> {
        vec![
            ContainerStats {
                timestamp_ms: 0,
                memory_usage_bytes: 1000 * 1024 * 1024,
                memory_limit_bytes: 16000 * 1024 * 1024,
                memory_percent: 6.25,
                memory_cache_bytes: 100 * 1024 * 1024,
                memory_rss_bytes: 900 * 1024 * 1024,
                cpu_percent: 100.0,
            },
            ContainerStats {
                timestamp_ms: 1000,
                memory_usage_bytes: 1500 * 1024 * 1024,
                memory_limit_bytes: 16000 * 1024 * 1024,
                memory_percent: 9.375,
                memory_cache_bytes: 150 * 1024 * 1024,
                memory_rss_bytes: 1350 * 1024 * 1024,
                cpu_percent: 105.0,
            },
            ContainerStats {
                timestamp_ms: 2000,
                memory_usage_bytes: 2000 * 1024 * 1024,
                memory_limit_bytes: 16000 * 1024 * 1024,
                memory_percent: 12.5,
                memory_cache_bytes: 200 * 1024 * 1024,
                memory_rss_bytes: 1800 * 1024 * 1024,
                cpu_percent: 110.0,
            },
        ]
    }

    fn make_test_result(name: &str, mode: NockchainMode) -> MiningResult {
        let config = MiningScenarioConfig {
            name: name.to_string(),
            mode,
            ..Default::default()
        };

        MiningResult {
            config,
            start_time_unix: 1700000000,
            actual_duration: Duration::from_secs(60),
            samples: make_test_stats(),
            peak_memory_bytes: 2000 * 1024 * 1024,
            avg_memory_bytes: 1500 * 1024 * 1024,
            peak_rss_bytes: 1800 * 1024 * 1024,
            avg_rss_bytes: 1350 * 1024 * 1024,
            final_logs: vec!["test log".to_string()],
            success: true,
            error: None,
        }
    }

    #[test]
    fn test_write_stats_parquet() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("stats.parquet");

        let stats = make_test_stats();
        let writer = ParquetWriter::new();

        writer.write_stats(&path, "test-run", &stats).unwrap();

        assert!(path.exists());

        // Verify file size is reasonable (should be small but non-zero)
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 100, "Parquet file should have content");
        assert!(metadata.len() < 10000, "Parquet file should be reasonably small");
    }

    #[test]
    fn test_write_stats_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.parquet");

        let writer = ParquetWriter::new();
        let result = writer.write_stats(&path, "empty", &[]);

        assert!(matches!(result, Err(ParquetError::NoData)));
    }

    #[test]
    fn test_write_multi_stats() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("multi.parquet");

        let stats_a = make_test_stats();
        let stats_b = make_test_stats();

        let writer = ParquetWriter::new();
        writer
            .write_multi_stats(&path, &[("run-a", &stats_a), ("run-b", &stats_b)])
            .unwrap();

        assert!(path.exists());
    }

    #[test]
    fn test_write_results_parquet() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("results.parquet");

        let result1 = make_test_result("checkpoint-test", NockchainMode::Checkpoint { save_interval_secs: 120 });
        let result2 = make_test_result("pma-test", NockchainMode::PmaPersist);

        let writer = ParquetWriter::new();
        writer.write_results(&path, &[&result1, &result2]).unwrap();

        assert!(path.exists());

        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() > 100);
    }

    #[test]
    fn test_convenience_functions() {
        let dir = tempdir().unwrap();

        // Test write_stats_parquet
        let stats_path = dir.path().join("stats.parquet");
        let stats = make_test_stats();
        write_stats_parquet(&stats_path, "test", &stats).unwrap();
        assert!(stats_path.exists());

        // Test write_results_parquet
        let results_path = dir.path().join("results.parquet");
        let result = make_test_result("test", NockchainMode::PmaPersist);
        write_results_parquet(&results_path, &[&result]).unwrap();
        assert!(results_path.exists());
    }

    #[test]
    fn test_compression_options() {
        let dir = tempdir().unwrap();
        let stats = make_test_stats();

        // Test compression algorithms that are enabled by default
        let compressions = [
            Compression::SNAPPY,
            Compression::UNCOMPRESSED,
        ];

        for (i, compression) in compressions.iter().enumerate() {
            let path = dir.path().join(format!("comp_{}.parquet", i));
            let writer = ParquetWriter::new().with_compression(compression.clone());
            writer.write_stats(&path, "test", &stats).unwrap();
            assert!(path.exists());
        }
    }
}
