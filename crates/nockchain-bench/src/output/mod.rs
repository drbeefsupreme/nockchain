//! Output formats for benchmark results
//!
//! This module provides writers for various output formats including Parquet
//! for time-series data analysis.

pub mod parquet;

pub use parquet::{ParquetWriter, ParquetError};
