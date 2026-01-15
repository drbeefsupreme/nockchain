//! Nockchain Bench - Benchmarking and memory profiling harness for Nockchain
//!
//! This crate provides tools for:
//! - Memory sampling and attribution (NockStack vs PMA vs heap)
//! - Docker-based test harness for reproducible benchmarks
//! - A/B comparison between different configurations
//! - Time-series data collection and analysis

pub mod output;
pub mod runner;
pub mod sampler;
pub mod scenario;
