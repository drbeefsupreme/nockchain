//! Nockchain Bench - Benchmarking and memory profiling harness for Nockchain
//!
//! This crate provides tools for:
//! - Memory sampling and attribution (NockStack vs heap/file-backed mappings)
//! - Docker-based test harness for reproducible benchmarks
//! - A/B comparison between different configurations
//! - Time-series data collection and analysis
//! - Event correlation from logs
//! - Speed-of-light benchmarks (maximum throughput without network)

pub mod events;
pub mod output;
pub mod runner;
pub mod sampler;
pub mod scenario;
pub mod speed_of_light;
