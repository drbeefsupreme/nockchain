//! Container and process runners for benchmarking
//!
//! This module provides abstractions for running Nockchain in various environments
//! and collecting metrics.

pub mod docker;

pub use docker::{DockerRunner, DockerRunnerConfig, ContainerStats, NockchainMode};
