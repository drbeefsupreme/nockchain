//! Nockchain Bench GUI
//!
//! A graphical user interface for nockchain-bench benchmarking and memory profiling.
//!
//! Features:
//! - Create and configure Docker containers with various CLI flags
//! - Create and run benchmark tests with customizable metrics
//! - Real-time visualization of memory usage and events
//! - Compare results across runs with statistical analysis
//! - Terminal output panes with tabbed interface
//! - Git integration for building from different branches/commits

pub mod app;
pub mod config;
pub mod docker_panel;
pub mod git_panel;
pub mod graph;
pub mod runner;
pub mod storage;
pub mod terminal;
pub mod test_panel;

pub use app::BenchApp;
pub use config::{
    BenchmarkMode, ContainerConfig, MetricType, SolBenchOptions, SolProofVersion, SolSweepOptions,
    TestConfig,
};
pub use storage::{TestResult, TestStorage};
