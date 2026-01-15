//! Benchmark scenarios for Nockchain
//!
//! Scenarios define structured benchmarks that can be run and compared.

pub mod mining;

pub use mining::{MiningScenario, MiningScenarioConfig, MiningResult, ScenarioError};
