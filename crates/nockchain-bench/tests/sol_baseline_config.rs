//! Contract tests for the baseline configuration system.
//!
//! These tests verify the TOML config file structure, profile resolution,
//! shell variable dumping, and config hashing.

use nockchain_bench::speed_of_light::config::{config_sha256, dump_shell_vars, load_config};

fn config_path() -> std::path::PathBuf {
    // Integration tests run from the crate directory; config is at workspace root.
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // crates/
    path.pop(); // workspace root
    path.push("benchmarks/baseline/sol-baseline.toml");
    path
}

#[test]
fn test_config_file_exists_and_is_valid_toml() {
    let path = &config_path();
    assert!(path.exists(), "Config file must exist at {:?}", path);
    let content = std::fs::read_to_string(path).expect("Should read config file");
    let _: toml::Value = toml::from_str(&content).expect("Config must be valid TOML");
}

#[test]
fn test_quick_profile_has_fewer_passes() {
    let quick = load_config(&config_path(), "quick").expect("quick profile");
    let full = load_config(&config_path(), "full").expect("full profile");
    assert!(
        quick.passes < full.passes,
        "Quick profile passes ({}) should be less than full ({})",
        quick.passes,
        full.passes,
    );
}

#[test]
fn test_full_profile_overrides_envs() {
    let full = load_config(&config_path(), "full").expect("full profile");
    assert!(
        full.envs.contains("docker"),
        "Full profile envs should include docker, got: {}",
        full.envs
    );
}

#[test]
fn test_dump_shell_vars_format() {
    let config = load_config(&config_path(), "quick").expect("quick profile");
    let output = dump_shell_vars(&config);
    assert!(
        output.contains("PASSES="),
        "Shell vars must contain PASSES="
    );
    assert!(
        output.contains("FIXTURES_DIR="),
        "Shell vars must contain FIXTURES_DIR="
    );
    assert!(
        output.contains("OUTPUT_ROOT="),
        "Shell vars must contain OUTPUT_ROOT="
    );
    assert!(
        output.contains("ENABLE_CHECKPOINTING="),
        "Shell vars must contain ENABLE_CHECKPOINTING="
    );
}

#[test]
fn test_config_sha256_is_deterministic() {
    let hash1 = config_sha256(&config_path()).expect("first hash");
    let hash2 = config_sha256(&config_path()).expect("second hash");
    assert_eq!(hash1, hash2, "SHA-256 must be deterministic");
    assert_eq!(hash1.len(), 64, "SHA-256 hex must be 64 characters");
}
