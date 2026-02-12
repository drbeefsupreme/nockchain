use nockchain_bench_gui::{BenchmarkMode, ContainerConfig, MetricType, TestConfig, TestStorage};

#[test]
fn test_legacy_config_json_defaults_new_fields() {
    let json = r#"
    {
        "id": "6f6d9aaf-2f5f-470f-a308-75d9f0227cd8",
        "name": "legacy config",
        "description": null,
        "containers": [],
        "metrics": ["VmRss"],
        "duration_secs": 300,
        "sample_interval_ms": 1000,
        "tags": [],
        "created_at": "2026-02-12T00:00:00Z"
    }
    "#;

    let config: TestConfig = serde_json::from_str(json).expect("legacy config should deserialize");
    assert_eq!(config.benchmark_mode, BenchmarkMode::Container);
    assert_eq!(config.sol_bench.profile_interval_ms, 500);
    assert_eq!(config.sol_sweep.repeats, 1);
}

#[test]
fn test_sol_sweep_config_roundtrip_via_storage() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = TestStorage::new(temp.path()).expect("storage");

    let mut config = TestConfig::new("sol sweep test");
    config.benchmark_mode = BenchmarkMode::SpeedOfLightSweep;
    config.containers.clear();
    config.metrics = vec![MetricType::VmRss];
    config.sol_sweep.candidates_csv = "a,b".to_string();
    config.sol_sweep.chunk_sizes_csv = "8,16".to_string();
    config.sol_sweep.memory_limits_csv = "8g,16g".to_string();

    storage.save_config(&config).expect("save");
    let loaded = storage.load_config(config.id).expect("load");

    assert_eq!(loaded.benchmark_mode, BenchmarkMode::SpeedOfLightSweep);
    assert_eq!(loaded.sol_sweep.repeats, 1);
    assert_eq!(loaded.sol_sweep.case_count().expect("case count"), 8);
}

#[test]
fn test_container_mode_validation_still_requires_containers() {
    let mut config = TestConfig::new("container");
    config.benchmark_mode = BenchmarkMode::Container;
    config.containers.clear();
    assert!(config.validate().is_err());

    config.containers.push(ContainerConfig::default());
    assert!(config.validate().is_ok());
}
