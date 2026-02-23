use std::path::{Path, PathBuf};
use std::process::Command;

fn bench_bin() -> &'static str {
    env!("CARGO_BIN_EXE_nockchain-bench")
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("guard")
        .join(name)
}

fn write_contract(path: &Path, min_samples: usize) {
    let contract = format!(
        r#"[metadata]
name = "guard-test"
version = "1"

[baseline]
window_runs = 20
max_age_days = 30
min_samples = {min_samples}

[rules.throughput]
metric = "throughput_blocks_s"
floor_pct_of_baseline = 0.95
severity = "fail"
weight = 1.0

[rules.init]
metric = "init_time_s"
ceiling_pct_of_baseline = 1.10
severity = "warn"
weight = 0.5

[rules.peak]
metric = "peak_rss_mib"
ceiling_pct_of_baseline = 1.08
severity = "fail"
weight = 1.0

[rules.major]
metric = "major_faults_delta"
absolute_ceiling = 2.0
severity = "fail"
weight = 0.8
"#
    );
    std::fs::write(path, contract).expect("write contract");
}

#[test]
fn guard_cli_returns_insufficient_baseline_exit_code() {
    let summary = fixture_path("combined_summary.tsv");
    let contract = fixture_path("contract.toml"); // min_samples=5, fixture only has 2 baseline rows

    let output = Command::new(bench_bin())
        .args([
            "sol",
            "guard",
            "--candidate-summary",
            summary.to_str().expect("summary path"),
            "--contract",
            contract.to_str().expect("contract path"),
            "--env",
            "native",
            "--branch",
            "master",
            "--fixture",
            "v0",
        ])
        .output()
        .expect("run guard");

    assert_eq!(output.status.code(), Some(3));
}

#[test]
fn guard_cli_returns_fail_exit_code() {
    let temp = tempfile::tempdir().expect("tempdir");
    let contract = temp.path().join("contract.toml");
    write_contract(&contract, 2);

    let summary = fixture_path("combined_summary.tsv");
    let output = Command::new(bench_bin())
        .args([
            "sol",
            "guard",
            "--candidate-summary",
            summary.to_str().expect("summary path"),
            "--contract",
            contract.to_str().expect("contract path"),
            "--env",
            "native",
            "--branch",
            "master",
            "--fixture",
            "v0",
        ])
        .output()
        .expect("run guard");

    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn guard_cli_returns_pass_exit_code() {
    let temp = tempfile::tempdir().expect("tempdir");
    let summary = temp.path().join("summary.tsv");
    let contract = temp.path().join("contract.toml");
    write_contract(&contract, 2);

    let tsv = "\
pass\tenv\tbranch\tfixture\tthroughput_blocks_s\tinit_time_s\tpeak_rss_mib\tmajor_faults_delta\tfailed_pokes\texit_status\n\
1\tnative\tmaster\tv0\t10.00\t0.10\t790.0\t1\t0\t0\n\
1\tnative\tmaster\tv0\t10.20\t0.11\t800.0\t1\t0\t0\n\
2\tnative\tmaster\tv0\t10.10\t0.11\t798.0\t1\t0\t0\n";
    std::fs::write(&summary, tsv).expect("write summary");

    let output = Command::new(bench_bin())
        .args([
            "sol",
            "guard",
            "--candidate-summary",
            summary.to_str().expect("summary path"),
            "--contract",
            contract.to_str().expect("contract path"),
            "--env",
            "native",
            "--branch",
            "master",
            "--fixture",
            "v0",
        ])
        .output()
        .expect("run guard");

    assert_eq!(output.status.code(), Some(0));
}
