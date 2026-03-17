use std::fs;
use std::process::Command;

fn script_path() -> &'static str {
    "../../scripts/build_nockchain_bench_image.sh"
}

#[test]
fn build_image_script_help_mentions_standard_and_profiling_variants() {
    let output = Command::new(script_path())
        .arg("--help")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--tag"));
    assert!(stdout.contains("--variant"));
    assert!(stdout.contains("standard"));
    assert!(stdout.contains("profiling"));
}

#[test]
fn profiling_variant_requires_samply_or_explicit_override() {
    let empty_path = tempfile::tempdir().expect("tempdir");
    let binary_dir = tempfile::tempdir().expect("binary tempdir");
    let binary_path = binary_dir.path().join("nockchain-bench");
    fs::write(&binary_path, b"placeholder").expect("write placeholder binary");

    let output = Command::new(script_path())
        .args([
            "--variant",
            "profiling",
            "--tag",
            "example:test",
            "--dry-run",
            "--skip-cargo-build",
            "--binary",
            binary_path.to_str().expect("binary path utf-8"),
        ])
        .env("PATH", empty_path.path())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run script");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("samply"));
}
