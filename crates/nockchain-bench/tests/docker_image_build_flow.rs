use std::fs;
use std::process::Command;

fn script_path() -> &'static str {
    "../../scripts/build_nockchain_bench_image.sh"
}

fn prepend_path(dir: &std::path::Path) -> std::ffi::OsString {
    let mut combined = std::ffi::OsString::new();
    combined.push(dir.as_os_str());
    combined.push(":");
    combined.push(std::env::var_os("PATH").expect("PATH"));
    combined
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

#[test]
fn standard_variant_completes_successfully_with_mocked_docker() {
    let bin_dir = tempfile::tempdir().expect("bin tempdir");
    let docker_path = bin_dir.path().join("docker");
    fs::write(
        &docker_path,
        "#!/bin/sh\nprintf 'docker %s\\n' \"$*\" > \"$MOCK_DOCKER_LOG\"\n",
    )
    .expect("write fake docker");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&docker_path)
            .expect("docker metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&docker_path, permissions).expect("chmod fake docker");
    }

    let binary_dir = tempfile::tempdir().expect("binary tempdir");
    let binary_path = binary_dir.path().join("nockchain-bench");
    fs::write(&binary_path, b"placeholder").expect("write placeholder binary");

    let log_path = binary_dir.path().join("docker.log");
    let output = Command::new(script_path())
        .args([
            "--variant",
            "standard",
            "--tag",
            "example:test",
            "--skip-cargo-build",
            "--binary",
            binary_path.to_str().expect("binary path utf-8"),
        ])
        .env("PATH", prepend_path(bin_dir.path()))
        .env("MOCK_DOCKER_LOG", &log_path)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run script");

    assert!(output.status.success(), "{output:?}");
    let docker_log = fs::read_to_string(log_path).expect("read docker log");
    assert!(docker_log.contains("build -t example:test"));
}

#[test]
fn profiling_variant_stages_samply_and_uses_profiling_dockerfile() {
    let bin_dir = tempfile::tempdir().expect("bin tempdir");
    let docker_path = bin_dir.path().join("docker");
    fs::write(
        &docker_path,
        "#!/bin/sh\nfor last do :; done\ncontext=\"$last\"\n[ -f \"$context/samply\" ] || { echo 'missing staged samply' >&2; exit 1; }\nprintf 'docker %s\\n' \"$*\" > \"$MOCK_DOCKER_LOG\"\n",
    )
    .expect("write fake docker");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&docker_path)
            .expect("docker metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&docker_path, permissions).expect("chmod fake docker");
    }

    let binary_dir = tempfile::tempdir().expect("binary tempdir");
    let binary_path = binary_dir.path().join("nockchain-bench");
    let samply_path = binary_dir.path().join("samply");
    fs::write(&binary_path, b"placeholder").expect("write placeholder binary");
    fs::write(&samply_path, b"placeholder").expect("write placeholder samply");

    let log_path = binary_dir.path().join("docker.log");
    let output = Command::new(script_path())
        .args([
            "--variant",
            "profiling",
            "--tag",
            "example:test",
            "--skip-cargo-build",
            "--binary",
            binary_path.to_str().expect("binary path utf-8"),
            "--samply-bin",
            samply_path.to_str().expect("samply path utf-8"),
        ])
        .env("PATH", prepend_path(bin_dir.path()))
        .env("MOCK_DOCKER_LOG", &log_path)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run script");

    assert!(output.status.success(), "{output:?}");
    let docker_log = fs::read_to_string(log_path).expect("read docker log");
    assert!(docker_log.contains("Dockerfile.profiling"));
}
