use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use super::HarnessError;

const DEFAULT_TRACY_CAPTURE_BIN: &str = "tracy-capture";
const TRACY_CAPTURE_BIN_ENV: &str = "NOCKCHAIN_BENCH_TRACY_CAPTURE_BIN";
const DEFAULT_TRACY_HOST: &str = "127.0.0.1";
const DEFAULT_TRACY_PORT: u16 = 8086;
const STOP_TIMEOUT: Duration = Duration::from_secs(10);
const EARLY_EXIT_GRACE_PERIOD: Duration = Duration::from_millis(200);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TracyEndpoint {
    pub host: String,
    pub port: u16,
}

impl TracyEndpoint {
    pub fn native() -> Self {
        Self {
            host: DEFAULT_TRACY_HOST.to_string(),
            port: DEFAULT_TRACY_PORT,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TracyCapturePlan {
    pub program: String,
    pub args: Vec<String>,
}

pub struct RunningTracyCapture {
    child: Child,
    output_path: PathBuf,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
}

pub fn ensure_tracy_capture_available() -> Result<(), HarnessError> {
    let tracy_capture_bin = tracy_capture_bin();
    Command::new(&tracy_capture_bin)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .map_err(|error| {
            HarnessError::CommandFailure(format!(
                "failed to execute {}: {error}",
                tracy_capture_bin
            ))
        })?;
    Ok(())
}

pub fn reserve_loopback_port() -> Result<u16, HarnessError> {
    let listener = TcpListener::bind((DEFAULT_TRACY_HOST, 0))?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

pub fn build_capture_plan(output_path: &Path, endpoint: &TracyEndpoint) -> TracyCapturePlan {
    build_capture_plan_with_follow_flag(output_path, endpoint, true)
}

pub fn build_native_capture_plan(
    output_path: &Path,
    endpoint: &TracyEndpoint,
) -> TracyCapturePlan {
    build_capture_plan_with_follow_flag(output_path, endpoint, false)
}

fn build_capture_plan_with_follow_flag(
    output_path: &Path,
    endpoint: &TracyEndpoint,
    follow_flag: bool,
) -> TracyCapturePlan {
    let mut args = Vec::new();
    if follow_flag {
        args.push("-f".to_string());
    }
    args.extend([
        "-o".to_string(),
        output_path.to_string_lossy().to_string(),
        "-a".to_string(),
        endpoint.host.clone(),
        "-p".to_string(),
        endpoint.port.to_string(),
    ]);

    TracyCapturePlan {
        program: tracy_capture_bin(),
        args,
    }
}

pub fn start_tracy_capture(
    output_path: &Path,
    endpoint: &TracyEndpoint,
) -> Result<RunningTracyCapture, HarnessError> {
    let plan = build_capture_plan(output_path, endpoint);
    start_tracy_capture_with_plan(output_path, plan, false)
}

pub fn start_native_tracy_capture(
    output_path: &Path,
    endpoint: &TracyEndpoint,
) -> Result<RunningTracyCapture, HarnessError> {
    let plan = build_native_capture_plan(output_path, endpoint);
    start_tracy_capture_with_plan(output_path, plan, true)
}

fn start_tracy_capture_with_plan(
    output_path: &Path,
    plan: TracyCapturePlan,
    use_shell_wrapper: bool,
) -> Result<RunningTracyCapture, HarnessError> {
    ensure_tracy_capture_available()?;
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = std::fs::remove_file(output_path);
    let (stdout_path, stderr_path) = capture_log_paths(output_path);
    std::fs::write(&stdout_path, b"")?;
    std::fs::write(&stderr_path, b"")?;

    let stdout = std::fs::File::create(&stdout_path)?;
    let stderr = std::fs::File::create(&stderr_path)?;
    let child = if use_shell_wrapper {
        Command::new("/bin/sh")
            .arg("-c")
            .arg("exec \"$0\" \"$@\"")
            .arg(&plan.program)
            .args(&plan.args)
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|error| {
                HarnessError::CommandFailure(format!(
                    "failed to start Tracy capture (/bin/sh -c exec \"$0\" \"$@\" {}): {error}",
                    shell_join_command(&plan)
                ))
            })?
    } else {
        Command::new(&plan.program)
            .args(&plan.args)
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|error| {
                HarnessError::CommandFailure(format!(
                    "failed to start Tracy capture ({} {}): {error}",
                    plan.program,
                    plan.args.join(" ")
                ))
            })?
    };

    Ok(RunningTracyCapture {
        child,
        output_path: output_path.to_path_buf(),
        stdout_path,
        stderr_path,
    })
}

impl RunningTracyCapture {
    pub fn ensure_started(&mut self) -> Result<(), HarnessError> {
        std::thread::sleep(EARLY_EXIT_GRACE_PERIOD);
        self.ensure_running()
    }

    pub fn ensure_running(&mut self) -> Result<(), HarnessError> {
        if let Some(status) = self.child.try_wait()? {
            return Err(capture_process_failure(
                status,
                &self.stdout_path,
                &self.stderr_path,
            ));
        }
        Ok(())
    }

    pub fn stop(self) -> Result<(), HarnessError> {
        self.stop_with_grace_period(Duration::ZERO)
    }

    pub fn wait_for_natural_exit(mut self, timeout: Duration) -> Result<(), HarnessError> {
        wait_for_capture_exit_or_timeout(
            &mut self.child,
            &self.stdout_path,
            &self.stderr_path,
            timeout,
        )?;
        validate_capture_file(&self.output_path)
    }

    pub fn stop_with_grace_period(
        mut self,
        natural_exit_grace_period: Duration,
    ) -> Result<(), HarnessError> {
        if wait_for_capture_exit(
            &mut self.child,
            &self.stdout_path,
            &self.stderr_path,
            natural_exit_grace_period,
        )? {
            return validate_capture_file(&self.output_path);
        }

        if self.child.try_wait()?.is_none() {
            send_sigint(&mut self.child)?;
        }
        wait_for_capture_exit_or_timeout(
            &mut self.child,
            &self.stdout_path,
            &self.stderr_path,
            STOP_TIMEOUT,
        )?;
        validate_capture_file(&self.output_path)
    }
}

fn validate_capture_file(path: &Path) -> Result<(), HarnessError> {
    let metadata = std::fs::metadata(path).map_err(|error| {
        HarnessError::CommandFailure(format!(
            "Tracy capture file {} missing: {error}",
            path.display()
        ))
    })?;
    if metadata.len() == 0 {
        return Err(HarnessError::CommandFailure(format!(
            "Tracy capture file {} is empty",
            path.display()
        )));
    }
    Ok(())
}

fn send_sigint(child: &mut Child) -> Result<(), HarnessError> {
    let pid = child.id() as i32;
    let rc = unsafe { libc::kill(pid, libc::SIGINT) };
    if rc == 0 {
        Ok(())
    } else {
        Err(HarnessError::CommandFailure(format!(
            "failed to stop Tracy capture process {pid}"
        )))
    }
}

fn wait_for_capture_exit(
    child: &mut Child,
    stdout_path: &Path,
    stderr_path: &Path,
    timeout: Duration,
) -> Result<bool, HarnessError> {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            if status.success() {
                return Ok(true);
            }
            return Err(capture_process_failure(status, stdout_path, stderr_path));
        }

        if start.elapsed() >= timeout {
            return Ok(false);
        }

        std::thread::sleep(Duration::from_millis(50));
    }
}

fn wait_for_capture_exit_or_timeout(
    child: &mut Child,
    stdout_path: &Path,
    stderr_path: &Path,
    timeout: Duration,
) -> Result<(), HarnessError> {
    if wait_for_capture_exit(child, stdout_path, stderr_path, timeout)? {
        Ok(())
    } else {
        let _ = child.kill();
        let _ = child.wait();
        Err(HarnessError::CommandFailure(
            "timed out waiting for Tracy capture process to exit".to_string(),
        ))
    }
}

fn capture_process_failure(
    status: std::process::ExitStatus,
    stdout_path: &Path,
    stderr_path: &Path,
) -> HarnessError {
    let stdout = read_capture_log(stdout_path);
    let stderr = read_capture_log(stderr_path);
    let mut details = Vec::new();
    if let Some(stdout) = stdout {
        details.push(format!("stdout: {stdout}"));
    }
    if let Some(stderr) = stderr {
        details.push(format!("stderr: {stderr}"));
    }
    let detail_suffix = if details.is_empty() {
        String::new()
    } else {
        format!(" ({})", details.join("; "))
    };
    HarnessError::CommandFailure(format!(
        "Tracy capture process exited with status {status}{detail_suffix}"
    ))
}

fn read_capture_log(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn capture_log_paths(output_path: &Path) -> (PathBuf, PathBuf) {
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = output_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("tracy_capture");
    (
        parent.join(format!("{stem}.stdout.log")),
        parent.join(format!("{stem}.stderr.log")),
    )
}

fn shell_join_command(plan: &TracyCapturePlan) -> String {
    let mut parts = Vec::with_capacity(1 + plan.args.len());
    parts.push(plan.program.clone());
    parts.extend(plan.args.clone());
    parts.join(" ")
}

fn tracy_capture_bin() -> String {
    std::env::var(TRACY_CAPTURE_BIN_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_TRACY_CAPTURE_BIN.to_string())
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn capture_plan_uses_expected_args() {
        let plan = build_capture_plan(
            Path::new("/tmp/run-0/tracy_capture.tracy"),
            &TracyEndpoint::native(),
        );

        assert_eq!(plan.program, "tracy-capture");
        assert_eq!(
            plan.args,
            vec![
                "-f",
                "-o",
                "/tmp/run-0/tracy_capture.tracy",
                "-a",
                "127.0.0.1",
                "-p",
                "8086",
            ]
        );
    }

    #[test]
    fn native_capture_plan_omits_follow_flag() {
        let plan = build_native_capture_plan(
            Path::new("/tmp/run-0/tracy_capture.tracy"),
            &TracyEndpoint::native(),
        );

        assert_eq!(plan.program, "tracy-capture");
        assert_eq!(
            plan.args,
            vec![
                "-o",
                "/tmp/run-0/tracy_capture.tracy",
                "-a",
                "127.0.0.1",
                "-p",
                "8086",
            ]
        );
    }

    #[test]
    fn capture_plan_uses_env_override_binary() {
        let tempdir = tempdir().expect("tempdir");
        let override_path = tempdir.path().join("tracy-capture-bin");
        std::fs::write(&override_path, "#!/bin/sh\nexit 0\n").expect("write override");
        let mut permissions = std::fs::metadata(&override_path)
            .expect("metadata")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&override_path, permissions).expect("chmod");

        let old = std::env::var(TRACY_CAPTURE_BIN_ENV).ok();
        std::env::set_var(TRACY_CAPTURE_BIN_ENV, &override_path);

        let plan = build_capture_plan(
            Path::new("/tmp/run-0/tracy_capture.tracy"),
            &TracyEndpoint::native(),
        );

        if let Some(old) = old {
            std::env::set_var(TRACY_CAPTURE_BIN_ENV, old);
        } else {
            std::env::remove_var(TRACY_CAPTURE_BIN_ENV);
        }

        assert_eq!(plan.program, override_path.to_string_lossy());
    }

    #[test]
    fn tracy_capture_stop_reports_child_stdout_on_failure() {
        let tempdir = tempdir().expect("tempdir");
        let fake_capture = tempdir.path().join("fake-capture.sh");
        std::fs::write(
            &fake_capture,
            "#!/bin/sh\nprintf 'protocol mismatch\\n'\nexit 1\n",
        )
        .expect("write fake capture");
        let mut permissions = std::fs::metadata(&fake_capture)
            .expect("metadata")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&fake_capture, permissions).expect("chmod");

        let old = std::env::var(TRACY_CAPTURE_BIN_ENV).ok();
        std::env::set_var(TRACY_CAPTURE_BIN_ENV, &fake_capture);

        let output_path = tempdir.path().join("tracy_capture.tracy");
        let running = start_tracy_capture(&output_path, &TracyEndpoint::native())
            .expect("start fake capture");
        let mut running = running;
        let error = running
            .ensure_started()
            .expect_err("capture should fail immediately");

        if let Some(old) = old {
            std::env::set_var(TRACY_CAPTURE_BIN_ENV, old);
        } else {
            std::env::remove_var(TRACY_CAPTURE_BIN_ENV);
        }

        let message = error.to_string();
        assert!(message.contains("protocol mismatch"), "{message}");
        assert!(output_path.with_file_name("tracy_capture.stdout.log").exists());
        assert!(output_path.with_file_name("tracy_capture.stderr.log").exists());
    }

    #[test]
    fn tracy_capture_stop_with_grace_period_allows_natural_exit() {
        let tempdir = tempdir().expect("tempdir");
        let fake_capture = tempdir.path().join("fake-capture.sh");
        std::fs::write(
            &fake_capture,
            "#!/bin/sh\n\
if [ \"$1\" = \"--help\" ]; then\n\
  exit 0\n\
fi\n\
trap 'exit 91' INT\n\
output=''\n\
while [ $# -gt 0 ]; do\n\
  if [ \"$1\" = \"-o\" ]; then\n\
    shift\n\
    output=\"$1\"\n\
  fi\n\
  shift\n\
done\n\
sleep 0.3\n\
printf 'trace-bytes' > \"$output\"\n\
exit 0\n",
        )
        .expect("write fake capture");
        let mut permissions = std::fs::metadata(&fake_capture)
            .expect("metadata")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&fake_capture, permissions).expect("chmod");

        let old = std::env::var(TRACY_CAPTURE_BIN_ENV).ok();
        std::env::set_var(TRACY_CAPTURE_BIN_ENV, &fake_capture);

        let output_path = tempdir.path().join("tracy_capture.tracy");
        let running = start_native_tracy_capture(&output_path, &TracyEndpoint::native())
            .expect("start fake capture");
        let result = running.stop_with_grace_period(Duration::from_secs(1));

        if let Some(old) = old {
            std::env::set_var(TRACY_CAPTURE_BIN_ENV, old);
        } else {
            std::env::remove_var(TRACY_CAPTURE_BIN_ENV);
        }

        result.expect("capture should exit cleanly before SIGINT");
        let bytes = std::fs::read(&output_path).expect("trace bytes");
        assert_eq!(bytes, b"trace-bytes");
    }

    #[test]
    fn tracy_capture_waits_for_natural_exit_without_signal() {
        let tempdir = tempdir().expect("tempdir");
        let fake_capture = tempdir.path().join("fake-capture.sh");
        std::fs::write(
            &fake_capture,
            "#!/bin/sh\n\
if [ \"$1\" = \"--help\" ]; then\n\
  exit 0\n\
fi\n\
trap 'exit 91' INT\n\
output=''\n\
while [ $# -gt 0 ]; do\n\
  if [ \"$1\" = \"-o\" ]; then\n\
    shift\n\
    output=\"$1\"\n\
  fi\n\
  shift\n\
done\n\
sleep 0.3\n\
printf 'trace-bytes' > \"$output\"\n\
exit 0\n",
        )
        .expect("write fake capture");
        let mut permissions = std::fs::metadata(&fake_capture)
            .expect("metadata")
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&fake_capture, permissions).expect("chmod");

        let old = std::env::var(TRACY_CAPTURE_BIN_ENV).ok();
        std::env::set_var(TRACY_CAPTURE_BIN_ENV, &fake_capture);

        let output_path = tempdir.path().join("tracy_capture.tracy");
        let running = start_native_tracy_capture(&output_path, &TracyEndpoint::native())
            .expect("start fake capture");
        let result = running.wait_for_natural_exit(Duration::from_secs(1));

        if let Some(old) = old {
            std::env::set_var(TRACY_CAPTURE_BIN_ENV, old);
        } else {
            std::env::remove_var(TRACY_CAPTURE_BIN_ENV);
        }

        result.expect("capture should exit cleanly without SIGINT");
        let bytes = std::fs::read(&output_path).expect("trace bytes");
        assert_eq!(bytes, b"trace-bytes");
    }
}
