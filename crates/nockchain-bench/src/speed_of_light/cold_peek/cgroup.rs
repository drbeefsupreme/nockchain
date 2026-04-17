use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fs, io};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColdStepOptions {
    pub tolerance_pages: u64,
    pub max_attempts: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdForceResult {
    pub cold_verified: bool,
    pub residency_pages_after: u64,
    pub residency_total_pages: u64,
    pub cold_attempts: u32,
    pub degraded_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ColdStepError {
    #[error("cold verify failed after {cold_attempts} attempts: {message}")]
    VerifyFailed {
        residency_pages_after: u64,
        residency_total_pages: u64,
        tolerance_pages: u64,
        cold_attempts: u32,
        message: String,
    },

    #[error("{0}")]
    System(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ColdInitError {
    #[error("cold peek requires cgroup v2 memory.reclaim support")]
    ReclaimUnsupported,

    #[error("cold peek requires memory.reclaim swappiness support; found kernel {found_kernel}")]
    SwappinessKeyUnsupported { found_kernel: String },

    #[error(
        "cold peek requires a delegated cgroup v2 parent with memory in cgroup.subtree_control"
    )]
    NoDelegatedMemory,

    #[error("failed to create cold peek leaf cgroup {path}: errno {errno}")]
    LeafCreateFailed { errno: i32, path: PathBuf },

    #[error("failed to probe memory.reclaim: errno {errno}")]
    ReclaimProbeFailed { errno: i32 },

    #[error("no PMA VMAs discovered under replay-pma")]
    NoPmaVmas,
}

#[derive(Debug)]
struct LeafCgroup {
    parent: PathBuf,
    leaf: PathBuf,
    pid: u32,
}

impl LeafCgroup {
    fn new(parent: PathBuf, leaf: PathBuf, pid: u32) -> Self {
        Self { parent, leaf, pid }
    }

    fn cleanup(&self) {
        let _ = fs::write(self.parent.join("cgroup.procs"), format!("{}\n", self.pid));
        let _ = fs::remove_dir(&self.leaf);
    }
}

impl Drop for LeafCgroup {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[derive(Debug)]
pub struct ColdRuntime {
    _leaf: LeafCgroup,
}

impl ColdRuntime {
    pub fn startup_if_needed(
        has_cold_steps: bool,
        _cold_mode: crate::speed_of_light::ColdMode,
    ) -> Result<Option<Self>, ColdInitError> {
        if !has_cold_steps {
            return Ok(None);
        }

        let parent = own_cgroup_path()?;
        ensure_memory_delegated(&parent)?;
        sweep_empty_bench_leaves(&parent);

        let pid = std::process::id();
        let leaf = parent.join(bench_leaf_name(pid));
        fs::create_dir(&leaf).map_err(|source| ColdInitError::LeafCreateFailed {
            errno: source.raw_os_error().unwrap_or(libc::EIO),
            path: leaf.clone(),
        })?;

        probe_memory_reclaim(&leaf)?;
        fs::write(leaf.join("cgroup.procs"), format!("{pid}\n")).map_err(|source| {
            classify_reclaim_probe_error(source, false, kernel_release_string())
        })?;

        Ok(Some(Self {
            _leaf: LeafCgroup::new(parent, leaf, pid),
        }))
    }

    pub fn bind_after_boot(&mut self, _work_dir: &Path, _fsync: bool) -> Result<(), ColdInitError> {
        Ok(())
    }

    pub fn force_cold(
        &mut self,
        _options: ColdStepOptions,
    ) -> Result<ColdForceResult, ColdStepError> {
        Err(ColdStepError::System(
            "cold step execution is not wired until Task 4".to_string(),
        ))
    }
}

pub fn own_cgroup_path() -> Result<PathBuf, ColdInitError> {
    let contents =
        fs::read_to_string("/proc/self/cgroup").map_err(|_| ColdInitError::ReclaimUnsupported)?;
    own_cgroup_path_from_str(&contents)
}

fn own_cgroup_path_from_str(contents: &str) -> Result<PathBuf, ColdInitError> {
    for line in contents.lines() {
        let mut parts = line.splitn(3, ':');
        let hierarchy = parts.next();
        let _controllers = parts.next();
        let path = parts.next();
        if hierarchy == Some("0") {
            let relative = path.unwrap_or_default().trim_start_matches('/');
            return Ok(PathBuf::from("/sys/fs/cgroup").join(relative));
        }
    }
    Err(ColdInitError::ReclaimUnsupported)
}

pub fn parse_subtree_control_tokens(contents: &str) -> Vec<&str> {
    contents
        .split_whitespace()
        .map(|token| token.trim_start_matches(['+', '-']))
        .filter(|token| !token.is_empty())
        .collect()
}

fn subtree_control_has_controller(contents: &str, controller: &str) -> bool {
    parse_subtree_control_tokens(contents)
        .into_iter()
        .any(|token| token == controller)
}

fn ensure_memory_delegated(parent: &Path) -> Result<(), ColdInitError> {
    let contents = fs::read_to_string(parent.join("cgroup.subtree_control"))
        .map_err(|_| ColdInitError::NoDelegatedMemory)?;
    if subtree_control_has_controller(&contents, "memory") {
        Ok(())
    } else {
        Err(ColdInitError::NoDelegatedMemory)
    }
}

fn sweep_empty_bench_leaves(parent: &Path) {
    let Ok(entries) = fs::read_dir(parent) else {
        return;
    };

    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if !file_name.starts_with("bench-") {
            continue;
        }

        let cgroup_procs = entry.path().join("cgroup.procs");
        let Ok(contents) = fs::read_to_string(&cgroup_procs) else {
            continue;
        };
        if !contents.trim().is_empty() {
            continue;
        }

        let _ = fs::remove_dir(entry.path());
    }
}

fn probe_memory_reclaim(leaf: &Path) -> Result<(), ColdInitError> {
    let reclaim_path = leaf.join("memory.reclaim");
    fs::write(&reclaim_path, "0")
        .map_err(|source| classify_reclaim_probe_error(source, false, kernel_release_string()))?;
    fs::write(&reclaim_path, "0 swappiness=0")
        .map_err(|source| classify_reclaim_probe_error(source, true, kernel_release_string()))
}

fn classify_reclaim_probe_error(
    source: io::Error,
    swappiness_probe: bool,
    kernel_release: String,
) -> ColdInitError {
    let errno = source.raw_os_error().unwrap_or(libc::EIO);
    classify_reclaim_probe_errno(errno, swappiness_probe, kernel_release)
}

fn classify_reclaim_probe_errno(
    errno: i32,
    swappiness_probe: bool,
    kernel_release: String,
) -> ColdInitError {
    if errno == libc::EINVAL && swappiness_probe {
        ColdInitError::SwappinessKeyUnsupported {
            found_kernel: kernel_release,
        }
    } else if errno == libc::ENOENT {
        ColdInitError::ReclaimUnsupported
    } else {
        ColdInitError::ReclaimProbeFailed { errno }
    }
}

fn bench_leaf_name(pid: u32) -> String {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    format!("bench-{pid}-{seed:08x}")
}

fn kernel_release_string() -> String {
    let mut uts = std::mem::MaybeUninit::<libc::utsname>::uninit();
    let ret = unsafe { libc::uname(uts.as_mut_ptr()) };
    if ret != 0 {
        return "unknown".to_string();
    }
    let uts = unsafe { uts.assume_init() };
    unsafe { CStr::from_ptr(uts.release.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_cgroup_path_uses_v2_entry() {
        let path = own_cgroup_path_from_str("0::/user.slice/user-1000.slice/session.scope\n")
            .expect("cgroup path");
        assert_eq!(
            path,
            PathBuf::from("/sys/fs/cgroup/user.slice/user-1000.slice/session.scope")
        );
    }

    #[test]
    fn subtree_control_parser_keeps_exact_tokens() {
        let tokens = parse_subtree_control_tokens("+cpu +memory +io");
        assert_eq!(tokens, vec!["cpu", "memory", "io"]);
    }

    #[test]
    fn subtree_control_memory_detection_is_exact_token_match() {
        assert!(subtree_control_has_controller("+cpu +memory +io", "memory"));
        assert!(!subtree_control_has_controller(
            "+cpu +memoryswap +io", "memory"
        ));
        assert!(!subtree_control_has_controller(
            "+cpu +mem ory +io", "memory"
        ));
    }

    #[test]
    fn reclaim_probe_einval_on_swappiness_maps_to_specific_error() {
        let error = classify_reclaim_probe_errno(libc::EINVAL, true, "6.10.0-test".to_string());
        assert_eq!(
            error,
            ColdInitError::SwappinessKeyUnsupported {
                found_kernel: "6.10.0-test".to_string()
            }
        );
    }

    #[test]
    fn reclaim_probe_eacces_maps_to_generic_probe_failure() {
        let error = classify_reclaim_probe_errno(libc::EACCES, true, "6.11.0-test".to_string());
        assert_eq!(
            error,
            ColdInitError::ReclaimProbeFailed {
                errno: libc::EACCES
            }
        );
    }
}
