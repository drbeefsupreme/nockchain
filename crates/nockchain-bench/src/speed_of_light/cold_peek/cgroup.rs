use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fs, io};

use super::vma::{read_pma_vmas, resident_pages, Vma};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffendingVmaResidency {
    pub path: PathBuf,
    pub resident_pages: u64,
    pub total_pages: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ColdStepError {
    #[error("cold verify failed after {cold_attempts} attempts: {message}")]
    VerifyFailed {
        residency_pages_after: u64,
        residency_total_pages: u64,
        tolerance_pages: u64,
        cold_attempts: u32,
        offending_vma: Option<OffendingVmaResidency>,
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

    fn reclaim_path(&self) -> PathBuf {
        self.leaf.join("memory.reclaim")
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
    leaf: LeafCgroup,
    fsync: bool,
    cold_mode: crate::speed_of_light::ColdMode,
    vmas: Vec<Vma>,
}

impl ColdRuntime {
    pub fn startup_if_needed(
        has_cold_steps: bool,
        cold_mode: crate::speed_of_light::ColdMode,
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
        fs::write(leaf.join("cgroup.procs"), format!("{pid}\n"))
            .map_err(|source| classify_leaf_join_error(source, &leaf))?;

        Ok(Some(Self {
            leaf: LeafCgroup::new(parent, leaf, pid),
            fsync: false,
            cold_mode,
            vmas: Vec::new(),
        }))
    }

    pub fn bind_after_boot(&mut self, work_dir: &Path, fsync: bool) -> Result<(), ColdInitError> {
        let vmas = read_pma_vmas(work_dir).map_err(|_| ColdInitError::NoPmaVmas)?;
        if vmas.is_empty() {
            return Err(ColdInitError::NoPmaVmas);
        }

        self.fsync = fsync;
        self.vmas = vmas;
        Ok(())
    }

    pub fn force_cold(
        &mut self,
        options: ColdStepOptions,
    ) -> Result<ColdForceResult, ColdStepError> {
        if self.vmas.is_empty() {
            return Err(ColdStepError::System(
                "cold runtime not bound to PMA VMAs after boot".to_string(),
            ));
        }

        let mut ops = LiveColdOps::new(self.leaf.reclaim_path());
        force_cold_with_ops(&mut ops, &self.vmas, self.fsync, options, self.cold_mode)
    }
}

trait ColdOps {
    fn sync_vmas(&mut self, vmas: &[Vma]) -> Result<(), ColdStepError>;
    fn pageout_vmas(&mut self, vmas: &[Vma]) -> Result<(), ColdStepError>;
    fn reclaim(&mut self, bytes: u64) -> Result<(), ColdStepError>;
    fn verify(&mut self, vmas: &[Vma]) -> Result<ColdVerifySummary, ColdStepError>;
}

struct LiveColdOps {
    reclaim_path: PathBuf,
}

impl LiveColdOps {
    fn new(reclaim_path: PathBuf) -> Self {
        Self { reclaim_path }
    }
}

impl ColdOps for LiveColdOps {
    fn sync_vmas(&mut self, vmas: &[Vma]) -> Result<(), ColdStepError> {
        for vma in vmas {
            let ret =
                unsafe { libc::msync(vma.start as *mut libc::c_void, vma.len(), libc::MS_SYNC) };
            if ret != 0 {
                return Err(ColdStepError::System(format!(
                    "msync(MS_SYNC) failed for {}: {}",
                    vma.path.display(),
                    io::Error::last_os_error()
                )));
            }
        }
        Ok(())
    }

    fn pageout_vmas(&mut self, vmas: &[Vma]) -> Result<(), ColdStepError> {
        for vma in vmas {
            let ret = unsafe {
                libc::madvise(
                    vma.start as *mut libc::c_void,
                    vma.len(),
                    libc::MADV_PAGEOUT,
                )
            };
            if ret != 0 {
                return Err(ColdStepError::System(format!(
                    "madvise(MADV_PAGEOUT) failed for {}: {}",
                    vma.path.display(),
                    io::Error::last_os_error()
                )));
            }
        }
        Ok(())
    }

    fn reclaim(&mut self, bytes: u64) -> Result<(), ColdStepError> {
        let reclaim_request = format!("{bytes} swappiness=0");
        match fs::write(&self.reclaim_path, reclaim_request) {
            Ok(()) => Ok(()),
            Err(source) if source.raw_os_error() == Some(libc::EAGAIN) => Ok(()),
            Err(source) => Err(ColdStepError::System(format!(
                "memory.reclaim failed for {}: {}",
                self.reclaim_path.display(),
                source
            ))),
        }
    }

    fn verify(&mut self, vmas: &[Vma]) -> Result<ColdVerifySummary, ColdStepError> {
        let mut resident = 0u64;
        let mut total = 0u64;
        let mut offending_vma = None;

        for vma in vmas {
            let (vma_resident, vma_total) = resident_pages(vma).map_err(|source| {
                ColdStepError::System(format!(
                    "mincore verify failed for {}: {}",
                    vma.path.display(),
                    source
                ))
            })?;
            resident = resident.saturating_add(vma_resident as u64);
            total = total.saturating_add(vma_total as u64);
            if offending_vma.is_none() && vma_resident > 0 {
                offending_vma = Some(OffendingVmaResidency {
                    path: vma.path.clone(),
                    resident_pages: vma_resident as u64,
                    total_pages: vma_total as u64,
                });
            }
        }

        Ok(ColdVerifySummary {
            residency_pages_after: resident,
            residency_total_pages: total,
            offending_vma,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ColdVerifySummary {
    residency_pages_after: u64,
    residency_total_pages: u64,
    offending_vma: Option<OffendingVmaResidency>,
}

fn force_cold_with_ops(
    ops: &mut impl ColdOps,
    vmas: &[Vma],
    fsync: bool,
    options: ColdStepOptions,
    cold_mode: crate::speed_of_light::ColdMode,
) -> Result<ColdForceResult, ColdStepError> {
    let max_attempts = options.max_attempts.max(1);
    if fsync {
        ops.sync_vmas(vmas)?;
    }

    let reclaim_bytes = vmas.iter().fold(0u64, |sum, vma| {
        sum.saturating_add(vma.len().try_into().unwrap_or(u64::MAX))
    });

    let mut last_summary = ColdVerifySummary {
        residency_pages_after: 0,
        residency_total_pages: 0,
        offending_vma: None,
    };
    for attempt in 1..=max_attempts {
        ops.pageout_vmas(vmas)?;
        ops.reclaim(reclaim_bytes)?;
        let summary = ops.verify(vmas)?;
        last_summary = summary;

        if last_summary.residency_pages_after <= options.tolerance_pages {
            return Ok(ColdForceResult {
                cold_verified: true,
                residency_pages_after: last_summary.residency_pages_after,
                residency_total_pages: last_summary.residency_total_pages,
                cold_attempts: attempt,
                degraded_reason: None,
            });
        }
    }

    if matches!(cold_mode, crate::speed_of_light::ColdMode::Soft) {
        return Ok(ColdForceResult {
            cold_verified: false,
            residency_pages_after: last_summary.residency_pages_after,
            residency_total_pages: last_summary.residency_total_pages,
            cold_attempts: max_attempts,
            degraded_reason: None,
        });
    }

    Err(ColdStepError::VerifyFailed {
        residency_pages_after: last_summary.residency_pages_after,
        residency_total_pages: last_summary.residency_total_pages,
        tolerance_pages: options.tolerance_pages,
        cold_attempts: max_attempts,
        offending_vma: last_summary.offending_vma.clone(),
        message: build_verify_failed_message(
            options.tolerance_pages,
            last_summary.residency_pages_after,
            last_summary.residency_total_pages,
            last_summary.offending_vma.as_ref(),
        ),
    })
}

fn build_verify_failed_message(
    tolerance_pages: u64,
    residency_pages_after: u64,
    residency_total_pages: u64,
    offending_vma: Option<&OffendingVmaResidency>,
) -> String {
    let aggregate = format!(
        "resident_pages_after={residency_pages_after}/{residency_total_pages} exceeded tolerance_pages={tolerance_pages}"
    );
    match offending_vma {
        Some(offending_vma) => format!(
            "offending_vma={} resident_pages={}/{}; {}",
            offending_vma.path.display(),
            offending_vma.resident_pages,
            offending_vma.total_pages,
            aggregate,
        ),
        None => aggregate,
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

fn classify_leaf_join_error(source: io::Error, leaf: &Path) -> ColdInitError {
    ColdInitError::LeafCreateFailed {
        errno: source.raw_os_error().unwrap_or(libc::EIO),
        path: leaf.to_path_buf(),
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
    use tempfile::tempdir;

    use super::*;
    use crate::speed_of_light::ColdMode;

    #[derive(Default)]
    struct FakeColdOps {
        calls: Vec<&'static str>,
        verify_results: Vec<ColdVerifySummary>,
    }

    impl FakeColdOps {
        fn new(verify_results: &[ColdVerifySummary]) -> Self {
            Self {
                calls: Vec::new(),
                verify_results: verify_results.to_vec(),
            }
        }
    }

    impl ColdOps for FakeColdOps {
        fn sync_vmas(&mut self, _vmas: &[Vma]) -> Result<(), ColdStepError> {
            self.calls.push("msync");
            Ok(())
        }

        fn pageout_vmas(&mut self, _vmas: &[Vma]) -> Result<(), ColdStepError> {
            self.calls.push("madvise");
            Ok(())
        }

        fn reclaim(&mut self, _bytes: u64) -> Result<(), ColdStepError> {
            self.calls.push("reclaim");
            Ok(())
        }

        fn verify(&mut self, _vmas: &[Vma]) -> Result<ColdVerifySummary, ColdStepError> {
            self.calls.push("verify");
            Ok(self.verify_results.remove(0))
        }
    }

    fn verify_summary(
        residency_pages_after: u64,
        residency_total_pages: u64,
        offending_vma: Option<OffendingVmaResidency>,
    ) -> ColdVerifySummary {
        ColdVerifySummary {
            residency_pages_after,
            residency_total_pages,
            offending_vma,
        }
    }

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

    #[test]
    fn leaf_join_failure_maps_to_leaf_create_failed() {
        let leaf = PathBuf::from("/sys/fs/cgroup/bench-123");
        let error = classify_leaf_join_error(io::Error::from_raw_os_error(libc::EACCES), &leaf);
        assert_eq!(
            error,
            ColdInitError::LeafCreateFailed {
                errno: libc::EACCES,
                path: leaf,
            }
        );
    }

    #[test]
    fn startup_if_needed_is_noop_for_warm_only_plans() {
        let runtime = ColdRuntime::startup_if_needed(false, ColdMode::Strict).expect("startup");
        assert!(runtime.is_none());
    }

    #[test]
    fn bind_after_boot_is_the_only_phase_that_reports_no_pma_vmas() {
        let temp_dir = tempdir().expect("temp dir");
        let mut runtime = ColdRuntime {
            leaf: LeafCgroup::new(
                temp_dir.path().join("parent"),
                temp_dir.path().join("leaf"),
                std::process::id(),
            ),
            fsync: false,
            cold_mode: ColdMode::Strict,
            vmas: Vec::new(),
        };

        let error = runtime
            .bind_after_boot(temp_dir.path(), false)
            .expect_err("bind should fail without replay-pma VMAs");
        assert_eq!(error, ColdInitError::NoPmaVmas);
    }

    #[test]
    fn force_cold_respects_fsync_setting() {
        let vmas = vec![Vma {
            start: 0x1000,
            end: 0x2000,
            perms: "rw-s".to_string(),
            path: PathBuf::from("/tmp/replay-pma/slab-0.bin"),
        }];
        let options = ColdStepOptions {
            tolerance_pages: 0,
            max_attempts: 3,
        };

        let mut with_fsync = FakeColdOps::new(&[verify_summary(0, 1, None)]);
        let result = force_cold_with_ops(&mut with_fsync, &vmas, true, options, ColdMode::Strict)
            .expect("force cold with fsync");
        assert!(result.cold_verified);
        assert_eq!(
            with_fsync.calls,
            vec!["msync", "madvise", "reclaim", "verify"]
        );

        let mut without_fsync = FakeColdOps::new(&[verify_summary(0, 1, None)]);
        let result =
            force_cold_with_ops(&mut without_fsync, &vmas, false, options, ColdMode::Strict)
                .expect("force cold without fsync");
        assert!(result.cold_verified);
        assert_eq!(without_fsync.calls, vec!["madvise", "reclaim", "verify"]);
    }

    #[test]
    fn force_cold_retries_without_repeating_msync() {
        let vmas = vec![Vma {
            start: 0x1000,
            end: 0x3000,
            perms: "rw-s".to_string(),
            path: PathBuf::from("/tmp/replay-pma/slab-0.bin"),
        }];
        let options = ColdStepOptions {
            tolerance_pages: 0,
            max_attempts: 3,
        };
        let mut ops = FakeColdOps::new(&[
            verify_summary(
                3,
                8,
                Some(OffendingVmaResidency {
                    path: PathBuf::from("/tmp/replay-pma/slab-0.bin"),
                    resident_pages: 3,
                    total_pages: 8,
                }),
            ),
            verify_summary(0, 8, None),
        ]);

        let result = force_cold_with_ops(&mut ops, &vmas, true, options, ColdMode::Strict)
            .expect("force cold retries");

        assert!(result.cold_verified);
        assert_eq!(result.cold_attempts, 2);
        assert_eq!(
            ops.calls,
            vec!["msync", "madvise", "reclaim", "verify", "madvise", "reclaim", "verify"]
        );
    }

    #[test]
    fn force_cold_soft_mode_returns_unverified_after_retry_budget() {
        let vmas = vec![Vma {
            start: 0x1000,
            end: 0x3000,
            perms: "rw-s".to_string(),
            path: PathBuf::from("/tmp/replay-pma/slab-0.bin"),
        }];
        let options = ColdStepOptions {
            tolerance_pages: 0,
            max_attempts: 2,
        };
        let mut ops = FakeColdOps::new(&[
            verify_summary(
                2,
                8,
                Some(OffendingVmaResidency {
                    path: PathBuf::from("/tmp/replay-pma/slab-0.bin"),
                    resident_pages: 2,
                    total_pages: 8,
                }),
            ),
            verify_summary(
                1,
                8,
                Some(OffendingVmaResidency {
                    path: PathBuf::from("/tmp/replay-pma/slab-0.bin"),
                    resident_pages: 1,
                    total_pages: 8,
                }),
            ),
        ]);

        let result = force_cold_with_ops(&mut ops, &vmas, false, options, ColdMode::Soft)
            .expect("soft mode should continue");

        assert!(!result.cold_verified);
        assert_eq!(result.residency_pages_after, 1);
        assert_eq!(result.residency_total_pages, 8);
        assert_eq!(result.cold_attempts, 2);
    }

    #[test]
    fn strict_verify_failure_names_offending_vma_and_residency() {
        let vmas = vec![Vma {
            start: 0x1000,
            end: 0x3000,
            perms: "rw-s".to_string(),
            path: PathBuf::from("/tmp/replay-pma/slab-0.bin"),
        }];
        let options = ColdStepOptions {
            tolerance_pages: 0,
            max_attempts: 2,
        };
        let offending_vma = OffendingVmaResidency {
            path: PathBuf::from("/tmp/replay-pma/slab-0.bin"),
            resident_pages: 2,
            total_pages: 8,
        };
        let mut ops = FakeColdOps::new(&[
            verify_summary(2, 8, Some(offending_vma.clone())),
            verify_summary(2, 8, Some(offending_vma.clone())),
        ]);

        let error = force_cold_with_ops(&mut ops, &vmas, false, options, ColdMode::Strict)
            .expect_err("strict mode should fail");

        match error {
            ColdStepError::VerifyFailed {
                residency_pages_after,
                residency_total_pages,
                cold_attempts,
                offending_vma: Some(found_offending_vma),
                message,
                ..
            } => {
                assert_eq!(residency_pages_after, 2);
                assert_eq!(residency_total_pages, 8);
                assert_eq!(cold_attempts, 2);
                assert_eq!(found_offending_vma, offending_vma);
                assert!(message.contains("/tmp/replay-pma/slab-0.bin"));
                assert!(message.contains("resident_pages=2/8"));
            }
            other => panic!("expected verify failure, got {other:?}"),
        }
    }
}
