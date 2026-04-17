#[cfg(not(target_os = "linux"))]
use std::path::Path;

#[cfg(target_os = "linux")]
mod cgroup;
mod measure;
#[cfg(target_os = "linux")]
mod vma;

#[cfg(target_os = "linux")]
pub use cgroup::{
    own_cgroup_path, parse_subtree_control_tokens, ColdForceResult, ColdInitError, ColdRuntime,
    ColdStepError, ColdStepOptions, OffendingVmaResidency,
};
pub use measure::{measure_peek, measure_sync, PeekMeasurement, StepMeasurement};
#[cfg(target_os = "linux")]
pub use vma::{
    page_size, parse_proc_maps, read_pma_vmas, reduce_mincore_bitmap, resident_pages, Vma,
};

#[cfg(not(target_os = "linux"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColdStepOptions {
    pub tolerance_pages: u64,
    pub max_attempts: u32,
}

#[cfg(not(target_os = "linux"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffendingVmaResidency {
    pub path: std::path::PathBuf,
    pub resident_pages: u64,
    pub total_pages: u64,
}

#[cfg(not(target_os = "linux"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdForceResult {
    pub cold_verified: bool,
    pub residency_pages_after: u64,
    pub residency_total_pages: u64,
    pub cold_attempts: u32,
    pub degraded_reason: Option<String>,
}

#[cfg(not(target_os = "linux"))]
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

#[cfg(not(target_os = "linux"))]
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
    LeafCreateFailed {
        errno: i32,
        path: std::path::PathBuf,
    },

    #[error("failed to probe memory.reclaim: errno {errno}")]
    ReclaimProbeFailed { errno: i32 },

    #[error("no PMA VMAs discovered under replay-pma")]
    NoPmaVmas,
}

#[cfg(not(target_os = "linux"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColdRuntime;

#[cfg(not(target_os = "linux"))]
impl ColdRuntime {
    pub fn startup_if_needed(
        has_cold_steps: bool,
        _cold_mode: crate::speed_of_light::ColdMode,
    ) -> Result<Option<Self>, ColdInitError> {
        Ok(has_cold_steps.then_some(Self))
    }

    pub fn bind_after_boot(&mut self, _work_dir: &Path, _fsync: bool) -> Result<(), ColdInitError> {
        Ok(())
    }

    pub fn force_cold(
        &mut self,
        options: ColdStepOptions,
    ) -> Result<ColdForceResult, ColdStepError> {
        Ok(ColdForceResult {
            cold_verified: false,
            residency_pages_after: 0,
            residency_total_pages: 0,
            cold_attempts: options.max_attempts,
            degraded_reason: Some("macos_unsupported".to_string()),
        })
    }
}
