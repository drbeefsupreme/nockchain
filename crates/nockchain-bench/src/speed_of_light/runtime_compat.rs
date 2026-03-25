//! Bench-local compile-time compatibility helpers for PMA runtime support.

use nockapp::noun::slab::NounSlab;
use nockvm::noun::Noun;

#[cfg(feature = "pma-runtime-compat")]
use std::path::{Path, PathBuf};

#[cfg(feature = "pma-runtime-compat")]
use nockapp::kernel::boot::TraceOpts;
#[cfg(feature = "pma-runtime-compat")]
use nockapp::kernel::form::Kernel;
#[cfg(feature = "pma-runtime-compat")]
use nockapp::nockapp::save::SaveableCheckpoint;
#[cfg(feature = "pma-runtime-compat")]
use nockapp::nockapp::NockApp;
#[cfg(feature = "pma-runtime-compat")]
use tracing::info;
#[cfg(feature = "pma-runtime-compat")]
use zkvm_jetpack::hot::produce_prover_hot_state;

#[cfg(feature = "pma-runtime-compat")]
use super::kernel_utils::KernelInitError;

#[cfg(feature = "pma-runtime-compat")]
pub async fn init_replay_nockapp(
    kernel_path: &Path,
    checkpoint: Option<SaveableCheckpoint>,
    _work_dir: &PathBuf,
) -> Result<NockApp, KernelInitError> {
    let kernel_bytes = std::fs::read(kernel_path)?;
    info!(kernel_size = kernel_bytes.len(), "Loaded kernel jam");

    let hot_state = produce_prover_hot_state();
    info!(jets = hot_state.len(), "Got hot state entries");

    let kernel = Kernel::load_with_hot_state_medium(
        &kernel_bytes,
        checkpoint,
        &hot_state,
        vec![],
        TraceOpts::default(),
        None,
    )
    .await
    .map_err(nockapp::nockapp::NockAppError::from)
    .map_err(KernelInitError::from)?;

    NockApp::new(move |_metrics| async move {
        Ok::<Kernel<SaveableCheckpoint>, nockapp::CrownError>(kernel)
    })
        .await
        .map_err(KernelInitError::from)
}

#[cfg(not(feature = "pma-runtime-compat"))]
pub fn copy_from_source_slab<J, K>(dst: &mut NounSlab<J>, noun: Noun, _src: &NounSlab<K>) -> Noun {
    dst.copy_into(noun)
}

#[cfg(feature = "pma-runtime-compat")]
pub fn copy_from_source_slab<J, K>(dst: &mut NounSlab<J>, noun: Noun, src: &NounSlab<K>) -> Noun {
    use nockvm::noun::NounAllocator;

    let space = src.noun_space();
    dst.copy_into(noun, &space)
}
