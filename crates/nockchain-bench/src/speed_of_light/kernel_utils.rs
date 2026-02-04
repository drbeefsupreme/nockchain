//! Shared kernel initialization and peek helpers for speed-of-light tooling.

use std::path::{Path, PathBuf};

use nockapp::kernel::boot::TraceOpts;
use nockapp::kernel::form::Kernel;
use nockapp::nockapp::save::SaveableCheckpoint;
use nockapp::nockapp::NockApp;
use nockapp::noun::slab::NounSlab;
use nockchain_types::tx_engine::common::{BlockHeight, Hash};
use nockvm::noun::{NounAllocator, SIG};
use noun_serde::NounDecode;
use thiserror::Error;
use tracing::info;
use zkvm_jetpack::hot::produce_prover_hot_state;

#[derive(Debug, Error)]
pub enum KernelInitError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("NockApp error: {0}")]
    NockApp(#[from] nockapp::nockapp::NockAppError),
}

#[derive(Debug, Error)]
pub enum PeekChainError {
    #[error("NockApp error: {0}")]
    NockApp(#[from] nockapp::nockapp::NockAppError),

    #[error("Noun decode error: {0}")]
    NounDecode(#[from] noun_serde::NounDecodeError),
}

/// Initialize a NockApp with a kernel and optional checkpoint.
pub async fn init_nockapp(
    kernel_path: &Path,
    checkpoint: Option<SaveableCheckpoint>,
    work_dir: &PathBuf,
    enable_checkpointing: bool,
    prefer_existing_checkpoint: bool,
) -> Result<NockApp, KernelInitError> {
    let kernel_bytes = std::fs::read(kernel_path)?;
    info!(kernel_size = kernel_bytes.len(), "Loaded kernel jam");

    let hot_state = produce_prover_hot_state();
    info!(jets = hot_state.len(), "Got hot state entries");

    let nockapp = NockApp::new(
        move |existing_checkpoint| {
            let checkpoint = if prefer_existing_checkpoint {
                existing_checkpoint.or(checkpoint)
            } else {
                checkpoint
            };
            async move {
                Kernel::load_with_hot_state_medium(
                    &kernel_bytes,
                    checkpoint,
                    &hot_state,
                    vec![],
                    TraceOpts::default(),
                    None,
                )
                .await
            }
        },
        work_dir,
        None,
        enable_checkpointing,
    )
    .await?;

    Ok(nockapp)
}

/// Peek the heaviest chain (height, hash) from a running NockApp.
pub async fn peek_heaviest_chain(
    nockapp: &mut NockApp,
) -> Result<Option<(BlockHeight, Hash)>, PeekChainError> {
    let mut path_slab = NounSlab::new();
    let tag = nockapp::utils::make_tas(&mut path_slab, "heaviest-chain").as_noun();
    let path_noun = nockvm::noun::T(&mut path_slab, &[tag, SIG]);
    path_slab.set_root(path_noun);

    let result = nockapp.peek(path_slab).await?;
    let result_noun = unsafe { result.root() };
    let space = result.noun_space();

    let opt: Option<Option<(BlockHeight, Hash)>> = NounDecode::from_noun(&result_noun, &space)?;
    Ok(opt.flatten())
}
