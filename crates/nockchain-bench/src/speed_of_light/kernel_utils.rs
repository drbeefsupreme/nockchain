//! Shared kernel initialization and peek helpers for speed-of-light tooling.

use std::path::{Path, PathBuf};

use nockapp::kernel::form::Kernel;
use nockapp::nockapp::save::SaveableCheckpoint;
use nockapp::nockapp::wire::WireRepr;
use nockapp::nockapp::NockApp;
use nockapp::noun::slab::NounSlab;
use nockchain_types::tx_engine::common::{BlockHeight, Hash};
use nockvm::noun::SIG;
use nockvm::trace::TraceInfo;
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

/// Minimal valid libp2p peer id used for synthetic SOL replay wires.
const SOL_REPLAY_PEER_ID: &str = "11";

/// Canonical wire for replaying archived `%heard-block` facts.
///
/// This mirrors the normal network ingress path:
/// `[%poke %libp2p 1 %gossip %peer-id <peer-id> ~]`.
pub fn sol_replay_wire() -> WireRepr {
    WireRepr::new(
        "libp2p",
        1,
        vec!["gossip".into(), "peer-id".into(), SOL_REPLAY_PEER_ID.into()],
    )
}

/// Initialize a NockApp with a kernel and optional checkpoint.
pub async fn init_nockapp(
    kernel_path: &Path,
    checkpoint: Option<SaveableCheckpoint>,
    work_dir: &PathBuf,
    prefer_existing_checkpoint: bool,
    trace_info: Option<TraceInfo>,
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
                Kernel::load_with_hot_state_medium_trace_info(
                    &kernel_bytes,
                    checkpoint,
                    &hot_state,
                    vec![],
                    trace_info,
                )
                .await
            }
        },
        work_dir,
        None,
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

    let opt: Option<Option<(BlockHeight, Hash)>> = NounDecode::from_noun(&result_noun)?;
    Ok(opt.flatten())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sol_replay_wire_matches_libp2p_gossip_shape() {
        let wire = sol_replay_wire();
        assert_eq!(wire.source, "libp2p");
        assert_eq!(wire.version, 1);
        assert_eq!(wire.tags_as_csv(), "libp2p,1,gossip,peer-id,11");
    }
}
