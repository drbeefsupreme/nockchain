//! Shared kernel initialization and peek helpers for speed-of-light tooling.

use std::path::{Path, PathBuf};

use nockapp::kernel::boot::{self, TraceOpts};
use nockapp::kernel::form::Kernel;
use nockapp::nockapp::save::SaveableCheckpoint;
use nockapp::nockapp::wire::WireRepr;
use nockapp::nockapp::NockApp;
use nockapp::noun::slab::{NockJammer, NounSlab};
use nockapp::noun::AtomExt;
use nockapp::utils::make_tas;
use nockapp::wire::{SystemWire, Wire};
use nockchain::setup::{self, SetupCommand};
#[cfg(not(feature = "pma-runtime-compat"))]
use nockchain_types::fakenet_blockchain_constants;
use nockchain_types::tx_engine::common::{BlockHeight, Hash, Page};
use nockvm::noun::{Atom, D, NO, SIG, T, YES};
use thiserror::Error;
use tracing::info;
use zkvm_jetpack::hot::produce_prover_hot_state;

use super::checkpoint::{load_saveable_checkpoint, CheckpointLoadError};
use super::noun_compat;
use super::pma_replay;

#[derive(Debug, Error)]
pub enum KernelInitError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("NockApp error: {0}")]
    NockApp(#[from] nockapp::nockapp::NockAppError),

    #[error("Kernel boot error: {0}")]
    Boot(String),
}

#[derive(Debug, Error)]
pub enum CheckpointBackedInitError {
    #[error("failed to load checkpoint: {0}")]
    CheckpointLoad(#[from] CheckpointLoadError),

    #[error("failed to initialize checkpoint-backed kernel: {0}")]
    KernelInit(#[from] KernelInitError),
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
const FULL_CHECKPOINT_BOOT_NAME: &str = ".";
const DEFAULT_FAKENET_POW_LEN: u64 = 2;
const DEFAULT_FAKENET_LOG_DIFFICULTY: u64 = 1;

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
    fsync: bool,
) -> Result<NockApp, KernelInitError> {
    if prefer_existing_checkpoint {
        return Err(KernelInitError::Boot(
            "prefer_existing_checkpoint replay is not supported by PMA replay".to_string(),
        ));
    }

    pma_replay::init_replay_nockapp(kernel_path, checkpoint, work_dir, fsync).await
}

/// Load a checkpoint from disk and boot a checkpoint-backed NockApp.
pub async fn init_checkpoint_backed_nockapp(
    checkpoint_path: &Path,
    kernel_path: &Path,
    work_dir: &PathBuf,
    fsync: bool,
) -> Result<NockApp, CheckpointBackedInitError> {
    let checkpoint = load_saveable_checkpoint(checkpoint_path)?;
    init_nockapp(kernel_path, Some(checkpoint), work_dir, false, fsync)
        .await
        .map_err(Into::into)
}

/// Initialize a NockApp through the runtime boot path and force the startup
/// pokes required to materialize a runtime-shaped checkpoint.
pub async fn init_full_checkpoint_nockapp(
    kernel_path: &Path,
    work_dir: &PathBuf,
) -> Result<NockApp, KernelInitError> {
    #[cfg(feature = "pma-runtime-compat")]
    {
        let _ = (kernel_path, work_dir);
        return Err(KernelInitError::Boot(
            "full checkpoint boot is not supported under pma-runtime-compat in Phase 1; boot::setup() integration is deferred to Phase 2B".to_string(),
        ));
    }

    #[cfg(not(feature = "pma-runtime-compat"))]
    {
        let kernel_bytes = std::fs::read(kernel_path)?;
        info!(
            kernel_size = kernel_bytes.len(),
            "Loaded full checkpoint kernel jam"
        );

        let hot_state = produce_prover_hot_state();
        info!(jets = hot_state.len(), "Got hot state entries");

        let mut boot_cli = boot::default_boot_cli(false);
        boot_cli.stack_size = boot::NockStackSize::Medium;
        boot_cli.save_interval = Some(0);

        let mut nockapp = boot::setup::<NockJammer>(
            &kernel_bytes,
            boot_cli,
            &hot_state,
            FULL_CHECKPOINT_BOOT_NAME,
            Some(work_dir.clone()),
        )
        .await
        .map_err(|err| KernelInitError::Boot(err.to_string()))?;

        bootstrap_full_checkpoint_runtime_state(&mut nockapp).await?;
        Ok(nockapp)
    }
}

#[cfg(not(feature = "pma-runtime-compat"))]
async fn bootstrap_full_checkpoint_runtime_state(
    nockapp: &mut NockApp,
) -> Result<(), KernelInitError> {
    let is_kernel_mainnet = peek_kernel_mainnet(nockapp).await?;
    let genesis_seal_set = peek_genesis_seal_initialized(nockapp).await?;

    if matches!(is_kernel_mainnet, Some(true) | None) {
        if is_kernel_mainnet.is_none() {
            info!("kernel did not expose `mainnet`; defaulting full bootstrap to mainnet");
        }
        if !genesis_seal_set {
            apply_setup_command(
                nockapp,
                SetupCommand::PokeSetGenesisSeal(setup::REALNET_GENESIS_MESSAGE.to_string()),
            )
            .await?;
        }
    } else {
        apply_setup_command(
            nockapp,
            SetupCommand::PokeFakenetConstants(Box::new(fakenet_blockchain_constants(
                DEFAULT_FAKENET_POW_LEN, DEFAULT_FAKENET_LOG_DIFFICULTY,
            ))),
        )
        .await?;
        if !genesis_seal_set {
            apply_setup_command(
                nockapp,
                SetupCommand::PokeSetGenesisSeal(setup::FAKENET_GENESIS_MESSAGE.to_string()),
            )
            .await?;
        }

        nockapp
            .poke(SystemWire.to_wire(), setup::heard_fake_genesis_block(None)?)
            .await?;
    }

    apply_setup_command(nockapp, SetupCommand::PokeSetBtcData).await?;
    nockapp
        .poke(full_checkpoint_mining_wire(), enable_mining_poke(false))
        .await?;
    nockapp
        .poke(SystemWire.to_wire(), born_poke())
        .await
        .map(|_| ())
        .map_err(KernelInitError::from)
}

async fn apply_setup_command(
    nockapp: &mut NockApp,
    command: SetupCommand,
) -> Result<(), KernelInitError> {
    setup::poke(nockapp, command)
        .await
        .map_err(|err| KernelInitError::Boot(err.to_string()))
}

async fn peek_kernel_mainnet(nockapp: &mut NockApp) -> Result<Option<bool>, KernelInitError> {
    let mut peek_slab = NounSlab::new();
    let tag = make_tas(&mut peek_slab, "mainnet").as_noun();
    let peek_noun = T(&mut peek_slab, &[tag, D(0)]);
    peek_slab.set_root(peek_noun);
    let Some(peek_res) = nockapp.peek_handle(peek_slab).await? else {
        return Ok(None);
    };
    let mainnet_flag = unsafe { peek_res.root() };
    if !mainnet_flag.is_atom() {
        return Err(KernelInitError::Boot(
            "kernel returned a non-atom `mainnet` bootstrap peek".to_string(),
        ));
    }

    Ok(Some(unsafe { mainnet_flag.raw_equals(&YES) }))
}

async fn peek_genesis_seal_initialized(nockapp: &mut NockApp) -> Result<bool, KernelInitError> {
    let mut peek_slab = NounSlab::new();
    let tag = make_tas(&mut peek_slab, "genesis-seal-set").as_noun();
    let peek_noun = T(&mut peek_slab, &[tag, D(0)]);
    peek_slab.set_root(peek_noun);
    let Some(peek_res) = nockapp.peek_handle(peek_slab).await? else {
        return Err(KernelInitError::Boot(
            "kernel did not expose a `genesis-seal-set` bootstrap peek".to_string(),
        ));
    };
    let genesis_seal = unsafe { peek_res.root() };
    if !genesis_seal.is_atom() {
        return Err(KernelInitError::Boot(
            "kernel returned a non-atom `genesis-seal-set` bootstrap peek".to_string(),
        ));
    }

    Ok(unsafe { genesis_seal.raw_equals(&YES) })
}

fn full_checkpoint_mining_wire() -> WireRepr {
    WireRepr::new("miner", 1, vec!["enable".into()])
}

fn enable_mining_poke(enable: bool) -> NounSlab {
    let mut enable_mining_slab = NounSlab::new();
    let command = make_tas(&mut enable_mining_slab, "command").as_noun();
    let enable_mining = Atom::from_value(&mut enable_mining_slab, "enable-mining")
        .expect("failed to build enable-mining atom");
    let enable_mining_poke = T(
        &mut enable_mining_slab,
        &[command, enable_mining.as_noun(), if enable { YES } else { NO }],
    );
    enable_mining_slab.set_root(enable_mining_poke);
    enable_mining_slab
}

fn born_poke() -> NounSlab {
    let mut born_slab = NounSlab::new();
    let command = make_tas(&mut born_slab, "command").as_noun();
    let born_tag = make_tas(&mut born_slab, "born").as_noun();
    let born = T(&mut born_slab, &[command, born_tag, D(0)]);
    born_slab.set_root(born);
    born_slab
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
    let result_space = noun_compat::space_for_slab(&result);
    decode_heaviest_chain_result(*result_noun, &result_space)
}

pub async fn peek_heaviest_chain_or_block(
    nockapp: &mut NockApp,
) -> Result<Option<(BlockHeight, Hash)>, PeekChainError> {
    if let Some(tip) = peek_heaviest_chain(nockapp).await? {
        return Ok(Some(tip));
    }

    peek_heaviest_block(nockapp).await
}

async fn peek_heaviest_block(
    nockapp: &mut NockApp,
) -> Result<Option<(BlockHeight, Hash)>, PeekChainError> {
    let mut path_slab = NounSlab::new();
    let tag = nockapp::utils::make_tas(&mut path_slab, "heaviest-block").as_noun();
    let path_noun = nockvm::noun::T(&mut path_slab, &[tag, SIG]);
    path_slab.set_root(path_noun);

    let result = nockapp.peek(path_slab).await?;
    let result_noun = unsafe { result.root() };
    let result_space = noun_compat::space_for_slab(&result);
    decode_heaviest_block_result(*result_noun, &result_space)
}

fn decode_heaviest_chain_result(
    result_noun: nockvm::noun::Noun,
    space: &noun_compat::NounSpace,
) -> Result<Option<(BlockHeight, Hash)>, PeekChainError> {
    let opt: Option<Option<(BlockHeight, Hash)>> =
        noun_compat::decode_with_space(&result_noun, space)?;
    Ok(opt.flatten())
}

fn decode_heaviest_block_result(
    result_noun: nockvm::noun::Noun,
    space: &noun_compat::NounSpace,
) -> Result<Option<(BlockHeight, Hash)>, PeekChainError> {
    let opt: Option<Option<Page>> = noun_compat::decode_with_space(&result_noun, space)?;
    Ok(opt.flatten().map(|page| {
        (
            BlockHeight(nockchain_math::belt::Belt(page.height)),
            page.digest,
        )
    }))
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "pma-runtime-compat")]
    use std::path::{Path, PathBuf};

    use nockchain_math::belt::Belt;
    use noun_serde::NounEncode;

    use super::*;

    #[test]
    fn test_sol_replay_wire_matches_libp2p_gossip_shape() {
        let wire = sol_replay_wire();
        assert_eq!(wire.source, "libp2p");
        assert_eq!(wire.version, 1);
        assert_eq!(wire.tags_as_csv(), "libp2p,1,gossip,peer-id,11");
    }

    #[test]
    fn test_full_checkpoint_mining_wire_matches_runtime_shape() {
        let wire = full_checkpoint_mining_wire();
        assert_eq!(wire.source, "miner");
        assert_eq!(wire.version, 1);
        assert_eq!(wire.tags_as_csv(), "miner,1,enable");
    }

    #[test]
    fn test_decode_heaviest_chain_result_reads_optional_height_and_hash() {
        let mut slab: NounSlab = NounSlab::new();
        let expected_hash = Hash([Belt(10), Belt(11), Belt(12), Belt(13), Belt(14)]);
        let response = Some(Some((BlockHeight(Belt(42)), expected_hash.clone())));
        let noun = response.to_noun(&mut slab);
        let space = noun_compat::space_for_slab(&slab);

        let decoded =
            decode_heaviest_chain_result(noun, &space).expect("heaviest-chain decode should work");

        let (height, hash) = decoded.expect("heaviest-chain peek should produce data");
        assert_eq!(height.0 .0, 42);
        assert_eq!(hash.to_base58(), expected_hash.to_base58());
    }

    #[test]
    fn test_decode_heaviest_chain_result_returns_none_when_chain_is_missing() {
        let mut slab: NounSlab = NounSlab::new();
        let response: Option<Option<(BlockHeight, Hash)>> = Some(None);
        let noun = response.to_noun(&mut slab);
        let space = noun_compat::space_for_slab(&slab);

        let decoded =
            decode_heaviest_chain_result(noun, &space).expect("heaviest-chain decode should work");

        assert!(decoded.is_none());
    }

    #[test]
    fn test_decode_heaviest_block_result_reads_page_height_and_hash() {
        let mut slab: NounSlab = NounSlab::new();
        let expected_hash = Hash([Belt(10), Belt(11), Belt(12), Belt(13), Belt(14)]);
        let page = Page {
            digest: expected_hash.clone(),
            pow: None,
            parent: Hash([Belt(20), Belt(21), Belt(22), Belt(23), Belt(24)]),
            tx_ids: Vec::new(),
            coinbase: nockchain_types::tx_engine::common::CoinbaseSplit::V1,
            timestamp: 0,
            epoch_counter: 0,
            target: nockchain_types::tx_engine::common::BigNum::from_u64(1),
            accumulated_work: nockchain_types::tx_engine::common::BigNum::from_u64(1),
            height: 42,
            msg: Vec::new(),
        };
        let response = Some(Some(page));
        let noun = response.to_noun(&mut slab);
        let space = noun_compat::space_for_slab(&slab);

        let decoded =
            decode_heaviest_block_result(noun, &space).expect("heaviest-block decode should work");

        let (height, hash) = decoded.expect("heaviest-block peek should produce data");
        assert_eq!(height.0 .0, 42);
        assert_eq!(hash.to_base58(), expected_hash.to_base58());
    }

    #[cfg(feature = "pma-runtime-compat")]
    #[tokio::test]
    async fn test_pma_init_nockapp_rejects_prefer_existing_checkpoint() {
        let err = match init_nockapp(
            Path::new("unused-kernel"),
            None,
            &PathBuf::from("."),
            true,
            true,
        )
        .await
        {
            Ok(_) => panic!("PMA replay wrapper should reject prefer_existing_checkpoint"),
            Err(err) => err,
        };

        match err {
            KernelInitError::Boot(message) => assert_eq!(
                message,
                "prefer_existing_checkpoint replay is not supported under pma-runtime-compat in Phase 1; existing-checkpoint PMA boot is deferred"
            ),
            other => panic!("expected boot error, got {other:?}"),
        }
    }
}
