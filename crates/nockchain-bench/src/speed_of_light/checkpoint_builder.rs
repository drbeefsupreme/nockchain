//! Build checkpoints by replaying archived blocks into a kernel.

use std::path::PathBuf;

use bytes::Bytes;
use nockapp::kernel::boot::TraceOpts;
use nockapp::kernel::form::Kernel;
use nockapp::nockapp::save::SaveableCheckpoint;
use nockapp::nockapp::wire::WireRepr;
use nockapp::nockapp::NockApp;
use nockapp::noun::slab::NounSlab;
use nockchain_types::tx_engine::common::{BlockHeight, Hash};
use nockvm::noun::{NounAllocator, SIG};
use noun_serde::NounDecode;
use thiserror::Error;
use tracing::info;
use zkvm_jetpack::hot::produce_prover_hot_state;

use super::archive::{ArchiveFilter, ArchiveReader};
use super::checkpoint::{
    load_checkpoint, select_latest_checkpoint_path, CheckpointLoadError, CheckpointMetaError,
};
use super::poke::{extract_page_from_entry, make_heard_block_cause};
use super::start_height::{resolve_start_height, StartHeightError};

#[derive(Debug, Error)]
pub enum CheckpointBuildError {
    #[error("Archive error: {0}")]
    Archive(#[from] super::archive::ArchiveError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Kernel load error: {0}")]
    KernelLoad(String),

    #[error("Checkpoint load error: {0}")]
    CheckpointLoad(#[from] CheckpointLoadError),

    #[error("Checkpoint metadata error: {0}")]
    CheckpointMeta(#[from] CheckpointMetaError),

    #[error("Cue error: {0}")]
    Cue(String),

    #[error("Poke error: {0}")]
    Poke(String),

    #[error("Noun decode error: {0}")]
    NounDecode(#[from] noun_serde::NounDecodeError),

    #[error("Start height error: {0}")]
    StartHeight(#[from] StartHeightError),

    #[error("Checkpoint chain height unavailable; pass --start-height explicitly")]
    CheckpointHeightUnavailable,

    #[error("Invalid height range: start {start} > target {target}")]
    InvalidHeightRange { start: u64, target: u64 },

    #[error("NockApp error: {0}")]
    NockApp(#[from] nockapp::nockapp::NockAppError),
}

#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub archive_path: String,
    pub kernel_path: String,
    pub checkpoint_path: Option<String>,
    pub start_height: Option<u64>,
    pub target_height: u64,
    pub output_path: PathBuf,
    pub work_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct CheckpointResult {
    pub start_height: u64,
    pub target_height: u64,
    pub blocks_poked: u64,
    pub output_path: PathBuf,
}

pub struct CheckpointBuilder {
    config: CheckpointConfig,
    nockapp: Option<NockApp>,
}

impl CheckpointBuilder {
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            nockapp: None,
        }
    }

    pub async fn initialize(&mut self) -> Result<(), CheckpointBuildError> {
        info!(kernel = %self.config.kernel_path, "Initializing kernel for checkpoint builder");

        let kernel_bytes = std::fs::read(&self.config.kernel_path)?;
        info!(kernel_size = kernel_bytes.len(), "Loaded kernel jam");

        let hot_state = produce_prover_hot_state();
        info!(jets = hot_state.len(), "Got hot state entries");

        let checkpoint = if let Some(path) = &self.config.checkpoint_path {
            let loaded = load_checkpoint(path)?;
            Some(SaveableCheckpoint {
                ker_hash: loaded.ker_hash,
                event_num: loaded.event_num,
                state: loaded.state,
                cold: loaded.cold,
            })
        } else {
            None
        };

        let work_dir = self.config.work_dir.clone();

        let nockapp = NockApp::new(
            move |_existing_checkpoint| {
                let checkpoint = checkpoint;
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
            &work_dir,
            None,
            true,
        )
        .await
        .map_err(|e| CheckpointBuildError::KernelLoad(e.to_string()))?;

        self.nockapp = Some(nockapp);
        Ok(())
    }

    async fn peek_chain_height(nockapp: &mut NockApp) -> Result<Option<u64>, CheckpointBuildError> {
        let mut path_slab = NounSlab::new();
        let tag = nockapp::utils::make_tas(&mut path_slab, "heaviest-chain").as_noun();
        let path_noun = nockvm::noun::T(&mut path_slab, &[tag, SIG]);
        path_slab.set_root(path_noun);

        let result = nockapp.peek(path_slab).await?;
        let result_noun = unsafe { result.root() };
        let space = result.noun_space();

        let opt: Option<Option<(BlockHeight, Hash)>> = NounDecode::from_noun(&result_noun, &space)?;
        Ok(opt.flatten().map(|(height, _)| height.0 .0))
    }

    pub async fn run(&mut self) -> Result<CheckpointResult, CheckpointBuildError> {
        let archive_bytes = std::fs::read(&self.config.archive_path)?;
        let reader = ArchiveReader::from_bytes(archive_bytes)?;

        self.initialize().await?;

        let nockapp = self.nockapp.as_mut().ok_or(CheckpointBuildError::KernelLoad(
            "NockApp not initialized".to_string(),
        ))?;

        let checkpoint_height = if self.config.checkpoint_path.is_some() {
            let height = Self::peek_chain_height(nockapp).await?;
            height.ok_or(CheckpointBuildError::CheckpointHeightUnavailable).map(Some)?
        } else {
            None
        };

        let start_height = resolve_start_height(self.config.start_height, checkpoint_height)?;
        if start_height > self.config.target_height {
            return Err(CheckpointBuildError::InvalidHeightRange {
                start: start_height,
                target: self.config.target_height,
            });
        }

        info!(
            start_height,
            target_height = self.config.target_height,
            "Replaying blocks for checkpoint"
        );

        let filter = ArchiveFilter {
            proof_version: None,
            start_height: Some(start_height),
            end_height: Some(self.config.target_height),
        };

        let mut blocks_poked = 0u64;
        let wire = WireRepr {
            source: "bench",
            version: 1,
            tags: vec![],
        };

        for (entry, jam_bytes) in reader.iter_filtered(filter) {
            let mut entry_slab: NounSlab = NounSlab::new();
            let entry_noun = entry_slab
                .cue_into(Bytes::from(jam_bytes.to_vec()))
                .map_err(|e| CheckpointBuildError::Cue(format!("cue failed: {e:?}")))?;

            let page = extract_page_from_entry(entry_noun, &entry_slab)
                .map_err(CheckpointBuildError::Cue)?;

            let mut poke_slab: NounSlab = NounSlab::new();
            let space = entry_slab.noun_space();
            let page_copy = poke_slab.copy_into(page, &space);
            let cause = make_heard_block_cause(page_copy, &mut poke_slab);
            poke_slab.set_root(cause);

            nockapp
                .poke(wire.clone(), poke_slab)
                .await
                .map_err(|e| CheckpointBuildError::Poke(format!("poke failed: {e:?}")))?;

            blocks_poked += 1;

            if blocks_poked % 100 == 0 {
                info!(blocks_poked, height = entry.height, "Checkpoint replay progress");
            }
        }

        info!(blocks_poked, "Replay complete; saving checkpoint");
        nockapp.save_blocking().await?;

        let latest_checkpoint = select_latest_checkpoint_path(&self.config.work_dir)?;
        if let Some(parent) = self.config.output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(&latest_checkpoint, &self.config.output_path)?;

        Ok(CheckpointResult {
            start_height,
            target_height: self.config.target_height,
            blocks_poked,
            output_path: self.config.output_path.clone(),
        })
    }
}
