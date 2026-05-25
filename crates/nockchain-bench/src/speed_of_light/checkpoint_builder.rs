//! Build checkpoints by replaying archived blocks into a kernel.

use std::path::PathBuf;

use nockapp::nockapp::save::SaveableCheckpoint;
use nockapp::nockapp::NockApp;
use thiserror::Error;
use tracing::info;

use super::archive::{ArchiveFilter, SolArchiveReader};
use super::checkpoint::{
    load_checkpoint, select_latest_checkpoint_path, CheckpointLoadError, CheckpointMetaError,
};
use super::kernel_utils::{
    init_full_checkpoint_nockapp, init_nockapp, peek_heaviest_chain, sol_replay_wire,
    KernelInitError, PeekChainError,
};
use super::poke::build_poke_slab_from_jam;
use super::start_height::{resolve_start_height, StartHeightError};
use super::types::SolHeight;

#[derive(Debug, Error)]
pub enum CheckpointBuildError {
    #[error("Archive error: {0}")]
    Archive(#[from] super::archive::ArchiveError),

    #[error("Unsupported checkpoint path: {0}")]
    Unsupported(String),

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

    #[error("Kernel init error: {0}")]
    KernelInit(#[from] KernelInitError),

    #[error("Chain height peek error: {0}")]
    ChainPeek(#[from] PeekChainError),
}

#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub archive_path: String,
    pub kernel_path: String,
    pub checkpoint_path: Option<String>,
    pub build_mode: CheckpointBuildMode,
    pub start_height: Option<SolHeight>,
    pub target_height: SolHeight,
    pub output_path: PathBuf,
    pub work_dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointBuildMode {
    Derived,
    Full,
}

#[derive(Debug, Clone)]
pub struct CheckpointResult {
    pub start_height: SolHeight,
    pub target_height: SolHeight,
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
        ensure_checkpoint_builder_supported(self.config.build_mode)?;
        info!(kernel = %self.config.kernel_path, "Initializing kernel for checkpoint builder");

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

        let nockapp = match self.config.build_mode {
            CheckpointBuildMode::Derived => {
                init_nockapp(
                    std::path::Path::new(&self.config.kernel_path),
                    checkpoint,
                    &work_dir,
                    false,
                    true,
                )
                .await?
            }
            CheckpointBuildMode::Full => {
                if checkpoint.is_some() {
                    return Err(CheckpointBuildError::KernelLoad(
                        "full checkpoint mode does not support --checkpoint input".to_string(),
                    ));
                }
                init_full_checkpoint_nockapp(
                    std::path::Path::new(&self.config.kernel_path),
                    &work_dir,
                )
                .await?
            }
        };

        self.nockapp = Some(nockapp);
        Ok(())
    }

    pub async fn run(&mut self) -> Result<CheckpointResult, CheckpointBuildError> {
        ensure_checkpoint_builder_supported(self.config.build_mode)?;

        let archive_bytes = std::fs::read(&self.config.archive_path)?;
        let reader = SolArchiveReader::from_bytes(archive_bytes)?;

        self.initialize().await?;

        let nockapp = self
            .nockapp
            .as_mut()
            .ok_or(CheckpointBuildError::KernelLoad(
                "NockApp not initialized".to_string(),
            ))?;

        let checkpoint_height = if self.config.checkpoint_path.is_some() {
            let height = peek_heaviest_chain(nockapp).await?;
            height
                .map(|(height, _)| SolHeight(height.0 .0))
                .ok_or(CheckpointBuildError::CheckpointHeightUnavailable)
                .map(Some)?
        } else {
            None
        };

        let start_height = resolve_start_height(self.config.start_height, checkpoint_height)?;
        if start_height > self.config.target_height {
            return Err(CheckpointBuildError::InvalidHeightRange {
                start: start_height.as_u64(),
                target: self.config.target_height.as_u64(),
            });
        }

        info!(
            start_height = start_height.as_u64(),
            target_height = self.config.target_height.as_u64(),
            "Replaying blocks for checkpoint"
        );

        let filter = ArchiveFilter {
            proof_version: None,
            start_height: Some(start_height),
            end_height: Some(self.config.target_height),
        };

        let mut blocks_poked = 0u64;
        let wire = sol_replay_wire();

        for (entry, jam_bytes) in reader.iter_filtered(filter) {
            let poke_slab =
                build_poke_slab_from_jam(jam_bytes).map_err(CheckpointBuildError::Cue)?;

            nockapp
                .poke(wire.clone(), poke_slab)
                .await
                .map_err(|e| CheckpointBuildError::Poke(format!("poke failed: {e:?}")))?;

            blocks_poked += 1;

            if blocks_poked % 100 == 0 {
                info!(
                    blocks_poked,
                    height = entry.height.as_u64(),
                    "Checkpoint replay progress"
                );
            }
        }

        #[cfg(feature = "pma-runtime-compat")]
        {
            let _ = (nockapp, start_height, blocks_poked);
            unreachable!("checkpoint builder is guarded above under pma-runtime-compat");
        }

        #[cfg(not(feature = "pma-runtime-compat"))]
        {
            info!(blocks_poked, "Replay complete; saving checkpoint");
            nockapp.save_blocking().await?;

            let latest_checkpoint = select_latest_checkpoint_path(self.snapshot_dir())?;
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

    fn snapshot_dir(&self) -> PathBuf {
        snapshot_dir_for_mode(&self.config.work_dir, self.config.build_mode)
    }
}

fn snapshot_dir_for_mode(work_dir: &std::path::Path, build_mode: CheckpointBuildMode) -> PathBuf {
    match build_mode {
        CheckpointBuildMode::Derived => work_dir.to_path_buf(),
        CheckpointBuildMode::Full => work_dir.join("checkpoints"),
    }
}

fn ensure_checkpoint_builder_supported(
    _build_mode: CheckpointBuildMode,
) -> Result<(), CheckpointBuildError> {
    #[cfg(feature = "pma-runtime-compat")]
    {
        return Err(CheckpointBuildError::Unsupported(
            "checkpoint builder is not supported under pma-runtime-compat in Phase 1; legacy .chkjam materialization is deferred to Phase 2".to_string(),
        ));
    }

    #[cfg(not(feature = "pma-runtime-compat"))]
    {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_builder_uses_plain_work_dir_for_derived_snapshots() {
        assert_eq!(
            snapshot_dir_for_mode(
                std::path::Path::new("/tmp/checkpoint-work"),
                CheckpointBuildMode::Derived,
            ),
            PathBuf::from("/tmp/checkpoint-work")
        );
    }

    #[test]
    fn checkpoint_builder_uses_runtime_checkpoint_subdir_for_full_snapshots() {
        assert_eq!(
            snapshot_dir_for_mode(
                std::path::Path::new("/tmp/checkpoint-work"),
                CheckpointBuildMode::Full,
            ),
            PathBuf::from("/tmp/checkpoint-work/checkpoints")
        );
    }

    #[cfg(feature = "pma-runtime-compat")]
    #[test]
    fn checkpoint_builder_rejects_derived_mode_under_pma_phase1() {
        let err = ensure_checkpoint_builder_supported(CheckpointBuildMode::Derived)
            .expect_err("derived mode should be rejected");
        assert!(matches!(err, CheckpointBuildError::Unsupported(_)));
        assert!(err
            .to_string()
            .contains("checkpoint builder is not supported under pma-runtime-compat in Phase 1"));
    }

    #[cfg(feature = "pma-runtime-compat")]
    #[test]
    fn checkpoint_builder_rejects_full_mode_under_pma_phase1() {
        let err = ensure_checkpoint_builder_supported(CheckpointBuildMode::Full)
            .expect_err("full mode should be rejected");
        assert!(matches!(err, CheckpointBuildError::Unsupported(_)));
        assert!(err
            .to_string()
            .contains("checkpoint builder is not supported under pma-runtime-compat in Phase 1"));
    }
}
