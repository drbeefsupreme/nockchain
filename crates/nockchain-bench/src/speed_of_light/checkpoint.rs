//! Checkpoint loading utilities for speed-of-light benchmarks

use std::path::Path;

use nockapp::nockapp::save::{CheckpointError, JammedCheckpointV2};
use nockapp::noun::slab::NounSlab;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CheckpointLoadError {
    #[error("IO error reading checkpoint: {0}")]
    Io(#[from] std::io::Error),

    #[error("Checkpoint decode error: {0}")]
    Checkpoint(#[from] CheckpointError),

    #[error("Cue error: {0}")]
    Cue(#[from] nockapp::noun::slab::CueError),
}

/// Loaded checkpoint data ready for use
pub struct LoadedCheckpoint {
    /// Kernel state as a NounSlab
    pub state: NounSlab,
    /// Cold jet state as a NounSlab
    pub cold: NounSlab,
    /// Event number at checkpoint
    pub event_num: u64,
    /// Kernel hash
    pub ker_hash: blake3::Hash,
}

/// Load a checkpoint from a .chkjam file
///
/// # Arguments
/// * `path` - Path to the checkpoint file (e.g., "0.chkjam")
///
/// # Returns
/// A `LoadedCheckpoint` containing the kernel state and cold state as NounSlabs
pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<LoadedCheckpoint, CheckpointLoadError> {
    let bytes = std::fs::read(path.as_ref())?;
    load_checkpoint_from_bytes(&bytes)
}

/// Load a checkpoint from raw bytes
pub fn load_checkpoint_from_bytes(bytes: &[u8]) -> Result<LoadedCheckpoint, CheckpointLoadError> {
    let jammed = JammedCheckpointV2::decode_from_bytes(bytes)?;

    let ker_hash = jammed.ker_hash;
    let event_num = jammed.event_num;

    // Cue the state jam into a NounSlab
    let mut state_slab = NounSlab::new();
    let state_root = state_slab.cue_into(jammed.state_jam.0.clone())?;
    state_slab.set_root(state_root);

    // Cue the cold jam into a NounSlab
    let mut cold_slab = NounSlab::new();
    let cold_root = cold_slab.cue_into(jammed.cold_jam.0.clone())?;
    cold_slab.set_root(cold_root);

    Ok(LoadedCheckpoint {
        state: state_slab,
        cold: cold_slab,
        event_num,
        ker_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires checkpoint file"]
    fn test_load_checkpoint() {
        let checkpoint_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("0.chkjam");

        if checkpoint_path.exists() {
            let loaded = load_checkpoint(&checkpoint_path).expect("should load checkpoint");
            println!("Loaded checkpoint at event_num: {}", loaded.event_num);
        }
    }
}
