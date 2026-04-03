//! Bench-local compile-time compatibility helpers for PMA runtime support.

#[cfg(feature = "pma-runtime-compat")]
use std::fs;
#[cfg(feature = "pma-runtime-compat")]
use std::path::{Path, PathBuf};

#[cfg(feature = "pma-runtime-compat")]
use nockapp::kernel::boot::TraceOpts;
#[cfg(feature = "pma-runtime-compat")]
use nockapp::kernel::form::{Kernel, PmaConfig};
#[cfg(feature = "pma-runtime-compat")]
use nockapp::nockapp::save::SaveableCheckpoint;
#[cfg(feature = "pma-runtime-compat")]
use nockapp::nockapp::NockApp;
use nockapp::noun::slab::NounSlab;
use nockvm::noun::Noun;
#[cfg(feature = "pma-runtime-compat")]
use tracing::info;
#[cfg(feature = "pma-runtime-compat")]
use zkvm_jetpack::hot::produce_prover_hot_state;

#[cfg(feature = "pma-runtime-compat")]
use super::kernel_utils::KernelInitError;

#[cfg(feature = "pma-runtime-compat")]
fn replay_pma_dir(work_dir: &Path) -> PathBuf {
    work_dir.join("replay-pma")
}

#[cfg(feature = "pma-runtime-compat")]
fn prepare_replay_pma_dir(work_dir: &Path) -> Result<PathBuf, std::io::Error> {
    let replay_pma_dir = replay_pma_dir(work_dir);
    if replay_pma_dir.exists() {
        fs::remove_dir_all(&replay_pma_dir)?;
    }
    fs::create_dir_all(&replay_pma_dir)?;
    Ok(replay_pma_dir)
}

#[cfg(feature = "pma-runtime-compat")]
fn replay_pma_words() -> usize {
    nockapp::utils::NOCK_STACK_SIZE_MEDIUM
}

#[cfg(feature = "pma-runtime-compat")]
fn replay_pma_config(work_dir: &Path, fsync_enabled: bool) -> Result<PmaConfig, std::io::Error> {
    let replay_pma_dir = prepare_replay_pma_dir(work_dir)?;
    Ok(PmaConfig::for_nc_bench_shim(
        replay_pma_dir.join("0.pma"),
        replay_pma_dir.join("1.pma"),
        replay_pma_words(),
        None,
        fsync_enabled,
    ))
}

#[cfg(feature = "pma-runtime-compat")]
pub async fn init_replay_nockapp(
    kernel_path: &Path,
    checkpoint: Option<SaveableCheckpoint>,
    work_dir: &PathBuf,
    fsync_enabled: bool,
) -> Result<NockApp, KernelInitError> {
    let kernel_bytes = std::fs::read(kernel_path)?;
    info!(kernel_size = kernel_bytes.len(), "Loaded kernel jam");

    let hot_state = produce_prover_hot_state();
    info!(jets = hot_state.len(), "Got hot state entries");
    let replay_pma_config = replay_pma_config(work_dir, fsync_enabled)?;

    let kernel = Kernel::load_with_hot_state_medium(
        &kernel_bytes,
        checkpoint,
        &hot_state,
        vec![],
        TraceOpts::default(),
        Some(replay_pma_config),
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

#[cfg(all(test, feature = "pma-runtime-compat"))]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::{prepare_replay_pma_dir, replay_pma_config, replay_pma_dir, replay_pma_words};

    #[test]
    fn test_prepare_replay_pma_dir_recreates_directory_and_removes_stale_files() {
        let tempdir = tempdir().expect("tempdir should be created");
        let replay_pma_dir = replay_pma_dir(tempdir.path());
        fs::create_dir_all(&replay_pma_dir).expect("replay-pma dir should be created");
        fs::write(replay_pma_dir.join("0.pma"), b"stale slab 0").expect("stale slab 0");
        fs::write(replay_pma_dir.join("1.pma"), b"stale slab 1").expect("stale slab 1");

        let prepared_dir =
            prepare_replay_pma_dir(tempdir.path()).expect("replay-pma dir should be prepared");

        assert_eq!(prepared_dir, replay_pma_dir);
        assert_eq!(prepared_dir, tempdir.path().join("replay-pma"));
        assert!(prepared_dir.is_dir());
        assert!(!prepared_dir.join("0.pma").exists());
        assert!(!prepared_dir.join("1.pma").exists());
    }

    #[test]
    fn test_replay_pma_words_matches_expected_medium_stack_size() {
        assert_eq!(replay_pma_words(), nockapp::utils::NOCK_STACK_SIZE_MEDIUM);
    }

    #[test]
    fn test_replay_pma_config_returns_fresh_replay_shape() {
        let tempdir = tempdir().expect("tempdir should be created");

        let config =
            replay_pma_config(tempdir.path(), true).expect("replay config should be prepared");
        let replay_pma_dir = replay_pma_dir(tempdir.path());

        assert_eq!(config.path_0, replay_pma_dir.join("0.pma"));
        assert_eq!(config.path_1, replay_pma_dir.join("1.pma"));
        assert_eq!(config.words, replay_pma_words());
        assert!(!config.open_existing);
        assert!(!config.create_snapshots);
        assert_eq!(config.rotating_snapshot_interval_event_time, None);
        assert_eq!(config.gc_interval, None);
    }

    #[test]
    fn replay_pma_config_passes_fsync_modes_to_nc_bench_shim() {
        let tempdir = tempdir().expect("tempdir should be created");

        let config_on =
            replay_pma_config(tempdir.path(), true).expect("replay config should enable fsync");
        let replay_pma_dir = replay_pma_dir(tempdir.path());
        assert_eq!(config_on.path_0, replay_pma_dir.join("0.pma"));
        assert_eq!(config_on.path_1, replay_pma_dir.join("1.pma"));
        assert_eq!(config_on.words, replay_pma_words());
        assert!(!config_on.open_existing);
        assert!(!config_on.create_snapshots);
        assert_eq!(config_on.rotating_snapshot_interval_event_time, None);
        assert_eq!(config_on.gc_interval, None);

        let config_off =
            replay_pma_config(tempdir.path(), false).expect("replay config should disable fsync");
        assert_eq!(config_off.path_0, replay_pma_dir.join("0.pma"));
        assert_eq!(config_off.path_1, replay_pma_dir.join("1.pma"));
        assert_eq!(config_off.words, replay_pma_words());
        assert!(!config_off.open_existing);
        assert!(!config_off.create_snapshots);
        assert_eq!(config_off.rotating_snapshot_interval_event_time, None);
        assert_eq!(config_off.gc_interval, None);
    }
}
