use std::path::PathBuf;

use nockchain_bench::speed_of_light::extractor::ExtractorError;
use nockchain_bench::speed_of_light::{
    slice_archive_file, BlockExtractor, CheckpointBuilder, CheckpointConfig, ExtractorConfig,
    NockStackProfile, SolHeight,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate layout should be crates/nockchain-bench")
        .to_path_buf()
}

fn full_checkpoint_path() -> PathBuf {
    std::env::var_os("NOCKCHAIN_SOL_FULL_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("checkpoints").join("0.chkjam"))
}

fn kernel_path() -> PathBuf {
    std::env::var_os("NOCKCHAIN_SOL_KERNEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("assets").join("dumb.jam"))
}

#[tokio::test]
#[ignore = "Requires full checkpoint and takes several minutes"]
async fn full_checkpoint_slice_and_derived_reextract_behavior() {
    let checkpoint = full_checkpoint_path();
    let kernel = kernel_path();

    if !checkpoint.is_file() || !kernel.is_file() {
        eprintln!(
            "Skipping full-checkpoint slicer test; missing inputs checkpoint={} kernel={}",
            checkpoint.display(),
            kernel.display()
        );
        return;
    }

    let temp = tempfile::tempdir().expect("temp dir");
    let work_dir = temp.path().join("work");
    let checkpoint_work = temp.path().join("checkpoint-work");
    std::fs::create_dir_all(&work_dir).expect("create work dir");
    std::fs::create_dir_all(&checkpoint_work).expect("create checkpoint work dir");

    let bootstrap_archive = temp.path().join("bootstrap-0-999.solarch");
    let derived_checkpoint = temp.path().join("derived-999.chkjam");
    let derived_archive_attempt = temp.path().join("derived-0-999.solarch");
    let sliced_archive = temp.path().join("slice-200-299.solarch");

    let mut extractor = BlockExtractor::new(ExtractorConfig {
        checkpoint_path: checkpoint.to_string_lossy().to_string(),
        kernel_path: kernel.to_string_lossy().to_string(),
        block_count: 1000,
        chunk_size: 8,
        work_dir: work_dir.clone(),
        include_mempool: false,
        stack_profile: NockStackProfile::Large,
    });
    extractor
        .initialize()
        .await
        .expect("initialize full extractor");
    extractor
        .extract_range_to_archive_with_progress(0, 999, &bootstrap_archive, |_| {})
        .await
        .expect("extract first 1000 blocks from full checkpoint");

    let mut checkpoint_builder = CheckpointBuilder::new(CheckpointConfig {
        archive_path: bootstrap_archive.to_string_lossy().to_string(),
        kernel_path: kernel.to_string_lossy().to_string(),
        checkpoint_path: None,
        start_height: None,
        target_height: SolHeight(999),
        output_path: derived_checkpoint.clone(),
        work_dir: checkpoint_work,
        stack_profile: NockStackProfile::Large,
    });
    checkpoint_builder
        .run()
        .await
        .expect("build derived checkpoint at 999");

    let slice_result = slice_archive_file(
        &bootstrap_archive,
        &sliced_archive,
        temp.path(),
        SolHeight(200),
        SolHeight(299),
        false,
    )
    .expect("slice 200..299 from full-checkpoint archive");
    assert_eq!(slice_result.block_count, 100);
    assert_eq!(slice_result.start_height, SolHeight(200));
    assert_eq!(slice_result.end_height, SolHeight(299));
    assert_eq!(slice_result.mempool_snapshot_count, 0);

    let mut derived_extractor = BlockExtractor::new(ExtractorConfig {
        checkpoint_path: derived_checkpoint.to_string_lossy().to_string(),
        kernel_path: kernel.to_string_lossy().to_string(),
        block_count: 1000,
        chunk_size: 8,
        work_dir,
        include_mempool: false,
        stack_profile: NockStackProfile::Large,
    });
    derived_extractor
        .initialize()
        .await
        .expect("initialize derived extractor");

    let err = derived_extractor
        .extract_range_to_archive_with_progress(0, 999, &derived_archive_attempt, |_| {})
        .await
        .expect_err("derived checkpoint should not replay historical 0..999 archive");
    match err {
        ExtractorError::StartAboveChainTip { start, tip } => {
            assert_eq!(start, 0);
            assert_eq!(tip, 999);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
