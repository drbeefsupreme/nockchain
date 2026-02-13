//! Unified speed-of-light fixture format and builder.
//!
//! A fixture bundles:
//! - a derived checkpoint at `start_height - 1`
//! - a `.solarch` archive spanning `start_height..=end_height`
//! - the kernel jam used to build and run the fixture

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::checkpoint::{checkpoint_event_num, CheckpointMetaError};
use super::checkpoint_builder::{CheckpointBuildError, CheckpointBuilder, CheckpointConfig};
use super::extractor::{
    ArchiveExtractionPhase, ArchiveExtractionProgress, BlockExtractor, ExtractorConfig,
    ExtractorError,
};
use super::types::SolHeight;

const FIXTURE_MAGIC: &[u8; 8] = b"SOLTEST\0";
const FIXTURE_VERSION: u16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolFixtureManifest {
    pub format_version: u16,
    pub source_checkpoint_path: String,
    pub source_checkpoint_event_num: u64,
    pub derived_checkpoint_height: SolHeight,
    pub derived_checkpoint_event_num: u64,
    pub archive_start_height: SolHeight,
    pub archive_end_height: SolHeight,
    pub include_mempool: bool,
    pub chunk_size: u64,
    pub kernel_hash_hex: String,
    pub checkpoint_hash_hex: String,
    pub archive_hash_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolFixtureFile {
    pub manifest: SolFixtureManifest,
    pub checkpoint_bytes: Vec<u8>,
    pub archive_bytes: Vec<u8>,
    pub kernel_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct FixtureBuildConfig {
    pub source_checkpoint_path: PathBuf,
    pub kernel_path: PathBuf,
    pub start_height: SolHeight,
    pub end_height: SolHeight,
    pub output_path: PathBuf,
    pub work_dir: PathBuf,
    pub chunk_size: u64,
    pub include_mempool: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureBuildPhase {
    BootstrapArchive,
    DerivedCheckpoint,
    TestArchive,
    Packaging,
}

#[derive(Debug, Clone)]
pub struct FixtureBuildProgress {
    pub phase: FixtureBuildPhase,
    pub message: String,
    pub blocks_done: Option<u64>,
    pub blocks_total: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct FixtureBuildResult {
    pub output_path: PathBuf,
    pub derived_checkpoint_height: SolHeight,
    pub archive_start_height: SolHeight,
    pub archive_end_height: SolHeight,
}

#[derive(Debug, Error)]
pub enum FixtureError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] Box<bincode::ErrorKind>),

    #[error("Invalid fixture magic")]
    InvalidMagic,

    #[error("Unsupported fixture version: {0}")]
    UnsupportedVersion(u16),

    #[error("Truncated fixture payload")]
    TruncatedPayload,
}

#[derive(Debug, Error)]
pub enum FixtureBuildError {
    #[error("Invalid fixture build configuration: {0}")]
    InvalidConfig(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Extractor error: {0}")]
    Extractor(#[from] ExtractorError),

    #[error("Checkpoint builder error: {0}")]
    CheckpointBuilder(#[from] CheckpointBuildError),

    #[error("Checkpoint metadata error: {0}")]
    CheckpointMeta(#[from] CheckpointMetaError),

    #[error("Fixture file error: {0}")]
    Fixture(#[from] FixtureError),
}

pub fn write_fixture_file<P: AsRef<Path>>(
    path: P,
    fixture: &SolFixtureFile,
) -> Result<(), FixtureError> {
    let payload = bincode::serialize(fixture)?;
    let mut file = File::create(path.as_ref())?;
    file.write_all(FIXTURE_MAGIC)?;
    file.write_all(&FIXTURE_VERSION.to_le_bytes())?;
    file.write_all(&(payload.len() as u64).to_le_bytes())?;
    file.write_all(&payload)?;
    Ok(())
}

pub fn read_fixture_file<P: AsRef<Path>>(path: P) -> Result<SolFixtureFile, FixtureError> {
    let mut file = File::open(path.as_ref())?;

    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != FIXTURE_MAGIC {
        return Err(FixtureError::InvalidMagic);
    }

    let mut version_bytes = [0u8; 2];
    file.read_exact(&mut version_bytes)?;
    let version = u16::from_le_bytes(version_bytes);
    if version != FIXTURE_VERSION {
        return Err(FixtureError::UnsupportedVersion(version));
    }

    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let payload_len = u64::from_le_bytes(len_bytes) as usize;
    let mut payload = vec![0u8; payload_len];
    file.read_exact(&mut payload)?;
    if payload.len() != payload_len {
        return Err(FixtureError::TruncatedPayload);
    }

    Ok(bincode::deserialize(&payload)?)
}

pub struct FixtureBuilder {
    config: FixtureBuildConfig,
}

impl FixtureBuilder {
    pub fn new(config: FixtureBuildConfig) -> Self {
        Self { config }
    }

    pub async fn run(&self) -> Result<FixtureBuildResult, FixtureBuildError> {
        self.run_with_progress(|_| {}).await
    }

    pub async fn run_with_progress<F>(
        &self,
        mut on_progress: F,
    ) -> Result<FixtureBuildResult, FixtureBuildError>
    where
        F: FnMut(FixtureBuildProgress),
    {
        validate_build_config(&self.config)?;

        std::fs::create_dir_all(&self.config.work_dir)?;
        if let Some(parent) = self.config.output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let base_checkpoint_height = self.config.start_height.saturating_sub(1);
        let run_id = format!("{}-{}", std::process::id(), monotonic_id());
        let run_dir = self.config.work_dir.join(format!("sol-fixture-{run_id}"));
        std::fs::create_dir_all(&run_dir)?;
        let bootstrap_archive_path = run_dir.join("bootstrap.solarch");
        let test_archive_path = run_dir.join("test.solarch");
        let checkpoint_output_path = run_dir.join("derived.chkjam");
        let checkpoint_work_dir = run_dir.join("checkpoint-work");
        std::fs::create_dir_all(&checkpoint_work_dir)?;

        let extractor_base = ExtractorConfig {
            checkpoint_path: self
                .config
                .source_checkpoint_path
                .to_string_lossy()
                .to_string(),
            kernel_path: self.config.kernel_path.to_string_lossy().to_string(),
            block_count: 0,
            chunk_size: self.config.chunk_size,
            work_dir: run_dir.clone(),
            include_mempool: false,
        };

        on_progress(FixtureBuildProgress {
            phase: FixtureBuildPhase::BootstrapArchive,
            message: format!(
                "Extracting bootstrap archive 0..{}",
                base_checkpoint_height.as_u64()
            ),
            blocks_done: Some(0),
            blocks_total: Some(base_checkpoint_height.as_u64().saturating_add(1)),
        });
        let mut bootstrap_extractor = BlockExtractor::new(extractor_base.clone());
        bootstrap_extractor.initialize().await?;
        bootstrap_extractor
            .extract_range_to_archive_with_progress(
                0,
                base_checkpoint_height.as_u64(),
                &bootstrap_archive_path,
                |progress| {
                    emit_extract_progress(
                        &mut on_progress,
                        FixtureBuildPhase::BootstrapArchive,
                        progress,
                    )
                },
            )
            .await?;

        on_progress(FixtureBuildProgress {
            phase: FixtureBuildPhase::DerivedCheckpoint,
            message: format!(
                "Building derived checkpoint at height {}",
                base_checkpoint_height.as_u64()
            ),
            blocks_done: None,
            blocks_total: None,
        });
        let mut checkpoint_builder = CheckpointBuilder::new(CheckpointConfig {
            archive_path: bootstrap_archive_path.to_string_lossy().to_string(),
            kernel_path: self.config.kernel_path.to_string_lossy().to_string(),
            checkpoint_path: None,
            start_height: None,
            target_height: base_checkpoint_height,
            output_path: checkpoint_output_path.clone(),
            work_dir: checkpoint_work_dir,
        });
        checkpoint_builder.run().await?;

        on_progress(FixtureBuildProgress {
            phase: FixtureBuildPhase::TestArchive,
            message: format!(
                "Extracting test archive {}..{}",
                self.config.start_height.as_u64(),
                self.config.end_height.as_u64()
            ),
            blocks_done: Some(0),
            blocks_total: Some(
                self.config
                    .end_height
                    .as_u64()
                    .saturating_sub(self.config.start_height.as_u64())
                    .saturating_add(1),
            ),
        });
        if self.config.include_mempool {
            let mut mempool_extractor = BlockExtractor::new(ExtractorConfig {
                include_mempool: true,
                ..extractor_base
            });
            mempool_extractor.initialize().await?;
            mempool_extractor
                .extract_range_to_archive_with_progress(
                    self.config.start_height.as_u64(),
                    self.config.end_height.as_u64(),
                    &test_archive_path,
                    |progress| {
                        emit_extract_progress(
                            &mut on_progress,
                            FixtureBuildPhase::TestArchive,
                            progress,
                        )
                    },
                )
                .await?;
        } else {
            bootstrap_extractor
                .extract_range_to_archive_with_progress(
                    self.config.start_height.as_u64(),
                    self.config.end_height.as_u64(),
                    &test_archive_path,
                    |progress| {
                        emit_extract_progress(
                            &mut on_progress,
                            FixtureBuildPhase::TestArchive,
                            progress,
                        )
                    },
                )
                .await?;
        }

        on_progress(FixtureBuildProgress {
            phase: FixtureBuildPhase::Packaging,
            message: "Packaging .soltest fixture".to_string(),
            blocks_done: None,
            blocks_total: None,
        });

        let source_event_num = checkpoint_event_num(&self.config.source_checkpoint_path)?;
        let derived_event_num = checkpoint_event_num(&checkpoint_output_path)?;
        let kernel_bytes = std::fs::read(&self.config.kernel_path)?;
        let checkpoint_bytes = std::fs::read(&checkpoint_output_path)?;
        let archive_bytes = std::fs::read(&test_archive_path)?;

        let fixture = SolFixtureFile {
            manifest: SolFixtureManifest {
                format_version: FIXTURE_VERSION,
                source_checkpoint_path: self
                    .config
                    .source_checkpoint_path
                    .to_string_lossy()
                    .to_string(),
                source_checkpoint_event_num: source_event_num,
                derived_checkpoint_height: base_checkpoint_height,
                derived_checkpoint_event_num: derived_event_num,
                archive_start_height: self.config.start_height,
                archive_end_height: self.config.end_height,
                include_mempool: self.config.include_mempool,
                chunk_size: self.config.chunk_size,
                kernel_hash_hex: blake3::hash(&kernel_bytes).to_hex().to_string(),
                checkpoint_hash_hex: blake3::hash(&checkpoint_bytes).to_hex().to_string(),
                archive_hash_hex: blake3::hash(&archive_bytes).to_hex().to_string(),
            },
            checkpoint_bytes,
            archive_bytes,
            kernel_bytes,
        };

        write_fixture_file(&self.config.output_path, &fixture)?;

        Ok(FixtureBuildResult {
            output_path: self.config.output_path.clone(),
            derived_checkpoint_height: base_checkpoint_height,
            archive_start_height: self.config.start_height,
            archive_end_height: self.config.end_height,
        })
    }
}

fn validate_build_config(config: &FixtureBuildConfig) -> Result<(), FixtureBuildError> {
    if config.start_height == SolHeight::ZERO {
        return Err(FixtureBuildError::InvalidConfig(
            "start height must be greater than 0 to build a derived checkpoint".to_string(),
        ));
    }
    if config.start_height > config.end_height {
        return Err(FixtureBuildError::InvalidConfig(format!(
            "start height {} must be <= end height {}",
            config.start_height, config.end_height
        )));
    }
    if config.chunk_size == 0 {
        return Err(FixtureBuildError::InvalidConfig(
            "chunk size must be greater than 0".to_string(),
        ));
    }
    if !config.source_checkpoint_path.exists() {
        return Err(FixtureBuildError::InvalidConfig(format!(
            "source checkpoint file not found: {}",
            config.source_checkpoint_path.display()
        )));
    }
    if !config.kernel_path.exists() {
        return Err(FixtureBuildError::InvalidConfig(format!(
            "kernel file not found: {}",
            config.kernel_path.display()
        )));
    }
    Ok(())
}

fn monotonic_id() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn emit_extract_progress<F>(
    on_progress: &mut F,
    phase: FixtureBuildPhase,
    progress: ArchiveExtractionProgress,
) where
    F: FnMut(FixtureBuildProgress),
{
    let (blocks_done, blocks_total) = match progress.phase {
        ArchiveExtractionPhase::Blocks => (
            Some(progress.blocks_archived as u64),
            Some(progress.target_blocks),
        ),
        ArchiveExtractionPhase::MempoolReplay => (
            Some(progress.mempool_snapshots_done as u64),
            Some(progress.mempool_snapshots_total as u64),
        ),
        ArchiveExtractionPhase::Complete => (None, None),
    };
    let message = match progress.phase {
        ArchiveExtractionPhase::Blocks => format!(
            "blocks {}/{}",
            progress.blocks_archived, progress.target_blocks
        ),
        ArchiveExtractionPhase::MempoolReplay => format!(
            "mempool snapshots {}/{}",
            progress.mempool_snapshots_done, progress.mempool_snapshots_total
        ),
        ArchiveExtractionPhase::Complete => "archive complete".to_string(),
    };

    on_progress(FixtureBuildProgress {
        phase,
        message,
        blocks_done,
        blocks_total,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_file_roundtrip() {
        let fixture = SolFixtureFile {
            manifest: SolFixtureManifest {
                format_version: 1,
                source_checkpoint_path: "/tmp/source.chkjam".to_string(),
                source_checkpoint_event_num: 100_000,
                derived_checkpoint_height: SolHeight(49_999),
                derived_checkpoint_event_num: 49_999,
                archive_start_height: SolHeight(50_000),
                archive_end_height: SolHeight(60_000),
                include_mempool: false,
                chunk_size: 8,
                kernel_hash_hex: "k".repeat(64),
                checkpoint_hash_hex: "c".repeat(64),
                archive_hash_hex: "a".repeat(64),
            },
            checkpoint_bytes: vec![1, 2, 3],
            archive_bytes: vec![4, 5, 6],
            kernel_bytes: vec![7, 8, 9],
        };

        let temp_dir = tempfile::tempdir().expect("temp dir");
        let path = temp_dir.path().join("fixture.soltest");
        write_fixture_file(&path, &fixture).expect("write fixture");
        let loaded = read_fixture_file(&path).expect("read fixture");

        assert_eq!(loaded.manifest.archive_start_height, SolHeight(50_000));
        assert_eq!(loaded.checkpoint_bytes, vec![1, 2, 3]);
        assert_eq!(loaded.archive_bytes, vec![4, 5, 6]);
        assert_eq!(loaded.kernel_bytes, vec![7, 8, 9]);
    }
}
