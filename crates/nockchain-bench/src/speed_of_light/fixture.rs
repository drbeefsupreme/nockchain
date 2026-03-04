//! Unified speed-of-light fixture format and builder.
//!
//! A fixture bundles:
//! - a derived checkpoint at `start_height - 1`
//! - a `.solarch` archive spanning `start_height..=end_height`
//! - the kernel jam used to build and run the fixture

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
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
const FIXTURE_VERSION: u16 = 2;
const MAX_FIXTURE_FILE_BYTES: u64 = 16 * 1024 * 1024 * 1024; // 16 GiB
const MAX_FIXTURE_MANIFEST_BYTES: u64 = 1 * 1024 * 1024; // 1 MiB
const MAX_FIXTURE_SECTION_BYTES: u64 = 8 * 1024 * 1024 * 1024; // 8 GiB per section

#[derive(Debug, Clone, Copy)]
struct FixtureSectionLayout {
    manifest_len: u64,
    checkpoint_len: u64,
    archive_len: u64,
    kernel_len: u64,
}

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

    #[error("Fixture section lengths overflow")]
    LengthOverflow,

    #[error("Limit exceeded for {field}: {value} > {max}")]
    LimitExceeded {
        field: &'static str,
        value: u64,
        max: u64,
    },
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
    let manifest_bytes = bincode::serialize(&fixture.manifest)?;
    let layout = FixtureSectionLayout {
        manifest_len: manifest_bytes.len() as u64,
        checkpoint_len: fixture.checkpoint_bytes.len() as u64,
        archive_len: fixture.archive_bytes.len() as u64,
        kernel_len: fixture.kernel_bytes.len() as u64,
    };
    validate_layout_limits(layout)?;

    let mut writer = BufWriter::new(File::create(path.as_ref())?);
    write_fixture_header(&mut writer)?;
    write_fixture_layout(&mut writer, layout)?;
    writer.write_all(&manifest_bytes)?;
    writer.write_all(&fixture.checkpoint_bytes)?;
    writer.write_all(&fixture.archive_bytes)?;
    writer.write_all(&fixture.kernel_bytes)?;
    writer.flush()?;
    Ok(())
}

pub fn write_fixture_file_from_paths<P: AsRef<Path>>(
    path: P,
    manifest: &SolFixtureManifest,
    checkpoint_path: &Path,
    archive_path: &Path,
    kernel_path: &Path,
) -> Result<(), FixtureError> {
    let manifest_bytes = bincode::serialize(manifest)?;
    let layout = FixtureSectionLayout {
        manifest_len: manifest_bytes.len() as u64,
        checkpoint_len: std::fs::metadata(checkpoint_path)?.len(),
        archive_len: std::fs::metadata(archive_path)?.len(),
        kernel_len: std::fs::metadata(kernel_path)?.len(),
    };
    validate_layout_limits(layout)?;

    let mut writer = BufWriter::new(File::create(path.as_ref())?);
    write_fixture_header(&mut writer)?;
    write_fixture_layout(&mut writer, layout)?;
    writer.write_all(&manifest_bytes)?;
    copy_path_to_writer(checkpoint_path, &mut writer)?;
    copy_path_to_writer(archive_path, &mut writer)?;
    copy_path_to_writer(kernel_path, &mut writer)?;
    writer.flush()?;
    Ok(())
}

pub fn read_fixture_file<P: AsRef<Path>>(path: P) -> Result<SolFixtureFile, FixtureError> {
    ensure_fixture_file_size(path.as_ref())?;
    let mut reader = BufReader::new(File::open(path.as_ref())?);
    let version = read_fixture_header(&mut reader)?;
    if version != FIXTURE_VERSION {
        return Err(FixtureError::UnsupportedVersion(version));
    }
    read_fixture_v2(&mut reader)
}

pub fn extract_fixture_to_paths<P: AsRef<Path>>(
    fixture_path: P,
    checkpoint_path: &Path,
    archive_path: &Path,
    kernel_path: &Path,
) -> Result<SolFixtureManifest, FixtureError> {
    ensure_fixture_file_size(fixture_path.as_ref())?;
    let mut reader = BufReader::new(File::open(fixture_path.as_ref())?);
    let version = read_fixture_header(&mut reader)?;
    if version != FIXTURE_VERSION {
        return Err(FixtureError::UnsupportedVersion(version));
    }
    let layout = read_fixture_layout(&mut reader)?;
    let manifest: SolFixtureManifest =
        bincode::deserialize(&read_exact_vec(&mut reader, layout.manifest_len)?)?;
    copy_reader_to_path_exact(&mut reader, checkpoint_path, layout.checkpoint_len)?;
    copy_reader_to_path_exact(&mut reader, archive_path, layout.archive_len)?;
    copy_reader_to_path_exact(&mut reader, kernel_path, layout.kernel_len)?;
    Ok(manifest)
}

fn write_fixture_header<W: Write>(writer: &mut W) -> Result<(), FixtureError> {
    writer.write_all(FIXTURE_MAGIC)?;
    writer.write_all(&FIXTURE_VERSION.to_le_bytes())?;
    Ok(())
}

fn read_fixture_header<R: Read>(reader: &mut R) -> Result<u16, FixtureError> {
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != FIXTURE_MAGIC {
        return Err(FixtureError::InvalidMagic);
    }
    let mut version_bytes = [0u8; 2];
    reader.read_exact(&mut version_bytes)?;
    Ok(u16::from_le_bytes(version_bytes))
}

fn write_fixture_layout<W: Write>(
    writer: &mut W,
    layout: FixtureSectionLayout,
) -> Result<(), FixtureError> {
    writer.write_all(&layout.manifest_len.to_le_bytes())?;
    writer.write_all(&layout.checkpoint_len.to_le_bytes())?;
    writer.write_all(&layout.archive_len.to_le_bytes())?;
    writer.write_all(&layout.kernel_len.to_le_bytes())?;
    Ok(())
}

fn read_fixture_layout<R: Read>(reader: &mut R) -> Result<FixtureSectionLayout, FixtureError> {
    let manifest_len = read_u64(reader)?;
    let checkpoint_len = read_u64(reader)?;
    let archive_len = read_u64(reader)?;
    let kernel_len = read_u64(reader)?;
    let layout = FixtureSectionLayout {
        manifest_len,
        checkpoint_len,
        archive_len,
        kernel_len,
    };
    validate_layout_limits(layout)?;
    Ok(layout)
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, FixtureError> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_exact_vec<R: Read>(reader: &mut R, len: u64) -> Result<Vec<u8>, FixtureError> {
    let len = usize::try_from(len).map_err(|_| FixtureError::TruncatedPayload)?;
    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload)?;
    Ok(payload)
}

fn read_fixture_v2<R: Read>(reader: &mut R) -> Result<SolFixtureFile, FixtureError> {
    let layout = read_fixture_layout(reader)?;
    let manifest: SolFixtureManifest =
        bincode::deserialize(&read_exact_vec(reader, layout.manifest_len)?)?;
    let checkpoint_bytes = read_exact_vec(reader, layout.checkpoint_len)?;
    let archive_bytes = read_exact_vec(reader, layout.archive_len)?;
    let kernel_bytes = read_exact_vec(reader, layout.kernel_len)?;
    Ok(SolFixtureFile {
        manifest,
        checkpoint_bytes,
        archive_bytes,
        kernel_bytes,
    })
}

fn copy_path_to_writer<W: Write>(source_path: &Path, writer: &mut W) -> Result<(), FixtureError> {
    let mut source = File::open(source_path)?;
    std::io::copy(&mut source, writer)?;
    Ok(())
}

fn copy_reader_to_path_exact<R: Read>(
    reader: &mut R,
    destination_path: &Path,
    len: u64,
) -> Result<(), FixtureError> {
    let mut destination = BufWriter::new(File::create(destination_path)?);
    let mut limited = reader.take(len);
    let copied = std::io::copy(&mut limited, &mut destination)?;
    if copied != len {
        return Err(FixtureError::TruncatedPayload);
    }
    destination.flush()?;
    Ok(())
}

fn ensure_fixture_file_size(path: &Path) -> Result<(), FixtureError> {
    let file_size = std::fs::metadata(path)?.len();
    enforce_limit("fixture.file_size", file_size, MAX_FIXTURE_FILE_BYTES)
}

fn validate_layout_limits(layout: FixtureSectionLayout) -> Result<(), FixtureError> {
    enforce_limit(
        "fixture.manifest_bytes", layout.manifest_len, MAX_FIXTURE_MANIFEST_BYTES,
    )?;
    enforce_limit(
        "fixture.checkpoint_bytes", layout.checkpoint_len, MAX_FIXTURE_SECTION_BYTES,
    )?;
    enforce_limit(
        "fixture.archive_bytes", layout.archive_len, MAX_FIXTURE_SECTION_BYTES,
    )?;
    enforce_limit(
        "fixture.kernel_bytes", layout.kernel_len, MAX_FIXTURE_SECTION_BYTES,
    )?;

    let total_size = fixture_stream_file_size(layout)?;
    enforce_limit("fixture.file_size", total_size, MAX_FIXTURE_FILE_BYTES)?;
    Ok(())
}

fn fixture_stream_file_size(layout: FixtureSectionLayout) -> Result<u64, FixtureError> {
    let header_bytes = 8u64 + 2 + 8 + 8 + 8 + 8;
    header_bytes
        .checked_add(layout.manifest_len)
        .and_then(|sum| sum.checked_add(layout.checkpoint_len))
        .and_then(|sum| sum.checked_add(layout.archive_len))
        .and_then(|sum| sum.checked_add(layout.kernel_len))
        .ok_or(FixtureError::LengthOverflow)
}

fn enforce_limit(field: &'static str, value: u64, max: u64) -> Result<(), FixtureError> {
    if value > max {
        return Err(FixtureError::LimitExceeded { field, value, max });
    }
    Ok(())
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
        let fixture_manifest = SolFixtureManifest {
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
            kernel_hash_hex: blake3_hash_hex_for_file(&self.config.kernel_path)?,
            checkpoint_hash_hex: blake3_hash_hex_for_file(&checkpoint_output_path)?,
            archive_hash_hex: blake3_hash_hex_for_file(&test_archive_path)?,
        };

        write_fixture_file_from_paths(
            &self.config.output_path, &fixture_manifest, &checkpoint_output_path,
            &test_archive_path, &self.config.kernel_path,
        )?;

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

fn blake3_hash_hex_for_file(path: &Path) -> Result<String, FixtureBuildError> {
    let mut file = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
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
    use std::io::Write;

    use super::*;

    #[test]
    fn test_fixture_file_roundtrip() {
        let fixture = SolFixtureFile {
            manifest: SolFixtureManifest {
                format_version: FIXTURE_VERSION,
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

    #[test]
    fn test_extract_fixture_to_paths_roundtrip() {
        let fixture = SolFixtureFile {
            manifest: SolFixtureManifest {
                format_version: FIXTURE_VERSION,
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
        let fixture_path = temp_dir.path().join("fixture.soltest");
        let checkpoint_path = temp_dir.path().join("fixture.chkjam");
        let archive_path = temp_dir.path().join("fixture.solarch");
        let kernel_path = temp_dir.path().join("fixture.jam");

        write_fixture_file(&fixture_path, &fixture).expect("write fixture");
        let manifest =
            extract_fixture_to_paths(&fixture_path, &checkpoint_path, &archive_path, &kernel_path)
                .expect("extract fixture");

        assert_eq!(manifest.archive_start_height, SolHeight(50_000));
        assert_eq!(
            std::fs::read(checkpoint_path).expect("read checkpoint"),
            vec![1, 2, 3]
        );
        assert_eq!(
            std::fs::read(archive_path).expect("read archive"),
            vec![4, 5, 6]
        );
        assert_eq!(
            std::fs::read(kernel_path).expect("read kernel"),
            vec![7, 8, 9]
        );
    }

    #[test]
    fn test_read_fixture_rejects_v1_files() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let path = temp_dir.path().join("legacy-v1.soltest");
        let mut file = File::create(&path).expect("create fixture");
        file.write_all(FIXTURE_MAGIC).expect("write magic");
        file.write_all(&1u16.to_le_bytes()).expect("write version");
        file.write_all(&0u64.to_le_bytes())
            .expect("write payload len");
        file.flush().expect("flush fixture");

        let err = read_fixture_file(&path).expect_err("v1 fixtures should be rejected");
        assert!(matches!(err, FixtureError::UnsupportedVersion(1)));
    }
}
