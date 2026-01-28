//! Speed-of-light archive format
//!
//! Stores extracted blocks as jammed noun blobs with bincode metadata
//! for fast loading during benchmark runs.
//!
//! File format:
//! ```text
//! ┌─────────────────────────────────────┐
//! │ Metadata Header (bincode)           │
//! │  - magic, version, counts           │
//! │  - Vec<BlockEntry> index            │
//! ├─────────────────────────────────────┤
//! │ Jammed Noun Blobs                   │
//! │  - block 0 jam bytes                │
//! │  - block 1 jam bytes                │
//! │  - ...                              │
//! └─────────────────────────────────────┘
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use nockchain_types::tx_engine::common::Hash;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Magic bytes to identify speed-of-light archive files
pub const ARCHIVE_MAGIC: &[u8; 8] = b"SOLARCH\0";

/// Current archive format version
pub const ARCHIVE_VERSION: u32 = 1;

/// Errors that can occur when working with archives
#[derive(Debug, Error)]
pub enum ArchiveError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),

    #[error("Invalid magic bytes")]
    InvalidMagic,

    #[error("Unsupported archive version: {0} (expected {1})")]
    UnsupportedVersion(u32, u32),

    #[error("Block not found at height {0}")]
    BlockNotFound(u64),

    #[error("Archive is empty")]
    EmptyArchive,
}

/// Metadata for a single block in the archive
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockEntry {
    /// Block height
    pub height: u64,
    /// Block ID hash
    pub block_id: Hash,
    /// Number of transactions in this block
    pub tx_count: usize,
    /// Offset into the jam blob section (bytes from start of blob section)
    pub jam_offset: u64,
    /// Size of the jammed noun blob in bytes
    pub jam_size: u64,
}

/// Archive header with metadata and block index
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArchiveMetadata {
    /// Magic bytes for file identification
    pub magic: [u8; 8],
    /// Archive format version
    pub version: u32,
    /// Total number of blocks in archive
    pub block_count: u64,
    /// Total number of transactions across all blocks
    pub total_tx_count: u64,
    /// Minimum block height in archive
    pub min_height: u64,
    /// Maximum block height in archive
    pub max_height: u64,
    /// Hash of source checkpoint (for provenance tracking)
    pub source_checkpoint_hash: Option<Hash>,
    /// Block entries (index)
    pub blocks: Vec<BlockEntry>,
}

impl ArchiveMetadata {
    /// Create a new empty archive metadata
    pub fn new() -> Self {
        Self {
            magic: *ARCHIVE_MAGIC,
            version: ARCHIVE_VERSION,
            block_count: 0,
            total_tx_count: 0,
            min_height: u64::MAX,
            max_height: 0,
            source_checkpoint_hash: None,
            blocks: Vec::new(),
        }
    }

    /// Create metadata with a source checkpoint hash
    pub fn with_source(source_hash: Hash) -> Self {
        let mut meta = Self::new();
        meta.source_checkpoint_hash = Some(source_hash);
        meta
    }

    /// Add a block entry to the metadata
    pub fn add_block(&mut self, entry: BlockEntry) {
        // Update stats
        if entry.height < self.min_height {
            self.min_height = entry.height;
        }
        if entry.height > self.max_height {
            self.max_height = entry.height;
        }
        self.total_tx_count += entry.tx_count as u64;
        self.block_count += 1;

        self.blocks.push(entry);
    }

    /// Validate the metadata
    pub fn validate(&self) -> Result<(), ArchiveError> {
        if self.magic != *ARCHIVE_MAGIC {
            return Err(ArchiveError::InvalidMagic);
        }
        if self.version != ARCHIVE_VERSION {
            return Err(ArchiveError::UnsupportedVersion(self.version, ARCHIVE_VERSION));
        }
        Ok(())
    }

    /// Serialize metadata to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, ArchiveError> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize metadata from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ArchiveError> {
        let meta: Self = bincode::deserialize(bytes)?;
        meta.validate()?;
        Ok(meta)
    }

    /// Get block entry by height
    pub fn get_block(&self, height: u64) -> Option<&BlockEntry> {
        self.blocks.iter().find(|b| b.height == height)
    }

    /// Get block entry by index
    pub fn get_block_by_index(&self, index: usize) -> Option<&BlockEntry> {
        self.blocks.get(index)
    }
}

impl Default for ArchiveMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Writer for creating speed-of-light archive files
///
/// Accumulates blocks with their jammed noun bytes, then writes
/// everything to a file in the archive format.
///
/// # Example
/// ```ignore
/// let mut writer = ArchiveWriter::new();
/// writer.add_block(0, block_id, 0, &jammed_bytes)?;
/// writer.add_block(1, block_id, 2, &jammed_bytes)?;
/// writer.write_to_file("blocks.solar")?;
/// ```
pub struct ArchiveWriter {
    metadata: ArchiveMetadata,
    /// Accumulated jammed noun blobs
    jam_blobs: Vec<u8>,
}

impl ArchiveWriter {
    /// Create a new archive writer
    pub fn new() -> Self {
        Self {
            metadata: ArchiveMetadata::new(),
            jam_blobs: Vec::new(),
        }
    }

    /// Create a new archive writer with source checkpoint hash
    pub fn with_source(source_hash: Hash) -> Self {
        Self {
            metadata: ArchiveMetadata::with_source(source_hash),
            jam_blobs: Vec::new(),
        }
    }

    /// Add a block to the archive
    ///
    /// # Arguments
    /// * `height` - Block height
    /// * `block_id` - Block ID hash
    /// * `tx_count` - Number of transactions in the block
    /// * `jam_bytes` - The jammed noun bytes for this block
    pub fn add_block(
        &mut self,
        height: u64,
        block_id: Hash,
        tx_count: usize,
        jam_bytes: &[u8],
    ) -> Result<(), ArchiveError> {
        let jam_offset = self.jam_blobs.len() as u64;
        let jam_size = jam_bytes.len() as u64;

        // Add jam bytes to blob section
        self.jam_blobs.extend_from_slice(jam_bytes);

        // Add entry to metadata
        self.metadata.add_block(BlockEntry {
            height,
            block_id,
            tx_count,
            jam_offset,
            jam_size,
        });

        Ok(())
    }

    /// Get the current number of blocks in the archive
    pub fn block_count(&self) -> u64 {
        self.metadata.block_count
    }

    /// Get the total size of jam blobs accumulated so far
    pub fn jam_blob_size(&self) -> usize {
        self.jam_blobs.len()
    }

    /// Write the archive to a file
    ///
    /// File format:
    /// - 8 bytes: metadata length (u64 little-endian)
    /// - N bytes: bincode-serialized metadata
    /// - M bytes: concatenated jam blobs
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ArchiveError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        self.write_to(&mut writer)?;
        writer.flush()?;

        Ok(())
    }

    /// Write the archive to any writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), ArchiveError> {
        // Serialize metadata
        let meta_bytes = self.metadata.to_bytes()?;
        let meta_len = meta_bytes.len() as u64;

        // Write metadata length (8 bytes, little-endian)
        writer.write_all(&meta_len.to_le_bytes())?;

        // Write metadata
        writer.write_all(&meta_bytes)?;

        // Write jam blobs
        writer.write_all(&self.jam_blobs)?;

        Ok(())
    }

    /// Convert to bytes (for testing)
    pub fn to_bytes(&self) -> Result<Vec<u8>, ArchiveError> {
        let mut buffer = Vec::new();
        self.write_to(&mut buffer)?;
        Ok(buffer)
    }

    /// Get a reference to the metadata
    pub fn metadata(&self) -> &ArchiveMetadata {
        &self.metadata
    }
}

impl Default for ArchiveWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Reader for speed-of-light archive files
///
/// Provides access to archived blocks and their jammed noun bytes.
/// Supports both in-memory and memory-mapped access patterns.
///
/// # Example
/// ```ignore
/// let reader = ArchiveReader::from_file("blocks.solar")?;
/// println!("Archive has {} blocks", reader.block_count());
///
/// // Get jam bytes for a specific block
/// let jam = reader.get_jam_by_height(5629)?;
///
/// // Iterate through all blocks
/// for (entry, jam_bytes) in reader.iter() {
///     println!("Block {}: {} bytes", entry.height, jam_bytes.len());
/// }
/// ```
pub struct ArchiveReader {
    /// Parsed metadata with block index
    metadata: ArchiveMetadata,
    /// Raw bytes of the jam blob section
    jam_section: Vec<u8>,
}

impl ArchiveReader {
    /// Open an archive from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ArchiveError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(bytes)
    }

    /// Parse an archive from a byte buffer
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, ArchiveError> {
        if bytes.len() < 8 {
            return Err(ArchiveError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "file too small for metadata length",
            )));
        }

        // Read metadata length
        let meta_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;

        if bytes.len() < 8 + meta_len {
            return Err(ArchiveError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "file too small for metadata",
            )));
        }

        // Parse metadata
        let meta_bytes = &bytes[8..8 + meta_len];
        let metadata = ArchiveMetadata::from_bytes(meta_bytes)?;

        // Extract jam section
        let jam_section = bytes[8 + meta_len..].to_vec();

        Ok(Self {
            metadata,
            jam_section,
        })
    }

    /// Get the archive metadata
    pub fn metadata(&self) -> &ArchiveMetadata {
        &self.metadata
    }

    /// Get the number of blocks in the archive
    pub fn block_count(&self) -> u64 {
        self.metadata.block_count
    }

    /// Get the total number of transactions across all blocks
    pub fn total_tx_count(&self) -> u64 {
        self.metadata.total_tx_count
    }

    /// Get the minimum block height in the archive
    pub fn min_height(&self) -> u64 {
        self.metadata.min_height
    }

    /// Get the maximum block height in the archive
    pub fn max_height(&self) -> u64 {
        self.metadata.max_height
    }

    /// Get jam bytes for a block by height
    pub fn get_jam_by_height(&self, height: u64) -> Result<&[u8], ArchiveError> {
        let entry = self
            .metadata
            .get_block(height)
            .ok_or(ArchiveError::BlockNotFound(height))?;

        self.get_jam_for_entry(entry)
    }

    /// Get jam bytes for a block by index
    pub fn get_jam_by_index(&self, index: usize) -> Result<&[u8], ArchiveError> {
        let entry = self
            .metadata
            .get_block_by_index(index)
            .ok_or(ArchiveError::BlockNotFound(index as u64))?;

        self.get_jam_for_entry(entry)
    }

    /// Get block entry by height
    pub fn get_entry_by_height(&self, height: u64) -> Option<&BlockEntry> {
        self.metadata.get_block(height)
    }

    /// Get block entry by index
    pub fn get_entry_by_index(&self, index: usize) -> Option<&BlockEntry> {
        self.metadata.get_block_by_index(index)
    }

    /// Internal: get jam bytes for a block entry
    fn get_jam_for_entry(&self, entry: &BlockEntry) -> Result<&[u8], ArchiveError> {
        let start = entry.jam_offset as usize;
        let end = start + entry.jam_size as usize;

        if end > self.jam_section.len() {
            return Err(ArchiveError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "jam blob out of bounds: offset={}, size={}, section_len={}",
                    entry.jam_offset,
                    entry.jam_size,
                    self.jam_section.len()
                ),
            )));
        }

        Ok(&self.jam_section[start..end])
    }

    /// Iterate over all blocks in index order (which is insertion order)
    pub fn iter(&self) -> ArchiveIterator<'_> {
        ArchiveIterator {
            reader: self,
            index: 0,
        }
    }

    /// Iterate over blocks in a height range (inclusive)
    /// Note: Only yields blocks that exist in the archive within the range
    pub fn iter_range(&self, start_height: u64, end_height: u64) -> ArchiveRangeIterator<'_> {
        ArchiveRangeIterator {
            reader: self,
            current_height: start_height,
            end_height,
        }
    }
}

/// Iterator over all blocks in an archive (by index order)
pub struct ArchiveIterator<'a> {
    reader: &'a ArchiveReader,
    index: usize,
}

impl<'a> Iterator for ArchiveIterator<'a> {
    type Item = (&'a BlockEntry, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.reader.get_entry_by_index(self.index)?;
        let jam = self.reader.get_jam_for_entry(entry).ok()?;
        self.index += 1;
        Some((entry, jam))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.reader.metadata.blocks.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for ArchiveIterator<'a> {}

/// Iterator over blocks in a height range
pub struct ArchiveRangeIterator<'a> {
    reader: &'a ArchiveReader,
    current_height: u64,
    end_height: u64,
}

impl<'a> Iterator for ArchiveRangeIterator<'a> {
    type Item = (&'a BlockEntry, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_height <= self.end_height {
            let height = self.current_height;
            self.current_height += 1;

            if let Some(entry) = self.reader.get_entry_by_height(height) {
                if let Ok(jam) = self.reader.get_jam_for_entry(entry) {
                    return Some((entry, jam));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nockchain_math::belt::Belt;

    fn dummy_hash(v: u64) -> Hash {
        Hash([Belt(v), Belt(v + 1), Belt(v + 2), Belt(v + 3), Belt(v + 4)])
    }

    /// Test that ArchiveMetadata serializes and deserializes correctly
    #[test]
    fn test_archive_metadata_roundtrip() {
        let mut meta = ArchiveMetadata::new();
        meta.source_checkpoint_hash = Some(dummy_hash(12345));

        // Add some block entries
        meta.add_block(BlockEntry {
            height: 0,
            block_id: dummy_hash(100),
            tx_count: 0,
            jam_offset: 0,
            jam_size: 1024,
        });
        meta.add_block(BlockEntry {
            height: 1,
            block_id: dummy_hash(101),
            tx_count: 2,
            jam_offset: 1024,
            jam_size: 2048,
        });
        meta.add_block(BlockEntry {
            height: 2,
            block_id: dummy_hash(102),
            tx_count: 1,
            jam_offset: 3072,
            jam_size: 512,
        });

        // Serialize
        let bytes = meta.to_bytes().expect("serialization should succeed");

        // Deserialize
        let restored = ArchiveMetadata::from_bytes(&bytes).expect("deserialization should succeed");

        // Verify
        assert_eq!(meta, restored);
        assert_eq!(restored.block_count, 3);
        assert_eq!(restored.total_tx_count, 3);
        assert_eq!(restored.min_height, 0);
        assert_eq!(restored.max_height, 2);
        assert_eq!(restored.blocks.len(), 3);
    }

    /// Test that BlockEntry serializes and deserializes correctly
    #[test]
    fn test_block_entry_roundtrip() {
        let entry = BlockEntry {
            height: 5629,
            block_id: dummy_hash(999),
            tx_count: 5,
            jam_offset: 123456,
            jam_size: 7890,
        };

        let bytes = bincode::serialize(&entry).expect("serialization should succeed");
        let restored: BlockEntry = bincode::deserialize(&bytes).expect("deserialization should succeed");

        assert_eq!(entry, restored);
        assert_eq!(restored.height, 5629);
        assert_eq!(restored.tx_count, 5);
        assert_eq!(restored.jam_offset, 123456);
        assert_eq!(restored.jam_size, 7890);
    }

    /// Test that version validation works correctly
    #[test]
    fn test_archive_version_check() {
        // Valid metadata should pass validation
        let valid_meta = ArchiveMetadata::new();
        assert!(valid_meta.validate().is_ok());

        // Invalid magic should fail
        let mut bad_magic = ArchiveMetadata::new();
        bad_magic.magic = *b"BADMAGIC";
        assert!(matches!(
            bad_magic.validate(),
            Err(ArchiveError::InvalidMagic)
        ));

        // Invalid version should fail
        let mut bad_version = ArchiveMetadata::new();
        bad_version.version = 999;
        assert!(matches!(
            bad_version.validate(),
            Err(ArchiveError::UnsupportedVersion(999, ARCHIVE_VERSION))
        ));
    }

    /// Test metadata stats are updated correctly when adding blocks
    #[test]
    fn test_archive_metadata_stats() {
        let mut meta = ArchiveMetadata::new();

        // Initially empty
        assert_eq!(meta.block_count, 0);
        assert_eq!(meta.total_tx_count, 0);

        // Add blocks out of order to test min/max tracking
        meta.add_block(BlockEntry {
            height: 100,
            block_id: dummy_hash(1),
            tx_count: 3,
            jam_offset: 0,
            jam_size: 100,
        });
        assert_eq!(meta.min_height, 100);
        assert_eq!(meta.max_height, 100);

        meta.add_block(BlockEntry {
            height: 50,
            block_id: dummy_hash(2),
            tx_count: 2,
            jam_offset: 100,
            jam_size: 100,
        });
        assert_eq!(meta.min_height, 50);
        assert_eq!(meta.max_height, 100);

        meta.add_block(BlockEntry {
            height: 200,
            block_id: dummy_hash(3),
            tx_count: 5,
            jam_offset: 200,
            jam_size: 100,
        });
        assert_eq!(meta.min_height, 50);
        assert_eq!(meta.max_height, 200);

        assert_eq!(meta.block_count, 3);
        assert_eq!(meta.total_tx_count, 10); // 3 + 2 + 5
    }

    /// Test block lookup methods
    #[test]
    fn test_archive_block_lookup() {
        let mut meta = ArchiveMetadata::new();

        meta.add_block(BlockEntry {
            height: 0,
            block_id: dummy_hash(100),
            tx_count: 0,
            jam_offset: 0,
            jam_size: 100,
        });
        meta.add_block(BlockEntry {
            height: 5,
            block_id: dummy_hash(105),
            tx_count: 1,
            jam_offset: 100,
            jam_size: 200,
        });
        meta.add_block(BlockEntry {
            height: 10,
            block_id: dummy_hash(110),
            tx_count: 2,
            jam_offset: 300,
            jam_size: 150,
        });

        // Lookup by height
        let block_5 = meta.get_block(5).expect("block 5 should exist");
        assert_eq!(block_5.height, 5);
        assert_eq!(block_5.tx_count, 1);

        // Missing height returns None
        assert!(meta.get_block(3).is_none());
        assert!(meta.get_block(100).is_none());

        // Lookup by index
        let block_idx_1 = meta.get_block_by_index(1).expect("index 1 should exist");
        assert_eq!(block_idx_1.height, 5);

        // Out of bounds returns None
        assert!(meta.get_block_by_index(100).is_none());
    }

    // ==================== ArchiveWriter Tests ====================

    /// Test writing an empty archive
    #[test]
    fn test_archive_writer_empty() {
        let writer = ArchiveWriter::new();

        assert_eq!(writer.block_count(), 0);
        assert_eq!(writer.jam_blob_size(), 0);

        let bytes = writer.to_bytes().expect("should serialize empty archive");

        // Should have at least the metadata length prefix (8 bytes) + metadata
        assert!(bytes.len() >= 8);

        // Verify the metadata length prefix
        let meta_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert!(meta_len > 0, "metadata should have non-zero length");

        // Verify we can parse the metadata
        let meta_bytes = &bytes[8..8 + meta_len as usize];
        let meta = ArchiveMetadata::from_bytes(meta_bytes).expect("should parse metadata");
        assert_eq!(meta.block_count, 0);
    }

    /// Test writing a single block
    #[test]
    fn test_archive_writer_single_block() {
        let mut writer = ArchiveWriter::new();

        // Add a block with some dummy jam bytes
        let jam_bytes = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
        writer
            .add_block(0, dummy_hash(100), 2, &jam_bytes)
            .expect("should add block");

        assert_eq!(writer.block_count(), 1);
        assert_eq!(writer.jam_blob_size(), 8);

        let bytes = writer.to_bytes().expect("should serialize archive");

        // Parse the archive
        let meta_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let meta_bytes = &bytes[8..8 + meta_len];
        let jam_section = &bytes[8 + meta_len..];

        let meta = ArchiveMetadata::from_bytes(meta_bytes).expect("should parse metadata");
        assert_eq!(meta.block_count, 1);
        assert_eq!(meta.blocks[0].height, 0);
        assert_eq!(meta.blocks[0].tx_count, 2);
        assert_eq!(meta.blocks[0].jam_offset, 0);
        assert_eq!(meta.blocks[0].jam_size, 8);

        // Verify jam bytes
        assert_eq!(jam_section, &jam_bytes);
    }

    /// Test writing multiple blocks and verify offsets
    #[test]
    fn test_archive_writer_multiple_blocks() {
        let mut writer = ArchiveWriter::new();

        // Add three blocks with different sizes
        let jam_0 = vec![0x00; 100]; // 100 bytes
        let jam_1 = vec![0x11; 250]; // 250 bytes
        let jam_2 = vec![0x22; 50];  // 50 bytes

        writer.add_block(0, dummy_hash(100), 0, &jam_0).unwrap();
        writer.add_block(1, dummy_hash(101), 3, &jam_1).unwrap();
        writer.add_block(2, dummy_hash(102), 1, &jam_2).unwrap();

        assert_eq!(writer.block_count(), 3);
        assert_eq!(writer.jam_blob_size(), 400); // 100 + 250 + 50

        let bytes = writer.to_bytes().expect("should serialize archive");

        // Parse the archive
        let meta_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let meta_bytes = &bytes[8..8 + meta_len];
        let jam_section = &bytes[8 + meta_len..];

        let meta = ArchiveMetadata::from_bytes(meta_bytes).expect("should parse metadata");

        // Verify block entries have correct offsets
        assert_eq!(meta.blocks[0].jam_offset, 0);
        assert_eq!(meta.blocks[0].jam_size, 100);

        assert_eq!(meta.blocks[1].jam_offset, 100);
        assert_eq!(meta.blocks[1].jam_size, 250);

        assert_eq!(meta.blocks[2].jam_offset, 350);
        assert_eq!(meta.blocks[2].jam_size, 50);

        // Verify we can extract each block's jam bytes using the offsets
        for entry in &meta.blocks {
            let start = entry.jam_offset as usize;
            let end = start + entry.jam_size as usize;
            let block_jam = &jam_section[start..end];

            // Verify content matches what we wrote
            let expected_byte = match entry.height {
                0 => 0x00,
                1 => 0x11,
                2 => 0x22,
                _ => panic!("unexpected height"),
            };
            assert!(block_jam.iter().all(|&b| b == expected_byte));
        }
    }

    /// Test that archive file can be written and structure is valid
    #[test]
    fn test_archive_writer_file_structure() {
        let mut writer = ArchiveWriter::with_source(dummy_hash(99999));

        // Add some blocks
        for i in 0u64..5 {
            let jam = vec![i as u8; (i as usize + 1) * 100];
            writer.add_block(i, dummy_hash(i), i as usize, &jam).unwrap();
        }

        // Write to a temp file
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_archive.solar");

        writer.write_to_file(&temp_path).expect("should write file");

        // Read the file back
        let file_bytes = std::fs::read(&temp_path).expect("should read file");

        // Clean up
        std::fs::remove_file(&temp_path).ok();

        // Parse and verify
        let meta_len = u64::from_le_bytes(file_bytes[0..8].try_into().unwrap()) as usize;
        let meta_bytes = &file_bytes[8..8 + meta_len];
        let jam_section = &file_bytes[8 + meta_len..];

        let meta = ArchiveMetadata::from_bytes(meta_bytes).expect("should parse metadata");

        assert_eq!(meta.block_count, 5);
        assert_eq!(meta.min_height, 0);
        assert_eq!(meta.max_height, 4);
        assert_eq!(meta.total_tx_count, 0 + 1 + 2 + 3 + 4);
        assert!(meta.source_checkpoint_hash.is_some());

        // Verify total jam size: 100 + 200 + 300 + 400 + 500 = 1500
        let expected_jam_size: u64 = (1..=5).map(|i| i * 100).sum();
        assert_eq!(jam_section.len() as u64, expected_jam_size);
    }

    // ==================== ArchiveReader Tests ====================

    /// Test parsing an archive from bytes
    #[test]
    fn test_archive_reader_from_bytes() {
        // Create an archive with the writer
        let mut writer = ArchiveWriter::new();
        writer.add_block(0, dummy_hash(100), 0, &[0x00; 50]).unwrap();
        writer.add_block(1, dummy_hash(101), 2, &[0x11; 100]).unwrap();
        writer.add_block(2, dummy_hash(102), 1, &[0x22; 75]).unwrap();

        let bytes = writer.to_bytes().unwrap();

        // Parse with reader
        let reader = ArchiveReader::from_bytes(bytes).expect("should parse archive");

        assert_eq!(reader.block_count(), 3);
        assert_eq!(reader.total_tx_count(), 3);
        assert_eq!(reader.min_height(), 0);
        assert_eq!(reader.max_height(), 2);
    }

    /// Test getting jam bytes by height
    #[test]
    fn test_archive_reader_get_jam_by_height() {
        let mut writer = ArchiveWriter::new();

        // Add blocks with distinctive content
        let jam_0 = vec![0xAA; 100];
        let jam_5 = vec![0xBB; 200];
        let jam_10 = vec![0xCC; 150];

        writer.add_block(0, dummy_hash(100), 0, &jam_0).unwrap();
        writer.add_block(5, dummy_hash(105), 1, &jam_5).unwrap();
        writer.add_block(10, dummy_hash(110), 2, &jam_10).unwrap();

        let bytes = writer.to_bytes().unwrap();
        let reader = ArchiveReader::from_bytes(bytes).unwrap();

        // Get jam by height
        let retrieved_0 = reader.get_jam_by_height(0).expect("block 0 should exist");
        assert_eq!(retrieved_0.len(), 100);
        assert!(retrieved_0.iter().all(|&b| b == 0xAA));

        let retrieved_5 = reader.get_jam_by_height(5).expect("block 5 should exist");
        assert_eq!(retrieved_5.len(), 200);
        assert!(retrieved_5.iter().all(|&b| b == 0xBB));

        let retrieved_10 = reader.get_jam_by_height(10).expect("block 10 should exist");
        assert_eq!(retrieved_10.len(), 150);
        assert!(retrieved_10.iter().all(|&b| b == 0xCC));

        // Missing height should error
        assert!(matches!(
            reader.get_jam_by_height(3),
            Err(ArchiveError::BlockNotFound(3))
        ));
    }

    /// Test getting jam bytes by index
    #[test]
    fn test_archive_reader_get_jam_by_index() {
        let mut writer = ArchiveWriter::new();

        writer.add_block(100, dummy_hash(100), 0, &[0x11; 50]).unwrap();
        writer.add_block(200, dummy_hash(200), 0, &[0x22; 60]).unwrap();
        writer.add_block(300, dummy_hash(300), 0, &[0x33; 70]).unwrap();

        let bytes = writer.to_bytes().unwrap();
        let reader = ArchiveReader::from_bytes(bytes).unwrap();

        // Index 0 = first block added (height 100)
        let (entry_0, jam_0) = (
            reader.get_entry_by_index(0).unwrap(),
            reader.get_jam_by_index(0).unwrap(),
        );
        assert_eq!(entry_0.height, 100);
        assert_eq!(jam_0.len(), 50);

        // Index 1 = second block added (height 200)
        let (entry_1, jam_1) = (
            reader.get_entry_by_index(1).unwrap(),
            reader.get_jam_by_index(1).unwrap(),
        );
        assert_eq!(entry_1.height, 200);
        assert_eq!(jam_1.len(), 60);

        // Index 2 = third block added (height 300)
        let (entry_2, jam_2) = (
            reader.get_entry_by_index(2).unwrap(),
            reader.get_jam_by_index(2).unwrap(),
        );
        assert_eq!(entry_2.height, 300);
        assert_eq!(jam_2.len(), 70);

        // Out of bounds
        assert!(reader.get_jam_by_index(99).is_err());
    }

    /// Test full roundtrip: write with Writer, read with Reader
    #[test]
    fn test_archive_reader_roundtrip() {
        let mut writer = ArchiveWriter::with_source(dummy_hash(99999));

        // Add blocks with varied content
        let test_data: Vec<(u64, Vec<u8>)> = vec![
            (0, (0..100).collect()),
            (1, (100..250).cycle().take(150).collect()),
            (5, vec![0xFF; 200]),
            (10, vec![0x42; 50]),
        ];

        for (height, jam) in &test_data {
            writer.add_block(*height, dummy_hash(*height), *height as usize, jam).unwrap();
        }

        // Write to bytes and read back
        let bytes = writer.to_bytes().unwrap();
        let reader = ArchiveReader::from_bytes(bytes).unwrap();

        // Verify metadata
        assert_eq!(reader.block_count(), 4);
        assert_eq!(reader.min_height(), 0);
        assert_eq!(reader.max_height(), 10);
        assert!(reader.metadata().source_checkpoint_hash.is_some());

        // Verify each block's content matches
        for (height, expected_jam) in &test_data {
            let actual_jam = reader.get_jam_by_height(*height).unwrap();
            assert_eq!(actual_jam, expected_jam.as_slice(), "jam mismatch at height {}", height);
        }
    }

    /// Test iterating through all blocks
    #[test]
    fn test_archive_reader_iterate() {
        let mut writer = ArchiveWriter::new();

        // Add blocks (note: not in height order to test iteration order)
        writer.add_block(5, dummy_hash(5), 1, &[0x55; 50]).unwrap();
        writer.add_block(2, dummy_hash(2), 0, &[0x22; 20]).unwrap();
        writer.add_block(8, dummy_hash(8), 3, &[0x88; 80]).unwrap();

        let bytes = writer.to_bytes().unwrap();
        let reader = ArchiveReader::from_bytes(bytes).unwrap();

        // Iterate - should be in insertion order (5, 2, 8)
        let entries: Vec<_> = reader.iter().collect();
        assert_eq!(entries.len(), 3);

        assert_eq!(entries[0].0.height, 5);
        assert_eq!(entries[0].1.len(), 50);

        assert_eq!(entries[1].0.height, 2);
        assert_eq!(entries[1].1.len(), 20);

        assert_eq!(entries[2].0.height, 8);
        assert_eq!(entries[2].1.len(), 80);

        // Test ExactSizeIterator
        assert_eq!(reader.iter().len(), 3);
    }

    /// Test iterating over a height range
    #[test]
    fn test_archive_reader_iterate_range() {
        let mut writer = ArchiveWriter::new();

        // Add blocks 0, 2, 4, 6, 8, 10 (even numbers only)
        for i in (0..=10).step_by(2) {
            writer.add_block(i, dummy_hash(i), 0, &[i as u8; 10]).unwrap();
        }

        let bytes = writer.to_bytes().unwrap();
        let reader = ArchiveReader::from_bytes(bytes).unwrap();

        // Range 3..=7 should only yield 4 and 6 (since odd numbers don't exist)
        let range_entries: Vec<_> = reader.iter_range(3, 7).collect();
        assert_eq!(range_entries.len(), 2);
        assert_eq!(range_entries[0].0.height, 4);
        assert_eq!(range_entries[1].0.height, 6);

        // Range 0..=4 should yield 0, 2, 4
        let range_entries: Vec<_> = reader.iter_range(0, 4).collect();
        assert_eq!(range_entries.len(), 3);
        assert_eq!(range_entries[0].0.height, 0);
        assert_eq!(range_entries[1].0.height, 2);
        assert_eq!(range_entries[2].0.height, 4);
    }
}
