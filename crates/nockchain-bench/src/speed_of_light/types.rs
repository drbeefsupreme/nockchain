//! Types for speed-of-light benchmark data

use bytes::Bytes;
use nockchain_math::noun_ext::NounMathExt;
use nockchain_types::tx_engine::common::Hash;
use nockchain_types::tx_engine::v0::{Lock, NoteV0, RawTx};
use nockvm::noun::Noun;
use noun_serde::{NounDecode, NounDecodeError};
use serde::{Deserialize, Serialize};

use super::compat::{HoonMapIterCompatExt, NounCompatExt, NounSpace};

/// Height wrapper for speed-of-light data structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SolHeight(pub u64);

impl SolHeight {
    pub const ZERO: SolHeight = SolHeight(0);
    pub const MAX: SolHeight = SolHeight(u64::MAX);

    pub fn as_u64(self) -> u64 {
        self.0
    }

    pub fn saturating_add(self, rhs: u64) -> SolHeight {
        SolHeight(self.0.saturating_add(rhs))
    }

    pub fn saturating_sub(self, rhs: u64) -> SolHeight {
        SolHeight(self.0.saturating_sub(rhs))
    }
}

impl From<u64> for SolHeight {
    fn from(value: u64) -> Self {
        SolHeight(value)
    }
}

impl From<SolHeight> for u64 {
    fn from(value: SolHeight) -> Self {
        value.0
    }
}

impl std::fmt::Display for SolHeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for SolHeight {
    fn default() -> Self {
        SolHeight::ZERO
    }
}

/// Proof version cutover heights (from hoon/apps/dumbnet/lib/consensus.hoon)
pub const PROOF_VERSION_1_START: u64 = 6_750;
pub const PROOF_VERSION_2_START: u64 = 12_000;

/// Proof version for a block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ProofVersion {
    V0 = 0,
    V1 = 1,
    V2 = 2,
}

impl ProofVersion {
    /// Map a block height to its proof version using consensus cutovers.
    pub fn for_height(height: SolHeight) -> Self {
        if height.0 >= PROOF_VERSION_2_START {
            ProofVersion::V2
        } else if height.0 >= PROOF_VERSION_1_START {
            ProofVersion::V1
        } else {
            ProofVersion::V0
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ProofVersion::V0 => "v0",
            ProofVersion::V1 => "v1",
            ProofVersion::V2 => "v2",
        }
    }
}

impl std::fmt::Display for ProofVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Metadata for a block (without full transaction data)
#[derive(Debug, Clone)]
pub struct BlockMetadata {
    pub height: SolHeight,
    pub block_id: Hash,
    pub parent_id: Hash,
    pub timestamp: u64,
    pub tx_ids: Vec<Hash>,
}

/// Full block data including transactions
#[derive(Debug, Clone)]
pub struct BlockData {
    pub height: SolHeight,
    pub block_id: Hash,
    pub parent_id: Hash,
    pub timestamp: u64,
    pub transactions: Vec<TransactionData>,
}

/// Block data with raw jammed noun bytes for archiving
///
/// This struct combines the decoded BlockData with the raw jammed noun bytes
/// that can be used to reconstruct the original noun without decoding loss.
#[derive(Debug, Clone)]
pub struct BlockDataWithJam {
    /// Decoded block data for easy access
    pub data: BlockData,
    /// Raw jammed noun bytes for the block entry
    pub jam_bytes: Bytes,
}

/// Minimal archive-safe summary of a block entry noun.
///
/// This intentionally avoids decoding page/transaction payload shapes, so it
/// remains valid across proof/page/tx encoding versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArchiveBlockSummary {
    pub height: SolHeight,
    pub block_id: Hash,
    pub tx_count: usize,
    pub proof_version: ProofVersion,
}

impl BlockData {
    pub fn tx_count(&self) -> usize {
        self.transactions.len()
    }

    pub fn proof_version(&self) -> ProofVersion {
        ProofVersion::for_height(self.height)
    }
}

/// Transaction data extracted from a block
#[derive(Debug, Clone)]
pub struct TransactionData {
    pub tx_id: Hash,
    pub version: u64,
    pub raw_tx: RawTx,
    pub total_size: u64,
    pub outputs: Vec<TxOutput>,
}

/// Transaction output (lock + note)
#[derive(Debug, Clone)]
pub struct TxOutput {
    pub lock: Lock,
    pub note: NoteV0,
}

// --- Internal decoding types (matching block_explorer.rs) ---

use nockchain_math::structs::HoonMapIter;
use nockchain_types::tx_engine::common::BlockHeight;

/// Raw decoded block range entry from peek
#[derive(Debug, Clone, NounDecode)]
pub(crate) struct BlockRangeEntryNoun {
    pub height: BlockHeight,
    pub tail: BlockRangeEntryTail,
}

#[derive(Debug, Clone, NounDecode)]
pub(crate) struct BlockRangeEntryTail {
    pub block_id: Hash,
    pub tail: PageAndTxs,
}

#[derive(Debug, Clone, NounDecode)]
pub(crate) struct PageAndTxs {
    pub page: PageNoun,
    pub txs: Noun,
}

#[derive(Debug, Clone, NounDecode)]
pub(crate) struct PageNoun {
    pub _digest: Hash,
    pub _pow: Noun,
    pub parent: Hash,
    pub _tx_ids: Noun,
    pub _coinbase: Noun,
    pub timestamp: Noun,
    pub _epoch_counter: Noun,
    pub _target: Noun,
    pub _accumulated_work: Noun,
    pub _height: BlockHeight,
    pub _msg: Noun,
}

impl BlockRangeEntryNoun {
    /// Convert raw noun structure to BlockData with full transaction data
    pub fn into_block_data(self, space: &NounSpace) -> Result<BlockData, NounDecodeError> {
        let BlockRangeEntryNoun { height, tail } = self;
        let BlockRangeEntryTail { block_id, tail } = tail;
        let PageAndTxs { page, txs } = tail;

        let parent_id = page.parent;
        let timestamp = u64::from_noun(&page.timestamp)?;
        let transactions = extract_transactions_from_map(&txs, space)?;

        Ok(BlockData {
            height: SolHeight(height.0 .0),
            block_id,
            parent_id,
            timestamp,
            transactions,
        })
    }

    /// Convert raw noun structure to BlockMetadata (tx IDs only, no full data)
    pub fn into_metadata(self, space: &NounSpace) -> Result<BlockMetadata, NounDecodeError> {
        let BlockRangeEntryNoun { height, tail } = self;
        let BlockRangeEntryTail { block_id, tail } = tail;
        let PageAndTxs { page, txs } = tail;

        let parent_id = page.parent;
        let timestamp = u64::from_noun(&page.timestamp)?;
        let tx_ids = extract_tx_ids_from_map(&txs, space)?;

        Ok(BlockMetadata {
            height: SolHeight(height.0 .0),
            block_id,
            parent_id,
            timestamp,
            tx_ids,
        })
    }
}

/// Decode only the stable outer structure of a block-range entry noun:
/// `[height [block-id [page txs]]]`.
///
/// This intentionally does not decode `page` or tx value payloads.
pub(crate) fn summarize_archive_entry(
    entry_noun: Noun,
    space: &NounSpace,
) -> Result<ArchiveBlockSummary, NounDecodeError> {
    let entry_cell = entry_noun
        .in_space(space)
        .as_cell()
        .map_err(|_| NounDecodeError::ExpectedCell)?;

    let height = BlockHeight::from_noun(&entry_cell.head().noun())?;

    let tail_cell = entry_cell
        .tail()
        .as_cell()
        .map_err(|_| NounDecodeError::ExpectedCell)?;
    let block_id = Hash::from_noun(&tail_cell.head().noun())?;

    let page_txs_cell = tail_cell
        .tail()
        .as_cell()
        .map_err(|_| NounDecodeError::ExpectedCell)?;
    let txs_noun = page_txs_cell.tail().noun();
    let tx_count = tx_map_len(&txs_noun, space)?;

    let height = SolHeight(height.0 .0);
    Ok(ArchiveBlockSummary {
        height,
        block_id,
        tx_count,
        proof_version: ProofVersion::for_height(height),
    })
}

fn tx_map_len(txs_noun: &Noun, space: &NounSpace) -> Result<usize, NounDecodeError> {
    if let Ok(atom) = txs_noun.in_space(space).as_atom() {
        if atom.as_u64()? == 0 {
            return Ok(0);
        }
        return Err(NounDecodeError::ExpectedCell);
    }

    Ok(HoonMapIter::new(*txs_noun, space)
        .filter(|entry| entry.is_cell())
        .count())
}

/// Extract just transaction IDs from the txs z-map
fn extract_tx_ids_from_map(
    txs_noun: &Noun,
    space: &NounSpace,
) -> Result<Vec<Hash>, NounDecodeError> {
    if let Ok(atom) = txs_noun.in_space(space).as_atom() {
        if atom.as_u64()? == 0 {
            return Ok(Vec::new());
        }
    }

    let tx_ids: Vec<Hash> = HoonMapIter::new(*txs_noun, space)
        .filter(|entry| entry.is_cell())
        .filter_map(|entry| {
            let [key, _value] = entry.uncell().ok()?;
            Hash::from_noun(&key).ok()
        })
        .collect();

    Ok(tx_ids)
}

/// Extract full transaction data from the txs z-map
fn extract_transactions_from_map(
    txs_noun: &Noun,
    space: &NounSpace,
) -> Result<Vec<TransactionData>, NounDecodeError> {
    if let Ok(atom) = txs_noun.in_space(space).as_atom() {
        if atom.as_u64()? == 0 {
            return Ok(Vec::new());
        }
    }

    let mut txs = Vec::new();
    for entry in HoonMapIter::new(*txs_noun, space) {
        if !entry.is_cell() {
            continue;
        }
        let [key, value] = entry
            .uncell()
            .map_err(|_| NounDecodeError::ExpectedCell)?;
        let tx_id = Hash::from_noun(&key)?;
        let tx = TxV0Internal::from_noun(&value)?;
        txs.push(TransactionData {
            tx_id,
            version: tx.version,
            raw_tx: tx.raw_tx,
            total_size: tx.total_size,
            outputs: tx.outputs,
        });
    }

    Ok(txs)
}

/// Internal transaction decoding (matches block_explorer.rs TxV0)
struct TxV0Internal {
    version: u64,
    raw_tx: RawTx,
    total_size: u64,
    outputs: Vec<TxOutput>,
}

impl NounDecode for TxV0Internal {
    fn from_noun(noun: &Noun) -> Result<Self, NounDecodeError> {
        let space = ();
        let cell = noun.in_space(&space).as_cell()?;
        let version_noun = cell.head().noun();
        let version = u64::from_noun(&version_noun)?;

        let tail = cell.tail();
        let cell = tail.as_cell()?;
        let raw_tx_noun = cell.head().noun();
        let raw_tx = RawTx::from_noun(&raw_tx_noun)?;

        let tail = cell.tail();
        let cell = tail.as_cell()?;
        let total_noun = cell.head().noun();
        let total_size = u64::from_noun(&total_noun)?;
        let outputs_noun = cell.tail().noun();
        let outputs = decode_outputs(&outputs_noun, &space)?;

        Ok(Self {
            version,
            raw_tx,
            total_size,
            outputs,
        })
    }
}

fn decode_outputs(noun: &Noun, space: &NounSpace) -> Result<Vec<TxOutput>, NounDecodeError> {
    if let Ok(atom) = noun.in_space(space).as_atom() {
        if atom.as_u64()? == 0 {
            return Ok(Vec::new());
        }
    }

    let mut outputs = Vec::new();
    for entry in HoonMapIter::new(*noun, space) {
        if !entry.is_cell() {
            continue;
        }
        let [key, value] = entry
            .uncell()
            .map_err(|_| NounDecodeError::ExpectedCell)?;
        let lock = Lock::from_noun(&key)?;
        let value_cell = value
            .in_space(space)
            .as_cell()
            .map_err(|_| NounDecodeError::ExpectedCell)?;
        let note_noun = value_cell.head().noun();
        let note = NoteV0::from_noun(&note_noun)?;
        outputs.push(TxOutput { lock, note });
    }

    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use nockapp::noun::slab::NounSlab;
    use nockchain_math::belt::Belt;
    use nockchain_math::zoon::common::DefaultTipHasher;
    use nockchain_math::zoon::zmap;
    use nockvm::noun::{D, T};
    use noun_serde::NounEncode;

    use super::*;

    fn dummy_hash(v: u64) -> Hash {
        Hash([Belt(v), Belt(v + 1), Belt(v + 2), Belt(v + 3), Belt(v + 4)])
    }

    fn dummy_block_data(height: u64) -> BlockData {
        BlockData {
            height: SolHeight(height),
            block_id: dummy_hash(height),
            parent_id: dummy_hash(height.saturating_sub(1)),
            timestamp: 1234567890 + height,
            transactions: vec![],
        }
    }

    #[test]
    fn test_proof_version_for_height_boundaries() {
        assert_eq!(ProofVersion::for_height(SolHeight(0)), ProofVersion::V0);
        assert_eq!(
            ProofVersion::for_height(SolHeight(PROOF_VERSION_1_START - 1)),
            ProofVersion::V0
        );
        assert_eq!(
            ProofVersion::for_height(SolHeight(PROOF_VERSION_1_START)),
            ProofVersion::V1
        );
        assert_eq!(
            ProofVersion::for_height(SolHeight(PROOF_VERSION_2_START - 1)),
            ProofVersion::V1
        );
        assert_eq!(
            ProofVersion::for_height(SolHeight(PROOF_VERSION_2_START)),
            ProofVersion::V2
        );
    }

    #[test]
    fn test_block_data_with_jam_creation() {
        let data = dummy_block_data(42);
        let jam_bytes = Bytes::from(vec![1, 2, 3, 4, 5]);

        let block_with_jam = BlockDataWithJam {
            data: data.clone(),
            jam_bytes: jam_bytes.clone(),
        };

        assert_eq!(block_with_jam.data.height, SolHeight(42));
        assert_eq!(block_with_jam.jam_bytes.len(), 5);
        assert_eq!(block_with_jam.jam_bytes[0], 1);
        assert_eq!(block_with_jam.jam_bytes[4], 5);
    }

    #[test]
    fn test_block_data_with_jam_clone() {
        let data = dummy_block_data(100);
        let jam_bytes = Bytes::from(vec![0xDE, 0xAD, 0xBE, 0xEF]);

        let original = BlockDataWithJam { data, jam_bytes };
        let cloned = original.clone();

        assert_eq!(original.data.height, cloned.data.height);
        assert_eq!(original.jam_bytes, cloned.jam_bytes);
    }

    fn make_txs_map_with_atom_values(slab: &mut NounSlab, values: &[u64]) -> Noun {
        let mut map = D(0);
        for (idx, value) in values.iter().enumerate() {
            let mut key = dummy_hash(10_000 + idx as u64).to_noun(slab);
            let mut val = D(*value);
            map = zmap::z_map_put(slab, &map, &mut key, &mut val, &DefaultTipHasher)
                .expect("z-map put should succeed");
        }
        map
    }

    fn make_page_v0(slab: &mut NounSlab, parent: Hash, height: u64) -> Noun {
        let digest = dummy_hash(height + 1_000).to_noun(slab);
        let parent = parent.to_noun(slab);
        T(
            slab,
            &[
                digest,
                D(0),
                parent,
                D(0),
                D(0),
                D(123_456),
                D(0),
                D(0),
                D(0),
                D(height),
                D(0),
            ],
        )
    }

    fn make_page_v1(slab: &mut NounSlab, parent: Hash, height: u64) -> Noun {
        let digest = dummy_hash(height + 2_000).to_noun(slab);
        let parent = parent.to_noun(slab);
        T(
            slab,
            &[
                D(1),
                digest,
                D(0),
                parent,
                D(0),
                D(0),
                D(123_456),
                D(0),
                D(0),
                D(0),
                D(height),
                D(0),
            ],
        )
    }

    fn make_entry(
        slab: &mut NounSlab,
        height: u64,
        block_id: Hash,
        page: Noun,
        txs: Noun,
    ) -> Noun {
        let block_id_noun = block_id.to_noun(slab);
        let page_txs = T(slab, &[page, txs]);
        let tail = T(slab, &[block_id_noun, page_txs]);
        T(slab, &[D(height), tail])
    }

    #[test]
    fn test_summarize_archive_entry_v0_page_counts_txs_without_value_decode() {
        let mut slab = NounSlab::new();
        let block_id = dummy_hash(7_777);
        let parent = dummy_hash(7_776);
        let txs = make_txs_map_with_atom_values(&mut slab, &[9, 8, 7]);
        let page = make_page_v0(&mut slab, parent, 42);
        let entry = make_entry(
            &mut slab,
            42,
            block_id.clone(),
            page,
            txs,
        );
        let space = ();

        let summary = summarize_archive_entry(entry, &space).expect("summary decode should succeed");
        assert_eq!(summary.height, SolHeight(42));
        assert_eq!(summary.block_id, block_id);
        assert_eq!(summary.tx_count, 3);
        assert_eq!(summary.proof_version, ProofVersion::V0);
    }

    #[test]
    fn test_summarize_archive_entry_v1_page_is_version_agnostic() {
        let mut slab = NounSlab::new();
        let height = PROOF_VERSION_1_START + 10;
        let block_id = dummy_hash(8_888);
        let parent = dummy_hash(8_887);
        let txs = make_txs_map_with_atom_values(&mut slab, &[1, 2]);
        let page = make_page_v1(&mut slab, parent, height);
        let entry = make_entry(
            &mut slab,
            height,
            block_id.clone(),
            page,
            txs,
        );
        let space = ();

        let summary = summarize_archive_entry(entry, &space).expect("summary decode should succeed");
        assert_eq!(summary.height, SolHeight(height));
        assert_eq!(summary.block_id, block_id);
        assert_eq!(summary.tx_count, 2);
        assert_eq!(summary.proof_version, ProofVersion::V1);
    }
}
