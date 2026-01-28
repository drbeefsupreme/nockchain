//! Types for speed-of-light benchmark data

use nockchain_math::noun_ext::NounMathExt;
use nockchain_types::tx_engine::common::Hash;
use nockchain_types::tx_engine::v0::{Lock, NoteV0, RawTx};
use nockvm::noun::{Noun, NounSpace};
use noun_serde::{NounDecode, NounDecodeError};

/// Metadata for a block (without full transaction data)
#[derive(Debug, Clone)]
pub struct BlockMetadata {
    pub height: u64,
    pub block_id: Hash,
    pub parent_id: Hash,
    pub timestamp: u64,
    pub tx_ids: Vec<Hash>,
}

/// Full block data including transactions
#[derive(Debug, Clone)]
pub struct BlockData {
    pub height: u64,
    pub block_id: Hash,
    pub parent_id: Hash,
    pub timestamp: u64,
    pub transactions: Vec<TransactionData>,
}

impl BlockData {
    pub fn tx_count(&self) -> usize {
        self.transactions.len()
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
        let timestamp = u64::from_noun(&page.timestamp, space)?;
        let transactions = extract_transactions_from_map(&txs, space)?;

        Ok(BlockData {
            height: height.0 .0,
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
        let timestamp = u64::from_noun(&page.timestamp, space)?;
        let tx_ids = extract_tx_ids_from_map(&txs, space)?;

        Ok(BlockMetadata {
            height: height.0 .0,
            block_id,
            parent_id,
            timestamp,
            tx_ids,
        })
    }
}

/// Extract just transaction IDs from the txs z-map
fn extract_tx_ids_from_map(txs_noun: &Noun, space: &NounSpace) -> Result<Vec<Hash>, NounDecodeError> {
    if let Ok(atom) = txs_noun.in_space(space).as_atom() {
        if atom.as_u64()? == 0 {
            return Ok(Vec::new());
        }
    }

    let tx_ids: Vec<Hash> = HoonMapIter::new(*txs_noun, space)
        .filter(|entry| entry.is_cell())
        .filter_map(|entry| {
            let [key, _value] = entry.uncell(space).ok()?;
            Hash::from_noun(&key, space).ok()
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
        let [key, value] = entry.uncell(space).map_err(|_| NounDecodeError::ExpectedCell)?;
        let tx_id = Hash::from_noun(&key, space)?;
        let tx = TxV0Internal::from_noun(&value, space)?;
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
    fn from_noun(noun: &Noun, space: &NounSpace) -> Result<Self, NounDecodeError> {
        let cell = noun.in_space(space).as_cell()?;
        let version_noun = cell.head().noun();
        let version = u64::from_noun(&version_noun, space)?;

        let tail = cell.tail();
        let cell = tail.as_cell()?;
        let raw_tx_noun = cell.head().noun();
        let raw_tx = RawTx::from_noun(&raw_tx_noun, space)?;

        let tail = cell.tail();
        let cell = tail.as_cell()?;
        let total_noun = cell.head().noun();
        let total_size = u64::from_noun(&total_noun, space)?;
        let outputs_noun = cell.tail().noun();
        let outputs = decode_outputs(&outputs_noun, space)?;

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
        let [key, value] = entry.uncell(space).map_err(|_| NounDecodeError::ExpectedCell)?;
        let lock = Lock::from_noun(&key, space)?;
        let value_cell = value
            .in_space(space)
            .as_cell()
            .map_err(|_| NounDecodeError::ExpectedCell)?;
        let note_noun = value_cell.head().noun();
        let note = NoteV0::from_noun(&note_noun, space)?;
        outputs.push(TxOutput { lock, note });
    }

    Ok(outputs)
}
