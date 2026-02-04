//! Helpers for building pokes from archived block entries.

use nockapp::noun::slab::NounSlab;
use nockvm::noun::{Noun, NounAllocator, T};
use bytes::Bytes;

/// Extract the page noun from a block entry noun.
///
/// Block entry structure: [height [block_id [page txs]]]
pub fn extract_page_from_entry(entry_noun: Noun, slab: &NounSlab) -> Result<Noun, String> {
    let space = slab.noun_space();

    // entry = [height tail]
    let entry_cell = entry_noun
        .in_space(&space)
        .as_cell()
        .map_err(|_| "entry not a cell".to_string())?;

    // tail = [block_id [page txs]]
    let tail = entry_cell.tail();
    let tail_cell = tail
        .as_cell()
        .map_err(|_| "tail not a cell".to_string())?;

    // [page txs]
    let page_txs = tail_cell.tail();
    let page_txs_cell = page_txs
        .as_cell()
        .map_err(|_| "page_txs not a cell".to_string())?;

    Ok(page_txs_cell.head().noun())
}

/// Construct a poke cause: [%fact [%heard-block page]]
pub fn make_heard_block_cause(page: Noun, slab: &mut NounSlab) -> Noun {
    let fact_tag = nockapp::utils::make_tas(slab, "fact").as_noun();
    let heard_block_tag = nockapp::utils::make_tas(slab, "heard-block").as_noun();

    let heard_block = T(slab, &[heard_block_tag, page]);
    T(slab, &[fact_tag, heard_block])
}

/// Build a poke slab from jammed block-entry bytes.
///
/// This cues the entry noun, extracts the page, and builds the [%fact [%heard-block page]] cause.
pub fn build_poke_slab_from_jam(jam_bytes: &[u8]) -> Result<NounSlab, String> {
    let mut entry_slab = NounSlab::new();
    let entry_noun = entry_slab
        .cue_into(Bytes::copy_from_slice(jam_bytes))
        .map_err(|e| format!("cue failed: {e:?}"))?;

    let page = extract_page_from_entry(entry_noun, &entry_slab)
        .map_err(|e| format!("extract page failed: {e}"))?;

    let mut poke_slab = NounSlab::new();
    let space = entry_slab.noun_space();
    let page_copy = poke_slab.copy_into(page, &space);
    let cause = make_heard_block_cause(page_copy, &mut poke_slab);
    poke_slab.set_root(cause);

    Ok(poke_slab)
}
