//! Helpers for building pokes from archived block entries.

use bytes::Bytes;
use nockapp::noun::slab::NounSlab;
use nockvm::noun::{Noun, NounAllocator, D, T};

use super::compat::{NounCompatExt, NounSlabCompatExt};

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
    let tail_cell = tail.as_cell().map_err(|_| "tail not a cell".to_string())?;

    // [page txs]
    let page_txs = tail_cell.tail();
    let page_txs_cell = page_txs
        .as_cell()
        .map_err(|_| "page_txs not a cell".to_string())?;

    Ok(page_txs_cell.head().noun())
}

/// Construct a poke cause: [%fact 0 [%heard-block page]]
pub fn make_heard_block_cause(page: Noun, slab: &mut NounSlab) -> Noun {
    let fact_tag = nockapp::utils::make_tas(slab, "fact").as_noun();
    let heard_block_tag = nockapp::utils::make_tas(slab, "heard-block").as_noun();

    let heard_block = T(slab, &[heard_block_tag, page]);
    // dumbnet fact payload now requires explicit version=%0.
    T(slab, &[fact_tag, D(0), heard_block])
}

/// Build a poke slab from jammed block-entry bytes.
///
/// This cues the entry noun, extracts the page, and builds the [%fact 0 [%heard-block page]] cause.
pub fn build_poke_slab_from_jam(jam_bytes: &[u8]) -> Result<NounSlab, String> {
    let mut entry_slab = NounSlab::new();
    let entry_noun = entry_slab
        .cue_into(Bytes::copy_from_slice(jam_bytes))
        .map_err(|e| format!("cue failed: {e:?}"))?;

    let page = extract_page_from_entry(entry_noun, &entry_slab)
        .map_err(|e| format!("extract page failed: {e}"))?;

    let mut poke_slab = NounSlab::new();
    let space = entry_slab.noun_space();
    let page_copy = poke_slab.copy_into(page);
    let cause = make_heard_block_cause(page_copy, &mut poke_slab);
    poke_slab.set_root(cause);

    Ok(poke_slab)
}

#[cfg(test)]
mod tests {
    use nockapp::noun::slab::NockJammer;

    use super::*;

    #[test]
    fn test_make_heard_block_cause_includes_fact_version_zero() {
        let mut slab: NounSlab<NockJammer> = NounSlab::new();
        let page = T(&mut slab, &[D(11), D(22)]);
        let cause = make_heard_block_cause(page, &mut slab);
        slab.set_root(cause);

        let root = unsafe { slab.root() };
        let space = slab.noun_space();
        let root_cell = root
            .in_space(&space)
            .as_cell()
            .expect("cause must be a cell");
        assert!(root_cell.head().eq_bytes(b"fact"));

        let fact_payload = root_cell
            .tail()
            .as_cell()
            .expect("fact payload must be a cell");
        let version = fact_payload
            .head()
            .as_atom()
            .expect("version must be an atom")
            .as_u64()
            .expect("version must fit in u64");
        assert_eq!(version, 0);

        let data = fact_payload.tail().noun();
        let data_cell = data
            .in_space(&space)
            .as_cell()
            .expect("fact data must be a cell");
        assert!(data_cell.head().eq_bytes(b"heard-block"));
    }

    #[test]
    fn test_build_poke_slab_from_jam_emits_versioned_fact() {
        let mut entry_slab: NounSlab<NockJammer> = NounSlab::new();
        let page = T(&mut entry_slab, &[D(1), D(2), D(3)]);
        let page_txs = T(&mut entry_slab, &[page, D(0)]);
        let tail = T(&mut entry_slab, &[D(42), page_txs]);
        let entry = T(&mut entry_slab, &[D(7), tail]);
        entry_slab.set_root(entry);
        let jammed = entry_slab.jam();

        let poke_slab = build_poke_slab_from_jam(jammed.as_ref()).expect("should build poke slab");
        let root = unsafe { poke_slab.root() };
        let space = poke_slab.noun_space();
        let root_cell = root
            .in_space(&space)
            .as_cell()
            .expect("cause must be a cell");
        assert!(root_cell.head().eq_bytes(b"fact"));

        let fact_payload = root_cell
            .tail()
            .as_cell()
            .expect("fact payload must be a cell");
        let version = fact_payload
            .head()
            .as_atom()
            .expect("version must be an atom")
            .as_u64()
            .expect("version must fit in u64");
        assert_eq!(version, 0);

        let data = fact_payload.tail().noun();
        let data_cell = data
            .in_space(&space)
            .as_cell()
            .expect("fact data must be a cell");
        assert!(data_cell.head().eq_bytes(b"heard-block"));
    }
}
