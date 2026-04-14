//! Helpers for building pokes from archived block entries.

use std::time::{Duration, Instant};

use bytes::Bytes;
use nockapp::nockapp::wire::WireRepr;
use nockapp::nockapp::{NockApp, NockAppError};
use nockapp::noun::slab::NounSlab;
use nockvm::noun::{Noun, D, T};
use thiserror::Error;

use super::{noun_compat, runtime_compat};

/// Extract the page noun from a block entry noun.
///
/// Block entry structure: [height [block_id [page txs]]]
pub(crate) fn extract_page_from_entry(
    entry_noun: Noun,
    space: &noun_compat::NounSpace,
) -> Result<Noun, String> {
    let tail =
        noun_compat::noun_tail(entry_noun, space).map_err(|_| "tail not a cell".to_string())?;
    let page_txs =
        noun_compat::noun_tail(tail, space).map_err(|_| "page_txs not a cell".to_string())?;
    noun_compat::noun_head(page_txs, space).map_err(|_| "page_txs not a cell".to_string())
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
    let mut entry_slab: NounSlab = NounSlab::new();
    let entry_noun = entry_slab
        .cue_into(Bytes::copy_from_slice(jam_bytes))
        .map_err(|e| format!("cue failed: {e:?}"))?;

    let entry_space = noun_compat::space_for_slab(&entry_slab);
    let page = extract_page_from_entry(entry_noun, &entry_space)
        .map_err(|e| format!("extract page failed: {e}"))?;

    let mut poke_slab = NounSlab::new();
    let page_copy = runtime_compat::copy_from_source_slab(&mut poke_slab, page, &entry_slab);
    let cause = make_heard_block_cause(page_copy, &mut poke_slab);
    poke_slab.set_root(cause);

    Ok(poke_slab)
}

#[derive(Debug, Error)]
pub enum PokeStepError {
    #[error("failed to build poke slab: {0}")]
    Build(String),

    #[error("failed to poke block: {0}")]
    Poke(#[from] NockAppError),
}

pub async fn poke_block_from_jam(
    nockapp: &mut NockApp,
    wire: WireRepr,
    jam_bytes: &[u8],
) -> Result<Duration, PokeStepError> {
    let started_at = Instant::now();
    let poke_slab = build_poke_slab_from_jam(jam_bytes).map_err(PokeStepError::Build)?;
    nockapp.poke(wire, poke_slab).await?;
    Ok(started_at.elapsed())
}

#[cfg(test)]
mod tests {
    use nockapp::noun::slab::NockJammer;

    use super::super::noun_compat;
    use super::*;

    fn assert_versioned_fact_cause<J>(slab: &NounSlab<J>) {
        let space = noun_compat::space_for_slab(slab);
        let root = unsafe { slab.root() };

        let fact_tag_noun = noun_compat::noun_head(*root, &space).expect("cause tag");
        let fact_tag =
            noun_compat::decode_with_space::<String>(&fact_tag_noun, &space).expect("fact tag");
        assert_eq!(fact_tag, "fact");

        let fact_payload = noun_compat::noun_tail(*root, &space).expect("fact payload");
        let version_noun = noun_compat::noun_head(fact_payload, &space).expect("version");
        let version =
            noun_compat::decode_with_space::<u64>(&version_noun, &space).expect("fact version");
        assert_eq!(version, 0);

        let data_noun = noun_compat::noun_tail(fact_payload, &space).expect("fact data");
        let heard_block_tag_noun =
            noun_compat::noun_head(data_noun, &space).expect("heard-block tag");
        let heard_block_tag =
            noun_compat::decode_with_space::<String>(&heard_block_tag_noun, &space)
                .expect("heard-block tag");
        assert_eq!(heard_block_tag, "heard-block");
    }

    #[test]
    fn test_extract_page_from_entry_reads_page_from_entry_shape() {
        let mut entry_slab: NounSlab<NockJammer> = NounSlab::new();
        let page = T(&mut entry_slab, &[D(1), D(2), D(3)]);
        let page_txs = T(&mut entry_slab, &[page, D(0)]);
        let tail = T(&mut entry_slab, &[D(42), page_txs]);
        let entry = T(&mut entry_slab, &[D(7), tail]);

        let entry_space = noun_compat::space_for_slab(&entry_slab);
        let extracted =
            extract_page_from_entry(entry, &entry_space).expect("page extraction should succeed");

        let mut extracted_slab: NounSlab<NockJammer> = NounSlab::new();
        let extracted_copy =
            runtime_compat::copy_from_source_slab(&mut extracted_slab, extracted, &entry_slab);
        extracted_slab.set_root(extracted_copy);

        let mut expected_slab: NounSlab<NockJammer> = NounSlab::new();
        let expected_copy =
            runtime_compat::copy_from_source_slab(&mut expected_slab, page, &entry_slab);
        expected_slab.set_root(expected_copy);

        assert_eq!(extracted_slab.jam().as_ref(), expected_slab.jam().as_ref());
    }

    #[test]
    fn test_make_heard_block_cause_includes_fact_version_zero() {
        let mut slab: NounSlab<NockJammer> = NounSlab::new();
        let page = T(&mut slab, &[D(11), D(22)]);
        let cause = make_heard_block_cause(page, &mut slab);
        slab.set_root(cause);

        assert_versioned_fact_cause(&slab);
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
        assert_versioned_fact_cause(&poke_slab);
    }

    #[test]
    fn test_build_poke_slab_from_jam_rejects_invalid_jam_bytes() {
        let error = build_poke_slab_from_jam(b"not-a-jam").expect_err("invalid jam should fail");
        assert!(
            error.contains("cue failed") || error.contains("extract page failed"),
            "unexpected error: {error}"
        );
    }
}
