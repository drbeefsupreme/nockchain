//! Helpers for building pokes from archived block entries.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

use bytes::Bytes;
use nockapp::nockapp::wire::WireRepr;
use nockapp::nockapp::{NockApp, NockAppError};
use nockapp::noun::slab::NounSlab;
use nockvm::noun::{Noun, D, T};
use thiserror::Error;

use super::archive::{ArchiveError, BlockEntryV4, SolArchiveReader};
use super::{noun_compat, pma_replay};

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

/// Construct a poke cause: [%fact 0 [%heard-tx raw-tx]]
pub fn make_heard_tx_cause(raw_tx: Noun, slab: &mut NounSlab) -> Noun {
    let fact_tag = nockapp::utils::make_tas(slab, "fact").as_noun();
    let heard_tx_tag = nockapp::utils::make_tas(slab, "heard-tx").as_noun();

    let heard_tx = T(slab, &[heard_tx_tag, raw_tx]);
    T(slab, &[fact_tag, D(0), heard_tx])
}

fn cue_jam_into_slab(slab: &mut NounSlab, jam_bytes: &[u8]) -> Result<Noun, String> {
    match catch_unwind(AssertUnwindSafe(|| {
        slab.cue_into(Bytes::copy_from_slice(jam_bytes))
    })) {
        Ok(Ok(noun)) => Ok(noun),
        Ok(Err(error)) => Err(format!("cue failed: {error:?}")),
        Err(_) => Err("cue panicked".to_string()),
    }
}

/// Build a poke slab from jammed block-entry bytes.
///
/// This cues the entry noun, extracts the page, and builds the [%fact 0 [%heard-block page]] cause.
pub fn build_poke_slab_from_jam(jam_bytes: &[u8]) -> Result<NounSlab, String> {
    let mut entry_slab: NounSlab = NounSlab::new();
    let entry_noun = cue_jam_into_slab(&mut entry_slab, jam_bytes)?;

    let entry_space = noun_compat::space_for_slab(&entry_slab);
    let page = extract_page_from_entry(entry_noun, &entry_space)
        .map_err(|e| format!("extract page failed: {e}"))?;

    let mut poke_slab = NounSlab::new();
    let page_copy = pma_replay::copy_from_source_slab(&mut poke_slab, page, &entry_slab);
    let cause = make_heard_block_cause(page_copy, &mut poke_slab);
    poke_slab.set_root(cause);

    Ok(poke_slab)
}

/// Build a poke slab from jammed raw transaction bytes.
pub fn build_heard_tx_poke_slab_from_jam(jam_bytes: &[u8]) -> Result<NounSlab, String> {
    let mut raw_tx_slab: NounSlab = NounSlab::new();
    let raw_tx = cue_jam_into_slab(&mut raw_tx_slab, jam_bytes)?;

    let mut poke_slab = NounSlab::new();
    let raw_tx_copy = pma_replay::copy_from_source_slab(&mut poke_slab, raw_tx, &raw_tx_slab);
    let cause = make_heard_tx_cause(raw_tx_copy, &mut poke_slab);
    poke_slab.set_root(cause);

    Ok(poke_slab)
}

#[derive(Debug, Error)]
pub enum PokeStepError {
    #[error("failed to build poke slab: {0}")]
    Build(String),

    #[error("failed to poke block: {0}")]
    Poke(#[from] NockAppError),

    #[error("archive error: {0}")]
    Archive(#[from] ArchiveError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchivePokeTimings {
    pub block_duration: Duration,
    pub raw_tx_duration: Duration,
    pub total_duration: Duration,
    pub raw_tx_pokes_completed: u64,
}

struct ArchiveBlockPokeSlabs {
    block: NounSlab,
    raw_txs: Vec<NounSlab>,
}

fn build_archive_block_poke_slabs(
    reader: &SolArchiveReader,
    entry: &BlockEntryV4,
) -> Result<ArchiveBlockPokeSlabs, PokeStepError> {
    let body = reader.as_v4().ok_or(ArchiveError::UnsupportedOperation {
        version: super::archive::ARCHIVE_VERSION_V3,
        operation: "poke_archive_block",
    })?;

    let block =
        build_poke_slab_from_jam(body.get_jam_for_entry(entry)?).map_err(PokeStepError::Build)?;
    let mut raw_txs = Vec::new();

    for raw_tx_entry in body.raw_tx_entries_for_block(entry)? {
        let payload = body.get_raw_tx_payload(raw_tx_entry)?;
        raw_txs.push(build_heard_tx_poke_slab_from_jam(payload).map_err(PokeStepError::Build)?);
    }

    Ok(ArchiveBlockPokeSlabs { block, raw_txs })
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

pub async fn poke_archive_block(
    nockapp: &mut NockApp,
    wire: WireRepr,
    reader: &SolArchiveReader,
    entry: &BlockEntryV4,
) -> Result<ArchivePokeTimings, PokeStepError> {
    let total_started_at = Instant::now();
    let ArchiveBlockPokeSlabs { block, raw_txs } = build_archive_block_poke_slabs(reader, entry)?;

    let block_started_at = Instant::now();
    nockapp.poke(wire.clone(), block).await?;
    let block_duration = block_started_at.elapsed();
    let raw_tx_started_at = Instant::now();
    let mut raw_tx_pokes_completed = 0u64;

    for poke_slab in raw_txs {
        nockapp.poke(wire.clone(), poke_slab).await?;
        raw_tx_pokes_completed = raw_tx_pokes_completed.saturating_add(1);
    }

    Ok(ArchivePokeTimings {
        block_duration,
        raw_tx_duration: raw_tx_started_at.elapsed(),
        total_duration: total_started_at.elapsed(),
        raw_tx_pokes_completed,
    })
}

#[cfg(test)]
mod tests {
    use nockapp::noun::slab::NockJammer;
    use nockchain_math::belt::Belt;
    use nockchain_types::tx_engine::common::Hash;

    use super::super::noun_compat;
    use super::*;
    use crate::speed_of_light::archive::{RawTxPayload, SolArchiveReader, SolArchiveWriterV4};
    use crate::speed_of_light::types::{ProofVersion, SolHeight};

    fn dummy_hash(v: u64) -> Hash {
        Hash([Belt(v), Belt(v + 1), Belt(v + 2), Belt(v + 3), Belt(v + 4)])
    }

    fn block_entry_jam(height: u64) -> Vec<u8> {
        let mut slab: NounSlab<NockJammer> = NounSlab::new();
        let page = T(&mut slab, &[D(1), D(2), D(3)]);
        let page_txs = T(&mut slab, &[page, D(0)]);
        let tail = T(&mut slab, &[D(height), page_txs]);
        let entry = T(&mut slab, &[D(height), tail]);
        slab.set_root(entry);
        slab.jam().as_ref().to_vec()
    }

    fn assert_versioned_fact_cause<J>(slab: &NounSlab<J>, expected_payload_tag: &str) {
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
        let payload_tag_noun = noun_compat::noun_head(data_noun, &space).expect("payload tag");
        let payload_tag = noun_compat::decode_with_space::<String>(&payload_tag_noun, &space)
            .expect("payload tag");
        assert_eq!(payload_tag, expected_payload_tag);
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
            pma_replay::copy_from_source_slab(&mut extracted_slab, extracted, &entry_slab);
        extracted_slab.set_root(extracted_copy);

        let mut expected_slab: NounSlab<NockJammer> = NounSlab::new();
        let expected_copy =
            pma_replay::copy_from_source_slab(&mut expected_slab, page, &entry_slab);
        expected_slab.set_root(expected_copy);

        assert_eq!(extracted_slab.jam().as_ref(), expected_slab.jam().as_ref());
    }

    #[test]
    fn test_make_heard_block_cause_includes_fact_version_zero() {
        let mut slab: NounSlab<NockJammer> = NounSlab::new();
        let page = T(&mut slab, &[D(11), D(22)]);
        let cause = make_heard_block_cause(page, &mut slab);
        slab.set_root(cause);

        assert_versioned_fact_cause(&slab, "heard-block");
    }

    #[test]
    fn test_make_heard_tx_cause_includes_fact_version_zero() {
        let mut slab: NounSlab<NockJammer> = NounSlab::new();
        let raw_tx = T(&mut slab, &[D(1), D(22)]);
        let cause = make_heard_tx_cause(raw_tx, &mut slab);
        slab.set_root(cause);

        assert_versioned_fact_cause(&slab, "heard-tx");
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
        assert_versioned_fact_cause(&poke_slab, "heard-block");
    }

    #[test]
    fn test_build_heard_tx_poke_slab_from_jam_emits_versioned_fact() {
        let mut raw_tx_slab: NounSlab<NockJammer> = NounSlab::new();
        let raw_tx = T(&mut raw_tx_slab, &[D(1), D(22)]);
        raw_tx_slab.set_root(raw_tx);
        let jammed = raw_tx_slab.jam();

        let poke_slab =
            build_heard_tx_poke_slab_from_jam(jammed.as_ref()).expect("should build poke slab");
        assert_versioned_fact_cause(&poke_slab, "heard-tx");
    }

    #[test]
    fn test_build_poke_slab_from_jam_rejects_invalid_jam_bytes() {
        let error = build_poke_slab_from_jam(b"not-a-jam").expect_err("invalid jam should fail");
        assert!(
            error.contains("cue failed") || error.contains("extract page failed"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_build_archive_block_poke_slabs_rejects_bad_raw_tx_before_poke() {
        let mut writer = SolArchiveWriterV4::new();
        writer
            .add_block_with_raw_txs(
                SolHeight(7),
                dummy_hash(700),
                ProofVersion::V0,
                &block_entry_jam(7),
                [RawTxPayload {
                    tx_id: dummy_hash(701),
                    jam_bytes: &[],
                }],
            )
            .expect("v4 block should be accepted");
        let reader = SolArchiveReader::from_bytes(writer.to_bytes().expect("serialize"))
            .expect("read archive");
        let body = reader.as_v4().expect("v4 body");
        let entry = body.get_entry_by_height(SolHeight(7)).expect("block entry");

        let error = match build_archive_block_poke_slabs(&reader, entry) {
            Ok(_) => panic!("bad raw tx jam should fail before any poke"),
            Err(error) => error,
        };
        assert!(matches!(error, PokeStepError::Build(_)));
    }
}
