use std::path::{Path, PathBuf};

use nockchain_bench::speed_of_light::{
    read_fixture_file, ArchiveReader, ProofVersion, SolHeight, PROOF_VERSION_1_START,
    PROOF_VERSION_2_START,
};

const V0_FIXTURE_FILE: &str = "v0-100.soltest";
const V1_FIXTURE_FILE: &str = "v1-100.soltest";
const V2_FIXTURE_FILE: &str = "v2-100.soltest";
const EXPECTED_BLOCK_COUNT: u64 = 100;

const ENV_FIXTURE_DIR: &str = "NOCKCHAIN_SOL_FIXTURES_100_DIR";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate layout should be crates/nockchain-bench")
        .to_path_buf()
}

fn fixture_dir_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(dir) = std::env::var_os(ENV_FIXTURE_DIR) {
        candidates.push(PathBuf::from(dir));
    }

    let root = repo_root();
    candidates.push(root.join("bench-artifacts").join("fixtures-legacy"));
    candidates.push(PathBuf::from(
        "/shared/nockchain-ext4-bench/artifacts/fixtures-legacy",
    ));

    candidates
}

fn fixture_paths_exist(dir: &Path) -> bool {
    dir.join(V0_FIXTURE_FILE).is_file()
        && dir.join(V1_FIXTURE_FILE).is_file()
        && dir.join(V2_FIXTURE_FILE).is_file()
}

fn resolve_fixture_dir() -> PathBuf {
    fixture_dir_candidates()
        .into_iter()
        .find(|dir| fixture_paths_exist(dir))
        .unwrap_or_else(|| {
            panic!(
                "Unable to locate 100-block SOL fixtures.\n\
                 Expected files: {}, {}, {}.\n\
                 Set {} to the fixture directory if needed.",
                V0_FIXTURE_FILE, V1_FIXTURE_FILE, V2_FIXTURE_FILE, ENV_FIXTURE_DIR
            )
        })
}

fn assert_fixture_range_and_versions(
    fixture_path: &Path,
    expected_start: u64,
    expected_end: u64,
    expected_proof: ProofVersion,
) {
    let fixture = read_fixture_file(fixture_path).unwrap_or_else(|err| {
        panic!(
            "failed to read fixture {}: {err}",
            fixture_path.display()
        )
    });

    assert_eq!(
        fixture.manifest.archive_start_height,
        SolHeight(expected_start),
        "manifest archive_start_height mismatch for {}",
        fixture_path.display()
    );
    assert_eq!(
        fixture.manifest.archive_end_height,
        SolHeight(expected_end),
        "manifest archive_end_height mismatch for {}",
        fixture_path.display()
    );
    assert_eq!(
        fixture.manifest.derived_checkpoint_height,
        SolHeight(expected_start.saturating_sub(1)),
        "manifest derived_checkpoint_height mismatch for {}",
        fixture_path.display()
    );

    let archive = ArchiveReader::from_bytes(fixture.archive_bytes).unwrap_or_else(|err| {
        panic!(
            "failed to parse archive bytes from fixture {}: {err}",
            fixture_path.display()
        )
    });

    assert_eq!(
        archive.block_count(),
        EXPECTED_BLOCK_COUNT,
        "archive block_count mismatch for {}",
        fixture_path.display()
    );
    assert_eq!(
        archive.min_height(),
        SolHeight(expected_start),
        "archive min_height mismatch for {}",
        fixture_path.display()
    );
    assert_eq!(
        archive.max_height(),
        SolHeight(expected_end),
        "archive max_height mismatch for {}",
        fixture_path.display()
    );

    let first = archive.get_entry_by_index(0).unwrap_or_else(|| {
        panic!(
            "archive missing first entry for {}",
            fixture_path.display()
        )
    });
    let last = archive
        .get_entry_by_index((EXPECTED_BLOCK_COUNT - 1) as usize)
        .unwrap_or_else(|| {
            panic!(
                "archive missing last entry for {}",
                fixture_path.display()
            )
        });

    assert_eq!(
        first.proof_version, expected_proof,
        "first entry proof_version mismatch for {}",
        fixture_path.display()
    );
    assert_eq!(
        last.proof_version, expected_proof,
        "last entry proof_version mismatch for {}",
        fixture_path.display()
    );
}

#[test]
fn proof_cutover_constants_match_expected_mainnet_values() {
    assert_eq!(PROOF_VERSION_1_START, 6_750);
    assert_eq!(PROOF_VERSION_2_START, 12_000);
    assert!(PROOF_VERSION_1_START < PROOF_VERSION_2_START);
}

#[test]
fn v0_v1_v2_100_block_fixtures_start_at_proof_cutovers() {
    let dir = resolve_fixture_dir();

    assert_fixture_range_and_versions(
        &dir.join(V0_FIXTURE_FILE),
        1,
        100,
        ProofVersion::V0,
    );
    assert_fixture_range_and_versions(
        &dir.join(V1_FIXTURE_FILE),
        PROOF_VERSION_1_START,
        PROOF_VERSION_1_START + EXPECTED_BLOCK_COUNT - 1,
        ProofVersion::V1,
    );
    assert_fixture_range_and_versions(
        &dir.join(V2_FIXTURE_FILE),
        PROOF_VERSION_2_START,
        PROOF_VERSION_2_START + EXPECTED_BLOCK_COUNT - 1,
        ProofVersion::V2,
    );
}
