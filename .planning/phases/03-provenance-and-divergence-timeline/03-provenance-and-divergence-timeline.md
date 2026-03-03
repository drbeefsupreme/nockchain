# Phase 3 Provenance And Divergence Timeline

## Metadata

- phase: `03-provenance-and-divergence-timeline`
- artifact_role: `phase-03-canonical-provenance-and-timeline`
- canonical_target_ref: `refs/remotes/upstream/master`
- pinned_master_sha: `cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c`
- merge_base_master_current: `cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c`
- merge_base_historical_current: `c5b13b6bf1808eb28b5a5019f9d73907fc02dee2`
- branch_geometry:
  - `upstream/master...nockchain-bench-master-candidate = 0/114`
  - `upstream/master...upstream/bitemyapp/ag2-opt-persistence-madvise-checkpoint-chaff-pma-gc-checkpoint-streaming = 17/121`
  - `upstream/bitemyapp/ag2-opt-persistence-madvise-checkpoint-chaff-pma-gc-checkpoint-streaming...nockchain-bench-master-candidate = 121/131`
- branch_horizon:
  - `upstream/master`
  - `upstream/bitemyapp/ag2-opt-persistence-madvise-checkpoint-chaff-pma-gc-checkpoint-streaming`
  - `nockchain-bench-master-candidate`

## Purpose

Define the canonical schema for provenance attribution and divergence chronology before population. This artifact locks attribution taxonomy, required evidence fields, unresolved handling, and timeline structure for all Phase 3 records.

## Attribution Taxonomy

Allowed classification values are locked to:

- `Inherited` - evidence ties origin to historical branch ancestry.
- `Local` - evidence indicates introduction in current-branch-only evolution.
- `Mixed` - evidence shows multiple distinct origins across horizons.

Any other classification value is invalid.

## Evidence Requirements

Every provenance record must include:

- Dependency linkage (`dependency_id`, `finding_ids`)
- Commit evidence (`origin_commit_sha`, `pivot_commit_shas`)
- Branch-horizon evidence (`historical_branch_evidence`, `current_branch_evidence`)
- Decision rationale and certainty (`rationale`, `confidence`, `status`, `notes`)

`confidence` is locked to `high|medium|low`.

## Dependency Provenance Records

| dependency_id | finding_ids | classification | origin_commit_sha | pivot_commit_shas | historical_branch_evidence | current_branch_evidence | rationale | confidence | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEP-001 | C001,C002 | TBD | `77cf156c170865a4b823871a6481c7c54babb81b` | `26710e534f5058f8be3cd89be7522337c679f72f` | `git log --reverse -S '--pma-persist' c5b13b6..497b016 -> 77cf156` | `git log --reverse -S '--pma-persist' c5b13b6..37397c4 -- crates/nockchain-bench -> 26710e5` | Evidence captured across historical + current horizons; classification pending Task 2. | low | unresolved | dependency_id=`DEP-001`; finding_ids=`C001,C002` |
| DEP-002 | C003 | TBD | `26710e534f5058f8be3cd89be7522337c679f72f` | `26710e534f5058f8be3cd89be7522337c679f72f` | No `-S 'NOCK_PMA_CANDIDATE'` hit in `c5b13b6..497b016`. | `git log --reverse -S 'NOCK_PMA_CANDIDATE' c5b13b6..37397c4 -- crates/nockchain-bench -> 26710e5` | Historical horizon lacks this env symbol; current bench origin captured. | low | unresolved | dependency_id=`DEP-002`; finding_ids=`C003` |
| DEP-003 | C004 | TBD | `26710e534f5058f8be3cd89be7522337c679f72f` | `26710e534f5058f8be3cd89be7522337c679f72f` | No `-S 'NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE'` hit in `c5b13b6..497b016`. | `git log --reverse -S 'NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE' c5b13b6..37397c4 -- crates/nockchain-bench -> 26710e5` | Stream checkpoint chunk env currently only evidenced on current bench lineage. | low | unresolved | dependency_id=`DEP-003`; finding_ids=`C004` |
| DEP-004 | C005,C006,C007,C008,C014,C015 | TBD | `e6cc94db771c0f3ea400559c6d67e0a2738da47d` | `26710e534f5058f8be3cd89be7522337c679f72f,73838f25a6872e1f577e577699e368892d783987` | `git log --reverse -S 'NounSpace' c5b13b6..497b016 -> 6b5ce44`; `-S 'result.noun_space()' -> e6cc94d` | `git log --reverse -S 'NounSpace' c5b13b6..37397c4 -- crates/nockchain-bench -> 26710e5,73838f2` | NounSpace concept and noun_space access appear in historical line and current bench graft/pivot. | low | unresolved | dependency_id=`DEP-004`; finding_ids=`C005,C006,C007,C008,C014,C015` |
| DEP-005 | C009 | TBD | `26710e534f5058f8be3cd89be7522337c679f72f` | `26710e534f5058f8be3cd89be7522337c679f72f` | `raw-transactions` token exists on historical/master hoon paths but no bench-path introduction signal in historical range. | `git log --reverse -S 'raw-transactions' c5b13b6..37397c4 -- crates/nockchain-bench -> 26710e5` | Token-level ancestry is cross-domain; semantic bench origin still needs adjudication. | low | unresolved | dependency_id=`DEP-005`; finding_ids=`C009` |
| DEP-006 | C011 | TBD | `d7a0e874194950549edb4a9dcd4bde38229606c0` | `26710e534f5058f8be3cd89be7522337c679f72f` | `git log --reverse -S '--data-dir' c5b13b6..497b016 -> d7a0e87,d46f296` | `git log --reverse -S '--data-dir' c5b13b6..37397c4 -- crates/nockchain-bench -> 26710e5` | Data-dir flag appears in historical and current lines with differing runtime certainty. | low | unresolved | dependency_id=`DEP-006`; finding_ids=`C011` |
| DEP-007 | C012 | TBD | `77cf156c170865a4b823871a6481c7c54babb81b` | `26710e534f5058f8be3cd89be7522337c679f72f` | `git log --reverse -S '--save-interval' c5b13b6..497b016 -> 949523c,77cf156` | `git log --reverse -S '--save-interval' c5b13b6..37397c4 -- crates/nockchain-bench -> 26710e5` | Save-interval controls were present before bench graft and then carried into current bench wiring. | low | unresolved | dependency_id=`DEP-007`; finding_ids=`C012` |
| DEP-008 | C013 | TBD | `c413814608fc3304f276977864d18059b27f3cdb` | `26710e534f5058f8be3cd89be7522337c679f72f` | `--new` exists on historical branch and upstream/master ancestry; historical range includes `77cf156`. | `git log --reverse -S '--new' c5b13b6..37397c4 -- crates/nockchain-bench -> 26710e5` | Bootstrap-new semantics predate bench graft and remain in current runner behavior. | low | unresolved | dependency_id=`DEP-008`; finding_ids=`C013` |
| DEP-009 | PMA-S001,PMA-S002 | TBD | `26710e534f5058f8be3cd89be7522337c679f72f` | `26710e534f5058f8be3cd89be7522337c679f72f` | No `-S 'is_pma_path'` or `-S 'MemoryBucket::Pma'` hit in `c5b13b6..497b016`. | `git log --reverse -S 'is_pma_path'/-S 'MemoryBucket::Pma' c5b13b6..37397c4 -- crates/nockchain-bench -> 26710e5` | Sampler PMA heuristics currently appear bench-local after graft introduction. | low | unresolved | dependency_id=`DEP-009`; finding_ids=`PMA-S001,PMA-S002` |
| DEP-ctl-001 | C010 | TBD | `1722a9827a16be4134310d5db7ba58e797693d68` | `26710e534f5058f8be3cd89be7522337c679f72f,73838f25a6872e1f577e577699e368892d783987` | `heaviest-chain-blocks-range` exists at historical ref `497b016` (`nockapp-grpc` block_explorer). | `git log --reverse -S 'heaviest-chain-blocks-range' c5b13b6..37397c4 -- crates/nockchain-bench -> 26710e5,73838f2` | Positive-control signal propagated into bench extraction path. | low | unresolved | dependency_id=`DEP-ctl-001`; finding_ids=`C010` |

## Unresolved Provenance

Task 1 intentionally leaves all records unresolved while collecting commit and branch evidence. Task 2 applies attribution taxonomy and escalates only truly ambiguous rows.

| dependency_id | finding_ids | classification | origin_commit_sha | pivot_commit_shas | historical_branch_evidence | current_branch_evidence | rationale | confidence | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Divergence Timeline (Thematic Buckets)

Timeline entries are grouped by theme and ordered chronologically within each bucket.

### Bucket: Graft Introduction And Early Compatibility Surface

| event_date | commit_sha | dependency_ids | finding_ids | classification_impact | summary | evidence |
| --- | --- | --- | --- | --- | --- | --- |

### Bucket: PMA/Checkpoint Persistence Divergence

| event_date | commit_sha | dependency_ids | finding_ids | classification_impact | summary | evidence |
| --- | --- | --- | --- | --- | --- | --- |

### Bucket: NounSpace/Adapter Divergence

| event_date | commit_sha | dependency_ids | finding_ids | classification_impact | summary | evidence |
| --- | --- | --- | --- | --- | --- | --- |

### Bucket: Current-Branch Local Mutations

| event_date | commit_sha | dependency_ids | finding_ids | classification_impact | summary | evidence |
| --- | --- | --- | --- | --- | --- | --- |
