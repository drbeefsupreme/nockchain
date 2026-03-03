# Phase 3 Provenance And Divergence Timeline

## Metadata

- phase: `03-provenance-and-divergence-timeline`
- artifact_role: `phase-03-canonical-provenance-and-timeline`
- canonical_target_ref: `refs/remotes/upstream/master`
- pinned_master_sha: `cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c`
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

## Unresolved Provenance

Use this section for rows where evidence is conflicting or insufficient. Unresolved rows must still populate all required schema fields and include explicit rationale for why attribution is not yet finalized.

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
