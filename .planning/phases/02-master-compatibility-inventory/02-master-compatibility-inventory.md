# Phase 2 Master Compatibility Inventory

## Metadata

- canonical_target_ref: `refs/remotes/upstream/master`
- pinned_master_sha: `cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c`
- generated_at_utc: `2026-03-03T21:05:00Z`
- artifact_role: `phase-02-canonical-inventory`

## Purpose

This is the canonical Phase 2 artifact for compatibility findings against pinned master. It defines one normalized schema and taxonomy for runtime-path incompatibilities, test-only incompatibilities, and positive controls.

## Locked Disposition Taxonomy

Allowed `disposition` values are locked to:

- `remove`
- `replace-with-master-equivalent`
- `feature-gate`
- `defer`

Any other value is invalid.

## Hybrid Entry Model

Each dependency is represented as one primary incompatibility row (`dependency_id`) and can link multiple code references (`finding_id`) that share the same underlying gap. This prevents duplicate dependency rows while preserving callsite evidence.

## Required Schema

Every inventory row must include these columns:

`dependency_id`, `finding_id`, `file_path`, `symbol_or_api`, `branch_context`, `master_evidence`, `impact_statement`, `confidence`, `match_rule`, `impact_level`, `disposition`, `disposition_rationale`, `tags`, `status`, `notes`

### Canonical Inventory Table

| dependency_id | finding_id | file_path | symbol_or_api | branch_context | master_evidence | impact_statement | confidence | match_rule | impact_level | disposition | disposition_rationale | tags | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dep-001 | C001 | crates/nockchain-bench/src/runner/docker.rs:174 | --pma-persist | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- '--pma-persist' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | pma persist mode flag has no pinned-master runtime flag match. | high | branch_env_config_toggle | medium |remove| PMA persist mode is optional branch-only behavior, so remove by default per locked Phase 2 rules. | cli-flag\|runtime\|persist-mode | open | master_presence=missing |
| dep-001 | C002 | crates/nockchain-bench/src/runner/docker.rs:220 | NOCK_PMA_PERSIST | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'NOCK_PMA_PERSIST' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | pma runtime env toggle is branch-only in bench wiring. | high | branch_env_config_toggle | medium | remove | PMA env toggle only enables optional persist behavior and should be removed for master graft baseline. | env-toggle\|runtime\|persist-mode | open | master_presence=missing |
| dep-002 | C003 | crates/nockchain-bench/src/main.rs:437 | NOCK_PMA_CANDIDATE | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'NOCK_PMA_CANDIDATE' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Sweep CLI default references pma-only candidate selection env var absent from master. | high | branch_env_config_toggle | low | remove | Candidate-selection env is optional PMA experiment wiring, not required for master-compatible benchmark baseline. | env-toggle\|sweep\|candidate-selection | open | master_presence=missing |
| dep-003 | C004 | crates/nockchain-bench/src/main.rs:441 | NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Sweep path depends on pma-centric streaming checkpoint env tuning. | high | branch_env_config_toggle | low | remove | Streaming checkpoint chunk tuning is PMA-centric optional behavior and defaults to remove under locked disposition rules. | checkpoint\|env-toggle\|sweep | open | master_presence=missing |
| dep-009 | PMA-S001 | crates/nockchain-bench/src/sampler/smaps.rs:133 | is_pma_path(path) | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'is_pma_path' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Sampler parser hard-codes pma path heuristics (`/pma/*.mmap`) that have no pinned-master equivalent contract. | medium | branch_env_config_toggle | low | remove | PMA path heuristics are optional branch-only observability logic and should be removed for master-compat baseline. | sampler\|path-heuristic\|runtime | open | master_presence=missing;non-candidate |
| dep-009 | PMA-S002 | crates/nockchain-bench/src/sampler/buckets.rs:14 | MemoryBucket::Pma | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'MemoryBucket::Pma' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Memory attribution pipeline includes pma-only bucket accounting not required by master benchmark runtime contracts. | medium | branch_env_config_toggle | low | remove | PMA bucket accounting is optional branch instrumentation and defaults to remove under PMA remove-bias. | sampler\|memory-bucket\|runtime | open | master_presence=missing;non-candidate |
| dep-004 | C005 | crates/nockchain-bench/src/speed_of_light/compat.rs:5 | NounSpace | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'NounSpace' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Branch compatibility layer introduces NounSpace concept not present in pinned master crates. | high | exact_missing_ref | medium | defer | Initial classification pending branch-only concept pass. | nounspace\|type\|sol | open | master_presence=missing |
| dep-004 | C006 | crates/nockchain-bench/src/speed_of_light/compat.rs:41 | NounCompatExt::in_space | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'NounCompatExt::in_space' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Branch adapter trait injects in-space conversion API not defined in master runtime surfaces. | high | exact_missing_ref | medium | defer | Initial classification pending branch-only concept pass. | nounspace\|adapter\|sol | open | master_presence=missing |
| dep-004 | C007 | crates/nockchain-bench/src/speed_of_light/compat.rs:8 | NounSlabCompatExt::noun_space | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'NounSlabCompatExt::noun_space' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Branch noun slab helper depends on noun-space API absent in master. | high | exact_missing_ref | medium | defer | Initial classification pending branch-only concept pass. | nounspace\|adapter\|sol | open | master_presence=missing |
| dep-004 | C008 | crates/nockchain-bench/src/speed_of_light/compat.rs:1 | speed_of_light::compat | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'speed_of_light::compat' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Dedicated compat module is branch-only glue with no master module counterpart. | high | exact_missing_ref | medium | defer | Initial classification pending branch-only concept pass. | nounspace\|module\|adapter | open | master_presence=missing |
| dep-005 | C009 | crates/nockchain-bench/src/speed_of_light/extractor.rs:234 | raw-transactions | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'raw-transactions' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Bench mempool snapshot extraction depends on raw-transactions peek path absent from master surfaces. | high | exact_missing_ref | high | defer | No concrete master equivalent identified yet for this extraction path. | sol\|peek-path\|mempool | open | master_presence=missing |
| dep-006 | C011 | crates/nockchain-bench/src/runner/docker.rs:179 | --data-dir | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `cd91acc3...:crates/hoonc/README.md:36` docs-only mention; no runtime symbol hit under crates | Runner always passes data-dir, but pinned master evidence is documentation-only and not a confirmed runtime contract. | medium | replaceable_gap | medium | defer | Flag may be replaceable, but concrete master CLI contract remains uncertain. | cli-flag\|runtime\|pathing | open | master_presence=uncertain |
| dep-007 | C012 | crates/nockchain-bench/src/runner/docker.rs:168 | --save-interval | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- '--save-interval' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Checkpoint mode in runner expects a save interval flag not found in pinned master crates. | high | exact_missing_ref | high | defer | Non-optional runner path lacks a concrete master flag equivalent so deferred pending graft design. | cli-flag\|runtime\|checkpoint | open | master_presence=missing |
| dep-008 | C013 | crates/nockchain-bench/src/runner/docker.rs:200 | --new | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `cd91acc3...:crates/hoonc/README.md:59` docs-only mention; no runtime symbol hit under crates | Runner bootstrap relies on --new while master evidence currently appears only in docs text. | medium | replaceable_gap | medium | defer | Runtime equivalence is uncertain until master CLI contract is confirmed from code paths. | cli-flag\|runtime\|bootstrap | open | master_presence=uncertain |
| dep-004 | C014 | crates/nockchain-bench/src/speed_of_light/extractor.rs:254 | HoonMapIter::new(map_noun, &space) | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'HoonMapIter::new(map_noun, &space)' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Bench uses compat shim constructor signature not present in pinned-master callsites. | high | replaceable_gap | medium | defer | Initial classification pending branch-only concept pass. | nounspace\|iterator\|sol | open | master_presence=missing |
| dep-004 | C015 | crates/nockchain-bench/src/speed_of_light/extractor.rs:240 | result.noun_space() | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'result.noun_space()' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Bench decoding path depends on noun-space compatibility accessor absent on master result handling. | high | replaceable_gap | medium | defer | Initial classification pending branch-only concept pass. | nounspace\|api-call\|sol | open | master_presence=missing |

## Runtime-Path Incompatibilities

Use this section for dependencies reached by runtime CLI flows, runner wiring, extraction/replay paths, and non-test behavior assumptions.

| dependency_id | finding_id | file_path | symbol_or_api | branch_context | master_evidence | impact_statement | confidence | match_rule | impact_level | disposition | disposition_rationale | tags | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dep-ctl-001 | C010 | crates/nockchain-bench/src/speed_of_light/extractor.rs:314 | heaviest-chain-blocks-range | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c:crates/nockapp-grpc/src/services/public_nockchain/v2/block_explorer.rs:659` | Positive control confirms compatibility process does not over-report this known master path as missing. | high | replaceable_gap | low | replace-with-master-equivalent | Documented present-control dependency retained as non-gap validation row. | sol\|peek-path\|positive-control | control | master_presence=present-control |

## Test-Only Incompatibilities

Use this section only for dependencies reached exclusively by test code (e.g., unit-test-only helpers or `#[cfg(test)]` paths).

| dependency_id | finding_id | file_path | symbol_or_api | branch_context | master_evidence | impact_statement | confidence | match_rule | impact_level | disposition | disposition_rationale | tags | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Positive Controls

Positive controls are references intentionally retained to prove the inventory process does not over-report non-gaps.

| dependency_id | finding_id | file_path | symbol_or_api | branch_context | master_evidence | impact_statement | confidence | match_rule | impact_level | disposition | disposition_rationale | tags | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Linked Reference Map

Use this map to associate one primary dependency row with multiple concrete references.

| dependency_id | finding_id | link_type | reference_note |
| --- | --- | --- | --- |
| dep-001 | C001 | primary | PMA persist runner flag gap |
| dep-001 | C002 | supporting | PMA runtime env toggle linked to same dependency |
| dep-002 | C003 | primary | PMA candidate sweep env default |
| dep-003 | C004 | primary | Streaming checkpoint chunk env default |
| dep-004 | C005 | primary | NounSpace compatibility concept |
| dep-004 | C006 | supporting | in_space adapter trait callsites |
| dep-004 | C007 | supporting | noun_space adapter trait callsites |
| dep-004 | C008 | supporting | branch-only compat module definition |
| dep-004 | C014 | supporting | HoonMapIter compat constructor call |
| dep-004 | C015 | supporting | noun_space accessor on peek results |
| dep-005 | C009 | primary | raw-transactions mempool peek path |
| dep-006 | C011 | primary | data-dir flag runtime contract uncertainty |
| dep-007 | C012 | primary | save-interval flag missing on pinned master |
| dep-008 | C013 | primary | new flag bootstrap uncertainty |
| dep-009 | PMA-S001 | primary | sampler PMA mmap-path heuristic |
| dep-009 | PMA-S002 | supporting | sampler PMA memory-bucket attribution |
| dep-ctl-001 | C010 | control | heaviest-chain-blocks-range present-control |
