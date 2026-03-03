# Phase 2 Master Compatibility Inventory

## Metadata

- canonical_target_ref: `refs/remotes/upstream/master`
- pinned_master_sha: `cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c`
- generated_at_utc: `2026-03-03T21:27:00Z`
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

## Runtime-Path Incompatibilities

Use this section for dependencies reached by runtime CLI flows, runner wiring, extraction/replay paths, and non-test behavior assumptions.

| dependency_id | finding_id | file_path | symbol_or_api | branch_context | master_evidence | impact_statement | confidence | match_rule | impact_level | disposition | disposition_rationale | tags | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEP-001 | C001 | crates/nockchain-bench/src/runner/docker.rs:174 | --pma-persist | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- '--pma-persist' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | pma persist mode flag has no pinned-master runtime flag match. | high | branch_env_config_toggle | medium | remove | PMA persist mode is optional branch-only behavior, so remove by default per locked Phase 2 rules. | cli-flag\|runtime\|persist-mode | open | master_presence=missing |
| DEP-001 | C002 | crates/nockchain-bench/src/runner/docker.rs:220 | NOCK_PMA_PERSIST | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'NOCK_PMA_PERSIST' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | pma runtime env toggle is branch-only in bench wiring. | high | branch_env_config_toggle | medium | remove | PMA env toggle only enables optional persist behavior and should be removed for master graft baseline. | env-toggle\|runtime\|persist-mode | open | master_presence=missing |
| DEP-002 | C003 | crates/nockchain-bench/src/main.rs:437 | NOCK_PMA_CANDIDATE | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'NOCK_PMA_CANDIDATE' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Sweep CLI default references pma-only candidate selection env var absent from master. | high | branch_env_config_toggle | low | remove | Candidate-selection env is optional PMA experiment wiring, not required for master-compatible benchmark baseline. | env-toggle\|sweep\|candidate-selection | open | master_presence=missing |
| DEP-003 | C004 | crates/nockchain-bench/src/main.rs:441 | NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Sweep path depends on pma-centric streaming checkpoint env tuning. | high | branch_env_config_toggle | low | remove | Streaming checkpoint chunk tuning is PMA-centric optional behavior and defaults to remove under locked disposition rules. | checkpoint\|env-toggle\|sweep | open | master_presence=missing |
| DEP-009 | PMA-S001 | crates/nockchain-bench/src/sampler/smaps.rs:133 | is_pma_path(path) | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'is_pma_path' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Sampler parser hard-codes pma path heuristics (`/pma/*.mmap`) that have no pinned-master equivalent contract. | medium | branch_env_config_toggle | low | remove | PMA path heuristics are optional branch-only observability logic and should be removed for master-compat baseline. | sampler\|path-heuristic\|runtime | open | master_presence=missing;non-candidate |
| DEP-009 | PMA-S002 | crates/nockchain-bench/src/sampler/buckets.rs:14 | MemoryBucket::Pma | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'MemoryBucket::Pma' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Memory attribution pipeline includes pma-only bucket accounting not required by master benchmark runtime contracts. | medium | branch_env_config_toggle | low | remove | PMA bucket accounting is optional branch instrumentation and defaults to remove under PMA remove-bias. | sampler\|memory-bucket\|runtime | open | master_presence=missing;non-candidate |
| DEP-004 | C005 | crates/nockchain-bench/src/speed_of_light/compat.rs:5 | NounSpace | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | No `NounSpace` symbol in pinned master (`git grep -n --fixed-strings -- 'NounSpace' ...` no matches), while master decoding and iteration use direct APIs: `block_explorer.rs:678 NounDecode::from_noun`, `block_explorer.rs:1422 HoonMapIter::from`. | Branch compatibility type can be removed by switching to direct master noun decode/iteration APIs. | high | replaceable_gap | medium | replace-with-master-equivalent | Concrete master equivalents exist without NounSpace adapters. | nounspace\|type\|sol | open | master_presence=missing |
| DEP-004 | C006 | crates/nockchain-bench/src/speed_of_light/compat.rs:41 | NounCompatExt::in_space | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | Master callsites decode nouns directly (`block_explorer.rs:678 NounDecode::from_noun`) and iterate via `HoonMapIter::from` (`block_explorer.rs:1422`) without `in_space`. | in_space adapter can be replaced with direct noun decode access patterns present in pinned master. | high | replaceable_gap | medium | replace-with-master-equivalent | Concrete master decode path is evidenced and does not require adapter traits. | nounspace\|adapter\|sol | open | master_presence=missing |
| DEP-004 | C007 | crates/nockchain-bench/src/speed_of_light/compat.rs:8 | NounSlabCompatExt::noun_space | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | Pinned-master decode path uses `NounDecode::from_noun` (`block_explorer.rs:678`) and direct `Noun` iteration (`HoonMapIter::from`, `block_explorer.rs:1422`) with no noun_space accessor. | noun_space compatibility accessor is branch-only shim and has direct master-equivalent decoding flow. | high | replaceable_gap | medium | replace-with-master-equivalent | Replace with master's direct noun decode path to drop shim dependency. | nounspace\|adapter\|sol | open | master_presence=missing |
| DEP-004 | C008 | crates/nockchain-bench/src/speed_of_light/compat.rs:1 | speed_of_light::compat | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | Module symbol absent from pinned master (`git grep -n --fixed-strings -- 'speed_of_light::compat' ...` no matches); equivalent behavior is represented by direct master noun decode APIs in `block_explorer.rs`. | Branch-only compatibility module can be removed when callsites adopt concrete master decode/iteration APIs. | high | replaceable_gap | medium | replace-with-master-equivalent | Concrete master API path exists and removes need for dedicated compat module. | nounspace\|module\|adapter | open | master_presence=missing |
| DEP-005 | C009 | crates/nockchain-bench/src/speed_of_light/extractor.rs:234 | raw-transactions | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `git grep -n --fixed-strings -- 'raw-transactions' cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c -- crates/*` returned no matches | Bench mempool snapshot extraction depends on raw-transactions peek path absent from master surfaces. | high | exact_missing_ref | high | defer | No concrete master equivalent identified yet for this extraction path. | sol\|peek-path\|mempool | open | master_presence=missing |
| DEP-006 | C011 | crates/nockchain-bench/src/runner/docker.rs:179 | --data-dir | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `cd91acc3...:crates/hoonc/README.md:36` docs-only mention; no pinned-master runtime symbol hit under crates. | Runner hard-codes data-dir pathing without concrete master runtime contract evidence. | medium | branch_env_config_toggle | medium | remove | Data-dir override is branch runner pathing glue and can be removed for baseline-compatible master runs. | cli-flag\|runtime\|pathing | open | master_presence=uncertain |
| DEP-007 | C012 | crates/nockchain-bench/src/runner/docker.rs:168 | --save-interval | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | Pinned master exposes save interval on boot CLI (`crates/nockapp/src/kernel/boot.rs:103 pub save_interval`, `#[arg(long)]`), providing equivalent control surface. | Runner checkpoint save interval can map to master boot CLI save-interval support. | high | replaceable_gap | medium | replace-with-master-equivalent | Concrete master boot CLI field exists for save interval semantics. | cli-flag\|runtime\|checkpoint | open | master_presence=uncertain |
| DEP-008 | C013 | crates/nockchain-bench/src/runner/docker.rs:200 | --new | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | Pinned master boot CLI includes `new` switch (`crates/nockapp/src/kernel/boot.rs:88-93` with `#[arg(long)] pub new: bool`). | Runner bootstrap can map --new semantics to concrete pinned-master boot CLI equivalent. | high | replaceable_gap | medium | replace-with-master-equivalent | Concrete master boot path supports new-state initialization. | cli-flag\|runtime\|bootstrap | open | master_presence=uncertain |
| DEP-004 | C014 | crates/nockchain-bench/src/speed_of_light/extractor.rs:254 | HoonMapIter::new(map_noun, &space) | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | Pinned master uses `HoonMapIter::from` directly (`block_explorer.rs:1422`, `1483`, `2428`) instead of compat shim constructor. | HoonMapIter compat constructor call is replaceable with direct master iterator API. | high | replaceable_gap | medium | replace-with-master-equivalent | Concrete iterator equivalent is present in pinned master callsites. | nounspace\|iterator\|sol | open | master_presence=missing |
| DEP-004 | C015 | crates/nockchain-bench/src/speed_of_light/extractor.rs:240 | result.noun_space() | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | Master decode path uses `NounDecode::from_noun` on peek result nouns (`block_explorer.rs:678`, `863`, `918`) with no noun_space accessor. | noun_space accessor usage is branch adapter glue and can be replaced by direct master noun decode handling. | high | replaceable_gap | medium | replace-with-master-equivalent | Concrete master noun decode callsites remove the need for result.noun_space adapters. | nounspace\|api-call\|sol | open | master_presence=missing |

## Test-Only Incompatibilities

Use this section only for dependencies reached exclusively by test code (e.g., unit-test-only helpers or `#[cfg(test)]` paths).

| dependency_id | finding_id | file_path | symbol_or_api | branch_context | master_evidence | impact_statement | confidence | match_rule | impact_level | disposition | disposition_rationale | tags | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Positive Controls

Positive controls are references intentionally retained to prove the inventory process does not over-report non-gaps.

| dependency_id | finding_id | file_path | symbol_or_api | branch_context | master_evidence | impact_statement | confidence | match_rule | impact_level | disposition | disposition_rationale | tags | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEP-ctl-001 | C010 | crates/nockchain-bench/src/speed_of_light/extractor.rs:314 | heaviest-chain-blocks-range | bench@nockchain-bench-master-candidate;target=upstream/master@cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c | `cd91acc3f2975ce2dc4f66ce73fb87a421e9b27c:crates/nockapp-grpc/src/services/public_nockchain/v2/block_explorer.rs:659` | Positive control confirms compatibility process does not over-report this known master path as missing. | high | replaceable_gap | low | replace-with-master-equivalent | Documented present-control dependency retained as non-gap validation row. | sol\|peek-path\|positive-control | control | master_presence=present-control |

## Linked Reference Map

Use this map to associate one primary dependency row with multiple concrete references.

| dependency_id | finding_id | link_type | reference_note |
| --- | --- | --- | --- |
| ref-001 | C001 | primary | maps to inventory-id D001: PMA persist runner flag gap |
| ref-002 | C002 | supporting | maps to inventory-id D001: PMA runtime env toggle linked to same dependency |
| ref-003 | C003 | primary | maps to inventory-id D002: PMA candidate sweep env default |
| ref-004 | C004 | primary | maps to inventory-id D003: Streaming checkpoint chunk env default |
| ref-005 | C005 | primary | maps to inventory-id D004: NounSpace compatibility concept |
| ref-006 | C006 | supporting | maps to inventory-id D004: in_space adapter trait callsites |
| ref-007 | C007 | supporting | maps to inventory-id D004: noun_space adapter trait callsites |
| ref-008 | C008 | supporting | maps to inventory-id D004: branch-only compat module definition |
| ref-009 | C014 | supporting | maps to inventory-id D004: HoonMapIter compat constructor call |
| ref-010 | C015 | supporting | maps to inventory-id D004: noun_space accessor on peek results |
| ref-011 | C009 | primary | maps to inventory-id D005: raw-transactions mempool peek path |
| ref-012 | C011 | primary | maps to inventory-id D006: data-dir flag runtime contract uncertainty |
| ref-013 | C012 | primary | maps to inventory-id D007: save-interval maps to nockapp boot CLI save_interval |
| ref-014 | C013 | primary | maps to inventory-id D008: new flag maps to nockapp boot CLI new |
| ref-015 | PMA-S001 | primary | maps to inventory-id D009: sampler PMA mmap-path heuristic |
| ref-016 | PMA-S002 | supporting | maps to inventory-id D009: sampler PMA memory-bucket attribution |
| ref-017 | C010 | control | maps to inventory-id CTRL001: heaviest-chain-blocks-range present-control |
