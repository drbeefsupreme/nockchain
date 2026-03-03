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

## Runtime-Path Incompatibilities

Use this section for dependencies reached by runtime CLI flows, runner wiring, extraction/replay paths, and non-test behavior assumptions.

| dependency_id | finding_id | file_path | symbol_or_api | branch_context | master_evidence | impact_statement | confidence | match_rule | impact_level | disposition | disposition_rationale | tags | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

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

