# Phase 1 Evidence Contract

## Purpose

Define the enforceable evidence schema for every incompatibility finding recorded during compatibility inventory and provenance work.

## Atomic Finding Record

Each finding is one atomic record in the canonical ledger and MUST include all required fields:

- `file_path` (file path)
- `symbol_or_api` (symbol/API reference)
- `branch_context` (branch context including pinned master SHA)
- `impact_statement` (impact statement)
- `confidence`
- `match_rule`
- `impact_level`

## Required Enums

### `match_rule` (locked)

- `exact_missing_ref`
- `replaceable_gap`
- `branch_env_config_toggle` (includes PMA and other branch env/config toggles)

No additional values are permitted in Phase 1 artifacts.

### `confidence` (locked)

Allowed values are exactly `high|medium|low`.

### `impact_level`

Use `low|medium|high` to indicate severity of unresolved or accepted incompatibility impact.

## Hard-Fail Requirements

- Missing any required field is a hard-fail and the record is invalid.
- Empty or non-normalized enum values are a hard-fail.
- `branch_context` MUST cite the pinned master SHA from `01-master-target.md`.
- Any unresolved finding with `impact_level=high` blocks phase closure (high-impact unresolved closure block).

## Unresolved and Low-Confidence Handling

- Findings with `confidence=low` are tracked in the unresolved section of the canonical ledger.
- Conflicting evidence between records requires manual resolution before final closure decisions.
- Unresolved records must still include full required evidence fields and enum-conformant values.

## Escalation Rule

When evidence conflicts cannot be resolved from Rust-code primary evidence alone, escalate for manual resolution and keep the finding in unresolved status until adjudicated.

