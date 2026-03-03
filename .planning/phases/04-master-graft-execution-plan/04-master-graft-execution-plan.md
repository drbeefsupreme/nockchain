# Phase 04 Canonical Master Graft Execution Plan

## Metadata

- Phase: `04-master-graft-execution-plan`
- Canonical master ref policy: use `refs/remotes/upstream/master` as source of truth, with local fallback only if upstream is unavailable.
- Scope reminder: this document locks runbook structure only; execution details are populated by later plans.
- Dependency linkage source: `.planning/phases/04-master-graft-execution-plan/04-graft-dependency-matrix.tsv`

## Deterministic Sequence

## R0 Bootstrap from upstream/master

Goal: start from a fresh baseline that can be replayed deterministically.

- Create a new execution branch from `refs/remotes/upstream/master`.
- Pin the baseline SHA in execution logs before any graft actions.
- Confirm clean working tree before proceeding.
- Rollback anchor: `R0-baseline`.

## R1 transplant staging and guardrails

Goal: reserve the transplant checkpoint for dependency-scoped graft operations.

- Lock the step boundary for transplant operations only.
- Require dependency IDs to be referenced from the matrix before changes.
- Enforce per-dependency verification command placeholders prior to population.
- Rollback anchor: `R1-pre-transplant`.

## R2 remove pass for branch-only dependencies

Goal: reserve removal sequence for dependencies marked remove or no-longer-needed.

- Execute in dependency ID order from the matrix.
- Require explicit rollback point IDs per row before implementation details are filled.
- Track negative-impact risk notes for each remove action.
- Rollback anchor: `R2-pre-remove`.

## R3 replace pass for master equivalents

Goal: reserve replacement sequence for dependencies with identified master alternatives.

- Map each replacement to exact target files and master references.
- Record compatibility risks and expected behavior parity checks.
- Keep implementation ordering deterministic by execution step ID.
- Rollback anchor: `R3-pre-replace`.

## R4 unresolved dependency decision gate

Goal: prevent silent continuation when unresolved dependencies remain (including `DEP-005`).

- Enumerate unresolved rows from the matrix with blocker rationale.
- Require explicit decision outcome per unresolved dependency: defer, redesign, or approved exception.
- Require sign-off before entering final verification.
- Rollback anchor: `R4-gate`.

## R5 verification and closure handoff

Goal: provide final verification checkpoint and verifier/make-gate handoff contract.

- Run plan-level verification commands for runbook, matrix, and checklist artifacts.
- Confirm closure checklist IDs are present and stable before execution population.
- Hand off to Phase 4 verifier integration without mutating reserved closure IDs.
- Rollback anchor: `R5-pre-close`.

## Dependency Treatment Linkage

- Every matrix row (`DEP-001` through `DEP-009`) must map to exactly one execution step ID in `R1`-`R4`.
- `execution_step_id`, `action`, `risk_note`, `rollback_point`, and `verification_command` are required before any execution pass is considered complete.
- Provenance class from Phase 3 must be preserved through execution planning and closure evidence.

## Stop-The-Line Criteria

- Baseline is not pinned to `upstream/master` SHA.
- Any dependency action is attempted without a matrix row and rollback point.
- A required verification command is missing for an in-scope dependency row.
- An unresolved dependency proceeds beyond `R4` without explicit decision record.
- Closure checklist IDs are changed or removed from checkpoint artifacts.

## Rollback Policy

- Use nearest checkpoint rollback anchor (`R0-baseline`, `R1-pre-transplant`, `R2-pre-remove`, `R3-pre-replace`, `R4-gate`, `R5-pre-close`) based on failing step.
- Revert only the current execution span first; preserve prior validated checkpoint evidence.
- Re-run verification commands for the reverted span before resuming.
- Record rollback trigger, impacted dependency IDs, and decision outcome in execution notes.

## Unresolved Decision Gate Requirements

- Unresolved items must include dependency ID, issue summary, options considered, selected disposition, approver, and timestamp.
- No unresolved item may be closed implicitly by progress in later checkpoints.
- Decision records must be linked from both this runbook and the dependency matrix notes field.
