# Bench Harness Phase 2 Refactor Gate Design

**Date:** 2026-03-10

**Goal:** Start `BENCH_HARNESS_SPEC_v7.md` Phase 2 by extracting a shared trusted orchestrator from the native harness, defining the backend contract it needs, and proving native artifact semantics remain equivalent before Docker execution work begins.

## Scope

This pass is limited to the Phase 2 refactor gate:

- extract the trusted orchestration loop from `harness/native.rs` into `harness/orchestrate.rs`
- define a backend contract for trusted runs
- convert native trusted execution into a backend adapter over that contract
- verify native artifact structure and JSON semantics remain equivalent after the refactor

This pass does not yet:

- implement actual Docker replay execution
- widen trusted CLI surface for Docker mode selection
- implement `sol run-once`
- add validation or sweep work

## Review Findings

The current Phase 1 code already has the right lower-level seam for this extraction:

- `harness/execute.rs` is the once-run engine for a single replay
- `harness/native.rs` still owns the trusted orchestration lifecycle around it
- `harness/provenance.rs` already models backend runtime facts, but only for native
- `RequestedCase.execution` still exposes only `Native`, so Phase 2 can start without changing trusted CLI behavior

That means the first Phase 2 slice should be a refactor that preserves behavior, not a Docker-first expansion.

## Design

### 1. Shared trusted orchestrator

Create `harness/orchestrate.rs` and move the trusted lifecycle there:

1. prepare the output root
2. resolve the requested case
3. let the backend perform setup and expose runtime facts
4. write root artifacts
5. run warmups
6. run measured iterations
7. compute summary and verdict

`native.rs` becomes a thin wrapper that constructs `NativeBackend` and delegates to the shared orchestrator.

### 2. Backend contract

Introduce a narrow backend trait for trusted execution. It only needs to express the seams Phase 2 actually requires:

- capture realized backend runtime facts after setup and before measured execution
- execute a single run for a resolved case into a run directory
- identify the backend facts to persist in `provenance.json`

The contract should preserve the current `CompletedRun`/summary flow so we do not rewrite metrics aggregation while extracting the orchestrator.

### 3. Provenance construction

Replace the native-only provenance constructor with a builder that accepts:

- the resolved case
- host identity and git metadata
- backend runtime facts supplied by the adapter

For this refactor, native still emits `BackendRuntimeFacts::Native`, so persisted output remains stable while the construction path becomes backend-agnostic.

### 4. Native-equivalence verification

Treat parity as semantic, not byte-for-byte. Tests should verify:

- the same root artifact set is emitted
- per-run artifacts remain in the same layout
- normalized JSON content remains equivalent for user-meaningful fields
- summary and verdict semantics match existing native behavior

Naturally variable values such as timestamps may differ.

## Testing

Follow TDD for the refactor gate:

1. add failing orchestrator/backend tests around the extracted contract
2. add a failing native parity test for artifact structure and normalized JSON semantics
3. implement the minimal orchestrator extraction and native adapter
4. run targeted tests for the new modules
5. run `cargo build -p nockchain-bench --release`
6. run `cargo test -p nockchain-bench --release`

## Outcome

This slice is complete when native trusted runs execute through the shared orchestrator with equivalent artifacts and no user-facing Docker behavior added yet.
