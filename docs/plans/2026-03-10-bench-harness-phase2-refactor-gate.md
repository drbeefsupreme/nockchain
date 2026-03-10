# Bench Harness Phase 2 Refactor Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract the native trusted benchmark orchestration into a shared orchestrator and backend contract while preserving native artifact semantics.

**Architecture:** Keep `execute.rs` as the once-run engine and move only the trusted run lifecycle into a new `orchestrate.rs`. Native execution becomes a thin backend adapter that supplies runtime facts and delegates each run to the existing once-run path, while provenance construction is made backend-agnostic.

**Tech Stack:** Rust, Tokio, Serde JSON, existing `nockchain-bench` harness modules

---

### Task 1: Add Failing Tests For The Shared Orchestrator Seam

**Files:**
- Create: `crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/mod.rs`
- Test: `crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs`

**Step 1: Write the failing tests**

Add orchestrator-focused tests in `orchestrate.rs` using a fake backend that records call order and returns canned `CompletedRun` values:

```rust
#[tokio::test]
async fn orchestrator_captures_runtime_facts_before_measured_runs() {
    let backend = FakeBackend::successful();

    let result = execute_trusted_run(&backend, requested_case(), tempdir.path(), false).await;

    assert!(result.is_ok());
    assert_eq!(backend.events(), vec![
        "setup",
        "warmup-0",
        "run-0",
        "run-1",
        "run-2",
    ]);
}
```

Add a second failing test that verifies summary/verdict behavior is preserved when one measured run fails.

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench --release speed_of_light::harness::orchestrate::tests -- --nocapture`
Expected: FAIL because `orchestrate.rs` and `execute_trusted_run` do not exist yet.

**Step 3: Write minimal implementation**

Create `orchestrate.rs` with:

- `TrustedBackend` trait
- `TrustedRunResult`
- skeletal `execute_trusted_run(...)`
- temporary fake-backend test helpers

Only add enough structure to compile and drive the tests.

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench --release speed_of_light::harness::orchestrate::tests -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs \
  crates/nockchain-bench/src/speed_of_light/harness/mod.rs
git commit -m "test: add trusted orchestrator seam coverage"
```

### Task 2: Make Provenance Construction Backend-Agnostic

**Files:**
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/provenance.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/artifacts.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/mod.rs`
- Test: `crates/nockchain-bench/src/speed_of_light/harness/provenance.rs`

**Step 1: Write the failing test**

Add a provenance test that builds a `Provenance` value from a resolved case plus explicit `BackendRuntimeFacts` and asserts the backend payload is preserved:

```rust
#[test]
fn provenance_builder_uses_supplied_backend_runtime_facts() {
    let provenance = build_provenance(&resolved_case(), BackendRuntimeFacts::Native);
    assert_eq!(provenance.backend, BackendRuntimeFacts::Native);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench --release speed_of_light::harness::provenance::tests -- --nocapture`
Expected: FAIL because the shared builder does not exist yet.

**Step 3: Write minimal implementation**

Refactor `provenance.rs` so native-specific host/git collection stays internal, but the final provenance assembly happens in a reusable function such as:

```rust
pub fn build_provenance(
    resolved: &ResolvedCase,
    backend: BackendRuntimeFacts,
) -> Provenance
```

Keep the serialized shape unchanged.

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench --release speed_of_light::harness::provenance::tests -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/nockchain-bench/src/speed_of_light/harness/provenance.rs \
  crates/nockchain-bench/src/speed_of_light/harness/artifacts.rs \
  crates/nockchain-bench/src/speed_of_light/harness/mod.rs
git commit -m "refactor: make harness provenance backend-aware"
```

### Task 3: Convert Native Trusted Execution Into A Backend Adapter

**Files:**
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/native.rs:1-175`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/mod.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/mod.rs`
- Test: `crates/nockchain-bench/src/speed_of_light/harness/native.rs`

**Step 1: Write the failing test**

Add a native adapter test that asserts `execute_native_trusted_run(...)` still returns a result with the resolved case, provenance, summary, and verdict fields populated via the shared orchestrator.

```rust
#[test]
fn native_run_result_still_exposes_refactored_outputs() {
    let result = NativeRunResult::from_trusted_run(sample_trusted_run_result());
    assert_eq!(result.provenance.backend, BackendRuntimeFacts::Native);
}
```

If a direct end-to-end test is practical with temporary fixtures, prefer that over a conversion-only unit test.

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench --release speed_of_light::harness::native::tests -- --nocapture`
Expected: FAIL because native still owns the orchestration loop directly.

**Step 3: Write minimal implementation**

Move the loop and artifact writing out of `native.rs`, then:

- define `NativeBackend`
- implement `TrustedBackend` for it by delegating to `execute_once(...)`
- keep `prepare_output_root(...)` in the shared path or move it into `orchestrate.rs`
- make `execute_native_trusted_run(...)` a thin wrapper around `execute_trusted_run(...)`

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench --release speed_of_light::harness::native::tests -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/nockchain-bench/src/speed_of_light/harness/native.rs \
  crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs \
  crates/nockchain-bench/src/speed_of_light/harness/mod.rs \
  crates/nockchain-bench/src/speed_of_light/mod.rs
git commit -m "refactor: route native trusted runs through orchestrator"
```

### Task 4: Add Native Artifact Parity Coverage

**Files:**
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/native.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/artifacts.rs`
- Test: `crates/nockchain-bench/src/speed_of_light/harness/native.rs`

**Step 1: Write the failing test**

Add a parity-oriented test that compares emitted artifact paths and normalized JSON payloads for a native trusted run against explicit semantic expectations:

```rust
#[test]
fn native_trusted_run_preserves_artifact_semantics_after_refactor() {
    let artifact = run_native_fixture_case(tempdir.path());
    assert!(artifact.root.join("summary.json").exists());
    assert_eq!(normalized_json("verdict.json"), expected_verdict_json());
}
```

Normalize timestamp-like fields before comparison.

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench --release native_trusted_run_preserves_artifact_semantics_after_refactor -- --nocapture`
Expected: FAIL until the shared orchestrator/native adapter emit the expected shape.

**Step 3: Write minimal implementation**

Adjust the orchestrator/native adapter only where needed to restore semantic parity. Do not widen CLI or Docker behavior in this task.

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench --release native_trusted_run_preserves_artifact_semantics_after_refactor -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/nockchain-bench/src/speed_of_light/harness/native.rs \
  crates/nockchain-bench/src/speed_of_light/harness/artifacts.rs
git commit -m "test: lock native harness artifact semantics"
```

### Task 5: Full Verification

**Files:**
- Modify: `docs/plans/2026-03-10-bench-harness-phase2-refactor-gate.md`

**Step 1: Run targeted harness tests**

Run: `cargo test -p nockchain-bench --release speed_of_light::harness -- --nocapture`
Expected: PASS

**Step 2: Run release build**

Run: `cargo build -p nockchain-bench --release`
Expected: PASS

**Step 3: Run release tests**

Run: `cargo test -p nockchain-bench --release`
Expected: PASS

**Step 4: Update plan progress notes**

Mark the completed tasks and capture any deferred Docker follow-up.

**Step 5: Commit**

```bash
git add docs/plans/2026-03-10-bench-harness-phase2-refactor-gate.md
git commit -m "docs: record phase 2 refactor gate verification"
```

## Progress Notes

### 2026-03-10 continuation

- Task 4 now has wrapper-level parity coverage via `native_trusted_run_preserves_artifact_semantics_after_refactor` in `crates/nockchain-bench/src/speed_of_light/harness/native.rs`.
- That parity test exercises the native trusted wrapper through an injected backend seam and locks:
  - root artifact tree layout
  - requested/resolved/provenance/summary/verdict JSON semantics after normalization of environment-specific fields
  - provenance binary git commit parity with the resolved case
- Verification commands completed successfully:
  - `cargo test -p nockchain-bench --release native_trusted_run_preserves_artifact_semantics_after_refactor -- --nocapture`
  - `cargo test -p nockchain-bench --release speed_of_light::harness -- --nocapture`
  - `cargo build -p nockchain-bench --release`
  - `cargo test -p nockchain-bench --release`
- Current refactor-gate status:
  - shared orchestrator extraction is in place
  - native trusted execution routes through the shared orchestrator
  - native wrapper artifact semantics are now covered by parity-oriented tests
  - Docker Phase 2 steps 4+ remain deferred
