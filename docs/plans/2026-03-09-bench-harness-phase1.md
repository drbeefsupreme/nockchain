# Bench Harness Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the BENCH_HARNESS_SPEC_v4.md Phase 1 native trustworthy SOL harness with a shared once-run core, native `sol run`, persisted artifacts, summary statistics, verdict computation, and a `sol bench` refactor onto the same execution seam.

**Architecture:** Add a Phase 1 `speed_of_light::harness` library surface that owns requested/resolved case modeling, provenance capture, artifact writing, summary math, and native run orchestration. Reuse `SolBenchRunner` as the measurement engine via one shared once-run function so `sol run` and `sol bench` stay behaviorally aligned.

**Tech Stack:** Rust, clap, serde, tokio, existing `speed_of_light` replay modules, cargo release builds/tests

---

### Task 1: Define the Phase 1 harness model and summary math

**Files:**
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/mod.rs`
- Create: `crates/nockchain-bench/src/speed_of_light/harness/case.rs`
- Create: `crates/nockchain-bench/src/speed_of_light/harness/summary.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/mod.rs`

**Step 1: Write the failing test**

Add focused unit tests for:
- `RequestedCase` default native run counts and minimum measured-run policy
- `ResolvedCase` fixture/hash/build-profile resolution behavior
- summary metric math for median/min/max/MAD/stddev/CV
- verdict downgrades for failed runs and unstable throughput

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench --release harness_summary`

Expected: FAIL because the new case/summary modules and types do not exist yet.

**Step 3: Write minimal implementation**

Add the harness case model, summary/value-stat helpers, and verdict calculation with only the fields Phase 1 needs.

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench --release harness_summary`

Expected: PASS.

### Task 2: Extract the shared once-run execution path and artifact writer

**Files:**
- Create: `crates/nockchain-bench/src/speed_of_light/harness/execute.rs`
- Create: `crates/nockchain-bench/src/speed_of_light/harness/artifacts.rs`
- Create: `crates/nockchain-bench/src/speed_of_light/harness/provenance.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/bench.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/mod.rs`

**Step 1: Write the failing test**

Add focused tests for:
- per-run artifact persistence (`result.json`, `profile.json`, `block_timings.ndjson`)
- provenance file shape for native runs
- shared once-run output struct serialization

Prefer pure unit tests for artifact/provenance writers and keep replay execution mocked only where unavoidable.

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench --release harness_artifacts`

Expected: FAIL because the artifact/provenance writers and shared execution seam are missing.

**Step 3: Write minimal implementation**

Factor the current `SolBenchRunner` usage into a machine-oriented once-run function that can persist artifacts into a provided run directory and return structured results.

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench --release harness_artifacts`

Expected: PASS.

### Task 3: Add native trusted `sol run` orchestration and refactor `sol bench`

**Files:**
- Create: `crates/nockchain-bench/src/speed_of_light/harness/native.rs`
- Modify: `crates/nockchain-bench/src/commands/sol.rs`
- Modify: `crates/nockchain-bench/src/main.rs`

**Step 1: Write the failing test**

Add CLI-focused tests proving:
- `sol run` parses with Phase 1 native trusted options
- non-release trusted runs are rejected by default
- measured runs below the Phase 1 minimum are rejected
- `sol bench` still parses and routes through the shared replay config path

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench --release sol_run_cli`

Expected: FAIL because `sol run` and its policy checks do not exist yet.

**Step 3: Write minimal implementation**

Add the `sol run` subcommand, native repetition loop, cooldown handling, artifact-tree root writing, release gating, and the `sol bench` refactor onto the shared once-run seam.

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench --release sol_run_cli`

Expected: PASS.

### Task 4: Verify Phase 1 exit criteria in release mode

**Files:**
- Modify: `crates/nockchain-bench/src/main.rs`
- Modify: `crates/nockchain-bench/src/commands/sol.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/*.rs` as needed

**Step 1: Write the failing check**

Use the full crate release build/test as the gate for unresolved integration issues.

**Step 2: Run build/test to identify failures**

Run: `cargo build -p nockchain-bench --release`

Run: `cargo test -p nockchain-bench --release`

Expected: FAIL until all new harness paths are wired together cleanly.

**Step 3: Write minimal implementation**

Fix any remaining integration issues, serialization mismatches, or CLI wiring problems required for the native trusted Phase 1 path.

**Step 4: Run full verification**

Run: `cargo build -p nockchain-bench --release`

Run: `cargo test -p nockchain-bench --release`

Expected: both PASS.
