# Bench Harness Phase 0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all mining-era and legacy sweep surfaces from `nockchain-bench`, keep only the Phase 0 SOL/sample interfaces, and leave the crate building and testing cleanly in `--release` mode.

**Architecture:** This phase is a hard deletion boundary. The crate becomes SOL-specific plus `sample`, and only the generic Docker stats/provenance helpers survive as source material under a new `speed_of_light::harness::docker` module. No compatibility wrappers remain.

**Tech Stack:** Rust, clap, tokio, bollard, serde, cargo release builds/tests

---

### Task 1: Add focused failing tests for the surviving helper surface

**Files:**
- Modify: `crates/nockchain-bench/src/commands/sol.rs`
- Create: `crates/nockchain-bench/src/speed_of_light/harness/mod.rs`
- Create: `crates/nockchain-bench/src/speed_of_light/harness/docker.rs`

**Step 1: Write the failing tests**

Add focused unit tests for:
- `latest_checkpoint_size_in_dir` in `commands/sol.rs`
- `parse_proc_stat_faults`
- `parse_memory_limit`

The Docker-helper tests should assert the generic salvage behavior defined by the spec, not the old mining API.

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench --release latest_checkpoint_size_in_dir`

Run: `cargo test -p nockchain-bench --release parse_proc_stat_faults`

Expected: missing `speed_of_light::harness::docker` module or missing functions/tests.

**Step 3: Write minimal implementation**

Create `speed_of_light::harness` with a `docker` module that exposes only the generic helpers needed for later phases:
- Docker connection/ping helper
- `ContainerStats`
- stats parsing helpers
- memory-limit parsing
- `/proc/1/stat` fault parsing

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench --release latest_checkpoint_size_in_dir`

Run: `cargo test -p nockchain-bench --release parse_proc_stat_faults`

Expected: PASS.

### Task 2: Remove legacy CLI commands and sweep entrypoints

**Files:**
- Modify: `crates/nockchain-bench/src/main.rs`
- Modify: `crates/nockchain-bench/src/commands/mod.rs`
- Modify: `crates/nockchain-bench/src/commands/sol.rs`

**Step 1: Write the failing test**

Add a focused CLI-shape regression test in `main.rs` or a nearby unit-testable helper that proves the surviving Phase 0 command set:
- keeps `sample`
- keeps `sol quick-bench|extract|checkpoint|inspect|fixture build|fixture inspect`
- removes `run|attach|compare|analyze|sol sweep`

If direct clap parser tests are awkward in `main.rs`, extract the clap enum construction into a testable helper first and test that helper.

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench --release phase0_cli_surface`

Expected: FAIL because legacy commands still parse or still appear in help/command definitions.

**Step 3: Write minimal implementation**

Delete the legacy command variants and routing from `main.rs`, remove `OutputFormat`, remove mining-module registration from `commands/mod.rs`, and remove `cmd_sol_sweep` plus related sweep helpers/imports from `commands/sol.rs`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench --release phase0_cli_surface`

Expected: PASS.

### Task 3: Delete mining-era modules and trim crate exports

**Files:**
- Modify: `crates/nockchain-bench/src/lib.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/mod.rs`
- Delete: `crates/nockchain-bench/src/scenario/mod.rs`
- Delete: `crates/nockchain-bench/src/scenario/mining.rs`
- Delete: `crates/nockchain-bench/src/events/mod.rs`
- Delete: `crates/nockchain-bench/src/events/log_parser.rs`
- Delete: `crates/nockchain-bench/src/output/mod.rs`
- Delete: `crates/nockchain-bench/src/output/parquet.rs`
- Delete: `crates/nockchain-bench/src/runner/mod.rs`
- Delete: `crates/nockchain-bench/src/runner/docker.rs`
- Delete: `crates/nockchain-bench/src/commands/mining.rs`
- Delete: `crates/nockchain-bench/src/speed_of_light/sweep.rs`

**Step 1: Write the failing test**

Add or update a crate-level compile-oriented test around the remaining `speed_of_light` exports so the surviving SOL APIs are explicit and the deleted sweep exports are gone.

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench --release speed_of_light_surviving_exports`

Expected: FAIL while deleted exports are still present.

**Step 3: Write minimal implementation**

Delete the mining/event/output/runner/sweep modules, remove their `pub mod` / `pub use` entries, and keep only SOL/sample-facing exports.

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench --release speed_of_light_surviving_exports`

Expected: PASS.

### Task 4: Remove dead dependencies and verify Phase 0 exit criteria

**Files:**
- Modify: `crates/nockchain-bench/Cargo.toml`
- Modify: `crates/nockchain-bench/Cargo.lock` (if needed)

**Step 1: Write the failing check**

Use the crate build itself as the gate for dependency cleanup and remaining references.

**Step 2: Run build/test to identify failures**

Run: `cargo build -p nockchain-bench --release`

Run: `cargo test -p nockchain-bench --release`

Expected: FAIL until imports/dependencies are cleaned up.

**Step 3: Write minimal implementation**

Remove `arrow`, `parquet`, and `chrono` if unused after deletions. Keep Docker dependencies. Fix any remaining import or warning-level fallout needed for a clean Phase 0 boundary.

**Step 4: Run full verification**

Run: `cargo build -p nockchain-bench --release`

Run: `cargo test -p nockchain-bench --release`

Expected: both PASS.
