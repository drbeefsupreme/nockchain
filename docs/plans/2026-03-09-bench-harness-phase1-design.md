# Bench Harness Phase 1 Design

**Date:** 2026-03-09

**Goal:** Implement BENCH_HARNESS_SPEC_v4.md Phase 1 by introducing a shared once-run SOL replay core, a native trusted `sol bench` path that emits auditable artifacts, and a thin `sol quick-bench` quick path over the same measurement engine.

## Scope

Phase 1 is native-only. It does not implement Docker execution, hidden container-only CLI entrypoints, or validation-gate behavior beyond the release-build policy required for trusted native runs.

The implementation must:

- define the Phase 1 harness data model under `speed_of_light::harness`
- extract a machine-oriented once-run library seam from the current SOL bench path
- add native `sol bench` orchestration with warmups, measured runs, cooldowns, artifacts, summary, and verdict
- keep `sol quick-bench` working as the quick ad hoc interface

## Design

### 1. Harness module layout

Add a Phase 1 subset of the spec’s new harness tree:

- `harness/case.rs`
- `harness/execute.rs`
- `harness/native.rs`
- `harness/artifacts.rs`
- `harness/provenance.rs`
- `harness/summary.rs`
- existing `harness/docker.rs` remains as Phase 0 salvage only

This keeps the new trusted-run logic out of `commands/sol.rs` and avoids coupling the future Docker path to CLI glue.

### 2. Shared once-run execution seam

Promote the current `SolBenchRunner` path into a reusable machine-oriented operation. The shared once-run API should:

- accept a resolved native case plus a per-run output directory
- extract fixture inputs into a temporary work area
- execute one replay through `SolBenchRunner`
- persist machine-readable run artifacts:
  - `result.json`
  - `profile.json` when present
  - `block_timings.ndjson`
- return a structured run record for summary aggregation

`sol quick-bench` should call this same library path, then print the existing human summary.

### 3. Requested vs resolved case

Add separate request and resolved structs:

- `RequestedCase` captures the user-declared native request: fixture path, blocks, checkpointing, profiling, threads placeholder, warmups, measured runs, cooldown, output directory intent, and optional label.
- `ResolvedCase` freezes defaults and static derived facts: absolute fixture path, fixture hash, embedded fixture manifest, schema version, tool version, build profile, and run-count defaults.

Phase 1 only supports native execution. The execution enum can still exist with a native variant so the model stays aligned with the spec without dragging in Docker behavior.

### 4. Provenance, summary, verdict

`provenance.json` for Phase 1 records host-side facts only:

- capture timestamp
- hostname, OS, kernel, CPU count, total memory when available
- git commit/branch/dirty status when discoverable
- binary version and build profile
- fixture identity and embedded manifest

`summary.json` is derived from measured runs only and retains raw arrays for each metric. It computes:

- median
- min
- max
- MAD
- stddev
- CV
- values

`verdict.json` is driven by policy and observed outcomes:

- `Invalid` for release-policy violations
- `Partial` when one or more measured runs fail or throughput CV exceeds the default threshold
- `Valid` otherwise

### 5. CLI surface

Add `sol bench` as the trusted native interface. It should:

- take the same replay controls as `sol quick-bench`
- add `--output`, `--warmup-runs`, `--measured-runs`, `--cooldown-secs`, `--label`
- reject non-release trusted runs unless an explicit allow flag is provided

Keep `sol quick-bench` as the quick path and do not force artifact-tree output there.

## Testing

Use TDD:

1. add failing tests for request/resolution defaults, summary math, artifact layout, CLI parsing, and release gating
2. implement the minimal harness core and native runner
3. refactor `sol quick-bench` onto the shared once-run seam
4. verify with:
   - `cargo build -p nockchain-bench --release`
   - `cargo test -p nockchain-bench --release`

## Non-goals

- Docker trusted runner
- `sol run-once`
- validation probing and cache
- sweep rewrite
