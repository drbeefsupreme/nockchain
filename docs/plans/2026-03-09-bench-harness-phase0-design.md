# Bench Harness Phase 0 Design

**Date:** 2026-03-09

**Goal:** Implement the BENCH_HARNESS_SPEC_v4.md Phase 0 hard deletion boundary for `nockchain-bench` while keeping the remaining SOL-focused CLI and library surfaces building in `--release` mode.

## Scope

Phase 0 is a clean break. This change removes mining-era subsystems, removes legacy CLI commands, removes the current `sol sweep` command, trims crate exports to SOL-only surfaces plus `sample`, and reevaluates dependencies that existed only for the deleted code.

This turn intentionally stops short of building the full new trustworthy harness. It only carries forward the generic Docker helpers that the spec explicitly names as salvageable source material.

## Design

### 1. Hard deletion boundary

Delete these modules and command surfaces:

- `src/scenario/`
- `src/events/`
- `src/output/`
- `src/runner/`
- `src/commands/mining.rs`
- `src/speed_of_light/sweep.rs`
- top-level CLI commands `run`, `attach`, `compare`, `analyze`
- `sol sweep`

After this trim, the CLI surface is:

- `sample`
- `sol quick-bench`
- `sol extract`
- `sol checkpoint`
- `sol inspect`
- `sol fixture build`
- `sol fixture inspect`

### 2. Minimal SOL-local salvage

Create a new `speed_of_light::harness` module with a `docker` submodule and move only the spec-listed generic helpers there:

- Docker connection logic with multi-socket discovery and ping
- `ContainerStats`
- `from_docker_stats`
- `parse_memory_limit`
- `parse_proc_stat_faults`
- `calculate_cpu_percent`

This does not preserve the old mining-oriented `DockerRunner`, `DockerRunnerConfig`, or `NockchainMode` API.

### 3. Crate surface cleanup

Trim `lib.rs`, `speed_of_light/mod.rs`, and `commands/sol.rs` so no deleted subsystem remains referenced. The remaining SOL code should compile without depending on mining-era abstractions.

### 4. Dependency cleanup

Remove `arrow`, `parquet`, and `chrono` if they are unused after the Phase 0 deletion. Keep Docker dependencies because later SOL harness phases still need them.

## Testing

Use TDD for behavior changes in the remaining code:

- first add or adjust focused tests around CLI/helper surfaces that remain
- then delete incompatible modules and rewire imports
- finally verify:
  - `cargo build -p nockchain-bench --release`
  - `cargo test -p nockchain-bench --release`

## Non-goals

- No compatibility shims
- No replacement `sol sweep`
- No full `speed_of_light::harness` implementation beyond minimal Docker helper salvage
- No mining benchmark support of any kind
