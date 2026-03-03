# Codebase Structure

**Analysis Date:** 2026-03-03

## Directory Layout

```
/shared/nockchain/
├── crates/                    # Rust workspace crates
│   ├── nockchain-bench/       # benchmark harness and SOL tooling
│   ├── nockchain/             # node runtime binary crate
│   ├── nockapp/               # runtime app/kernel framework
│   ├── nockvm/                # VM/runtime internals
│   └── ...                    # supporting domain/network crates
├── scripts/                   # benchmark orchestration scripts
├── benchmarks/                # baseline configs and benchmark definitions
├── bench-artifacts/           # generated benchmark artifacts/history
├── bench-worktrees/           # branch-specific worktree pointers
├── .github/workflows/         # CI automation
└── .planning/                 # GSD planning and map artifacts
```

## Directory Purposes

**`crates/nockchain-bench/`:**
- Purpose: benchmark CLI and orchestration logic
- Contains: `main.rs`, `scenario/`, `runner/`, `speed_of_light/`, tests
- Key files: `src/main.rs`, `src/speed_of_light/*`, `tests/*`

**`crates/nockchain/`:**
- Purpose: node entry and runtime composition
- Contains: CLI config and runtime wiring
- Key files: `src/main.rs`, `src/lib.rs`, `src/config.rs`

**`scripts/`:**
- Purpose: reproducible matrix/baseline automation
- Key files: `sol_bench_matrix_trace.sh`, `sol_baseline_run.sh`

## Key File Locations

**Entry Points:**
- `crates/nockchain-bench/src/main.rs` - benchmark command router
- `crates/nockchain/src/main.rs` - node startup

**Configuration:**
- `Cargo.toml` (workspace)
- `rust-toolchain.toml` and `rustfmt.toml`
- `benchmarks/baseline/sol-baseline.toml`

**Core Logic:**
- `crates/nockchain-bench/src/speed_of_light/` - SOL subsystem
- `crates/nockchain-bench/src/runner/` - runner abstractions
- `crates/nockchain-bench/src/scenario/` - scenario orchestration

**Testing:**
- `crates/nockchain-bench/tests/` - integration tests
- module-local tests in `crates/nockchain-bench/src/**`

## Where to Add New Code

**New benchmark capability:**
- command wiring: `crates/nockchain-bench/src/main.rs`
- implementation: `crates/nockchain-bench/src/speed_of_light/` or `scenario/`
- tests: `crates/nockchain-bench/tests/`

**New branch-compat analysis logic:**
- place under `crates/nockchain-bench/src/speed_of_light/guard/` or dedicated analysis module
- add fixtures in `crates/nockchain-bench/tests/fixtures/`

---
*Structure analysis: 2026-03-03*
*Update when directory structure changes*
