# Cold Peeking Simplification Ledger

Run id: `2026-04-27-cold-peeking-simplify`
Base range reviewed: `05d24e26..HEAD`

## Baseline

| Check | Result |
|---|---:|
| `cargo build -p nockchain-bench --release` | pass |
| `cargo test -p nockchain-bench --release quick_orchestrate_` | 13 passed |
| `cargo test -p nockchain-bench --release force_cold_` | 4 passed |
| `cargo test -p nockchain-bench --release cold_init_` | 0 matched |
| `cargo check -p nockchain-bench --release --features pma-runtime-compat` in standalone checkout | expected failure: current branch lacks PMA-only helper/API |

## Candidate Matrix

| Candidate | Clone Type | LOC | Confidence | Risk | Score | Decision |
|---|---|---:|---:|---:|---:|---|
| Hoist duplicated cold-peek public data/error types from Linux `cgroup.rs` and non-Linux fallback `mod.rs` into shared `cold_peek/mod.rs` definitions | I/II | 3 | 5 | 1 | 15.0 | accept |

## Isomorphism Card

### Change: share cold-peek public data/error types across Linux and non-Linux modules

#### Equivalence contract

- **Inputs covered:** `ColdStepOptions`, `ColdForceResult`, `OffendingVmaResidency`, `ColdStepError`, and `ColdInitError` construction and matching from current Linux tests plus non-feature orchestrator validation.
- **Ordering preserved:** N/A; these are passive data and error types.
- **Tie-breaking:** N/A.
- **Error semantics:** Same enum variants and same `thiserror` messages move unchanged.
- **Laziness:** N/A.
- **Short-circuit eval:** N/A.
- **Floating-point:** N/A.
- **RNG / hash order:** N/A.
- **Observable side-effects:** None; no runtime logic changes.
- **Type narrowing:** Rust variant names and field names remain unchanged; exhaustiveness stays the same.
- **Rerender behavior:** N/A.

#### Verification

- [x] `cargo test -p nockchain-bench --release quick_orchestrate_`: 13 passed.
- [x] `cargo test -p nockchain-bench --release force_cold`: 4 passed.
- [x] `uv run --project scripts/bench_sync scripts/bench_sync/pma_bench_sync.py --target-dir /shared/nockchain/.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure --force --allow-dirty-source`: build ran.
- [x] `cargo test -p nockapp --release for_nc_bench_shim` in PMA worktree: 1 passed.
- [x] `cargo test -p nockchain-bench --release --features pma-runtime-compat cold_init_` in PMA worktree: 2 passed.
- [x] `cargo test -p nockchain-bench --release --features pma-runtime-compat quick_orchestrate_` in PMA worktree: 15 passed.
- [x] `cargo test -p nockchain-bench --release --features pma-runtime-compat force_cold_` in PMA worktree after final sync: 8 passed, 2 ignored checkpoint-backed smokes.
- [x] `cargo build -p nockchain-bench --release --features pma-runtime-compat` in PMA worktree: pass.
- [x] `/shared/nockchain/target/release/nockchain-bench sol quick-orchestrate --plan ... --cold-mode strict`: standalone current branch rejects `peek_height_cold` with the expected `requires --features pma-runtime-compat` error.
- [ ] PMA checkpoint-backed cold smoke: attempted both sandboxed and unsandboxed; blocked before boot by host cgroup setup: `cold peek requires a delegated cgroup v2 parent with memory in cgroup.subtree_control`.

## Result

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `cold_peek/cgroup.rs` LOC | 856 | 800 | -56 |
| `cold_peek/mod.rs` LOC | 121 | 111 | -10 |
| Net source LOC | 977 | 911 | -66 |

