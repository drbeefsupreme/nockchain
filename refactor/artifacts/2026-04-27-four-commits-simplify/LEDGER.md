# Refactor Ledger — 2026-04-27-four-commits-simplify

## Scope

- Four commits from 2026-04-27:
  - `d7e7cdaf refactor: share cold peek types`
  - `7e9f33a1 remove madv_pageout`
  - `ed4bdbcc bench: support current branch cold peeks`
  - `6ce5082b bench: cold PMA and NockStack by default`
- Code surface: `crates/nockchain-bench/src/speed_of_light/cold_peek/*` and `orchestrator.rs`.

## Candidate Ledger

| ID | Decision | Change | LOC Delta | Proof |
|----|----------|--------|-----------|-------|
| D1 | accepted | Extracted `StepResult` builder helpers for measurement and cold metadata; removed unused `PeekMeasurement` accessors. | `orchestrator.rs` 72 insertions / 89 deletions; `cgroup.rs` 8 insertions / 7 deletions from `cargo fmt`; scoped total 3875 -> 3859 lines. | `cargo fmt --check`; `cargo test -p nockchain-bench --release cold_peek`; `cargo test -p nockchain-bench --release orchestrator::tests`; `cargo check -p nockchain-bench --release`; `git diff --check`. |

## Baseline Note

The broad release baseline `cargo test -p nockchain-bench --release --lib speed_of_light` was not green before edits because `speed_of_light::checkpoint_builder::tests::full_checkpoint_mode_includes_runtime_startup_events` failed with `UnexpectedEof` while reading a local archive fixture. The scoped cold-peek and orchestrator release tests were green before and after the refactor.

## Warnings

Existing warnings remain:

- `cold_peek/mod.rs`: unused test re-export `set_test_cold_init_overrides` in some filtered test builds.
- `cold_peek/cgroup.rs`: `ColdInitTestOverrideGuard` and `set_test_cold_init_overrides` unused in some filtered test builds.
- `cold_peek/measure.rs`: `PeekMeasurement.sample` dead-code warning.
