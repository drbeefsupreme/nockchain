# Isomorphism Card — D1

## 1. Identity

- **Candidate ID:** D1
- **Run ID:** 2026-04-27-four-commits-simplify
- **Clone type:** II (parametric)
- **Expected LOC saved:** 16
- **Score:** (LOC_saved 3 * Confidence 5) / Risk 1 = 15

## 2. Sites

```
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:124
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:886
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:917
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:955
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:1018
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:1082
```

## 3. Observable Contract

- Return/output: `StepResult` fields and serialized JSON are unchanged.
- Side effects: none; assignments stay local to the `StepResult`.
- Error modes: `ColdStepError::VerifyFailed` still becomes an error `StepResult` with `cold_verified=false`; generic cold errors still omit cold metadata.
- Timing/order: no async, lock, I/O, or measurement ordering changes.
- Observability: JSON field names and omission behavior remain defined by `StepResultWire`.

## 4. Hidden Differences Between Sites

- Successful cold force and successful cold peek copy `degraded_reason` from `ColdForceResult`.
- Verify-failure cold force has no `ColdForceResult`; it copies the failure fields and explicitly sets `cold_verified=false`.
- Generic cold errors do not have residency or target metadata and must remain unchanged.
- Plain peek steps copy only measurement counters, not cold metadata.

Strategy: extract builder helpers on `StepResult` for measurement fields, `ColdForceResult`, and verify-failure metadata. Remove the now-unused `PeekMeasurement` accessors after direct use of the embedded `StepMeasurement`.

## 5. Proof Strategy

- Baseline targeted tests passed:
  - `cargo test -p nockchain-bench --release cold_peek`
  - `cargo test -p nockchain-bench --release orchestrator::tests`
- After edit, rerun the same targeted tests plus `cargo fmt --check`, `cargo check -p nockchain-bench --release`, and `git diff --check`.
- Broad `speed_of_light` baseline was not green due to an unrelated checkpoint fixture EOF in `checkpoint_builder::tests::full_checkpoint_mode_includes_runtime_startup_events`.

## 6. Risk

- Reversibility: simple helper extraction, reversible with one patch.
- Blast radius: private methods and private orchestration code only; no type signature change.
- Concurrency hazard: none; no shared mutable state introduced.

## 7. UBS Prompts

- No `unwrap`, `ignore`, env-var branch, feature flag, dependency, or test deletion is planned.

## 8. Commit Plan

- Commit 1: `refactor(bench): extract step result metadata builders`
- Verify: targeted release tests, formatter, diff check, and release check.
