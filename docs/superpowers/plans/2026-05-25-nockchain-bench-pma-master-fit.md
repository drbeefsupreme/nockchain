# Nockchain Bench PMA Master Fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the old PMA-less master compatibility layer from `nockchain-bench` so the crate is shaped around current `master`, where PMA is the normal runtime path.

**Architecture:** Make PMA replay the single runtime implementation instead of a feature-gated compatibility path. Then remove or narrow the legacy checkpoint-production and non-PMA fallback surfaces that existed only because `nockchain-bench` straddled old master and the PMA branch. Keep existing `.soltest` replay, trusted native/Docker benchmarking, sweeps, cold-peek evidence, and bench_pages compatibility.

**Tech Stack:** Rust workspace, `nockchain-bench`, `nockapp::kernel::form::PmaConfig::for_replay`, PMA-backed `NockApp`, Cargo release builds, Docker trusted harness, `scripts/bench_pages`.

---

## Current State

The release-candidate branch has PMA merged into master and one small `nockapp` API addition:

- `crates/nockapp/src/kernel/form.rs`: `PmaConfig::for_replay(...)`
- `crates/nockchain-bench/src/speed_of_light/runtime_compat.rs`: PMA replay boot helper behind `pma-runtime-compat`
- `crates/nockchain-bench/Cargo.toml`: feature `pma-runtime-compat = []`

The branch still contains compatibility code for the old PMA-less master:

- `#[cfg(feature = "pma-runtime-compat")]` and `#[cfg(not(feature = "pma-runtime-compat"))]` throughout the bench crate
- `runtime_compat.rs` name and docs that still describe transplant/PMA-compat workflows
- non-PMA noun access shims in `noun_compat.rs`
- non-PMA `Kernel::load_with_hot_state_medium(...)` and `boot::setup(...)` paths in `kernel_utils.rs`
- checkpoint production paths that are guarded as unsupported under PMA
- CLI/docs for checkpoint cadence, full checkpoint construction, and dual master/PMA behavior

## Desired End State

- `cargo build -p nockchain-bench --release` builds the PMA-backed bench binary without feature flags.
- `--features pma-runtime-compat` no longer exists and no code references that feature.
- PMA replay boot is named as the normal path, not "compat".
- `--fsync on|off` is a normal bench CLI field wherever it is supported today.
- Existing fixture replay, quick read, quick orchestrate, trusted native bench, trusted Docker bench, and trusted sweep flows continue to work.
- Unsupported checkpoint-production behavior is either removed from the CLI or retained only as explicit "not supported on current PMA master" stubs with no old implementation hidden behind cfgs.
- README/spec text describes current master directly; no transplant instructions, no "final dual branch", no "PMA-only branch later" language.

## File Map

Core runtime:

- Modify: `crates/nockchain-bench/Cargo.toml`
- Modify: `crates/nockchain-bench/src/speed_of_light/mod.rs`
- Rename/modify: `crates/nockchain-bench/src/speed_of_light/runtime_compat.rs` -> `crates/nockchain-bench/src/speed_of_light/pma_replay.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/noun_compat.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/extractor.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/poke.rs`
- Keep: `crates/nockapp/src/kernel/form.rs` (`PmaConfig::for_replay`)

CLI and command wiring:

- Modify: `crates/nockchain-bench/src/main.rs`
- Modify: `crates/nockchain-bench/src/commands/sol.rs`

Benchmark execution and checkpoint-related cleanup:

- Modify: `crates/nockchain-bench/src/speed_of_light/bench.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/checkpoint.rs` only if command surfaces change
- Modify: `crates/nockchain-bench/src/speed_of_light/fixture.rs` only if fixture-build behavior changes

Cold-peek and PMA evidence:

- Modify: `crates/nockchain-bench/src/speed_of_light/cold_peek/cgroup.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/cold_peek/mod.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/cold_peek/vma.rs` only if names change
- Modify: `crates/nockchain-bench/src/speed_of_light/orchestrator.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/orchestrate_execute.rs`

Trusted harness:

- Modify: `crates/nockchain-bench/src/speed_of_light/harness/case.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/provenance.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/sweep.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/native.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/validate.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/docker.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/artifacts.rs` only if schemas change

Tests and docs:

- Modify/delete: `crates/nockchain-bench/tests/binary_identity_build_profile.rs`
- Modify: `crates/nockchain-bench/tests/build_support.rs`
- Modify: `crates/nockchain-bench/tests/docker_image_build_flow.rs`
- Modify: `crates/nockchain-bench/README.md`
- Modify or delete superseded copies: `crates/nockchain-bench/BENCH_HARNESS_SPEC_v5.md`, `crates/nockchain-bench/BENCH_HARNESS_SPEC_v6.md`, `crates/nockchain-bench/BENCH_HARNESS_SPEC_v7.md`
- Modify: `crates/nockchain-bench/specs/bench-harness-spec.md`

## Review Questions For Claude

Ask Claude to specifically challenge these points before implementation:

- This plan now deletes `pma-runtime-compat` only at the final audit after every cfg site is gone. Challenge only if a downstream-compatible no-op feature should survive one release.
- This plan now commits to keeping `sol checkpoint` and `sol fixture build` as explicit unsupported commands for one branch, while keeping `sol fixture inspect` working and deleting non-PMA implementation bodies. Challenge only if the command surfaces should be removed outright.
- Should non-PMA NockStack cold-target override remain as a developer diagnostic (`NOCKCHAIN_BENCH_COLD_TARGET=nockstack`), or should PMA replay be the only cold target? This plan recommends keeping the diagnostic override if it remains useful, but deleting non-PMA default behavior.
- Should historical specs v5/v6/v7 be deleted, moved under `docs/archive`, or left as-is with a current-spec pointer? This plan recommends deleting v5/v6 from the crate and making `specs/bench-harness-spec.md` the canonical spec.

## Implementation Invariant

Every commit in this plan must build. Do not delete the `pma-runtime-compat`
feature until all `#[cfg(...pma-runtime-compat...)]` sites are gone. The
transitional step is to make `pma-runtime-compat` a default feature, then remove
cfgs file-by-file, then delete the feature entry near the end.

---

## Task 1: Add A No-Cruft Audit Baseline

**Files:**

- Modify: none initially
- Test: shell audit commands

- [ ] **Step 1: Record the current PMA compatibility references**

Run:

```bash
rg 'pma-runtime-compat|runtime_compat|PMA-compatible|PMA compatibility|transplant|PMA checkout|master-style|PMA-only branch|without `pma-runtime-compat`|not\\(feature = "pma-runtime-compat"\\)' crates/nockchain-bench crates/nockapp/src/kernel/form.rs Cargo.toml
```

Expected: many matches. Save this output in the implementation notes for comparison.

- [ ] **Step 2: Record currently passing PMA release checks**

Run:

```bash
cargo test -p nockapp --release for_replay
cargo test -p nockchain-bench --release --features pma-runtime-compat cold_init_
cargo test -p nockchain-bench --release --features pma-runtime-compat quick_orchestrate_
cargo test -p nockchain-bench --release --features pma-runtime-compat force_cold_
cargo build -p nockchain-bench --release --features pma-runtime-compat
```

Expected: all pass before cleanup.

- [ ] **Step 3: Commit only if a baseline note file is created**

If an implementation notes file is created, commit it separately. Otherwise do not commit.

```bash
git add <notes-file>
git commit -m "docs: record bench PMA cleanup baseline"
```

## Task 2: Make PMA Replay The Normal Build

**Files:**

- Modify: `crates/nockchain-bench/Cargo.toml`
- Modify: `crates/nockchain-bench/src/speed_of_light/mod.rs`
- Rename: `crates/nockchain-bench/src/speed_of_light/runtime_compat.rs` -> `crates/nockchain-bench/src/speed_of_light/pma_replay.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs`

- [ ] **Step 1: Write/adjust a compile test expectation**

Update tests or add a small assertion in an existing test module so the normal release build uses PMA replay with no feature flag. Prefer an existing runtime helper test in the renamed `pma_replay.rs`.

Expected test intent:

```rust
#[test]
fn replay_pma_config_returns_fresh_replay_shape() {
    let tempdir = tempdir().expect("tempdir should be created");
    let config = replay_pma_config(tempdir.path(), true).expect("replay config should be prepared");

    assert_eq!(config.path_0, tempdir.path().join("replay-pma/0.pma"));
    assert_eq!(config.path_1, tempdir.path().join("replay-pma/1.pma"));
    assert_eq!(config.words, nockapp::utils::NOCK_STACK_SIZE_MEDIUM);
    assert_eq!(config.reserved_words, None);
    assert!(!config.open_existing);
    assert!(!config.create_snapshots);
}
```

- [ ] **Step 2: Turn PMA runtime on by default, but do not delete the feature yet**

Change this block in `crates/nockchain-bench/Cargo.toml`:

```toml
[features]
default = ["pma-runtime-compat"]
pma-runtime-compat = []
```

This keeps intermediate commits compiling while cfgs are removed across the
crate. The feature entry is deleted only after the final cfg audit.

- [ ] **Step 3: Rename `runtime_compat.rs`**

Rename the file:

```bash
git mv crates/nockchain-bench/src/speed_of_light/runtime_compat.rs crates/nockchain-bench/src/speed_of_light/pma_replay.rs
```

Update `crates/nockchain-bench/src/speed_of_light/mod.rs` from:

```rust
mod runtime_compat;
```

to:

```rust
mod pma_replay;
```

Update all direct callers of the old module name:

- `crates/nockchain-bench/src/speed_of_light/extractor.rs`
- `crates/nockchain-bench/src/speed_of_light/poke.rs`

Run:

```bash
git grep -n runtime_compat crates/nockchain-bench
```

Expected: no remaining matches after imports and callsites are renamed.

- [ ] **Step 4: Remove feature gates inside the renamed module**

In `pma_replay.rs`, remove all `#[cfg(feature = "pma-runtime-compat")]` attributes. Delete the non-PMA implementation of `copy_from_source_slab`.

Keep one unconditional implementation:

```rust
pub fn copy_from_source_slab<J, K>(dst: &mut NounSlab<J>, noun: Noun, src: &NounSlab<K>) -> Noun {
    use nockvm::noun::NounAllocator;

    let space = src.noun_space();
    dst.copy_into(noun, &space)
}
```

- [ ] **Step 5: Update `kernel_utils.rs` to call PMA replay unconditionally**

In `init_nockapp`, remove both cfg blocks. Keep the guard against `prefer_existing_checkpoint` for now unless Task 5 removes that parameter entirely.

Target shape:

```rust
pub async fn init_nockapp(
    kernel_path: &Path,
    checkpoint: Option<SaveableCheckpoint>,
    work_dir: &PathBuf,
    prefer_existing_checkpoint: bool,
    fsync: bool,
) -> Result<NockApp, KernelInitError> {
    if prefer_existing_checkpoint {
        return Err(KernelInitError::Boot(
            "prefer_existing_checkpoint replay is not supported by PMA replay".to_string(),
        ));
    }

    pma_replay::init_replay_nockapp(kernel_path, checkpoint, work_dir, fsync).await
}
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
cargo test -p nockchain-bench --release replay_pma_config
cargo test -p nockchain-bench --release checkpoint_to_load_state
cargo build -p nockchain-bench --release
```

Expected: all pass with no explicit `--features pma-runtime-compat`, because it
is now a temporary default feature.

- [ ] **Step 7: Commit**

```bash
git add crates/nockchain-bench/Cargo.toml crates/nockchain-bench/src/speed_of_light/mod.rs crates/nockchain-bench/src/speed_of_light/pma_replay.rs crates/nockchain-bench/src/speed_of_light/kernel_utils.rs crates/nockchain-bench/src/speed_of_light/extractor.rs crates/nockchain-bench/src/speed_of_light/poke.rs
git commit -m "refactor(bench): make PMA replay the default runtime"
```

## Task 3: Remove PMA Feature Gates From CLI And Case Models

**Files:**

- Modify: `crates/nockchain-bench/src/main.rs`
- Modify: `crates/nockchain-bench/src/commands/sol.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/case.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/sweep.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/provenance.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/native.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/validate.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/docker.rs`

- [ ] **Step 1: Make `BenchFsyncMode` unconditional**

In `src/main.rs`, remove `#[cfg(feature = "pma-runtime-compat")]` around:

- `BenchFsyncMode`
- `BenchFsyncMode::enabled`
- every `fsync: BenchFsyncMode` CLI field
- every destructuring/use site for `fsync`

Delete the `DEFAULT_FSYNC_ENABLED` import from `main.rs`; the CLI should pass real `fsync.enabled()` everywhere.

- [ ] **Step 2: Make option structs accept fsync unconditionally**

In `src/commands/sol.rs`, remove cfg guards from:

- `QuickReadBenchOptions::fsync`
- any construction of `PeekBenchConfig { fsync: ... }`
- quick-read CPU profiling command fsync forwarding tests

Expected behavior: quick read profiling always forwards `--fsync on|off`.

- [ ] **Step 3: Make requested-case fsync schema unconditional**

In `harness/case.rs`:

- remove `#[cfg(feature = "pma-runtime-compat")]` around `RequestedCase::fsync`
- keep `serialize_fsync_bool` and `deserialize_fsync_bool` unconditionally
- simplify `fsync_enabled()` and `set_fsync_enabled()` to direct field access

Target shape:

```rust
pub fn fsync_enabled(&self) -> bool {
    self.fsync
}

pub fn set_fsync_enabled(&mut self, enabled: bool) {
    self.fsync = enabled;
}
```

- [ ] **Step 4: Remove dual sweep deserialization**

In `harness/sweep.rs`, collapse `SweepBaseCaseSerde` into a single struct with an unconditional `fsync: bool`.

Delete:

- the `#[cfg(not(feature = "pma-runtime-compat"))]` serde struct
- `deserialize_present_fsync`
- tests that reject `fsync` without the feature

Keep/update tests:

- `sweep_base_case_sets_fsync_when_feature_enabled` -> rename to `sweep_base_case_sets_fsync`
- `sweep_base_case_defaults_fsync_on_when_field_is_missing`
- `sweep_expands_fsync_axis_when_feature_enabled` -> rename to `sweep_expands_fsync_axis`

- [ ] **Step 5: Make PMA provenance unconditional**

In `harness/provenance.rs`, remove cfg gates around:

- `PmaReplayProvenance`
- `phase2_pma_provenance`
- `pma_fsync_mode`
- tests that expect PMA fields in provenance

Delete tests that assert PMA fields are absent without the feature.

- [ ] **Step 6: Preserve Docker validation artifact schema while making PMA proof unconditional for new records**

In `harness/validate.rs`, keep these published/cross-boundary fields for
backward compatibility:

- `ValidationProbeResult::pma_runtime_compat`
- `ValidationRecord::observed_pma_runtime_compat`
- `BackendValidationOutcome::pma_replay_proven`

Do not remove these fields from serde structs in this cleanup. New validation
probes should report PMA support as true unconditionally:

```rust
pma_runtime_compat: true,
```

because current master always uses PMA replay for this crate.

Keep `observed_pma_runtime_compat: Option<bool>` so old validation records can
still be read. Keep `BackendValidationOutcome::from_validation_record` gated by
`observed_pma_runtime_compat == Some(true)` for cached/old records, but ensure
new records set it to `Some(true)`. This preserves artifact compatibility and
prevents the "feature was deleted so cfg!(feature) became false" failure mode.

In `harness/docker.rs`, update any field names or comments that mention
`pma-runtime-compat` as a feature. Keep the JSON field unless a deliberate probe
schema migration is done.

Probe version policy:

- If the field remains present and the meaning is "this binary is PMA-replay capable", do not bump `VALIDATION_PROBE_VERSION`.
- If the field is removed or renamed, bump `VALIDATION_PROBE_VERSION` and update all validation cache tests. This plan recommends keeping the field and not bumping the version.

Add or update tests that prove:

```rust
run_validation_probe(...).pma_runtime_compat == true
BackendValidationOutcome::from_validation_record(...).pma_replay_proven == true
```

for a new current-master Docker validation record.

- [ ] **Step 7: Run CLI/model tests**

Before running tests, update `crates/nockchain-bench/tests/binary_identity_build_profile.rs`:

- delete the `#[cfg(not(feature = "pma-runtime-compat"))]` gates and run the tests unconditionally, or
- delete the file only if the build-profile identity behavior is covered elsewhere.

Preferred: delete the cfgs and keep the tests.

Run:

```bash
cargo test -p nockchain-bench --release test_sol_quick_read_bench_cli_parses_fsync_modes
cargo test -p nockchain-bench --release test_sol_quick_orchestrate_cli_parses_fsync_modes
cargo test -p nockchain-bench --release sweep_base_case_sets_fsync
cargo test -p nockchain-bench --release sweep_expands_fsync_axis
cargo test -p nockchain-bench --release provenance_records_pma
cargo test -p nockchain-bench --release validation
```

If exact test names differ after renaming, use:

```bash
cargo test -p nockchain-bench --release fsync
cargo test -p nockchain-bench --release pma
```

- [ ] **Step 8: Commit**

```bash
git add crates/nockchain-bench/src/main.rs crates/nockchain-bench/src/commands/sol.rs crates/nockchain-bench/src/speed_of_light/harness/case.rs crates/nockchain-bench/src/speed_of_light/harness/sweep.rs crates/nockchain-bench/src/speed_of_light/harness/provenance.rs crates/nockchain-bench/src/speed_of_light/harness/native.rs crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs crates/nockchain-bench/src/speed_of_light/harness/validate.rs crates/nockchain-bench/src/speed_of_light/harness/docker.rs
git commit -m "refactor(bench): remove PMA feature gates from CLI and harness models"
```

## Task 4: Collapse Noun Compatibility To Current PMA APIs

**Files:**

- Modify: `crates/nockchain-bench/src/speed_of_light/noun_compat.rs`
- Modify callers only if compiler errors surface

- [ ] **Step 1: Remove fake `NounSpace`**

Delete:

```rust
#[cfg(not(feature = "pma-runtime-compat"))]
pub(crate) struct NounSpace;
```

Make `NounSpace` import unconditional:

```rust
pub(crate) use nockvm::noun::NounSpace;
use nockvm::noun::NounAllocator;
```

- [ ] **Step 2: Remove non-PMA helper branches**

For each helper in `noun_compat.rs`, keep only the PMA/current-master implementation:

- `space_for_slab`
- `decode_with_space`
- `atom_is_zero`
- `hoon_list_items`
- `hoon_map_entries`
- `noun_head`
- `noun_tail`

Example target:

```rust
pub(crate) fn decode_with_space<T: NounDecode>(
    noun: &Noun,
    space: &NounSpace,
) -> Result<T, NounDecodeError> {
    T::from_noun(noun, space)
}
```

- [ ] **Step 3: Run noun-dependent tests**

Run:

```bash
cargo test -p nockchain-bench --release mempool
cargo test -p nockchain-bench --release archive
cargo test -p nockchain-bench --release quick_orchestrate_
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add crates/nockchain-bench/src/speed_of_light/noun_compat.rs
git commit -m "refactor(bench): use current PMA noun access APIs"
```

## Task 5: Remove Legacy Checkpoint Production Or Make It Explicitly Unsupported

**Files:**

- Modify: `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/bench.rs`
- Modify: `crates/nockchain-bench/src/main.rs`
- Modify: `crates/nockchain-bench/src/commands/sol.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/case.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/harness/sweep.rs`

**Decision:** use Option B for this branch. Keep the `sol checkpoint` and
`sol fixture build` command surfaces as explicit unsupported stubs, keep
`sol fixture inspect`, and delete the hidden old-master implementation bodies.
This preserves CLI discoverability for one branch without carrying stale
non-PMA checkpoint-production logic.

- [ ] **Step 1: Replace checkpoint-production command bodies with unsupported stubs**

In `commands/sol.rs`, replace the bodies of `cmd_sol_checkpoint` and
`cmd_sol_fixture_build` with immediate unsupported errors. Do not merely guard
at the top while leaving old implementation code reachable.

Use current-master wording:

```rust
return Err("checkpoint materialization is not supported by current PMA replay; use existing .soltest fixtures or wait for PMA-native fixture generation".into());
```

Keep `sol fixture inspect` working.

- [ ] **Step 2: Delete non-PMA full checkpoint boot internals**

In `kernel_utils.rs`, remove:

- `init_full_checkpoint_nockapp`
- `bootstrap_full_checkpoint_runtime_state`
- `apply_setup_command`
- `peek_kernel_mainnet`
- `peek_genesis_seal_initialized`
- `full_checkpoint_mining_wire`
- `enable_mining_poke`
- `born_poke`
- old imports used only by these helpers: `boot`, `NockJammer`, `SystemWire`, `Wire`, `setup`, `SetupCommand`, `fakenet_blockchain_constants`, `Atom`, `D`, `NO`, `T`, `YES`, `make_tas` if no longer used

- [ ] **Step 3: Strip `CheckpointBuilder` old implementation**

In `checkpoint_builder.rs`, keep a minimal type whose `initialize()` and
`run()` return:

```rust
CheckpointBuildError::Unsupported(
    "checkpoint materialization is not supported by current PMA replay".to_string(),
)
```

Delete imports and code for:

- `init_full_checkpoint_nockapp`
- `init_nockapp`
- `select_latest_checkpoint_path`
- `peek_heaviest_chain`
- archive replay loop
- `snapshot_dir_for_mode`

- [ ] **Step 4: Remove checkpoint cadence from replay bench**

In `bench.rs`, remove or hard-disable:

- `checkpoint_every_blocks`
- checkpoint recovery timeout/tolerance fields
- `CheckpointProfile` accumulation if it only exists for checkpoint cadence
- `latest_checkpoint_size`
- `estimate_checkpoint_size`
- `ensure_checkpoint_cadence_supported`
- the inner `nockapp.save_blocking()` checkpoint cadence branch

Do not remove ordinary process memory profiling, GC inference, page-fault bursts, or phase summaries.

- [ ] **Step 5: Remove checkpoint cadence from execution, but keep artifact fields backward-compatible**

In `main.rs`, `commands/sol.rs`, `harness/case.rs`, and `harness/sweep.rs`, remove operational CLI/sweep-axis support for:

- `--checkpoint-every-blocks`
- `--enable-checkpointing` if it has no remaining PMA meaning
- checkpoint recovery tuning flags if they no longer affect anything

Do not remove `RequestedCase::checkpoint_every_blocks` or
`RequestedCase::enable_checkpointing` from serde structs in this branch unless
bench_pages compatibility is proven. Keep them with defaults and continue
writing `0` / `false` in new artifacts if that is the least disruptive schema.

Remove or reject the corresponding sweep axes so new matrices cannot vary
checkpoint cadence. Keep old artifact deserialization working.

Before deleting any field from newly emitted artifacts, run the renderer test
suite in the checkout that owns `scripts/bench_pages`. On this machine that is
the main checkout, not necessarily this release-candidate worktree:

```bash
cd /shared/nockchain
uv run --project scripts/bench_pages pytest
```

If renderer tolerance is not proven, keep the fields.

- [ ] **Step 6: Update tests**

Delete or rewrite tests that exist only for removed checkpoint cadence or old checkpoint generation. Keep tests that inspect existing checkpoint fixtures.

Run:

```bash
cargo test -p nockchain-bench --release checkpoint
cargo test -p nockchain-bench --release fixture
cargo test -p nockchain-bench --release bench_config_default_profile_values
```

Expected:

- fixture inspection tests pass
- removed command tests are gone, or unsupported-command tests pass
- no test depends on non-PMA checkpoint materialization

- [ ] **Step 7: Commit**

```bash
git add crates/nockchain-bench/src/speed_of_light/kernel_utils.rs crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs crates/nockchain-bench/src/speed_of_light/bench.rs crates/nockchain-bench/src/main.rs crates/nockchain-bench/src/commands/sol.rs crates/nockchain-bench/src/speed_of_light/harness/case.rs crates/nockchain-bench/src/speed_of_light/harness/sweep.rs
git commit -m "refactor(bench): remove old-master checkpoint production paths"
```

## Task 6: Make Cold-Peek PMA Targeting Current-Master Native

**Files:**

- Modify: `crates/nockchain-bench/src/speed_of_light/cold_peek/cgroup.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/cold_peek/mod.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/orchestrator.rs`

- [ ] **Step 1: Remove cfg-gated PMA cold target code**

In `cold_peek/cgroup.rs`, remove feature cfgs around:

- `read_pma_vmas`
- `ColdTarget::pma_replay`
- `ColdTarget::pma_replay_nockstack`
- `ColdTargetComponent::pma_replay`
- `startup_reclaim_swappinesses`
- `bind_target_after_boot`
- `ColdTargetSelection`
- `cold_target_selection`

Delete the non-PMA `startup_reclaim_swappinesses` and `bind_target_after_boot` implementations.

- [ ] **Step 2: Keep or remove `nockstack` diagnostic override**

If keeping it, document it as a diagnostic override only:

```text
NOCKCHAIN_BENCH_COLD_TARGET=nockstack
```

Do not describe it as the default or old-master fallback.

- [ ] **Step 3: Update orchestrator wording**

In `orchestrator.rs`, replace messages like:

```text
requires --features pma-runtime-compat
```

with current-master wording, for example:

```text
requires PMA replay cold-runtime support
```

Update ignored test messages:

```text
checkpoint-backed cold-peek smoke; run from transplanted PMA checkout
```

to:

```text
checkpoint-backed cold-peek smoke; requires local checkpoint/cgroup setup
```

- [ ] **Step 4: Run cold-peek tests**

Run:

```bash
cargo test -p nockchain-bench --release cold_init_
cargo test -p nockchain-bench --release force_cold_
cargo test -p nockchain-bench --release quick_orchestrate_
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add crates/nockchain-bench/src/speed_of_light/cold_peek crates/nockchain-bench/src/speed_of_light/orchestrator.rs
git commit -m "refactor(bench): make cold peek PMA targeting unconditional"
```

## Task 7: Rewrite Docs And Specs Around Current Master

**Files:**

- Modify: `crates/nockchain-bench/README.md`
- Modify: `crates/nockchain-bench/specs/bench-harness-spec.md`
- Delete or archive: `crates/nockchain-bench/BENCH_HARNESS_SPEC_v5.md`
- Delete or archive: `crates/nockchain-bench/BENCH_HARNESS_SPEC_v6.md`
- Delete or archive: `crates/nockchain-bench/BENCH_HARNESS_SPEC_v7.md`

- [ ] **Step 1: Remove transplant and dual-branch language**

In `README.md`, delete sections that describe:

- "final compatibility line"
- "current master-style runtime and PMA runtime compatibility transplant"
- `scripts/bench_sync`
- PMA checkout sync/build/test
- "PMA-only branch later"
- keeping code compiling without `pma-runtime-compat`

- [ ] **Step 2: Replace build docs**

Use:

```markdown
## Build

Use release builds for benchmark work:

```bash
cargo build -p nockchain-bench --release
./target/release/nockchain-bench sol --help
```

PMA replay is the normal runtime path on current master. `--fsync on|off`
controls PMA durability behavior where exposed by the command.
```
```

- [ ] **Step 3: Replace workflow docs**

Keep docs for:

- `sol quick-bench`
- `sol quick-read-bench`
- `sol quick-orchestrate`
- `sol bench`
- `sol sweep`
- Docker image build
- bench_pages publish

Update docs for:

- `sol checkpoint`
- `sol fixture build`

according to the Task 5 decision.

- [ ] **Step 4: Make one canonical spec**

Make `crates/nockchain-bench/specs/bench-harness-spec.md` the current canonical spec.

Remove compatibility addenda:

- "final master/PMA-compatible branch"
- transplant instructions
- "master-style and PMA-compatible builds"

If keeping historical specs, move them under an archive directory and make clear they are not current implementation guidance.

- [ ] **Step 5: Run doc greps**

Run:

```bash
rg 'pma-runtime-compat|runtime_compat|transplant|PMA checkout|master-style|PMA-compatible|PMA-only branch|bench_sync' crates/nockchain-bench
```

Also run the same grep over adjacent bench tooling that may still contain
operator-facing wording:

```bash
rg 'pma-runtime-compat|runtime_compat|transplant|PMA checkout|master-style|PMA-compatible|PMA-only branch|bench_sync' scripts docker crates/nockchain-bench
```

Expected: no matches, except possibly archived historical docs if the team chooses to keep them.

- [ ] **Step 6: Commit**

```bash
git add crates/nockchain-bench/README.md crates/nockchain-bench/specs/bench-harness-spec.md crates/nockchain-bench/BENCH_HARNESS_SPEC_v5.md crates/nockchain-bench/BENCH_HARNESS_SPEC_v6.md crates/nockchain-bench/BENCH_HARNESS_SPEC_v7.md
git commit -m "docs(bench): describe current master PMA runtime"
```

## Task 8: Final Audit And Smoke Verification

**Files:**

- Modify: `crates/nockchain-bench/Cargo.toml`
- Modify: other files only if fixing issues found by verification

- [ ] **Step 1: Verify all PMA feature cfgs are gone**

Run:

```bash
rg '#\\[cfg.*pma-runtime-compat|cfg!\\(feature = "pma-runtime-compat"\\)' crates/nockchain-bench
```

Expected: no matches. Do not continue until this is true.

- [ ] **Step 2: Delete the transitional Cargo feature**

Only after Step 1 passes, delete the entire feature section from
`crates/nockchain-bench/Cargo.toml`:

```toml
[features]
default = ["pma-runtime-compat"]
pma-runtime-compat = []
```

Run:

```bash
cargo metadata --no-deps
```

Expected: succeeds and does not mention `pma-runtime-compat` for
`nockchain-bench`.

- [ ] **Step 3: Audit for removed compatibility terms**

Run:

```bash
rg 'pma-runtime-compat|runtime_compat|transplant|PMA checkout|master-style|PMA-compatible|PMA-only branch|not\\(feature = "pma-runtime-compat"\\)' crates/nockchain-bench Cargo.toml
```

Expected: no matches.

- [ ] **Step 4: Full bench crate test**

Run:

```bash
cargo test -p nockchain-bench --release
```

Expected: pass.

- [ ] **Step 5: nockapp replay helper test**

Run:

```bash
cargo test -p nockapp --release for_replay
```

Expected: pass.

- [ ] **Step 6: Release build**

Run:

```bash
cargo build -p nockchain-bench --release
```

Expected: pass.

- [ ] **Step 7: Native quick smoke**

Run:

```bash
target/release/nockchain-bench sol quick-bench \
  --fixture /shared/nockchain/fixtures/first-100-v2-derived-checkpoint-no-mempool.soltest \
  --blocks 10 \
  --fsync on
```

Expected: 10 blocks poked, 0 failed pokes.

- [ ] **Step 8: Docker trusted smoke**

Build a fresh image:

```bash
scripts/build_nockchain_bench_image.sh \
  --variant standard \
  --tag nockchain-bench:pma-master-fit \
  --binary /shared/nockchain/.worktrees/nockchain-bench-release-candidate/target/release/nockchain-bench \
  --skip-cargo-build
```

Run a short trusted Docker bench:

```bash
mkdir -p /home/drbeefsupreme/nockchain-bench-pma-master-fit-smoke
DOCKER_HOST=unix:///home/drbeefsupreme/.docker/desktop/docker.sock \
target/release/nockchain-bench sol bench \
  --fixture /shared/nockchain/fixtures/first-100-v2-derived-checkpoint-no-mempool.soltest \
  --output /home/drbeefsupreme/nockchain-bench-pma-master-fit-smoke \
  --docker-image nockchain-bench:pma-master-fit \
  --memory-limit 8g \
  --work-dir-mode docker-tmpfs \
  --blocks 10 \
  --warmup-runs 0 \
  --measured-runs 3 \
  --cooldown-secs 0
```

Expected: verdict `Valid`.

- [ ] **Step 9: Fixture-axis sweep smoke**

Run a current-master Docker sweep over v0/v1/v2 derived fixtures:

```bash
DOCKER_HOST=unix:///home/drbeefsupreme/.docker/desktop/docker.sock \
target/release/nockchain-bench sol sweep \
  --matrix tmp/rc-sweeps/docker-derived-v0-v1-v2-100-matrix.json \
  --output /home/drbeefsupreme/nockchain-bench-pma-master-fit-v0-v1-v2 \
  --comparison-markdown
```

Expected: verdict `Valid`; v0/v1/v2 all complete.

- [ ] **Step 10: Commit final fixes**

Commit the feature deletion and any verification fixes:

```bash
git add crates/nockchain-bench/Cargo.toml <fixed-files>
git commit -m "fix(bench): finish PMA master cleanup"
```

## Risks

- Removing `pma-runtime-compat` will break any downstream script that still builds with `--features pma-runtime-compat`. That is intentional if current master is the only target, but release notes should call it out.
- Removing `pma-runtime-compat` before all cfg sites are gone will produce a broken intermediate commit. Keep it as a default feature until the final audit.
- Deleting old checkpoint-production code may remove a local utility even though it is not valid under PMA replay. Claude should verify whether operators still need `sol fixture build` in this branch.
- Artifact schema cleanup can break bench_pages if fields are removed too aggressively. Preserve Docker validation PMA fields and checkpoint-cadence requested-case fields unless renderer compatibility is proven.
- Docker Desktop shared-path behavior is environment-specific. Use `/home/drbeefsupreme/...` for local Docker trusted smokes on this machine unless Docker file sharing is updated.

## Definition Of Done

- `rg 'pma-runtime-compat|runtime_compat|transplant|PMA checkout|master-style|PMA-compatible|PMA-only branch' crates/nockchain-bench Cargo.toml` has no non-archival matches.
- `cargo test -p nockchain-bench --release` passes.
- `cargo test -p nockapp --release for_replay` passes.
- `cargo build -p nockchain-bench --release` passes.
- Native quick smoke passes with `--fsync on`.
- Trusted Docker smoke passes with a freshly built image.
- README describes current master directly.
- The branch no longer needs `scripts/bench_sync` or a transplanted PMA checkout for acceptance.
