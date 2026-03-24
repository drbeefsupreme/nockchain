# PMA Bench Compatibility Implementation Spec

> **For implementation:** Land `nockchain-bench sol quick-bench` first, then a PMA-enabled direct-replay quick-bench phase, then checkpoint materialization and fixture generation, then trusted harness refinements.

**Goal:** Make `nockchain-bench` compatible with the PMA runtime with the smallest practical blast radius.

**Design center:** Keep `.soltest` v4, `.solarch`, and the trusted harness artifact model intact. Add a narrow compatibility seam only at the `nockapp` API boundaries that diverged between `master` and the PMA branch. Prefer compile-time feature gating so the master path keeps its current behavior and replay hot path. When PMA later needs a materialized checkpoint, normalize PMA state back into legacy `.chkjam` bytes instead of introducing a new fixture format.

## 1. Executive Summary

- Phase 1 remains narrowly scoped to making `nockchain-bench sol quick-bench` run on the PMA branch from an existing legacy `.soltest` fixture.
- Phase 1 is PMA-branch compatible, but not PMA-backed: the PMA replay helper still passes `pma: None`.
- New Phase 1.5 upgrades the same direct replay path to PMA-enabled replay by passing `pma: Some(PmaConfig)` with fresh bench-owned PMA files, while still avoiding `boot::setup()`, event-log replay, and snapshot restore.
- `.soltest` v4 stays unchanged. Its payload remains `checkpoint_bytes + archive_bytes + kernel_bytes`.
- `.solarch` stays unchanged.
- The trusted harness and orchestrator stay structurally unchanged. `resolved_case.json`, `provenance.json`, and `summary.json` remain bench-owned artifacts.
- Phase 2 adds checkpoint materialization by normalizing PMA runtime state back into legacy `.chkjam`: minimal `NockApp` checkpoint hook -> `SaveableCheckpoint` -> `to_jammed_checkpoint::<NockJammer>()` -> `encode()` -> write file.
- Phase 2B, not Phase 1.5, is still the first place the spec enables full PMA boot semantics via `boot::setup()`, `data_dir`, `event_log_path`, snapshot policy, and PMA boot-source selection.
- Phase 3 adds PMA-aware trusted provenance fields additively while preserving schema compatibility and harness ownership of final artifacts.

## 2. Scope And Constraints

### Phase 1 in scope

- Compile `nockchain-bench` against the PMA runtime branch with a dedicated feature gate.
- Make quick-bench replay execute on PMA from existing `.soltest` fixtures without changing fixture format.
- Keep master builds unchanged when the PMA feature is not enabled.
- Defer PMA-backed replay, checkpoint materialization, full-checkpoint boot, trusted provenance changes, and Docker-specific PMA polish unless they are required to keep the crate compiling cleanly.

### Phase 1.5 in scope

- Keep the same quick-bench replay shape and fixture model as Phase 1.
- Switch PMA replay initialization from `pma: None` to fresh direct `pma: Some(PmaConfig)`.
- Do this without `boot::setup()`, without PMA boot-source selection, and without new fixture formats.

### Later phases in scope

- Phase 2: checkpoint materialization back to legacy `.chkjam`, then derived fixture generation, then full-checkpoint fixture generation.
- Phase 3: trusted native bench, trusted Docker bench, and additive provenance/schema refinements.

### Hard preservation constraints

- Keep `.soltest` layout version 4 intact.
- Keep `.solarch` intact.
- Keep the current bench harness/orchestrator contract intact.
- Keep the harness as the owner of `resolved_case.json`, `provenance.json`, `summary.json`, and related trusted outputs.
- Keep master-path behavior and performance as close to unchanged as practical.
- Prefer compile-time separation over runtime branching for PMA/master API-shape differences.

### Explicit non-goals

- No `.soltest` v5 or PMA-native fixture format.
- No broad runtime-enum abstraction threaded through `RequestedCase`, `ResolvedCase`, fixture manifests, or harness backends.
- No subprocess-wrapper design around `bench_nockchain_checkpoint_block` or another PMA helper binary.
- No trusted-artifact rewrite.
- No code implementation in this task.

## 3. Ground-Truth Code Findings

### Observed facts

- `.soltest` v4 is checkpoint-first. `SolFixtureFile` stores `checkpoint_bytes`, `archive_bytes`, and `kernel_bytes`, and extraction writes them back out as `fixture.chkjam`, `fixture.solarch`, and `fixture.jam`. See `crates/nockchain-bench/src/speed_of_light/fixture.rs:17-29`, `crates/nockchain-bench/src/speed_of_light/fixture.rs:54-60`, `crates/nockchain-bench/src/speed_of_light/fixture.rs:136-147`, and `crates/nockchain-bench/src/speed_of_light/harness/execute.rs:165-173`.
- Quick-bench already runs through the shared once-run engine. `cmd_sol_quick_bench()` calls `execute_once_with_options()`, which extracts the fixture and constructs `SolBenchRunner`. See `crates/nockchain-bench/src/commands/sol.rs:161-315` and `crates/nockchain-bench/src/speed_of_light/harness/execute.rs:146-196`.
- The current bench-local runtime init seam is concentrated in `kernel_utils.rs`. `init_nockapp()` owns the direct `NockApp::new(...)` and `Kernel::load_with_hot_state_medium(...)` call, and `init_full_checkpoint_nockapp()` owns the boot-path initialization. See `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs:61-98` and `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs:100-130`.
- Current bench code uses the old `NounSlab::copy_into(noun)` signature in only two bench files: `poke.rs` and `extractor.rs`. See `crates/nockchain-bench/src/speed_of_light/poke.rs:42-56` and `crates/nockchain-bench/src/speed_of_light/extractor.rs:317-332`.
- Checkpoint materialization today still depends on `save_blocking()` plus reading the latest `.chkjam` from disk. `CheckpointBuilder::run()` does this directly. See `crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs:217-224`.
- On the PMA branch, `Kernel::load_with_hot_state_medium` takes `pma: Option<PmaConfig>`. Compare `master:crates/nockapp/src/kernel/form.rs:606-612` with `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/kernel/form.rs:1261-1268`.
- The PMA helper binary already demonstrates direct replay with `Kernel::load_with_hot_state_medium(..., None)` plus PMA-shaped `NockApp::new(move |_| async move { Ok(kernel) })`. See `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockchain/src/bin/bench_nockchain_checkpoint_block.rs:339-347` and `...:367-410`.
- PMA direct kernel load also supports `Some(PmaConfig)` outside `boot::setup()`. The PMA branch’s own tests construct `PmaConfig` directly and pass it into `Kernel::load_with_hot_state_{tiny,small,medium,...}` with `open_existing: false`, `create_snapshots: false`, `rotating_snapshot_interval_event_time: None`, `restore_manifest: None`, and `gc_interval: None`. See `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/nockapp/test.rs:138-159` and `...:195-216`.
- `PmaConfig` is concrete and bench-constructible: `{ path_0, path_1, words, open_existing, create_snapshots, rotating_snapshot_interval_event_time, restore_manifest, gc_interval }`. See `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/kernel/form.rs:65-74`.
- Full PMA boot semantics live in `boot::setup()`, which creates `data_dir`, `pma/`, `checkpoints/`, `event-log.sqlite3`, then constructs `PmaConfig` and boot-source selection inside the kernel closure. See `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/kernel/boot.rs:1578-1592` and `...:1663-1690`.
- PMA boot priority is explicit: valid PMA first, then verified snapshot plus replay, then checkpoint bootstrap, then fresh boot. See `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/kernel/boot.rs:1125-1355`.
- `save_blocking()` exists on `master` but is absent on PMA. See `master:crates/nockapp/src/nockapp/mod.rs:315-318`.
- PMA still exposes `Kernel::checkpoint()` and `SaveableCheckpoint::to_jammed_checkpoint::<J>()`. See `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/kernel/form.rs:1418-1420` and `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/nockapp/save.rs:74-90`.
- `to_jammed_checkpoint` has different signatures on each branch. On master it takes `(self, metrics: Arc<NockAppMetrics>)`; on PMA it takes only `(self)`.
- `NockApp.kernel` is `pub(crate)` on PMA. External crates like `nockchain-bench` cannot call `self.kernel.checkpoint()` directly.
- `ExportedState` only captures `LoadState { ker_hash, event_num, kernel_state }`; it does not include `cold`.
- `init_nockapp()` currently takes `prefer_existing_checkpoint: bool`. Quick-bench and checkpoint builder pass `false`; extractor passes `true`. See `crates/nockchain-bench/src/speed_of_light/bench.rs:256-261`, `crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs:125-130`, and `crates/nockchain-bench/src/speed_of_light/extractor.rs:175-180`.

### Inferences

- Phase 1 does not need PMA-backed replay if the goal is strictly “make quick-bench run on the PMA branch.”
- Phase 1.5 is feasible without `boot::setup()` because `Kernel::load_with_hot_state_medium(..., Some(PmaConfig))` is already supported directly on the PMA branch.
- The direct-replay PMA path and the full PMA boot path should remain separate phases because they solve different problems:
  - direct replay with fresh PMA files does not introduce boot-source ambiguity
  - full boot via `boot::setup()` does
- Because the entire `nockchain-bench` binary is compiled in Phase 1 and 1.5, the spec must preserve or explicitly feature-gate the existing `init_nockapp(..., prefer_existing_checkpoint)` call surface. Narrowing only the low-level replay helper is not enough by itself.

## 4. Proposed Design

### 4.1 Architecture And Module Seam

Add one bench-local compatibility module under `crates/nockchain-bench/src/speed_of_light/`:

- New: `runtime_compat.rs`

The seam owns only branch-shape differences:

- constructing a replay `NockApp`
- constructing a PMA-enabled replay `NockApp`
- full-checkpoint boot setup
- copying nouns between slabs when PMA requires `NounSpace`
- later, materializing a legacy `.chkjam` from PMA runtime state

It does not own archive iteration, fixture format, harness orchestration, case resolution, summary math, or Docker policy.

### 4.2 Compile-Time Feature Strategy

Add a bench-local Cargo feature:

- `pma-runtime-compat`

Rules:

- Master and `nockchain-bench-mega-pr` builds use the default build with the feature disabled.
- PMA branch builds enable `--features pma-runtime-compat`.
- The feature only changes bench code shape. It is not threaded into `.soltest`, `.solarch`, `RequestedCase`, or `ResolvedCase`.
- Use `#[cfg(feature = "pma-runtime-compat")]` and `#[cfg(not(feature = "pma-runtime-compat"))]` at function or module granularity so the master replay hot path does not pay runtime branches.

### 4.3 Exact Helper And Wrapper Shape

Introduce one low-level replay helper, staged internally across phases, and keep one wrapper stable:

- `runtime_compat::init_replay_nockapp(kernel_path, checkpoint, work_dir) -> Result<NockApp, KernelInitError>`
  - Master implementation: current replay construction with caller-supplied checkpoint semantics.
  - PMA Phase 1 implementation:
    - build the kernel with `Kernel::load_with_hot_state_medium(..., None)`
    - wrap it in PMA-shaped `NockApp::new(move |_metrics| async move { Ok(kernel) })`
  - PMA Phase 1.5 implementation:
    - create a fresh bench-owned PMA replay directory under `work_dir`, for example `work_dir/replay-pma/`
    - use `path_0 = replay-pma/0.pma`, `path_1 = replay-pma/1.pma`
    - construct `PmaConfig` with:
      - `words = nockapp::utils::NOCK_STACK_SIZE_MEDIUM` if publicly importable; otherwise use a bench-local mirror only after verifying the constant matches PMA medium-stack expectations
      - `open_existing = false`
      - `create_snapshots = false`
      - `rotating_snapshot_interval_event_time = None`
      - `restore_manifest = None`
      - `gc_interval = None`
    - build the kernel with `Kernel::load_with_hot_state_medium(..., Some(pma_config))`
    - still wrap it in PMA-shaped `NockApp::new(move |_metrics| async move { Ok(kernel) })`
  - No `boot::setup()` call in either Phase 1 or Phase 1.5.

- Existing public wrapper to preserve in `kernel_utils.rs`:
  - `init_nockapp(kernel_path, checkpoint, work_dir, prefer_existing_checkpoint) -> Result<NockApp, KernelInitError>`
  - Keep the four-argument signature in Phase 1 and 1.5 so existing bench callers still compile.
  - Master implementation: preserve current behavior, including `prefer_existing_checkpoint`.
  - PMA implementation in both Phase 1 and Phase 1.5:
    - if `prefer_existing_checkpoint == false`, delegate to `runtime_compat::init_replay_nockapp(...)`
    - if `prefer_existing_checkpoint == true`, return an explicit unsupported error such as `KernelInitError::Boot("prefer_existing_checkpoint replay is not yet supported under pma-runtime-compat")`

- `runtime_compat::copy_from_source_slab(dst, noun, src) -> Noun`
  - Master implementation: `dst.copy_into(noun)`.
  - PMA implementation: `dst.copy_into(noun, &src.noun_space())`.
  - PMA compilation requires `use nockvm::noun::NounAllocator;`.

- `runtime_compat::init_full_checkpoint_nockapp(kernel_path, work_dir) -> Result<NockApp, KernelInitError>`
  - Master implementation: current `boot::setup()` plus bootstrap pokes.
  - PMA Phase 1 and 1.5 implementation: explicit unsupported error.
  - PMA Phase 2B implementation: real `boot::setup()` path with fresh data dir.

- `runtime_compat::materialize_legacy_chkjam(nockapp, output_path, work_dir) -> Result<(), CheckpointMaterializationError>`
  - Master implementation: current `save_blocking()` plus file-copy behavior.
  - PMA Phase 2 implementation: use the minimal runtime checkpoint hook to obtain `SaveableCheckpoint`, convert with `to_jammed_checkpoint::<NockJammer>()`, then encode and write a real legacy `.chkjam`.

### 4.4 File-By-File Change List

#### Phase 1: PMA-branch-compatible quick-bench

Files that should change:

- `crates/nockchain-bench/Cargo.toml`
  - Add `[features] pma-runtime-compat = []`.

- `crates/nockchain-bench/src/speed_of_light/mod.rs`
  - Add private `runtime_compat` module wiring.

- `crates/nockchain-bench/src/speed_of_light/runtime_compat.rs`
  - New compile-time compatibility helpers for replay init and slab copying.

- `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs`
  - Keep `init_nockapp(...)` signature stable.
  - Route `prefer_existing_checkpoint = false` through `runtime_compat::init_replay_nockapp()`.
  - Under `pma-runtime-compat`, return an explicit unsupported error when `prefer_existing_checkpoint = true`.
  - Feature-gate the entire body of `init_full_checkpoint_nockapp()`.

- `crates/nockchain-bench/src/speed_of_light/poke.rs`
  - Replace direct `copy_into(page)` with the compatibility helper.

- `crates/nockchain-bench/src/speed_of_light/extractor.rs`
  - Replace direct `copy_into(entry_noun)` with the compatibility helper.
  - Do not change the extractor init call surface in Phase 1.

- `crates/nockchain-bench/src/speed_of_light/bench.rs`
  - Keep using `kernel_utils::init_nockapp(...)`.
  - Add an explicit PMA unsupported error for `checkpoint_every_blocks > 0`.

- `crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs`
  - Add an explicit PMA unsupported error for Phase-2-only paths.

#### Phase 1.5: PMA-enabled direct replay quick-bench

Files that should change:

- `crates/nockchain-bench/src/speed_of_light/runtime_compat.rs`
  - Upgrade the PMA replay helper from `pma: None` to fresh direct `pma: Some(PmaConfig)`.
  - Create fresh PMA files under the bench work dir and never reuse stale `.pma` slabs.

- `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs`
  - Keep the same wrapper shape; only the PMA `prefer_existing_checkpoint = false` implementation changes underneath.

- `crates/nockchain-bench/src/speed_of_light/bench.rs`
  - Keep Phase-1 unsupported checkpoint cadence behavior. Phase 1.5 is still replay-only.

Files that should not change in Phase 1.5:

- fixture format files
- harness/orchestrator structure
- provenance schema
- checkpoint builder save/materialization logic

#### Phase 2: checkpoint materialization and fixture support

Phase 2A files that should change:

- `crates/nockapp/src/nockapp/mod.rs`
  - Add the smallest needed runtime checkpoint hook if bench cannot otherwise capture a checkpoint because `kernel` is private.

- `crates/nockchain-bench/src/speed_of_light/runtime_compat.rs`
  - Implement PMA legacy checkpoint materialization on top of the now-PMA-enabled direct replay path.

- `crates/nockchain-bench/src/speed_of_light/bench.rs`
  - Replace the Phase 1/1.5 unsupported checkpoint-cadence path with real PMA `.chkjam` materialization.

- `crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs`
  - Replace the Phase 1/1.5 unsupported save path with real PMA `.chkjam` materialization.

Phase 2B files that should change:

- `crates/nockchain-bench/src/speed_of_light/runtime_compat.rs`
  - Implement PMA full-checkpoint boot setup via `boot::setup()`.

- `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs`
  - Switch PMA `init_full_checkpoint_nockapp()` from unsupported to real implementation.

No file-format changes:

- `crates/nockchain-bench/src/speed_of_light/fixture.rs` stays structurally unchanged.
- `crates/nockchain-bench/src/speed_of_light/checkpoint.rs` stays the reader/metadata layer for legacy `.chkjam`.

#### Phase 3: trusted harness and provenance refinements

Files that should change:

- `crates/nockchain-bench/src/speed_of_light/harness/provenance.rs`
- `crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs`
- `crates/nockchain-bench/src/speed_of_light/harness/native.rs`
- `crates/nockchain-bench/src/speed_of_light/harness/docker.rs`
- `crates/nockchain-bench/src/speed_of_light/harness/sweep.rs`
- `crates/nockchain-bench/README.md`
- `crates/nockchain-bench/specs/bench-harness-spec.md`

### 4.5 Phase 1 Design: PMA-Branch-Compatible Quick-Bench

Phase 1 is the initial compatibility milestone only.

Replay path:

1. Read the embedded legacy `.chkjam` from the fixture.
2. Decode it into `LoadedCheckpoint`.
3. Rebuild `SaveableCheckpoint { ker_hash, event_num, state, cold }`.
4. Build the kernel with `Kernel::load_with_hot_state_medium(..., None)`.
5. Wrap it in PMA-shaped `NockApp::new(move |_metrics| async move { Ok(kernel) })`.
6. Replay archive pokes exactly as today.

Unsupported in Phase 1:

- `--checkpoint-every-blocks > 0`
- `sol checkpoint`
- `sol fixture build`
- any path that requires `prefer_existing_checkpoint = true`, including current `sol extract`

### 4.6 Phase 1.5 Design: PMA-Enabled Direct Replay

Phase 1.5 keeps the same user-visible quick-bench deliverable but changes the PMA replay helper from compatibility-only to PMA-backed direct replay.

Replay path:

1. Keep the Phase 1 direct replay flow from legacy `.chkjam`.
2. Before kernel load, create a fresh PMA replay directory under the bench work dir.
3. Construct `PmaConfig` with fresh `0.pma` and `1.pma` paths, medium-stack-sized `words`, `open_existing = false`, `create_snapshots = false`, `rotating_snapshot_interval_event_time = None`, `restore_manifest = None`, and `gc_interval = None`.
4. Build the kernel with `Kernel::load_with_hot_state_medium(..., Some(pma_config))`.
5. Still avoid `boot::setup()`, `data_dir`, `event_log_path`, and snapshot restore.
6. Replay archive pokes exactly as in Phase 1.

Why this phase exists:

- It enables PMA-backed replay and PMA-relevant memory measurements earlier.
- It still avoids the larger blast radius of full PMA boot semantics.
- It provides a cleaner foundation for Phase 2A materialization from PMA-backed state.

Unsupported in Phase 1.5:

- everything Phase 1 still forbids
- full boot-source selection and provenance around PMA vs snapshot vs checkpoint bootstrap
- PMA snapshot creation, PMA event-log replay, and checkpoint materialization

### 4.7 Phase 2 Design: Checkpoint And Fixture Support

#### Phase 2A: derived checkpoint materialization

Implement PMA `.chkjam` output without changing bench artifacts:

1. Add the smallest runtime hook needed to get a `SaveableCheckpoint` out of `NockApp`.
2. In bench code, materialize legacy checkpoint bytes as:
   - `checkpoint = nockapp.checkpoint().await?`
   - `jammed = checkpoint.to_jammed_checkpoint::<NockJammer>()` on PMA
   - `bytes = jammed.encode()?`
   - `std::fs::write(output_path, bytes)?`
3. Use that path for:
   - `sol checkpoint`
   - fixture build with `checkpoint_kind = derived`
   - replay checkpoint cadence in `SolBenchRunner`

Under `pma-runtime-compat`, Phase 2A should operate on the Phase 1.5 PMA-enabled direct replay state, not revert to the earlier `pma: None` helper.

#### Phase 2B: full checkpoint boot support

Only full checkpoint generation needs full PMA boot semantics.

Implementation rules:

- `init_full_checkpoint_nockapp()` on PMA must build a fresh PMA data dir under the bench work dir.
- The PMA boot CLI should set:
  - `new = true`
  - `gc_interval = Some(0)`
  - `rotating_snapshot_interval_event_time = None`
  - `data_dir = Some(runtime_data_dir)`
  - `event_log_path = Some(runtime_data_dir.join("event-log.sqlite3"))`
  - `disable_fsync = false` unless a later explicitly local benchmark mode decides otherwise
- The data dir must be created fresh per invocation; do not reuse a dirty directory.
- After boot plus existing bootstrap pokes plus archive replay, materialize a legacy `.chkjam` through the same Phase 2A normalization path.

### 4.8 Phase 3 Design: Trusted Native And Docker Bench

The harness structure remains unchanged:

- `ResolvedCase` stays the resolved request plus fixture identity.
- `execute_once()` stays the shared once-run engine.
- `execute_trusted_run()` stays the shared orchestrator.
- `artifacts.rs` keeps writing all final trusted artifacts.

Phase 3 work is PMA-awareness, not a harness rewrite:

- mark trusted runs with `runtime_flavor`
- keep direct replay trusted runs reporting `boot_source = "checkpoint"` even when PMA-backed replay is enabled internally
- preserve the current summary and verdict pipeline
- ensure Docker provenance keeps bench as the source of truth instead of delegating to external helper output

## 5. PMA Runtime Hook

### Required for Phase 1?

No.

### Required for Phase 1.5?

No.

Phase 1.5 enables PMA-backed replay through existing `PmaConfig` kernel-load support. It still does not need checkpoint materialization.

### Required for later phases?

Yes, for any PMA path that must emit a real `.chkjam`.

### Smallest viable API

Recommended:

- `NockApp::checkpoint() -> Result<SaveableCheckpoint, NockAppError>`

Fallback if exposing `SaveableCheckpoint` is undesirable:

- `NockApp::checkpoint_to_jam<J: Jammer>() -> Result<JammedCheckpointV2, NockAppError>`

### Why `export()` is insufficient

`ExportedState` omits `cold`, so it cannot preserve current legacy checkpoint semantics or produce a complete legacy `.chkjam` payload.

### Exact normalization path

Use this PMA normalization path in Phase 2:

- `NockApp::checkpoint()` -> `SaveableCheckpoint`
- `SaveableCheckpoint::to_jammed_checkpoint::<NockJammer>()`
- `JammedCheckpointV2::encode()`
- write bytes to `*.chkjam`

Do not introduce a PMA-native bench fixture format.

## 6. Provenance And Schema Handling

### Phase 1 and Phase 1.5

- No trusted provenance change is required.
- `sol quick-bench` is not a trusted artifact path and should remain so.
- `resolved_case.json`, `summary.json`, and `provenance.json` stay unchanged in these phases.

### Phase 2

- Derived checkpoint generation and fixture generation do not require a trusted provenance schema change.
- The fixture format stays unchanged because the output is still legacy `.chkjam` bytes embedded into `.soltest`.

### Phase 3

Keep `schema_version = "1"` and extend `Provenance` additively with optional fields:

- `runtime_flavor: Option<String>` with values `"legacy"` or `"pma"`
- `boot_source: Option<String>` with values `"checkpoint"`, `"pma"`, `"snapshot"`, `"fresh"`
- `boot_event_num: Option<u64>`
- `boot_snapshot_manifest_path: Option<PathBuf>`

Direct PMA-backed replay in Phase 1.5 and later trusted replay runs should still report `boot_source = "checkpoint"` because the source of runtime state is the extracted legacy checkpoint, not PMA boot-source selection.

## 7. Checkpoint Profiling Semantics

Phase 1 and Phase 1.5:

- not required
- PMA quick-bench should forbid replay checkpoint cadence rather than pretend PMA durability equals legacy `save_blocking()`

Phase 2 and Phase 3 decision:

- preserve current checkpoint timing summary fields only for explicit bench-triggered legacy `.chkjam` materialization
- do not reinterpret those fields as total PMA durability cost
- if PMA-specific raw timing evidence is later useful, add it as raw evidence or optional profile payloads rather than changing the meaning of `checkpoint_count` and `average_checkpoint_time_secs`

## 8. Verification Plan

All verification uses release assumptions.

### Phase 1 gate checks

1. Master compile still works:

   ```bash
   cargo build -p nockchain-bench --release
   ```

2. PMA compile works with the feature:

   ```bash
   cargo build -p nockchain-bench --release --features pma-runtime-compat
   ```

3. PMA quick-bench native smoke with `pma: None`:

   ```bash
   /shared/nockchain/target/release/nockchain-bench sol quick-bench \
     --fixture /shared/nockchain/fixtures/first-100-v2-derived-checkpoint-no-mempool.soltest \
     --blocks 100 \
     --checkpoint-every-blocks 0
   ```

   Success observations:
   - command exits `0`
   - the summary prints `Blocks poked:` with a value greater than `0`
   - no runtime-compat unsupported error is emitted
   - no PMA API-shape panic occurs during init or poke replay

4. Phase-1 out-of-scope PMA paths fail clearly:

   - `sol checkpoint`
   - `sol fixture build`
   - current `sol extract`
   - quick-bench with `--checkpoint-every-blocks > 0`

### Phase 1.5 gate checks

5. PMA quick-bench native smoke with direct `Some(PmaConfig)` replay also succeeds.

   Success observations:
   - command exits `0`
   - replay still boots from the extracted legacy checkpoint
   - a focused helper-level test or preserved work-dir probe confirms fresh PMA files were created under the bench work dir
   - no `boot::setup()`-style boot-source selection or snapshot restore occurs

6. Phase-1 unsupported-path behavior remains unchanged in Phase 1.5.

### Phase 2 checks

7. Master derived checkpoint build still works.
8. PMA derived checkpoint build works and produces a legacy-loadable `.chkjam`.
9. Master and PMA derived fixture build both work.
10. Master and PMA full fixture build both work, with PMA logs or assertions confirming fresh boot instead of PMA or snapshot reuse.

### Phase 3 checks

11. PMA trusted native bench writes a normal trusted artifact tree with additive PMA provenance fields.
12. PMA trusted Docker bench does the same, with Docker launched via `sol bench` and the bench harness still owning the final artifact set.

## 9. Risks And Open Questions

### Material implementation risks

- Replay compatibility risk in Phase 1:
  - verify that direct replay from legacy `.chkjam` into PMA `Kernel::load_with_hot_state_medium(..., None)` behaves the same across existing v0/v1/v2 fixtures.

- PMA-enabled replay risk in Phase 1.5:
  - verify that switching the same direct replay helper to `Some(PmaConfig)` does not introduce hidden assumptions about `boot::setup()`, event-log availability, or snapshot policy.

- PMA work-dir hygiene risk in Phase 1.5:
  - the helper must create fresh `.pma` files per run and never accidentally reopen stale slabs.

- Slab-copy risk in Phase 1 and 1.5:
  - the PMA `copy_into(..., &NounSpace)` migration is small but correctness depends on passing the source slab's noun space at every call site.

- Extractor containment risk in Phase 1 and 1.5:
  - because extractor is compiled into the same binary, the `init_nockapp(..., true)` wrapper must remain present or be explicitly cfg-gated.

- Checkpoint materialization risk in Phase 2:
  - `NockApp.kernel` is `pub(crate)`, so a runtime hook is mandatory.

- Full checkpoint determinism risk in Phase 2B:
  - PMA boot priority can silently select PMA or snapshot state unless the bench-created data dir is fresh.

### Open questions that must be verified, not guessed

- Is `nockapp::utils::NOCK_STACK_SIZE_MEDIUM` publicly importable from the PMA target branch for bench use, or does Phase 1.5 need a verified bench-local mirror?
- Is a direct PMA-enabled replay quick-bench stable across existing v0/v1/v2 fixtures, not just the current smoke fixture?
- Does the exact PMA target branch expose `SaveableCheckpoint::to_jammed_checkpoint::<J>()` publicly in the final cherry-pick destination, not only in the inspected remote branch?

## 10. Suggested Implementation Order

1. Land the feature gate plus `runtime_compat` replay-init and slab-copy helpers.
2. Preserve the existing `init_nockapp(..., prefer_existing_checkpoint)` wrapper and make the PMA `true` path fail explicitly.
3. Make PMA quick-bench pass with `pma: None` and `checkpoint_every_blocks = 0`.
4. Upgrade the PMA replay helper to fresh direct `Some(PmaConfig)` replay and verify the same quick-bench path still passes.
5. Keep explicit unsupported errors for PMA checkpoint-materialization paths instead of partially emulating them.
6. Add the minimal PMA runtime checkpoint hook.
7. Implement PMA derived `.chkjam` materialization and unblock `sol checkpoint`, derived fixture build, and replay checkpoint cadence.
8. Implement PMA full-checkpoint boot on a clean data dir and unblock full fixture build.
9. Add trusted native and Docker PMA provenance without changing the harness artifact model.

## 11. Success Definitions

### Phase 1 success

- `cargo build -p nockchain-bench --release` still succeeds on the master-style bench branch with no PMA feature enabled.
- `cargo build -p nockchain-bench --release --features pma-runtime-compat` succeeds against the PMA runtime branch.
- quick-bench runs to completion on PMA with the Phase-1 direct replay helper using `pma: None`.
- No file format changes are required for `.soltest` or `.solarch`.
- No trusted harness or orchestrator rewrite is required.

### Phase 1.5 success

- The same quick-bench command also runs to completion on PMA after upgrading the replay helper to `pma: Some(PmaConfig)`.
- PMA files are bench-owned, fresh per run, and confined to the replay work dir.
- No `boot::setup()` path, PMA boot-source selection, or event-log restore is required yet.
- Checkpoint materialization, fixture generation, and trusted provenance changes remain explicitly deferred to later phases.
