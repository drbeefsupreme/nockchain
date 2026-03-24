# PMA Bench Compatibility Implementation Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `nockchain-bench` compatible with the PMA runtime with the smallest practical blast radius, landing `nockchain-bench sol quick-bench` first and extending the same seam to checkpoint materialization, fixture generation, and trusted harness flows later.

**Architecture:** Keep `.soltest` v4, `.solarch`, and the trusted harness artifact model intact. Add a narrow, bench-local compatibility seam only where `nockapp` API shape diverged between `master` and the PMA branch, selected at compile time with a bench feature gate so the master build keeps its current behavior and replay hot path. When PMA needs a materialized checkpoint later, normalize PMA state back into legacy `.chkjam` bytes instead of introducing a new fixture format.

**Tech Stack:** Rust, Cargo feature gating, `nockchain-bench`, `nockapp`, existing SOL fixture and harness modules

---

## 1. Executive Summary

- Phase 1 is narrowly scoped to making `nockchain-bench sol quick-bench` run on the PMA runtime from an existing legacy `.soltest` fixture.
- The recommended design is a bench-local compatibility seam at the runtime API boundaries that actually changed: `NockApp::new`, `Kernel::load_with_hot_state_*`, `NounSlab::copy_into`, and checkpoint save/materialization.
- `.soltest` v4 stays unchanged. Its payload remains embedded checkpoint bytes plus archive bytes plus kernel bytes, and trusted harness unpacking stays unchanged.
- `.solarch` stays unchanged.
- The trusted harness and orchestrator stay structurally unchanged. `resolved_case.json`, `provenance.json`, and `summary.json` remain bench-owned artifacts, not outputs scraped from a helper subprocess.
- Phase 1 does not require checkpoint materialization. Quick-bench replay already flows through the shared once-run engine, and periodic replay checkpointing is only reached when `checkpoint_every_blocks > 0`.
- PMA replay boot for Phase 1 should use the execution-only path suggested by the PMA helper binary: load legacy `.chkjam` into `SaveableCheckpoint`, then call `Kernel::load_with_hot_state_medium(..., None)` inside the PMA-shaped `NockApp::new` closure.
- Phase 2 adds checkpoint materialization by normalizing PMA state back into legacy `.chkjam`: `NockApp::checkpoint()` or equivalent minimal hook -> `SaveableCheckpoint` -> `to_jammed_checkpoint::<NockJammer>()` -> `encode()` -> write file.
- Phase 2 should split internally into derived-checkpoint support first, then full-checkpoint boot support, because PMA boot-source priority only matters for full checkpoint materialization paths.
- Phase 3 formalizes trusted native and Docker PMA support with additive provenance fields such as `runtime_flavor`, `boot_source`, and `boot_event_num`, while keeping schema compatibility.

## 2. Scope And Constraints

### Phase 1 in scope

- Compile `nockchain-bench` against the PMA runtime branch with a dedicated feature gate.
- Make quick-bench replay execute on PMA from existing `.soltest` fixtures without changing fixture format.
- Keep master builds unchanged when the PMA feature is not enabled.
- Defer checkpoint materialization, full checkpoint boot, trusted provenance changes, and Docker-specific PMA polish unless they are required to keep the crate compiling.

### Later phases in scope

- Phase 2: checkpoint materialization back to legacy `.chkjam`, then derived fixture generation, then full checkpoint fixture generation.
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
- No broad runtime-enum abstraction threaded through requested/resolved case models.
- No subprocess-wrapper design around `bench_nockchain_checkpoint_block` or another PMA helper binary.
- No trusted-artifact rewrite.
- No code implementation in this task.

## 3. Ground-Truth Code Findings

### Observed facts

- Observed: `.soltest` v4 is still a checkpoint-first bundle. `SolFixtureFile` stores `checkpoint_bytes`, `archive_bytes`, and `kernel_bytes`, and extraction writes them back out as `fixture.chkjam`, `fixture.solarch`, and `fixture.jam`. See `crates/nockchain-bench/src/speed_of_light/fixture.rs:17-29`, `crates/nockchain-bench/src/speed_of_light/fixture.rs:54-60`, `crates/nockchain-bench/src/speed_of_light/fixture.rs:136-147`, and `crates/nockchain-bench/src/speed_of_light/harness/execute.rs:165-173`.
- Observed: quick-bench already runs through the shared once-run engine, not a separate replay implementation. `cmd_sol_quick_bench()` calls `execute_once_with_options()`, which extracts the fixture and constructs `SolBenchRunner`. See `crates/nockchain-bench/src/commands/sol.rs:265-271` and `crates/nockchain-bench/src/speed_of_light/harness/execute.rs:146-196`.
- Observed: quick-bench defaults `checkpoint_every_blocks` to `0`, and replay checkpointing only occurs inside the cadence branch that currently calls `save_blocking()`. See `crates/nockchain-bench/src/main.rs:142-148` and `crates/nockchain-bench/src/speed_of_light/bench.rs:407-424`.
- Observed: the bench-local runtime initialization seam is already concentrated in `kernel_utils.rs`. `init_nockapp()` owns the direct `NockApp::new(...)` and `Kernel::load_with_hot_state_medium(...)` call, and `init_full_checkpoint_nockapp()` owns the boot-path initialization. See `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs:61-98` and `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs:100-130`.
- Observed: current bench code uses the old `NounSlab::copy_into(noun)` signature in only two bench files: `poke.rs` and `extractor.rs`. See `crates/nockchain-bench/src/speed_of_light/poke.rs:42-56` and `crates/nockchain-bench/src/speed_of_light/extractor.rs:320-332`.
- Observed: checkpoint materialization today is still master-style and depends on `save_blocking()` plus reading the latest `.chkjam` from disk. `CheckpointBuilder::run()` does this directly. See `crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs:217-224`.
- Observed: the harness still owns trusted artifacts directly. `artifacts.rs` writes `resolved_case.json`, `provenance.json`, and `summary.json`, and `orchestrate.rs` calls those writers. See `crates/nockchain-bench/src/speed_of_light/harness/artifacts.rs:24-43` and `crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs:67-75`, `crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs:185-196`.
- Observed: provenance schema version is currently `"1"`. See `crates/nockchain-bench/src/speed_of_light/harness/mod.rs:56`.
- Observed: on `master`, `NockApp::new` takes `(kernel_from_checkpoint, snapshot_path, save_interval)`. On the PMA branch it takes only a metrics-accepting closure. Compare `master:crates/nockapp/src/nockapp/mod.rs:142-146` with `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/nockapp/mod.rs:156-160`.
- Observed: on `master`, `Kernel::load_with_hot_state_medium` takes five arguments. On the PMA branch it adds `pma: Option<PmaConfig>`. Compare `master:crates/nockapp/src/kernel/form.rs:606-612` with `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/kernel/form.rs:1261-1268`.
- Observed: on `master`, `NounSlab::copy_into` takes only a `Noun`. On the PMA branch it requires `&NounSpace`. Compare `master:crates/nockapp/src/noun/slab.rs:264-266` with `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/noun/slab.rs:281-283`.
- Observed: `save_blocking()` exists on `master` but is absent on the PMA branch. See `master:crates/nockapp/src/nockapp/mod.rs:315-318`; there is no corresponding PMA symbol.
- Observed: PMA boot CLI removed `save_interval` and added PMA- and durability-related fields such as `gc_interval`, `rotating_snapshot_interval_event_time`, `data_dir`, `event_log_path`, and `disable_fsync`. Compare `master:crates/nockapp/src/kernel/boot.rs:99-103`, `master:crates/nockapp/src/kernel/boot.rs:407-452` with `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/kernel/boot.rs:103-168`, `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/kernel/boot.rs:1578-1799`.
- Observed: PMA boot-source priority is real and explicit: valid PMA first, then verified snapshot plus replay, then checkpoint bootstrap, then fresh boot. See `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/kernel/boot.rs:1125-1355`.
- Observed: PMA still exposes `Kernel::checkpoint()` and `SaveableCheckpoint::to_jammed_checkpoint::<J>()`. See `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/kernel/form.rs:1418-1420` and `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/nockapp/save.rs:74-90`.
- Observed: `ExportedState` still only captures `LoadState { ker_hash, event_num, kernel_state }`; it does not include `cold`. See `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/nockapp/export.rs:38-60`.
- Observed: the existing PMA helper binary already demonstrates direct checkpoint replay using `Kernel::load_with_hot_state_medium(..., None)` and the PMA-shaped `NockApp::new(kernel_f)`. See `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockchain/src/bin/bench_nockchain_checkpoint_block.rs:339-347` and `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockchain/src/bin/bench_nockchain_checkpoint_block.rs:367-410`.

### Inferences

- Inference: Phase 1 does not need checkpoint materialization if it keeps `checkpoint_every_blocks = 0`, because quick-bench replay only needs to load the embedded checkpoint and then poke archive blocks.
- Inference: the minimal compile break surface is small enough that a bench-local seam is sufficient; the code does not justify a new runtime abstraction threaded through `RequestedCase`, `ResolvedCase`, fixture manifests, or harness backends.
- Inference: the current shared once-run engine is already the correct place to gain PMA quick-bench compatibility, because both quick-bench and trusted bench reuse it.
- Inference: PMA boot-source complexity should be isolated to full-checkpoint materialization paths. Replay execution from a legacy fixture checkpoint can stay on the direct kernel-load path and avoid `boot::setup()` entirely.

## 4. Proposed Design

### 4.1 Architecture And Module Seam

Add one bench-local compatibility module under `crates/nockchain-bench/src/speed_of_light/` and keep call-site changes narrow:

- New: `crates/nockchain-bench/src/speed_of_light/runtime_compat.rs`
- Existing call sites stay mostly unchanged and delegate into this module.

The seam owns only branch-shape differences:

- constructing a replay `NockApp`
- full-checkpoint boot setup
- copying nouns between slabs when PMA requires `NounSpace`
- later, materializing a legacy `.chkjam` from PMA runtime state

The seam does not own:

- archive iteration
- fixture format
- harness orchestration
- case resolution
- summary math
- Docker policy

### 4.2 Compile-Time Feature Strategy

Add a bench-local Cargo feature:

- `pma-runtime-compat`

Rules:

- `master` or `nockchain-bench-mega-pr` builds use the default build with the feature disabled.
- PMA branch builds enable `--features pma-runtime-compat`.
- The feature only changes bench code shape. It is not threaded into `RequestedCase`, `ResolvedCase`, `.soltest`, or `.solarch`.
- Use `#[cfg(feature = "pma-runtime-compat")]` and `#[cfg(not(feature = "pma-runtime-compat"))]` at function/module granularity so the replay hot path does not pay runtime branches.

### 4.3 Exact Bench-Local Helpers

Introduce or refactor to the following helpers:

- `runtime_compat::init_replay_nockapp(kernel_path, checkpoint, work_dir) -> Result<NockApp, KernelInitError>`
  - Master implementation: current `NockApp::new(..., work_dir, None)` plus `Kernel::load_with_hot_state_medium(...)`.
  - PMA implementation: `NockApp::new(move |metrics| async move { Kernel::load_with_hot_state_medium(..., None) })`.
  - `work_dir` is accepted for signature stability even though Phase 1 PMA replay does not use it.

- `runtime_compat::copy_from_source_slab(dst, noun, src) -> Noun`
  - Master implementation: `dst.copy_into(noun)`.
  - PMA implementation: `dst.copy_into(noun, &src.noun_space())`.
  - Keep this helper `#[inline]`.

- `runtime_compat::init_full_checkpoint_nockapp(kernel_path, work_dir) -> Result<NockApp, KernelInitError>`
  - Master implementation: current `boot::setup()` plus runtime bootstrap pokes.
  - Phase 1 PMA implementation: explicit unsupported error.
  - Phase 2 PMA implementation: clean `boot::setup()` path with a fresh PMA data dir.

- Phase 2 helper:
  - `runtime_compat::materialize_legacy_chkjam(nockapp, output_path, work_dir) -> Result<(), CheckpointMaterializationError>`
  - Master implementation: keep current `save_blocking()` plus file copy behavior.
  - PMA implementation: use the minimal runtime hook to capture `SaveableCheckpoint`, convert with `to_jammed_checkpoint::<NockJammer>()`, then encode and write a real legacy `.chkjam`.

### 4.4 File-By-File Change List

#### Phase 1: quick-bench execution

Files that should change:

- `crates/nockchain-bench/Cargo.toml`
  - Add `[features] pma-runtime-compat = []`.

- `crates/nockchain-bench/src/speed_of_light/mod.rs`
  - Add private `runtime_compat` module wiring.

- `crates/nockchain-bench/src/speed_of_light/runtime_compat.rs`
  - New compile-time compatibility helpers for replay init and slab copying.

- `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs`
  - Rewrite `init_nockapp()` to delegate to `runtime_compat::init_replay_nockapp()`.
  - Keep `sol_replay_wire()` and chain-peek helpers unchanged.
  - Make `init_full_checkpoint_nockapp()` return a PMA-feature-gated unsupported error in Phase 1.

- `crates/nockchain-bench/src/speed_of_light/poke.rs`
  - Replace direct `copy_into(page)` with `runtime_compat::copy_from_source_slab(...)`.

- `crates/nockchain-bench/src/speed_of_light/extractor.rs`
  - Replace direct `copy_into(entry_noun)` with the same helper so the crate compiles on PMA.

- `crates/nockchain-bench/src/speed_of_light/bench.rs`
  - Replace direct PMA-breaking initialization with the compatibility helper through `kernel_utils`.
  - Add an explicit PMA unsupported error for `checkpoint_every_blocks > 0` in Phase 1 instead of trying to emulate `save_blocking()`.

- `crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs`
  - Add an explicit PMA unsupported error in Phase 1 for any path that still depends on checkpoint materialization.

Files that should not change in Phase 1:

- `crates/nockchain-bench/src/speed_of_light/fixture.rs`
- `crates/nockchain-bench/src/speed_of_light/harness/execute.rs`
- `crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs`
- `crates/nockchain-bench/src/speed_of_light/harness/artifacts.rs`
- `crates/nockchain-bench/src/speed_of_light/harness/summary.rs`
- `.soltest` and `.solarch` formats

#### Phase 2: checkpoint materialization and fixture support

Files that should change:

- `crates/nockapp/src/nockapp/mod.rs`
  - Add the smallest needed runtime hook if bench cannot otherwise capture a checkpoint because `kernel` is private.

- `crates/nockchain-bench/src/speed_of_light/runtime_compat.rs`
  - Implement PMA legacy checkpoint materialization.
  - Implement PMA full-checkpoint boot setup.

- `crates/nockchain-bench/src/speed_of_light/kernel_utils.rs`
  - Switch PMA `init_full_checkpoint_nockapp()` from unsupported to real implementation.

- `crates/nockchain-bench/src/speed_of_light/bench.rs`
  - Replace Phase 1 unsupported checkpoint cadence path with real PMA `.chkjam` materialization.

- `crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs`
  - Replace Phase 1 unsupported save path with real PMA `.chkjam` materialization.

No file-format changes:

- `crates/nockchain-bench/src/speed_of_light/fixture.rs` stays structurally unchanged.
- `crates/nockchain-bench/src/speed_of_light/checkpoint.rs` stays the reader/metadata layer for legacy `.chkjam`.

#### Phase 3: trusted harness and provenance refinements

Files that should change:

- `crates/nockchain-bench/src/speed_of_light/harness/provenance.rs`
  - Add optional PMA-aware provenance fields.

- `crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs`
  - Thread the additional provenance facts into final artifact writing.

- `crates/nockchain-bench/src/speed_of_light/harness/native.rs`
  - Capture runtime flavor and boot facts for native trusted PMA runs.

- `crates/nockchain-bench/src/speed_of_light/harness/docker.rs`
  - Capture the same runtime flavor and boot facts for Docker trusted PMA runs.

- `crates/nockchain-bench/src/speed_of_light/harness/sweep.rs`
  - Treat the new provenance fields as comparison invariants when present.

- `crates/nockchain-bench/README.md`
- `crates/nockchain-bench/specs/bench-harness-spec.md`

### 4.5 Phase 1 Design: PMA Quick-Bench Execution

#### Absolute minimum change set

Phase 1 needs only:

- compile-time gating for `NockApp::new`
- compile-time gating for `Kernel::load_with_hot_state_medium(..., None)`
- compile-time gating for `NounSlab::copy_into(..., &NounSpace)`
- explicit deferral of `save_blocking()`-dependent code paths

It does not need:

- checkpoint materialization
- boot CLI translation for `save_interval`
- PMA event-log or snapshot handling
- new trusted artifact fields

#### Replay path

Phase 1 replay on PMA should be:

1. Read the embedded legacy `.chkjam` from the fixture.
2. Decode it into `LoadedCheckpoint`.
3. Rebuild `SaveableCheckpoint { ker_hash, event_num, state, cold }`.
4. Call PMA-shaped `NockApp::new(move |metrics| async move { Kernel::load_with_hot_state_medium(&kernel_bytes, checkpoint, &hot_state, vec![], TraceOpts::default(), None).await })`.
5. Replay archive pokes exactly as today.

That matches the helper-binary evidence and avoids PMA boot-source ambiguity entirely for Phase 1.

#### PMA differences deferred out of Phase 1

- `save_blocking()` replacement
- `boot::Cli` field translation
- `boot::setup()` work-dir/data-dir management
- full checkpoint mode
- trusted provenance changes
- PMA-specific checkpoint timing semantics

#### Phase 1 unsupported behavior

Under `--features pma-runtime-compat`, Phase 1 should fail clearly for:

- `--checkpoint-every-blocks > 0`
- `sol checkpoint`
- `sol fixture build`

Those are Phase 2 because they require `.chkjam` output, not just replay execution.

### 4.6 Phase 2 Design: Checkpoint And Fixture Support

#### Phase 2A: derived checkpoint materialization

Implement PMA `.chkjam` output without changing bench artifacts:

1. Add the smallest runtime hook needed to get a `SaveableCheckpoint` out of `NockApp`.
2. In bench code, materialize legacy checkpoint bytes as:
   - `checkpoint = nockapp.checkpoint().await?`
   - `jammed = checkpoint.to_jammed_checkpoint::<NockJammer>()`
   - `bytes = jammed.encode()?`
   - `std::fs::write(output_path, bytes)?`
3. Use that path for:
   - `sol checkpoint`
   - fixture build with `checkpoint_kind = derived`
   - replay checkpoint cadence in `SolBenchRunner`

This keeps `.chkjam` as the materialized checkpoint artifact and avoids introducing PMA-native fixtures.

#### Phase 2B: full checkpoint boot support

Only full checkpoint generation needs PMA boot semantics.

Implementation rules:

- `init_full_checkpoint_nockapp()` on PMA must build a fresh PMA data dir under the bench work dir, for example:
  - `<work_dir>/runtime-data/checkpoints/`
  - `<work_dir>/runtime-data/pma/`
  - `<work_dir>/runtime-data/event-log.sqlite3`
- The PMA boot CLI should set:
  - `new = true`
  - `gc_interval = Some(0)`
  - `rotating_snapshot_interval_event_time = None`
  - `data_dir = Some(runtime_data_dir)`
  - `event_log_path = Some(runtime_data_dir.join("event-log.sqlite3"))`
  - `disable_fsync = false` unless an explicit local benchmark mode later decides otherwise
- The data dir must be created fresh per invocation; do not reuse an operator-supplied dirty directory for full checkpoint generation.
- After boot plus existing bootstrap pokes plus archive replay, materialize a legacy `.chkjam` through the same Phase 2A normalization path.

This keeps full checkpoint behavior deterministic even though PMA boot priority prefers PMA and snapshots over checkpoint bootstrap.

### 4.7 Phase 3 Design: Trusted Native And Docker Bench

The harness structure remains unchanged:

- `ResolvedCase` stays the resolved request plus fixture identity.
- `execute_once()` stays the shared once-run engine.
- `execute_trusted_run()` stays the shared orchestrator.
- `artifacts.rs` keeps writing all final trusted artifacts.

Phase 3 work is mostly PMA-awareness, not a harness rewrite:

- mark trusted runs with `runtime_flavor`
- capture logical boot source for replay runs
- preserve current summary/verdict pipeline
- ensure Docker provenance keeps bench as the source of truth instead of delegating to external helper output

Important point:

- Replay trusted runs should still boot from the extracted legacy fixture checkpoint through the direct replay init path, not via PMA `boot::setup()`.
- Therefore trusted native and Docker PMA replay runs should report `boot_source = "checkpoint"` unless a future execution mode deliberately switches to PMA boot artifacts.

## 5. PMA Runtime Hook

### Required for Phase 1?

No.

Phase 1 can run quick-bench by replaying from the embedded legacy checkpoint and never materializing a new checkpoint.

### Required for later phases?

Yes, for any PMA path that must emit a real `.chkjam`.

### Smallest viable API

Recommended:

- `NockApp::checkpoint() -> Result<SaveableCheckpoint, NockAppError>`

Why:

- `Kernel::checkpoint()` already exists on both master and PMA.
- `SaveableCheckpoint::to_jammed_checkpoint::<J>()` is already public on the PMA branch.
- Bench already knows how to write bytes and name output files.
- This keeps the runtime addition narrow and avoids inventing a new bench-facing serialization API.

Fallback if exposing `SaveableCheckpoint` is undesirable:

- `NockApp::checkpoint_to_jam<J: Jammer>() -> Result<JammedCheckpointV2, NockAppError>`

This is still acceptable, but `checkpoint()` is the smaller and more generally useful primitive.

### Why `export()` is insufficient

`ExportedState` only round-trips `LoadState { ker_hash, event_num, kernel_state }` and omits `cold`. See `remotes/nockchain/bitemyapp/bump-pma-post-throughput-event-log-and-snapshots-squashed-rebased-fsync:crates/nockapp/src/nockapp/export.rs:38-60`.

That means `export()` cannot preserve current legacy checkpoint semantics or produce a complete legacy `.chkjam` payload.

### Exact normalization path

Use this exact PMA normalization path in Phase 2:

- `NockApp::checkpoint()` -> `SaveableCheckpoint`
- `SaveableCheckpoint::to_jammed_checkpoint::<NockJammer>()`
- `JammedCheckpointV2::encode()`
- write bytes to `*.chkjam`

Do not introduce a PMA-native bench fixture format.

## 6. Provenance And Schema Handling

### Phase 1

- No trusted provenance change is required.
- `sol quick-bench` is not a trusted artifact path and should remain so.
- `resolved_case.json`, `summary.json`, and `provenance.json` stay unchanged in this phase.

### Phase 2

- Derived checkpoint generation and fixture generation do not require a trusted provenance schema change.
- The fixture format stays unchanged because the output is still legacy `.chkjam` bytes embedded into `.soltest`.

### Phase 3

Keep `schema_version = "1"` and extend `Provenance` additively with optional fields.

Recommended additive fields:

- `runtime_flavor: Option<String>`
  - Values: `"legacy"` or `"pma"`

- `boot_source: Option<String>`
  - Values: `"checkpoint"`, `"pma"`, `"snapshot"`, `"fresh"`

- `boot_event_num: Option<u64>`
  - For replay runs from fixtures, use `fixture_manifest.checkpoint_event_num`

- `boot_snapshot_manifest_path: Option<PathBuf>`
  - Only populate when a PMA boot path actually restored from a snapshot

Do not add `boot_path` for ordinary replay runs from extracted fixtures:

- the path would only be a temp extraction path such as `fixture.chkjam`
- that path is not stable or useful for trusted comparison

Backward compatibility story:

- existing consumers that ignore unknown fields continue to work
- old artifacts remain valid because required fields do not change
- `BackendRuntimeFacts::Native` can remain structurally unchanged; do not convert it from a unit variant just to carry PMA facts

### Trusted artifact model compatibility

- `resolved_case.json`
  - remains the requested and resolved execution plan
  - should not gain PMA branch-shape knobs

- `summary.json`
  - remains the statistical summary over measured runs
  - Phase 3 should not redefine summary shape just because PMA is in use

- `provenance.json`
  - remains the realized environment artifact
  - gains only additive optional runtime facts in Phase 3

- Harness ownership
  - remains in `harness/artifacts.rs` and `harness/orchestrate.rs`
  - is not delegated to an external PMA helper process

## 7. Checkpoint Profiling Semantics

Phase 1:

- Not required.
- PMA quick-bench should forbid replay checkpoint cadence rather than pretend PMA durability equals legacy `save_blocking()`.

Phase 2 and Phase 3 decision:

- Preserve current checkpoint timing summary fields only for explicit bench-triggered legacy `.chkjam` materialization.
- Do not reinterpret those fields as total PMA durability cost.
- If PMA-specific raw timing evidence becomes useful later, add it as raw evidence or optional profile payloads rather than changing the meaning of `checkpoint_count` and `average_checkpoint_time_secs`.

Reasoning:

- legacy `save_blocking()` and PMA persistence are semantically different
- the bench command still needs a timing for explicit `.chkjam` capture when `checkpoint_every_blocks > 0`
- keeping summary semantics narrow avoids misleading comparisons

## 8. Verification Plan

All verification uses release assumptions.

### Phase 1 gate checks

1. Master compile still works:

   ```bash
   cargo build -p nockchain-bench --release
   ```

   Success observation:
   - build succeeds without enabling `pma-runtime-compat`

2. PMA compile works with the feature:

   ```bash
   cargo build -p nockchain-bench --release --features pma-runtime-compat
   ```

   Success observation:
   - build succeeds against the PMA branch API shape

3. PMA quick-bench native smoke:

   ```bash
   /shared/nockchain/target/release/nockchain-bench sol quick-bench \
     --fixture /shared/nockchain/fixtures/first-100-v2-derived-checkpoint-no-mempool.soltest \
     --blocks 100 \
     --checkpoint-every-blocks 0
   ```

   Success observation:
   - command exits `0`
   - the summary prints `Blocks poked:` with a value greater than `0`
   - no runtime-compat unsupported error is emitted
   - no PMA API-shape panic occurs during init or poke replay

### Phase 2 checks

4. Master derived checkpoint build still works:

   ```bash
   cargo build -p nockchain-bench --release
   /shared/nockchain/target/release/nockchain-bench sol checkpoint \
     --archive /shared/nockchain/solarch/38394-blocks-no-mempool.solarch \
     --kernel /shared/nockchain/assets/dumb.jam \
     --target-height 100 \
     --output /shared/nockchain/tmp/master-derived-100.chkjam
   ```

   Success observation:
   - output file exists
   - `checkpoint_event_num()` succeeds on the file

5. PMA derived checkpoint build works:

   ```bash
   /shared/nockchain/target/release/nockchain-bench sol checkpoint \
     --archive /shared/nockchain/solarch/38394-blocks-no-mempool.solarch \
     --kernel /shared/nockchain/assets/dumb.jam \
     --target-height 100 \
     --output /shared/nockchain/tmp/pma-derived-100.chkjam
   ```

   Success observation:
   - output file exists
   - `checkpoint_event_num()` succeeds on the file
   - file loads through existing `load_checkpoint()`

6. Master and PMA derived fixture build both work:

   ```bash
   /shared/nockchain/target/release/nockchain-bench sol fixture build \
     --archive /shared/nockchain/solarch/38394-blocks-no-mempool.solarch \
     --kernel /shared/nockchain/assets/dumb.jam \
     --start-height 0 \
     --end-height 100 \
     --checkpoint-kind derived \
     --output /shared/nockchain/tmp/pma-derived.soltest \
     --work-dir /shared/nockchain/tmp/pma-derived-build
   ```

   Success observation:
   - fixture file exists
   - `sol fixture inspect` reports the expected checkpoint kind, checkpoint height, and archive range

7. Master and PMA full fixture build both work:

   ```bash
   /shared/nockchain/target/release/nockchain-bench sol fixture build \
     --archive /shared/nockchain/solarch/38394-blocks-no-mempool.solarch \
     --kernel /shared/nockchain/assets/dumb.jam \
     --start-height 0 \
     --end-height 100 \
     --checkpoint-kind full \
     --output /shared/nockchain/tmp/pma-full.soltest \
     --work-dir /shared/nockchain/tmp/pma-full-build
   ```

   Success observation:
   - fixture file exists
   - the embedded checkpoint is a real legacy `.chkjam`
   - PMA builder logs or assertions confirm fresh boot instead of accidental PMA/snapshot reuse

### Phase 3 checks

8. PMA trusted native bench:

   ```bash
   /shared/nockchain/target/release/nockchain-bench sol bench \
     --fixture /shared/nockchain/fixtures/first-100-v2-derived-checkpoint-no-mempool.soltest \
     --output /shared/nockchain/tmp/pma-native-bench-smoke \
     --warmup-runs 0 \
     --measured-runs 3 \
     --cooldown-secs 0
   ```

   Success observation:
   - trusted artifact tree exists
   - `resolved_case.json`, `provenance.json`, and `summary.json` exist
   - `summary.json.measured_runs_succeeded == 3`
   - `provenance.json.runtime_flavor == "pma"`
   - `provenance.json.boot_source == "checkpoint"`

9. PMA trusted Docker bench:

   ```bash
   DOCKER_HOST=unix:///home/drbeefsupreme/.docker/desktop/docker.sock \
   /shared/nockchain/target/release/nockchain-bench sol bench \
     --fixture /shared/nockchain/fixtures/first-100-v2-derived-checkpoint-no-mempool.soltest \
     --output /shared/nockchain/tmp/pma-docker-bench-smoke \
     --docker-image nockchain-bench:local \
     --memory-limit 8g \
     --work-dir-mode docker-tmpfs \
     --warmup-runs 0 \
     --measured-runs 3 \
     --cooldown-secs 0
   ```

   Success observation:
   - trusted Docker artifact tree exists
   - host/container binary identity is still bench-recorded
   - `provenance.json.runtime_flavor == "pma"`
   - `provenance.json.boot_source == "checkpoint"`

## 9. Risks And Open Questions

### Material implementation risks

- Replay compatibility risk in Phase 1:
  - verify that direct replay from legacy `.chkjam` into PMA `Kernel::load_with_hot_state_medium(..., None)` behaves the same across existing v0/v1/v2 fixtures.

- Slab-copy risk in Phase 1:
  - the PMA `copy_into(..., &NounSpace)` migration is small but correctness depends on passing the source slab's noun space at every call site.

- Checkpoint materialization risk in Phase 2:
  - if bench cannot reach `Kernel::checkpoint()` because `NockApp.kernel` remains private, the PMA runtime hook is mandatory.

- Full checkpoint determinism risk in Phase 2:
  - PMA boot priority can silently select PMA or snapshot state unless the bench-created data dir is fresh.

- Trusted Docker rollout risk in Phase 3:
  - the runtime seam itself is enough for replay, but image-build flows may still need explicit `pma-runtime-compat` enablement when producing the Docker binary.

### Open questions that must be verified, not guessed

- Does the exact PMA target branch expose `SaveableCheckpoint::to_jammed_checkpoint::<J>()` publicly in the cherry-pick destination, not only in the inspected remote branch?
- Is a bench-local PMA quick-bench smoke with existing fixtures fully stable across proof-version windows, or are there fixture-specific replay differences?
- For full checkpoint mode, do PMA background snapshot artifacts appear even with `gc_interval = Some(0)` and rotating snapshots disabled, and if so do they affect determinism?

## 10. Suggested Implementation Order

1. Land the feature gate and the replay-init/slab-copy seam only.
2. Make PMA quick-bench pass with `checkpoint_every_blocks = 0`.
3. Add explicit unsupported errors for PMA checkpoint materialization paths instead of partially emulating them.
4. Add the minimal PMA runtime checkpoint hook.
5. Implement PMA derived `.chkjam` materialization and unblock `sol checkpoint`, derived fixture build, and replay checkpoint cadence.
6. Implement PMA full checkpoint boot on a clean data dir and unblock full fixture build.
7. Add trusted native PMA provenance fields without changing the harness artifact model.
8. Add trusted Docker PMA provenance and verification, keeping the harness as artifact owner.

## 11. Phase 1 Success Definition

Phase 1 is successful only when all of the following are true:

- `cargo build -p nockchain-bench --release` still succeeds on the master-style bench branch with no PMA feature enabled.
- `cargo build -p nockchain-bench --release --features pma-runtime-compat` succeeds against the PMA runtime branch.
- `/shared/nockchain/target/release/nockchain-bench sol quick-bench --fixture /shared/nockchain/fixtures/first-100-v2-derived-checkpoint-no-mempool.soltest --blocks 100 --checkpoint-every-blocks 0` runs to completion on PMA and prints a normal benchmark summary.
- No file format changes are required for `.soltest` or `.solarch`.
- No trusted harness/orchestrator rewrite is required.
- PMA checkpoint materialization, fixture generation, and trusted provenance changes remain explicitly deferred to later phases instead of being hand-waved into Phase 1.
