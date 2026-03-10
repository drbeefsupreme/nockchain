# Spec: Trustworthy SOL Benchmark Harness v4 (Consensus)

## Status

This spec represents consensus between both reviewers. It supersedes
BENCH_HARNESS_SPEC.md, BENCH_HARNESS_SPEC_v3.md, and all prior drafts.

## 1. Purpose

Build a trustworthy benchmark harness for SOL replay workloads in `nockchain-bench`.

The harness must make four things explicit and auditable:
- requested inputs
- resolved execution plan
- realized execution environment
- raw measurement evidence

Mining benchmarks are out of scope and are removed before any new harness work
begins.

## 2. Scope

This spec applies only to SOL replay benchmarking driven by `SolBenchRunner` and
the unified `.soltest` fixture format.

It does not apply to:
- mining benchmarks
- `MiningScenario`
- event correlation for mining logs
- Parquet export for mining stats
- the current `sol sweep` implementation

## 3. Design Axioms

1. One measured run equals one fully specified case.
2. One trustworthy comparison changes only declared axes.
3. Raw evidence is retained even when parsed summaries exist.
4. Requested configuration and realized provenance are separate artifacts.
5. Trusted Docker runs require validation.
6. Trusted results require release builds.
7. Sweep orchestration contains no measurement logic.
8. Human labels never drive logic.
9. Invalid runs are preserved, not discarded.
10. Prefix replay is a first-class supported mode; arbitrary in-fixture slicing
    is not required in v1.

## 4. Phase 0: Hard Deletion Boundary

Phase 0 is a clean break. No compatibility stubs.

### 4.1 Delete Entire Subsystems

Delete these directories or modules entirely:
- `src/scenario/`
- `src/events/`
- `src/output/`
- `src/runner/`
- `src/commands/mining.rs`
- `src/speed_of_light/sweep.rs`

### 4.2 Delete CLI Surfaces

Remove these top-level commands from `src/main.rs`:
- `Run`
- `Attach`
- `Compare`
- `Analyze`

Remove the current `sol sweep` subcommand entirely. It will be replaced later.

### 4.3 Delete Mining-Specific Types And Re-exports

Remove:
- `MiningScenario`
- `MiningScenarioConfig`
- `MiningResult`
- `NockchainMode`
- `OutputFormat`
- all re-exports tied to mining, events, parquet, or the old runner

### 4.4 Cargo Dependency Cleanup

After Phase 0, remove dependencies that only supported deleted subsystems.
At minimum, reevaluate and likely remove:
- `arrow`
- `parquet`
- `chrono`

Keep Docker-related dependencies (`bollard`, `futures`) because the new SOL
harness will still need them.

### 4.5 Salvage Generic Helpers

The deleted `src/runner/docker.rs` contains generic code that is not
mining-specific and should be carried into the new harness module (Section 5):

- Docker connection logic (multi-socket discovery, ping)
- `ContainerStats` struct and `from_docker_stats` (v1/v2 cgroup handling)
- `parse_memory_limit`
- `parse_proc_stat_faults`
- `calculate_cpu_percent`
- Page fault reading via `docker exec` of `/proc/1/stat`

Carry these as source into the new `speed_of_light::harness::docker` module.
Do not preserve the old module structure or its mining-specific API surface
(`DockerRunnerConfig`, `NockchainMode`, `build_args`, `build_env`).

Similarly, `page_fault_bursts` from `sweep.rs` is conceptually a profiling
helper. Move it to `profiling.rs` if it is still useful.

### 4.6 Phase 0 Exit Criteria

After deletion:
- `cargo build -p nockchain-bench --release` passes
- `cargo test -p nockchain-bench --release` passes
- remaining CLI surface is SOL-focused plus `sample`

Remaining commands:
- `sample`
- `sol quick-bench`
- `sol extract`
- `sol checkpoint`
- `sol inspect`
- `sol fixture build`
- `sol fixture inspect`

## 5. New Module Layout

Do not repurpose the mining-era `runner/docker.rs`.

Create a new SOL-specific harness module tree under `src/speed_of_light/harness/`.

Suggested layout:

```text
src/speed_of_light/harness/
├── mod.rs
├── case.rs          # RequestedCase, ResolvedCase, matrix schema
├── artifacts.rs     # output tree writing, raw/log/result persistence
├── provenance.rs    # host/container provenance capture
├── summary.rs       # stats aggregation, verdicts
├── execute.rs       # shared once-run execution contract
├── native.rs        # native execution path
├── docker.rs        # Docker execution path for SOL replay
├── validate.rs      # Docker validation gate and probe protocol
└── sweep.rs         # matrix expansion and orchestration only
```

Reason:
- the old `runner/` module is not actually generic
- the new harness is SOL-specific by scope
- isolating the new code avoids dragging mining assumptions into the redesign

## 6. Reuse From Existing Code

Reuse these existing SOL pieces:
- `SolBenchRunner`
- `SolBenchConfig`
- `SolBenchResults`
- `MemoryProfile`
- `ProcessMemoryProfiler`
- `SolScorecard`
- `SolFixtureManifest`
- fixture parsing and extraction helpers
- archive reader/writer
- checkpoint builder and extractor utilities
- `sampler::smaps`, `sampler::buckets` (process memory attribution)

Do not make the old mining-oriented abstractions part of the new design.

## 7. Benchmark Semantics

### 7.1 `--blocks`

In v1, `--blocks N` means:
- replay at most the first `N` replayable blocks from the fixture's archive window

It does not mean:
- slice the fixture file physically
- skip into the middle of the fixture window
- benchmark an arbitrary offset without building a different fixture

This matches the current behavior and is enough for a minimal trustworthy protocol.

### 7.2 Fixture Handling

In v1:
- the fixture is treated as an immutable input blob
- full fixture extraction is acceptable
- fixture bind-mounting into Docker is acceptable
- partial extraction optimization is explicitly out of scope

Future optimization may add partial extraction or offset-based replay, but that
is not part of this rewrite.

## 8. Execution Architecture

The central execution contract is: "run one resolved case once and emit
machine-readable artifacts."

### 8.1 Shared Once-Run Contract

Introduce one shared execution entrypoint used by both:
- native trusted runs
- Docker trusted runs

This must be implemented as a library-level operation, not by scraping
human-readable CLI output.

A hidden/internal CLI is acceptable for container execution, for example:
- `nockchain-bench sol run-once --resolved-case /bench/input/resolved_case.json --run-dir /bench/output/run-0`

The important requirement:
- native and Docker trusted execution share the same machine-oriented once-run path

### 8.2 Relationship To `sol quick-bench`

`sol quick-bench` remains as the quick ad hoc interface.

`sol bench` must not depend on parsing `sol quick-bench` stdout.

Recommended structure:
- extract the current single-run logic into a library function
- let `sol quick-bench` call that function and print a human summary
- let `sol bench` call that same function and manage provenance, repetitions, and
  artifacts
- let Docker mode invoke the same code path inside the container via a
  hidden/internal subcommand

One measurement engine, two interfaces:
- `sol quick-bench` for ad hoc use
- `sol bench` for trusted measurement

## 9. Data Model

### 9.1 RequestedCase

Requested input only. No auto-captured facts.

```rust
pub struct RequestedCase {
    pub benchmark: String,                // "sol-replay"
    pub label: Option<String>,

    pub fixture_path: PathBuf,
    pub blocks: u64,                      // 0 = all blocks in fixture window
    pub skip_genesis: bool,

    pub enable_checkpointing: bool,
    pub checkpoint_every_blocks: u64,

    pub profile_memory: bool,
    pub profile_interval_ms: u64,

    pub execution: ExecutionRequest,
    pub threads: u32,

    pub warmup_runs: u32,                 // default 1
    pub measured_runs: u32,               // default 5, minimum 3
    pub cooldown_secs: u64,               // default 10
}

pub enum ExecutionRequest {
    Native,
    Docker {
        image_tag: String,
        memory_limit: String,
        cpuset: Option<String>,
        cpu_quota: Option<i64>,
        cpu_period: Option<i64>,
        work_dir_mode: WorkDirMode,
    },
}

pub enum WorkDirMode {
    HostBind,
    DockerVolume,
    DockerTmpfs,
}
```

### 9.2 ResolvedCase

Normalized execution plan after applying defaults and computing static inputs.

Includes:
- absolute paths
- parsed memory limit bytes
- fixture SHA256 (computed by harness)
- embedded fixture manifest
- schema version
- build profile
- tool version and commit

Does not include runtime facts like container id or realized cgroup values.

### 9.3 Provenance

Realized environment captured after setup and before the first measured block.

Provenance must distinguish host-side and container-side identity in Docker mode.

`provenance.json` must include:
- schema version
- capture timestamp
- host identity (hostname, kernel, OS, total memory, CPU count, CPU model)
- git identity (commit, branch, dirty, commit date)
- fixture identity (path, SHA256, embedded `SolFixtureManifest`)
- binary identity (version, commit, profile)

For Docker mode, additionally record:
- `host_binary`: version and commit of the orchestrator binary on the host
- `container_binary`: version and commit of the `nockchain-bench` binary inside
  the container

Container provenance must include:
- image tag
- image digest (resolved sha256)
- container id
- Docker engine version
- Docker context
- cgroup version (v1 or v2)
- storage driver
- realized `memory.max`
- realized `memory.current` snapshot
- realized cpuset
- realized `cpu.max` or quota/period

### 9.4 Version Skew Policy

Trusted Docker mode requires host/container binary version agreement by default.

- record host binary version/commit
- record container binary version/commit
- mark the run `Invalid` unless they match, or unless `--allow-version-skew` is set

This prevents the orchestrator and the measured binary from silently diverging.

### 9.5 Verdict

```rust
pub enum Validity {
    Valid,
    Partial { reasons: Vec<String> },
    Invalid { reasons: Vec<String> },
}
```

Examples:
- `Invalid`: requested memory limit does not match realized limit
- `Invalid`: debug build used without `--allow-debug-benchmark`
- `Invalid`: host/container binary mismatch without `--allow-version-skew`
- `Partial`: one measured repetition failed
- `Partial`: throughput CV exceeded threshold

## 10. Artifact Model

### 10.1 Single Trusted Run

```text
<output_root>/
├── schema_version.txt
├── requested_case.json
├── resolved_case.json
├── provenance.json
├── validation.json                 # Docker only
├── verdict.json
├── summary.json
├── raw/
│   ├── docker_inspect.json         # Docker only
│   ├── docker_info.json            # Docker only
│   ├── host_env.json               # selected host env and runtime facts
│   └── container_env.json          # Docker only, selected runtime facts
├── runs/
│   ├── warmup-0/
│   │   ├── result.json
│   │   ├── profile.json
│   │   ├── block_timings.ndjson
│   │   ├── stdout.log
│   │   └── stderr.log
│   ├── run-0/
│   │   ├── result.json
│   │   ├── profile.json
│   │   ├── block_timings.ndjson
│   │   ├── container_samples.ndjson   # Docker only
│   │   ├── stdout.log
│   │   └── stderr.log
│   └── ...
```

Notes:
- warmups are persisted but excluded from summary statistics
- raw Docker evidence is retained alongside parsed provenance
- `block_timings.ndjson`: line-delimited `{"height": N, "duration_ms": F}` per
  block, enables streaming analysis without loading the full result blob
- `container_samples.ndjson`: Docker stats API polls from the host at
  `profile_interval_ms`. Each line:
  `{"timestamp_ms": N, "memory_usage_bytes": N, "memory_limit_bytes": N,
  "memory_rss_bytes": N, "cpu_percent": F, ...}`
- no markdown is required for a single trusted run

### 10.2 Sweep Output

```text
<sweep_root>/
├── schema_version.txt
├── matrix.json
├── matrix_expanded.json
├── schedule.json
├── verdict.json
├── comparison.json
├── comparison.md                 # optional derived output
└── cases/
    ├── case-000-memory_limit_4g/
    ├── case-001-memory_limit_8g/
    └── ...
```

Case directory naming: zero-padded index plus axis values for readability and
sort stability. Multi-axis: `case-000-memory_limit_4g+checkpoint_every_10`.

Do not use timestamps or labels as the primary case identifier.

## 11. Docker Execution Model

### 11.1 Core Rule

In Docker mode, the SOL replay workload must run inside the constrained container.

### 11.2 What The Host Does

The host orchestrator:
- validates requested Docker parameters
- resolves image tag to digest
- captures Docker engine facts
- creates the container with requested resource limits
- bind-mounts fixture input read-only at `/bench/fixture.soltest`
- bind-mounts artifact output at `/bench/output/`
- configures work dir at `/bench/work/` per `work_dir_mode`
- polls container-level stats concurrently during execution
- gathers raw inspect/info evidence
- collects exit status and logs

### 11.3 What The Container Does

The container executes the shared once-run SOL path via a hidden/internal CLI
designed for machine execution:

```
nockchain-bench sol run-once \
  --resolved-case /bench/input/resolved_case.json \
  --run-dir /bench/output/run-0
```

The container does NOT run the public `sol quick-bench` command. The trusted Docker
path must not depend on parsing human-readable stdout.

### 11.4 Image Requirements

The image must contain:
- `nockchain-bench` release binary

It does not need:
- `nockchain`
- mining logic
- network configuration

Minimal Dockerfile:

```dockerfile
FROM ubuntu:24.04
COPY target/release/nockchain-bench /usr/local/bin/nockchain-bench
ENTRYPOINT ["/usr/local/bin/nockchain-bench"]
```

Fixture input is mounted read-only from the host.

### 11.5 Work Directory Modes

Explicit and benchmark-relevant:
- `HostBind`: bind mount from host
- `DockerVolume`: named Docker volume
- `DockerTmpfs`: tmpfs mount

No silent default in trusted Docker mode. Storage mode changes I/O behavior,
which changes replay results. The user must choose explicitly.

## 12. Validation Gate

Trusted Docker runs require validation.

### 12.1 Required Checks

1. container starts successfully
2. realized `memory.max` matches requested limit
3. CPU controls are readable and recorded
4. a known allocation (64 MiB, pages touched) changes `memory.current` as
   expected (+/- 20%)
5. required cgroup files are readable inside the runtime environment

### 12.2 Validation Mechanism

Preferred implementation:
- `nockchain-bench sol validate-probe` runs inside the container

This is better than shell-based probing because it:
- avoids shell/tooling assumptions in minimal images
- keeps the validation protocol versioned with the harness
- emits structured JSON directly

### 12.3 OOM Policy

OOM testing is not a mandatory gate. It is destructive and complicates recovery.
Optional diagnostic mode only (`sol validate --stress-oom`).

### 12.4 Validation Cache

Validation may be cached per unique tuple of:
- Docker engine version
- cgroup version
- image digest
- memory limit
- cpuset
- cpu quota/period
- work dir mode
- validation probe version

Cache is local to the sweep output directory. If a valid `validation.json`
exists for the same tuple, the gate is skipped.

### 12.5 Abort Behavior

If validation fails, the run aborts immediately with a clear error identifying
what did not match. The partial `validation.json` is preserved.

## 13. Measurement Sources

Trust hierarchy (most trusted first):
1. `SolBenchResults` — benchmark-native metrics
2. `MemoryProfile` — in-process RSS/VM/fault sampling
3. cgroup snapshots from inside the container
4. time-series container samples via Docker stats API

Never use as primary evidence:
- `/usr/bin/time docker ...`
- Docker client memory
- human summaries without machine artifacts
- artifact directory names as benchmark dimensions

## 14. Summary Rules

`summary.json` must include raw values and dispersion metrics.

For each summarized metric, include:
- `median`
- `min`
- `max`
- `mad` (Median Absolute Deviation — robust to outliers for small N)
- `stddev`
- `cv` (Coefficient of Variation = stddev/mean)
- `values` (raw array, always retained)

Minimum metrics to summarize:
- throughput (blocks/s)
- init time
- total replay time
- average block time
- failed pokes
- checkpoint count
- average checkpoint time
- peak process RSS
- peak container memory (Docker mode)
- major/minor fault totals (where available)

Defaults:
- measured runs default to 5
- minimum measured runs is 3

### Stability Flagging

If `cv` exceeds a configured threshold (default 0.10) on throughput, the verdict
is `Partial` with reason.

High spread does NOT trigger automatic extra runs in v1. The harness flags
instability and records the reason. The operator decides whether to rerun with
`--measured-runs N`.

## 15. Sweep Semantics

### 15.1 Matrix Schema

The matrix always uses `axes` (plural, a map of axis name to value list).
Single-axis and multi-axis sweeps use the same schema shape.

Single-axis example:

```json
{
  "benchmark": "sol-replay",
  "base": {
    "fixture": "bench-artifacts/fixtures/v2-100.soltest",
    "blocks": 100,
    "enable_checkpointing": true,
    "checkpoint_every_blocks": 10,
    "profile_memory": true,
    "threads": 4,
    "warmup_runs": 1,
    "measured_runs": 5,
    "mode": {
      "docker": {
        "image_tag": "nockbench-master:latest",
        "cpuset": "0-3",
        "work_dir_mode": "DockerTmpfs"
      }
    }
  },
  "axes": {
    "memory_limit": ["4g", "8g", "16g", "32g"]
  }
}
```

Multi-axis example (requires `--allow-multi-axis`):

```json
{
  "benchmark": "sol-replay",
  "base": { "..." },
  "axes": {
    "memory_limit": ["4g", "32g"],
    "checkpoint_every_blocks": [5, 10, 20]
  }
}
```

Using `axes` as a map means one schema shape, one parser. The
`--allow-multi-axis` flag gates `len(axes) > 1`, not a schema change.

Without `--allow-multi-axis`, a matrix with more than one axis is an error with
a message explaining that multi-axis sweeps measure multiple things simultaneously
and the user must explicitly opt in.

### 15.2 Default Sweep Policy

Trusted sweeps are single-axis by default.

### 15.3 Scheduling

Allowed modes:
- sequential (default)
- `--interleave` — round-robin across cases (defends against temporal confounds)
- `--randomize-order`

Not allowed in trusted mode by default:
- concurrent measured execution

### 15.4 Cooldown

`cooldown_secs` (default 10) applies between all runs: warmup-to-warmup,
warmup-to-measured, measured-to-measured. Purpose: let host stabilize.

### 15.5 Invariants

Across a trusted comparison, all non-axis fields must remain constant, including:
- fixture SHA256 and manifest
- git commit and dirty state
- build profile
- execution mode
- image digest
- work dir mode
- checkpointing config
- thread count
- CPU control policy
- host identity (unless explicitly overridden)
- host/container binary identity policy

The sweep wrapper verifies invariants at matrix expansion time.

## 16. CLI Surface

### 16.1 Keep

- `sample`
- `sol quick-bench`
- `sol extract`
- `sol checkpoint`
- `sol inspect`
- `sol fixture build`
- `sol fixture inspect`

### 16.2 Add

- `sol bench` — trusted single-case measurement with provenance and repetition
- `sol sweep` — trusted matrix orchestration over `sol bench`
- `sol validate` — standalone Docker validation gate
- `sol run-once` — hidden/internal, machine-oriented single execution for
  container use
- `sol validate-probe` — hidden/internal, runs inside container for cgroup checks

### 16.3 `sol quick-bench` Positioning

- `sol quick-bench` is for quick ad hoc single runs and inner-loop debugging only
- `sol quick-bench` must not be used as reproducible benchmark evidence
- `sol bench` is for trustworthy measured runs
- `sol sweep` is for trustworthy orchestration over `sol bench`

## 17. Build and Release Policy

Trusted mode enforces release builds.

- The harness records build profile in `resolved_case.json` and `provenance.json`
- Trusted mode refuses debug binaries unless `--allow-debug-benchmark` is set
- If the override is used, the verdict includes the reason
- The Makefile target for benchmarking should always use `--release`

## 18. Failure Policy

Do not discard partial evidence.

If a run fails:
- preserve all artifacts collected so far
- emit `verdict.json` with `Invalid` and the failure reason
- the run is not eligible for comparison

If one of N measured repetitions fails:
- preserve its artifacts
- exclude it from summary statistics
- emit `Partial` verdict with reason
- do NOT silently average the remaining runs into a `Valid` result

If a sweep is interrupted:
- completed cases retain their artifacts
- sweep-level `verdict.json` reflects incomplete state
- automatic sweep resume is a non-goal for v1

## 19. Implementation Phases

### Phase 0: Delete Mining And Legacy Harness

1. delete `src/scenario/`, `src/events/`, `src/output/`, `src/runner/`
2. delete `src/commands/mining.rs`, `src/speed_of_light/sweep.rs`
3. delete CLI commands: `Run`, `Attach`, `Compare`, `Analyze`, `sol sweep`
4. delete mining-specific types and re-exports
5. clean Cargo dependencies (`arrow`, `parquet`, `chrono`)
6. salvage generic Docker/cgroup helpers into notes for Phase 2
7. update `src/lib.rs`, `src/commands/mod.rs`, `src/speed_of_light/mod.rs`

Exit criteria:
- `cargo build -p nockchain-bench --release` passes
- `cargo test -p nockchain-bench --release` passes
- remaining CLI: `sample`, `sol quick-bench`, `sol extract`, `sol checkpoint`,
  `sol inspect`, `sol fixture build`, `sol fixture inspect`

### Phase 1: Shared Once-Run Core + Native Trusted Runner

1. create `speed_of_light::harness` module tree
2. define `RequestedCase`, `ResolvedCase`, `Provenance`, `Summary`, `Verdict`
3. extract shared once-run execution from current SOL bench path into library fn
4. implement `sol bench` native mode with repetition loop and cooldown
5. refactor `sol quick-bench` to call the shared library function
6. write artifact tree (`requested_case.json`, `resolved_case.json`,
   `provenance.json`, per-run `result.json`/`profile.json`/`block_timings.ndjson`)
7. compute `summary.json` with median/min/max/MAD/stddev/CV
8. compute `verdict.json`
9. enforce release-build policy

Exit criteria:
- native `sol bench` produces a complete valid artifact tree
- `sol quick-bench` still works as the quick path
- summary statistics are correct for 3+ measured runs

### Phase 2: Docker Trusted Runner

1. implement SOL-specific Docker execution in `speed_of_light::harness::docker`,
   carrying generic helpers from the deleted `runner/docker.rs`
2. implement hidden `sol run-once` subcommand
3. add host/container provenance capture (dual binary identity)
4. add concurrent Docker stats API polling → `container_samples.ndjson`
5. add host/container version skew check
6. support explicit work dir modes
7. capture `raw/docker_inspect.json`, `raw/docker_info.json`

Exit criteria:
- Docker `sol bench` executes replay inside container via `sol run-once`
- emits full artifact tree with both process-level and container-level evidence
- version skew between host and container binary is detected

### Phase 3: Validation Gate

1. implement `sol validate`
2. implement `sol validate-probe` (runs inside container)
3. implement memory-limit verification and allocation sanity probe
4. add validation caching by resource tuple
5. wire validation into Docker `sol bench`: auto-validate before first measured run,
   abort on failure

Exit criteria:
- `sol validate` passes with correct limits, fails with incorrect
- Docker `sol bench` fails fast when limits are not realized

### Phase 4: Sweep Rewrite

1. implement `axes` map matrix schema and cartesian expansion
2. implement `sol sweep` as orchestration over `sol bench`
3. implement single-axis trusted sweep with invariant checking
4. add `--allow-multi-axis`, `--interleave`, `--randomize-order`
5. generate `comparison.json` and optional `comparison.md`
6. generate sweep-level `verdict.json`

Exit criteria:
- single-axis trusted memory sweep produces correct comparison output
- multi-axis sweep rejected without `--allow-multi-axis`
- invariant violations detected and reported

### Phase 5: Documentation And Follow-Through

1. document trusted benchmark protocol
2. document `sol quick-bench` vs `sol bench` distinction
3. document `--blocks` prefix-replay semantics
4. document host/container version policy
5. update any scripts or CI that used deleted mining commands

## 20. Acceptance Criteria

The redesign is acceptable when all of these hold:

1. `MiningScenario` and related subsystems are gone.
2. No trusted SOL path depends on mining-era abstractions.
3. A trusted Docker run records both host and container binary identity.
4. A trusted Docker run proves whether the requested memory limit was realized.
5. A trusted comparison can be traced back to raw per-run artifacts.
6. `sol bench` native and Docker modes share one machine-oriented once-run
   execution contract.
7. `sol quick-bench` remains available as the quick path but is not the source of
   truth for trusted orchestration.
8. `--blocks N` is explicitly documented as prefix replay of the fixture window.
9. Sweeps use `axes` map schema and no longer rely on phase labels or
   mining-oriented naming.
10. `cargo build -p nockchain-bench --release` and
    `cargo test -p nockchain-bench --release` pass after each phase boundary.
