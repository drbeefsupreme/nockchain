# Spec: Trustworthy SOL Benchmark Harness v5

## Status

This spec supersedes `BENCH_HARNESS_SPEC.md`, `BENCH_HARNESS_SPEC_v3.md`,
and `BENCH_HARNESS_SPEC_v4.md`.

v5 preserves the benchmark protocol and acceptance criteria from v4, but
revises the execution architecture:

- one machine-oriented once-run engine
- one shared trusted orchestrator
- backend adapters for native and Docker execution

The intent is to avoid separate native and Docker trusted-run stacks that drift
in policy, artifacts, or summary logic over time.

## 1. Purpose

Build a trustworthy benchmark harness for SOL replay workloads in
`nockchain-bench`.

The harness must make four things explicit and auditable:
- requested inputs
- resolved execution plan
- realized execution environment
- raw measurement evidence

Mining benchmarks are out of scope and are removed before any new harness work
begins.

## 2. Scope

This spec applies only to SOL replay benchmarking driven by `SolBenchRunner`
and the unified `.soltest` fixture format.

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
11. Native and Docker trusted runs share one orchestration contract, not only
    one measurement engine.

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
mining-specific and should be carried into the new harness module:

- Docker connection logic (multi-socket discovery, ping)
- `ContainerStats` struct and `from_docker_stats` (v1/v2 cgroup handling)
- `parse_memory_limit`
- `parse_proc_stat_faults`
- `calculate_cpu_percent`
- page fault reading via `docker exec` of `/proc/1/stat`

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

## 5. Module Layout

Do not repurpose the mining-era `runner/docker.rs`.

Create a new SOL-specific harness module tree under
`src/speed_of_light/harness/`.

Suggested layout:

```text
src/speed_of_light/harness/
├── mod.rs
├── case.rs          # RequestedCase, ResolvedCase, execution/backend schema
├── artifacts.rs     # output tree writing, raw/log/result persistence
├── provenance.rs    # host/container provenance capture
├── summary.rs       # stats aggregation, verdicts
├── execute.rs       # shared once-run execution contract
├── orchestrate.rs   # shared trusted orchestration pipeline
├── native.rs        # native backend adapter
├── docker.rs        # Docker backend adapter and Docker helpers
├── validate.rs      # Docker validation gate and probe protocol
└── sweep.rs         # matrix expansion and orchestration only
```

Reason:
- the old `runner/` module is not actually generic
- the new harness is SOL-specific by scope
- trusted policy should be centralized in one orchestrator
- backend-specific setup should be isolated from common artifact and summary
  behavior

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
- replay at most the first `N` replayable blocks from the fixture's archive
  window

It does not mean:
- slice the fixture file physically
- skip into the middle of the fixture window
- benchmark an arbitrary offset without building a different fixture

This matches the current behavior and is enough for a minimal trustworthy
protocol.

### 7.2 Fixture Handling

In v1:
- the fixture is treated as an immutable input blob
- full fixture extraction is acceptable
- fixture bind-mounting into Docker is acceptable
- partial extraction optimization is explicitly out of scope

Future optimization may add partial extraction or offset-based replay, but that
is not part of this rewrite.

## 8. Execution Architecture

The execution model has three layers:

1. one shared once-run engine
2. one shared trusted orchestrator
3. backend adapters for native and Docker

The central orchestration contract is:
"run one resolved case N times under one backend, persist all artifacts, and
emit summary and verdict."

### 8.1 Shared Once-Run Engine

Introduce one shared execution entrypoint used by both:
- native trusted runs
- Docker trusted runs
- `sol quick-bench`

This must be implemented as a library-level operation, not by scraping
human-readable CLI output.

The once-run engine is responsible only for:
- executing one resolved SOL replay case once
- collecting machine-oriented per-run results
- persisting per-run artifacts for that one execution

It is not responsible for:
- warmup/measured repetition policy
- cooldown
- verdicts
- cross-run summary
- backend-specific validation

### 8.2 Hidden `sol run-once`

A hidden/internal CLI is required for container execution, for example:

- `nockchain-bench sol run-once --resolved-case /bench/input/resolved_case.json --run-dir /bench/output/run-0`

`sol run-once` is a machine-oriented wrapper over the shared once-run engine.

It exists so the host-side Docker backend can invoke the same execution path
inside the container without parsing stdout or reimplementing replay logic.

### 8.3 Shared Trusted Orchestrator

Trusted benchmarking must use one shared orchestration pipeline for both native
and Docker backends.

The orchestrator owns:
- output-root preparation
- `requested_case.json`
- `resolved_case.json`
- warmup and measured run scheduling
- cooldown policy
- run-failure accounting
- `summary.json`
- `verdict.json`
- shared failure handling policy

The orchestrator does not know Docker API details or native-only setup details.
It delegates environment-specific behavior to a backend adapter.

### 8.4 Backend Adapter Contract

Each backend adapter must provide a small, explicit contract to the
orchestrator:

- resolve backend-specific static facts needed before execution
- prepare the runtime environment
- execute one run
- capture backend-specific provenance
- capture backend-specific raw evidence
- clean up backend-owned resources

The backend adapter must not own:
- summary math
- verdict policy
- repetition scheduling
- artifact naming conventions shared across backends

### 8.5 Relationship To `sol bench`

`sol bench` is the public trusted interface.

It resolves the requested case, selects a backend, and runs the shared
orchestrator.

There is one trusted command:
- `sol bench` for trustworthy measured runs

There are multiple backends behind it:
- native backend
- Docker backend

### 8.6 Relationship To `sol quick-bench`

`sol quick-bench` remains the quick ad hoc interface.

`sol quick-bench` must not depend on trusted orchestration.

Recommended structure:
- factor replay execution into the shared once-run engine
- let `sol quick-bench` call that engine directly
- let `sol bench` call the shared trusted orchestrator
- let Docker mode invoke `sol run-once` inside the container

One measurement engine, two public roles:
- `sol quick-bench` for ad hoc use
- `sol bench` for trusted measurement

## 9. Data Model

### 9.1 RequestedCase

Requested input only. No auto-captured facts.

```rust
pub struct RequestedCase {
    pub benchmark: String,
    pub label: Option<String>,

    pub fixture_path: PathBuf,
    pub blocks: u64,
    pub skip_genesis: bool,

    pub enable_checkpointing: bool,
    pub checkpoint_every_blocks: u64,

    pub profile_memory: bool,
    pub profile_interval_ms: u64,

    pub execution: ExecutionRequest,
    pub threads: u32,

    pub warmup_runs: u32,
    pub measured_runs: u32,
    pub cooldown_secs: u64,
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
        allow_version_skew: bool,
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
- fixture SHA256
- embedded fixture manifest
- schema version
- build profile
- tool version and commit
- normalized backend configuration suitable for handoff to the backend adapter

Does not include runtime facts like container id or realized cgroup values.

### 9.3 Provenance

Realized environment captured after setup and before the first measured block.

Provenance must distinguish:
- shared run identity
- host-side identity
- backend-specific realized runtime identity

`provenance.json` must include:
- schema version
- capture timestamp
- host identity (hostname, kernel, OS, total memory, CPU count, CPU model)
- git identity (commit, branch, dirty, commit date)
- fixture identity (path, SHA256, embedded `SolFixtureManifest`)
- binary identity (version, commit, profile)

For Docker mode, additionally record:
- `host_binary`
- `container_binary`
- image tag
- image digest
- container id
- Docker engine version
- Docker context
- cgroup version
- storage driver
- realized `memory.max`
- realized `memory.current` snapshot
- realized cpuset
- realized `cpu.max` or quota/period

### 9.4 Version Skew Policy

Trusted Docker mode requires host/container binary version agreement by default.

- record host binary version/commit
- record container binary version/commit
- mark the run `Invalid` unless they match, or unless
  `--allow-version-skew` is set

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
├── validation.json                 # Docker only, Phase 3+
├── verdict.json
├── summary.json
├── raw/
│   ├── host_env.json
│   ├── docker_inspect.json         # Docker only
│   ├── docker_info.json            # Docker only
│   └── container_env.json          # Docker only
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

Artifact ownership is explicit:

- orchestrator-owned:
  - `schema_version.txt`
  - `requested_case.json`
  - `resolved_case.json`
  - `summary.json`
  - `verdict.json`
- backend-owned additions:
  - backend-specific provenance fields
  - backend-specific raw evidence under `raw/`
  - backend-specific per-run evidence such as `container_samples.ndjson`

Notes:
- warmups are persisted but excluded from summary statistics
- raw Docker evidence is retained alongside parsed provenance
- `block_timings.ndjson` is line-delimited
  `{"height": N, "duration_ms": F}` per block
- `container_samples.ndjson` is line-delimited Docker stats API output sampled
  at `profile_interval_ms`
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
sort stability. Multi-axis:
`case-000-memory_limit_4g+checkpoint_every_10`.

Do not use timestamps or labels as the primary case identifier.

## 11. Backend Models

### 11.1 Native Backend

The native backend runs the shared once-run engine directly on the host.

It is responsible for:
- host-side environment preparation
- host provenance capture
- direct invocation of the once-run engine

It is not a separate trusted runner architecture. It is one backend
implementation of the shared orchestrator contract.

### 11.2 Docker Backend Core Rule

In Docker mode, the SOL replay workload must run inside the constrained
container.

### 11.3 What The Docker Backend Does On The Host

The Docker backend on the host:
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

### 11.4 What The Docker Backend Does In The Container

The container executes the shared once-run engine via hidden/internal CLI:

```bash
nockchain-bench sol run-once \
  --resolved-case /bench/input/resolved_case.json \
  --run-dir /bench/output/run-0
```

The container does not run the public `sol quick-bench` command. The trusted
Docker path must not depend on human-readable stdout.

### 11.5 Image Requirements

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

### 11.6 Work Directory Modes

Explicit and benchmark-relevant:
- `HostBind`
- `DockerVolume`
- `DockerTmpfs`

No silent default in trusted Docker mode. Storage mode changes I/O behavior,
which changes replay results. The user must choose explicitly.

## 12. Validation Gate

Trusted Docker runs require validation, but validation remains a distinct layer
on top of the Docker backend rather than part of the core orchestrator.

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

OOM testing is not a mandatory gate. It is destructive and complicates
recovery. Optional diagnostic mode only (`sol validate --stress-oom`).

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

Trust hierarchy, most trusted first:
1. `SolBenchResults`
2. `MemoryProfile`
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
- `mad`
- `stddev`
- `cv`
- `values`

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

If `cv` exceeds a configured threshold (default 0.10) on throughput, the
verdict is `Partial` with reason.

High spread does not trigger automatic extra runs in v1. The harness flags
instability and records the reason. The operator decides whether to rerun with
`--measured-runs N`.

## 15. Sweep Semantics

### 15.1 Matrix Schema

The matrix always uses `axes` as a map of axis name to value list.

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

Multi-axis example:

```json
{
  "benchmark": "sol-replay",
  "base": { "...": "..." },
  "axes": {
    "memory_limit": ["4g", "32g"],
    "checkpoint_every_blocks": [5, 10, 20]
  }
}
```

Without `--allow-multi-axis`, a matrix with more than one axis is an error.

### 15.2 Default Sweep Policy

Trusted sweeps are single-axis by default.

### 15.3 Scheduling

Allowed modes:
- sequential
- `--interleave`
- `--randomize-order`

Not allowed in trusted mode by default:
- concurrent measured execution

### 15.4 Cooldown

`cooldown_secs` applies between all runs: warmup-to-warmup, warmup-to-measured,
and measured-to-measured.

### 15.5 Invariants

Across a trusted comparison, all non-axis fields must remain constant,
including:
- fixture SHA256 and manifest
- git commit and dirty state
- build profile
- execution mode
- image digest
- work dir mode
- checkpointing config
- thread count
- CPU control policy
- host identity unless explicitly overridden
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

- `sol bench`
- `sol sweep`
- `sol validate`
- `sol run-once`
- `sol validate-probe`

### 16.3 `sol quick-bench` Positioning

- `sol quick-bench` is for quick ad hoc single runs and inner-loop debugging
  only
- `sol quick-bench` must not be used as reproducible benchmark evidence
- `sol bench` is for trustworthy measured runs
- `sol sweep` is for trustworthy orchestration over `sol bench`

## 17. Build and Release Policy

Trusted mode enforces release builds.

- the harness records build profile in `resolved_case.json` and
  `provenance.json`
- trusted mode refuses debug binaries unless `--allow-debug-benchmark` is set
- if the override is used, the verdict includes the reason
- benchmark make targets should always use `--release`

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
- do not silently average the remaining runs into a `Valid` result

If environment setup fails:
- preserve any raw evidence already captured
- emit a structured error artifact when possible
- do not fall back to another backend automatically

## 19. Implementation Phases

### Phase 0: Delete Mining And Legacy Harness

1. delete mining-specific subsystems and CLI surfaces
2. clean Cargo dependencies
3. salvage generic Docker helpers into the new harness area
4. preserve only SOL-focused commands

Exit criteria:
- deleted mining code is gone
- crate builds and tests pass in release mode

### Phase 1: Shared Once-Run Core + Shared Orchestrator + Native Backend

1. implement the shared once-run engine
2. implement the shared trusted orchestrator
3. implement the native backend adapter on top of that orchestrator
4. refactor `sol quick-bench` to call the shared once-run engine
5. write the trusted native artifact tree
6. compute `summary.json`
7. compute `verdict.json`
8. enforce release-build policy

Exit criteria:
- native `sol bench` produces a complete valid artifact tree
- `sol quick-bench` still works as the quick path
- summary statistics are correct for 3+ measured runs
- native trusted execution uses the shared orchestrator, not a separate runner

### Phase 2: Docker Backend On The Shared Orchestrator

1. implement SOL-specific Docker backend logic in
   `speed_of_light::harness::docker`
2. implement hidden `sol run-once`
3. add Docker execution to `ExecutionRequest`
4. add host/container provenance capture
5. add concurrent Docker stats API polling to `container_samples.ndjson`
6. add host/container version skew check
7. support explicit work dir modes
8. capture `raw/docker_inspect.json`, `raw/docker_info.json`, and
   `raw/container_env.json`
9. keep the orchestrator shared; do not add a second trusted-run control flow

Exit criteria:
- Docker `sol bench` executes replay inside the container via `sol run-once`
- Docker and native trusted runs both use the same shared orchestrator
- full artifact tree is emitted with both process-level and container-level
  evidence
- version skew between host and container binary is detected

### Phase 3: Validation Gate

1. implement `sol validate`
2. implement `sol validate-probe`
3. implement memory-limit verification and allocation sanity probe
4. add validation caching by resource tuple
5. wire validation into Docker `sol bench`: auto-validate before first measured
   run and abort on failure

Exit criteria:
- `sol validate` passes with correct limits and fails with incorrect limits
- Docker `sol bench` fails fast when limits are not realized

### Phase 4: Sweep Rewrite

1. implement `axes` map matrix schema and cartesian expansion
2. implement `sol sweep` as orchestration over `sol bench`
3. implement single-axis trusted sweep with invariant checking
4. add `--allow-multi-axis`, `--interleave`, and `--randomize-order`
5. generate `comparison.json` and optional `comparison.md`
6. generate sweep-level `verdict.json`

Exit criteria:
- single-axis trusted memory sweep produces correct comparison output
- multi-axis sweep is rejected without `--allow-multi-axis`
- invariant violations are detected and reported

### Phase 5: Documentation And Follow-Through

1. document trusted benchmark protocol
2. document `sol quick-bench` vs `sol bench`
3. document `--blocks` prefix-replay semantics
4. document host/container version policy
5. update scripts or CI that used deleted mining commands

## 20. Acceptance Criteria

The redesign is acceptable when all of these hold:

1. `MiningScenario` and related subsystems are gone.
2. No trusted SOL path depends on mining-era abstractions.
3. A trusted Docker run records both host and container binary identity.
4. A trusted Docker run proves whether the requested memory limit was realized.
5. A trusted comparison can be traced back to raw per-run artifacts.
6. `sol bench` native and Docker modes share one machine-oriented once-run
   execution contract.
7. `sol bench` native and Docker modes share one trusted orchestration
   contract.
8. `sol quick-bench` remains available as the quick path but is not the source
   of truth for trusted orchestration.
9. `--blocks N` is explicitly documented as prefix replay of the fixture
   window.
10. Sweeps use `axes` map schema and no longer rely on phase labels or
    mining-oriented naming.
11. `cargo build -p nockchain-bench --release` and
    `cargo test -p nockchain-bench --release` pass after each phase boundary.
