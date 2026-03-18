# nockchain-bench

`nockchain-bench` is the Nockchain benchmarking and replay-analysis crate. It
contains both the CLI binary and the library code used to:

- extract speed-of-light `.solarch` replay archives from checkpoints
- build and inspect unified `.soltest` fixtures from `.solarch` archives
- run ad hoc SOL replay benchmarks for inner-loop work
- run trusted native and Docker SOL benchmarks with persisted artifacts
- validate Docker benchmark environments before replay
- execute trusted benchmark sweeps over matrix-defined cases

At a high level, the crate has two operator-facing modes:

- quick local investigation through `nockchain-bench sol quick-bench`
- auditable benchmark execution through `nockchain-bench sol bench`,
  `nockchain-bench sol validate`, and `nockchain-bench sol sweep`

The trusted harness behavior is specified in
`crates/nockchain-bench/specs/bench-harness-spec.md`. This README is the
operator-facing summary of that harness, including the trusted benchmark
protocol, the boundary between `sol quick-bench` and `sol bench`, the meaning
of `--blocks`, the host/container version policy, and a matrix-sweep example.

We recommend running all benchmarks in `--release` mode unless strictly
necessary. Nockchain performance is very negatively impacted by running on lower
optimization settings.

All command and path examples below assume you are running them from the
`nockchain` repository root.

## Key Commands

- `nockchain-bench sol extract` for archive extraction
- `nockchain-bench sol fixture build` and `fixture inspect` for unified fixture
  workflows
- `nockchain-bench sol quick-bench` for ad hoc replay profiling
- `nockchain-bench sol bench` for trusted measured runs
- `nockchain-bench sol validate` for trusted Docker preflight checks
- `nockchain-bench sol sweep` for trusted benchmark matrices

## Archive And Fixture Workflow

The SOL toolchain has three main artifact types:

- `.chkjam`: a checkpoint used to bootstrap the kernel state
- `.solarch`: an extracted replay archive containing a block range, and
  optionally mempool snapshots
- `.soltest`: a unified fixture that bundles checkpoint + archive + kernel for
  repeatable replay and benchmarking

About `dumb.jam` and `--kernel`:

- `--kernel` is the actual jammed kernel loaded into `NockApp`. It's typically
  given as `assets/dumb.jam`, the Nockchain kernel.
- `sol extract` uses that kernel together with the checkpoint to boot the node
  state and replay blocks while producing the archive.
- `sol fixture build` uses that kernel again while deriving the embedded
  checkpoint for the fixture, then stores the exact kernel bytes inside the
  `.soltest` file. You must use the same kernel jamfile for each.
- Later `sol quick-bench`, `sol bench`, and `sol sweep` runs unpack the embedded
  kernel from the fixture and use it for replay, which keeps the checkpoint,
  archive, and kernel tied to one reproducible bundle.

In practice the workflow is:

1. Use `sol extract` to turn a checkpoint into a `.solarch` archive.
2. Use `sol fixture build` to turn that source archive into a replay-ready
   `.soltest` fixture for a specific benchmark window.
3. Use `sol fixture inspect` to confirm the fixture manifest, embedded heights,
   hashes, and payload sizes before benchmarking.

### `sol extract`

Use `nockchain-bench sol extract` when you have a checkpoint and kernel and need
an archive of accepted blocks for later slicing or fixture construction.

Important behavior:

- `--start-height` is inclusive.
- `--end-height` is inclusive and overrides `--blocks`.
- If `--end-height` is omitted, the command extracts `--blocks` accepted blocks
  starting at `--start-height`.
- `--kernel` selects the jammed kernel binary to load with the checkpoint before
  replaying blocks. The default is `assets/dumb.jam`.
- `--blocks` must be greater than `0` unless `--end-height` is provided.
- `--include-mempool` records mempool snapshots in the archive so later fixture
  builds can preserve them (NOTE this feature is currently untested and likely
  results in an empty mempool)
- If `--output` is omitted, the command writes `blocks_<N>.solarch` or
  `blocks_<start>-<end>.solarch` depending on the requested range.

Example:

```bash
./target/release/nockchain-bench sol extract \
  --checkpoint ./path/to/0.chkjam \
  --kernel ./assets/dumb.jam \
  --start-height 0 \
  --end-height 1000 \
  --output ./tmp/first-1001.solarch
```

That command extracts heights `0..=1000` into
`./tmp/first-1001.solarch`.

### `sol fixture build`

Use `nockchain-bench sol fixture build` when you already have a source
`.solarch` archive and want a reusable `.soltest` fixture for replay or trusted
benchmarks.

The fixture builder does two things:

- builds an embedded checkpoint at exactly `--start-height`
- slices the source archive so the fixture replay payload begins at
  `start_height + 1` and runs through `--end-height` inclusive

This means the source archive must cover both the checkpoint target height and
the requested replay window. `--end-height` must be strictly greater than
`--start-height`.

Important behavior:

- `--archive` must already contain the requested range and enough bootstrap
  prefix to derive the embedded checkpoint at `--start-height`.
- `--kernel` selects the jammed kernel binary used while deriving the embedded
  checkpoint, and those exact kernel bytes are then stored inside the fixture.
- `--include-mempool` controls whether the sliced fixture archive keeps mempool
  snapshots.
- `--work-dir` is used for temporary artifacts such as the sliced archive and
  the derived embedded checkpoint.

Example:

```bash
./target/release/nockchain-bench sol fixture build \
  --archive ./tmp/first-1001.solarch \
  --kernel ./assets/dumb.jam \
  --start-height 0 \
  --end-height 100 \
  --work-dir ./tmp \
  --output ./fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest
```

That command derives an embedded checkpoint at height `0` using the specified
kernel, slices the archive to heights `1..=100`, and packages the checkpoint,
sliced archive, and the same kernel bytes into the output fixture.

### `sol fixture inspect`

Use `nockchain-bench sol fixture inspect` to verify what a `.soltest` fixture
actually contains before using it in `sol quick-bench`, `sol bench`, or
`sol sweep`.

The inspect command prints:

- manifest format version
- source archive path and source event number
- derived checkpoint height and event number
- embedded archive replay range
- whether mempool snapshots are included
- kernel, checkpoint, and archive content hashes
- embedded payload sizes for checkpoint, archive, and kernel

Example:

```bash
./target/release/nockchain-bench sol fixture inspect \
  --fixture ./fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest
```

Use this output to confirm that the fixture range, checkpoint height, and
embedded payload hashes match the data you intended to benchmark.

## Trusted SOL Benchmarks

## Command Roles

- `nockchain-bench sol quick-bench` is the ad hoc path for single-run local
  investigation and inner-loop debugging.
- `nockchain-bench sol bench` is the public trusted interface for measured SOL
  replay runs.
- `nockchain-bench sol validate` preflights a trusted Docker case without
  running replay.
- `nockchain-bench sol sweep` expands a trusted matrix and runs each case
  through `sol bench`.

The trusted Docker path does not depend on `sol quick-bench` output. Inside the
container it uses the hidden `sol run-once` command so native and Docker trusted
runs share the same once-run execution contract.

## Trusted Benchmark Protocol

Use this protocol when you want evidence that can be compared or archived:

1. Build the release binary.
2. Choose a unified `.soltest` fixture and keep that fixture constant across the
   comparison you care about.
3. For trusted `sol bench` and `sol validate`, point `--output` at an existing
   empty directory so the artifact tree starts from a clean root.
4. Prefer at least `--measured-runs 3` for trusted results, especially in
   Docker mode.
5. Treat `summary.json`, `verdict.json`, `provenance.json`, and the per-run
   artifacts under `runs/` as the record of truth.

Trusted mode records build/profile identity in `resolved_case.json` and
`provenance.json`. Release builds are required unless you intentionally use
`--allow-debug-benchmark`, in which case the verdict records that override.

### Artifact Expectations

A trusted single-case run writes an auditable tree rooted at `--output`,
including:

- `schema_version.txt`
- `requested_case.json`
- `resolved_case.json`
- `provenance.json`
- `summary.json`
- `verdict.json`
- `runs/<run-id>/...`
- `raw/...` host and Docker evidence files

Docker runs also persist validation and container evidence so a result can be
traced back to raw host/container facts.

Standalone `sol validate` writes the same requested/resolved-case scaffold plus
`validation.json` and raw Docker evidence, and also maintains a sibling
`validation_cache.json` next to the chosen output directory so repeated
preflights with the same engine/image/limit tuple can reuse the cached result.

When trusted sweep profiling is enabled, each profiled case also writes:

- `cpu_profile.json`
- `profiles/samply-profile.json.gz`
- `profile-run/...`

`sol quick-bench` does not persist that trusted case-local profiling tree. Its
CPU profiling mode copies only the raw profile artifact to the explicit
`--cpu-profile-output` path and then removes its temporary working directory.

## `sol quick-bench` vs `sol bench`

Use `sol quick-bench` when you want speed and iteration:

- fast ad hoc runs
- inner-loop profiling
- one-off debugging while changing replay behavior
- optional extra CPU profiling pass via `--cpu-profiler samply`

Do not use `sol quick-bench` as reproducible benchmark evidence. It is not the
trusted orchestration surface and is not the source of truth for published
comparisons.

Use `sol bench` when you want trustworthy measurements:

- repeated measured runs with cooldown control
- persisted requested/resolved case records
- explicit provenance and verdicts
- one shared trusted contract across native and Docker backends

Direct `sol bench` intentionally has no CPU-profiling flags. Trusted CPU
profiling is exposed through `sol sweep`, which layers one extra profiled pass
per case on top of the normal trusted run contract.

Use `sol sweep` when you need a trusted comparison across a matrix. It is an
orchestrator over `sol bench`, not a separate measurement engine.

CPU profiling is intentionally separate from trusted measured-run statistics:

- `sol quick-bench --cpu-profiler samply --cpu-profile-output <path>` runs the
  normal quick benchmark, then one extra profiled replay pass and copies the
  raw `samply` artifact to the requested path
- `sol sweep --cpu-profiler samply` runs warmups/measured runs normally for
  each case, then one extra profiled replay pass per case
- native profiling preflights Linux perf access and fails early when
  `kernel.perf_event_paranoid > 1`
- trusted `summary.json` and verdict math exclude that extra profiled pass
- if CPU profiling itself fails, the explicitly profiled trusted case is marked
  invalid rather than silently degrading

## SOL Sweep Matrix

`sol sweep` reads a JSON matrix file with three top-level keys:

- `benchmark`: currently must be `"sol-replay"`
- `base`: the template requested case used as the starting point for every
  expanded sweep case
- `axes`: a map of field name to value list; the sweep expands one case per
  value combination

Simple matrix example:

```json
{
  "benchmark": "sol-replay",
  "base": {
    "fixture": "./fixtures/first-100.soltest",
    "warmup_runs": 0,
    "measured_runs": 3,
    "cooldown_secs": 0,
    "mode": {
      "docker": {
        "image_tag": "nockchain-bench:local",
        "work_dir_mode": "DockerTmpfs"
      }
    }
  },
  "axes": {
    "memory_limit": ["4g", "8g"]
  }
}
```

That matrix produces two cases. Both start from the same `base` template, and
the only field that varies is `memory_limit`.

### `base`

`base` is the default requested case. For each expanded case, the sweep clones
`base` and then applies that case's axis assignments on top.

| Property | Type | Default when omitted | What it controls | Example |
| --- | --- | --- | --- | --- |
| `fixture` | string/path | required | Fixture path used as the default input for every expanded case. | `"./fixtures/first-100.soltest"` |
| `blocks` | integer | `0` | Prefix replay length. `0` means replay the full fixture window. | `100` |
| `skip_genesis` | boolean | `false` | Whether replay skips the genesis entry. | `true` |
| `enable_checkpointing` | boolean | `true` | Whether replay-generated checkpoints are enabled during the run. | `false` |
| `checkpoint_every_blocks` | integer | `0` | Write a replay checkpoint every `N` accepted blocks. `0` disables periodic checkpoints. | `50` |
| `profile_memory` | boolean | `false` | Enable process/container memory sampling during the run. | `true` |
| `profile_interval_ms` | integer | `500` | Memory profiling sample interval in milliseconds when profiling is enabled. | `250` |
| `warmup_runs` | integer | `1` | Number of warmup runs before measured runs begin. | `0` |
| `measured_runs` | integer | `5` | Number of measured runs included in the summary and verdict. Trusted runs still require at least `3`. | `3` |
| `cooldown_secs` | integer | `10` | Delay between runs in seconds. | `0` |
| `label` | string | unset | Optional human label persisted with the case metadata. | `"docker-8g"` |
| `mode` | object | `native` | Execution backend template for the sweep. Use this to select Docker mode and set Docker-specific defaults. | `{ "docker": { "image_tag": "nockchain-bench:local", "memory_limit": "8g", "work_dir_mode": "DockerTmpfs" } }` |

`mode` currently selects one backend for the entire sweep: every expanded case
is either native or Docker, not a mix of both. Mixed native and Docker cases in
a single matrix are not supported yet. Support for mixed-backend sweeps is
planned, but this release still requires a sweep to choose one execution mode.

`mode` may specify either `native` or `docker`, not both. When `mode` is
omitted, the sweep defaults to native execution.

When `mode.docker` is used, omitted Docker subfields fall back to defaults
listed in the table below.

#### `mode.docker`

`mode.docker` supplies Docker defaults for every expanded case in the sweep.

| Property | Type | Default when omitted | What it controls | Example |
| --- | --- | --- | --- | --- |
| `image_tag` | string | empty string | Docker image tag used for trusted Docker cases. Trusted execution still requires a non-empty value after axis overrides. | `"nockchain-bench:local"` |
| `memory_limit` | string | empty string | Docker memory limit passed to the container. Trusted execution still requires a positive value after axis overrides. | `"8g"` |
| `cpuset` | string | unset | Docker CPU affinity mask/list. | `"0-3"` |
| `cpu_quota` | integer | unset | Docker CPU quota (`--cpu-quota`). | `200000` |
| `cpu_period` | integer | unset | Docker CPU period (`--cpu-period`). | `100000` |
| `work_dir_mode` | string | `DockerTmpfs` | Docker work directory strategy. Valid values are `HostBind`, `DockerVolume`, and `DockerTmpfs`. | `"DockerTmpfs"` |
| `allow_version_skew` | boolean | `false` | Allow host/container binary identity mismatch without treating the Docker run as invalid by default. | `true` |

`base` is a template, not necessarily a runnable case by itself. Validity is
checked on the final expanded cases after axis overrides are applied.

Expanded-case validity requirements:

- `measured_runs >= 3`
- `checkpoint_every_blocks > 0` requires `enable_checkpointing = true`
- Docker cases must end up with a non-empty `image_tag` and a positive
  `memory_limit`
- Docker cases must not set empty `cpuset` values, and provided `cpu_quota` /
  `cpu_period` values must be positive

This means a Docker sweep may leave `image_tag` or `memory_limit` out of
`base` if those values are supplied by axes, as long as every final expanded
case still resolves to a valid trusted Docker request.

### `axes`

`axes` is a map from field name to a list of values. The sweep computes the
cartesian product of those value lists.

Rules:

- the matrix must contain at least one axis
- each axis must have at least one value
- the sweep expands one case per value combination across all provided axes
- axis values override the corresponding field from `base`

This override behavior is general. For example, `base.mode.docker.memory_limit =
"8g"` means the default Docker memory limit is `8g`, and a `memory_limit` axis
such as `["4g", "8g"]` produces cases with `memory_limit = "4g"` and
`memory_limit = "8g"`.

The `fixture` axis is supported. It is the mechanism for sweeping across more
than one `.soltest` fixture.

- If there is no `fixture` axis, every case uses `base.fixture`.
- If there is a `fixture` axis, each expanded case replaces `base.fixture` with
  the assigned fixture path.
- `base.fixture` is still required because it is the only non-defaulted required
  field in the `base` schema.

When `fixture` is an axis, fixture identity is allowed to differ across cases.
That means changes in fixture hash and fixture manifest are treated as expected
axis variation rather than invariant violations. This matters because changing
fixture usually changes the embedded checkpoint, archive window, mempool
setting, and embedded kernel together.

Fixture-axis example:

```json
{
  "benchmark": "sol-replay",
  "base": {
    "fixture": "./fixtures/a.soltest",
    "warmup_runs": 0,
    "measured_runs": 3,
    "cooldown_secs": 0
  },
  "axes": {
    "fixture": [
      "./fixtures/a.soltest",
      "./fixtures/b.soltest"
    ]
  }
}
```

That matrix produces one case for `a.soltest` and one case for `b.soltest`.

Supported axis names:

| Axis | Type | Default when omitted | What it controls | Example |
| --- | --- | --- | --- | --- |
| `blocks` | integer | `0` | Prefix replay length. `0` means replay the full fixture window. | `100` |
| `skip_genesis` | boolean | `false` | Whether to skip the genesis entry during replay. | `true` |
| `enable_checkpointing` | boolean | `true` | Whether replay-generated checkpoints are enabled during the run. | `false` |
| `checkpoint_every_blocks` | integer | `0` | Write a replay checkpoint every `N` accepted blocks. `0` disables periodic checkpoints. | `50` |
| `profile_memory` | boolean | `false` | Enable process/container memory sampling during the run. This is what turns on page-fault sampling. | `true` |
| `profile_interval_ms` | integer | `500` | Memory profiling sample interval in milliseconds when profiling is enabled. | `250` |
| `warmup_runs` | integer | `1` | Number of warmup runs before measured runs begin. | `0` |
| `measured_runs` | integer | `5` | Number of measured runs included in the summary and verdict. Trusted runs still require at least `3`. | `3` |
| `cooldown_secs` | integer | `10` | Delay between runs in seconds. | `0` |
| `fixture` | string/path | required | Fixture path for the trusted case. | `"./fixtures/first-100.soltest"` |
| `label` | string | unset | Human label persisted with the case metadata. | `"docker-8g"` |
| `image_tag` | string | empty string in Docker mode | Docker image tag used for trusted Docker cases. Docker-only. A trusted Docker run still requires a non-empty value. | `"nockchain-bench:local"` |
| `memory_limit` | string | empty string in Docker mode | Docker memory limit passed to the container. Docker-only. A trusted Docker run still requires a positive value. | `"8g"` |
| `cpuset` | string | unset | Docker CPU affinity mask/list. Docker-only. | `"0-3"` |
| `cpu_quota` | integer | unset | Docker CPU quota (`--cpu-quota`). Docker-only. | `200000` |
| `cpu_period` | integer | unset | Docker CPU period (`--cpu-period`). Docker-only. | `100000` |
| `work_dir_mode` | string | `DockerTmpfs` in Docker mode | Docker work directory strategy. Valid values are `HostBind`, `DockerVolume`, and `DockerTmpfs`. Docker-only. | `"DockerTmpfs"` |
| `allow_version_skew` | boolean | `false` | Allow host/container binary identity mismatch without treating the Docker run as invalid by default. Docker-only. | `true` |

Docker-only axes require `base.mode.docker`; using them with a native base case
is an error. In Docker mode, the parser defaults `image_tag` and `memory_limit`
to empty strings, but trusted execution validation still requires a non-empty
image tag and a positive memory limit.

## CPU Profiling with `samply`

- `sol quick-bench` supports `--cpu-profiler samply`, `--cpu-profile-rate`, and
  `--cpu-profile-output`
- `sol sweep` supports `--cpu-profiler samply` and `--cpu-profile-rate`
- direct trusted `sol bench` does not expose CPU-profiling flags
- native profiling wraps the hidden `sol run-once` entrypoint with host
  `samply`
- Docker profiling runs `samply record` inside a dedicated replay container so
  the captured profile is for the replay work, not just the host orchestrator
- wrapping the top-level `sol sweep` command in `samply record` is not
  equivalent for Docker; that only captures host-side orchestration
- on Linux, `samply` requires `kernel.perf_event_paranoid <= 1` for
  unprivileged profiling
- native profiling checks that Linux setting before launching `samply`, so the
  operator gets a direct error instead of an opaque profiler failure
- on high-core Linux hosts, `samply` may also fail with `mmap failed` when it
  tries to set up profiling across all CPUs; for single-threaded workloads, a
  practical workaround is to run the benchmark under `taskset`, for example
  `taskset -c 0` or `taskset -c 0-3`
- Docker profiling additionally requires both `nockchain-bench` and `samply` in
  the image, plus container perf permissions that allow sampling

Tracked Docker image builds:

```bash
scripts/build_nockchain_bench_image.sh --variant standard --tag nockchain-bench:local
scripts/build_nockchain_bench_image.sh --variant profiling --tag nockchain-bench:local-samply
```

- the script builds `target/release/nockchain-bench` by default before staging
  a temporary Docker build context
- the profiling-enabled image is only required when using Docker CPU profiling
- Docker CPU profiling still requires container perf permissions at runtime

Quick benchmark CPU profiling example:

```bash
./target/release/nockchain-bench sol quick-bench \
  --fixture ./fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest \
  --cpu-profiler samply \
  --cpu-profile-output ./tmp/quick-bench-profile.json.gz
```

Trusted sweep CPU profiling example:

```bash
./target/release/nockchain-bench sol sweep \
  --matrix ./tmp/native-sweep-matrix.json \
  --output ./tmp/native-sweep-out \
  --cpu-profiler samply \
  --cpu-profile-rate 1000 \
  --comparison-markdown
```

## `--blocks` Prefix Replay Semantics

`--blocks N` is a prefix replay control, not an arbitrary slicing mechanism.

- `--blocks 0` means replay the full fixture archive window.
- `--blocks N` means replay the first `N` accepted blocks from the fixture's
  archive window.
- The starting point comes from the fixture manifest and resolved start height.
- The same prefix semantics apply to `sol quick-bench`, `sol bench`, and every
  expanded case in `sol sweep`.

If you need a different replay window, build a new fixture rather than treating
`--blocks` as an in-fixture range selector.

## Host/Container Version Policy

Trusted Docker execution records both host and container binary identity.

By default, a trusted Docker run is only valid when the host and container agree
on binary version and git commit identity. If they differ:

- `sol bench` marks the run invalid by default
- `--allow-version-skew` permits the run to continue
- the resulting provenance still records both identities and the override

Trusted Docker provenance also records the image tag, resolved image digest,
container id, Docker engine/context data, and realized cgroup values such as
`memory.max`, `memory.current`, `cpuset`, and `cpu.max` when available.

Use `sol validate` when you want to confirm the container runtime can realize
the requested limits before spending time on measured replay.

`sol validate` uses the same Docker request shape as trusted Docker `sol bench`
and requires the same `--image-tag`, `--memory-limit`, and `--work-dir-mode`
inputs. Like trusted bench, its `--output` directory must already exist and be
empty.

## Practical Examples

Native trusted bench:

```bash
./target/release/nockchain-bench sol bench \
  --fixture ./fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest \
  --output ./tmp/native-bench-example \
  --warmup-runs 0 \
  --measured-runs 3 \
  --cooldown-secs 0
```

Docker trusted bench:

```bash
./target/release/nockchain-bench sol bench \
  --fixture ./fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest \
  --output ./tmp/docker-bench-example \
  --image-tag nockchain-bench:local \
  --memory-limit 8g \
  --work-dir-mode docker-tmpfs \
  --warmup-runs 0 \
  --measured-runs 3 \
  --cooldown-secs 0
```

Docker validation preflight:

```bash
./target/release/nockchain-bench sol validate \
  --fixture ./fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest \
  --output ./tmp/docker-validate-example \
  --image-tag nockchain-bench:local \
  --memory-limit 8g \
  --work-dir-mode docker-tmpfs
```

Trusted sweep with a matrix file:

`matrix.json`

```json
{
  "benchmark": "sol-replay",
  "base": {
    "fixture": "./fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest",
    "warmup_runs": 0,
    "measured_runs": 3,
    "cooldown_secs": 0,
    "mode": {
      "docker": {
        "image_tag": "nockchain-bench:local",
        "work_dir_mode": "DockerTmpfs"
      }
    }
  },
  "axes": {
    "memory_limit": ["4g", "8g", "16g"]
  }
}
```

```bash
./target/release/nockchain-bench sol sweep \
  --matrix ./tmp/memory-matrix.json \
  --output ./tmp/live-sol-sweep \
  --comparison-markdown
```

The sweep writes per-case outputs under `cases/` plus top-level
`schema_version.txt`, `matrix.json`, `matrix_expanded.json`, `schedule.json`,
`comparison.json`, and `verdict.json`.

Passing `--comparison-markdown` also writes `comparison.md`, a human-readable
Markdown rendering of the same comparison data for quick review in a terminal,
editor, or PR.

Multi-axis trusted sweep example:

`matrix-multi-axis.json`

```json
{
  "benchmark": "sol-replay",
  "base": {
    "fixture": "./fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest",
    "blocks": 0,
    "enable_checkpointing": true,
    "checkpoint_every_blocks": 0,
    "profile_memory": true,
    "profile_interval_ms": 500,
    "warmup_runs": 0,
    "measured_runs": 3,
    "cooldown_secs": 0,
    "label": "docker-sol-sweep",
    "mode": {
      "docker": {
        "image_tag": "nockchain-bench:local",
        "memory_limit": "8g",
        "cpuset": "0-3",
        "cpu_quota": 200000,
        "cpu_period": 100000,
        "work_dir_mode": "DockerTmpfs",
        "allow_version_skew": false
      }
    }
  },
  "axes": {
    "memory_limit": ["4g", "8g"],
    "work_dir_mode": ["DockerTmpfs", "DockerVolume"],
    "allow_version_skew": [false, true]
  }
}
```

```bash
./target/release/nockchain-bench sol sweep \
  --matrix ./tmp/matrix-multi-axis.json \
  --output ./tmp/live-sol-sweep-multi-axis \
  --comparison-markdown
```
