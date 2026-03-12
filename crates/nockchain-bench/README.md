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

About `chunk_size`:

- `chunk_size` is the batch size used by `sol extract` when it asks the running
  kernel for block ranges.
- With the default `chunk_size` of `8`, extraction works in windows like
  `0..=7`, `8..=15`, `16..=23`, and so on until the requested end height.
- Larger values mean fewer, larger extraction range queries. Smaller values mean
  more, smaller queries.
- `chunk_size` does not change replay semantics once a fixture has been built.
- `sol fixture build` records `chunk_size` in the fixture manifest as provenance
  metadata so later inspection shows how the archive/fixture was prepared.

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
- `--chunk-size` must be greater than `0`. It controls how many heights are
  requested per extraction range query. The default is `8`.
- `--include-mempool` records mempool snapshots in the archive so later fixture
  builds can preserve them (NOTE this feature is currently untested and likely
  results in an empty mempool)
- If `--output` is omitted, the command writes `blocks_<N>.solarch` or
  `blocks_<start>-<end>.solarch` depending on the requested range.

Example:

```bash
/shared/nockchain/target/release/nockchain-bench sol extract \
  --checkpoint /shared/Dropbox/zorp/agents/nockchain/0.chkjam \
  --kernel /shared/Dropbox/zorp/agents/nockchain/assets/dumb.jam \
  --start-height 0 \
  --end-height 1000 \
  --chunk-size 8 \
  --output /shared/nockchain/tmp/first-1001.solarch
```

That command extracts heights `0..=1000` into
`/shared/nockchain/tmp/first-1001.solarch`.

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
- `--chunk-size` is recorded in the fixture manifest as archive-preparation
  metadata. The default is `8`.
- `--work-dir` is used for temporary artifacts such as the sliced archive and
  the derived embedded checkpoint.

Example:

```bash
/shared/nockchain/target/release/nockchain-bench sol fixture build \
  --archive /shared/nockchain/tmp/first-1001.solarch \
  --kernel /shared/Dropbox/zorp/agents/nockchain/assets/dumb.jam \
  --start-height 0 \
  --end-height 100 \
  --chunk-size 8 \
  --work-dir /shared/nockchain/tmp \
  --output /shared/nockchain/fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest
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
- recorded chunk size
- kernel, checkpoint, and archive content hashes
- embedded payload sizes for checkpoint, archive, and kernel

Example:

```bash
/shared/nockchain/target/release/nockchain-bench sol fixture inspect \
  --fixture /shared/nockchain/fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest
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
3. For trusted `sol bench`, point `--output` at an existing empty directory so
   the artifact tree starts from a clean root.
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

- `requested_case.json`
- `resolved_case.json`
- `provenance.json`
- `summary.json`
- `verdict.json`
- `runs/<run-id>/...`
- `raw/...` host and Docker evidence files

Docker runs also persist validation and container evidence so a result can be
traced back to raw host/container facts.

## `sol quick-bench` vs `sol bench`

Use `sol quick-bench` when you want speed and iteration:

- fast ad hoc runs
- inner-loop profiling
- one-off debugging while changing replay behavior

Do not use `sol quick-bench` as reproducible benchmark evidence. It is not the
trusted orchestration surface and is not the source of truth for published
comparisons.

Use `sol bench` when you want trustworthy measurements:

- repeated measured runs with cooldown control
- persisted requested/resolved case records
- explicit provenance and verdicts
- one shared trusted contract across native and Docker backends

Use `sol sweep` when you need a trusted comparison across a matrix. It is an
orchestrator over `sol bench`, not a separate measurement engine.

## SOL Sweep Matrix Axes

`sol sweep` reads a matrix file with `benchmark`, `base`, and `axes` keys. Each
entry under `axes` is a case field to vary, and each axis value list is expanded
into one case per combination. Without `--allow-multi-axis`, the matrix may only
contain one axis.

If an axis is not set in `base` and is not varied under `axes`, the sweep uses
the same defaults as a trusted single-case run. The `fixture` field has no
default and must be provided in `base`. The default execution mode is native.

Each expanded sweep case is built by starting with `base` and then applying that
case's axis assignments on top. In other words, `base` provides the default
requested case, and `axes` override specific fields per case.

### Fixture Axis Semantics

The `fixture` axis is supported, and it is the mechanism for sweeping across
more than one `.soltest` fixture.

- If `base.fixture` is set and there is no `fixture` axis, every case uses the
  same base fixture.
- If there is a `fixture` axis, each expanded case replaces `base.fixture` with
  the fixture path from that axis assignment.
- `base.fixture` still has to exist in the matrix file because `base` must be a
  complete valid requested case before axis overrides are applied.
- In practice, when you include a `fixture` axis, `base.fixture` acts as the
  default/fallback value and as the prototype used to build each case before the
  per-case fixture override is applied.

When `fixture` is an axis, the sweep comparison intentionally allows fixture
identity to differ across cases. That means changes in fixture hash and fixture
manifest are treated as expected axis variation rather than invariant
violations. This is necessary because changing fixture usually changes the
embedded checkpoint, archive window, mempool setting, chunk-size metadata, and
embedded kernel together.

Example:

```json
{
  "benchmark": "sol-replay",
  "base": {
    "fixture": "/shared/nockchain/fixtures/a.soltest",
    "warmup_runs": 0,
    "measured_runs": 3,
    "cooldown_secs": 0
  },
  "axes": {
    "fixture": [
      "/shared/nockchain/fixtures/a.soltest",
      "/shared/nockchain/fixtures/b.soltest"
    ]
  }
}
```

In that example, the sweep produces one case for `a.soltest` and one case for
`b.soltest`. The `base.fixture` value is simply the starting value before the
axis override is applied to each case.

The supported axis names are:

| Axis | Type | Default when omitted | What it controls | Example |
| --- | --- | --- | --- | --- |
| `threads` | integer | `1` | Replay worker thread count for each trusted case. | `4` |
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
| `image_tag` | string | empty string in Docker mode | Docker image tag used for trusted Docker cases. Docker-only. A trusted Docker run still requires a non-empty value. | `"nockchain-bench:phase2-local"` |
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

## Practical Examples

Native trusted bench:

```bash
/shared/nockchain/target/release/nockchain-bench sol bench \
  --fixture /shared/nockchain/fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest \
  --output /shared/nockchain/tmp/native-bench-example \
  --warmup-runs 0 \
  --measured-runs 3 \
  --cooldown-secs 0
```

Docker trusted bench:

```bash
/shared/nockchain/target/release/nockchain-bench sol bench \
  --fixture /shared/nockchain/fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest \
  --output /shared/nockchain/tmp/docker-bench-example \
  --image-tag nockchain-bench:phase2-local \
  --memory-limit 8g \
  --work-dir-mode docker-tmpfs \
  --warmup-runs 0 \
  --measured-runs 3 \
  --cooldown-secs 0
```

Trusted sweep with a matrix file:

`matrix.json`

```json
{
  "benchmark": "sol-replay",
  "base": {
    "fixture": "/shared/nockchain/fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest",
    "warmup_runs": 0,
    "measured_runs": 3,
    "cooldown_secs": 0,
    "mode": {
      "docker": {
        "image_tag": "nockchain-bench:phase2-local",
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
/shared/nockchain/target/release/nockchain-bench sol sweep \
  --matrix /shared/nockchain/tmp/memory-matrix.json \
  --output /shared/nockchain/tmp/live-sol-sweep \
  --comparison-markdown
```

The sweep writes per-case outputs under `cases/` plus top-level
`matrix.json`, `matrix_expanded.json`, `schedule.json`, `comparison.json`, and
`verdict.json`.

Multi-axis trusted sweep example:

`matrix-multi-axis.json`

```json
{
  "benchmark": "sol-replay",
  "base": {
    "fixture": "/shared/nockchain/fixtures/first-100-v0-derived-checkpoint-no-mempool.soltest",
    "blocks": 0,
    "enable_checkpointing": true,
    "checkpoint_every_blocks": 0,
    "profile_memory": true,
    "profile_interval_ms": 500,
    "threads": 4,
    "warmup_runs": 0,
    "measured_runs": 3,
    "cooldown_secs": 0,
    "label": "docker-sol-sweep",
    "mode": {
      "docker": {
        "image_tag": "nockchain-bench:phase2-local",
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
    "threads": [2, 4],
    "memory_limit": ["4g", "8g"],
    "work_dir_mode": ["DockerTmpfs", "DockerVolume"],
    "allow_version_skew": [false, true]
  }
}
```

```bash
/shared/nockchain/target/release/nockchain-bench sol sweep \
  --matrix /shared/nockchain/tmp/matrix-multi-axis.json \
  --output /shared/nockchain/tmp/live-sol-sweep-multi-axis \
  --allow-multi-axis \
  --comparison-markdown
```
