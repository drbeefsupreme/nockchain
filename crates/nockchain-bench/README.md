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
