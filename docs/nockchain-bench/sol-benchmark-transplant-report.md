# SOL Benchmark Transplant And Performance Report

Date: 2026-02-16

## 1. Transplanted Benchmark Branches (Pushed)

These branches contain the transplanted `crates/nockchain-bench` integration used for this benchmark campaign and are pushed to `github.com/drbeefsupreme/nockchain`.

| Base branch | Pushed branch | Commit SHA | PR shortcut |
|---|---|---|---|
| `master` | `bench-transplant-master` | `26710e534f5058f8be3cd89be7522337c679f72f` | https://github.com/drbeefsupreme/nockchain/pull/new/bench-transplant-master |
| `bitemyapp/ag2-opt-persistence-madvise-checkpoint-chaff-pma-gc-checkpoint-streaming` | `bench-transplant-streaming` | `6ee5c97a5bb63e795fe32bb6258f2a51653ffa07` | https://github.com/drbeefsupreme/nockchain/pull/new/bench-transplant-streaming |
| `bitemyapp/ag2-opt-persistence-madvise-checkpoint-stream-from-pma-slab-but-btree` | `bench-transplant-btree` | `5eb712ec58d747de14c0fa82e87989e5e71cd107` | https://github.com/drbeefsupreme/nockchain/pull/new/bench-transplant-btree |

## 2. Benchmark Matrix Scope

- Fixtures/tests:
  - `v0 proofs`
  - `v1 proofs`
  - `v2 proofs` (available range from provided checkpoint: `12000..=42985`)
- Environments:
  - Native host
  - Docker (`nockchain-local:latest`, `--memory=16g`)
- Checkpoint modes:
  - `master`, `btree`: checkpoint `off` and `on`
  - `streaming`: checkpoint `off` and checkpoint `on` with chunk sizes `32`, `64`, `256`
- Checkpoint cadence for `on` runs: every `5000` blocks

## 3. Coverage Summary

- Native: `24/24` successful benchmark JSON outputs
- Docker: `16/24` successful benchmark JSON outputs
- Docker unresolved failures: `8`, all on `v2`

Docker unresolved failures (no JSON output):

| Branch | Test | Mode | Observed fail time |
|---|---|---|---|
| `master` | `v2` | `checkpoint_off` | `20s` |
| `master` | `v2` | `checkpoint_on` | `6s` |
| `streaming` | `v2` | `checkpoint_off` | `5s` |
| `streaming` | `v2` | `checkpoint_on_chunk32` | `5s` |
| `streaming` | `v2` | `checkpoint_on_chunk64` | `5s` |
| `streaming` | `v2` | `checkpoint_on_chunk256` | `6s` |
| `btree` | `v2` | `checkpoint_off` | `5s` |
| `btree` | `v2` | `checkpoint_on` | `5s` |

A direct repro of docker `master v2 checkpoint_off` returned `rc=137`, consistent with container kill under the `16GB` memory cap.

## 4. Metrics Tracked In JSON

Each successful run tracked these metrics:

- `blocks_poked`
- `failed_pokes`
- `init_time_secs`
- `total_poke_time_secs`
- `blocks_per_second`
- `checkpoint_count`
- `checkpoint_total_time_secs`
- `checkpoint_avg_time_secs`
- `memory_profile`

Observed distributions across successful runs:

- `failed_pokes`: all successful runs had `0`
- `memory_profile`: all successful runs had `null` (profiling disabled)
- `checkpoint_count`:
  - `0` for checkpoint-off runs
  - `1` for v0/v1 checkpoint-on runs
  - `6` for native v2 checkpoint-on runs (cadence 5000 over 30986 blocks)

## 5. Aggregate Throughput (blocks/s)

### Native Aggregate (all successful native runs)

| Branch | Mode | Aggregate blocks | Aggregate time (s) | Aggregate bps |
|---|---|---:|---:|---:|
| `master` | `checkpoint_off` | 42985 | 1641.111 | 26.193 |
| `master` | `checkpoint_on` | 42985 | 1597.833 | 26.902 |
| `streaming` | `checkpoint_off` | 42985 | 4088.920 | 10.513 |
| `streaming` | `checkpoint_on_chunk32` | 42985 | 549.570 | 78.216 |
| `streaming` | `checkpoint_on_chunk64` | 42985 | 551.056 | 78.005 |
| `streaming` | `checkpoint_on_chunk256` | 42985 | 555.324 | 77.405 |
| `btree` | `checkpoint_off` | 42985 | 4128.684 | 10.411 |
| `btree` | `checkpoint_on` | 42985 | 4163.901 | 10.323 |

### Docker Aggregate (successful v0+v1 runs)

| Branch | Mode | Aggregate blocks | Aggregate time (s) | Aggregate bps |
|---|---|---:|---:|---:|
| `master` | `checkpoint_off` | 11999 | 534.119 | 22.465 |
| `master` | `checkpoint_on` | 11999 | 538.772 | 22.271 |
| `streaming` | `checkpoint_off` | 11999 | 1190.744 | 10.077 |
| `streaming` | `checkpoint_on_chunk32` | 11999 | 159.721 | 75.125 |
| `streaming` | `checkpoint_on_chunk64` | 11999 | 158.858 | 75.533 |
| `streaming` | `checkpoint_on_chunk256` | 11999 | 165.010 | 72.717 |
| `btree` | `checkpoint_off` | 11999 | 1216.443 | 9.864 |
| `btree` | `checkpoint_on` | 11999 | 1216.662 | 9.862 |

## 6. Throughput Deltas (Checkpoint On vs Off)

### Native

- `master`: `+3%` aggregate (`26.193 -> 26.902` bps)
- `btree`: `-1%` aggregate (`10.411 -> 10.323` bps)
- `streaming`: `~7.4x` aggregate with checkpoint-on chunk modes vs off

### Docker (successful runs)

- `master`: `-1%` aggregate (`22.465 -> 22.271` bps)
- `btree`: approximately flat (`9.864 -> 9.862` bps)
- `streaming`: `~7.2x to 7.5x` aggregate with checkpoint-on chunk modes vs off

## 7. Chunk-Size Sensitivity (Streaming, Checkpoint On)

Observed spread in throughput between tested chunk sizes:

- Native:
  - `v0`: `0.44%` spread (very stable)
  - `v1`: `0.64%` spread (very stable)
  - `v2`: `1.38%` spread (still small)
- Docker:
  - `v0`: `0.28%` spread (very stable)
  - `v1`: `9.21%` spread (`chunk256` slower than `chunk32/64`)

## 8. Checkpoint Timing Characteristics

Mean `checkpoint_avg_time_secs` on successful checkpoint-enabled runs:

- Native:
  - `master`: `0.933s`
  - `streaming`: `0.202s`
  - `btree`: `1.065s`
- Docker:
  - `master`: `1.242s`
  - `streaming`: `0.261s`
  - `btree`: `1.243s`

Checkpoint time share of total poke time remained low in all successful checkpoint-on runs, roughly `0.17%` to `0.52%`.

## 9. Notable Outliers

- `native master v0 checkpoint_off` had an initialization outlier:
  - `init_time_secs = 138.741`
  - typical native init times otherwise clustered around `~2.8s to ~3.4s`
- Docker v2 modes across all branches failed rapidly under 16GB cap, so no successful docker-v2 performance comparison is available.

## 10. Data Files

Primary machine-generated artifacts used for this report:

- Native run outputs: `/tmp/nockchain-sol-artifacts/results/native/...`
- Docker run outputs: `/home/drbeefsupreme/nockchain-docker-bench/results/...`
- Consolidated tables:
  - `/tmp/nockchain-sol-artifacts/results/combined_metrics.tsv`
  - `/tmp/nockchain-sol-artifacts/results/detailed_metrics.tsv`
- Docker run status timeline:
  - `/home/drbeefsupreme/nockchain-docker-bench/results/docker_manifest.tsv`
