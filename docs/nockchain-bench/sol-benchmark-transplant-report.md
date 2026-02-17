# SOL Benchmark Transplant Report

Date: 2026-02-17
Run ID: `20260217_093534`

## 1. Scope

Benchmarks run for these 3 branches:

- `current` (`bench-transplant-master` @ `40b91786bb1595542057d68819c12955a9f13444`)
- `streaming` (`bench-transplant-streaming` @ `6ee5c97a5bb63e795fe32bb6258f2a51653ffa07`)
- `btree` (`bench-transplant-btree` @ `5eb712ec58d747de14c0fa82e87989e5e71cd107`)

Matrix:

- Fixtures: `v0`, `v1`, `v2` first 1000 blocks
- Environments: `native`, `docker --memory=16g`
- Checkpointing: `off` for all runs
- Memory profiling: `on` for all runs
- Execution policy: one test at a time

## 2. Completion

- Native success: `9/9`
- Docker success: `9/9`
- Total successful runs: `18/18`
- All runs exited with `exit_status=0`

## 3. Throughput Summary

Average throughput by branch/runtime (blocks/s):

| branch/runtime | avg throughput | avg peak RSS (MiB) |
|---|---:|---:|
| `current/native` | `24.74` | `1137.97` |
| `current/docker` | `24.06` | `1138.16` |
| `btree/native` | `10.49` | `1272.26` |
| `btree/docker` | `10.44` | `1272.36` |
| `streaming/native` | `9.72` | `1272.39` |
| `streaming/docker` | `9.97` | `1272.35` |

Fixture winners (throughput):

- Native `v0`: `current` at `24.32 bps`
- Native `v1`: `current` at `24.98 bps`
- Native `v2`: `current` at `24.91 bps`
- Docker `v0`: `current` at `23.01 bps`
- Docker `v1`: `current` at `24.37 bps`
- Docker `v2`: `current` at `24.81 bps`

## 4. Memory/Profile Metrics

Tracked for every run:

- `samples`, `gc_events`, `fault_bursts`, `peak_rss_mib`, `p95_rss_mib`, `gc_per_1k_blocks`

Observed rollups:

- `failed_pokes`: `0` for all `18/18`
- `checkpoints`: `0` for all `18/18` (expected: checkpointing disabled)
- `gc_events`: `0` for all `18/18`
- `fault_bursts`: `1` in all native runs, `0` in all docker runs
- `peak_rss_mib` range: `1137.60` to `1272.71`

Note on `max_rss_kb` from `/usr/bin/time -v`:

- Native values reflect benchmark process RSS.
- Docker values reflect the host `docker` client process RSS, not in-container RSS.

## 5. Artifacts

Primary source data:

- `bench-artifacts/benchmark-matrix/20260217_093534/combined_summary.tsv`
- `bench-artifacts/benchmark-matrix/20260217_093534/runs/**/command.log`
- `bench-artifacts/benchmark-matrix/20260217_093534/runs/**/profile.json`

Published dashboard:

- `docs/nockchain-bench/sol-benchmark-transplant-report.html`
- `docs/nockchain-bench/sol-benchmark-transplant-report.md`
