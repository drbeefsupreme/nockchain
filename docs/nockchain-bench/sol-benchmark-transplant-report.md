# SOL Benchmark Transplant Report

Date: 2026-02-17
Run ID: `20260217_115605`

## 1. Scope

Benchmarks represented in this scoreboard:

- `master` (`bench-transplant-master` @ `40b91786bb1595542057d68819c12955a9f13444`) reused from run `20260217_093534`
- `bump PMA` (`bench-transplant-streaming` local patched worktree rerun)
- `btree` (`bench-transplant-btree` local patched worktree rerun)

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
| `master/native` | `24.74` | `1137.97` |
| `master/docker` | `24.06` | `1138.16` |
| `btree/native` | `70.15` | `1487.37` |
| `btree/docker` | `6.12` | `1348.47` |
| `bump PMA/native` | `72.33` | `1359.13` |
| `bump PMA/docker` | `71.41` | `1279.43` |

Fixture winners (throughput):

- Native `v0`: `bump PMA` at `72.10 bps`
- Native `v1`: `bump PMA` at `72.60 bps`
- Native `v2`: `bump PMA` at `72.29 bps`
- Docker `v0`: `bump PMA` at `71.44 bps`
- Docker `v1`: `bump PMA` at `71.55 bps`
- Docker `v2`: `bump PMA` at `71.23 bps`

## 4. Memory Explorer Features

The published dashboard includes an interactive memory panel with:

- shared fixture selector (applies to both stacked charts)
- branch selector for graph 1 and graph 2
- environment selector for graph 1 and graph 2
- metric toggles to add/remove any tracked memory or fault metric
- stacked comparison mode enabled by default
- optional y-axis synchronization across stacked charts
- metric presets (`RSS`, `all`, `faults`, `clear`)
- per-run memory summary cards (peak/p95 RSS, PMA vs nockstack peaks, fault deltas, phase peaks)
- sortable memory leaderboard across all runs

## 5. Memory/Profile Metrics

Tracked for every run:

- `samples`, `gc_events`, `fault_bursts`, `peak_rss_mib`, `p95_rss_mib`, `gc_per_1k_blocks`

Observed rollups:

- `failed_pokes`: `0` for `18/18`
- `checkpoints`: `0` for `18/18` (expected: checkpointing disabled)
- `gc_events`: `0` for `18/18`
- `fault_bursts`: `3` runs with `0`, `15` runs with `>0` (max `251`)
- `peak_rss_mib` range: `1137.60` to `1487.70`

Note on `max_rss_kb` from `/usr/bin/time -v`:

- Native values reflect benchmark process RSS.
- Docker values reflect the host `docker` client process RSS, not in-container RSS.

## 6. Artifacts

Primary source data:

- `bench-artifacts/benchmark-matrix/20260217_115605/combined_summary.tsv`
- `bench-artifacts/benchmark-matrix/20260217_115605/runs/**/command.log`
- `bench-artifacts/benchmark-matrix/20260217_115605/runs/**/profile.json`
- `docs/nockchain-bench/sol-benchmark-transplant-memory-profiles.json`

Published dashboard:

- `docs/nockchain-bench/sol-benchmark-transplant-report.html`
- `docs/nockchain-bench/sol-benchmark-transplant-report.md`
