# SOL Benchmark Transplant Report

Date: 2026-02-17
Run ID: `20260217_161000`

## Scope

- Branches: `master`, `bump PMA`, `btree`
- Fixtures: `v0`, `v1`, `v2` 100-block fixtures
- Environments: `native`, `docker --memory=16g --memory-swap=16g`
- Checkpointing: off
- Memory profiling: on
- Execution: one test at a time
- Repeats: 2 full matrix passes (36 runs total); dashboard rollups use pass 2 profiles for 1:1 memory explorer mapping.

## Completion

- Pass-2 rows in scoreboard: `18/18` exit success
- Full matrix rows: `36/36` exit success

## Throughput (Pass 2)

| branch/runtime | avg throughput (bps) | avg peak RSS (MiB) |
|---|---:|---:|
| `master/native` | `23.31` | `1063.72` |
| `master/docker` | `23.42` | `1063.61` |
| `bump PMA/native` | `75.30` | `1167.85` |
| `bump PMA/docker` | `74.71` | `1169.84` |
| `btree/native` | `70.56` | `1296.17` |
| `btree/docker` | `69.40` | `1298.31` |

## Artifacts

- Full matrix TSV: `/shared/nockchain-ext4-bench/artifacts/runs/20260217_161000-sol-100x2/combined_summary.tsv`
- Dashboard memory dataset: `docs/nockchain-bench/sol-benchmark-transplant-memory-profiles.json`
- Published report HTML: `docs/nockchain-bench/sol-benchmark-transplant-report.html`
- Published report MD: `docs/nockchain-bench/sol-benchmark-transplant-report.md`
