# SOL Benchmark Transplant Report

Date: 2026-02-17
Run ID: `20260217_135129`

## Scope

- Branches: `master`, `bump PMA`, `btree`
- Fixtures: `v0`, `v1`, `v2` 100-block fixtures
- Environments: `native`, `docker --memory=16g`
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
| `master/native` | `25.47` | `1833.77` |
| `master/docker` | `25.60` | `1063.71` |
| `bump PMA/native` | `76.16` | `1167.75` |
| `bump PMA/docker` | `74.53` | `1170.20` |
| `btree/native` | `71.22` | `1296.21` |
| `btree/docker` | `70.31` | `1298.38` |

## Artifacts

- Full matrix TSV: `/shared/nockchain-ext4-bench/artifacts/runs/20260217_135129-sol-100x2/combined_summary.tsv`
- Dashboard memory dataset: `docs/nockchain-bench/sol-benchmark-transplant-memory-profiles.json`
- Published report HTML: `docs/nockchain-bench/sol-benchmark-transplant-report.html`
- Published report MD: `docs/nockchain-bench/sol-benchmark-transplant-report.md`
