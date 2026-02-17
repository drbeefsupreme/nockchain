# SOL Benchmark Transplant Report

Date: 2026-02-17
Run ID: `20260217_164120`

## Scope

- Branches: `master`, `bump PMA`, `btree`
- Fixtures: `v0`, `v1`, `v2` 100-block fixtures
- Environments: `native`, `docker --memory=16g --memory-swap=16g`
- Checkpointing: off
- Memory profiling: off
- Execution: one test at a time
- Repeats: 2 full matrix passes (36 runs total); dashboard rollups use pass 2 (18 rows).

## Completion

- Pass-2 rows in scoreboard: `18/18` exit success
- Full matrix rows: `36/36` exit success

## Throughput (Pass 2)

| branch/runtime | avg throughput (bps) |
|---|---:|
| `master/native` | `24.53` |
| `master/docker` | `23.88` |
| `bump PMA/native` | `76.19` |
| `bump PMA/docker` | `73.62` |
| `btree/native` | `69.67` |
| `btree/docker` | `68.09` |

## Fixture Winners (Pass 2)

- Native `v0`: `bump PMA` at `76.78 bps`
- Native `v1`: `bump PMA` at `77.37 bps`
- Native `v2`: `bump PMA` at `74.41 bps`
- Docker `v0`: `bump PMA` at `73.66 bps`
- Docker `v1`: `bump PMA` at `73.64 bps`
- Docker `v2`: `bump PMA` at `73.55 bps`

## Artifacts

- Full matrix TSV: `/shared/nockchain-ext4-bench/artifacts/runs/20260217_164120-sol-100x2-noprofile/combined_summary.tsv`
- Dashboard memory dataset: `docs/nockchain-bench/sol-benchmark-transplant-memory-profiles.json` (empty in this run because profiling was off)
- Published report HTML: `docs/nockchain-bench/sol-benchmark-transplant-report.html`
- Published report MD: `docs/nockchain-bench/sol-benchmark-transplant-report.md`
