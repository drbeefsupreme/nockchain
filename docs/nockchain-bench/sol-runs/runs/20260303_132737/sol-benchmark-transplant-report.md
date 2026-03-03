# SOL Trace + Sampling Report

- Date: `2026-03-03`
- Run ID: `20260303_132737`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (100-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `n/a / n/a / 0.00 bps`
- Best docker: `bump PMA / v1 / 53.67 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | v0 | 9.935 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1 | 9.626 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2 | 9.661 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0 | 53.087 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1 | 53.670 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2 | 53.308 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0 | 23.411 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1 | 21.022 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2 | 21.797 | 0 | 0 | 0.00 | 0 | n/a |

## Guard Verdicts

| env | branch | fixture | verdict | baseline samples | failed rules | reports |
|---|---|---|---|---:|---:|---|
| n/a | n/a | n/a | n/a | 0 | 0 | n/a |

## Why Did It Change? (Causal Attribution)

| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| docker | btree | v0 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 2 | 1.019215 | 1.107723 |
| docker | btree | v1 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 2 | -0.295006 | -0.100289 |
| docker | btree | v2 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 2 | -1.171658 | -0.950662 |
| docker | bump PMA | v0 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 2 | -0.06002 | -0.42861 |
| docker | bump PMA | v1 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 2 | 0.915725 | 2.125745 |
| docker | bump PMA | v2 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 2 | 0.833797 | 4.310367 |
| docker | master | v0 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 2 | 2.273992 | 57.075126 |
| docker | master | v1 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 2 | -3.912263 | -1.109868 |
| docker | master | v2 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 2 | -4.15902 | -7.165442 |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
- `causal-attribution.json`
- `calibration-eval.json`
