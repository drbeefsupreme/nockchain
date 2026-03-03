# SOL Trace + Sampling Report

- Date: `2026-03-03`
- Run ID: `20260303_145036`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (100-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `n/a / n/a / 0.00 bps`
- Best docker: `btree / v0 / 52.98 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | v0 | 52.977 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1 | 52.461 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2 | 52.454 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0 | 52.582 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1 | 52.853 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2 | 52.436 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0 | 21.880 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1 | 21.127 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2 | 22.450 | 0 | 0 | 0.00 | 0 | n/a |

## Guard Verdicts

| env | branch | fixture | verdict | baseline samples | failed rules | reports |
|---|---|---|---|---:|---:|---|
| n/a | n/a | n/a | n/a | 0 | 0 | n/a |

## Why Did It Change? (Causal Attribution)

| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| docker | btree | v0 | improvement | 100.0% | 0.999999 | 0.999999 | raw_z_tail | 4 | 438.641066 | 476.738611 |
| docker | btree | v1 | improvement | 100.0% | 0.999999 | 0.999999 | raw_z_tail | 4 | 442.216693 | 262.314828 |
| docker | btree | v2 | improvement | 100.0% | 0.999999 | 0.999999 | raw_z_tail | 4 | 437.17967 | 354.313989 |
| docker | bump PMA | v0 | regression | 100.0% | 0.999999 | 0.999999 | raw_z_tail | 4 | -0.964828 | -20.363924 |
| docker | bump PMA | v1 | regression | 97.8% | 0.978415 | 0.978415 | raw_z_tail | 4 | -1.215438 | -2.297585 |
| docker | bump PMA | v2 | regression | 97.1% | 0.970622 | 0.970622 | raw_z_tail | 4 | -1.291658 | -2.178381 |
| docker | master | v0 | regression | 100.0% | 0.999811 | 0.999811 | raw_z_tail | 4 | -4.412795 | -3.733271 |
| docker | master | v1 | stable | 42.7% | 0.573407 | 0.573407 | raw_z_tail | 4 | -0.927932 | -0.795036 |
| docker | master | v2 | stable | 66.8% | 0.332227 | 0.332227 | raw_z_tail | 4 | -0.505259 | -0.429207 |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
- `causal-attribution.json`
- `calibration-eval.json`
