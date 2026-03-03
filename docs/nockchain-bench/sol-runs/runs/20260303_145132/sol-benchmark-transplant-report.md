# SOL Trace + Sampling Report

- Date: `2026-03-03`
- Run ID: `20260303_145132`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (100-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `n/a / n/a / 0.00 bps`
- Best docker: `btree / v2 / 53.37 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | v0 | 52.667 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1 | 52.300 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2 | 53.368 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0 | 52.758 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1 | 52.373 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2 | 51.876 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0 | 22.854 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1 | 21.923 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2 | 21.804 | 0 | 0 | 0.00 | 0 | n/a |

## Guard Verdicts

| env | branch | fixture | verdict | baseline samples | failed rules | reports |
|---|---|---|---|---:|---:|---|
| n/a | n/a | n/a | n/a | 0 | 0 | n/a |

## Why Did It Change? (Causal Attribution)

| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| docker | btree | v0 | improvement | 100.0% | 0.999999 | 0.999999 | raw_z_tail | 5 | 432.195073 | 236.819528 |
| docker | btree | v1 | improvement | 100.0% | 0.999999 | 0.999999 | raw_z_tail | 5 | 437.808942 | 236.624074 |
| docker | btree | v2 | improvement | 100.0% | 0.999999 | 0.999999 | raw_z_tail | 5 | 442.648844 | 209.482791 |
| docker | bump PMA | v0 | regression | 100.0% | 0.999999 | 0.999999 | raw_z_tail | 5 | -0.617927 | -12.097432 |
| docker | bump PMA | v1 | stable | 5.0% | 0.949629 | 0.949629 | raw_z_tail | 5 | -1.80846 | -1.956804 |
| docker | bump PMA | v2 | stable | 5.5% | 0.945516 | 0.945516 | raw_z_tail | 5 | -2.002561 | -1.922971 |
| docker | master | v0 | stable | 95.4% | 0.046314 | 0.046314 | raw_z_tail | 5 | -0.132706 | -0.058078 |
| docker | master | v1 | improvement | 99.0% | 0.989886 | 0.989886 | raw_z_tail | 5 | 2.961111 | 2.571896 |
| docker | master | v2 | regression | 98.8% | 0.987822 | 0.987822 | raw_z_tail | 5 | -2.980913 | -2.506952 |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
- `causal-attribution.json`
- `calibration-eval.json`
