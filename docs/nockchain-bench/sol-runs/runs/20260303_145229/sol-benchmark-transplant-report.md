# SOL Trace + Sampling Report

- Date: `2026-03-03`
- Run ID: `20260303_145229`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (100-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `n/a / n/a / 0.00 bps`
- Best docker: `btree / v0 / 53.44 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | v0 | 53.444 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1 | 51.726 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2 | 52.466 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0 | 51.994 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1 | 52.446 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2 | 52.841 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0 | 21.286 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1 | 22.582 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2 | 21.559 | 0 | 0 | 0.00 | 0 | n/a |

## Guard Verdicts

| env | branch | fixture | verdict | baseline samples | failed rules | reports |
|---|---|---|---|---:|---:|---|
| n/a | n/a | n/a | n/a | 0 | 0 | n/a |

## Why Did It Change? (Causal Attribution)

| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| docker | btree | v0 | improvement | 100.0% | 0.999999 | 0.999999 | isotonic_online | 6 | 438.980182 | 207.418061 |
| docker | btree | v1 | improvement | 100.0% | 0.999999 | 0.999999 | isotonic_online | 6 | 428.607732 | 117.436209 |
| docker | btree | v2 | improvement | 100.0% | 0.999999 | 0.999999 | isotonic_online | 6 | 432.866189 | 171.124337 |
| docker | bump PMA | v0 | regression | 100.0% | 0.999999 | 0.999999 | isotonic_online | 6 | -2.041504 | -12.60241 |
| docker | bump PMA | v1 | stable | 0.0% | 0.776583 | 0.999999 | isotonic_online | 6 | -1.385699 | -1.217493 |
| docker | bump PMA | v2 | stable | 72.2% | 0.032009 | 0.277778 | isotonic_online | 6 | -0.049069 | -0.040128 |
| docker | master | v0 | regression | 100.0% | 0.999999 | 0.999999 | isotonic_online | 6 | -6.920876 | -5.849676 |
| docker | master | v1 | improvement | 100.0% | 0.999288 | 0.999999 | isotonic_online | 6 | 5.896061 | 3.384757 |
| docker | master | v2 | regression | 100.0% | 0.969708 | 0.999999 | isotonic_online | 6 | -4.021027 | -2.166253 |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
- `causal-attribution.json`
- `calibration-eval.json`
