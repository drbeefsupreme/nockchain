# SOL Trace + Sampling Report

- Date: `2026-03-03`
- Run ID: `20260303_145327`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (100-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `n/a / n/a / 0.00 bps`
- Best docker: `btree / v0 / 52.65 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | v0 | 52.651 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1 | 52.237 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2 | 52.113 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0 | 52.412 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1 | 52.525 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2 | 52.631 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0 | 23.198 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1 | 21.596 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2 | 22.281 | 0 | 0 | 0.00 | 0 | n/a |

## Guard Verdicts

| env | branch | fixture | verdict | baseline samples | failed rules | reports |
|---|---|---|---|---:|---:|---|
| n/a | n/a | n/a | n/a | 0 | 0 | n/a |

## Why Did It Change? (Causal Attribution)

| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| docker | btree | v0 | improvement | 100.0% | 0.999999 | 0.999999 | isotonic_online | 7 | 429.928541 | 178.641291 |
| docker | btree | v1 | improvement | 100.0% | 0.999999 | 0.999999 | isotonic_online | 7 | 430.541208 | 74.634571 |
| docker | btree | v2 | improvement | 100.0% | 0.999999 | 0.999999 | isotonic_online | 7 | 428.678374 | 145.557308 |
| docker | bump PMA | v0 | regression | 100.0% | 0.99999 | 0.999999 | isotonic_online | 7 | -1.235662 | -4.407824 |
| docker | bump PMA | v1 | stable | 66.7% | 0.440136 | 0.333333 | isotonic_online | 7 | -0.949423 | -0.583044 |
| docker | bump PMA | v2 | stable | 66.7% | 0.273835 | 0.333333 | isotonic_online | 7 | -0.398297 | -0.350232 |
| docker | master | v0 | stable | 23.1% | 0.528991 | 0.769231 | isotonic_online | 7 | 1.507267 | 0.720838 |
| docker | master | v1 | stable | 66.7% | 0.368316 | 0.333333 | isotonic_online | 7 | 1.117268 | 0.479358 |
| docker | master | v2 | stable | 66.7% | 0.233924 | 0.333333 | isotonic_online | 7 | -0.750943 | -0.297512 |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
- `causal-attribution.json`
- `calibration-eval.json`
