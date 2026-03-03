# SOL Trace + Sampling Report

- Date: `2026-03-03`
- Run ID: `20260303_132616`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (100-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `n/a / n/a / 0.00 bps`
- Best docker: `bump PMA / v1 / 53.34 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | v0 | 9.896 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1 | 9.463 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2 | 9.695 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0 | 53.169 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1 | 53.337 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2 | 52.936 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0 | 22.884 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1 | 21.358 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2 | 22.654 | 0 | 0 | 0.00 | 0 | n/a |

## Guard Verdicts

| env | branch | fixture | verdict | baseline samples | failed rules | reports |
|---|---|---|---|---:|---:|---|
| n/a | n/a | n/a | n/a | 0 | 0 | n/a |

## Why Did It Change? (Causal Attribution)

| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| docker | btree | v0 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 1.248948 | 0.624474 |
| docker | btree | v1 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -3.890906 | -1.945453 |
| docker | btree | v2 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -1.648866 | -0.824433 |
| docker | bump PMA | v0 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.189083 | 0.094541 |
| docker | bump PMA | v1 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.582806 | 0.291403 |
| docker | bump PMA | v2 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.261288 | 0.130644 |
| docker | master | v0 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -0.053732 | -0.026866 |
| docker | master | v1 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -4.644701 | -2.32235 |
| docker | master | v2 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -0.779933 | -0.389966 |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
- `causal-attribution.json`
- `calibration-eval.json`
