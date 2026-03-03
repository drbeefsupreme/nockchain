# SOL Trace + Sampling Report

- Date: `2026-03-03`
- Run ID: `20260303_132456`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (100-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `n/a / n/a / 0.00 bps`
- Best docker: `bump PMA / v0 / 53.07 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | v0 | 9.774 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1 | 9.846 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2 | 9.857 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0 | 53.068 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1 | 53.028 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2 | 52.798 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0 | 22.896 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1 | 22.398 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2 | 22.832 | 0 | 0 | 0.00 | 0 | n/a |

## Guard Verdicts

| env | branch | fixture | verdict | baseline samples | failed rules | reports |
|---|---|---|---|---:|---:|---|
| n/a | n/a | n/a | n/a | 0 | 0 | n/a |

## Why Did It Change? (Causal Attribution)

| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| docker | btree | v0 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v1 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v2 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v0 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v1 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v2 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v0 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v1 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v2 | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
- `causal-attribution.json`
- `calibration-eval.json`
