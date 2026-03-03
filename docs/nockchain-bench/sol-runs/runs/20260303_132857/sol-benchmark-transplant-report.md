# SOL Trace + Sampling Report

- Date: `2026-03-03`
- Run ID: `20260303_132857`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (100-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `n/a / n/a / 0.00 bps`
- Best docker: `bump PMA / v1 / 53.72 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | v0 | 9.774 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1 | 9.725 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2 | 9.835 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0 | 53.102 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1 | 53.719 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2 | 53.361 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0 | 22.531 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1 | 21.292 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2 | 22.474 | 0 | 0 | 0.00 | 0 | n/a |

## Guard Verdicts

| env | branch | fixture | verdict | baseline samples | failed rules | reports |
|---|---|---|---|---:|---:|---|
| n/a | n/a | n/a | n/a | 0 | 0 | n/a |

## Why Did It Change? (Causal Attribution)

| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| docker | btree | v0 | regression | 96.4% | 0.963898 | 0.963898 | raw_z_tail | 3 | -1.230944 | -2.095781 |
| docker | btree | v1 | stable | 68.3% | 0.31682 | 0.31682 | raw_z_tail | 3 | 1.025046 | 0.408127 |
| docker | btree | v2 | improvement | 99.6% | 0.995507 | 0.995507 | raw_z_tail | 3 | 1.445843 | 2.84131 |
| docker | bump PMA | v0 | stable | 56.4% | 0.436036 | 0.436036 | raw_z_tail | 3 | 0.029471 | 0.576963 |
| docker | bump PMA | v1 | stable | 40.5% | 0.595317 | 0.595317 | raw_z_tail | 3 | 0.715845 | 0.833287 |
| docker | bump PMA | v2 | improvement | 96.2% | 0.962241 | 0.962241 | raw_z_tail | 3 | 0.802679 | 2.077458 |
| docker | master | v0 | regression | 100.0% | 0.999999 | 0.999999 | raw_z_tail | 3 | -1.594093 | -20.01051 |
| docker | master | v1 | stable | 89.6% | 0.103991 | 0.103991 | raw_z_tail | 3 | -0.30464 | -0.130704 |
| docker | master | v2 | stable | 49.5% | 0.505202 | 0.505202 | raw_z_tail | 3 | -0.795628 | -0.682698 |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
- `causal-attribution.json`
- `calibration-eval.json`
