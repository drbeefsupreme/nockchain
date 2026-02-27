# SOL Trace + Sampling Report

- Date: `2026-02-27`
- Run ID: `20260227_184500`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (100-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `n/a / n/a / 0.00 bps`
- Best docker: `master / v1-8g / 25.78 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | v0-16g | 10.518 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v0-32g | 10.387 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v0-4g | 10.480 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v0-8g | 10.644 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1-16g | 10.435 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1-32g | 10.321 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1-4g | 10.347 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1-8g | 10.411 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2-16g | 10.476 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2-32g | 10.276 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2-4g | 10.508 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2-8g | 10.450 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0-16g | 10.477 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0-32g | 10.342 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0-4g | 10.527 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0-8g | 10.447 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1-16g | 10.102 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1-32g | 10.377 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1-4g | 10.499 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1-8g | 10.245 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2-16g | 10.531 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2-32g | 10.537 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2-4g | 10.369 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2-8g | 10.631 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0-16g | 25.557 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0-32g | 25.459 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0-4g | 25.416 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0-8g | 23.973 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1-16g | 25.170 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1-32g | 25.593 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1-4g | 25.510 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1-8g | 25.780 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2-16g | 24.484 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2-32g | 25.617 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2-4g | 25.580 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2-8g | 23.303 | 0 | 0 | 0.00 | 0 | n/a |

## Guard Verdicts

| env | branch | fixture | verdict | baseline samples | failed rules | reports |
|---|---|---|---|---:|---:|---|
| n/a | n/a | n/a | n/a | 0 | 0 | n/a |

## Why Did It Change? (Causal Attribution)

| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| docker | btree | v0-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v0-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v0-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v0-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v1-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v1-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v1-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v1-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v2-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v2-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v2-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | btree | v2-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v0-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v0-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v0-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v0-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v1-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v1-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v1-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v1-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v2-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v2-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v2-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | bump PMA | v2-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v0-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v0-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v0-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v0-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v1-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v1-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v1-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v1-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v2-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v2-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v2-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |
| docker | master | v2-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 0 | n/a | n/a |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
- `causal-attribution.json`
- `calibration-eval.json`
