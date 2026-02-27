# SOL Trace + Sampling Report

- Date: `2026-02-27`
- Run ID: `20260227_164520`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (v0/v1/v2 (100 blocks), derived fixtures, memory-sweep encoded in fixture ids)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `n/a / n/a / 0.00 bps`
- Best docker: `master / v0-8g / 26.42 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | v0-16g | 10.480 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v0-32g | 10.390 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v0-4g | 10.410 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v0-8g | 10.290 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1-16g | 10.390 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1-32g | 10.390 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1-4g | 10.380 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1-8g | 10.480 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2-16g | 10.420 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2-32g | 10.360 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2-4g | 10.580 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2-8g | 10.460 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0-16g | 10.370 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0-32g | 10.410 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0-4g | 10.640 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0-8g | 10.350 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1-16g | 10.570 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1-32g | 10.560 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1-4g | 10.610 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1-8g | 10.560 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2-16g | 10.400 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2-32g | 10.680 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2-4g | 10.590 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2-8g | 10.620 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0-16g | 24.350 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0-32g | 25.990 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0-4g | 26.320 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0-8g | 26.420 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1-16g | 26.360 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1-32g | 23.950 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1-4g | 24.740 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1-8g | 24.110 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2-16g | 24.510 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2-32g | 25.980 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2-4g | 25.030 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2-8g | 25.800 | 0 | 0 | 0.00 | 0 | n/a |

## Guard Verdicts

| env | branch | fixture | verdict | baseline samples | failed rules | reports |
|---|---|---|---|---:|---:|---|
| n/a | n/a | n/a | n/a | 0 | 0 | n/a |

## Why Did It Change? (Causal Attribution)

| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| docker | btree | v0-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -0.358253 | -0.179126 |
| docker | btree | v0-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.028175 | 0.014088 |
| docker | btree | v0-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -0.67072 | -0.33536 |
| docker | btree | v0-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -3.329964 | -1.664982 |
| docker | btree | v1-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -0.430203 | -0.215102 |
| docker | btree | v1-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.669817 | 0.334909 |
| docker | btree | v1-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.321478 | 0.160739 |
| docker | btree | v1-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.660538 | 0.330269 |
| docker | btree | v2-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -0.535788 | -0.267894 |
| docker | btree | v2-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.819778 | 0.409889 |
| docker | btree | v2-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.688987 | 0.344494 |
| docker | btree | v2-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.098661 | 0.04933 |
| docker | bump PMA | v0-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -1.022999 | -0.5115 |
| docker | bump PMA | v0-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.654782 | 0.327391 |
| docker | bump PMA | v0-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 1.069096 | 0.534548 |
| docker | bump PMA | v0-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -0.928725 | -0.464362 |
| docker | bump PMA | v1-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 4.634266 | 2.317133 |
| docker | bump PMA | v1-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 1.768362 | 0.884181 |
| docker | bump PMA | v1-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 1.06178 | 0.53089 |
| docker | bump PMA | v1-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 3.070711 | 1.535356 |
| docker | bump PMA | v2-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -1.24801 | -0.624005 |
| docker | bump PMA | v2-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 1.356451 | 0.678225 |
| docker | bump PMA | v2-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 2.132791 | 1.066396 |
| docker | bump PMA | v2-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -0.099974 | -0.049987 |
| docker | master | v0-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -4.722371 | -2.361185 |
| docker | master | v0-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 2.08414 | 1.04207 |
| docker | master | v0-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 3.558787 | 1.779394 |
| docker | master | v0-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 10.207316 | 5.103658 |
| docker | master | v1-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 4.727188 | 2.363594 |
| docker | master | v1-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -6.419305 | -3.209653 |
| docker | master | v1-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -3.016744 | -1.508372 |
| docker | master | v1-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -6.478425 | -3.239213 |
| docker | master | v2-16g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 0.108003 | 0.054001 |
| docker | master | v2-32g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 1.418754 | 0.709377 |
| docker | master | v2-4g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | -2.151388 | -1.075694 |
| docker | master | v2-8g | insufficient_baseline | 50.0% | 0.5 | 0.5 | raw_z_tail | 1 | 10.715219 | 5.35761 |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
- `causal-attribution.json`
- `calibration-eval.json`
