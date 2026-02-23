# SOL Trace + Sampling Report

- Date: `2026-02-23`
- Run ID: `20260223_112036`
- Report pass: `2`
- Matrix: `3 branches x 3 fixtures x native+docker (100-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `bump PMA / v0 / 71.22 bps`
- Best docker: `bump PMA / v0 / 75.57 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | v0 | 70.240 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v1 | 69.490 | 0 | 0 | 0.00 | 0 | n/a |
| docker | btree | v2 | 68.130 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v0 | 75.570 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v1 | 74.590 | 0 | 0 | 0.00 | 0 | n/a |
| docker | bump PMA | v2 | 75.510 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v0 | 24.300 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v1 | 23.980 | 0 | 0 | 0.00 | 0 | n/a |
| docker | master | v2 | 24.540 | 0 | 0 | 0.00 | 0 | n/a |
| native | btree | v0 | 68.400 | 1916 | 392 | 6.86 | 1105 | [trace.tracy](trace/native/btree/v0/trace.tracy) [trace.log](trace/native/btree/v0/trace-capture.log) [flamegraph](trace/native/btree/v0/perf-flamegraph.svg) [summary](trace/native/btree/v0/perf-summary.json) [report](trace/native/btree/v0/perf-report.txt) [folded](trace/native/btree/v0/perf.folded) [perf.script](trace/native/btree/v0/perf.script) [perf.data](trace/native/btree/v0/perf.data) |
| native | btree | v1 | 67.910 | 1911 | 391 | 7.23 | 1105 | [trace.tracy](trace/native/btree/v1/trace.tracy) [trace.log](trace/native/btree/v1/trace-capture.log) [flamegraph](trace/native/btree/v1/perf-flamegraph.svg) [summary](trace/native/btree/v1/perf-summary.json) [report](trace/native/btree/v1/perf-report.txt) [folded](trace/native/btree/v1/perf.folded) [perf.script](trace/native/btree/v1/perf.script) [perf.data](trace/native/btree/v1/perf.data) |
| native | btree | v2 | 67.030 | 1844 | 394 | 7.15 | 1105 | [trace.tracy](trace/native/btree/v2/trace.tracy) [trace.log](trace/native/btree/v2/trace-capture.log) [flamegraph](trace/native/btree/v2/perf-flamegraph.svg) [summary](trace/native/btree/v2/perf-summary.json) [report](trace/native/btree/v2/perf-report.txt) [folded](trace/native/btree/v2/perf.folded) [perf.script](trace/native/btree/v2/perf.script) [perf.data](trace/native/btree/v2/perf.data) |
| native | bump PMA | v0 | 71.220 | 1817 | 356 | 7.23 | 1105 | [trace.tracy](trace/native/bump-pma/v0/trace.tracy) [trace.log](trace/native/bump-pma/v0/trace-capture.log) [flamegraph](trace/native/bump-pma/v0/perf-flamegraph.svg) [summary](trace/native/bump-pma/v0/perf-summary.json) [report](trace/native/bump-pma/v0/perf-report.txt) [folded](trace/native/bump-pma/v0/perf.folded) [perf.script](trace/native/bump-pma/v0/perf.script) [perf.data](trace/native/bump-pma/v0/perf.data) |
| native | bump PMA | v1 | 70.630 | 1654 | 372 | 7.27 | 1105 | [trace.tracy](trace/native/bump-pma/v1/trace.tracy) [trace.log](trace/native/bump-pma/v1/trace-capture.log) [flamegraph](trace/native/bump-pma/v1/perf-flamegraph.svg) [summary](trace/native/bump-pma/v1/perf-summary.json) [report](trace/native/bump-pma/v1/perf-report.txt) [folded](trace/native/bump-pma/v1/perf.folded) [perf.script](trace/native/bump-pma/v1/perf.script) [perf.data](trace/native/bump-pma/v1/perf.data) |
| native | bump PMA | v2 | 71.060 | 1657 | 388 | 7.10 | 1105 | [trace.tracy](trace/native/bump-pma/v2/trace.tracy) [trace.log](trace/native/bump-pma/v2/trace-capture.log) [flamegraph](trace/native/bump-pma/v2/perf-flamegraph.svg) [summary](trace/native/bump-pma/v2/perf-summary.json) [report](trace/native/bump-pma/v2/perf-report.txt) [folded](trace/native/bump-pma/v2/perf.folded) [perf.script](trace/native/bump-pma/v2/perf.script) [perf.data](trace/native/bump-pma/v2/perf.data) |
| native | master | v0 | 22.320 | 2300 | 477 | 6.67 | 1105 | [trace.tracy](trace/native/master/v0/trace.tracy) [trace.log](trace/native/master/v0/trace-capture.log) [flamegraph](trace/native/master/v0/perf-flamegraph.svg) [summary](trace/native/master/v0/perf-summary.json) [report](trace/native/master/v0/perf-report.txt) [folded](trace/native/master/v0/perf.folded) [perf.script](trace/native/master/v0/perf.script) [perf.data](trace/native/master/v0/perf.data) |
| native | master | v1 | 22.750 | 2196 | 443 | 6.66 | 1105 | [trace.tracy](trace/native/master/v1/trace.tracy) [trace.log](trace/native/master/v1/trace-capture.log) [flamegraph](trace/native/master/v1/perf-flamegraph.svg) [summary](trace/native/master/v1/perf-summary.json) [report](trace/native/master/v1/perf-report.txt) [folded](trace/native/master/v1/perf.folded) [perf.script](trace/native/master/v1/perf.script) [perf.data](trace/native/master/v1/perf.data) |
| native | master | v2 | 22.150 | 2162 | 448 | 6.58 | 1105 | [trace.tracy](trace/native/master/v2/trace.tracy) [trace.log](trace/native/master/v2/trace-capture.log) [flamegraph](trace/native/master/v2/perf-flamegraph.svg) [summary](trace/native/master/v2/perf-summary.json) [report](trace/native/master/v2/perf-report.txt) [folded](trace/native/master/v2/perf.folded) [perf.script](trace/native/master/v2/perf.script) [perf.data](trace/native/master/v2/perf.data) |

## Guard Verdicts

| env | branch | fixture | verdict | baseline samples | failed rules | reports |
|---|---|---|---|---:|---:|---|
| docker | btree | v0 | pass | 10 | 0 | [json](guard/guard-docker-btree-v0.json) [md](guard/guard-docker-btree-v0.md) |
| docker | btree | v1 | pass | 10 | 0 | [json](guard/guard-docker-btree-v1.json) [md](guard/guard-docker-btree-v1.md) |
| docker | btree | v2 | fail | 10 | 1 | [json](guard/guard-docker-btree-v2.json) [md](guard/guard-docker-btree-v2.md) |
| docker | bump PMA | v0 | pass | 10 | 0 | [json](guard/guard-docker-bump_PMA-v0.json) [md](guard/guard-docker-bump_PMA-v0.md) |
| docker | bump PMA | v1 | pass | 10 | 0 | [json](guard/guard-docker-bump_PMA-v1.json) [md](guard/guard-docker-bump_PMA-v1.md) |
| docker | bump PMA | v2 | pass | 10 | 0 | [json](guard/guard-docker-bump_PMA-v2.json) [md](guard/guard-docker-bump_PMA-v2.md) |
| docker | master | v0 | fail | 10 | 1 | [json](guard/guard-docker-master-v0.json) [md](guard/guard-docker-master-v0.md) |
| docker | master | v1 | fail | 10 | 1 | [json](guard/guard-docker-master-v1.json) [md](guard/guard-docker-master-v1.md) |
| docker | master | v2 | fail | 10 | 1 | [json](guard/guard-docker-master-v2.json) [md](guard/guard-docker-master-v2.md) |
| native | btree | v0 | pass | 5 | 0 | [json](guard/guard-native-btree-v0.json) [md](guard/guard-native-btree-v0.md) |
| native | btree | v1 | pass | 5 | 0 | [json](guard/guard-native-btree-v1.json) [md](guard/guard-native-btree-v1.md) |
| native | btree | v2 | pass | 5 | 0 | [json](guard/guard-native-btree-v2.json) [md](guard/guard-native-btree-v2.md) |
| native | bump PMA | v0 | pass | 5 | 0 | [json](guard/guard-native-bump_PMA-v0.json) [md](guard/guard-native-bump_PMA-v0.md) |
| native | bump PMA | v1 | pass | 5 | 0 | [json](guard/guard-native-bump_PMA-v1.json) [md](guard/guard-native-bump_PMA-v1.md) |
| native | bump PMA | v2 | pass | 5 | 0 | [json](guard/guard-native-bump_PMA-v2.json) [md](guard/guard-native-bump_PMA-v2.md) |
| native | master | v0 | fail | 6 | 1 | [json](guard/guard-native-master-v0.json) [md](guard/guard-native-master-v0.md) |
| native | master | v1 | pass | 6 | 0 | [json](guard/guard-native-master-v1.json) [md](guard/guard-native-master-v1.md) |
| native | master | v2 | pass | 6 | 0 | [json](guard/guard-native-master-v2.json) [md](guard/guard-native-master-v2.md) |

## Why Did It Change? (Causal Attribution)

| env | branch | fixture | classification | confidence (class) | p(change) raw | p(change) calibrated | model | baseline samples | throughput delta (%) | z-score |
|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| docker | btree | v0 | stable | 99.9% | 0.001412 | 0.001412 | raw_z_tail | 10 | 0.007119 | 0.00177 |
| docker | btree | v1 | stable | 97.7% | 0.022851 | 0.022851 | raw_z_tail | 10 | -0.097376 | -0.028643 |
| docker | btree | v2 | stable | 69.7% | 0.303083 | 0.303083 | raw_z_tail | 10 | -1.358594 | -0.389486 |
| docker | bump PMA | v0 | stable | 49.7% | 0.503323 | 0.503323 | raw_z_tail | 8 | 1.747439 | 0.679728 |
| docker | bump PMA | v1 | stable | 87.0% | 0.13007 | 0.13007 | raw_z_tail | 8 | -0.200668 | -0.163748 |
| docker | bump PMA | v2 | stable | 40.0% | 0.599969 | 0.599969 | raw_z_tail | 8 | 0.908726 | 0.841566 |
| docker | master | v0 | stable | 91.1% | 0.088584 | 0.088584 | raw_z_tail | 10 | 0.57947 | 0.111253 |
| docker | master | v1 | stable | 48.4% | 0.515459 | 0.515459 | raw_z_tail | 10 | -3.442722 | -0.699018 |
| docker | master | v2 | stable | 75.3% | 0.246469 | 0.246469 | raw_z_tail | 10 | -1.088271 | -0.313987 |
| native | btree | v0 | stable | 50.6% | 0.493663 | 0.493663 | raw_z_tail | 10 | -1.822879 | -0.664552 |
| native | btree | v1 | stable | 22.2% | 0.778163 | 0.778163 | raw_z_tail | 10 | -2.686824 | -1.221657 |
| native | btree | v2 | stable | 5.6% | 0.944024 | 0.944024 | raw_z_tail | 10 | -4.590421 | -1.911225 |
| native | bump PMA | v0 | regression | 100.0% | 0.999809 | 0.999809 | raw_z_tail | 8 | -6.451824 | -3.730449 |
| native | bump PMA | v1 | stable | 22.6% | 0.774006 | 0.774006 | raw_z_tail | 8 | -5.887532 | -1.210744 |
| native | bump PMA | v2 | stable | 31.0% | 0.690407 | 0.690407 | raw_z_tail | 8 | -5.400822 | -1.016076 |
| native | master | v0 | stable | 7.6% | 0.924098 | 0.924098 | raw_z_tail | 10 | -8.223684 | -1.774976 |
| native | master | v1 | stable | 23.2% | 0.768105 | 0.768105 | raw_z_tail | 10 | -7.801418 | -1.195491 |
| native | master | v2 | stable | 15.7% | 0.843013 | 0.843013 | raw_z_tail | 10 | -9.977647 | -1.415278 |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
- `causal-attribution.json`
- `calibration-eval.json`
