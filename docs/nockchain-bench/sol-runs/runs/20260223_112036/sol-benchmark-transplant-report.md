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

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
