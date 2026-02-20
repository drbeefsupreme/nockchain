# SOL Trace + Sampling Report

- Date: `2026-02-19`
- Run ID: `20260219_180712`
- Report pass: `2`
- Matrix: `3 branches x 3 fixtures x native+docker (derived fixtures (docker subset))`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native and docker runs`
- Stack sampling: `perf record -g` (native and docker runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `n/a / n/a / 0.00 bps`
- Best docker: `master / v0d / 24.80 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| env | branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---|---:|---:|---:|---:|---:|---|
| docker | btree | h51000d | 9.790 | 3029 | 650 | 0.00 | 0 | [trace.log](trace/docker/btree/h51000d/trace-capture.log) [flamegraph](trace/docker/btree/h51000d/perf-flamegraph.svg) [summary](trace/docker/btree/h51000d/perf-summary.json) [report](trace/docker/btree/h51000d/perf-report.txt) [folded](trace/docker/btree/h51000d/perf.folded) [perf.script](trace/docker/btree/h51000d/perf.script) [perf.data](trace/docker/btree/h51000d/perf.data) |
| docker | btree | v0d | 9.880 | 2958 | 601 | 0.00 | 0 | [trace.log](trace/docker/btree/v0d/trace-capture.log) [flamegraph](trace/docker/btree/v0d/perf-flamegraph.svg) [summary](trace/docker/btree/v0d/perf-summary.json) [report](trace/docker/btree/v0d/perf-report.txt) [folded](trace/docker/btree/v0d/perf.folded) [perf.script](trace/docker/btree/v0d/perf.script) [perf.data](trace/docker/btree/v0d/perf.data) |
| docker | btree | v1d | 9.930 | 2955 | 635 | 0.00 | 0 | [trace.log](trace/docker/btree/v1d/trace-capture.log) [flamegraph](trace/docker/btree/v1d/perf-flamegraph.svg) [summary](trace/docker/btree/v1d/perf-summary.json) [report](trace/docker/btree/v1d/perf-report.txt) [folded](trace/docker/btree/v1d/perf.folded) [perf.script](trace/docker/btree/v1d/perf.script) [perf.data](trace/docker/btree/v1d/perf.data) |
| docker | btree | v2d | 9.850 | 2949 | 581 | 0.00 | 0 | [trace.log](trace/docker/btree/v2d/trace-capture.log) [flamegraph](trace/docker/btree/v2d/perf-flamegraph.svg) [summary](trace/docker/btree/v2d/perf-summary.json) [report](trace/docker/btree/v2d/perf-report.txt) [folded](trace/docker/btree/v2d/perf.folded) [perf.script](trace/docker/btree/v2d/perf.script) [perf.data](trace/docker/btree/v2d/perf.data) |
| docker | bump PMA | h51000d | 10.010 | 2898 | 566 | 0.00 | 0 | [trace.log](trace/docker/bump-pma/h51000d/trace-capture.log) [flamegraph](trace/docker/bump-pma/h51000d/perf-flamegraph.svg) [summary](trace/docker/bump-pma/h51000d/perf-summary.json) [report](trace/docker/bump-pma/h51000d/perf-report.txt) [folded](trace/docker/bump-pma/h51000d/perf.folded) [perf.script](trace/docker/bump-pma/h51000d/perf.script) [perf.data](trace/docker/bump-pma/h51000d/perf.data) |
| docker | bump PMA | v0d | 9.560 | 3041 | 595 | 0.00 | 0 | [trace.log](trace/docker/bump-pma/v0d/trace-capture.log) [flamegraph](trace/docker/bump-pma/v0d/perf-flamegraph.svg) [summary](trace/docker/bump-pma/v0d/perf-summary.json) [report](trace/docker/bump-pma/v0d/perf-report.txt) [folded](trace/docker/bump-pma/v0d/perf.folded) [perf.script](trace/docker/bump-pma/v0d/perf.script) [perf.data](trace/docker/bump-pma/v0d/perf.data) |
| docker | bump PMA | v1d | 10.020 | 2923 | 635 | 0.00 | 0 | [trace.log](trace/docker/bump-pma/v1d/trace-capture.log) [flamegraph](trace/docker/bump-pma/v1d/perf-flamegraph.svg) [summary](trace/docker/bump-pma/v1d/perf-summary.json) [report](trace/docker/bump-pma/v1d/perf-report.txt) [folded](trace/docker/bump-pma/v1d/perf.folded) [perf.script](trace/docker/bump-pma/v1d/perf.script) [perf.data](trace/docker/bump-pma/v1d/perf.data) |
| docker | bump PMA | v2d | 9.610 | 2999 | 635 | 0.00 | 0 | [trace.log](trace/docker/bump-pma/v2d/trace-capture.log) [flamegraph](trace/docker/bump-pma/v2d/perf-flamegraph.svg) [summary](trace/docker/bump-pma/v2d/perf-summary.json) [report](trace/docker/bump-pma/v2d/perf-report.txt) [folded](trace/docker/bump-pma/v2d/perf.folded) [perf.script](trace/docker/bump-pma/v2d/perf.script) [perf.data](trace/docker/bump-pma/v2d/perf.data) |
| docker | master | h51000d | 23.780 | 1621 | 632 | 0.00 | 0 | [trace.log](trace/docker/master/h51000d/trace-capture.log) [flamegraph](trace/docker/master/h51000d/perf-flamegraph.svg) [summary](trace/docker/master/h51000d/perf-summary.json) [report](trace/docker/master/h51000d/perf-report.txt) [folded](trace/docker/master/h51000d/perf.folded) [perf.script](trace/docker/master/h51000d/perf.script) [perf.data](trace/docker/master/h51000d/perf.data) |
| docker | master | v0d | 24.800 | 1571 | 615 | 0.00 | 0 | [trace.log](trace/docker/master/v0d/trace-capture.log) [flamegraph](trace/docker/master/v0d/perf-flamegraph.svg) [summary](trace/docker/master/v0d/perf-summary.json) [report](trace/docker/master/v0d/perf-report.txt) [folded](trace/docker/master/v0d/perf.folded) [perf.script](trace/docker/master/v0d/perf.script) [perf.data](trace/docker/master/v0d/perf.data) |
| docker | master | v1d | 24.400 | 1574 | 634 | 0.00 | 0 | [trace.log](trace/docker/master/v1d/trace-capture.log) [flamegraph](trace/docker/master/v1d/perf-flamegraph.svg) [summary](trace/docker/master/v1d/perf-summary.json) [report](trace/docker/master/v1d/perf-report.txt) [folded](trace/docker/master/v1d/perf.folded) [perf.script](trace/docker/master/v1d/perf.script) [perf.data](trace/docker/master/v1d/perf.data) |
| docker | master | v2d | 24.550 | 1581 | 624 | 0.00 | 0 | [trace.log](trace/docker/master/v2d/trace-capture.log) [flamegraph](trace/docker/master/v2d/perf-flamegraph.svg) [summary](trace/docker/master/v2d/perf-summary.json) [report](trace/docker/master/v2d/perf-report.txt) [folded](trace/docker/master/v2d/perf.folded) [perf.script](trace/docker/master/v2d/perf.script) [perf.data](trace/docker/master/v2d/perf.data) |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
