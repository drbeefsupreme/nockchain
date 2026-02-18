# SOL Trace + Sampling Report

- Date: `2026-02-18`
- Run ID: `20260218_105729`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (100-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native runs`
- Stack sampling: `perf record -g` (native runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `btree / v1 / 69.20 bps`
- Best docker: `bump PMA / v1 / 75.45 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per native run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---:|---:|---:|---:|---:|---|
| btree | v0 | 67.950 | 1795 | 344 | 7.10 | 1105 | [trace.tracy](trace/btree/v0/trace.tracy) [trace.log](trace/btree/v0/trace-capture.log) [flamegraph](trace/btree/v0/perf-flamegraph.svg) [summary](trace/btree/v0/perf-summary.json) [report](trace/btree/v0/perf-report.txt) [folded](trace/btree/v0/perf.folded) [perf.script](trace/btree/v0/perf.script) [perf.data](trace/btree/v0/perf.data) |
| btree | v1 | 69.200 | 1741 | 372 | 7.14 | 1105 | [trace.tracy](trace/btree/v1/trace.tracy) [trace.log](trace/btree/v1/trace-capture.log) [flamegraph](trace/btree/v1/perf-flamegraph.svg) [summary](trace/btree/v1/perf-summary.json) [report](trace/btree/v1/perf-report.txt) [folded](trace/btree/v1/perf.folded) [perf.script](trace/btree/v1/perf.script) [perf.data](trace/btree/v1/perf.data) |
| btree | v2 | 68.780 | 1788 | 359 | 7.02 | 1105 | [trace.tracy](trace/btree/v2/trace.tracy) [trace.log](trace/btree/v2/trace-capture.log) [flamegraph](trace/btree/v2/perf-flamegraph.svg) [summary](trace/btree/v2/perf-summary.json) [report](trace/btree/v2/perf-report.txt) [folded](trace/btree/v2/perf.folded) [perf.script](trace/btree/v2/perf.script) [perf.data](trace/btree/v2/perf.data) |
| bump PMA | v0 | 66.580 | 1752 | 383 | 6.94 | 1105 | [trace.tracy](trace/bump-pma/v0/trace.tracy) [trace.log](trace/bump-pma/v0/trace-capture.log) [flamegraph](trace/bump-pma/v0/perf-flamegraph.svg) [summary](trace/bump-pma/v0/perf-summary.json) [report](trace/bump-pma/v0/perf-report.txt) [folded](trace/bump-pma/v0/perf.folded) [perf.script](trace/bump-pma/v0/perf.script) [perf.data](trace/bump-pma/v0/perf.data) |
| bump PMA | v1 | 68.530 | 1613 | 354 | 7.05 | 1105 | [trace.tracy](trace/bump-pma/v1/trace.tracy) [trace.log](trace/bump-pma/v1/trace-capture.log) [flamegraph](trace/bump-pma/v1/perf-flamegraph.svg) [summary](trace/bump-pma/v1/perf-summary.json) [report](trace/bump-pma/v1/perf-report.txt) [folded](trace/bump-pma/v1/perf.folded) [perf.script](trace/bump-pma/v1/perf.script) [perf.data](trace/bump-pma/v1/perf.data) |
| bump PMA | v2 | 68.550 | 1709 | 374 | 7.14 | 1105 | [trace.tracy](trace/bump-pma/v2/trace.tracy) [trace.log](trace/bump-pma/v2/trace-capture.log) [flamegraph](trace/bump-pma/v2/perf-flamegraph.svg) [summary](trace/bump-pma/v2/perf-summary.json) [report](trace/bump-pma/v2/perf-report.txt) [folded](trace/bump-pma/v2/perf.folded) [perf.script](trace/bump-pma/v2/perf.script) [perf.data](trace/bump-pma/v2/perf.data) |
| master | v0 | 23.410 | 2104 | 444 | 6.70 | 1105 | [trace.tracy](trace/master/v0/trace.tracy) [trace.log](trace/master/v0/trace-capture.log) [flamegraph](trace/master/v0/perf-flamegraph.svg) [summary](trace/master/v0/perf-summary.json) [report](trace/master/v0/perf-report.txt) [folded](trace/master/v0/perf.folded) [perf.script](trace/master/v0/perf.script) [perf.data](trace/master/v0/perf.data) |
| master | v1 | 23.030 | 2178 | 487 | 14.13 | 1105 | [trace.tracy](trace/master/v1/trace.tracy) [trace.log](trace/master/v1/trace-capture.log) [flamegraph](trace/master/v1/perf-flamegraph.svg) [summary](trace/master/v1/perf-summary.json) [report](trace/master/v1/perf-report.txt) [folded](trace/master/v1/perf.folded) [perf.script](trace/master/v1/perf.script) [perf.data](trace/master/v1/perf.data) |
| master | v2 | 23.220 | 2059 | 438 | 6.37 | 1105 | [trace.tracy](trace/master/v2/trace.tracy) [trace.log](trace/master/v2/trace-capture.log) [flamegraph](trace/master/v2/perf-flamegraph.svg) [summary](trace/master/v2/perf-summary.json) [report](trace/master/v2/perf-report.txt) [folded](trace/master/v2/perf.folded) [perf.script](trace/master/v2/perf.script) [perf.data](trace/master/v2/perf.data) |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
