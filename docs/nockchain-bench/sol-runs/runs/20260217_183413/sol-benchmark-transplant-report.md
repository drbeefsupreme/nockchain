# SOL Trace + Sampling Report

- Date: `2026-02-17`
- Run ID: `20260217_183413`
- Report pass: `2`
- Matrix: `3 branches x 3 fixtures x native+docker`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native runs`
- Stack sampling: `perf record -g` (native runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `bump PMA / v0 / 71.22 bps`
- Best docker: `bump PMA / v0 / 75.57 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per native run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---:|---:|---:|---:|---:|---|
| btree | v0 | 68.400 | 1916 | 392 | 6.86 | 1105 | [trace.tracy](trace/btree/v0/trace.tracy) [trace.log](trace/btree/v0/trace-capture.log) [flamegraph](trace/btree/v0/perf-flamegraph.svg) [summary](trace/btree/v0/perf-summary.json) [report](trace/btree/v0/perf-report.txt) [folded](trace/btree/v0/perf.folded) [perf.script](trace/btree/v0/perf.script) [perf.data](trace/btree/v0/perf.data) |
| btree | v1 | 67.910 | 1911 | 391 | 7.23 | 1105 | [trace.tracy](trace/btree/v1/trace.tracy) [trace.log](trace/btree/v1/trace-capture.log) [flamegraph](trace/btree/v1/perf-flamegraph.svg) [summary](trace/btree/v1/perf-summary.json) [report](trace/btree/v1/perf-report.txt) [folded](trace/btree/v1/perf.folded) [perf.script](trace/btree/v1/perf.script) [perf.data](trace/btree/v1/perf.data) |
| btree | v2 | 67.030 | 1844 | 394 | 7.15 | 1105 | [trace.tracy](trace/btree/v2/trace.tracy) [trace.log](trace/btree/v2/trace-capture.log) [flamegraph](trace/btree/v2/perf-flamegraph.svg) [summary](trace/btree/v2/perf-summary.json) [report](trace/btree/v2/perf-report.txt) [folded](trace/btree/v2/perf.folded) [perf.script](trace/btree/v2/perf.script) [perf.data](trace/btree/v2/perf.data) |
| bump PMA | v0 | 71.220 | 1817 | 356 | 7.23 | 1105 | [trace.tracy](trace/bump-pma/v0/trace.tracy) [trace.log](trace/bump-pma/v0/trace-capture.log) [flamegraph](trace/bump-pma/v0/perf-flamegraph.svg) [summary](trace/bump-pma/v0/perf-summary.json) [report](trace/bump-pma/v0/perf-report.txt) [folded](trace/bump-pma/v0/perf.folded) [perf.script](trace/bump-pma/v0/perf.script) [perf.data](trace/bump-pma/v0/perf.data) |
| bump PMA | v1 | 70.630 | 1654 | 372 | 7.27 | 1105 | [trace.tracy](trace/bump-pma/v1/trace.tracy) [trace.log](trace/bump-pma/v1/trace-capture.log) [flamegraph](trace/bump-pma/v1/perf-flamegraph.svg) [summary](trace/bump-pma/v1/perf-summary.json) [report](trace/bump-pma/v1/perf-report.txt) [folded](trace/bump-pma/v1/perf.folded) [perf.script](trace/bump-pma/v1/perf.script) [perf.data](trace/bump-pma/v1/perf.data) |
| bump PMA | v2 | 71.060 | 1657 | 388 | 7.10 | 1105 | [trace.tracy](trace/bump-pma/v2/trace.tracy) [trace.log](trace/bump-pma/v2/trace-capture.log) [flamegraph](trace/bump-pma/v2/perf-flamegraph.svg) [summary](trace/bump-pma/v2/perf-summary.json) [report](trace/bump-pma/v2/perf-report.txt) [folded](trace/bump-pma/v2/perf.folded) [perf.script](trace/bump-pma/v2/perf.script) [perf.data](trace/bump-pma/v2/perf.data) |
| master | v0 | 22.320 | 2300 | 477 | 6.67 | 1105 | [trace.tracy](trace/master/v0/trace.tracy) [trace.log](trace/master/v0/trace-capture.log) [flamegraph](trace/master/v0/perf-flamegraph.svg) [summary](trace/master/v0/perf-summary.json) [report](trace/master/v0/perf-report.txt) [folded](trace/master/v0/perf.folded) [perf.script](trace/master/v0/perf.script) [perf.data](trace/master/v0/perf.data) |
| master | v1 | 22.750 | 2196 | 443 | 6.66 | 1105 | [trace.tracy](trace/master/v1/trace.tracy) [trace.log](trace/master/v1/trace-capture.log) [flamegraph](trace/master/v1/perf-flamegraph.svg) [summary](trace/master/v1/perf-summary.json) [report](trace/master/v1/perf-report.txt) [folded](trace/master/v1/perf.folded) [perf.script](trace/master/v1/perf.script) [perf.data](trace/master/v1/perf.data) |
| master | v2 | 22.150 | 2162 | 448 | 6.58 | 1105 | [trace.tracy](trace/master/v2/trace.tracy) [trace.log](trace/master/v2/trace-capture.log) [flamegraph](trace/master/v2/perf-flamegraph.svg) [summary](trace/master/v2/perf-summary.json) [report](trace/master/v2/perf-report.txt) [folded](trace/master/v2/perf.folded) [perf.script](trace/master/v2/perf.script) [perf.data](trace/master/v2/perf.data) |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
