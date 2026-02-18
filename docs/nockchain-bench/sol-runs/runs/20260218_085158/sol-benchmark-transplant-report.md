# SOL Trace + Sampling Report

- Date: `2026-02-18`
- Run ID: `20260218_085158`
- Report pass: `1`
- Matrix: `3 branches x 3 fixtures x native+docker (1000-block fixtures)`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `Tracy capture (.tracy) collected for native runs`
- Stack sampling: `perf record -g` (native runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `btree / v2 / 69.82 bps`
- Best docker: `btree / v2 / 75.84 bps`

## Tracing + Sampling Artifacts

Raw downloads are provided per native run: `trace.tracy`, `trace-capture.log`, `perf.folded`, `perf.script`, and `perf.data`.

| branch | fixture | bps | perf samples | unique stacks | tracy MiB | tracy zones | artifacts |
|---|---|---:|---:|---:|---:|---:|---|
| btree | v0 | 69.050 | 5426 | 614 | 16.15 | 11005 | [trace.tracy](trace/btree/v0/trace.tracy) [trace.log](trace/btree/v0/trace-capture.log) [flamegraph](trace/btree/v0/perf-flamegraph.svg) [summary](trace/btree/v0/perf-summary.json) [report](trace/btree/v0/perf-report.txt) [folded](trace/btree/v0/perf.folded) [perf.script](trace/btree/v0/perf.script) [perf.data](trace/btree/v0/perf.data) |
| btree | v1 | 68.530 | 5276 | 598 | 15.94 | 11005 | [trace.tracy](trace/btree/v1/trace.tracy) [trace.log](trace/btree/v1/trace-capture.log) [flamegraph](trace/btree/v1/perf-flamegraph.svg) [summary](trace/btree/v1/perf-summary.json) [report](trace/btree/v1/perf-report.txt) [folded](trace/btree/v1/perf.folded) [perf.script](trace/btree/v1/perf.script) [perf.data](trace/btree/v1/perf.data) |
| btree | v2 | 69.820 | 5267 | 573 | 16.08 | 11005 | [trace.tracy](trace/btree/v2/trace.tracy) [trace.log](trace/btree/v2/trace-capture.log) [flamegraph](trace/btree/v2/perf-flamegraph.svg) [summary](trace/btree/v2/perf-summary.json) [report](trace/btree/v2/perf-report.txt) [folded](trace/btree/v2/perf.folded) [perf.script](trace/btree/v2/perf.script) [perf.data](trace/btree/v2/perf.data) |
| bump PMA | v0 | 68.500 | 4323 | 510 | 8.06 | 11005 | [trace.tracy](trace/bump-pma/v0/trace.tracy) [trace.log](trace/bump-pma/v0/trace-capture.log) [flamegraph](trace/bump-pma/v0/perf-flamegraph.svg) [summary](trace/bump-pma/v0/perf-summary.json) [report](trace/bump-pma/v0/perf-report.txt) [folded](trace/bump-pma/v0/perf.folded) [perf.script](trace/bump-pma/v0/perf.script) [perf.data](trace/bump-pma/v0/perf.data) |
| bump PMA | v1 | 69.020 | 4476 | 554 | 16.03 | 11005 | [trace.tracy](trace/bump-pma/v1/trace.tracy) [trace.log](trace/bump-pma/v1/trace-capture.log) [flamegraph](trace/bump-pma/v1/perf-flamegraph.svg) [summary](trace/bump-pma/v1/perf-summary.json) [report](trace/bump-pma/v1/perf-report.txt) [folded](trace/bump-pma/v1/perf.folded) [perf.script](trace/bump-pma/v1/perf.script) [perf.data](trace/bump-pma/v1/perf.data) |
| bump PMA | v2 | 68.750 | 4349 | 507 | 8.07 | 11005 | [trace.tracy](trace/bump-pma/v2/trace.tracy) [trace.log](trace/bump-pma/v2/trace-capture.log) [flamegraph](trace/bump-pma/v2/perf-flamegraph.svg) [summary](trace/bump-pma/v2/perf-summary.json) [report](trace/bump-pma/v2/perf-report.txt) [folded](trace/bump-pma/v2/perf.folded) [perf.script](trace/bump-pma/v2/perf.script) [perf.data](trace/bump-pma/v2/perf.data) |
| master | v0 | 23.860 | 9893 | 559 | 15.55 | 11005 | [trace.tracy](trace/master/v0/trace.tracy) [trace.log](trace/master/v0/trace-capture.log) [flamegraph](trace/master/v0/perf-flamegraph.svg) [summary](trace/master/v0/perf-summary.json) [report](trace/master/v0/perf-report.txt) [folded](trace/master/v0/perf.folded) [perf.script](trace/master/v0/perf.script) [perf.data](trace/master/v0/perf.data) |
| master | v1 | 23.800 | 9842 | 595 | 15.65 | 11005 | [trace.tracy](trace/master/v1/trace.tracy) [trace.log](trace/master/v1/trace-capture.log) [flamegraph](trace/master/v1/perf-flamegraph.svg) [summary](trace/master/v1/perf-summary.json) [report](trace/master/v1/perf-report.txt) [folded](trace/master/v1/perf.folded) [perf.script](trace/master/v1/perf.script) [perf.data](trace/master/v1/perf.data) |
| master | v2 | 22.030 | 10491 | 554 | 15.31 | 11005 | [trace.tracy](trace/master/v2/trace.tracy) [trace.log](trace/master/v2/trace-capture.log) [flamegraph](trace/master/v2/perf-flamegraph.svg) [summary](trace/master/v2/perf-summary.json) [report](trace/master/v2/perf-report.txt) [folded](trace/master/v2/perf.folded) [perf.script](trace/master/v2/perf.script) [perf.data](trace/master/v2/perf.data) |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
