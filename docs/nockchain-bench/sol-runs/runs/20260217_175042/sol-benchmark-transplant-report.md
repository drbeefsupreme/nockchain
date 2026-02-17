# SOL Trace + Sampling Report

- Date: `2026-02-17`
- Run ID: `20260217_175042`
- Report pass: `2`
- Matrix: `3 branches x 3 fixtures x native+docker`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `tracy instrumentation enabled by default`
- Stack sampling: `perf record -g` (native runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `bump PMA / v2 / 77.92 bps`
- Best docker: `bump PMA / v0 / 75.56 bps`

## Sampling Artifacts

No headless `TracyCapture` binary was available on the runner, so published tracing data for this run is perf sampling output (summary + report + flamegraph SVG) for native runs.

| branch | fixture | bps | samples | unique stacks | artifacts |
|---|---|---:|---:|---:|---|
| btree | v0 | 72.720 | 1342 | 265 | [flamegraph](trace/btree/v0/perf-flamegraph.svg) [summary](trace/btree/v0/perf-summary.json) [report](trace/btree/v0/perf-report.txt) |
| btree | v1 | 69.790 | 1464 | 274 | [flamegraph](trace/btree/v1/perf-flamegraph.svg) [summary](trace/btree/v1/perf-summary.json) [report](trace/btree/v1/perf-report.txt) |
| btree | v2 | 71.450 | 1372 | 262 | [flamegraph](trace/btree/v2/perf-flamegraph.svg) [summary](trace/btree/v2/perf-summary.json) [report](trace/btree/v2/perf-report.txt) |
| bump PMA | v0 | 76.340 | 1257 | 243 | [flamegraph](trace/bump-pma/v0/perf-flamegraph.svg) [summary](trace/bump-pma/v0/perf-summary.json) [report](trace/bump-pma/v0/perf-report.txt) |
| bump PMA | v1 | 77.650 | 1373 | 274 | [flamegraph](trace/bump-pma/v1/perf-flamegraph.svg) [summary](trace/bump-pma/v1/perf-summary.json) [report](trace/bump-pma/v1/perf-report.txt) |
| bump PMA | v2 | 77.920 | 1231 | 257 | [flamegraph](trace/bump-pma/v2/perf-flamegraph.svg) [summary](trace/bump-pma/v2/perf-summary.json) [report](trace/bump-pma/v2/perf-report.txt) |
| master | v0 | 25.480 | 1787 | 255 | [flamegraph](trace/master/v0/perf-flamegraph.svg) [summary](trace/master/v0/perf-summary.json) [report](trace/master/v0/perf-report.txt) |
| master | v1 | 26.110 | 1729 | 231 | [flamegraph](trace/master/v1/perf-flamegraph.svg) [summary](trace/master/v1/perf-summary.json) [report](trace/master/v1/perf-report.txt) |
| master | v2 | 25.560 | 1707 | 244 | [flamegraph](trace/master/v2/perf-flamegraph.svg) [summary](trace/master/v2/perf-summary.json) [report](trace/master/v2/perf-report.txt) |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
