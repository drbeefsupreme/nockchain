# SOL Trace + Sampling Report

- Date: `2026-02-17`
- Run ID: `20260217_180344`
- Report pass: `2`
- Matrix: `3 branches x 3 fixtures x native+docker`
- Checkpointing: `off`
- Memory profiling: `off`
- Tracing: `tracy instrumentation enabled by default`
- Stack sampling: `perf record -g` (native runs); flamegraphs generated from folded stacks

## Throughput Summary

- Best native: `bump PMA / v2 / 77.70 bps`
- Best docker: `bump PMA / v0 / 75.56 bps`

## Sampling Artifacts

No headless `TracyCapture` binary was available on the runner, so published tracing data for this run is perf sampling output (summary + report + flamegraph SVG) for native runs.
Raw downloads are also provided per run: `perf.folded`, `perf.script`, and `perf.data`.

| branch | fixture | bps | samples | unique stacks | artifacts |
|---|---|---:|---:|---:|---|
| btree | v0 | 72.530 | 1309 | 216 | [flamegraph](trace/btree/v0/perf-flamegraph.svg) [summary](trace/btree/v0/perf-summary.json) [report](trace/btree/v0/perf-report.txt) [folded](trace/btree/v0/perf.folded) [perf.script](trace/btree/v0/perf.script) [perf.data](trace/btree/v0/perf.data) |
| btree | v1 | 72.080 | 1403 | 236 | [flamegraph](trace/btree/v1/perf-flamegraph.svg) [summary](trace/btree/v1/perf-summary.json) [report](trace/btree/v1/perf-report.txt) [folded](trace/btree/v1/perf.folded) [perf.script](trace/btree/v1/perf.script) [perf.data](trace/btree/v1/perf.data) |
| btree | v2 | 71.810 | 1356 | 220 | [flamegraph](trace/btree/v2/perf-flamegraph.svg) [summary](trace/btree/v2/perf-summary.json) [report](trace/btree/v2/perf-report.txt) [folded](trace/btree/v2/perf.folded) [perf.script](trace/btree/v2/perf.script) [perf.data](trace/btree/v2/perf.data) |
| bump PMA | v0 | 77.260 | 1287 | 235 | [flamegraph](trace/bump-pma/v0/perf-flamegraph.svg) [summary](trace/bump-pma/v0/perf-summary.json) [report](trace/bump-pma/v0/perf-report.txt) [folded](trace/bump-pma/v0/perf.folded) [perf.script](trace/bump-pma/v0/perf.script) [perf.data](trace/bump-pma/v0/perf.data) |
| bump PMA | v1 | 76.440 | 1232 | 224 | [flamegraph](trace/bump-pma/v1/perf-flamegraph.svg) [summary](trace/bump-pma/v1/perf-summary.json) [report](trace/bump-pma/v1/perf-report.txt) [folded](trace/bump-pma/v1/perf.folded) [perf.script](trace/bump-pma/v1/perf.script) [perf.data](trace/bump-pma/v1/perf.data) |
| bump PMA | v2 | 77.700 | 1357 | 249 | [flamegraph](trace/bump-pma/v2/perf-flamegraph.svg) [summary](trace/bump-pma/v2/perf-summary.json) [report](trace/bump-pma/v2/perf-report.txt) [folded](trace/bump-pma/v2/perf.folded) [perf.script](trace/bump-pma/v2/perf.script) [perf.data](trace/bump-pma/v2/perf.data) |
| master | v0 | 26.310 | 1610 | 303 | [flamegraph](trace/master/v0/perf-flamegraph.svg) [summary](trace/master/v0/perf-summary.json) [report](trace/master/v0/perf-report.txt) [folded](trace/master/v0/perf.folded) [perf.script](trace/master/v0/perf.script) [perf.data](trace/master/v0/perf.data) |
| master | v1 | 26.600 | 1645 | 315 | [flamegraph](trace/master/v1/perf-flamegraph.svg) [summary](trace/master/v1/perf-summary.json) [report](trace/master/v1/perf-report.txt) [folded](trace/master/v1/perf.folded) [perf.script](trace/master/v1/perf.script) [perf.data](trace/master/v1/perf.data) |
| master | v2 | 26.210 | 1652 | 309 | [flamegraph](trace/master/v2/perf-flamegraph.svg) [summary](trace/master/v2/perf-summary.json) [report](trace/master/v2/perf-report.txt) [folded](trace/master/v2/perf.folded) [perf.script](trace/master/v2/perf.script) [perf.data](trace/master/v2/perf.data) |

## Files

- `combined_summary.tsv`
- `sol-benchmark-transplant-report.html`
- `sol-benchmark-transplant-report.md`
- `sol-benchmark-transplant-memory-profiles.json`
