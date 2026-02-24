---
phase: 02-regression-comparison-and-pr-gates
plan: 01
subsystem: testing
tags: [benchmark, statistics, bootstrap, regression, comparison]

requires:
  - phase: 01-reproducible-baseline-execution
    provides: combined_summary.tsv canonical format, sol CLI framework, guard module patterns
provides:
  - ComparisonVerdict four-way classification engine
  - Bootstrap CI overlap comparison logic
  - Dual-format reporting (GFM Markdown + JSON delta)
  - sol compare CLI subcommand
  - CI wrapper script for comparison
  - Contract tests for comparison logic
affects: [02-regression-comparison-and-pr-gates, 03-durable-history-and-pages-publication]

tech-stack:
  added: []
  patterns: [bootstrap CI overlap for verdict classification, two-tier GFM reports]

key-files:
  created:
    - crates/nockchain-bench/src/speed_of_light/guard/compare.rs
    - crates/nockchain-bench/src/speed_of_light/guard/compare_report.rs
    - scripts/sol_compare_ci.sh
    - crates/nockchain-bench/tests/sol_comparison.rs
  modified:
    - crates/nockchain-bench/src/speed_of_light/guard/model.rs
    - crates/nockchain-bench/src/speed_of_light/guard/mod.rs
    - crates/nockchain-bench/src/main.rs

key-decisions:
  - "Bootstrap CI overlap for verdict classification rather than simple threshold comparison"
  - "Advisory-only exit code 0 for all comparison outcomes per user decision"
  - "Two-tier Markdown with compact summary table and expandable per-metric detail"

patterns-established:
  - "ComparisonVerdict four-way classification: Improvement, Regression, NoSignificantChange, Inconclusive"
  - "MetricDirection enum for higher-is-better vs lower-is-better semantics"
  - "CI wrapper pattern: graceful no-baseline handling, cargo fallback"

requirements-completed: [STAT-01, STAT-02]

duration: 8min
completed: 2026-02-24
---

# Plan 02-01: Statistical Comparison Engine Summary

**Four-way bootstrap CI comparison engine with dual-format GFM/JSON reporting and `sol compare` CLI subcommand**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-24
- **Completed:** 2026-02-24
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- ComparisonVerdict four-way enum classifies metrics via bootstrap CI overlap and MetricDirection
- run_comparison consumes two TSV paths and produces ComparisonReport with per-metric verdicts
- Two-tier GitHub-flavored Markdown report with compact summary table and expandable detail section
- Machine-readable JSON delta with per-metric verdicts, effect sizes, and confidence levels
- `sol compare` CLI subcommand with configurable significance threshold and min samples
- CI wrapper script with graceful missing-baseline handling
- 5 contract tests all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: ComparisonVerdict model, compare.rs engine, compare_report.rs rendering** - `7e98a32` (feat)
2. **Task 2: sol compare CLI, CI wrapper, contract tests** - `b1781d3` (feat)

## Files Created/Modified
- `crates/nockchain-bench/src/speed_of_light/guard/compare.rs` - Classification engine with CI overlap logic
- `crates/nockchain-bench/src/speed_of_light/guard/compare_report.rs` - Dual-format Markdown and JSON rendering
- `crates/nockchain-bench/src/speed_of_light/guard/model.rs` - ComparisonVerdict, ComparisonConfig, MetricDirection types
- `crates/nockchain-bench/src/speed_of_light/guard/mod.rs` - Module registration and re-exports
- `crates/nockchain-bench/src/main.rs` - sol compare subcommand and cmd_sol_compare handler
- `scripts/sol_compare_ci.sh` - CI wrapper for sol compare
- `crates/nockchain-bench/tests/sol_comparison.rs` - 5 contract tests

## Decisions Made
- Bootstrap CI overlap for verdict classification rather than simple threshold comparison
- Advisory-only exit code 0 for all comparison outcomes per user decision
- Two-tier Markdown with compact summary table and expandable per-metric detail
- CanonicalMetric gets Hash + Ord derives for use as BTreeMap key

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Comparison engine ready for PR workflow integration (Plan 02-02)
- sol compare CLI and CI wrapper available for GitHub Actions integration

---
*Phase: 02-regression-comparison-and-pr-gates*
*Completed: 2026-02-24*
