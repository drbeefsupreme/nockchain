---
phase: 03-durable-history-and-pages-publication
plan: 01
subsystem: infra
tags: [bash, github-actions, gh-pages, jq, awk, peaceiris, history, baseline]

requires:
  - phase: 01-reproducible-baseline-and-provenance
    provides: manifest.json provenance format, combined_summary.tsv TSV schema
  - phase: 02-regression-comparison-and-pr-gates
    provides: sol-baseline.yml workflow with artifact upload steps

provides:
  - sol_history_append.sh: generates per-run JSON from manifest + TSV, appends to index.json
  - gh-pages/history/{run_id}.json: immutable per-run provenance + metric record
  - gh-pages/history/index.json: single source of truth for run discovery
  - gh-pages/history/baseline-active.json: tracks the promoted active baseline run
  - sol-advance-baseline.yml: manual workflow_dispatch for promoting active baseline

affects:
  - phase-04: any dashboard or pages publication that reads history/index.json

tech-stack:
  added: [peaceiris/actions-gh-pages@v4]
  patterns:
    - keep_files: true on all gh-pages pushes to preserve existing history files
    - bench-history-write concurrency group serializes all history writes across workflows
    - PUBLISH_DIR contains only new/changed files (not entire history) to work with keep_files
    - Column-name lookup via awk header scan (not hardcoded column positions) for TSV robustness

key-files:
  created:
    - scripts/sol_history_append.sh
    - .github/workflows/sol-advance-baseline.yml
    - crates/nockchain-bench/tests/fixtures/guard/manifest.json
  modified:
    - .github/workflows/sol-baseline.yml

key-decisions:
  - "PUBLISH_DIR contains only new/changed files so peaceiris keep_files: true preserves all prior run JSONs"
  - "bench-history-write concurrency group (cancel-in-progress: false) used by both workflows to serialize writes"
  - "Column-name lookup via awk avoids hardcoded column positions for TSV schema resilience"
  - "manifest.json symlink added in fixtures so verify command matches actual fixture filename (run-manifest.json)"

patterns-established:
  - "History append: write only delta files + updated index, never full history directory"
  - "Active baseline tracking: separate baseline-active.json + is_active_baseline flag in index.json"
  - "Concurrency serialization: same group name across all workflows that touch gh-pages history branch"

requirements-completed: [DATA-03, PIPE-01]

duration: 2min
completed: 2026-02-26
---

# Phase 3 Plan 01: Durable History and Pages Publication Summary

**Bash history append pipeline and manual baseline promotion: sol_history_append.sh generates per-run JSON with median metrics, sol-baseline.yml pushes each run to gh-pages/history/ via peaceiris with keep_files, and sol-advance-baseline.yml enables workflow_dispatch promotion with provenance validation.**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-26T17:51:33Z
- **Completed:** 2026-02-26T17:53:37Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- sol_history_append.sh computes 4 metric medians (throughput_blocks_s, init_time_s, avg_per_block_ms, peak_rss_mib) via awk column-name lookup from TSV header, then writes per-run JSON + updated index.json to PUBLISH_DIR
- sol-baseline.yml extended with contents/pages/id-token write permissions, bench-history-write concurrency group, and two new steps: "Build history payload" (fetches existing index from gh-pages, runs append script) and "Push to history branch" (peaceiris@v4, keep_files: true)
- sol-advance-baseline.yml created as workflow_dispatch-only workflow: validates run JSON exists and has git_commit/timestamp/metrics fields, writes baseline-active.json, updates is_active_baseline flag in index.json, pushes via peaceiris with keep_files: true using same concurrency group

## Task Commits

Each task was committed atomically:

1. **Task 1: Create sol_history_append.sh and per-run JSON generation** - `4eecea1` (feat)
2. **Task 2: Extend sol-baseline.yml with history append and create advancement workflow** - `a02f19d` (feat)

## Files Created/Modified
- `scripts/sol_history_append.sh` - Bash script: per-run JSON generation from manifest + TSV, index.json append. Reads provenance from manifest.json, computes medians via awk, writes {run_id}.json + index.json to PUBLISH_DIR
- `.github/workflows/sol-advance-baseline.yml` - New workflow_dispatch workflow for manual baseline promotion with provenance validation
- `.github/workflows/sol-baseline.yml` - Extended with write permissions, concurrency group, history-append + peaceiris push steps
- `crates/nockchain-bench/tests/fixtures/guard/manifest.json` - Symlink to run-manifest.json for test verification compatibility

## Decisions Made
- PUBLISH_DIR pattern: only write new/changed files (per-run JSON + updated index) so peaceiris keep_files: true can preserve all prior history files on gh-pages
- bench-history-write concurrency group with cancel-in-progress: false used across both workflows to serialize all writes to gh-pages history branch and prevent concurrent corruption
- Column-name lookup via awk header scan instead of hardcoded column positions to avoid breakage if TSV schema gains new columns in future phases
- manifest.json symlink in guard fixtures so the plan's verify command (`bash ... manifest.json ...`) resolves correctly to the existing run-manifest.json fixture

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created manifest.json symlink for verify command compatibility**
- **Found during:** Task 1 (sol_history_append.sh verification)
- **Issue:** Plan's verify command references `crates/nockchain-bench/tests/fixtures/guard/manifest.json` but the actual fixture file is named `run-manifest.json`
- **Fix:** Created `manifest.json` as a symlink to `run-manifest.json` in the guard fixtures directory
- **Files modified:** `crates/nockchain-bench/tests/fixtures/guard/manifest.json`
- **Verification:** verify command runs successfully and produces expected output
- **Committed in:** 4eecea1 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary compatibility fix for test verification. No scope creep.

## Issues Encountered
None beyond the manifest.json filename mismatch handled above.

## User Setup Required

**External services require manual configuration.** Repository must have GitHub Pages enabled:
- Task: "Enable GitHub Pages with 'GitHub Actions' as source"
- Location: Repository Settings -> Pages -> Source -> select 'GitHub Actions'

This is required for the peaceiris/actions-gh-pages push to be served as a GitHub Pages site.

## Next Phase Readiness
- History append infrastructure is complete and CI-automated
- gh-pages branch will accumulate run JSONs and maintain index.json after each baseline run
- baseline-active.json can be promoted via workflow_dispatch at any time
- Next: GitHub Pages publication (serving history/index.json as a discoverable API or dashboard)

---
*Phase: 03-durable-history-and-pages-publication*
*Completed: 2026-02-26*
