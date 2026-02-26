---
phase: 03-durable-history-and-pages-publication
verified: 2026-02-26T19:00:00Z
status: human_needed
score: 10/10 must-haves verified
re_verification: true
  previous_status: gaps_found
  previous_score: 8/10
  gaps_closed:
    - "Dashboard displays trend charts for all key metrics across historical runs"
    - "Dashboard allows drill-down into individual benchmark details"
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Verify dashboard renders correctly in browser with real history data"
    expected: "Four trend line charts display numeric metric values. Active baseline run appears with a star-shaped point and yellow ACTIVE badge in both charts and run table. Header shows run count, latest date, and active baseline info."
    why_human: "Visual rendering, chart interactivity, responsive layout, and color scheme cannot be verified programmatically."
  - test: "Trigger the sol-baseline workflow manually (workflow_dispatch) and confirm gh-pages branch is updated with history/ files and Pages deployment completes"
    expected: "gh-pages branch gains a new history/{run_id}.json and updated history/index.json; sol-pages-deploy.yml deploy-pages job completes; GitHub Pages URL serves updated dashboard within ~2 minutes"
    why_human: "Requires live GitHub Actions execution, gh-pages branch write access, and GitHub Pages deployment to verify."
  - test: "Trigger sol-advance-baseline workflow with a valid run_id and verify baseline-active.json is written and Pages redeploys"
    expected: "history/baseline-active.json appears on gh-pages branch, history/index.json has is_active_baseline=true for the promoted run, Pages redeploy completes, and dashboard highlights that run."
    why_human: "Requires live workflow execution and inspection of gh-pages branch content and deployed Pages site."
---

# Phase 3: Durable History and Pages Publication - Verification Report

**Phase Goal:** Baseline history is immutable and continuously extended, with latest baseline datasets and history automatically published for team access.
**Verified:** 2026-02-26T19:00:00Z
**Status:** HUMAN NEEDED (all automated checks pass)
**Re-verification:** Yes — after gap closure (commit 8f4ec28 fixed metric key name mismatch)

## Re-Verification Summary

The previous verification (2026-02-26T18:02:17Z) found one root-cause gap: `scripts/sol_history_append.sh` wrote metric keys WITHOUT the `_median` suffix (`metrics.throughput_blocks_s`) while `pages/index.html` read metric keys WITH the `_median` suffix (`metrics.throughput_blocks_s_median`). This caused all four metric values to resolve to `null` in trend charts, the run table, and detail panels.

Commit 8f4ec28 fixed this by updating lines 173-176 of `scripts/sol_history_append.sh` to write `throughput_blocks_s_median`, `init_time_s_median`, `avg_per_block_ms_median`, and `peak_rss_mib_median` — aligning with the interface spec in 03-02-PLAN.md and the dashboard's consumption of those keys.

All 10 truths now verify as VERIFIED. No regressions found.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Baseline run produces a per-run JSON file with provenance and metric medians | VERIFIED | `scripts/sol_history_append.sh` (214 lines, syntax valid) writes `{RUN_ID}.json` with `metrics.throughput_blocks_s_median`, `init_time_s_median`, `avg_per_block_ms_median`, `peak_rss_mib_median` — now matching dashboard expectations |
| 2 | CI baseline workflow appends new run to gh-pages branch history without deleting prior runs | VERIFIED | `sol-baseline.yml` "Push to history branch" step uses `peaceiris/actions-gh-pages@v4` with `keep_files: true` and `destination_dir: history` |
| 3 | history/index.json is updated with each new run entry | VERIFIED | Script appends new entry to `EXISTING_INDEX` (fetched from gh-pages before the script runs) and writes to `PUBLISH_DIR/index.json`; published via peaceiris action |
| 4 | Concurrent history writes are serialized via concurrency group | VERIFIED | Both `sol-baseline.yml` and `sol-advance-baseline.yml` declare `concurrency: group: bench-history-write, cancel-in-progress: false` |
| 5 | Maintainer can advance active baseline via workflow_dispatch with provenance validation | VERIFIED | `sol-advance-baseline.yml` accepts `run_id` input, validates `history/{run_id}.json` exists and has `git_commit`, `timestamp`, `metrics` fields before writing |
| 6 | baseline-active.json records promoted run_id and timestamp | VERIFIED | Workflow writes JSON with `run_id`, `promoted_at`, `git_commit`, `promoted_by` fields to `advance-payload/baseline-active.json` pushed to `history/` destination with `keep_files: true` |
| 7 | Dashboard displays trend charts for all key metrics across historical runs | VERIFIED | **GAP CLOSED.** Script now writes `metrics.throughput_blocks_s_median` (and 3 other `_median` keys) at lines 173-176. Dashboard reads identical keys at lines 271-292 and 510-513. Key alignment is now correct. |
| 8 | Dashboard highlights which run is the active baseline | VERIFIED | Dashboard compares `entry.run_id === activeBaselineId` for star point style in Chart.js and `star-badge ACTIVE` in table; metric display is no longer broken by the key mismatch |
| 9 | Dashboard allows drill-down into individual benchmark details | VERIFIED | **GAP CLOSED.** Detail panel reads `m.throughput_blocks_s_median` etc. at lines 547-550. These keys now exist in per-run JSON produced by the fixed script. Expand/collapse mechanism is also wired correctly. |
| 10 | Pages are automatically deployed after history append and baseline advancement | VERIFIED | Both `sol-baseline.yml` and `sol-advance-baseline.yml` have `deploy-pages` job calling `sol-pages-deploy.yml` via `workflow_call` after their primary job completes |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/sol_history_append.sh` | Per-run JSON generation from manifest + TSV, index.json append | VERIFIED | 214 lines (min_lines: 60 satisfied), syntax valid (`bash -n` passes), jq output block at lines 172-177 now writes all four `_median`-suffixed metric keys |
| `.github/workflows/sol-baseline.yml` | History-append step after artifact upload | VERIFIED | 127 lines, YAML valid, contains `peaceiris/actions-gh-pages@v4`, `bash scripts/sol_history_append.sh`, `keep_files: true`, `concurrency` group, `deploy-pages` job |
| `.github/workflows/sol-advance-baseline.yml` | Manual baseline advancement with validation | VERIFIED | 97 lines, YAML valid, `workflow_dispatch` trigger with `run_id` input, provenance validation step (`jq -e '.git_commit and .timestamp and .metrics'`), `keep_files: true`, same concurrency group, `deploy-pages` job |
| `pages/index.html` | Client-side rendered dashboard with Chart.js trend charts | VERIFIED | 678 lines (min_lines: 100 satisfied), Chart.js 4.4.1 CDN present, fetches `history/index.json` and `history/baseline-active.json`, renders 4 charts with metric keys now aligned to script output |
| `.github/workflows/sol-pages-deploy.yml` | GitHub Pages deployment pipeline | VERIFIED | 51 lines, YAML valid, `workflow_call` and `workflow_dispatch` triggers, `configure-pages@v5`, `upload-pages-artifact@v3`, `deploy-pages@v4` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scripts/sol_history_append.sh` | `bench-artifacts/.../manifest.json` | reads provenance from manifest (jq) | WIRED | Lines 60-66: `jq -r '.timestamp'`, `.git_commit`, `.git_branch`, `.benchmark_config.profile`, `.environment.*` from `$MANIFEST_JSON` |
| `scripts/sol_history_append.sh` | `bench-artifacts/.../combined_summary.tsv` | computes metric medians from TSV (awk) | WIRED | `compute_median()` function uses `awk` with column-name lookup; called at lines 139-142 with `$SUMMARY_TSV` |
| `.github/workflows/sol-baseline.yml` | `scripts/sol_history_append.sh` | bash invocation after artifact upload | WIRED | Line 104: `bash scripts/sol_history_append.sh` in "Build history payload" step |
| `.github/workflows/sol-advance-baseline.yml` | `gh-pages:baseline-active.json` | writes manifest via peaceiris action | WIRED | Line 87: `keep_files: true`; writes to `advance-payload/baseline-active.json` pushed to `history/` destination |
| `pages/index.html` | `gh-pages:history/index.json` | fetch() call to load run manifest | WIRED | Line 596 (approx): `index = await fetchJSON('history/index.json')` |
| `pages/index.html` | `gh-pages:baseline-active.json` | fetch() call to load active baseline | WIRED | `activeBaseline = await fetchJSON('history/baseline-active.json')` |
| `.github/workflows/sol-baseline.yml` | `.github/workflows/sol-pages-deploy.yml` | triggers Pages deploy after history append | WIRED | Line 122: `uses: ./.github/workflows/sol-pages-deploy.yml` in `deploy-pages` job |
| `.github/workflows/sol-advance-baseline.yml` | `.github/workflows/sol-pages-deploy.yml` | triggers Pages deploy after baseline advancement | WIRED | Line 92: `uses: ./.github/workflows/sol-pages-deploy.yml` in `deploy-pages` job |
| `pages/index.html` | `metrics.{key}_median` (per-run JSON) | metric values from run JSON files | WIRED | **GAP CLOSED.** Script lines 173-176 now write `throughput_blocks_s_median`, `init_time_s_median`, `avg_per_block_ms_median`, `peak_rss_mib_median`. Dashboard reads identical keys at lines 271-292, 510-513, 547-550. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DATA-03 | 03-01-PLAN.md | Maintainer can preserve immutable historical run records while tracking the active baseline reference | SATISFIED | `sol_history_append.sh` writes per-run JSONs; `peaceiris@v4` with `keep_files: true` preserves immutability; `baseline-active.json` tracks active reference; `REQUIREMENTS.md` marks as complete for Phase 3 |
| PIPE-01 | 03-01-PLAN.md | Maintainer can run scheduled baseline generation in CI that appends new baseline history without deleting prior runs | SATISFIED | `sol-baseline.yml` runs on push to master and workflow_dispatch; appends via `keep_files: true`; concurrency group `bench-history-write` prevents race conditions; `REQUIREMENTS.md` marks as complete for Phase 3 |
| PIPE-03 | 03-02-PLAN.md | Maintainer can publish benchmark history and latest baseline artifacts to GitHub Pages through an automated workflow | SATISFIED | Full publication pipeline wired: `sol-pages-deploy.yml` (configure-pages, upload-pages-artifact, deploy-pages), triggered via `workflow_call` from both history workflows; dashboard now correctly reads all metric keys from produced JSON; `REQUIREMENTS.md` marks as complete for Phase 3 |

No orphaned requirements: REQUIREMENTS.md maps DATA-03, PIPE-01, PIPE-03 to Phase 3 (lines 20, 29, 31 and table rows 74, 77, 79). All three appear in plan frontmatter and are verified.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `pages/index.html` | 265, 495 | `'OWNER/REPO'` fallback string | Info | Not a blocker — these are dynamic fallbacks for non-github.io contexts, computed from `window.location.hostname`; not a static placeholder |

No blocker anti-patterns remain. The `_median` key mismatch that was a blocker in the prior verification has been resolved by commit 8f4ec28.

### Human Verification Required

All automated checks pass. The following items require live execution to verify end-to-end behavior.

#### 1. Dashboard Visual Rendering

**Test:** Serve the dashboard in a browser with real history data. The simplest approach is to checkout the gh-pages branch (after at least one baseline run has pushed to it) and run `python3 -m http.server 8080` from that checkout, then visit http://localhost:8080.
**Expected:** Four trend line charts display numeric metric values for throughput, init time, per-block time, and peak RSS. The active baseline run appears with a star-shaped point style and yellow ACTIVE badge in both charts and the run table. Header shows run count, latest date, and active baseline info.
**Why human:** Visual rendering, chart interactivity, responsive layout, and color scheme cannot be verified programmatically.

#### 2. End-to-End CI Baseline Run + Pages Deploy

**Test:** Trigger `sol-baseline.yml` via workflow_dispatch with `profile: quick`. Monitor the workflow run in the GitHub Actions UI.
**Expected:** (1) benchmark completes, (2) "Build history payload" step succeeds and produces a valid RUN_ID.json and index.json, (3) "Push to history branch" step pushes to gh-pages branch preserving existing files, (4) `deploy-pages` job triggers `sol-pages-deploy.yml` and completes, (5) GitHub Pages URL updates within ~2 minutes showing the new run in the dashboard with metric values displayed (not `-`).
**Why human:** Requires live GitHub Actions execution, gh-pages branch write access, and GitHub Pages deployment to verify.

#### 3. Baseline Advancement Workflow

**Test:** Trigger `sol-advance-baseline.yml` via workflow_dispatch with a valid `run_id` from a previous baseline run.
**Expected:** `history/baseline-active.json` is created/updated on gh-pages branch, `history/index.json` shows `is_active_baseline: true` for the promoted run and `false` for all others, Pages redeploy completes, and the dashboard highlights the promoted run with the star point style and ACTIVE badge.
**Why human:** Requires live workflow execution, access to gh-pages branch content inspection, and visual verification of the deployed dashboard.

### Gaps Summary

No gaps remain. The single root-cause gap from the previous verification — the metric key name mismatch between `scripts/sol_history_append.sh` and `pages/index.html` — was fixed in commit 8f4ec28. All phase infrastructure is correctly implemented, wired, and substantive:

- History append pipeline: immutable gh-pages storage (`keep_files: true`), concurrency serialization (`bench-history-write`), column-name-lookup median computation
- Metric key alignment: script writes `*_median` keys at lines 173-176; dashboard reads identical `*_median` keys at all three consumption sites (chart METRICS array, table rows, detail panel)
- Index management: `index.json` updated per run with correct schema, `is_active_baseline` flag managed by advancement workflow
- Baseline advancement: `workflow_dispatch` only, provenance validation, `baseline-active.json` with provenance fields
- Pages deployment: official GitHub Actions pipeline (`configure-pages@v5`, `upload-pages-artifact@v3`, `deploy-pages@v4`), triggered automatically via `workflow_call` after both history operations

---

_Verified: 2026-02-26T19:00:00Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification: Yes — gap closure after commit 8f4ec28_
