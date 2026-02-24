# Phase 2: Regression Comparison and PR Gates - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Turn Phase 1's baseline benchmark artifacts into statistically defensible comparison outcomes and enforce them during PR review. Maintainers can compare a candidate run against the active baseline and receive a clear classification with effect size, inspect machine-readable and human-readable outputs, and see PR-time CI report regression results. Scheduled baseline generation, immutable history, and Pages publication are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Statistical verdict
- Balanced confidence threshold (~90% significance) as default
- Per-benchmark threshold overrides supported — global defaults with optional per-benchmark config for noisy vs stable benchmarks
- Four-way classification: improvement, regression, no significant change, inconclusive (insufficient data/noisy)
- Report both verdict AND effect size (percentage change / absolute delta) for every benchmark

### Comparison output
- Human-readable summary in GitHub-flavored Markdown — two tiers:
  - Compact summary table at top (one line per benchmark: name, verdict, effect size, confidence)
  - Expandable per-benchmark detail section below with statistical reasoning
- Machine-readable delta format at Claude's discretion (JSON likely, but pick what integrates best with CI and Phase 1 artifacts)
- Output file location at Claude's discretion (fit with Phase 1's artifact layout)

### PR gate behavior
- Warning only — advisory check, does not block merge
- Results appear in BOTH PR comment (bot-posted Markdown summary) AND GitHub Actions check run summary
- Inconclusive verdict treated as warning — flagged visually but doesn't change check status
- PR workflow compares pre-existing artifacts only — does NOT run benchmarks itself (assumes candidate run artifacts already exist)

### Baseline reference
- Pinned reference file pointing to a specific baseline run
- Stored as CI artifact/cache (not checked into the repo)
- When no baseline exists (first run, cache expired): skip gracefully — report "no baseline available" and pass
- Auto-update baseline reference on merge to main — CI promotes the latest run after successful merge

### Claude's Discretion
- Statistical test selection (e.g., Mann-Whitney U, t-test, bootstrap)
- Machine-readable delta format and schema
- Comparison output file location within artifact structure
- CI cache key strategy for baseline reference
- PR comment formatting and bot identity
- Exact effect size calculation methodology

</decisions>

<specifics>
## Specific Ideas

- Phase 1 already produces run directories in `bench-artifacts/` with `data/`, `meta/`, `logs/` structure and a `latest` symlink — comparison tool consumes these directly
- Phase 1's provenance guard module (`crates/nockchain-bench/src/speed_of_light/guard/`) already has baseline and stats submodules — extend rather than duplicate
- The "compare pre-existing artifacts" PR model means the workflow is fast and cheap — just runs the statistical comparison, not the benchmarks themselves
- Auto-baseline-on-merge dovetails with Phase 3's scheduled baseline refresh — Phase 2 sets the update mechanism, Phase 3 adds scheduling and history

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-regression-comparison-and-pr-gates*
*Context gathered: 2026-02-24*
