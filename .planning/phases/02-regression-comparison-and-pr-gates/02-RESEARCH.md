# Phase 2: Regression Comparison and PR Gates - Research

**Researched:** 2026-02-24
**Domain:** Statistical benchmark comparison, CI artifact caching, GitHub Actions PR integration (Rust/Bash)
**Confidence:** HIGH

## Summary

Phase 2 has an unusually favorable starting position: the hard parts are already built. The `guard/` module in `crates/nockchain-bench/src/speed_of_light/guard/` already implements the full statistical comparison pipeline — median/MAD, bootstrap confidence intervals, contract evaluation with per-rule severity/weight, JSON and Markdown report writing, and a working `sol guard` CLI subcommand with exit codes. The existing `sol_guard_ci.sh` script already wraps that subcommand for CI use.

What Phase 2 adds is the missing operational layer: (1) a four-way classification verdict aligned with the user-defined semantics (improvement / regression / no significant change / inconclusive), (2) a pinned baseline reference mechanism (cache key pointing to a specific artifact), (3) a PR CI workflow that compares pre-existing artifacts against that cached baseline, (4) auto-update of the baseline reference on merge to main, and (5) a GitHub PR comment posted with the Markdown summary. The statistical engine, contract DSL, TSV ingestion, and report rendering are all present and tested — Phase 2 extends rather than replaces them.

The primary architectural challenge is the baseline reference mechanism. GitHub Actions cache is the right tool: a stable cache key points to the canonical baseline artifact, the PR workflow restores it, compares, and posts results. On merge to main, a separate workflow step saves the new run as the new baseline cache. The "no baseline" graceful path (first run, cache miss) is handled by checking for the cached file before running the comparison.

**Primary recommendation:** Extend the existing `guard/` module with a four-way `ComparisonVerdict` type and a `sol compare` CLI subcommand that consumes two run directories (candidate and baseline). Wire it into a new PR workflow that restores baseline from cache, runs the comparison, posts a PR comment via `gh`, and saves the new baseline on merge.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Statistical verdict:**
- Balanced confidence threshold (~90% significance) as default
- Per-benchmark threshold overrides supported — global defaults with optional per-benchmark config for noisy vs stable benchmarks
- Four-way classification: improvement, regression, no significant change, inconclusive (insufficient data/noisy)
- Report both verdict AND effect size (percentage change / absolute delta) for every benchmark

**Comparison output:**
- Human-readable summary in GitHub-flavored Markdown — two tiers:
  - Compact summary table at top (one line per benchmark: name, verdict, effect size, confidence)
  - Expandable per-benchmark detail section below with statistical reasoning
- Machine-readable delta format at Claude's discretion (JSON likely, but pick what integrates best with CI and Phase 1 artifacts)
- Output file location at Claude's discretion (fit with Phase 1's artifact layout)

**PR gate behavior:**
- Warning only — advisory check, does not block merge
- Results appear in BOTH PR comment (bot-posted Markdown summary) AND GitHub Actions check run summary
- Inconclusive verdict treated as warning — flagged visually but doesn't change check status
- PR workflow compares pre-existing artifacts only — does NOT run benchmarks itself (assumes candidate run artifacts already exist)

**Baseline reference:**
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

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| STAT-01 | Maintainer can compare a candidate run against baseline data using statistically defensible classification (improvement, regression, no significant change) | Existing `evaluate_contract()` + bootstrap CI in `guard/contract.rs` and `guard/stats.rs` — extend with four-way `ComparisonVerdict`; use bootstrap median CI overlap as significance test |
| STAT-02 | Maintainer can inspect comparison output with both machine-readable deltas and human-readable summary suitable for review | `write_json()` and `render_markdown()` already exist in `guard/report.rs` — extend for two-tier GFM layout and delta JSON schema |
| PIPE-02 | Maintainer can run PR-time regression checks in CI against established baseline data | New `.github/workflows/sol-pr-regression.yml`; GitHub Actions cache for baseline artifact; `gh` CLI to post PR comment; advisory check-run status |
</phase_requirements>

## Standard Stack

### Core

| Library/Tool | Version | Purpose | Why Standard |
|-------------|---------|---------|--------------|
| Rust (existing guard module) | stable | Statistical comparison engine, report generation | Already implemented; bootstrap CI, MAD, median in `guard/stats.rs`; contract eval in `guard/contract.rs` |
| clap | 4 | CLI subcommand for `sol compare` | Already in nockchain-bench; derive macros |
| serde_json | 1.0 | Machine-readable delta output (JSON) | Already in dependencies; canonical for CI consumption |
| Bash | system | Shell wrapper for PR workflow steps | Existing pattern; `sol_guard_ci.sh` already wraps the guard subcommand |
| GitHub Actions cache | v4 | Pinned baseline reference storage | Actions cache is free, fast, and survives across workflow runs; 7-day TTL with restore-keys fallback |
| `gh` CLI | bundled on GitHub-hosted runners | Post PR comment, update check summary | Pre-installed on `ubuntu-latest`; `gh pr comment` is the idiomatic approach |
| actions/upload-artifact | v4 | Upload comparison report for inspection | Already used in sol-baseline.yml |

### Supporting

| Library/Tool | Version | Purpose | When to Use |
|-------------|---------|---------|-------------|
| actions/download-artifact | v4 | Retrieve candidate run artifacts in PR workflow | Needed if candidate artifacts are uploaded by a prior job in the same run |
| actions/cache | v4 | Store and restore pinned baseline artifact | Core mechanism for baseline reference persistence across runs |
| peter-evans/create-or-update-comment | v4 | Idiomatic PR comment creation/update (alternative to `gh`) | Use if `gh` auth setup is complicated in the workflow context |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| GitHub Actions cache for baseline | External storage (S3, GCS) | Cache is zero-config and free; external storage needs credentials and setup; acceptable for Phase 2 |
| Bootstrap median CI | Mann-Whitney U test | Bootstrap is already implemented in `guard/stats.rs`; Mann-Whitney would require a new implementation or external crate; bootstrap is appropriate for small samples (n=5–20) |
| `gh` CLI for PR comment | GitHub REST API via `curl` | `gh` is cleaner; pre-installed on GitHub-hosted runners; handles auth transparently |
| JSON for machine-readable delta | TOML or CSV | JSON integrates with `jq`, CI policy scripts, and Phase 1's existing serde_json usage |

**Installation:** No new dependencies needed. All required libraries are already in `Cargo.toml` or pre-installed on GitHub-hosted runners.

## Architecture Patterns

### Recommended Project Structure

Phase 2 adds these files to the existing structure:

```
crates/nockchain-bench/src/speed_of_light/guard/
├── compare.rs          # NEW: four-way ComparisonVerdict + per-metric ComparisonResult
├── compare_report.rs   # NEW: two-tier GFM Markdown + delta JSON rendering
├── baseline.rs         # EXISTING: extend with no-baseline detection
├── stats.rs            # EXISTING: bootstrap CI already present
└── model.rs            # EXISTING: add ComparisonVerdict enum

scripts/
├── sol_compare_ci.sh   # NEW: thin wrapper for `sol compare` in CI
└── sol_baseline_run.sh # EXISTING: unchanged

.github/workflows/
├── sol-baseline.yml    # EXISTING: add "save baseline cache" step on push to main
└── sol-pr-regression.yml  # NEW: PR trigger, restore cache, compare, post comment
```

### Pattern 1: Four-Way ComparisonVerdict

**What:** Extend `GuardVerdict` or add a parallel `ComparisonVerdict` enum for the statistical comparison context. The four-way classification maps CI overlap of bootstrap CIs to a verdict.

**When to use:** When comparing a single candidate metric against a distribution of baseline runs.

**Verdict logic (recommended):**
```rust
// Source: internal design, informed by bootstrap CI semantics in guard/stats.rs
pub enum ComparisonVerdict {
    Improvement,           // candidate CI entirely above baseline median + threshold
    Regression,            // candidate CI entirely below baseline median - threshold
    NoSignificantChange,   // CIs overlap within threshold
    Inconclusive,          // insufficient data or excessive noise (CV > threshold)
}

pub struct ComparisonResult {
    pub metric: CanonicalMetric,
    pub verdict: ComparisonVerdict,
    pub candidate_value: f64,
    pub baseline_median: f64,
    pub baseline_mad: f64,
    pub delta_pct: f64,
    pub delta_abs: f64,
    pub confidence: f64,         // 1.0 - alpha (e.g. 0.90)
    pub baseline_samples: usize,
    pub reason: String,
}
```

**Threshold logic:** Use `significance_threshold` (default 0.10 for ~90% confidence). If bootstrap CI of candidate does not overlap baseline CI → significant. Direction determines Improvement vs Regression.

### Pattern 2: Two-Tier GitHub-Flavored Markdown Report

**What:** Compact table at top (one row per metric), expandable `<details>` section below with full statistical reasoning per metric.

**When to use:** PR comment output and `--output-md` flag.

**Example structure:**
```markdown
## SOL Benchmark Regression Report

| Benchmark | Verdict | Effect | Confidence |
|-----------|---------|--------|-----------|
| throughput_blocks_s | ✅ no change | -1.2% | 90% |
| peak_rss_mib | 🔴 regression | +9.4% | 90% |
| init_time_s | ❓ inconclusive | +3.1% | — |

<details>
<summary>Per-benchmark statistical detail</summary>

### throughput_blocks_s
- Candidate: 10.02 blocks/s
- Baseline median: 10.15 blocks/s (MAD: 0.12, n=8)
- Delta: -0.13 blocks/s (-1.2%)
- Bootstrap 90% CI: [9.91, 10.14] overlaps baseline CI [10.03, 10.27]
- **Verdict: no significant change**

...
</details>

> Advisory only — this check does not block merge.
> Baseline: run `2026-02-20T12-00-00Z_abc1234` (8 samples)
```

### Pattern 3: GitHub Actions Cache-Based Baseline Reference

**What:** A stable cache key (e.g., `sol-baseline-ref-v1`) stores the path to the pinned baseline `combined_summary.tsv`. On merge to main, the baseline workflow saves the latest run's TSV as the new cache entry.

**When to use:** PR workflow restores cache, checks if file exists, skips gracefully if not.

```yaml
# In PR workflow: restore baseline
- name: Restore baseline reference
  id: baseline-cache
  uses: actions/cache/restore@v4
  with:
    path: .cache/sol-baseline-ref/
    key: sol-baseline-ref-v1
    restore-keys: sol-baseline-ref-

# Graceful skip if no baseline
- name: Run regression comparison
  run: |
    if [ ! -f .cache/sol-baseline-ref/combined_summary.tsv ]; then
      echo "No baseline available — skipping comparison"
      echo "## SOL Regression: No baseline available" >> "$GITHUB_STEP_SUMMARY"
      exit 0
    fi
    bash scripts/sol_compare_ci.sh \
      --candidate bench-artifacts/sol-baseline/latest/data/combined_summary.tsv \
      --baseline .cache/sol-baseline-ref/combined_summary.tsv \
      --output-json comparison/delta.json \
      --output-md comparison/summary.md
```

```yaml
# In baseline workflow (push to main): save new baseline
- name: Save baseline reference cache
  if: github.event_name == 'push' && github.ref == 'refs/heads/master'
  uses: actions/cache/save@v4
  with:
    path: .cache/sol-baseline-ref/
    key: sol-baseline-ref-v1-${{ github.sha }}
```

**Cache key strategy:** Use a versioned prefix (`sol-baseline-ref-v1`) so the cache can be invalidated intentionally by bumping the version. The `restore-keys` fallback ensures the most recent matching entry is used even if the exact key is absent.

### Pattern 4: PR Comment via `gh` CLI

**What:** Post or update a PR comment with the Markdown summary using the `gh` CLI.

```bash
# Post PR comment (idempotent via find-comment + update pattern)
gh pr comment "$PR_NUMBER" \
  --body-file comparison/summary.md \
  --repo "$GITHUB_REPOSITORY"
```

**Note:** The `gh` CLI on `ubuntu-latest` runners is authenticated via `GITHUB_TOKEN` automatically. The workflow needs `pull-requests: write` permission.

### Anti-Patterns to Avoid

- **Re-running benchmarks in the PR workflow:** The user decision is "compare pre-existing artifacts only." Never run `sol_bench_matrix.sh` in the PR workflow — it would make PRs expensive and slow.
- **Blocking merge on regression:** The decision is "warning only." Never set `continue-on-error: false` on the comparison step or use exit code 2 to fail the check run.
- **Using `GuardContract` for the comparison:** The existing `evaluate_contract()` is contract-based (floor/ceiling thresholds). Phase 2 needs a statistical comparison approach (CI overlap) that maps to the four-way verdict. Use the existing `stats.rs` primitives (bootstrap CI, MAD) but implement a separate `compare.rs` module.
- **Checking baseline reference into the repo:** It must live in CI artifact cache to avoid repo bloat and to allow non-repo environments to participate.
- **Creating a new stats engine:** `bootstrap_median_ci()`, `median()`, and `mad()` in `guard/stats.rs` are already correct and tested. Use them directly.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bootstrap confidence interval | Custom resampling loop | `bootstrap_median_ci()` in `guard/stats.rs` | Already implemented with deterministic seed, tested against edge cases |
| Median + MAD | Custom stat functions | `median()` + `mad()` in `guard/stats.rs` | Already handles outliers, tested |
| JSON serialization | String formatting | `serde_json` + existing derive macros | Type safety, no escaping bugs |
| PR comment posting | GitHub REST API via `curl` | `gh pr comment` | Pre-installed, handles auth, idiomatic |
| Significance classification | Custom threshold logic | Bootstrap CI overlap: if candidate CI does not overlap baseline CI, result is significant | CI overlap is a well-understood, robust approach for small samples |

**Key insight:** The guard module already solves the hardest problem (statistical comparison). Phase 2 is 80% about wiring existing code together and adding the four-way verdict semantics.

## Common Pitfalls

### Pitfall 1: Confusing `GuardVerdict` with `ComparisonVerdict`

**What goes wrong:** Reusing the existing `GuardVerdict` (Pass/Warn/Fail/InsufficientBaseline) for the comparison context forces the four-way classification into a three-way bucket and loses the Improvement/Regression distinction.

**Why it happens:** The guard module already has verdict types; it's tempting to extend rather than add.

**How to avoid:** Add a separate `ComparisonVerdict` enum in `model.rs` or a new `compare.rs`. `GuardVerdict` is contract-based (floor/ceiling pass/fail); `ComparisonVerdict` is statistics-based (CI overlap, direction, and magnitude).

**Warning signs:** If Improvement and NoSignificantChange are both mapped to "Pass," the four-way classification is lost.

### Pitfall 2: Cache Key Collisions Making Baseline Stale

**What goes wrong:** Using a fixed cache key (e.g., `sol-baseline-ref-v1`) without a save-side key increment means the cache is never updated — every PR sees the same old baseline.

**Why it happens:** GitHub Actions cache is immutable: once a key exists, it cannot be overwritten. Save with a unique key (e.g., append `${{ github.sha }}`); restore with the stable prefix via `restore-keys`.

**How to avoid:** Save baseline with key `sol-baseline-ref-v1-${{ github.sha }}`; restore with `restore-keys: sol-baseline-ref-v1-`. This ensures restore always gets the most recent matching entry.

**Warning signs:** PR comparison always shows the same baseline samples; `gh run list` shows the same cached entry age for all PRs.

### Pitfall 3: Graceful-Skip Logic Not Triggering

**What goes wrong:** The cache restore step exits non-zero when the cache is missing, aborting the workflow before the graceful-skip check runs.

**Why it happens:** `actions/cache/restore@v4` exits 0 even on cache miss; but if the step is configured with `fail-on-cache-miss: true`, it exits non-zero.

**How to avoid:** Do not set `fail-on-cache-miss: true` on the restore step. Use the step output `steps.baseline-cache.outputs.cache-hit` to branch logic.

**Warning signs:** Workflow fails on first-ever run with "cache not found" rather than producing a "no baseline" summary.

### Pitfall 4: Posting PR Comment Without write Permission

**What goes wrong:** `gh pr comment` fails with 403 because the workflow lacks `pull-requests: write`.

**Why it happens:** GitHub Actions defaults restrict write permissions.

**How to avoid:** Add `permissions: pull-requests: write` to the PR workflow job (or top-level). Do NOT add `contents: write` unless needed — least-privilege.

**Warning signs:** `gh pr comment` exits non-zero with a permissions error in CI logs.

### Pitfall 5: `combined_summary.tsv` Not in Expected Location

**What goes wrong:** PR workflow looks for `bench-artifacts/sol-baseline/latest/data/combined_summary.tsv` but the artifact was uploaded with a different internal structure.

**Why it happens:** Phase 1 uploads `bench-artifacts/sol-baseline/latest/` as the artifact root, but `actions/download-artifact` restores to a named directory, shifting the path.

**How to avoid:** In the PR workflow, either (a) download the candidate artifact and check the exact restored path, or (b) require the candidate run to set a known output path as an env variable passed via workflow outputs. Document the expected path contract explicitly.

**Warning signs:** `No such file` errors when the comparison script tries to open the TSV.

### Pitfall 6: Effect Size Direction for "Higher is Better" vs "Lower is Better" Metrics

**What goes wrong:** A +9% delta in `peak_rss_mib` (memory — lower is better) is a regression, but a +9% delta in `throughput_blocks_s` (throughput — higher is better) is an improvement. If delta sign is used naively, direction is inverted.

**Why it happens:** Effect size as `(candidate - baseline) / baseline * 100` is directionless without metric semantics.

**How to avoid:** Add a `metric_direction` property to the comparison config (or hardcode in the `CanonicalMetric` enum): `Higher` for throughput, `Lower` for timing and memory. Use direction when mapping delta sign to verdict.

**Warning signs:** Memory increase reported as "improvement"; throughput decrease reported as "regression."

## Code Examples

Verified patterns from existing source:

### Existing Bootstrap CI (guard/stats.rs)

```rust
// Source: crates/nockchain-bench/src/speed_of_light/guard/stats.rs
// Already implemented and tested — use directly in compare.rs
pub fn bootstrap_median_ci(
    values: &[f64],
    iterations: usize,  // use 500 for balance of speed/accuracy
    alpha: f64,          // 0.10 for 90% confidence
    seed: u64,           // deterministic: use 42
) -> Option<ConfidenceInterval>
```

### Existing Markdown Report Writer (guard/report.rs)

```rust
// Source: crates/nockchain-bench/src/speed_of_light/guard/report.rs
// Extend render_markdown() or add a new render_comparison_markdown()
pub fn write_markdown(path: &Path, report: &GuardReport) -> Result<(), ReportError>
pub fn render_markdown(report: &GuardReport) -> String
```

### Existing CLI Guard Subcommand Pattern (main.rs)

```rust
// Source: crates/nockchain-bench/src/main.rs (lines 447–492)
// Pattern for adding `sol compare` subcommand — mirror Guard structure
SolCommands::Guard {
    candidate_summary,
    baseline_summary,
    contract,
    env, branch, fixture, pass, run_id,
    output_json, output_md, strict,
} => { ... }
```

### Existing Exit Codes (guard/mod.rs)

```rust
// Source: crates/nockchain-bench/src/speed_of_light/guard/mod.rs
pub const EXIT_PASS: i32 = 0;
pub const EXIT_REGRESSION: i32 = 2;
pub const EXIT_INSUFFICIENT_BASELINE: i32 = 3;
pub const EXIT_CONFIG_ERROR: i32 = 4;
// Phase 2 adds: EXIT_IMPROVEMENT = 1 (advisory, or reuse EXIT_PASS)
// But per decisions: PR gate is warning-only — always exit 0 for CI pass
```

### CI Comment Pattern (GitHub Actions YAML)

```yaml
# Source: GitHub Actions documentation + gh CLI pattern
# Pre-installed on ubuntu-latest runners
- name: Post PR comment
  if: github.event_name == 'pull_request'
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    gh pr comment "${{ github.event.pull_request.number }}" \
      --body-file comparison/summary.md \
      --repo "${{ github.repository }}"
```

### Four-Way Verdict Mapping (New compare.rs)

```rust
// Source: internal design — extend from bootstrap_median_ci() semantics
fn classify_verdict(
    candidate_ci: ConfidenceInterval,
    baseline_ci: ConfidenceInterval,
    baseline_median: f64,
    delta_pct: f64,
    direction: MetricDirection,  // Higher or Lower (is better)
    min_samples: usize,
    actual_samples: usize,
) -> ComparisonVerdict {
    if actual_samples < min_samples {
        return ComparisonVerdict::Inconclusive;
    }
    let overlap = candidate_ci.low <= baseline_ci.high && baseline_ci.low <= candidate_ci.high;
    if overlap {
        return ComparisonVerdict::NoSignificantChange;
    }
    // No overlap — check direction
    let candidate_above = candidate_ci.low > baseline_ci.high;
    match (direction, candidate_above) {
        (MetricDirection::Higher, true) | (MetricDirection::Lower, false) => ComparisonVerdict::Improvement,
        _ => ComparisonVerdict::Regression,
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|-----------------|--------------|--------|
| Hardcoded floor/ceiling thresholds (GuardContract) | Bootstrap CI overlap for statistical significance | Phase 2 | More robust to noisy benchmarks; avoids false positives from fixed percentages |
| Single `gh pr comment --create` | `peter-evans/create-or-update-comment` or `gh pr comment` | 2023+ | Modern GitHub Actions avoids duplicate comments on re-runs |
| Stale GitHub Actions cache (immutable key) | Save with unique key, restore with prefix | 2022+ | Correct cache invalidation pattern |

**Deprecated/outdated:**
- `set-output` syntax in GitHub Actions: Use `$GITHUB_OUTPUT` environment file instead (deprecated 2022, removed 2023)
- `actions/cache@v2`: Use `v4` with separate `cache/restore` and `cache/save` actions for fine-grained control

## Open Questions

1. **Per-benchmark threshold overrides in config**
   - What we know: CONTEXT.md says "global defaults with optional per-benchmark config for noisy vs stable benchmarks"
   - What's unclear: Whether this means a new TOML section in `sol-baseline.toml`, a separate `contract.toml`, or CLI flags
   - Recommendation: Reuse the existing `GuardContract` TOML format (`[rules.metric_name]`) with an added `significance_threshold` field per rule. This is additive and backward compatible.

2. **Candidate artifact source in PR workflow**
   - What we know: "PR workflow compares pre-existing artifacts only — does NOT run benchmarks itself"
   - What's unclear: How the candidate `combined_summary.tsv` gets into the PR workflow. Options: (a) uploaded in a prior job of the same workflow run, (b) triggered after a separate manual baseline dispatch, (c) the PR workflow downloads the most recent baseline artifact by commit SHA
   - Recommendation: Most practical for Phase 2 is to accept the candidate TSV path as a workflow input, with a note in docs that the maintainer must first run the baseline workflow for the PR branch. This avoids complex job chaining while delivering the core feature. Phase 3 can add scheduling.

3. **GitHub Actions cache TTL and Phase 3 interaction**
   - What we know: GitHub Actions cache entries expire after 7 days without use and have a 10 GB per-repo limit
   - What's unclear: Whether the 7-day TTL is acceptable for the Phase 2 baseline reference or if Phase 3's scheduled refresh is needed to keep it alive
   - Recommendation: The PR workflow restoring the cache counts as "access" and resets the TTL. The scheduled baseline run in Phase 3 will also save a new cache entry. For Phase 2, the 7-day TTL is acceptable — document the limitation.

## Sources

### Primary (HIGH confidence)

- `/shared/nockchain/crates/nockchain-bench/src/speed_of_light/guard/stats.rs` — bootstrap CI, median, MAD implementations (verified by reading source)
- `/shared/nockchain/crates/nockchain-bench/src/speed_of_light/guard/contract.rs` — `evaluate_contract()`, contract evaluation logic (verified by reading source)
- `/shared/nockchain/crates/nockchain-bench/src/speed_of_light/guard/model.rs` — `GuardVerdict`, `CanonicalMetric`, `GuardReport` types (verified by reading source)
- `/shared/nockchain/crates/nockchain-bench/src/speed_of_light/guard/report.rs` — `render_markdown()`, `write_json()` (verified by reading source)
- `/shared/nockchain/crates/nockchain-bench/src/speed_of_light/guard/mod.rs` — exit code constants, module exports (verified by reading source)
- `/shared/nockchain/crates/nockchain-bench/src/main.rs` — `Guard` subcommand CLI definition (verified by reading source)
- `/shared/nockchain/scripts/sol_guard_ci.sh` — existing guard CI wrapper pattern (verified by reading source)
- `/shared/nockchain/.github/workflows/sol-baseline.yml` — existing workflow structure and conventions (verified by reading source)
- `/shared/nockchain/crates/nockchain-bench/tests/fixtures/guard/combined_summary.tsv` — Phase 1 TSV schema (verified by reading source)

### Secondary (MEDIUM confidence)

- GitHub Actions `actions/cache@v4` save/restore pattern — cache key immutability and `restore-keys` fallback (well-documented; standard pattern)
- `gh pr comment` authentication via `GITHUB_TOKEN` on ubuntu-latest — standard GitHub-hosted runner behavior
- `$GITHUB_OUTPUT` env file for step outputs — replacement for deprecated `set-output` (documented in GitHub Actions migration guide)

### Tertiary (LOW confidence)

- 7-day GitHub Actions cache TTL — from GitHub documentation; may vary by plan/repo configuration

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies already in Cargo.toml; no new libraries needed
- Architecture: HIGH — existing guard module provides the foundation; patterns verified by reading source
- Pitfalls: HIGH — cache key strategy and permission requirements verified against GitHub Actions documentation patterns; direction-aware metrics are an observable gap in the existing `CanonicalMetric` enum

**Research date:** 2026-02-24
**Valid until:** 2026-03-24 (30 days — GitHub Actions API is stable; Rust crate versions pinned)
