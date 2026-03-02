# Phase 3: Durable History and Pages Publication - Research

**Researched:** 2026-02-26
**Domain:** GitHub Pages CI deployment, immutable run history management, client-side charting dashboard (Bash/YAML/HTML)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**History storage model:**
- Each run stored as a separate JSON file in a history directory (e.g., `history/2026-02-25_abc1234.json`)
- History lives on a separate git branch (e.g., `gh-pages` or `bench-history`), not in master
- All runs coexist in a flat directory — no archiving or subdirectory separation
- Active baseline tracked via a manifest file (`baseline-active.json`) that references the current run by name/SHA
- Runs are never modified or deleted — immutability by convention

**Baseline advancement:**
- Manual workflow dispatch only — maintainer explicitly triggers promotion of a specific run
- Validate provenance metadata (SHA, config, environment) before promoting — lightweight sanity check, no comparison
- Git history of manifest changes serves as the audit trail — no in-file promotion history
- Advancement workflow triggers a Pages rebuild immediately so the site reflects the new baseline

**Pages publication:**
- Dashboard with trend charts and per-benchmark drill-down
- Client-side rendering: publish JSON data files + a single index.html that fetches and renders in-browser
- Lightweight charting library loaded from CDN (e.g., Chart.js or uPlot)
- Overview page shows all benchmarks with trend lines; click into individual benchmarks for detailed history

**Scheduled generation:**
- Triggered on merge to master and via manual workflow dispatch — no cron schedule
- Runs auto-append to history on the data branch — no manual accept step required
- All runs treated identically in the data regardless of trigger source (merge vs manual)

### Claude's Discretion
- Specific charting library choice (Chart.js, uPlot, or similar lightweight option)
- Branch naming for history/pages data
- HTML/CSS design of the dashboard
- File naming convention for individual run files
- How the index.html discovers and loads history data

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-03 | Maintainer can preserve immutable historical run records while tracking the active baseline reference. | History-branch pattern (flat dir of run JSON files + `baseline-active.json` manifest); append-only git commits via `peaceiris/actions-gh-pages@v4` with `keep_files: true`; manifest-only updates for baseline advancement |
| PIPE-01 | Maintainer can run scheduled baseline generation in CI that appends new baseline history without deleting prior runs. | `sol-baseline.yml` extended with history-append step; `peaceiris/actions-gh-pages@v4` `keep_files: true` pushes only new run file without touching existing history; triggered on push to master + `workflow_dispatch` |
| PIPE-03 | Maintainer can publish benchmark history and latest baseline artifacts to GitHub Pages through an automated workflow. | Three-action Pages pipeline (`configure-pages`, `upload-pages-artifact`, `deploy-pages`); OR history branch doubles as gh-pages source; Chart.js from CDN; client-side `fetch()` of JSON data files |
</phase_requirements>

## Summary

Phase 3 has two largely independent concerns: (1) durable history storage with an active-baseline manifest, and (2) a GitHub Pages dashboard that reads that history. Both are CI-orchestrated via GitHub Actions with no application code changes required — this phase is entirely Bash scripts, YAML workflows, and a single static HTML file.

The history storage model is already decided: a flat directory of per-run JSON files on a dedicated branch, with a `baseline-active.json` manifest pointing to the current run. The key implementation question is how to append to that branch in CI without destroying existing files. `peaceiris/actions-gh-pages@v4` with `keep_files: true` is the right tool — it pushes only the new file to the branch, leaving prior run files untouched. The git commit history of the branch serves as an immutable audit log by construction. Baseline advancement is a separate `workflow_dispatch`-only workflow that writes only the manifest file.

The Pages dashboard is a single `index.html` with a CDN-loaded Chart.js, a `fetch()` call to load `history/index.json` (a manifest listing all run file names), and client-side rendering of trend lines. Chart.js v4 is the right choice over uPlot: better documentation, CDN availability, line/time-series support, and adequate performance for the small datasets involved (tens of runs, ~40 columns per run). The official GitHub Pages deployment pipeline (`configure-pages` + `upload-pages-artifact` + `deploy-pages`) is the current standard and should be used over direct `gh-pages` branch pushes for the site content — though the data branch can be the same branch if configured as the Pages source.

**Primary recommendation:** Use `peaceiris/actions-gh-pages@v4` with `keep_files: true` to append run JSON files to the history branch. Use the official `actions/deploy-pages@v4` pipeline for the static site (or configure the history branch as the Pages source directly). Use Chart.js v4 loaded from jsdelivr CDN.

## Standard Stack

### Core

| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| `peaceiris/actions-gh-pages` | v4 | Push new run JSON files to history/data branch without deleting existing files | `keep_files: true` is the documented solution for append-only branch updates; maintained and widely used |
| `actions/configure-pages` | v5 | Configure the GitHub Pages deployment environment | Required first step in the official Pages Actions pipeline |
| `actions/upload-pages-artifact` | v3 | Package static site dir as Pages deployment artifact | Required second step; v3 required (v2 deprecated Dec 2024) |
| `actions/deploy-pages` | v4 | Deploy the packaged artifact to GitHub Pages | Required third step; outputs `page_url`; current version |
| Chart.js | 4.4.x | Client-side trend charts in the dashboard HTML | Better docs than uPlot, adequate performance for small datasets (~50 runs), CDN-available, extensive line chart support |
| Bash + jq | system | Append-logic scripts, manifest generation | Existing project pattern; `jq` already used in Phase 1 |

### Supporting

| Tool | Version | Purpose | When to Use |
|------|---------|---------|-------------|
| `gh` CLI | bundled | Trigger advancement workflow, post summaries | Pre-installed on ubuntu-latest; already used in Phase 2 for PR comments |
| `actions/checkout@v4` | v4 | Checkout with full history for branch operations | Required for all workflows in this project |
| jsDelivr CDN | — | Serve Chart.js from CDN without bundling | No build toolchain needed; chart.js at `https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `peaceiris/actions-gh-pages@v4` | Manual `git worktree` + `git push` in Bash | Manual approach works but requires handling SSH keys or token auth and worktree setup; the action abstracts this correctly |
| `peaceiris/actions-gh-pages@v4` | `JamesIves/github-pages-deploy-action@v4` | JamesIves supports `clean: false` too; `peaceiris` is simpler and more widely referenced for this pattern; either works |
| Chart.js | uPlot | uPlot is faster and smaller (47KB vs 254KB) but has minimal docs and a harder API; not worthwhile for this use case with small datasets |
| Official Pages pipeline | Push HTML directly to gh-pages branch | Direct branch push is simpler if history branch IS the Pages source; but official pipeline integrates with Pages environment protections and deployment history in the GitHub UI |
| Single gh-pages branch | Separate `bench-history` data branch + `gh-pages` site branch | One branch doubles as both data store and site source — simpler; Pages can serve from the branch root |

**Installation:** No npm/cargo dependencies. All tools are GitHub Actions or CDN-loaded.

## Architecture Patterns

### Recommended Project Structure

New files added by Phase 3:

```
.github/workflows/
├── sol-baseline.yml          # EXISTING: add history-append step
├── sol-advance-baseline.yml  # NEW: workflow_dispatch to promote run
└── sol-pages-deploy.yml      # NEW: deploy Pages site (or merged into sol-baseline.yml)

scripts/
├── sol_history_append.sh     # NEW: writes run JSON to history branch
└── sol_pages_build.sh        # NEW: builds index.html from template + generates history/index.json

bench-artifacts/sol-baseline/latest/   # EXISTING: produced by Phase 1
  data/combined_summary.tsv
  meta/manifest.json
```

On the data/pages branch (not in master):
```
history/
├── index.json                          # Run index: [{id, timestamp, git_commit, ...}, ...]
├── 2026-02-25T12-00-00Z_abc1234.json   # Per-run data: provenance + metric summaries
├── 2026-02-26T08-00-00Z_def5678.json
└── ...
baseline-active.json                    # Active baseline reference: {run_id, timestamp, git_commit}
index.html                              # Dashboard: fetches history/index.json, renders charts
```

### Pattern 1: Append-Only History via `peaceiris/actions-gh-pages@v4` with `keep_files: true`

**What:** Each baseline run produces a JSON file containing provenance + metric medians. A CI step pushes only that new file to the history branch. `keep_files: true` prevents the action from deleting existing run files.

**When to use:** Every time a baseline run completes (merge to master or manual dispatch).

**Example:**
```yaml
# Source: peaceiris/actions-gh-pages documentation
- name: Append run to history branch
  uses: peaceiris/actions-gh-pages@v4
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./new-run-payload   # directory containing only the new run JSON
    destination_dir: history         # maps into history/ on the branch
    keep_files: true                 # DO NOT delete other run files
    commit_message: "history: add run ${{ github.sha }}"
```

**Key constraint:** `publish_dir` must contain ONLY the new run file plus an updated `history/index.json`. Do not include the entire history dir or other files.

### Pattern 2: history/index.json as Run Discovery Manifest

**What:** A JSON array that the dashboard's `fetch()` call uses to discover all run files. Updated on every new run append and on every baseline advancement.

**When to use:** Client-side JS reads this to build the run list without directory listing (GitHub Pages doesn't support directory listings).

**Schema:**
```json
[
  {
    "id": "2026-02-26T08-00-00Z_def5678",
    "filename": "2026-02-26T08-00-00Z_def5678.json",
    "timestamp": "2026-02-26T08:00:00Z",
    "git_commit": "def5678...",
    "git_branch": "master",
    "profile": "full",
    "is_active_baseline": false
  },
  ...
]
```

**How to update:** The `sol_history_append.sh` script:
1. Checks out the history branch
2. Writes the new run JSON file to `history/`
3. Reads existing `history/index.json` (or initializes empty array)
4. Appends the new entry
5. Writes updated `history/index.json`
6. Lets `peaceiris/actions-gh-pages@v4` push both files with `keep_files: true`

### Pattern 3: baseline-active.json Manifest

**What:** A single JSON file on the history branch that records which run is the current active baseline.

**When to use:** Written only by the `sol-advance-baseline.yml` workflow via `workflow_dispatch`.

**Schema:**
```json
{
  "run_id": "2026-02-25T12-00-00Z_abc1234",
  "promoted_at": "2026-02-25T14:30:00Z",
  "git_commit": "abc1234...",
  "promoted_by": "workflow_dispatch"
}
```

**Advancement workflow steps:**
1. Accept `run_id` as workflow input
2. Validate: confirm `history/{run_id}.json` exists on the history branch
3. Validate: confirm provenance fields (sha, config, environment) are populated
4. Write `baseline-active.json` with the new reference
5. Push via `peaceiris/actions-gh-pages@v4` with `keep_files: true`
6. Trigger Pages redeploy (by dispatching `sol-pages-deploy.yml` or combining into same job)

### Pattern 4: Official GitHub Pages Deployment Pipeline

**What:** Three-step pipeline using `configure-pages`, `upload-pages-artifact`, `deploy-pages` to publish the static dashboard.

**When to use:** Whenever history or baseline changes — triggered after history append and after baseline advancement.

**Required permissions:**
```yaml
permissions:
  pages: write
  id-token: write
  contents: read
```

**Minimal workflow:**
```yaml
# Source: https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: gh-pages       # checkout the history/pages branch
      - name: Build index.json (ensure current)
        run: bash scripts/sol_pages_build.sh
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: '.'           # entire branch root is the site
  deploy:
    needs: build
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/deploy-pages@v4
        id: deployment
```

### Pattern 5: Client-Side Dashboard (index.html)

**What:** A single static HTML file that fetches `history/index.json`, loads per-run files, and renders trend charts with Chart.js.

**When to use:** Served directly from GitHub Pages; no server-side rendering.

**Structure:**
```html
<!DOCTYPE html>
<html>
<head>
  <title>SOL Benchmark History</title>
</head>
<body>
  <canvas id="throughput-chart"></canvas>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <script>
    // Source: https://www.chartjs.org/docs/latest/getting-started/
    async function main() {
      const index = await fetch('history/index.json').then(r => r.json());
      // Load each run's data, build chart datasets
      const runs = await Promise.all(
        index.map(entry => fetch(`history/${entry.filename}`).then(r => r.json()))
      );
      new Chart(document.getElementById('throughput-chart'), {
        type: 'line',
        data: {
          labels: runs.map(r => r.timestamp.slice(0, 10)),
          datasets: [{
            label: 'throughput_blocks_s',
            data: runs.map(r => r.metrics.throughput_blocks_s_median),
            tension: 0.1
          }]
        }
      });
    }
    main();
  </script>
</body>
</html>
```

### Pattern 6: Per-Run JSON File Schema

**What:** Each baseline run is serialized as a compact JSON file combining provenance + metric medians from the TSV. This is the source of truth for the dashboard.

**How to generate:** `sol_history_append.sh` reads `meta/manifest.json` (provenance) and `data/combined_summary.tsv` (raw rows), computes per-metric medians, and writes the combined file.

**Schema:**
```json
{
  "run_id": "2026-02-26T08-00-00Z_def5678",
  "timestamp": "2026-02-26T08:00:00Z",
  "git_commit": "def5678...",
  "git_branch": "master",
  "profile": "full",
  "passes": 5,
  "metrics": {
    "throughput_blocks_s_median": 10.12,
    "init_time_s_median": 0.142,
    "avg_per_block_ms_median": 98.8,
    "peak_rss_mib_median": 812.4
  },
  "environment": {
    "cpu_model": "AMD EPYC 7763",
    "cpu_cores": 16,
    "ram_bytes": 34359738368
  }
}
```

**Key metrics to include:** `throughput_blocks_s`, `init_time_s`, `avg_per_block_ms`, `peak_rss_mib` — the columns present in `combined_summary.tsv` that have benchmark signal.

### Anti-Patterns to Avoid

- **`keep_files: false` (default):** Never omit `keep_files: true` — default behavior deletes all existing history files on every push, destroying immutability.
- **Committing history to master:** History branch must remain separate; run JSON files in master would cause repo bloat and pollute commit history.
- **Directory listing for run discovery:** GitHub Pages does not serve directory listings. Always use `history/index.json` as the explicit manifest.
- **Appending to history/index.json without reading existing:** Always read the current index before appending. Writing a new array containing only the latest run would lose all prior entries.
- **Using `actions/cache` for history storage:** Cache has 7-day TTL and 10GB limit; not suitable for immutable permanent records. Use a git branch.
- **OIDC/id-token permission missing from deploy job:** The `deploy-pages` action requires `id-token: write` in addition to `pages: write`. Missing this causes cryptic auth failures.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Push to branch without deleting existing files | Custom git push script with merge/cherry-pick | `peaceiris/actions-gh-pages@v4` with `keep_files: true` | Handles auth, branch init, and merge conflict avoidance; keep_files is the documented idiom |
| Pages deployment | Direct `git push` to gh-pages + Pages auto-build | `configure-pages` + `upload-pages-artifact` + `deploy-pages` | Official pipeline integrates with GitHub Pages environment, deployment history, and branch protection |
| Chart rendering | Custom SVG/Canvas drawing | Chart.js v4 from CDN | Handles axes, tooltips, responsiveness, time-series labeling; well-documented |
| JSON median computation from TSV | Rust binary | `awk` + `jq` in `sol_history_append.sh` | The TSV already has per-pass rows; median is a simple sort/select in awk; no need for a compiled tool |

**Key insight:** Phase 3 is entirely orchestration and data transformation. No Rust code changes are needed. All logic is Bash + YAML + HTML.

## Common Pitfalls

### Pitfall 1: `keep_files: true` Silently Does Nothing If `publish_dir` Contains Stale Files

**What goes wrong:** If `publish_dir` accidentally contains the entire checked-out history directory (not just the new file), `keep_files: true` still merges it — but any stale or accidentally-modified files in `publish_dir` overwrite the canonical history on the branch.

**Why it happens:** The script checks out the history branch to read `index.json`, then builds `publish_dir` from that checkout, inadvertently including all existing run files in the publish payload.

**How to avoid:** Build `publish_dir` as a fresh empty directory. Copy only the new run JSON file and the updated `index.json` into it. Never copy existing run files into it.

**Warning signs:** The history branch commit shows hundreds of files changed rather than 2 (new run + updated index).

### Pitfall 2: history/index.json Update Race on Concurrent Runs

**What goes wrong:** Two baseline runs finish simultaneously. Both read the current `index.json`, both append their entry, and one overwrites the other. The index loses one entry.

**Why it happens:** Read-modify-write on a file in a branch without locking.

**How to avoid:** GitHub Actions branch-push serialization: `peaceiris/actions-gh-pages@v4` does a git push; the second concurrent push will fail with a non-fast-forward error. Add retry logic or use GitHub's serialization (one job at a time via `concurrency` group with `cancel-in-progress: false`). The concurrency group approach is simpler.

```yaml
concurrency:
  group: bench-history-write
  cancel-in-progress: false  # queue, don't cancel
```

**Warning signs:** `index.json` has gaps (run files exist on branch but not listed in index).

### Pitfall 3: Pages Source Misconfiguration

**What goes wrong:** The Pages site shows 404 because the repo has Pages disabled, configured for a different branch, or configured for "Deploy from branch" instead of "GitHub Actions."

**Why it happens:** Pages configuration must be manually set in the repository Settings UI before the first workflow run. The workflow itself cannot set this.

**How to avoid:** Document as a one-time setup step: Settings → Pages → Source → "GitHub Actions". The `actions/configure-pages` step will warn if this is not set.

**Warning signs:** `deploy-pages` step succeeds but the site URL returns 404.

### Pitfall 4: Missing `id-token: write` Permission on Deploy Job

**What goes wrong:** `actions/deploy-pages@v4` fails with a permissions/OIDC error.

**Why it happens:** The deploy job needs `id-token: write` in addition to `pages: write`. If permissions are declared at workflow level without the deploy job inheriting them, the job uses defaults.

**How to avoid:** Always declare permissions at the job level for the deploy job:
```yaml
jobs:
  deploy:
    permissions:
      pages: write
      id-token: write
```

**Warning signs:** Error message containing "OIDC" or "id-token" in the deploy-pages step output.

### Pitfall 5: Baseline Advancement Workflow Overwrites History Files

**What goes wrong:** The advancement workflow uses `peaceiris/actions-gh-pages@v4` without `keep_files: true` and only publishes `baseline-active.json` — the action wipes all history run files from the branch.

**Why it happens:** Default behavior of the action is to replace all files in `destination_dir` (or the branch root if no destination_dir).

**How to avoid:** Always set `keep_files: true` on every push to the history branch. This applies to both the history-append workflow and the advancement workflow.

**Warning signs:** History branch has only `baseline-active.json` after advancement.

### Pitfall 6: Dashboard fetch() Fails on CORS / Local File

**What goes wrong:** Opening `index.html` directly from the filesystem (file://) causes `fetch()` to fail with a CORS error.

**Why it happens:** Browser security prevents `fetch()` from reading local files via `file://` protocol.

**How to avoid:** This is expected behavior. Dashboard only works when served from GitHub Pages (https://). Document that local testing requires `python3 -m http.server` or similar. Do not attempt to work around it in the HTML.

**Warning signs:** Developers report blank dashboard when testing locally; works fine on the live Pages URL.

### Pitfall 7: Chart.js CDN @latest Tag Changes

**What goes wrong:** Using `chart.js@latest` in the CDN URL picks up a breaking major version when Chart.js releases v5.

**Why it happens:** `@latest` resolves to the newest version at load time.

**How to avoid:** Pin to a specific minor version: `chart.js@4.4.1/dist/chart.umd.min.js`. Update intentionally.

**Warning signs:** Dashboard breaks after a Chart.js major release without any local changes.

## Code Examples

Verified patterns from official sources:

### sol_history_append.sh - Core Logic

```bash
#!/usr/bin/env bash
# Source: project pattern; uses jq (already in Phase 1 toolchain)
set -euo pipefail

MANIFEST_JSON="$1"     # meta/manifest.json from the baseline run
SUMMARY_TSV="$2"       # data/combined_summary.tsv from the baseline run
PUBLISH_DIR="$3"       # empty directory to write new files into

# Compute run ID from manifest
TIMESTAMP=$(jq -r '.timestamp' "$MANIFEST_JSON" | tr ':' '-')
COMMIT_SHORT=$(jq -r '.git_commit' "$MANIFEST_JSON" | cut -c1-7)
RUN_ID="${TIMESTAMP}_${COMMIT_SHORT}"

# Compute metric medians from TSV using awk
# (sort numeric column, take middle value)
THROUGHPUT_MEDIAN=$(awk 'NR>1{print $10}' "$SUMMARY_TSV" | sort -n | awk 'NR==int((NF+1)/2)')

# Build per-run JSON
jq -n \
  --arg run_id "$RUN_ID" \
  --arg timestamp "$(jq -r '.timestamp' "$MANIFEST_JSON")" \
  --arg git_commit "$(jq -r '.git_commit' "$MANIFEST_JSON")" \
  --arg git_branch "$(jq -r '.git_branch' "$MANIFEST_JSON")" \
  --argjson throughput "${THROUGHPUT_MEDIAN:-null}" \
  '{run_id: $run_id, timestamp: $timestamp, git_commit: $git_commit,
    git_branch: $git_branch, metrics: {throughput_blocks_s_median: $throughput}}' \
  > "$PUBLISH_DIR/${RUN_ID}.json"

# Update index.json (read from history branch checkout, then append)
EXISTING_INDEX="${4:-[]}"   # pass contents of existing index.json or default to []
echo "$EXISTING_INDEX" | jq \
  --arg run_id "$RUN_ID" \
  --arg filename "${RUN_ID}.json" \
  --arg timestamp "$(jq -r '.timestamp' "$MANIFEST_JSON")" \
  --arg git_commit "$(jq -r '.git_commit' "$MANIFEST_JSON")" \
  '. + [{run_id: $run_id, filename: $filename, timestamp: $timestamp, git_commit: $git_commit}]' \
  > "$PUBLISH_DIR/index.json"
```

### sol-baseline.yml — History Append Step Addition

```yaml
# Source: peaceiris/actions-gh-pages documentation
# Add to existing sol-baseline.yml after "Upload artifacts" step

- name: Build history payload
  run: |
    mkdir -p history-payload
    bash scripts/sol_history_append.sh \
      bench-artifacts/sol-baseline/latest/meta/manifest.json \
      bench-artifacts/sol-baseline/latest/data/combined_summary.tsv \
      history-payload \
      "$(git fetch origin gh-pages 2>/dev/null; \
         git show origin/gh-pages:history/index.json 2>/dev/null || echo '[]')"

- name: Push to history branch
  uses: peaceiris/actions-gh-pages@v4
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./history-payload
    publish_branch: gh-pages
    destination_dir: history
    keep_files: true
    commit_message: "history: run ${{ github.sha }} (${{ github.run_number }})"
```

### sol-advance-baseline.yml — Advancement Workflow

```yaml
# Source: GitHub Actions documentation + project pattern
name: SOL Advance Baseline

on:
  workflow_dispatch:
    inputs:
      run_id:
        description: 'Run ID to promote (e.g. 2026-02-25T12-00-00Z_abc1234)'
        required: true
        type: string

permissions:
  contents: write
  pages: write
  id-token: write

jobs:
  advance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: gh-pages
          fetch-depth: 1

      - name: Validate run exists
        run: |
          RUN_FILE="history/${{ inputs.run_id }}.json"
          if [ ! -f "$RUN_FILE" ]; then
            echo "ERROR: Run file not found: $RUN_FILE" >&2
            exit 1
          fi
          # Validate provenance fields
          jq -e '.git_commit and .timestamp and .metrics' "$RUN_FILE" > /dev/null || {
            echo "ERROR: Run file missing required provenance fields" >&2; exit 1
          }

      - name: Write baseline-active.json
        run: |
          jq -n \
            --arg run_id "${{ inputs.run_id }}" \
            --arg promoted_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            '{run_id: $run_id, promoted_at: $promoted_at, promoted_by: "workflow_dispatch"}' \
            > baseline-active.json

      - name: Push updated manifest
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: .
          include_files: baseline-active.json   # only push this file
          publish_branch: gh-pages
          keep_files: true
          commit_message: "baseline: advance to ${{ inputs.run_id }}"
```

### Chart.js Line Chart (Dashboard index.html Snippet)

```html
<!-- Source: https://www.chartjs.org/docs/latest/getting-started/ -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<canvas id="throughput-chart"></canvas>
<script>
async function buildDashboard() {
  const index = await fetch('history/index.json').then(r => r.json());
  // Sort by timestamp ascending
  index.sort((a, b) => a.timestamp.localeCompare(b.timestamp));

  const labels = index.map(e => e.timestamp.slice(0, 10));
  const runData = await Promise.all(
    index.map(e => fetch('history/' + e.filename).then(r => r.json()))
  );

  new Chart(document.getElementById('throughput-chart'), {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'throughput_blocks_s (median)',
        data: runData.map(r => r.metrics.throughput_blocks_s_median),
        tension: 0.1,
        fill: false
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'top' } },
      scales: { y: { beginAtZero: false } }
    }
  });
}
buildDashboard();
</script>
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|-----------------|--------------|--------|
| `actions/cache` for persistent data | Git branch (gh-pages) for immutable records | 2022+ | Cache expires after 7 days; git branch is permanent |
| `actions/upload-pages-artifact@v2/v3` (old) | Must use v3+ (v2 deprecated Dec 2024) | Dec 2024 | Use `actions/upload-pages-artifact@v3` — v2 will stop working |
| `set-output` workflow command | `$GITHUB_OUTPUT` env file | 2022, enforced 2023 | Use `echo "name=value" >> $GITHUB_OUTPUT` |
| `peaceiris/actions-gh-pages@v3` | `peaceiris/actions-gh-pages@v4` | 2023 | v4 is current; v3 still works but use v4 |
| `actions/configure-pages@v3` | `actions/configure-pages@v5` | 2024 | v5 is current |

**Deprecated/outdated:**
- `actions/upload-pages-artifact@v2`: Deprecated December 2024, use v3.
- `gh-pages` npm package for deployment: Not applicable here; action-based approach is correct for CI.
- `chart.js@latest` CDN tag: Pin to `@4.4.1` to avoid breaking changes from future major release.

## Open Questions

1. **include_files option in peaceiris/actions-gh-pages**
   - What we know: The action has a `keep_files` option; the `include_files` option behavior needs verification
   - What's unclear: Whether `include_files` can restrict a push to a single file (e.g., only `baseline-active.json`) without requiring a full `publish_dir` containing only that file
   - Recommendation: Use a `publish_dir` containing only the files you want to push. Safer and well-documented. Create a temp directory with only the target file.

2. **History branch naming: `gh-pages` vs `bench-history`**
   - What we know: Using `gh-pages` as both the data branch and Pages source is simpler; a separate `bench-history` branch requires an extra step to copy data files to `gh-pages` for serving
   - What's unclear: Whether the team wants the Pages site and data files intermingled on one branch
   - Recommendation: Use a single `gh-pages` branch for both data and site. The dashboard `index.html` and `history/` directory live together. This eliminates any data-to-site sync step.

3. **TSV column positions for median computation**
   - What we know: The TSV has column headers in row 1; key columns include `throughput_blocks_s` (col 10), `init_time_s` (col 7), `avg_per_block_ms` (col 9), `peak_rss_mib` (col 15)
   - What's unclear: Whether column positions are stable across TSV schema versions
   - Recommendation: Use `awk -F'\t' -v col="$(head -1 tsv | tr '\t' '\n' | grep -n throughput | cut -d: -f1)" ...` to locate columns by name rather than hardcoded position. Safer against future schema changes.

## Sources

### Primary (HIGH confidence)
- `https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages` — required permissions, configure-pages + upload-pages-artifact + deploy-pages pipeline (verified via WebFetch)
- `https://github.com/actions/deploy-pages` — version v4.0.5, minimal workflow example (verified via WebFetch)
- `https://github.com/peaceiris/actions-gh-pages` — v4, `keep_files: true` option, `destination_dir` option (verified via WebFetch)
- `https://www.chartjs.org/docs/latest/getting-started/` — Chart.js CDN URL, minimal chart example (verified via WebFetch)
- `/home/drbeefsupreme/git/nockchain/crates/nockchain-bench/tests/fixtures/guard/combined_summary.tsv` — TSV column schema (verified by reading source)
- `/home/drbeefsupreme/git/nockchain/crates/nockchain-bench/src/speed_of_light/guard/provenance.rs` — RunProvenance JSON schema (verified by reading source)
- `/home/drbeefsupreme/git/nockchain/.github/workflows/sol-baseline.yml` — existing workflow structure for extension (verified by reading source)

### Secondary (MEDIUM confidence)
- uPlot GitHub README — size comparison: uPlot 47.9KB vs Chart.js 254KB; Chart.js better documented (WebSearch verified against multiple sources)
- `peaceiris/actions-gh-pages` discussion threads — `keep_files: true` behavior and `destination_dir` interaction (WebSearch; consistent with official docs)
- GitHub Changelog Dec 2024 — deprecation of `upload-pages-artifact@v2` (WebSearch, one source)

### Tertiary (LOW confidence)
- `chart.js@4.4.1` being the latest 4.x version — CDN URL pattern is verified; specific patch version may have incremented (use `@4` tag in CDN if exact pin not critical)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all tools verified via official docs/source; existing project patterns confirmed
- Architecture: HIGH — patterns derived from official action docs and existing Phase 1/2 workflow conventions
- Pitfalls: HIGH — most pitfalls derived from documented action behaviors and GitHub Pages requirements; race condition analysis is architectural reasoning

**Research date:** 2026-02-26
**Valid until:** 2026-03-26 (30 days — GitHub Actions API and Pages pipeline are stable; Chart.js v4 is stable)
