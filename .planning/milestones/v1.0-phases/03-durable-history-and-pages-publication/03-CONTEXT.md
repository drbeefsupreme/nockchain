# Phase 3: Durable History and Pages Publication - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Automate baseline refresh, preserve immutable run history, and publish artifacts to GitHub Pages. Covers: history storage and append logic, active baseline advancement workflow, GitHub Pages dashboard publication, and merge/manual-triggered baseline generation. Does NOT cover: new comparison logic (Phase 2), baseline config or provenance schema changes (Phase 1), or new benchmark additions.

</domain>

<decisions>
## Implementation Decisions

### History storage model
- Each run stored as a separate JSON file in a history directory (e.g., `history/2026-02-25_abc1234.json`)
- History lives on a separate git branch (e.g., `gh-pages` or `bench-history`), not in main
- All runs coexist in a flat directory — no archiving or subdirectory separation
- Active baseline tracked via a manifest file (`baseline-active.json`) that references the current run by name/SHA
- Runs are never modified or deleted — immutability by convention

### Baseline advancement
- Manual workflow dispatch only — maintainer explicitly triggers promotion of a specific run
- Validate provenance metadata (SHA, config, environment) before promoting — lightweight sanity check, no comparison
- Git history of manifest changes serves as the audit trail — no in-file promotion history
- Advancement workflow triggers a Pages rebuild immediately so the site reflects the new baseline

### Pages publication
- Dashboard with trend charts and per-benchmark drill-down
- Client-side rendering: publish JSON data files + a single index.html that fetches and renders in-browser
- Lightweight charting library loaded from CDN (e.g., Chart.js or uPlot)
- Overview page shows all benchmarks with trend lines; click into individual benchmarks for detailed history

### Scheduled generation
- Triggered on merge to master and via manual workflow dispatch — no cron schedule
- Runs auto-append to history on the data branch — no manual accept step required
- All runs treated identically in the data regardless of trigger source (merge vs manual)

### Claude's Discretion
- Specific charting library choice (Chart.js, uPlot, or similar lightweight option)
- Branch naming for history/pages data
- HTML/CSS design of the dashboard
- File naming convention for individual run files
- How the index.html discovers and loads history data

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-durable-history-and-pages-publication*
*Context gathered: 2026-02-25*
