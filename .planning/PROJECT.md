# Nockchain Benchmark Baseline Framework

## What This Is

A reproducible benchmark baseline system for `nockchain-bench` that collects statistically valid performance data, compares candidates against baselines with defensible verdicts, and publishes immutable history to GitHub Pages. Built for maintainers working on `nockchain` and `nockvm` who need trustworthy regression/improvement signals across changes.

## Core Value

Every benchmark comparison uses a reproducible, statistically valid baseline so performance changes can be interpreted with confidence.

## Requirements

### Validated

- ✓ Scripted baseline workflow with TOML profile config, local + CI parity — v1.0
- ✓ Canonical machine-readable artifacts with full provenance (commit, env, config) — v1.0
- ✓ Statistical comparison with four-way verdict (improvement/regression/no-change/inconclusive) — v1.0
- ✓ PR-time regression checks with advisory markdown reports — v1.0
- ✓ Immutable run history on gh-pages with append-only writes — v1.0
- ✓ GitHub Pages dashboard with trend charts and auto-deploy — v1.0

### Active

(None — next milestone requirements TBD via `/gsd:new-milestone`)

### Out of Scope

- Replacing the existing statistical testing engine in `nockchain-bench` — this effort seeds and feeds it
- Building new performance optimizations in `nockchain`/`nockvm` — this focuses on measurement infrastructure
- Offline mode or local-only dashboard — GitHub Pages is the publication target

## Context

Shipped v1.0 with ~3,800 LOC across Bash scripts, GitHub Actions YAML, Rust (comparison engine), and HTML (dashboard).

Tech stack: Rust (nockchain-bench CLI extensions), Bash (orchestration scripts), GitHub Actions (CI workflows), Chart.js (dashboard), jq/awk (data processing).

Key architecture: TOML config → Bash runner → provenance manifest → TSV artifacts → comparison engine → PR reports. Parallel path: history append → gh-pages branch → Pages deploy.

## Constraints

- **Compatibility**: Must integrate with existing `nockchain-bench` workflows and crate structure
- **Reproducibility**: Baseline generation must be deterministic enough for statistical comparison
- **Automation**: GitHub Pages updates run unattended via workflow_call triggers
- **Traceability**: Each baseline run attributable to code/config/environment context

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| TOML config with profile overlays | Supports quick (CI) and full (baseline) modes from one config | ✓ Good |
| Provenance in Bash via jq | Avoids Rust compilation for metadata collection | ✓ Good |
| Strict manifest validation (all 9 fields required) | No incomplete artifacts on disk | ✓ Good |
| Bootstrap CI overlap for comparison | More robust than t-test for small sample sizes | ✓ Good |
| Advisory-only PR gates (exit 0 always) | Prevents blocking PRs on noisy benchmarks | ✓ Good |
| Cache key with versioned prefix + SHA suffix | Proper invalidation without stale baseline data | ✓ Good |
| peaceiris/actions-gh-pages with keep_files: true | Append-only history without deleting prior runs | ✓ Good |
| workflow_call for Pages deploy reuse | Single deploy workflow called from baseline + advancement | ✓ Good |
| Chart.js v4.4.1 pinned from CDN | Better docs than uPlot, adequate for small datasets | ✓ Good |
| Concurrency group serialization | Prevents index.json corruption from parallel writes | ✓ Good |

---
*Last updated: 2026-02-26 after v1.0 milestone*
