# Nockchain Benchmark Baseline Framework

## What This Is

This project adds a repeatable script framework to establish and maintain baseline statistical benchmark data for `nockchain-bench`. It is for maintainers working on `nockchain` and `nockvm` who need trustworthy regression/improvement signals across changes. It also includes publication updates so historical benchmark runs and metadata remain available via GitHub Pages.

## Core Value

Every benchmark comparison uses a reproducible, statistically valid baseline so performance changes can be interpreted with confidence.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Build a script builder framework that can define and run baseline-generation benchmark workflows for `nockchain-bench`.
- [ ] Persist baseline outputs in a structured format suitable for longitudinal statistical comparison across runs.
- [ ] Automate GitHub Pages updates so prior runs and current baseline datasets are published and retained.
- [ ] Support comparison-ready metadata (commit, environment, benchmark config) so changes in `nockchain`/`nockvm` can be analyzed.
- [ ] Make baseline workflow repeatable in CI and local environments with clear invocation paths.

### Out of Scope

- Replacing the existing statistical testing engine in `nockchain-bench` — this effort seeds and feeds it.
- Building new performance optimizations in `nockchain`/`nockvm` — this focuses on measurement infrastructure.
- Building a bespoke dashboard UI beyond GitHub Pages publication — defer until baseline reliability is proven.

## Context

`nockchain-bench` recently gained statistical testing support but currently needs initial baseline data to make future comparisons meaningful. The team expects frequent changes in `nockchain` and `nockvm`, and wants mathematically informed attribution of regressions and improvements. Current needs include scripted baseline creation, durable storage of run history, and automated publication of benchmark artifacts to GitHub Pages as the historical source of truth.

## Constraints

- **Compatibility**: Must integrate with existing `nockchain-bench` workflows and crate structure — avoid disrupting current benchmark usage.
- **Reproducibility**: Baseline generation must be deterministic enough for statistical comparison — results without consistent methodology are low value.
- **Automation**: GitHub Pages updates must run unattended where possible — manual publication does not scale with frequent benchmark runs.
- **Traceability**: Each baseline run must be attributable to code/config/environment context — attribution is necessary for root-cause analysis.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build a reusable script framework rather than one-off scripts | Supports ongoing baseline refreshes as benchmarks evolve | — Pending |
| Keep benchmark history on GitHub Pages | Preserves accessible longitudinal data for team-wide comparison | — Pending |
| Prioritize baseline quality before optimization experiments | Statistical confidence is required before trusting perf deltas | — Pending |

---
*Last updated: 2026-02-24 after initialization*
