# Roadmap: Nockchain Benchmark Baseline Framework

## Overview

This roadmap delivers a reproducible benchmark baseline system from trusted data collection to decision-ready comparison and durable publication, so maintainers can interpret performance changes in `nockchain` and `nockvm` with confidence.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Reproducible Baseline Execution** - Establish a single local/CI workflow that produces canonical benchmark artifacts with full provenance.
- [ ] **Phase 2: Regression Comparison and PR Gates** - Turn baseline data into statistically defensible comparison outcomes and enforce them in PR checks.
- [ ] **Phase 3: Durable History and Pages Publication** - Automate baseline refresh, preserve immutable run history, and publish artifacts to GitHub Pages.

## Phase Details

### Phase 1: Reproducible Baseline Execution
**Goal**: Maintainers can run a single deterministic baseline workflow locally and in CI that emits canonical, comparison-ready artifacts with provenance.
**Depends on**: Nothing (first phase)
**Requirements**: ORCH-01, ORCH-02, ORCH-03, DATA-01, DATA-02
**Success Criteria** (what must be TRUE):
  1. Maintainer can run one scripted command locally to generate baseline benchmark artifacts for `nockchain-bench`.
  2. Maintainer can run the same workflow in CI with equivalent configuration semantics and produce equivalent artifact structure.
  3. Maintainer can set warmup, sample size, and iteration controls via versioned configuration and see those settings reflected in outputs.
  4. Maintainer can inspect each run artifact as canonical machine-readable data that includes commit SHA, branch, benchmark config, environment fingerprint, and tool versions.
**Plans**: 3 plans
- [ ] 01-01-PLAN.md — Versioned baseline config contract with profile support (ORCH-03)
- [ ] 01-02-PLAN.md — Provenance manifest model and validation (DATA-01, DATA-02)
- [ ] 01-03-PLAN.md — Local runner script and CI parity workflow (ORCH-01, ORCH-02)

### Phase 2: Regression Comparison and PR Gates
**Goal**: Maintainers can compare candidate performance against baseline with clear statistical verdicts and use those verdicts during PR review.
**Depends on**: Phase 1
**Requirements**: STAT-01, STAT-02, PIPE-02
**Success Criteria** (what must be TRUE):
  1. Maintainer can compare a candidate run against the active baseline and receive a classification of improvement, regression, or no significant change.
  2. Maintainer can inspect machine-readable delta output suitable for CI policy and downstream automation.
  3. Maintainer can read a human-readable comparison summary that explains the regression decision for review.
  4. Maintainer can see PR-time CI run the regression check against established baseline data and report the result in the pull request.
**Plans**: 2 plans
- [ ] 02-01-PLAN.md — Statistical comparison engine with four-way verdict and dual-format reports (STAT-01, STAT-02)
- [ ] 02-02-PLAN.md — PR regression workflow and baseline cache integration (PIPE-02)

### Phase 3: Durable History and Pages Publication
**Goal**: Baseline history is immutable and continuously extended, with latest baseline datasets and history automatically published for team access.
**Depends on**: Phase 1, Phase 2
**Requirements**: DATA-03, PIPE-01, PIPE-03
**Success Criteria** (what must be TRUE):
  1. Maintainer can run scheduled CI baseline generation that appends new historical records without deleting prior runs.
  2. Maintainer can advance the active baseline reference while prior historical run records remain immutable and retrievable.
  3. Maintainer can see GitHub Pages automatically updated from CI with benchmark history and the latest baseline artifacts.
  4. Maintainer can access published baseline history to retrieve prior runs for longitudinal analysis.
**Plans**: 2 plans
- [ ] 03-01-PLAN.md — History append infrastructure and baseline advancement workflow (DATA-03, PIPE-01)
- [ ] 03-02-PLAN.md — GitHub Pages dashboard and deployment pipeline (PIPE-03)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 1.1 -> 1.2 -> 2 -> 2.1 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Reproducible Baseline Execution | 0/3 | Planned | - |
| 2. Regression Comparison and PR Gates | 0/2 | Planned | - |
| 3. Durable History and Pages Publication | 0/2 | Planned | - |
