# Requirements: Nockchain Benchmark Baseline Framework

**Defined:** 2026-02-24
**Core Value:** Every benchmark comparison uses a reproducible, statistically valid baseline so performance changes can be interpreted with confidence.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Orchestration

- [x] **ORCH-01**: Maintainer can run a single scripted command to generate baseline benchmark runs for `nockchain-bench` locally.
- [x] **ORCH-02**: Maintainer can run the same scripted workflow in CI with equivalent configuration semantics.
- [x] **ORCH-03**: Maintainer can configure warmup, sample size, and iteration controls for baseline runs using versioned configuration.

### Data and Provenance

- [x] **DATA-01**: Maintainer can persist each benchmark run as a machine-readable canonical artifact for longitudinal comparison.
- [x] **DATA-02**: Maintainer can view captured run provenance including commit SHA, branch, benchmark configuration, environment fingerprint, and tool versions.
- [x] **DATA-03**: Maintainer can preserve immutable historical run records while tracking the active baseline reference.

### Statistical Comparison

- [x] **STAT-01**: Maintainer can compare a candidate run against baseline data using statistically defensible classification (improvement, regression, no significant change).
- [x] **STAT-02**: Maintainer can inspect comparison output with both machine-readable deltas and human-readable summary suitable for review.

### CI and Publication

- [x] **PIPE-01**: Maintainer can run scheduled baseline generation in CI that appends new baseline history without deleting prior runs.
- [x] **PIPE-02**: Maintainer can run PR-time regression checks in CI against established baseline data.
- [ ] **PIPE-03**: Maintainer can publish benchmark history and latest baseline artifacts to GitHub Pages through an automated workflow.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Baseline Strategy

- **BASE-01**: Maintainer can clone baseline start-points per branch to reduce false deltas from unrelated history.
- **BASE-02**: Maintainer can partition baseline lineage by testbed/spec identity as a first-class dimension.

### Advanced Analysis

- **ANLY-01**: Maintainer can configure metric-specific regression policies (for example, t-test, robust IQR, or static thresholds).
- **ANLY-02**: Maintainer can view benchmark quality scores (noise/outlier confidence) before trusting regression verdicts.

### UX and Correlation

- **UX-01**: Maintainer can explore interactive dashboard drill-downs for historical benchmark trends.
- **UX-02**: Maintainer can correlate regressions automatically across benchmark, infra, and code-change signals.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Replacing `nockchain-bench` statistical testing core | Current goal is baseline data seeding and pipeline integration, not rewriting analysis engine |
| Implementing performance optimizations in `nockchain` or `nockvm` | This project builds measurement infrastructure to support future optimization work |
| Dashboard-first implementation in v1 | Correctness and reproducibility must be established before interactive UX investment |
| Single "golden run" baseline snapshot model | One-off snapshots are too noisy and undermine reliable statistical interpretation |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ORCH-01 | Phase 1 | Complete |
| ORCH-02 | Phase 1 | Complete |
| ORCH-03 | Phase 1 | Complete |
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 3 | Complete |
| STAT-01 | Phase 2 | Complete |
| STAT-02 | Phase 2 | Complete |
| PIPE-01 | Phase 3 | Complete |
| PIPE-02 | Phase 2 | Complete |
| PIPE-03 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-24*
*Last updated: 2026-02-24 after Phase 2 completion*
