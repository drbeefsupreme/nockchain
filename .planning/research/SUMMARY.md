# Project Research Summary

**Project:** Nockchain Benchmark Baseline Framework
**Domain:** Reproducible benchmark baseline generation and longitudinal regression analysis in a Rust monorepo
**Researched:** 2026-02-24
**Confidence:** MEDIUM-HIGH

## Executive Summary

This project is a benchmarking infrastructure product, not a benchmarking algorithm project. The research converges on a correctness-first architecture: deterministic benchmark execution, schema-first normalized artifacts, explicit baseline governance, and CI automation that publishes durable history. Experts in this space consistently separate measurement collection from statistical decision logic and separate immutable run history from mutable baseline pointers to preserve auditability.

The recommended approach for Nockchain is a dual-lane benchmark pipeline anchored in pinned Rust toolchains and controlled environments: wall-time/statistical signals with `criterion` plus deterministic CI signal from `iai-callgrind` where needed. Build local and CI entrypoints from the same scripts, emit canonical JSON with full provenance metadata, feed `nockchain-bench` regression logic, and publish compact historical datasets through GitHub Pages custom workflows.

The biggest risks are baseline drift, environment drift, and weak statistical gates that create noisy or untrusted alerts. Mitigation is clear in the research: immutable baseline IDs with reviewed reseed flow, benchmark execution contracts on stable runner classes, strict metadata/schema validation, and threshold calibration from no-change replay data before turning hard fail gates on.

## Key Findings

### Recommended Stack

The stack is mature and conservative: keep the analysis core in Rust, pin versions aggressively, and use GitHub-native deployment primitives for publication. The critical architectural decision is to treat schema and provenance as first-class, then layer automation and visualization on top.

**Core technologies:**
- `rust-toolchain.toml` pinned stable (e.g., `1.89.x`): deterministic compiler/runtime behavior across local and CI runs.
- `criterion@0.8.x`: statistically grounded microbenchmarking and comparison workflows.
- `iai-callgrind@0.16.x`: deterministic instruction/cache-event lane for noisy CI environments.
- GitHub Pages custom workflow (`configure-pages`, `upload-pages-artifact`, `deploy-pages`): durable static history publication with explicit deploy permissions.
- `benchmark-action/github-action-benchmark@v1`: fast path to longitudinal charts and alerting without bespoke dashboard work.

Critical version constraints: pin Rust toolchain and action versions; install Valgrind on Linux runners for `iai-callgrind`; use `pages: write` and `id-token: write` permissions for Pages deploy.

### Expected Features

MVP must prioritize trust and attribution, not UX polish. Table-stake expectations are reproducible orchestration, statistically defensible comparisons, durable history, rich metadata, and CI-ready gating with machine-readable outputs.

**Must have (table stakes):**
- Deterministic benchmark orchestration with persisted run configuration.
- Full run provenance (commit, branch, toolchain, host/testbed, benchmark settings).
- Durable longitudinal baseline/history store with static publication.
- Statistical compare + regression classification usable by CI gates.
- Scheduled and PR-based automation with human-readable reports plus JSON artifacts.

**Should have (competitive):**
- Branch-aware baseline lineage with start-point cloning.
- Testbed/spec-aware baseline partitioning as part of identity.
- Metric-specific threshold models (t-test/IQR/z-score/static policy).
- Data quality scoring for noise/outlier confidence.

**Defer (v2+):**
- Interactive dashboard/drill-down UX.
- Automated cross-signal root-cause correlation.

### Architecture Approach

Architecture should follow a contract-first pipeline: deterministic runner -> environment fingerprinting -> normalizer to canonical schema -> analyzer/verdict gate -> immutable run artifact write -> compact site build -> Pages deploy. Keep a hard boundary where the analyzer never parses raw harness output and where deploy consumes built artifacts only (no rebuild-at-deploy).

**Major components:**
1. Benchmark Runner + Environment Fingerprinter - execute deterministic runs and capture attribution metadata.
2. Result Adapter/Normalizer - transform raw harness outputs into schema-validated canonical JSON.
3. Baseline Store + Update Policy - maintain mutable accepted baselines separately from immutable run history.
4. Regression Analyzer + Verdict Gate - produce pass/warn/fail with machine-readable deltas.
5. Site Builder + Pages Publisher - convert history into compact static data and deploy through split build/deploy jobs.

### Critical Pitfalls

1. **Silent baseline rewrites** - enforce immutable baseline IDs and reviewed reseed-only promotion.
2. **Environment drift mistaken for regressions** - pin runner class/CPU policy and maintain trusted benchmark lanes.
3. **Missing provenance metadata** - require schema validation that blocks incomplete run manifests.
4. **Build/profile mismatch** - pin toolchain/profile and fail if baseline/candidate manifests differ.
5. **Mis-tuned statistical gates** - calibrate thresholds on replay/no-change runs before strict CI failure policy.

## Implications for Roadmap

Based on combined research, the roadmap should be dependency-driven and governance-first.

### Phase 1: Data Contracts and Baseline Governance
**Rationale:** Every downstream step depends on stable schemas and immutable baseline policy.
**Delivers:** Run manifest schema, metric schema, baseline registry layout, reseed approval workflow.
**Addresses:** Reproducible orchestration prerequisites, metadata envelope, durable baseline identity.
**Avoids:** Mutable baseline pitfall and missing provenance pitfall.

### Phase 2: Deterministic Execution Pipeline
**Rationale:** Trusted data must exist before analysis quality can be trusted.
**Delivers:** Pinned-toolchain runner scripts, env fingerprint capture, raw->canonical normalization, validation gates.
**Uses:** `rust-toolchain.toml`, `criterion`, optional `iai-callgrind`, `serde`/`serde_json`/`sysinfo`.
**Implements:** Runner, fingerprinter, normalizer boundaries from architecture research.
**Avoids:** Environment drift, wrong-work measurement, and build/profile mismatch.

### Phase 3: Regression Decisions and CI Policy
**Rationale:** Once trustworthy inputs exist, convert them into operational decisions.
**Delivers:** Regression analyzer wiring, warn/fail semantics, scheduled + PR CI lanes, explainable delta reports.
**Addresses:** Statistical comparison table stakes and CI gating requirements.
**Avoids:** Noisy or ignored alerts via calibration and tiered policies.

### Phase 4: Durable History Publication
**Rationale:** Longitudinal value only exists if history is durable and accessible.
**Delivers:** Immutable run archival strategy, compact series generation, GitHub Pages deploy workflow, retention/size guardrails.
**Uses:** GitHub artifact + Pages workflow pattern, `github-action-benchmark` optional charting.
**Avoids:** History loss, monolithic data growth, and publication timeouts.

### Phase 5: Differentiators and Scale Hardening
**Rationale:** Add complexity only after baseline trust and workflow adoption are proven.
**Delivers:** Branch start-point cloning, testbed/spec-aware lineage, multi-model thresholds, quality scoring/canaries.
**Addresses:** P2 differentiators and governance hardening.
**Avoids:** Premature dashboard-first scope and false confidence from unmonitored measurement drift.

### Phase Ordering Rationale

- Schema/governance first prevents expensive rewrites and baseline integrity failures.
- Deterministic collection before gating keeps CI decisions anchored to reliable signal.
- Publication after analyzer integration ensures published history reflects validated semantics.
- Differentiators last reduces risk while preserving a clear path to competitive advantages.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Statistical calibration specifics (false-positive budget, per-metric policy defaults) require replay analysis on real Nockchain distributions.
- **Phase 4:** Storage/compaction strategy needs repo growth and Pages limit modeling for expected benchmark cadence.
- **Phase 5:** Branch/testbed lineage and multi-model policy design need product-level decisions on baseline promotion ownership and UX.

Phases with standard patterns (can likely skip research-phase):
- **Phase 1:** Schema-first + immutable baseline governance is well-established and strongly documented.
- **Phase 2:** Deterministic runner + normalization pipeline follows common CI benchmark architecture patterns.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Strong official documentation for toolchain pinning/Pages/actions; some ecosystem choices rely on community maturity signals. |
| Features | HIGH | Cross-ecosystem agreement from Criterion, pyperf, benchstat, pytest-benchmark, and Bencher on core expectations. |
| Architecture | MEDIUM | Pattern quality is high, but some implementation details are inferred from mixed official/community examples. |
| Pitfalls | MEDIUM | Risks are credible and consistent, but many mitigations depend on operational discipline and local calibration data. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Variance envelope by benchmark class:** run repeated same-SHA experiments on target runners to set SLOs and gate thresholds.
- **Baseline promotion governance owner:** define who can reseed and approval rules before enabling automatic workflows.
- **Long-term storage economics:** choose canonical archive location (repo vs object store) based on projected run volume.
- **Fast-vs-full suite split:** decide PR gate subset versus scheduled full suite to balance CI latency and signal coverage.

## Sources

### Primary (HIGH confidence)
- Rust/Cargo official docs (`rustup` overrides, Cargo profiles/bench) - deterministic toolchain and build behavior.
- GitHub official docs (Actions artifacts/jobs, Pages custom workflows/limits, runner models) - CI and publication architecture.
- Official benchmarking docs: `criterion`, `pyperf`, `benchstat`, `pytest-benchmark`, Bencher docs - expected feature and statistical patterns.

### Secondary (MEDIUM confidence)
- `iai-callgrind` crate documentation - deterministic CI measurement lane practices.
- `benchmark-action/github-action-benchmark` project docs - practical history/charting integration model.
- Criterion-specific FAQ/advanced config guidance - operational tuning and false-detection handling.

### Tertiary (LOW confidence)
- Community implementation conventions for long-horizon retention/compaction at scale - useful direction, requires local validation.

---
*Research completed: 2026-02-24*
*Ready for roadmap: yes*
