# Feature Research

**Domain:** Benchmark baseline and longitudinal performance regression analysis infrastructure
**Researched:** 2026-02-24
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Reproducible benchmark run orchestration (warmup, sample size, iteration control) | Without repeatable execution, longitudinal comparisons are not trustworthy | MEDIUM | Standard in `criterion`, `pyperf`, `pytest-benchmark`, and `hyperfine`; include explicit run config in saved metadata |
| Statistical comparison with significance + noise handling | Teams expect regression calls to be statistically defensible, not based on single samples | HIGH | `benchstat`, `Criterion.rs`, `pyperf`, and Bencher all emphasize statistical tests, confidence intervals, outlier handling, and noise thresholds |
| Baseline persistence with historical series | Baselines must survive local machine state and CI job lifetimes | MEDIUM | Store run outputs + summaries in versioned files/repo-backed static hosting; avoid artifact-only retention for long history |
| Rich run metadata capture (commit SHA, branch, benchmark config, testbed/hardware, tool versions) | Attribution of regressions requires environment and code context | MEDIUM | Bencher models this explicitly as Branch + Testbed (+ optional Spec); include host/env fingerprint in every run record |
| CI-ready automation and regression gating | Teams expect unattended regression checks in pull requests and scheduled runs | MEDIUM | `pytest-benchmark --benchmark-compare-fail` and Bencher threshold alerts/gating set baseline expectation |
| Machine-readable outputs + human-readable reports | Infra pipelines need parsable data, while maintainers need fast visual review | LOW | JSON/CSV/event-stream output plus static report publishing (GitHub Pages or similar) is now table stakes |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Branch-aware baseline lineage with start-point cloning | Feature branches inherit meaningful baselines and avoid false deltas against unrelated history | HIGH | Bencher's branch/start-point model is a strong pattern; useful for active perf work across many concurrent branches |
| First-class testbed/spec dimensioning (hardware/software fingerprint as model key) | Prevents mixed-hardware contamination and enables fair cross-host analysis | HIGH | Treat testbed/spec as part of the baseline identity, not just metadata for display |
| Multi-model regression policies per metric (t-test, IQR, z-score, static thresholds) | Teams can tune sensitivity by metric behavior instead of one-size-fits-all thresholds | HIGH | Bencher demonstrates this clearly; improves signal quality in noisy suites |
| Immutable publication pipeline with provenance checks | Creates auditability for historical benchmark evidence and reduces accidental tampering | MEDIUM | GitHub Actions artifact digest validation + static-site publication pattern enables traceable records |
| Benchmark data quality scoring (noise, outlier density, insufficient sample flags) | Surfaces when not to trust a regression result, reducing wasted debugging | MEDIUM | Derived from Criterion/pyperf/benchstat guidance on noise/outliers; operationalized as explicit health signals |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Single "golden run" baseline snapshot | Feels simple to maintain | High variance and drift make one-off baselines brittle; causes false alarms and missed regressions | Rolling historical windows with minimum sample sizes and significance tests |
| Mean-only pass/fail thresholds without uncertainty stats | Easy to explain in CI | Ignores variance/outliers; encourages p-hacking by rerunning until pass | Require confidence intervals/p-values or robust stats + explicit noise policy |
| Mixing all environments into one baseline | Simplifies storage schema | Hides hardware/OS/runtime effects; attribution becomes impossible | Partition baselines by branch + testbed/spec and compare like-for-like |
| Artifact-only storage strategy for longitudinal history | Quick to implement in CI | Artifact retention policies are bounded and not designed as permanent history | Promote curated benchmark datasets to durable storage (repo/Pages/object store) |
| Real-time dashboard-first scope in v1 | Stakeholders like immediate visuals | Pulls effort from data correctness/reproducibility; can lock in weak data model early | Build correctness-first pipeline + static reports first, then add interactive dashboards |

## Feature Dependencies

```text
[Reproducible run orchestration]
    └──requires──> [Run configuration schema]
                        └──requires──> [Metadata capture]

[Metadata capture]
    └──requires──> [Baseline persistence]
                        └──requires──> [Machine-readable outputs]

[Baseline persistence]
    └──enables──> [Statistical comparison + regression detection]
                        └──enables──> [CI gating + alerts]

[Branch/testbed-aware baseline lineage]
    └──enhances──> [Statistical comparison + regression detection]

[Real-time dashboard-first scope]
    └──conflicts──> [MVP delivery of reliable baseline pipeline]
```

### Dependency Notes

- **Reproducible run orchestration requires run configuration schema:** fixed warmup/sample/iteration knobs must be persisted to replay experiments.
- **Run configuration schema requires metadata capture:** without commit/environment linkage, identical configs still produce unactionable results.
- **Metadata capture requires baseline persistence:** metadata has value only when tied to durable historical run records.
- **Baseline persistence requires machine-readable outputs:** CI and downstream analysis need stable parseable artifacts, not console text.
- **Statistical comparison requires baseline persistence:** significance and trend detection need enough prior runs to form a distribution.
- **CI gating depends on statistical comparison:** pass/fail needs robust change detection to avoid flaky build signals.
- **Branch/testbed-aware lineage enhances regression detection:** it reduces false positives caused by unrelated branch history or host differences.
- **Dashboard-first conflicts with MVP reliability:** prioritizing UI before data integrity increases rewrite risk.

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [ ] Deterministic baseline generation workflow (local + CI invocation parity) — core trust requirement
- [ ] Structured run record format with full metadata envelope — prerequisite for attribution
- [ ] Longitudinal baseline store + static publication updates — preserves history for comparison
- [ ] Statistical compare command with regression classification — converts raw data into decisions
- [ ] CI integration for scheduled runs + PR regression checks — validates operational usefulness

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] Branch start-point baseline cloning — add when feature-branch volume starts creating baseline friction
- [ ] Multi-model threshold policies per metric — add when one-threshold policy yields noisy gates
- [ ] Benchmark quality score (noise/outlier confidence) — add when teams need faster triage of flaky results

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] Interactive dashboarding and drill-down UX — defer until data model and pipeline semantics stabilize
- [ ] Automated root-cause correlation across perf, infra, and code-change metadata — defer until enough historical scale exists

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Reproducible run orchestration + metadata envelope | HIGH | MEDIUM | P1 |
| Durable longitudinal baseline store + publication | HIGH | MEDIUM | P1 |
| Statistical regression detection + CI gate | HIGH | HIGH | P1 |
| Branch/testbed-aware baseline lineage | HIGH | HIGH | P2 |
| Multi-model threshold policies | MEDIUM | HIGH | P2 |
| Quality scoring and advanced diagnostics | MEDIUM | MEDIUM | P2 |
| Interactive dashboard UX | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Competitor A | Competitor B | Our Approach |
|---------|--------------|--------------|--------------|
| Historical benchmark tracking | `asv` tracks over project lifetime with static web frontend | Bencher tracks reports by branch/testbed with alerts | Keep static publication as source-of-truth and add strict metadata schema for reproducibility |
| Statistical regression detection | `benchstat` provides robust A/B significance and geomean summaries | Bencher offers configurable statistical threshold models | Use statistically rigorous defaults (robust comparisons) and configurable policy profiles per metric |
| CI integration and gating | `pytest-benchmark` supports compare and compare-fail workflows | Bencher provides CI-centric threshold alerts and fail-on-alert mode | Build CI-first workflows with deterministic runs and explainable fail reasons |
| Machine-readable output for downstream tools | cargo-criterion supports JSON message stream | pytest-benchmark outputs JSON and saved run files | Standardize a stable run schema and publish both raw + derived artifacts |

## Sources

- Criterion.rs docs: command-line output, analysis process, and machine-readable output guidance (`https://bheisler.github.io/criterion.rs/book/`) — MEDIUM/HIGH confidence (official docs)
- pyperf docs: analysis commands, comparison semantics, and system tuning (`https://pyperf.readthedocs.io/en/latest/`) — HIGH confidence (official docs)
- Go `benchstat` docs: significance testing workflow, sample-size recommendations, and anti-p-hacking guidance (`https://pkg.go.dev/golang.org/x/perf/cmd/benchstat`) — HIGH confidence (official docs; published 2026-02-11 for referenced version)
- pytest-benchmark docs: save/compare/fail options and storage concepts (`https://pytest-benchmark.readthedocs.io/en/latest/usage.html`) — HIGH confidence (official docs; last updated 2026-02-16)
- Bencher docs: branch/testbed/spec model and configurable threshold tests (`https://bencher.dev/docs/explanation/benchmarking/`, `https://bencher.dev/docs/explanation/thresholds/`) — HIGH confidence (official docs; benchmarking page updated 2026-01-30)
- GitHub Docs: GitHub Pages static hosting and workflow artifact retention behavior (`https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages`, `https://docs.github.com/en/actions/tutorials/store-and-share-data`) — HIGH confidence (official docs)

---
*Feature research for: benchmark baseline and performance regression infrastructure*
*Researched: 2026-02-24*
