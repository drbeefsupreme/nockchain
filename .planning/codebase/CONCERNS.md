# Codebase Concerns

**Analysis Date:** 2026-03-03

## Tech Debt

**SOL artifact handling:**
- Issue: fixture extraction and eager archive loading add overhead and extra state transitions
- Impact: benchmark memory signals can be influenced by harness behavior
- Fix approach: stream/load-shape controls and tighter isolation between harness and measured runtime

**Heuristic profiling buckets:**
- Issue: some memory/checkpoint metrics are inference-heavy
- Impact: interpretation can drift across runtime changes
- Fix approach: add invariant checks and explicit provenance fields per metric

## Known Bugs / Validity Risks

**Baseline selection drift:**
- Symptoms: stale or branch-misaligned baseline windows can be selected
- Trigger: broad fallback selection without strict recency/branch gates
- Workaround: manually pin baseline inputs when running comparisons

**Silent coercion in ingest paths:**
- Symptoms: parse failures becoming `0` can hide malformed data
- Trigger: permissive parsing defaults in guard ingest
- Workaround: fail-fast validation for required numeric fields

## Security Considerations

**Generated reports/artifacts:**
- Risk: accidental path leakage or environment-specific references in published artifacts
- Current mitigation: mostly procedural
- Recommendation: add validation/scrubbing before publish

## Performance Bottlenecks

**Large archive replay:**
- Problem: replay paths and artifact materialization can become I/O and memory bound
- Improvement path: incremental processing and more deterministic replay boundaries

## Fragile Areas

**Branch-coupled scripts and worktrees:**
- Why fragile: hardcoded branch names/path assumptions in scripts and worktree pointers
- Common failures: non-portable runs and apples-to-oranges comparisons
- Safe modification: centralize branch manifest and enforce branch identity checks per run

## Scaling Limits

**Benchmark matrix combinatorics:**
- Limit: parameter sweeps explode runtime and artifact volume quickly
- Symptoms at limit: long wall times, difficult result curation
- Scaling path: canonical matrix subsets + mandatory provenance metadata

## Dependencies at Risk

**Branch-specific harness assumptions:**
- Risk: `nockchain-bench` behavior tied to branches diverged from `nockchain/master`
- Impact: invalid cross-branch comparisons and misleading regression verdicts
- Mitigation: explicit compatibility matrix and feature-gated behavior

## Test Coverage Gaps

**Always-on validation of benchmark correctness:**
- Gap: CI frequently exercises only a subset of realistic heavy paths
- Risk: regressions in graftability/comparability slip through
- Priority: High

---
*Concerns audit: 2026-03-03*
*Update as issues are fixed or new ones discovered*
