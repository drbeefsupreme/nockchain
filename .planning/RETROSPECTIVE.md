# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Benchmark Baseline Framework

**Shipped:** 2026-02-26
**Phases:** 3 | **Plans:** 7 | **Sessions:** 3

### What Was Built
- Reproducible baseline execution with TOML config profiles, local runner, and CI parity workflow
- Statistical comparison engine with bootstrap CI four-way verdict and dual-format reports (Markdown + JSON)
- PR regression gates with cached baseline restore and advisory reporting
- Immutable history append pipeline with concurrency-serialized gh-pages writes
- GitHub Pages dashboard with Chart.js trend charts and auto-deploy from both baseline and advancement workflows

### What Worked
- Wave-based parallel execution kept phase execution fast (~5 min/plan average)
- TOML config with profile overlays cleanly separates quick (CI) and full (baseline) modes
- Strict provenance validation caught issues early — no incomplete artifacts
- Advisory-only PR gates (exit 0) avoid blocking PRs on noisy benchmarks while still surfacing regressions
- workflow_call reuse pattern eliminated deploy logic duplication

### What Was Inefficient
- Metric key name mismatch between producer (sol_history_append.sh) and consumer (pages/index.html) slipped through planning — caught by verifier
- Phase 1 and 2 roadmap progress tracking wasn't properly updated during execution (disk_status showed complete but roadmap showed "Planned")
- Phase 2 verification was noted as a pending todo but never formally run before Phase 3

### Patterns Established
- PUBLISH_DIR delta pattern: write only new files + updated index, let keep_files: true preserve history
- Column-name lookup via awk header scan instead of hardcoded positions for TSV schema resilience
- Concurrency group serialization for any workflow writing to shared branches

### Key Lessons
1. Interface contracts between plans (especially data schemas) need explicit verification at plan boundaries — the _median suffix mismatch was a cross-plan integration bug
2. Verifier agents are valuable — the automated spot-check caught the metric key mismatch that would have produced a silently broken dashboard
3. Advisory-only gates are the right starting posture for noisy benchmarks — can tighten later with confidence data

### Cost Observations
- Model mix: ~20% opus (orchestration), ~80% sonnet (research, planning, execution, verification)
- Sessions: 3 (Phase 1+2 execution, Phase 3 plan+execute, milestone completion)
- Notable: Phase 3 plans executed in ~5 min total — Bash/YAML plans are significantly faster than Rust-heavy phases

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 3 | 3 | Initial baseline — established wave execution, verification loop, advisory gates |

### Top Lessons (Verified Across Milestones)

1. Cross-plan data contracts need explicit schema verification (v1.0: metric key mismatch)
2. Verifier agents catch integration bugs that pass individual plan checks (v1.0: dashboard + script mismatch)
