# Roadmap: Nockchain Bench Reset And Master Graft

## Overview

This roadmap restores benchmark trust by moving from bounded analysis to evidence-backed compatibility inventory, then provenance, then an execution-ready master graft, and finally objective comparability verification for SOL benchmarks.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Scope Baseline And Evidence Contract** - Lock the analysis boundary and evidence standard for all findings.
- [x] **Phase 2: Master Compatibility Inventory** - Produce a complete incompatibility inventory against `nockchain/master` with action classifications.
- [ ] **Phase 3: Provenance And Divergence Timeline** - Explain where incompatibilities came from and how SOL behavior diverged over time.
- [ ] **Phase 4: Master Graft Execution Plan** - Define a stepwise, rollback-aware strategy to transplant bench onto a fresh master branch.
- [ ] **Phase 5: Comparability Verification Baseline** - Define pass/fail criteria and validation matrix for apples-to-apples SOL comparisons.

## Phase Details

### Phase 1: Scope Baseline And Evidence Contract
**Goal**: The maintainer has a clear, enforceable scope and evidence contract for every compatibility finding.
**Depends on**: Nothing (first phase)
**Requirements**: SCOP-01, SCOP-02, SCOP-03
**Success Criteria** (what must be TRUE):
  1. Maintainer can see a documented scope boundary limited to `nockchain-bench` and directly referenced runtime interfaces.
  2. Maintainer can verify `nockchain/master` is explicitly recorded as the canonical compatibility target.
  3. Every incompatibility finding is documented with file path, symbol/API reference, and branch context evidence.
**Plans**: 2 plans
- [x] 01-01-PLAN.md - Establish scope boundary and canonical master target artifacts.
- [x] 01-02-PLAN.md - Add hard-fail evidence contract verification and one-command make target.

### Phase 2: Master Compatibility Inventory
**Goal**: The maintainer has a complete list of bench dependencies that do not exist in `nockchain/master`, each with a disposition.
**Depends on**: Phase 1
**Requirements**: COMP-01, COMP-02, COMP-03, COMP-04
**Success Criteria** (what must be TRUE):
  1. Maintainer can review one inventory that enumerates all references absent in `nockchain/master` across symbols, modules, configs, files, and behavior assumptions.
  2. PMA-related dependencies appear as explicit, searchable incompatibility entries.
  3. Additional branch-only concepts (including NounSpaces-like dependencies if present) appear as explicit incompatibility entries.
  4. Every incompatibility entry is classified as `remove`, `replace-with-master-equivalent`, `feature-gate`, or `defer`.
**Plans**: 3 plans
- [x] 02-01-PLAN.md - Establish canonical inventory schema, bench candidate index, and closure checklist IDs.
- [x] 02-02-PLAN.md - Populate incompatibility inventory with explicit PMA and branch-only concept entries plus dispositions.
- [x] 02-03-PLAN.md - Add hard-fail verifier and one-command make gate for inventory completeness and classification quality.

### Phase 3: Provenance And Divergence Timeline
**Goal**: The maintainer can explain when and where incompatible bench behavior entered the branch history.
**Depends on**: Phase 2
**Requirements**: PROV-01, PROV-02, PROV-03
**Success Criteria** (what must be TRUE):
  1. Maintainer can trace incompatibility entries to source commits/branches where feasible, including `bitemyapp/ag2-opt-persistence-madvise-checkpoint-chaff-pma-gc-checkpoint-streaming`.
  2. Maintainer can distinguish inherited historical-branch divergences from local/current-branch additions.
  3. Maintainer can read a concise timeline of major divergence events affecting SOL benchmark behavior.
**Plans**: 3 plans
- [x] 03-01-PLAN.md - Create canonical Phase 3 provenance schema, seeded evidence TSV, and stable closure checklist IDs.
- [ ] 03-02-PLAN.md - Populate dependency-level provenance evidence, classifications, and thematic timeline events.
- [ ] 03-03-PLAN.md - Add hard-fail verifier automation and make-gate closure checks for Phase 3.

### Phase 4: Master Graft Execution Plan
**Goal**: The maintainer has an execution-ready method to transplant `nockchain-bench` onto a fresh master-based branch without non-master coupling.
**Depends on**: Phase 3
**Requirements**: GRAF-01, GRAF-02, GRAF-03
**Success Criteria** (what must be TRUE):
  1. Maintainer can execute a documented graft sequence from clean `nockchain/master` branch creation through bench transplant.
  2. The graft sequence removes references to non-master features and identifies exact replacements or gates where required.
  3. Each execution step includes explicit risk notes and rollback points the maintainer can follow.
**Plans**: TBD

### Phase 5: Comparability Verification Baseline
**Goal**: The maintainer can objectively determine whether SOL benchmark outputs are apples-to-apples across branches.
**Depends on**: Phase 4
**Requirements**: VERI-01, VERI-02, VERI-03
**Success Criteria** (what must be TRUE):
  1. Maintainer can apply objective acceptance criteria and produce a clear pass/fail comparability verdict.
  2. Maintainer can run a minimal reproducible validation matrix that demonstrates comparability on the grafted bench.
  3. Maintainer can detect and reject misleading benchmark outputs using documented data-quality guards.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Scope Baseline And Evidence Contract | 2/2 | Complete | 2026-03-03 |
| 2. Master Compatibility Inventory | 3/3 | Complete | 2026-03-03 |
| 3. Provenance And Divergence Timeline | 1/3 | In Progress | - |
| 4. Master Graft Execution Plan | 0/TBD | Not started | - |
| 5. Comparability Verification Baseline | 0/TBD | Not started | - |
