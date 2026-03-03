# Requirements: Nockchain Bench Reset And Master Graft

**Defined:** 2026-03-03
**Core Value:** Benchmark outputs must reflect `nockchain` runtime behavior, not branch-specific harness cruft.

## v1 Requirements

### Scope Baseline

- [x] **SCOP-01**: Analysis scope is limited to `nockchain-bench` and its directly referenced runtime interfaces.
- [x] **SCOP-02**: `nockchain/master` is treated as the canonical compatibility target for grafting.
- [x] **SCOP-03**: Each incompatibility finding includes concrete evidence (file path, symbol/API reference, and branch context).

### Compatibility Inventory

- [x] **COMP-01**: Enumerate every `nockchain-bench` reference to symbols, types, modules, config fields, files, or behaviors absent in `nockchain/master`.
- [x] **COMP-02**: Capture PMA-related dependencies as explicit incompatibility entries.
- [x] **COMP-03**: Capture additional branch-only concepts (including potential NounSpaces-like dependencies) as explicit incompatibility entries.
- [x] **COMP-04**: Classify each incompatibility as `remove`, `replace-with-master-equivalent`, `feature-gate`, or `defer`.

### Provenance Analysis

- [x] **PROV-01**: Trace provenance for incompatibility entries to commits/branches where feasible, including the branch `bitemyapp/ag2-opt-persistence-madvise-checkpoint-chaff-pma-gc-checkpoint-streaming`.
- [ ] **PROV-02**: Distinguish divergences inherited from historical branch work versus local/current-branch additions.
- [ ] **PROV-03**: Produce a concise timeline describing major divergence events affecting SOL benchmark behavior.

### Graft Plan

- [ ] **GRAF-01**: Define a clean graft strategy to transplant `nockchain-bench` onto a new branch based on `nockchain/master`.
- [ ] **GRAF-02**: Ensure graft strategy removes references to non-master features and avoids introducing extra cruft.
- [ ] **GRAF-03**: Break graft strategy into execution-ready steps with risk notes and rollback points.

### Verification And Comparability

- [ ] **VERI-01**: Define objective acceptance criteria for apples-to-apples SOL benchmark comparisons across branches.
- [ ] **VERI-02**: Define a minimal reproducible validation matrix that proves comparability on the grafted bench.
- [ ] **VERI-03**: Identify data-quality guards that prevent misleading benchmark outputs from being treated as valid.

## v2 Requirements

### Optional Automation

- **AUTO-01**: Add automation to continuously detect new non-master dependencies introduced into `nockchain-bench`.
- **AUTO-02**: Add automated compatibility checks that fail when graft assumptions drift from `nockchain/master`.

## Out of Scope

| Feature | Reason |
|---------|--------|
| GitHub Actions redesign | Explicitly excluded by user for this project cycle |
| New runtime features in `nockchain/master` to satisfy bench assumptions | Project goal is adapting bench to master, not changing master scope |
| New benchmark categories unrelated to SOL comparability reset | Avoids scope creep before core benchmark trust is restored |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SCOP-01 | Phase 1 | Complete |
| SCOP-02 | Phase 1 | Complete |
| SCOP-03 | Phase 1 | Complete |
| COMP-01 | Phase 2 | Complete |
| COMP-02 | Phase 2 | Complete |
| COMP-03 | Phase 2 | Complete |
| COMP-04 | Phase 2 | Complete |
| PROV-01 | Phase 3 | Complete |
| PROV-02 | Phase 3 | Pending |
| PROV-03 | Phase 3 | Pending |
| GRAF-01 | Phase 4 | Pending |
| GRAF-02 | Phase 4 | Pending |
| GRAF-03 | Phase 4 | Pending |
| VERI-01 | Phase 5 | Pending |
| VERI-02 | Phase 5 | Pending |
| VERI-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-03*
*Last updated: 2026-03-03 after initial definition*
