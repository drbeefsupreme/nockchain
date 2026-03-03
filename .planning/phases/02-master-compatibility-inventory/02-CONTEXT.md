# Phase 2: Master Compatibility Inventory - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Produce a complete incompatibility inventory against `nockchain/master` for `nockchain-bench`, including explicit dispositions per entry. This phase defines and records inventory findings; it does not add new benchmark capabilities.

</domain>

<decisions>
## Implementation Decisions

### Inventory Unit And Coverage
- Use a hybrid entry model: one primary incompatibility entry with linked symbol/API references.
- Keep sweep scope to `nockchain-bench` code references only.
- Within that scope, include operational assumptions (for example PMA/env/config toggles) as explicit incompatibility entries when evidenced in bench code paths.
- Use static references as primary evidence (no mandatory execution proof for every entry).

### Disposition Rules
- If branch-only dependency has no clear master equivalent yet, default disposition is `defer`.
- For optional branch-only behavior, prefer `remove` rather than `feature-gate`.
- Only classify as `replace-with-master-equivalent` when a concrete master target is identified.
- For PMA-related dependencies, apply a default bias toward `remove`.

### Claude's Discretion
- Exact inventory field layout and searchable tag schema for phase artifacts.
- Final inventory ordering/presentation strategy (for example by subsystem or by disposition) as long as it remains auditable.

</decisions>

<specifics>
## Specific Ideas

No specific stylistic references were requested; prioritize auditability and decision traceability for maintainers.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/nockchain-bench/src/main.rs`: central command dispatch and bench/SOL command surface to anchor inventory sweep coverage.
- `crates/nockchain-bench/src/speed_of_light/compat.rs`: existing compatibility adapter layer (`NounSlabCompatExt`, `NounCompatExt`) likely to contain branch-vs-master drift.
- `crates/nockchain-bench/src/speed_of_light/guard/`: existing report model and comparison structures that demonstrate current contract/report patterns.
- `.planning/phases/01-scope-baseline-and-evidence-contract/01-evidence-contract.md`: locked evidence semantics from Phase 1 to preserve.

### Established Patterns
- Typed Rust models and enums (`serde`, `thiserror`) for contract/report artifacts.
- CLI-first workflow with subsystem-oriented module boundaries (`speed_of_light`, `runner`, `scenario`).
- Hard-fail contract validation approach already established in Phase 1 tooling (`scripts/verify_scope_evidence_contract.sh` + `Makefile` target).

### Integration Points
- Inventory findings must map to `nockchain-bench` CLI and SOL module references, then be checked against `nockchain/master` availability.
- PMA-related and other branch-only assumptions should be surfaced from bench code paths where they appear, not inferred from unrelated crates.
- Output artifacts should remain aligned with existing planning/evidence workflow so later provenance (Phase 3) can consume them.

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-master-compatibility-inventory*
*Context gathered: 2026-03-03*
