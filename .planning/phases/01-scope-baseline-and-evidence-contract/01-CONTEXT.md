# Phase 1: Scope Baseline And Evidence Contract - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Define and lock the analysis scope and evidence contract used to identify `nockchain-bench` dependencies that are incompatible with `nockchain/master`. This phase sets rules for how findings are counted and evidenced; it does not execute the compatibility inventory itself.

</domain>

<decisions>
## Implementation Decisions

### Scope Boundary Rules
- Scope surface is **Rust code only** for compatibility findings.
- Runtime dependency tracing should include **direct references plus explicit behavior assumptions encoded in Rust bench logic**.
- Test-only incompatibilities should be tracked in a **separate section**, not mixed with runtime-path findings.
- Non-Rust artifacts (shell outputs, manifests, generated reports) are **not used as evidence inputs** for finding inclusion.

### Evidence Record Contract
- Use **one atomic record per incompatibility finding**.
- Mandatory fields per record: **file path, symbol/API reference, branch context, impact statement**.
- Keep a **single canonical Markdown artifact** as the source of truth.
- Evidence validation is **hard-fail on missing required fields**.

### Match Rules Against Master
- "Absent in master" is determined using **exact missing references** as the primary rule.
- Comparisons must use a **single pinned `nockchain/master` commit SHA** during this phase.
- Renamed-but-equivalent interfaces are recorded as **replaceable gaps**.
- Branch-only env/config toggles (including PMA-related toggles) are treated as **compatibility-relevant incompatibilities**.

### Uncertainty Handling
- Every finding carries **three-level confidence** (high/medium/low).
- Low-confidence findings live in a **separate unresolved section**.
- Conflicting evidence must be **escalated for manual resolution**.
- Phase closure is blocked if **high-impact unresolved findings** remain.

### Claude's Discretion
- Exact field naming convention and heading taxonomy inside the canonical Markdown artifact.
- How confidence tags are rendered (text labels vs compact badges) as long as semantics stay high/medium/low.

</decisions>

<specifics>
## Specific Ideas

- Keep the evidence contract strict in the style of existing provenance validation behavior.
- Prefer clean, auditable records over broad narrative summaries.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/nockchain-bench/src/speed_of_light/guard/provenance.rs`: strict schema + validation pattern that can inform Phase 1 evidence contract behavior.
- `crates/nockchain-bench/src/speed_of_light/guard/model.rs`: existing enum/config structures useful for consistent classification vocabulary.

### Established Patterns
- Benchmark integrity already relies on deterministic metadata capture and validation (e.g., baseline/provenance flows), so strict required-field checks fit existing project direction.
- Bench modules are organized by domain (`runner`, `scenario`, `speed_of_light`), enabling scoped compatibility scanning by module boundary.

### Integration Points
- Phase 2 compatibility inventory should connect to `crates/nockchain-bench/src/main.rs` command surfaces and `speed_of_light/*` modules first.
- PMA-related compatibility checks should inspect both runtime-facing sampler code (`crates/nockchain-bench/src/sampler/buckets.rs`) and bench-side config/toggle usage.

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---
*Phase: 01-scope-baseline-and-evidence-contract*
*Context gathered: 2026-03-03*
