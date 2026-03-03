# Phase 3: Provenance And Divergence Timeline - Context

**Gathered:** 2026-03-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Explain when and where incompatibilities from Phase 2 entered branch history, classify each divergence as inherited vs local vs mixed where possible, and produce a concise SOL-relevant divergence timeline. This phase clarifies provenance and chronology; it does not execute grafting or add new benchmark capabilities.

</domain>

<decisions>
## Implementation Decisions

### Provenance Trace Scope
- Trace provenance primarily at `dependency_id` level, and link all related `finding_id` entries to each provenance record.
- For each traced dependency, capture origin commit plus major pivot commits (not every touching commit).
- Branch horizon must explicitly include `bitemyapp/ag2-opt-persistence-madvise-checkpoint-chaff-pma-gc-checkpoint-streaming` plus current branch ancestry.
- Timeline/provenance entries must include both dependency and finding identifiers so Phase 2 artifacts remain directly navigable.

### Evidence And Uncertainty Handling
- Reuse locked confidence values `high|medium|low` from Phase 1 evidence contract.
- Keep unresolved provenance in a dedicated unresolved section (not hidden inline only).
- Require structured evidence fields per event: commit SHA(s), branch context, and explicit rationale for divergence mapping.
- On conflicting evidence, escalate and keep entry unresolved until adjudicated.

### Timeline Shape And Readability
- Timeline should be organized as thematic buckets, not strict commit-by-commit prose.
- Within that structure, maintain chronological ordering of events.
- Focus on major divergence events affecting SOL behavior (concise, high-signal timeline).
- Include inline dependency/finding IDs and key evidence references in each event.

### Divergence Attribution Rules
- Use explicit classes: `Inherited`, `Local`, `Mixed`.
- `Inherited`: evidence ties origin to historical branch ancestry, including the named historical branch line.
- `Local`: evidence indicates introduction after current branch point by commits unique to current branch evolution.
- Ambiguous cases use `Mixed` when multi-origin is evidenced; otherwise unresolved with confidence and rationale.

### Claude's Discretion
- Exact presentation format inside thematic buckets (table-first vs prose-first) as long as chronology and traceability remain explicit.
- Final threshold for what qualifies as a "major" event, provided PROV-03 remains concise and auditable.

</decisions>

<specifics>
## Specific Ideas

- Keep provenance outputs directly cross-linked to Phase 2 IDs (`dependency_id`, `finding_id`) rather than narrative-only descriptions.
- Prefer evidence-backed chronology over discovery-order storytelling.

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `.planning/phases/02-master-compatibility-inventory/02-master-compatibility-inventory.md`: canonical dependency/finding inventory to serve as provenance input surface.
- `.planning/phases/02-master-compatibility-inventory/02-compat-candidate-index.tsv`: deterministic candidate IDs and dependency linkages to anchor trace coverage.
- `.planning/phases/01-scope-baseline-and-evidence-contract/01-evidence-contract.md`: locked evidence semantics (`confidence`, required fields, unresolved handling) to preserve in Phase 3.
- `scripts/verify_master_compat_inventory.sh`: hard-fail validation style that can inform provenance artifact quality gates.

### Established Patterns
- Documentation artifacts in `.planning/phases/**` use strict schemas, explicit IDs, and deterministic verifier workflows.
- Compatibility analysis already uses pinned-master branch context and evidence-first assertions, which Phase 3 should extend to commit provenance.
- Closure quality is enforced through one-command Make/script gates rather than reviewer discretion.

### Integration Points
- Phase 3 provenance outputs should map directly from Phase 2 dependency/finding rows to commit/branch evidence records.
- Provenance classification must remain compatible with downstream Phase 4 graft planning (remove/replace/defer actions need origin clarity).
- Timeline content should focus on SOL-impacting divergence events surfaced from existing bench modules and inventory tags.

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-provenance-and-divergence-timeline*
*Context gathered: 2026-03-03*
