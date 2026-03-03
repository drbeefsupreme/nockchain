# Provenance Timeline Implementation Checklist

Stable IDs in this checklist are binary gates for Phase 3 provenance implementation and closure.

## Implementation Gates

- [ ] P001 Canonical artifact exists at `.planning/phases/03-provenance-and-divergence-timeline/03-provenance-and-divergence-timeline.md` with pinned-master context and branch horizon.
- [ ] P002 Provenance workspace exists at `.planning/phases/03-provenance-and-divergence-timeline/03-provenance-evidence.tsv` with one seeded row per Phase 2 dependency.
- [ ] P003 Every provenance row links a `dependency_id` to one or more Phase 2 `finding_ids`.
- [ ] P004 Attribution taxonomy is locked to `Inherited|Local|Mixed` across canonical artifact and machine-readable rows.
- [ ] P005 Canonical artifact includes visible `Unresolved Provenance` and thematic timeline sections for deterministic review.

## Final Closure Validation (Reserved For Verifier + Make Gate)

- [ ] P006 Provenance verifier enforces required schema columns in `03-provenance-evidence.tsv`.
- [ ] P007 Provenance verifier enforces dependency/finding lineage completeness against Phase 2 IDs.
- [ ] P008 Provenance verifier enforces attribution taxonomy and confidence/status enum validity.
- [ ] P009 Provenance verifier enforces timeline major-event traceability to dependency/finding identifiers.
- [ ] P010 One-command closure gate is available at `make provenance-timeline-verify` (verifier + checklist hard-fail checks).
