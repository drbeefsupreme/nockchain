# Provenance Timeline Implementation Checklist

Stable IDs in this checklist are binary gates for Phase 3 provenance implementation and closure.

## Implementation Gates

- [x] P001 Canonical artifact exists at `.planning/phases/03-provenance-and-divergence-timeline/03-provenance-and-divergence-timeline.md` with pinned-master context and branch horizon.
- [x] P002 Provenance workspace exists at `.planning/phases/03-provenance-and-divergence-timeline/03-provenance-evidence.tsv` with one seeded row per Phase 2 dependency.
- [x] P003 Every provenance row links a `dependency_id` to one or more Phase 2 `finding_ids`.
- [x] P004 Attribution taxonomy is locked to `Inherited|Local|Mixed` across canonical artifact and machine-readable rows.
- [x] P005 Canonical artifact includes visible `Unresolved Provenance` and thematic timeline sections for deterministic review.

## Final Closure Validation (Verifier + Make Gate Complete)

- [x] P006 `scripts/verify_provenance_timeline.sh` hard-fails when required provenance TSV schema columns are missing.
- [x] P007 `scripts/verify_provenance_timeline.sh` hard-fails when Phase 2 missing/uncertain dependencies or finding lineage do not map to Phase 3 provenance rows.
- [x] P008 `scripts/verify_provenance_timeline.sh` hard-fails when resolved-row `classification` or row-level `confidence`/`status` values drift from locked enums.
- [x] P009 `scripts/verify_provenance_timeline.sh` hard-fails when timeline event rows lack commit SHA plus dependency/finding traceability.
- [x] P010 `make provenance-timeline-verify` provides the one-command closure gate and fails if `P006..P010` are missing/unchecked.
