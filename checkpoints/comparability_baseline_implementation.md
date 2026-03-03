# Comparability Baseline Implementation Checklist

Stable IDs in this checklist are binary gates for Phase 5 comparability baseline implementation and closure.

## Implementation Gates

- [x] V001 Canonical comparability contract includes populated `## Critical Metrics`, `### PASS Conditions`, and `### FAIL Conditions` sections at `.planning/phases/05-comparability-verification-baseline/05-comparability-verification-baseline.md`.
- [x] V002 Canonical contract includes explicit `## Data-Quality Guards` and `## Baseline Fallback Policy` rules with fallback disallowed for final PASS unless pre-approved and documented.
- [x] V003 Validation matrix schema at `.planning/phases/05-comparability-verification-baseline/05-validation-matrix.tsv` includes deterministic `matrix_command`, `compare_output`, `guard_output`, and `verdict` columns.
- [x] V004 At least one canonical tuple row (`native-v0-master-vs-grafted-p5-cpfalse`) is populated with tuple extraction policy and concrete compare/guard artifact paths.
- [x] V005 Results template exists at `.planning/phases/05-comparability-verification-baseline/05-comparability-results-template.md` with mandatory `Final Verdict`, `Tuple Verdicts`, `Rejected Rows`, and `Evidence Index` sections.

## Reserved For Plan 05-03 Closure

- [ ] V006 `make comparability-baseline-verify` invokes `./scripts/verify_comparability_baseline.sh` and exits non-zero on schema/checklist drift.
- [ ] V007 Verifier asserts required contract section headers and policy enums exist exactly once (`PASS/FAIL`, guard IDs, fallback policy requirements).
- [ ] V008 Verifier asserts validation matrix required columns are present and each tuple row has non-empty deterministic identity/evidence fields.
- [ ] V009 Verifier asserts checklist IDs `V006..V010` all exist and remain unchecked until closure criteria are met.
- [ ] V010 One-command closure run produces auditable evidence output proving verifier + make-gate enforcement succeeded deterministically.
