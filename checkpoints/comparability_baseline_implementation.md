# Comparability Baseline Implementation Checklist

Stable IDs in this checklist are binary gates for Phase 5 comparability baseline implementation and closure.

## Implementation Gates

- [x] V001 Canonical comparability contract scaffold exists at `.planning/phases/05-comparability-verification-baseline/05-comparability-verification-baseline.md` with deterministic section headers.
- [x] V002 Validation matrix schema is locked at `.planning/phases/05-comparability-verification-baseline/05-validation-matrix.tsv` with required reproducibility columns.
- [x] V003 Minimal baseline tuple row exists for `native` + `v0` comparing `master` vs `grafted` with fixed pass-count policy.
- [x] V004 Tuple identity and tuple-purity policy is explicitly present in the canonical contract.
- [x] V005 Contract defines objective verdict and rejection scaffolding needed for downstream verifier/make-gate wiring.

## Reserved For Plan 05-03 Closure

- [ ] V006 Final verifier integration complete (`make comparability-baseline-verify` invokes `./scripts/verify_comparability_baseline.sh` and fails on schema/checklist drift).
- [ ] V007 Closure gate confirms required contract sections and locked policy enums are present exactly once.
- [ ] V008 Closure gate confirms validation matrix rows are complete, non-empty, and tuple identity fields remain deterministic.
- [ ] V009 Closure gate confirms reserved checklist IDs `V006..V010` are enforced and cannot be bypassed by unchecked/missing entries.
- [ ] V010 Closure gate confirms one-command Phase 5 closure evidence is deterministic and auditable through verifier + make-gate enforcement.
