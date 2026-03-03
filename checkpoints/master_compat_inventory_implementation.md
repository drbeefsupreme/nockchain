# Master Compatibility Inventory Implementation Checklist

Stable IDs in this checklist are binary gates for Phase 2 inventory implementation and closure.

## Implementation Gates

- [ ] M001 Canonical artifact exists at `.planning/phases/02-master-compatibility-inventory/02-master-compatibility-inventory.md` with pinned master SHA context.
- [ ] M002 Candidate sweep index exists at `.planning/phases/02-master-compatibility-inventory/02-compat-candidate-index.tsv` and is scoped to `crates/nockchain-bench/src/**`.
- [x] M003 PMA dependencies are explicit inventory candidates (`--pma-persist`, `NOCK_PMA_*`, and `NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE`) and mapped to `dep-001..dep-003` plus sampler PMA assumptions in `dep-009`.
- [x] M004 Branch-only concept dependencies are explicit inventory candidates (NounSpace-style adapters including `NounSpace`, `in_space`, `noun_space`) and linked under `dep-004` with master-evidence-backed replacements.
- [x] M005 Inventory taxonomy enforces allowed dispositions only: `remove|replace-with-master-equivalent|feature-gate|defer` (validated against all `dep-*` rows).

## Final Closure Validation (Reserved For Make Gate)

- [x] M006 Disposition enum lock is machine-enforced by `./scripts/verify_master_compat_inventory.sh` (`remove|replace-with-master-equivalent|feature-gate|defer` only).
- [x] M007 PMA coverage is machine-enforced by `./scripts/verify_master_compat_inventory.sh` requiring PMA-tagged inventory rows.
- [x] M008 Branch-only/NounSpace coverage is machine-enforced by `./scripts/verify_master_compat_inventory.sh` requiring `NounSpace`/`noun_space`/`in_space`-evidenced rows.
- [x] M009 Candidate-link completeness is machine-enforced by `./scripts/verify_master_compat_inventory.sh` requiring every `missing|uncertain` candidate to map to matching inventory `dependency_id` + `finding_id`.
- [x] M010 One-command closure gate is available at `make master-compat-verify` (verifier + checklist hard-fail checks).
