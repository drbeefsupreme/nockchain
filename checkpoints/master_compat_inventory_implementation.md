# Master Compatibility Inventory Implementation Checklist

Stable IDs in this checklist are binary gates for Phase 2 inventory implementation and closure.

## Implementation Gates

- [ ] M001 Canonical artifact exists at `.planning/phases/02-master-compatibility-inventory/02-master-compatibility-inventory.md` with pinned master SHA context.
- [ ] M002 Candidate sweep index exists at `.planning/phases/02-master-compatibility-inventory/02-compat-candidate-index.tsv` and is scoped to `crates/nockchain-bench/src/**`.
- [x] M003 PMA dependencies are explicit inventory candidates (`--pma-persist`, `NOCK_PMA_*`, and `NOCK_STREAMING_CHECKPOINT_CHUNK_SIZE`) and mapped to `dep-001..dep-003` plus sampler PMA assumptions in `dep-009`.
- [ ] M004 Branch-only concept dependencies are explicit inventory candidates (NounSpace-style adapters including `NounSpace`, `in_space`, `noun_space`).
- [ ] M005 Inventory taxonomy enforces allowed dispositions only: `remove|replace-with-master-equivalent|feature-gate|defer`.

## Final Closure Validation (Reserved For Make Gate)

- [ ] M006 Default-to-defer rule is applied when no concrete master equivalent is evidenced.
- [x] M007 Optional branch-only behavior is biased toward `remove` unless a stronger rationale is documented (all PMA-tagged rows currently classified `remove` with explicit rationale).
- [ ] M008 Positive controls are retained and marked as non-gaps (for example `heaviest-chain-blocks-range`).
- [ ] M009 Every candidate row has deterministic ID linkage and pinned-SHA `branch_context`.
- [ ] M010 Closure review confirms runtime-path and test-only sections are both covered or explicitly marked N/A.
