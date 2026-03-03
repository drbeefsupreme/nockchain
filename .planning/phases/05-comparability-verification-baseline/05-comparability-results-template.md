# Phase 05 Comparability Results Template

Use this template for every Phase 5 comparability verdict package. Do not remove sections. Empty cells must be filled with `N/A` plus justification.

## Final Verdict

`VERDICT: PASS | FAIL`

- Evaluation timestamp (UTC):
- Candidate commit SHA:
- Baseline commit SHA:
- Matrix run root:
- Contract version/path:
- Notes:

## Tuple Verdicts

| tuple_id | verdict | compare_outcome_summary | guard_status | compare_output | guard_output | tuple_extract_evidence | reviewer_notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `native-v0-master-vs-grafted-p5-cpfalse` | `PASS/FAIL/REJECTED` | `Improvement/NoSignificantChange/Regression/Inconclusive` | `PASS/FAIL` | `path/to/*.compare.json` | `path/to/*.guard.json` | `path/to/tuple-filter-command-or-log` | `required for non-PASS` |

Rules:
- `verdict` must be explicit for every tuple row (no blanks).
- `compare_output` and `guard_output` must point to concrete artifacts.
- Any tuple with `Regression`, `Inconclusive`, or failed guard must not be marked `PASS`.

## Rejected Rows

List every excluded candidate/baseline row that was not used in tuple verdicting. Silent row drops are forbidden.

| tuple_id | source_file | row_selector | reason_code | reason_detail | evidence_ref |
| --- | --- | --- | --- | --- | --- |
| `<tuple-id>` | `path/to/combined_summary.tsv` | `NR=...` or key fields | `RJ-001..RJ-005` | `why rejected` | `path/to/rejection-log` |

Accepted reason codes:
- `RJ-001`: missing required tuple fields
- `RJ-002`: tuple purity violation or identity mismatch
- `RJ-003`: data-quality guard failure
- `RJ-004`: missing required evidence payload field(s)
- `RJ-005`: invalid verdict enum/value

## Evidence Index

| evidence_id | description | path | required |
| --- | --- | --- | --- |
| `E-CMD-MATRIX` | exact matrix command transcript | `path/to/matrix-command.txt` | yes |
| `E-CMD-COMPARE` | exact compare command transcript | `path/to/compare-command.txt` | yes |
| `E-CMD-GUARD` | exact guard command transcript | `path/to/guard-command.txt` | yes |
| `E-TSV-CAND` | tuple-pure candidate TSV | `path/to/candidate.tuple.tsv` | yes |
| `E-TSV-BASE` | tuple-pure baseline TSV | `path/to/baseline.tuple.tsv` | yes |
| `E-JSON-COMPARE` | comparator JSON report | `path/to/*.compare.json` | yes |
| `E-JSON-GUARD` | guard JSON report | `path/to/*.guard.json` | yes |
| `E-MANIFEST-CAND` | candidate run manifest | `path/to/candidate/manifest.json` | yes |
| `E-MANIFEST-BASE` | baseline run manifest | `path/to/baseline/manifest.json` | yes |
| `E-FALLBACK-APPROVAL` | fallback exception approval record (if used) | `path/to/approval.md` | conditional |
