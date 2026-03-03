# Phase 05 Comparability Verification Baseline

## Metadata

- Phase: `05-comparability-verification-baseline`
- Purpose: lock a deterministic comparability contract before verifier and make-gate wiring.
- Requirements scaffolded here: `VERI-01`, `VERI-02`, `VERI-03`.
- Canonical matrix source: `.planning/phases/05-comparability-verification-baseline/05-validation-matrix.tsv`.

## Tuple Identity Contract

### Required Tuple Fields

| Field | Type | Allowed Values | Required |
| --- | --- | --- | --- |
| `tuple_id` | string | non-empty | yes |
| `env` | enum | `native`, `docker` | yes |
| `fixture` | enum | `v0`, `v1`, `v2` | yes |
| `baseline_branch` | string | non-empty | yes |
| `candidate_branch` | string | non-empty and not equal to `baseline_branch` | yes |
| `passes` | integer | `>=5` for baseline contract rows | yes |
| `enable_checkpointing` | enum | `true`, `false` | yes |

### Deterministic Tuple Purity Rules

1. A comparability verdict MUST evaluate one tuple identity at a time.
2. Candidate and baseline summary inputs MUST contain rows only for the selected tuple.
3. Tuple identity drift across `env`, `fixture`, branch labels, `passes`, or `enable_checkpointing` is rejection-worthy (see Rejection Rules).

## Verdict Policy

### Allowed Comparator Outcomes

- `Improvement`
- `NoSignificantChange`
- `Regression`
- `Inconclusive`

### Metric Classes

| Class | Contract Role | Default Policy |
| --- | --- | --- |
| `critical` | phase-gating metrics for PASS/FAIL | any `Regression` or `Inconclusive` fails verdict |
| `informational` | reviewer context metrics | does not independently fail verdict |

### PASS/FAIL Contract Scaffold

1. PASS requires data-quality guards to pass and no critical-metric `Regression`/`Inconclusive`.
2. FAIL occurs on any critical-metric `Regression`, any critical-metric `Inconclusive`, or any data-quality guard failure.
3. Final metric membership for `critical` and `informational` classes is populated by a later Phase 5 plan without changing this section schema.

## Data-Quality Guards

| Guard ID | Category | Description | Severity |
| --- | --- | --- | --- |
| `QG-001` | schema-integrity | required columns exist and numeric fields are parseable finite values | fail |
| `QG-002` | tuple-purity | candidate and baseline inputs are single-tuple and identity-matched | fail |
| `QG-003` | runtime-success | rows indicate successful execution (`exit_status=0`, no failed pokes) | fail |
| `QG-004` | sample-sufficiency | row counts satisfy baseline minimum sample policy | fail |
| `QG-005` | provenance-parity | manifests/configs show compatible run environment assumptions | fail |
| `QG-006` | baseline-fallback-discipline | fallback use is explicit and policy-compliant | fail |

## Evidence Requirements

### Required Evidence Payload

| Field | Description |
| --- | --- |
| `evaluation_timestamp_utc` | ISO-8601 UTC timestamp for the verdict run |
| `tuple_id` | tuple identity used for the verdict |
| `candidate_summary_path` | path to tuple-pure candidate summary TSV |
| `baseline_summary_path` | path to tuple-pure baseline summary TSV |
| `guard_report_json` | machine-readable guard output path |
| `compare_report_json` | machine-readable comparison output path |
| `guard_command` | exact guard command executed |
| `compare_command` | exact comparison command executed |
| `baseline_commit_sha` | source SHA for baseline branch binary/context |
| `candidate_commit_sha` | source SHA for candidate branch binary/context |

### Reporting Requirements

1. Evidence payload fields are mandatory for accepted verdicts.
2. Missing payload fields are rejection-worthy and MUST produce explicit rejection reasons.

## Rejection Rules

| Rejection ID | Trigger | Required Handling |
| --- | --- | --- |
| `RJ-001` | missing required tuple fields | reject verdict, report missing fields |
| `RJ-002` | tuple purity violation or mixed tuple input | reject verdict, report conflicting identities |
| `RJ-003` | data-quality guard failure | reject verdict, include failing guard IDs |
| `RJ-004` | missing required evidence payload fields | reject verdict, enumerate missing evidence keys |
| `RJ-005` | invalid verdict enum outside policy | reject verdict, report invalid value |

Rejected evaluations MUST be recorded as explicit `FAIL` or `REJECTED` outcomes with machine-readable reason codes; silent row dropping is not allowed.
