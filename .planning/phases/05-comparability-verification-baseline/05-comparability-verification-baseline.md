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
4. Tuple extraction must be deterministic and auditable; accepted verdicts MUST reference exact filter criteria used to produce tuple-pure candidate/baseline inputs.

## Critical Metrics

These metrics are phase-gating. Final PASS is not allowed when any critical metric reports `Regression` or `Inconclusive`.

| Metric | Comparator Direction | Why Critical |
| --- | --- | --- |
| `throughput_blocks_s` | higher is better | captures effective SOL replay throughput and is the primary speed indicator |
| `avg_per_block_ms` | lower is better | normalizes latency per block and detects per-unit slowdowns |
| `peak_rss_mib` | lower is better | protects against gross memory regressions under equivalent tuple conditions |
| `p95_rss_mib` | lower is better | guards sustained high-memory behavior beyond single-sample spikes |
| `failed_pokes` | lower is better (target `0`) | enforces runtime correctness; non-zero indicates unstable benchmark execution |

Informational metrics may be reported but do not override critical verdict outcomes.

## Verdict Policy

### Allowed Comparator Outcomes

- `Improvement`
- `NoSignificantChange`
- `Regression`
- `Inconclusive`

### Statistical Verdict Interpretation

| Comparator Outcome | Interpretation | Verdict Impact |
| --- | --- | --- |
| `Improvement` | candidate materially outperforms baseline for the metric | contributes toward PASS when all other gates also pass |
| `NoSignificantChange` | no statistically significant difference | acceptable for PASS when all other gates also pass |
| `Regression` | candidate materially underperforms baseline | hard FAIL when metric is critical |
| `Inconclusive` | comparison lacks sufficient confidence or samples | hard FAIL when metric is critical |

### PASS Conditions

All conditions below are mandatory:

1. Guard run is successful and all fail-severity data-quality guards pass.
2. Candidate and baseline comparison inputs are tuple-pure and identity-matched (`env`, `fixture`, branches, `passes`, checkpointing mode).
3. Every critical metric result is either `Improvement` or `NoSignificantChange`.
4. No critical metric result is `Regression` or `Inconclusive`.
5. Evidence payload is complete (commands, tuple IDs, report paths, SHAs, timestamp).
6. Baseline fallback policy is satisfied (see Baseline Fallback Policy).

### FAIL Conditions

Any single condition below is sufficient for FAIL:

1. Any fail-severity data-quality guard fails.
2. Tuple-purity violation, mixed tuple rows, or tuple identity mismatch is detected.
3. Any critical metric returns `Regression`.
4. Any critical metric returns `Inconclusive`.
5. Runtime-success guard detects failed execution (`exit_status != 0` or failed pokes present).
6. Required evidence payload fields are missing.
7. Baseline fallback usage is not explicitly approved/documented per policy.

## Data-Quality Guards

| Guard ID | Category | Description | Severity |
| --- | --- | --- | --- |
| `QG-001` | schema-integrity | required columns exist and numeric fields are parseable finite values | fail |
| `QG-002` | tuple-purity | candidate and baseline inputs are single-tuple and identity-matched | fail |
| `QG-003` | runtime-success | rows indicate successful execution (`exit_status=0`, no failed pokes) | fail |
| `QG-004` | sample-sufficiency | row counts satisfy baseline minimum sample policy | fail |
| `QG-005` | provenance-parity | manifests/configs show compatible run environment assumptions | fail |
| `QG-006` | baseline-fallback-discipline | fallback use is explicit and policy-compliant | fail |

## Baseline Fallback Policy

1. Branch-agnostic baseline fallback (`env+fixture` fallback that ignores baseline branch identity) is disallowed for final PASS by default.
2. If fallback is used for any tuple, the run outcome is FAIL unless explicit maintainer approval is recorded before verdict acceptance.
3. Any approved exception MUST include:
   - approver identity and UTC timestamp,
   - reason fallback was unavoidable,
   - impacted tuple IDs,
   - evidence that fallback does not hide branch-mixing risk.
4. Approved fallback exceptions still require all other PASS conditions to hold.

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
