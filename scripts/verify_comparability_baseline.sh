#!/usr/bin/env bash
set -euo pipefail

PHASE_DIR=".planning/phases/05-comparability-verification-baseline"
CONTRACT_FILE="${PHASE_DIR}/05-comparability-verification-baseline.md"
MATRIX_FILE="${PHASE_DIR}/05-validation-matrix.tsv"
RESULTS_TEMPLATE_FILE="${PHASE_DIR}/05-comparability-results-template.md"
CHECKLIST_FILE="checkpoints/comparability_baseline_implementation.md"

failures=0

fail() {
  echo "ERROR: $*" >&2
  failures=$((failures + 1))
}

require_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    fail "required artifact missing: ${file}"
  fi
}

require_pattern() {
  local file="$1"
  local pattern="$2"
  local message="$3"
  if ! rg -q -- "$pattern" "$file"; then
    fail "${message} (${file})"
  fi
}

require_unique_pattern() {
  local file="$1"
  local pattern="$2"
  local message="$3"
  local count
  count="$(rg -c -- "$pattern" "$file" || true)"
  if [[ "$count" -ne 1 ]]; then
    fail "${message} (expected exactly one match, found ${count}) (${file})"
  fi
}

for artifact in "$CONTRACT_FILE" "$MATRIX_FILE" "$RESULTS_TEMPLATE_FILE" "$CHECKLIST_FILE"; do
  require_file "$artifact"
done

# Contract integrity checks: required sections and policy markers.
for required_section in \
  '^## Tuple Identity Contract$' \
  '^### Required Tuple Fields$' \
  '^### Deterministic Tuple Purity Rules$' \
  '^## Critical Metrics$' \
  '^## Verdict Policy$' \
  '^### Allowed Comparator Outcomes$' \
  '^### PASS Conditions$' \
  '^### FAIL Conditions$' \
  '^## Data-Quality Guards$' \
  '^## Baseline Fallback Policy$' \
  '^## Evidence Requirements$' \
  '^### Required Evidence Payload$' \
  '^## Rejection Rules$'; do
  require_unique_pattern "$CONTRACT_FILE" "$required_section" "contract missing required section"
done

for required_marker in \
  'QG-002.*tuple-purity' \
  'QG-006.*baseline-fallback-discipline' \
  'Branch-agnostic baseline fallback .* disallowed for final PASS' \
  'Rejected evaluations MUST be recorded as explicit `FAIL` or `REJECTED` outcomes' \
  'silent row dropping is not allowed'; do
  require_pattern "$CONTRACT_FILE" "$required_marker" "contract missing required policy marker"
done

for outcome in Improvement NoSignificantChange Regression Inconclusive; do
  require_pattern "$CONTRACT_FILE" "^- \`${outcome}\`$" "contract missing allowed comparator outcome '${outcome}'"
done

for rejection_code in RJ-001 RJ-002 RJ-003 RJ-004 RJ-005; do
  require_pattern "$CONTRACT_FILE" "\\| \`${rejection_code}\` \\|" "contract missing rejection rule ${rejection_code}"
done

# Results template checks: required sections, explicit verdict rules, and rejection handling.
for required_section in \
  '^## Final Verdict$' \
  '^## Tuple Verdicts$' \
  '^## Rejected Rows$' \
  '^## Evidence Index$'; do
  require_unique_pattern "$RESULTS_TEMPLATE_FILE" "$required_section" "results template missing required section"
done

require_pattern "$RESULTS_TEMPLATE_FILE" '`VERDICT: PASS \| FAIL`' "results template missing explicit final verdict marker"
require_pattern "$RESULTS_TEMPLATE_FILE" '`verdict` must be explicit for every tuple row' "results template allows implicit tuple verdicts"
require_pattern "$RESULTS_TEMPLATE_FILE" 'no blanks' "results template must explicitly forbid blank tuple verdict fields"
require_pattern "$RESULTS_TEMPLATE_FILE" 'Silent row drops are forbidden\.' "results template missing rejection handling rule"
require_pattern "$RESULTS_TEMPLATE_FILE" '^\| tuple_id \| source_file \| row_selector \| reason_code \| reason_detail \| evidence_ref \|$' "results template missing rejected-row schema table"

for rejection_code in RJ-001 RJ-002 RJ-003 RJ-004 RJ-005; do
  require_pattern "$RESULTS_TEMPLATE_FILE" "- \`${rejection_code}\`:" "results template missing accepted rejection code ${rejection_code}"
done

# Matrix schema + tuple row integrity checks.
header="$(head -n1 "$MATRIX_FILE")"
for required_col in \
  tuple_id \
  env \
  fixture \
  baseline_branch \
  candidate_branch \
  passes \
  enable_checkpointing \
  status \
  notes \
  matrix_command \
  tuple_extract_policy \
  compare_output \
  guard_output \
  verdict; do
  if ! tr '\t' '\n' <<<"$header" | rg -qx -- "$required_col"; then
    fail "validation matrix missing required column '${required_col}'"
  fi
done

declare -A tuple_counts=()
required_tuple_id="native-v0-master-vs-grafted-p5-cpfalse"
row_count=0

while IFS=$'\t' read -r tuple_id env fixture baseline_branch candidate_branch passes enable_checkpointing status notes matrix_command tuple_extract_policy compare_output guard_output verdict; do
  if [[ -z "$tuple_id" ]]; then
    continue
  fi
  row_count=$((row_count + 1))

  tuple_counts["$tuple_id"]=$(( ${tuple_counts["$tuple_id"]:-0} + 1 ))

  for required_field in tuple_id env fixture baseline_branch candidate_branch passes enable_checkpointing matrix_command tuple_extract_policy compare_output guard_output verdict; do
    value="${!required_field}"
    if [[ -z "$value" ]]; then
      fail "${tuple_id}: missing required field '${required_field}'"
    fi
  done

  if [[ "$env" != "native" && "$env" != "docker" ]]; then
    fail "${tuple_id}: invalid env '${env}' (allowed: native|docker)"
  fi

  if [[ "$fixture" != "v0" && "$fixture" != "v1" && "$fixture" != "v2" ]]; then
    fail "${tuple_id}: invalid fixture '${fixture}' (allowed: v0|v1|v2)"
  fi

  if [[ "$enable_checkpointing" != "true" && "$enable_checkpointing" != "false" ]]; then
    fail "${tuple_id}: invalid enable_checkpointing '${enable_checkpointing}' (allowed: true|false)"
  fi

  if ! [[ "$passes" =~ ^[0-9]+$ ]]; then
    fail "${tuple_id}: passes must be an integer (found '${passes}')"
  elif (( passes < 5 )); then
    fail "${tuple_id}: passes must be >= 5 for baseline contract rows (found '${passes}')"
  fi

  if [[ "$candidate_branch" == "$baseline_branch" ]]; then
    fail "${tuple_id}: candidate_branch must differ from baseline_branch"
  fi

  if [[ "$tuple_extract_policy" != *"tuple-pure"* ]]; then
    fail "${tuple_id}: tuple_extract_policy must explicitly mention tuple-pure extraction"
  fi

  if [[ "$compare_output" != *.json ]]; then
    fail "${tuple_id}: compare_output must reference a JSON artifact path"
  fi

  if [[ "$guard_output" != *.json ]]; then
    fail "${tuple_id}: guard_output must reference a JSON artifact path"
  fi
done < <(tail -n +2 "$MATRIX_FILE")

if [[ "$row_count" -eq 0 ]]; then
  fail "validation matrix has no tuple rows"
fi

if [[ "${tuple_counts[$required_tuple_id]:-0}" -ne 1 ]]; then
  fail "required tuple row '${required_tuple_id}' must appear exactly once"
fi

# Checklist ID guard checks: IDs reserved for closure must exist in checklist.
for required_id in V006 V007 V008 V009 V010; do
  require_pattern "$CHECKLIST_FILE" "^- \\[[ xX]\\] ${required_id}\\b" "checklist missing reserved closure ID ${required_id}"
done

if (( failures > 0 )); then
  echo "Comparability baseline verification failed with ${failures} error(s)." >&2
  exit 1
fi

echo "Comparability baseline verification passed."
