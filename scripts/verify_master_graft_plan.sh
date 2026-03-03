#!/usr/bin/env bash
set -euo pipefail

PHASE_DIR=".planning/phases/04-master-graft-execution-plan"
RUNBOOK_FILE="${PHASE_DIR}/04-master-graft-execution-plan.md"
MATRIX_FILE="${PHASE_DIR}/04-graft-dependency-matrix.tsv"
CHECKLIST_FILE="checkpoints/master_graft_plan_implementation.md"

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

section_has_subheading() {
  local section_pattern="$1"
  local subheading="$2"
  awk -v section_pattern="$section_pattern" -v subheading="$subheading" '
    $0 ~ section_pattern { in_section = 1; next }
    $0 ~ /^## / && in_section { exit }
    in_section && $0 == subheading { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "$RUNBOOK_FILE"
}

for artifact in "$RUNBOOK_FILE" "$MATRIX_FILE" "$CHECKLIST_FILE"; do
  require_file "$artifact"
done

for section_num in 0 1 2 3 4 5; do
  section_pattern="^## R${section_num} "
  require_pattern "$RUNBOOK_FILE" "$section_pattern" "runbook missing required R${section_num} section"

  for subheading in "### Preconditions" "### Commands" "### Expected Output" "### Risk Notes" "### Rollback"; do
    if ! section_has_subheading "$section_pattern" "$subheading"; then
      fail "runbook section R${section_num} missing subheading '${subheading}'"
    fi
  done
done

require_pattern "$RUNBOOK_FILE" '^## Stop-The-Line Criteria$' "runbook missing stop-the-line criteria section"
require_pattern "$RUNBOOK_FILE" '^## Rollback Policy$' "runbook missing rollback policy section"
require_pattern "$RUNBOOK_FILE" '^## DEP-005 Decision Gate$' "runbook missing DEP-005 decision gate section"
require_pattern "$RUNBOOK_FILE" '^### Outcome A:' "runbook missing DEP-005 Outcome A"
require_pattern "$RUNBOOK_FILE" '^### Outcome B:' "runbook missing DEP-005 Outcome B"
require_pattern "$RUNBOOK_FILE" '^### Outcome C:' "runbook missing DEP-005 Outcome C"
require_pattern "$RUNBOOK_FILE" '^## Unresolved Decision Gate Requirements$' "runbook missing unresolved decision gate requirements section"
require_pattern "$RUNBOOK_FILE" 'No unresolved item may be closed implicitly' "runbook missing explicit unresolved-handling guard"
require_pattern "$RUNBOOK_FILE" 'exactly one outcome is recorded with approver and timestamp' "runbook missing DEP-005 outcome closure condition"

header="$(head -n1 "$MATRIX_FILE")"
for required_col in \
  dependency_id \
  execution_step_id \
  action \
  exact_target_files \
  master_reference \
  risk_note \
  rollback_point \
  verification_command \
  provenance_class \
  status \
  notes; do
  if ! tr '\t' '\n' <<<"$header" | rg -qx -- "$required_col"; then
    fail "dependency matrix missing required column '${required_col}'"
  fi
done

declare -A required_deps=(
  ["DEP-001"]=1
  ["DEP-002"]=1
  ["DEP-003"]=1
  ["DEP-004"]=1
  ["DEP-005"]=1
  ["DEP-006"]=1
  ["DEP-007"]=1
  ["DEP-008"]=1
  ["DEP-009"]=1
)
declare -A dep_counts=()

dep005_found=0
dep005_status=""
dep005_notes=""
dep005_action=""

while IFS=$'\t' read -r dependency_id execution_step_id action exact_target_files master_reference risk_note rollback_point verification_command provenance_class status notes; do
  if [[ -z "$dependency_id" ]]; then
    continue
  fi

  if [[ -z "${required_deps[$dependency_id]:-}" ]]; then
    fail "unexpected dependency row '${dependency_id}' (expected only DEP-001..DEP-009)"
    continue
  fi

  dep_counts["$dependency_id"]=$(( ${dep_counts["$dependency_id"]:-0} + 1 ))

  if [[ -z "$execution_step_id" ]]; then
    fail "${dependency_id}: missing execution_step_id"
  fi
  if [[ -z "$action" ]]; then
    fail "${dependency_id}: missing action"
  fi
  if [[ -z "$exact_target_files" ]]; then
    fail "${dependency_id}: missing exact_target_files"
  fi
  if [[ -z "$master_reference" ]]; then
    fail "${dependency_id}: missing master_reference"
  fi
  if [[ -z "$risk_note" ]]; then
    fail "${dependency_id}: missing risk_note"
  fi
  if [[ -z "$rollback_point" ]]; then
    fail "${dependency_id}: missing rollback_point"
  fi
  if [[ -z "$verification_command" ]]; then
    fail "${dependency_id}: missing verification_command"
  fi
  if [[ -z "$status" ]]; then
    fail "${dependency_id}: missing status"
  fi
  if [[ -z "$notes" ]]; then
    fail "${dependency_id}: missing notes"
  fi

  if [[ "$action" != "remove" && "$action" != "replace-with-master-equivalent" && "$action" != "feature-gate" && "$action" != "defer" ]]; then
    fail "${dependency_id}: invalid action '${action}' (allowed: remove|replace-with-master-equivalent|feature-gate|defer)"
  fi

  if [[ "$status" == "unresolved" ]] && ! grep -Eqi 'outcome|decision|gate|approver' <<<"$notes"; then
    fail "${dependency_id}: unresolved handling is implicit; notes must include explicit decision/outcome guidance"
  fi

  if [[ "$dependency_id" == "DEP-005" ]]; then
    dep005_found=1
    dep005_status="$status"
    dep005_notes="$notes"
    dep005_action="$action"
  fi
done < <(tail -n +2 "$MATRIX_FILE")

for dep_id in "${!required_deps[@]}"; do
  count="${dep_counts[$dep_id]:-0}"
  if [[ "$count" -ne 1 ]]; then
    fail "${dep_id} must appear exactly once in dependency matrix (found ${count})"
  fi
done

if [[ "$dep005_found" -ne 1 ]]; then
  fail "dependency matrix missing DEP-005 row"
else
  if ! grep -Eq 'Outcome A|Outcome B|Outcome C|Outcome A/B/C' <<<"$dep005_notes"; then
    fail "DEP-005 notes must explicitly reference decision outcomes (Outcome A/B/C)"
  fi
  if [[ "$dep005_status" == "unresolved" && "$dep005_action" != "defer" && "$dep005_action" != "feature-gate" ]]; then
    fail "DEP-005 unresolved row must remain behind defer/feature-gate action until decision is explicit"
  fi
fi

if (( failures > 0 )); then
  echo "Master graft plan verification failed with ${failures} error(s)." >&2
  exit 1
fi

echo "Master graft plan verification passed."
