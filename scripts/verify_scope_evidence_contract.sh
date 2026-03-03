#!/usr/bin/env bash
set -euo pipefail

PHASE_DIR=".planning/phases/01-scope-baseline-and-evidence-contract"
SCOPE_FILE="${PHASE_DIR}/01-scope-boundary.md"
MASTER_FILE="${PHASE_DIR}/01-master-target.md"
CONTRACT_FILE="${PHASE_DIR}/01-evidence-contract.md"
LEDGER_FILE="${PHASE_DIR}/01-findings-ledger.md"

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

for artifact in "$SCOPE_FILE" "$MASTER_FILE" "$CONTRACT_FILE" "$LEDGER_FILE"; do
  require_file "$artifact"
done

require_pattern "$SCOPE_FILE" '^## In Scope$' "scope boundary missing 'In Scope' marker"
require_pattern "$SCOPE_FILE" '^## Out of Scope$' "scope boundary missing 'Out of Scope' marker"
require_pattern "$SCOPE_FILE" '^## Scope Gate Checklist$' "scope boundary missing scope gate checklist"
require_pattern "$SCOPE_FILE" 'crates/nockchain-bench/src/\*\*' "scope boundary missing bench Rust path gate"

require_pattern "$MASTER_FILE" '^## Target Record$' "master target missing target record section"
require_pattern "$MASTER_FILE" 'Preferred remote/ref:' "master target missing preferred remote/ref"
require_pattern "$MASTER_FILE" 'Pinned SHA:' "master target missing pinned SHA"
require_pattern "$MASTER_FILE" '^## Pinning Rules$' "master target missing pinning rules"

require_pattern "$CONTRACT_FILE" '^## Atomic Finding Record$' "evidence contract missing atomic record section"
require_pattern "$CONTRACT_FILE" '^## Required Enums$' "evidence contract missing required enum section"
require_pattern "$CONTRACT_FILE" '^### `match_rule` \(locked\)$' "evidence contract missing locked match_rule enum"
require_pattern "$CONTRACT_FILE" '^### `confidence` \(locked\)$' "evidence contract missing locked confidence enum"
require_pattern "$CONTRACT_FILE" '^## Hard-Fail Requirements$' "evidence contract missing hard-fail requirements"
require_pattern "$CONTRACT_FILE" '^## Unresolved and Low-Confidence Handling$' "evidence contract missing unresolved handling section"
require_pattern "$CONTRACT_FILE" '^## Escalation Rule$' "evidence contract missing escalation rule"

require_pattern "$LEDGER_FILE" '^## Runtime-Path Incompatibilities$' "ledger missing runtime-path section"
require_pattern "$LEDGER_FILE" '^## Test-Only Incompatibilities$' "ledger missing test-only section"
require_pattern "$LEDGER_FILE" '^## Unresolved Low-Confidence Findings$' "ledger missing unresolved section"

header_count="$(rg -c '^\| finding_id \| file_path \| symbol_or_api \| branch_context \| impact_statement \| confidence \| match_rule \| impact_level \| status \| notes \|$' "$LEDGER_FILE")"
if [[ "$header_count" -lt 3 ]]; then
  fail "ledger must contain full required header row for runtime/test-only/unresolved sections"
fi

pinned_sha="$(sed -nE 's/^- Pinned SHA: `?([0-9a-f]{40})`?$/\1/p' "$MASTER_FILE" | head -n1)"
if [[ -z "$pinned_sha" ]]; then
  fail "unable to parse pinned SHA from ${MASTER_FILE}"
fi

while IFS=$'\t' read -r section finding_id file_path symbol_or_api branch_context impact_statement confidence match_rule impact_level status notes; do
  if [[ -z "$file_path" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: file_path"
  fi
  if [[ -z "$symbol_or_api" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: symbol_or_api"
  fi
  if [[ -z "$branch_context" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: branch_context"
  fi
  if [[ -z "$impact_statement" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: impact_statement"
  fi
  if [[ -z "$confidence" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: confidence"
  fi
  if [[ -z "$match_rule" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: match_rule"
  fi
  if [[ -z "$impact_level" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: impact_level"
  fi

  if [[ "$confidence" != "high" && "$confidence" != "medium" && "$confidence" != "low" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} invalid confidence '${confidence}' (allowed: high|medium|low)"
  fi

  if [[ "$match_rule" != "exact_missing_ref" && "$match_rule" != "replaceable_gap" && "$match_rule" != "branch_env_config_toggle" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} invalid match_rule '${match_rule}' (allowed: exact_missing_ref|replaceable_gap|branch_env_config_toggle)"
  fi

  if [[ "$branch_context" != *"$pinned_sha"* ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} branch_context must include pinned SHA ${pinned_sha}"
  fi

  if [[ "$match_rule" == "branch_env_config_toggle" ]]; then
    toggle_blob="$(printf '%s %s %s %s' "$symbol_or_api" "$branch_context" "$impact_statement" "$notes" | tr '[:upper:]' '[:lower:]')"
    if ! grep -Eq '(pma|env|config|toggle)' <<<"$toggle_blob"; then
      fail "${section}:${finding_id:-<missing finding_id>} branch_env_config_toggle row must include PMA/env/config marker evidence"
    fi
  fi

  if [[ "$section" == "unresolved" && "$impact_level" == "high" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} unresolved high-impact finding blocks phase closure"
  fi
done < <(
  awk '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    BEGIN { section = "" }
    /^## Runtime-Path Incompatibilities$/ { section = "runtime"; next }
    /^## Test-Only Incompatibilities$/ { section = "test-only"; next }
    /^## Unresolved Low-Confidence Findings$/ { section = "unresolved"; next }
    section != "" && /^\|/ {
      n = split($0, raw, "|")
      m = 0
      for (i = 2; i <= n - 1; i++) {
        m++
        cols[m] = trim(raw[i])
      }

      is_sep = 1
      for (i = 1; i <= m; i++) {
        tmp = cols[i]
        gsub(/[-: ]/, "", tmp)
        if (tmp != "") {
          is_sep = 0
          break
        }
      }
      if (is_sep) {
        next
      }

      if (tolower(cols[1]) == "finding_id") {
        next
      }

      all_empty = 1
      for (i = 1; i <= m; i++) {
        if (cols[i] != "") {
          all_empty = 0
          break
        }
      }
      if (all_empty) {
        next
      }

      for (i = m + 1; i <= 10; i++) {
        cols[i] = ""
      }

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", section, cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9], cols[10]
    }
  ' "$LEDGER_FILE"
)

if (( failures > 0 )); then
  echo "Scope/evidence contract verification failed with ${failures} error(s)." >&2
  exit 1
fi

echo "Scope/evidence contract verification passed."
