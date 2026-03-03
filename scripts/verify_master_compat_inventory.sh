#!/usr/bin/env bash
set -euo pipefail

PHASE_DIR=".planning/phases/02-master-compatibility-inventory"
INVENTORY_FILE="${PHASE_DIR}/02-master-compatibility-inventory.md"
CANDIDATE_FILE="${PHASE_DIR}/02-compat-candidate-index.tsv"
MASTER_TARGET_FILE=".planning/phases/01-scope-baseline-and-evidence-contract/01-master-target.md"

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

normalize_dep_id() {
  local dep_id="$1"
  printf '%s' "$dep_id" | tr '[:upper:]' '[:lower:]'
}

for artifact in "$INVENTORY_FILE" "$CANDIDATE_FILE" "$MASTER_TARGET_FILE"; do
  require_file "$artifact"
done

require_pattern "$INVENTORY_FILE" '^## Runtime-Path Incompatibilities$' "inventory missing runtime section marker"
require_pattern "$INVENTORY_FILE" '^## Test-Only Incompatibilities$' "inventory missing test-only section marker"
require_pattern "$INVENTORY_FILE" '^## Positive Controls$' "inventory missing positive-controls section marker"
require_pattern "$INVENTORY_FILE" '^## Locked Disposition Taxonomy$' "inventory missing locked disposition taxonomy section"

inventory_header='^\| dependency_id \| finding_id \| file_path \| symbol_or_api \| branch_context \| master_evidence \| impact_statement \| confidence \| match_rule \| impact_level \| disposition \| disposition_rationale \| tags \| status \| notes \|$'
header_count="$(rg -c -- "$inventory_header" "$INVENTORY_FILE")"
if [[ "$header_count" -lt 4 ]]; then
  fail "inventory must define the full required header row for canonical/runtime/test-only/positive-control tables"
fi

pinned_sha="$(sed -nE 's/^- Pinned SHA: `?([0-9a-f]{40})`?$/\1/p' "$MASTER_TARGET_FILE" | head -n1)"
if [[ -z "$pinned_sha" ]]; then
  fail "unable to parse pinned SHA from ${MASTER_TARGET_FILE}"
fi

declare -A inventory_dep_seen=()
declare -A inventory_finding_to_dep=()

pma_row_count=0
nounspace_row_count=0
positive_control_seen=0

while IFS=$'\t' read -r section dependency_id finding_id file_path symbol_or_api branch_context master_evidence impact_statement confidence match_rule impact_level disposition disposition_rationale tags status notes; do
  if [[ -z "$dependency_id" || -z "$finding_id" ]]; then
    fail "${section}: missing dependency_id or finding_id"
  fi
  if [[ -z "$file_path" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: file_path"
  fi
  if [[ -z "$symbol_or_api" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: symbol_or_api"
  fi
  if [[ -z "$branch_context" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: branch_context"
  fi
  if [[ -z "$master_evidence" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: master_evidence"
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
  if [[ -z "$disposition" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: disposition"
  fi
  if [[ -z "$disposition_rationale" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: disposition_rationale"
  fi
  if [[ -z "$tags" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: tags"
  fi
  if [[ -z "$status" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: status"
  fi
  if [[ -z "$notes" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} missing required field: notes"
  fi

  if [[ "$confidence" != "high" && "$confidence" != "medium" && "$confidence" != "low" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} invalid confidence '${confidence}' (allowed: high|medium|low)"
  fi

  if [[ "$match_rule" != "exact_missing_ref" && "$match_rule" != "replaceable_gap" && "$match_rule" != "branch_env_config_toggle" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} invalid match_rule '${match_rule}' (allowed: exact_missing_ref|replaceable_gap|branch_env_config_toggle)"
  fi

  if [[ "$disposition" != "remove" && "$disposition" != "replace-with-master-equivalent" && "$disposition" != "feature-gate" && "$disposition" != "defer" ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} invalid disposition '${disposition}' (allowed: remove|replace-with-master-equivalent|feature-gate|defer)"
  fi

  if [[ "$branch_context" != *"$pinned_sha"* ]]; then
    fail "${section}:${finding_id:-<missing finding_id>} branch_context must include pinned SHA ${pinned_sha}"
  fi

  dep_norm="$(normalize_dep_id "$dependency_id")"
  inventory_dep_seen["$dep_norm"]=1
  inventory_finding_to_dep["$finding_id"]="$dep_norm"

  evidence_blob="$(printf '%s %s %s %s' "$symbol_or_api" "$tags" "$notes" "$impact_statement" | tr '[:upper:]' '[:lower:]')"
  if grep -q "pma" <<<"$evidence_blob"; then
    pma_row_count=$((pma_row_count + 1))
  fi
  if grep -Eq "(nounspace|noun_space|in_space)" <<<"$evidence_blob"; then
    nounspace_row_count=$((nounspace_row_count + 1))
  fi

  if [[ "$symbol_or_api" == "heaviest-chain-blocks-range" ]]; then
    positive_control_seen=1
    lower_master_evidence="$(printf '%s' "$master_evidence" | tr '[:upper:]' '[:lower:]')"
    lower_notes="$(printf '%s' "$notes" | tr '[:upper:]' '[:lower:]')"
    if grep -Eq "(no matches|returned no matches)" <<<"$lower_master_evidence"; then
      fail "${section}:${finding_id} positive control has missing master evidence"
    fi
    if grep -q "master_presence=missing" <<<"$lower_notes"; then
      fail "${section}:${finding_id} positive control is incorrectly tagged as missing"
    fi
  fi
done < <(
  awk '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    BEGIN { section = "" }
    /^## Runtime-Path Incompatibilities$/ { section = "runtime"; next }
    /^## Test-Only Incompatibilities$/ { section = "test-only"; next }
    /^## Positive Controls$/ { section = "positive-controls"; next }
    /^## / { if (section != "") section = ""; next }
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
      if (is_sep) next
      if (tolower(cols[1]) == "dependency_id") next

      all_empty = 1
      for (i = 1; i <= m; i++) {
        if (cols[i] != "") {
          all_empty = 0
          break
        }
      }
      if (all_empty) next

      for (i = m + 1; i <= 15; i++) {
        cols[i] = ""
      }

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", section, cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9], cols[10], cols[11], cols[12], cols[13], cols[14], cols[15]
    }
  ' "$INVENTORY_FILE"
)

if [[ "$pma_row_count" -eq 0 ]]; then
  fail "inventory must include at least one PMA-tagged row"
fi

if [[ "$nounspace_row_count" -eq 0 ]]; then
  fail "inventory must include at least one NounSpace/branch-only concept row"
fi

if [[ "$positive_control_seen" -eq 0 ]]; then
  fail "inventory must include positive control row for heaviest-chain-blocks-range"
fi

candidate_header="$(head -n1 "$CANDIDATE_FILE")"
for required_col in candidate_id master_presence dependency_id; do
  if ! tr '\t' '\n' <<<"$candidate_header" | rg -qx -- "$required_col"; then
    fail "candidate index missing required column '${required_col}'"
  fi
done

candidate_link_rows=0
while IFS=$'\t' read -r candidate_id kind file_path symbol_or_api branch_context master_presence disposition tags master_evidence dependency_id; do
  if [[ "$candidate_id" == "candidate_id" ]]; then
    continue
  fi
  if [[ -z "$candidate_id" ]]; then
    continue
  fi

  if [[ "$master_presence" == "missing" || "$master_presence" == "uncertain" ]]; then
    candidate_link_rows=$((candidate_link_rows + 1))
    if [[ -z "$dependency_id" ]]; then
      fail "candidate ${candidate_id} (${master_presence}) missing dependency_id mapping"
      continue
    fi

    dep_norm="$(normalize_dep_id "$dependency_id")"
    if [[ -z "${inventory_dep_seen[$dep_norm]:-}" ]]; then
      fail "candidate ${candidate_id} maps to ${dependency_id} but inventory has no matching dependency_id row"
    fi

    mapped_dep="${inventory_finding_to_dep[$candidate_id]:-}"
    if [[ -z "$mapped_dep" ]]; then
      fail "candidate ${candidate_id} (${master_presence}) must map to an inventory finding_id row"
    elif [[ "$mapped_dep" != "$dep_norm" ]]; then
      fail "candidate ${candidate_id} dependency mismatch (candidate: ${dependency_id}, inventory: ${mapped_dep})"
    fi
  fi
done < "$CANDIDATE_FILE"

if [[ "$candidate_link_rows" -eq 0 ]]; then
  fail "candidate linkage check found no missing/uncertain rows; index parse may be invalid"
fi

if (( failures > 0 )); then
  echo "Master compatibility inventory verification failed with ${failures} error(s)." >&2
  exit 1
fi

echo "Master compatibility inventory verification passed."
