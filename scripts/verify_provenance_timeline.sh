#!/usr/bin/env bash
set -euo pipefail

PHASE3_DIR=".planning/phases/03-provenance-and-divergence-timeline"
PHASE2_DIR=".planning/phases/02-master-compatibility-inventory"

PROVENANCE_DOC="${PHASE3_DIR}/03-provenance-and-divergence-timeline.md"
PROVENANCE_TSV="${PHASE3_DIR}/03-provenance-evidence.tsv"
PHASE2_INVENTORY="${PHASE2_DIR}/02-master-compatibility-inventory.md"
PHASE2_CANDIDATES="${PHASE2_DIR}/02-compat-candidate-index.tsv"

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

normalize_id() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

require_file "$PROVENANCE_DOC"
require_file "$PROVENANCE_TSV"
require_file "$PHASE2_INVENTORY"
require_file "$PHASE2_CANDIDATES"

require_pattern "$PROVENANCE_DOC" '^## Attribution Taxonomy$' "provenance doc missing attribution taxonomy section"
require_pattern "$PROVENANCE_DOC" '^## Unresolved Provenance$' "provenance doc missing unresolved provenance section"
require_pattern "$PROVENANCE_DOC" '^## Divergence Timeline \(Thematic Buckets\)$' "provenance doc missing thematic timeline section"
require_pattern "$PROVENANCE_DOC" '^- `Inherited`' "provenance doc missing locked Inherited classification"
require_pattern "$PROVENANCE_DOC" '^- `Local`' "provenance doc missing locked Local classification"
require_pattern "$PROVENANCE_DOC" '^- `Mixed`' "provenance doc missing locked Mixed classification"

tsv_header="$(head -n1 "$PROVENANCE_TSV")"
for required_col in \
  dependency_id \
  finding_ids \
  classification \
  origin_commit_sha \
  pivot_commit_shas \
  historical_branch_evidence \
  current_branch_evidence \
  rationale \
  confidence \
  status \
  notes \
  branch_master_ref \
  branch_historical_ref \
  branch_current_ref; do
  if ! tr '\t' '\n' <<<"$tsv_header" | rg -qx -- "$required_col"; then
    fail "provenance TSV missing required column '${required_col}'"
  fi
done

declare -A phase2_dep_seen=()
declare -A phase2_finding_to_dep=()

while IFS=$'\t' read -r dependency_id finding_id; do
  dep_norm="$(normalize_id "$dependency_id")"
  finding_norm="$(normalize_id "$finding_id")"
  phase2_dep_seen["$dep_norm"]=1
  phase2_finding_to_dep["$finding_norm"]="$dep_norm"
done < <(
  awk '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    /^## Runtime-Path Incompatibilities$/ { section = 1; next }
    /^## Test-Only Incompatibilities$/ { section = 1; next }
    /^## Positive Controls$/ { section = 1; next }
    /^## / { section = 0; next }
    section == 1 && /^\|/ {
      n = split($0, raw, "|")
      dep = trim(raw[2])
      finding = trim(raw[3])
      if (dep == "" || finding == "" || tolower(dep) == "dependency_id" || dep ~ /^-+$/) next
      print dep "\t" finding
    }
  ' "$PHASE2_INVENTORY"
)

if [[ "${#phase2_dep_seen[@]}" -eq 0 ]]; then
  fail "unable to parse dependency IDs from ${PHASE2_INVENTORY}"
fi

declare -A required_phase3_deps=()

while IFS=$'\t' read -r _candidate_id _kind _file_path _symbol_or_api _branch_context master_presence _disposition _tags _master_evidence dependency_id; do
  if [[ "$master_presence" != "missing" && "$master_presence" != "uncertain" ]]; then
    continue
  fi

  if [[ -z "$dependency_id" ]]; then
    fail "candidate row with master_presence=${master_presence} missing dependency_id"
    continue
  fi
  required_phase3_deps["$(normalize_id "$dependency_id")"]=1
done < <(tail -n +2 "$PHASE2_CANDIDATES")

if [[ "${#required_phase3_deps[@]}" -eq 0 ]]; then
  fail "no missing/uncertain dependency IDs discovered in ${PHASE2_CANDIDATES}"
fi

declare -A provenance_dep_seen=()
declare -A unresolved_dep_seen=()
declare -A unresolved_dep_label=()

while IFS=$'\t' read -r dependency_id finding_ids classification origin_commit_sha pivot_commit_shas historical_branch_evidence current_branch_evidence rationale confidence status notes branch_master_ref branch_historical_ref branch_current_ref; do
  if [[ "$dependency_id" == "dependency_id" ]]; then
    continue
  fi

  if [[ -z "$dependency_id" ]]; then
    fail "provenance row missing dependency_id"
    continue
  fi
  if [[ -z "$finding_ids" ]]; then
    fail "${dependency_id}: missing finding_ids"
  fi
  if [[ -z "$origin_commit_sha" || -z "$pivot_commit_shas" || -z "$historical_branch_evidence" || -z "$current_branch_evidence" ]]; then
    fail "${dependency_id}: missing required evidence fields"
  fi
  if [[ -z "$rationale" ]]; then
    fail "${dependency_id}: missing rationale"
  fi
  if [[ -z "$confidence" ]]; then
    fail "${dependency_id}: missing confidence"
  fi
  if [[ -z "$status" ]]; then
    fail "${dependency_id}: missing status"
  fi

  dep_norm="$(normalize_id "$dependency_id")"
  provenance_dep_seen["$dep_norm"]=1

  if [[ -z "${phase2_dep_seen[$dep_norm]:-}" ]]; then
    fail "${dependency_id}: orphaned dependency_id not present in Phase 2 inventory"
  fi

  if [[ "$confidence" != "high" && "$confidence" != "medium" && "$confidence" != "low" ]]; then
    fail "${dependency_id}: invalid confidence '${confidence}' (allowed: high|medium|low)"
  fi

  if [[ "$status" != "resolved" && "$status" != "unresolved" ]]; then
    fail "${dependency_id}: invalid status '${status}' (allowed: resolved|unresolved)"
  fi

  if [[ "$status" == "resolved" ]]; then
    if [[ "$classification" != "Inherited" && "$classification" != "Local" && "$classification" != "Mixed" ]]; then
      fail "${dependency_id}: resolved row has invalid classification '${classification}' (allowed: Inherited|Local|Mixed)"
    fi
  fi

  if [[ "$status" == "unresolved" ]]; then
    unresolved_dep_seen["$dep_norm"]=1
    unresolved_dep_label["$dep_norm"]="$dependency_id"
  fi

  IFS=',' read -ra finding_tokens <<<"$finding_ids"
  for finding_id in "${finding_tokens[@]}"; do
    finding_id="$(sed 's/^ *//; s/ *$//' <<<"$finding_id")"
    if [[ -z "$finding_id" ]]; then
      continue
    fi
    finding_norm="$(normalize_id "$finding_id")"
    mapped_dep="${phase2_finding_to_dep[$finding_norm]:-}"
    if [[ -z "$mapped_dep" ]]; then
      fail "${dependency_id}: finding_id '${finding_id}' not present in Phase 2 inventory"
      continue
    fi
    if [[ "$mapped_dep" != "$dep_norm" ]]; then
      fail "${dependency_id}: finding_id '${finding_id}' maps to different Phase 2 dependency (${mapped_dep})"
    fi
  done
done < "$PROVENANCE_TSV"

for dep_norm in "${!required_phase3_deps[@]}"; do
  if [[ -z "${provenance_dep_seen[$dep_norm]:-}" ]]; then
    fail "missing/uncertain Phase 2 dependency not mapped in Phase 3 provenance TSV: ${dep_norm}"
  fi
done

for dep_norm in "${!unresolved_dep_seen[@]}"; do
  dep_id="${unresolved_dep_label[$dep_norm]}"
  if ! awk '
    /^## Unresolved Provenance$/ { in_unresolved = 1; next }
    /^## / && in_unresolved { exit }
    in_unresolved { print }
  ' "$PROVENANCE_DOC" | rg -q "\\|[[:space:]]*${dep_id}[[:space:]]*\\|"; then
    fail "${dep_id}: unresolved row missing from Unresolved Provenance section"
  fi
done

timeline_rows=0

while IFS=$'\t' read -r event_date commit_cell dependency_ids finding_ids; do
  timeline_rows=$((timeline_rows + 1))

  commit_sha="$(tr -d '` ' <<<"$commit_cell")"
  if ! [[ "$commit_sha" =~ ^[0-9a-f]{40}$ ]]; then
    fail "timeline row ${event_date} has invalid commit_sha '${commit_cell}'"
  fi

  if ! grep -Eq 'DEP-[A-Za-z0-9-]+' <<<"$dependency_ids"; then
    fail "timeline row ${event_date} missing dependency reference"
  fi

  if ! grep -Eq '(C[0-9]{3}|PMA-S[0-9]{3})' <<<"$finding_ids"; then
    fail "timeline row ${event_date} missing finding reference"
  fi

  IFS=',' read -ra dep_tokens <<<"$dependency_ids"
  for dep_id in "${dep_tokens[@]}"; do
    dep_id="$(sed 's/^ *//; s/ *$//' <<<"$dep_id")"
    if [[ -z "$dep_id" ]]; then
      continue
    fi
    dep_norm="$(normalize_id "$dep_id")"
    if [[ -z "${provenance_dep_seen[$dep_norm]:-}" ]]; then
      fail "timeline dependency reference '${dep_id}' does not map to a Phase 3 provenance row"
    fi
  done

  IFS=',' read -ra finding_tokens <<<"$finding_ids"
  for finding_id in "${finding_tokens[@]}"; do
    finding_id="$(sed 's/^ *//; s/ *$//' <<<"$finding_id")"
    if [[ -z "$finding_id" ]]; then
      continue
    fi
    finding_norm="$(normalize_id "$finding_id")"
    if [[ -z "${phase2_finding_to_dep[$finding_norm]:-}" ]]; then
      fail "timeline finding reference '${finding_id}' not present in Phase 2 inventory"
    fi
  done
done < <(
  awk '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    /^## Divergence Timeline \(Thematic Buckets\)$/ { in_timeline = 1; next }
    /^## / && in_timeline { exit }
    in_timeline && /^\|/ {
      n = split($0, raw, "|")
      c1 = trim(raw[2])
      c2 = trim(raw[3])
      c3 = trim(raw[4])
      c4 = trim(raw[5])

      if (tolower(c1) == "event_date" || c1 ~ /^-+$/) next
      if (c1 !~ /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/) next

      printf "%s\t%s\t%s\t%s\n", c1, c2, c3, c4
    }
  ' "$PROVENANCE_DOC"
)

if [[ "$timeline_rows" -eq 0 ]]; then
  fail "no timeline event rows parsed from ${PROVENANCE_DOC}"
fi

if (( failures > 0 )); then
  echo "Provenance timeline verification failed with ${failures} error(s)." >&2
  exit 1
fi

echo "Provenance timeline verification passed."
