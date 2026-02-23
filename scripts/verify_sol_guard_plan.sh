#!/usr/bin/env bash
set -euo pipefail

CHECKLIST="${1:-checkpoints/sol_guard_implementation.md}"

if [[ ! -f "$CHECKLIST" ]]; then
  echo "Checklist not found: $CHECKLIST" >&2
  exit 1
fi

missing_ids=()
for i in $(seq 1 45); do
  id=$(printf "G%03d" "$i")
  if ! rg -q "^- \\[[ xX]\\] ${id}\\b" "$CHECKLIST"; then
    missing_ids+=("$id")
  fi
done

if [[ ${#missing_ids[@]} -gt 0 ]]; then
  echo "Checklist is missing expected step IDs: ${missing_ids[*]}" >&2
  exit 1
fi

unchecked="$(rg -n "^- \\[ \\] G[0-9]{3}\\b" "$CHECKLIST" || true)"
if [[ -n "$unchecked" ]]; then
  echo "Checklist has unchecked steps:" >&2
  echo "$unchecked" >&2
  exit 1
fi

echo "SOL guard checklist complete: $CHECKLIST"
