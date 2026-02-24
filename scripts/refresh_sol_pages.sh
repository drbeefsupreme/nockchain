#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Refresh docs/nockchain-bench GitHub Pages content from the latest SOL run metadata.

Usage:
  scripts/refresh_sol_pages.sh [options]

Options:
  --run-id <id>               Run id to republish (defaults to latest from manifest)
  --run-root <path>           Explicit run root (defaults to newest matching artifacts dir)
  --pages-root <path>         Repo root that contains docs/ (default: current repo root)
  --artifacts-root <path>     Parent folder for run artifacts
                              (default: /shared/nockchain-ext4-bench/artifacts/runs)
  --title <text>              Override run title (default: value from runs-manifest.json)
  --scope <text>              Override run scope (default: value from runs-manifest.json)
  --fixture-label <text>      Optional fixture label override
  --keep-raw-artifacts        Keep generated untracked trace/guard/raw files
  --dry-run                   Print publish command without executing it
  -h, --help                  Show this help
EOF
}

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "error: ${flag} requires a value" >&2
    exit 2
  fi
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
pages_root="${repo_root}"
artifacts_root="/shared/nockchain-ext4-bench/artifacts/runs"
run_id=""
run_root=""
title=""
scope=""
fixture_label=""
keep_raw_artifacts=0
dry_run=0

while (($# > 0)); do
  case "$1" in
    --run-id)
      require_value "$1" "${2:-}"
      run_id="$2"
      shift 2
      ;;
    --run-root)
      require_value "$1" "${2:-}"
      run_root="$2"
      shift 2
      ;;
    --pages-root)
      require_value "$1" "${2:-}"
      pages_root="$2"
      shift 2
      ;;
    --artifacts-root)
      require_value "$1" "${2:-}"
      artifacts_root="$2"
      shift 2
      ;;
    --title)
      require_value "$1" "${2:-}"
      title="$2"
      shift 2
      ;;
    --scope)
      require_value "$1" "${2:-}"
      scope="$2"
      shift 2
      ;;
    --fixture-label)
      require_value "$1" "${2:-}"
      fixture_label="$2"
      shift 2
      ;;
    --keep-raw-artifacts)
      keep_raw_artifacts=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

pages_root="$(realpath "${pages_root}")"
manifest_path="${pages_root}/docs/nockchain-bench/sol-runs/runs-manifest.json"
if [[ ! -f "${manifest_path}" ]]; then
  echo "error: manifest not found: ${manifest_path}" >&2
  exit 1
fi

if [[ -z "${run_id}" || -z "${title}" || -z "${scope}" ]]; then
  mapfile -t manifest_row < <(
    python3 - "${manifest_path}" "${run_id}" <<'PY'
import json
import sys

manifest_path = sys.argv[1]
requested_run_id = sys.argv[2].strip()
payload = json.loads(open(manifest_path, encoding="utf-8").read())
runs = payload.get("runs") or []

record = None
if requested_run_id:
    for row in runs:
        if str(row.get("run_id", "")).strip() == requested_run_id:
            record = row
            break
else:
    if runs:
        record = runs[0]

if record is None:
    raise SystemExit(2)

print(str(record.get("run_id", "")).strip())
print(str(record.get("title", "")).strip())
print(str(record.get("scope", "")).strip())
PY
  ) || {
    echo "error: could not resolve run metadata for run_id='${run_id}' from ${manifest_path}" >&2
    exit 1
  }

  if [[ -z "${run_id}" ]]; then
    run_id="${manifest_row[0]}"
  fi
  if [[ -z "${title}" ]]; then
    title="${manifest_row[1]}"
  fi
  if [[ -z "${scope}" ]]; then
    scope="${manifest_row[2]}"
  fi
fi

if [[ -z "${run_root}" ]]; then
  run_root="$(
    python3 - "${artifacts_root}" "${run_id}" <<'PY'
import glob
import os
import sys

artifacts_root = sys.argv[1]
run_id = sys.argv[2]
matches = [p for p in glob.glob(os.path.join(artifacts_root, f"{run_id}*")) if os.path.isdir(p)]
if not matches:
    raise SystemExit(2)
matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
print(matches[0])
PY
  )" || {
    echo "error: no artifact directory found for run_id='${run_id}' under ${artifacts_root}" >&2
    exit 1
  }
fi

run_root="$(realpath "${run_root}")"
if [[ ! -d "${run_root}" ]]; then
  echo "error: run root does not exist: ${run_root}" >&2
  exit 1
fi

publish_cmd=(
  python3 "${repo_root}/scripts/publish_sol_trace_run.py"
  --run-root "${run_root}"
  --pages-root "${pages_root}"
  --title "${title}"
  --scope "${scope}"
)
if [[ -n "${fixture_label}" ]]; then
  publish_cmd+=(--fixture-label "${fixture_label}")
fi

echo "Run id:      ${run_id}"
echo "Run root:    ${run_root}"
echo "Pages root:  ${pages_root}"
echo "Title:       ${title}"
echo "Scope:       ${scope}"
if [[ -n "${fixture_label}" ]]; then
  echo "Fixture:     ${fixture_label}"
fi

if ((dry_run)); then
  printf 'Dry run publish command:\n'
  printf '  %q' "${publish_cmd[@]}"
  printf '\n'
  exit 0
fi

"${publish_cmd[@]}"

if ((keep_raw_artifacts == 0)); then
  run_rel="docs/nockchain-bench/sol-runs/runs/${run_id}"
  maybe_cleanup=(
    "docs/nockchain-bench/sol-benchmark-transplant-memory-profiles.json"
    "${run_rel}/combined_summary.tsv"
    "${run_rel}/guard"
    "${run_rel}/trace"
    "${run_rel}/sol-benchmark-transplant-memory-profiles.json"
  )
  for rel in "${maybe_cleanup[@]}"; do
    abs="${pages_root}/${rel}"
    if [[ ! -e "${abs}" ]]; then
      continue
    fi
    if [[ -n "$(git -C "${pages_root}" ls-files "${rel}")" ]]; then
      continue
    fi
    rm -rf "${abs}"
    echo "Removed generated untracked artifact: ${rel}"
  done
fi

echo
echo "Changed files:"
git -C "${pages_root}" status --short docs/nockchain-bench

