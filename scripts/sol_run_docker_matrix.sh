#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run SOL docker matrix across memory limits with phased fixture rollout.

This wraps scripts/sol_bench_matrix_trace.sh and runs:
  1) phase1 with v0 fixture only
  2) phase2 with v0,v1,v2 fixtures (only if phase1 succeeds)

Usage:
  scripts/sol_run_docker_matrix.sh [options]

Options:
  --root <path>               Root with branch build trees (default: /shared/nockchain-ext4-bench)
  --fixtures-dir <path>       Fixture directory (default: bench-artifacts/fixtures)
  --run-base <path>           Output base dir (default: bench-artifacts/runs)
  --memories <csv>            Docker memory limits CSV (default: 4g,8g,16g,32g)
  --passes <csv>              Passes CSV for underlying script (default: 1)
  --envs <csv>                Envs CSV for underlying script (default: docker)
  --phase1 <csv>              Phase1 fixtures (default: v0)
  --phase2 <csv>              Phase2 fixtures (default: v0,v1,v2)
  --skip-phase2               Do not run phase2
  --continue-on-failure       Keep running all memories in a phase and summarize failures
  --run-tag-prefix <string>   Tag prefix for run directories (default: sol-docker-matrix)
  --profile-memory <bool>     true/false (default: true)
  --perf-native <bool>        true/false (default: false)
  --perf-docker <bool>        true/false (default: false)
  --tracy-native <bool>       true/false (default: false)
  --tracy-docker <bool>       true/false (default: false)
  --dry-run                   Print commands only
  -h, --help                  Show help

Examples:
  scripts/sol_run_docker_matrix.sh

  scripts/sol_run_docker_matrix.sh \
    --memories 8g,16g \
    --passes 1,2 \
    --run-tag-prefix pma-throughput
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRACE_SCRIPT="$SCRIPT_DIR/sol_bench_matrix_trace.sh"

ROOT="/shared/nockchain-ext4-bench"
FIX_DIR="bench-artifacts/fixtures"
RUN_BASE="bench-artifacts/runs"
MEMORIES_CSV="4g,8g,16g,32g"
PASSES_CSV="1"
ENVS_CSV="docker"
PHASE1_FIXTURES_CSV="v0"
PHASE2_FIXTURES_CSV="v0,v1,v2"
RUN_TAG_PREFIX="sol-docker-matrix"
PROFILE_MEMORY="true"
PERF_NATIVE="false"
PERF_DOCKER="false"
TRACY_CAPTURE_NATIVE="false"
TRACY_CAPTURE_DOCKER="false"
RUN_PHASE2="true"
DRY_RUN="false"
FAIL_FAST="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --fixtures-dir) FIX_DIR="$2"; shift 2 ;;
    --run-base) RUN_BASE="$2"; shift 2 ;;
    --memories) MEMORIES_CSV="$2"; shift 2 ;;
    --passes) PASSES_CSV="$2"; shift 2 ;;
    --envs) ENVS_CSV="$2"; shift 2 ;;
    --phase1) PHASE1_FIXTURES_CSV="$2"; shift 2 ;;
    --phase2) PHASE2_FIXTURES_CSV="$2"; shift 2 ;;
    --skip-phase2) RUN_PHASE2="false"; shift ;;
    --continue-on-failure) FAIL_FAST="false"; shift ;;
    --run-tag-prefix) RUN_TAG_PREFIX="$2"; shift 2 ;;
    --profile-memory) PROFILE_MEMORY="$2"; shift 2 ;;
    --perf-native) PERF_NATIVE="$2"; shift 2 ;;
    --perf-docker) PERF_DOCKER="$2"; shift 2 ;;
    --tracy-native) TRACY_CAPTURE_NATIVE="$2"; shift 2 ;;
    --tracy-docker) TRACY_CAPTURE_DOCKER="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

abspath() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "$(pwd)/$p"
  fi
}

[[ -x "$TRACE_SCRIPT" ]] || { echo "Trace script missing/executable: $TRACE_SCRIPT" >&2; exit 2; }
ROOT="$(abspath "$ROOT")"
FIX_DIR="$(abspath "$FIX_DIR")"
RUN_BASE="$(abspath "$RUN_BASE")"

[[ -d "$ROOT" ]] || { echo "Root dir not found: $ROOT" >&2; exit 2; }
[[ -d "$FIX_DIR" ]] || { echo "Fixtures dir not found: $FIX_DIR" >&2; exit 2; }
mkdir -p "$RUN_BASE"

declare -a MEMORIES=()
declare -a PHASE1_FIXTURES=()
declare -a PHASE2_FIXTURES=()

IFS=',' read -r -a MEMORIES <<< "$MEMORIES_CSV"
IFS=',' read -r -a PHASE1_FIXTURES <<< "$PHASE1_FIXTURES_CSV"
IFS=',' read -r -a PHASE2_FIXTURES <<< "$PHASE2_FIXTURES_CSV"

[[ ${#MEMORIES[@]} -gt 0 ]] || { echo "No memories provided" >&2; exit 2; }
[[ ${#PHASE1_FIXTURES[@]} -gt 0 ]] || { echo "No phase1 fixtures provided" >&2; exit 2; }
[[ ${#PHASE2_FIXTURES[@]} -gt 0 ]] || { echo "No phase2 fixtures provided" >&2; exit 2; }

fixture_specs_from_ids() {
  local ids=("$@")
  local specs=()
  local id
  for id in "${ids[@]}"; do
    [[ "$id" =~ ^v[0-9]+$ ]] || {
      echo "Invalid fixture id '$id' (expected v0/v1/v2...)" >&2
      exit 2
    }
    local filename="${id}-100.soltest"
    local path="$FIX_DIR/$filename"
    [[ -f "$path" ]] || {
      echo "Missing fixture file for '$id': $path" >&2
      exit 2
    }
    specs+=("${id}=${filename}")
  done
  local joined
  joined="$(IFS=','; echo "${specs[*]}")"
  echo "$joined"
}

latest_run_dir_for_tag() {
  local tag="$1"
  find "$RUN_BASE" -mindepth 1 -maxdepth 1 -type d -name "*-${tag}" | sort | tail -n 1
}

summary_exit_column_index() {
  local summary="$1"
  awk -F'\t' '
    NR==1 {
      for (i=1; i<=NF; i++) {
        if ($i == "exit_status") {
          print i
          exit
        }
      }
    }
  ' "$summary"
}

summary_all_exit_zero() {
  local summary="$1"
  local idx
  idx="$(summary_exit_column_index "$summary")"
  [[ -n "$idx" ]] || { echo "exit_status column not found: $summary" >&2; return 1; }
  awk -F'\t' -v idx="$idx" 'NR>1 { if ($idx != "0") { bad=1 } } END { exit bad ? 1 : 0 }' "$summary"
}

run_one_matrix() {
  local phase="$1"
  local memory="$2"
  local fixture_specs="$3"
  local tag="${RUN_TAG_PREFIX}-${phase}-${memory}"
  local cmd=(
    env
    ROOT="$ROOT"
    FIX_DIR="$FIX_DIR"
    RUN_BASE="$RUN_BASE"
    RUN_TAG="$tag"
    PASSES_CSV="$PASSES_CSV"
    ENVS_CSV="$ENVS_CSV"
    FIXTURE_SPECS="$fixture_specs"
    DOCKER_MEMORY="$memory"
    PROFILE_MEMORY="$PROFILE_MEMORY"
    PERF_NATIVE="$PERF_NATIVE"
    PERF_DOCKER="$PERF_DOCKER"
    TRACY_CAPTURE_NATIVE="$TRACY_CAPTURE_NATIVE"
    TRACY_CAPTURE_DOCKER="$TRACY_CAPTURE_DOCKER"
    "$TRACE_SCRIPT"
  )

  echo
  echo "=== Running phase=$phase memory=$memory fixtures=$fixture_specs ==="
  echo "Tag: $tag"

  if [[ "$DRY_RUN" == "true" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"

  local run_dir
  run_dir="$(latest_run_dir_for_tag "$tag")"
  [[ -n "$run_dir" && -d "$run_dir" ]] || {
    echo "Could not find run directory for tag '$tag'" >&2
    return 1
  }

  local summary="$run_dir/combined_summary.tsv"
  [[ -f "$summary" ]] || {
    echo "Missing summary TSV: $summary" >&2
    return 1
  }

  if summary_all_exit_zero "$summary"; then
    echo "SUCCESS: all exit_status == 0 ($summary)"
    return 0
  fi

  echo "FAILURE: non-zero exit_status detected ($summary)" >&2
  return 1
}

PHASE1_SPECS="$(fixture_specs_from_ids "${PHASE1_FIXTURES[@]}")"
PHASE2_SPECS="$(fixture_specs_from_ids "${PHASE2_FIXTURES[@]}")"

echo "Root: $ROOT"
echo "Fixtures dir: $FIX_DIR"
echo "Run base: $RUN_BASE"
echo "Memories: $MEMORIES_CSV"
echo "Passes: $PASSES_CSV"
echo "Envs: $ENVS_CSV"
echo "Phase1 fixtures: $PHASE1_SPECS"
echo "Phase2 fixtures: $PHASE2_SPECS"
echo "Profile memory: $PROFILE_MEMORY"
echo "Perf native/docker: $PERF_NATIVE/$PERF_DOCKER"
echo "Tracy native/docker: $TRACY_CAPTURE_NATIVE/$TRACY_CAPTURE_DOCKER"
echo "Fail fast: $FAIL_FAST"

declare -a FAILURES=()
PHASE1_FAILED="false"
PHASE2_FAILED="false"

for memory in "${MEMORIES[@]}"; do
  if ! run_one_matrix "phase1" "$memory" "$PHASE1_SPECS"; then
    PHASE1_FAILED="true"
    FAILURES+=("phase1/$memory")
    if [[ "$FAIL_FAST" == "true" ]]; then
      break
    fi
  fi
done

if [[ "$PHASE1_FAILED" == "true" ]]; then
  echo
  echo "Phase1 did not fully succeed. Skipping phase2."
  if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "Failed runs: ${FAILURES[*]}"
  fi
  exit 1
fi

if [[ "$RUN_PHASE2" != "true" ]]; then
  echo "Phase1 complete. Phase2 skipped by flag."
  exit 0
fi

for memory in "${MEMORIES[@]}"; do
  if ! run_one_matrix "phase2" "$memory" "$PHASE2_SPECS"; then
    PHASE2_FAILED="true"
    FAILURES+=("phase2/$memory")
    if [[ "$FAIL_FAST" == "true" ]]; then
      break
    fi
  fi
done

echo
if [[ "$PHASE2_FAILED" == "true" ]]; then
  echo "Matrix finished with failures."
  echo "Failed runs: ${FAILURES[*]}"
  exit 1
fi

echo "All requested matrix runs completed successfully."
