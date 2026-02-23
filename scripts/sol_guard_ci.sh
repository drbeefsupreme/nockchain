#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_PATH="${SOL_GUARD_BIN:-$ROOT_DIR/target/debug/nockchain-bench}"

usage() {
  cat <<'EOF'
Usage:
  scripts/sol_guard_ci.sh \
    --candidate-summary <combined_summary.tsv> \
    --contract <contract.toml> \
    --env <native|docker> \
    --branch <branch-id> \
    --fixture <fixture-id> \
    [--baseline-summary <combined_summary.tsv>] \
    [--pass <n>] \
    [--run-id <id>] \
    [--output-json <path>] \
    [--output-md <path>] \
    [--strict]
EOF
}

CANDIDATE_SUMMARY=""
BASELINE_SUMMARY=""
CONTRACT=""
ENV_NAME=""
BRANCH=""
FIXTURE=""
PASS_NO=""
RUN_ID=""
OUTPUT_JSON=""
OUTPUT_MD=""
STRICT="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate-summary) CANDIDATE_SUMMARY="$2"; shift 2 ;;
    --baseline-summary) BASELINE_SUMMARY="$2"; shift 2 ;;
    --contract) CONTRACT="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --fixture) FIXTURE="$2"; shift 2 ;;
    --pass) PASS_NO="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --output-md) OUTPUT_MD="$2"; shift 2 ;;
    --strict) STRICT="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$CANDIDATE_SUMMARY" ]] || { echo "--candidate-summary is required" >&2; exit 2; }
[[ -n "$CONTRACT" ]] || { echo "--contract is required" >&2; exit 2; }
[[ -n "$ENV_NAME" ]] || { echo "--env is required" >&2; exit 2; }
[[ -n "$BRANCH" ]] || { echo "--branch is required" >&2; exit 2; }
[[ -n "$FIXTURE" ]] || { echo "--fixture is required" >&2; exit 2; }

cmd=(sol guard
  --candidate-summary "$CANDIDATE_SUMMARY"
  --contract "$CONTRACT"
  --env "$ENV_NAME"
  --branch "$BRANCH"
  --fixture "$FIXTURE"
)
if [[ -n "$BASELINE_SUMMARY" ]]; then
  cmd+=(--baseline-summary "$BASELINE_SUMMARY")
fi
if [[ -n "$PASS_NO" ]]; then
  cmd+=(--pass "$PASS_NO")
fi
if [[ -n "$RUN_ID" ]]; then
  cmd+=(--run-id "$RUN_ID")
fi
if [[ -n "$OUTPUT_JSON" ]]; then
  cmd+=(--output-json "$OUTPUT_JSON")
fi
if [[ -n "$OUTPUT_MD" ]]; then
  cmd+=(--output-md "$OUTPUT_MD")
fi
if [[ "$STRICT" == "true" ]]; then
  cmd+=(--strict)
fi

if [[ -x "$BIN_PATH" ]]; then
  echo "Using guard binary: $BIN_PATH"
  "$BIN_PATH" "${cmd[@]}"
else
  echo "Using cargo fallback (binary not found at $BIN_PATH)"
  (cd "$ROOT_DIR" && cargo run -p nockchain-bench -- "${cmd[@]}")
fi
