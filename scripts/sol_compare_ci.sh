#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_PATH="${SOL_COMPARE_BIN:-$ROOT_DIR/target/release/nockchain-bench}"

usage() {
  cat <<'EOF'
Usage:
  scripts/sol_compare_ci.sh \
    --candidate <combined_summary.tsv> \
    --baseline <combined_summary.tsv> \
    [--output-json <path>] \
    [--output-md <path>] \
    [--significance <alpha>] \
    [--min-samples <n>]
EOF
}

CANDIDATE=""
BASELINE=""
OUTPUT_JSON=""
OUTPUT_MD=""
SIGNIFICANCE=""
MIN_SAMPLES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --baseline) BASELINE="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --output-md) OUTPUT_MD="$2"; shift 2 ;;
    --significance) SIGNIFICANCE="$2"; shift 2 ;;
    --min-samples) MIN_SAMPLES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$CANDIDATE" ]] || { echo "--candidate is required" >&2; exit 2; }
[[ -n "$BASELINE" ]] || { echo "--baseline is required" >&2; exit 2; }

# If baseline file does not exist, skip comparison gracefully
if [ ! -f "$BASELINE" ]; then
  echo "No baseline available — skipping comparison"
  exit 0
fi

cmd=(sol compare
  --candidate-summary "$CANDIDATE"
  --baseline-summary "$BASELINE"
)

if [[ -n "$OUTPUT_JSON" ]]; then
  cmd+=(--output-json "$OUTPUT_JSON")
fi
if [[ -n "$OUTPUT_MD" ]]; then
  cmd+=(--output-md "$OUTPUT_MD")
fi
if [[ -n "$SIGNIFICANCE" ]]; then
  cmd+=(--significance "$SIGNIFICANCE")
fi
if [[ -n "$MIN_SAMPLES" ]]; then
  cmd+=(--min-samples "$MIN_SAMPLES")
fi

if [[ -x "$BIN_PATH" ]]; then
  echo "Using compare binary: $BIN_PATH"
  "$BIN_PATH" "${cmd[@]}"
else
  echo "Using cargo fallback (binary not found at $BIN_PATH)"
  (cd "$ROOT_DIR" && cargo run --release -p nockchain-bench -- "${cmd[@]}")
fi
