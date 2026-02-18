#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Build v0/v1/v2 100-block SOL fixtures in one command.

Usage:
  scripts/sol_build_fixtures.sh \
    --bench-bin <path/to/nockchain-bench> \
    --kernel <path/to/dumb.jam> \
    --checkpoint-v01 <path/to/0-or-1.chkjam> \
    [--checkpoint-v2 <path/to/1-or-large.chkjam>] \
    --out-dir <fixtures-dir> \
    --work-dir <tmp-work-dir>

Optional range overrides (start:end):
  --v0-range 1:100
  --v1-range 6750:6849
  --v2-range 12000:12099
USAGE
}

BENCH_BIN=""
KERNEL=""
CHECKPOINT_V01=""
CHECKPOINT_V2=""
OUT_DIR=""
WORK_DIR=""
V0_RANGE="1:100"
V1_RANGE="6750:6849"
V2_RANGE="12000:12099"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench-bin) BENCH_BIN="$2"; shift 2 ;;
    --kernel) KERNEL="$2"; shift 2 ;;
    --checkpoint-v01) CHECKPOINT_V01="$2"; shift 2 ;;
    --checkpoint-v2) CHECKPOINT_V2="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --v0-range) V0_RANGE="$2"; shift 2 ;;
    --v1-range) V1_RANGE="$2"; shift 2 ;;
    --v2-range) V2_RANGE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$BENCH_BIN" || -z "$KERNEL" || -z "$CHECKPOINT_V01" || -z "$OUT_DIR" || -z "$WORK_DIR" ]]; then
  usage
  exit 2
fi

if [[ -z "$CHECKPOINT_V2" ]]; then
  CHECKPOINT_V2="$CHECKPOINT_V01"
fi

for path in "$BENCH_BIN" "$KERNEL" "$CHECKPOINT_V01" "$CHECKPOINT_V2"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 2
  fi
done

parse_range() {
  local in="$1"
  local label="$2"
  if [[ "$in" != *:* ]]; then
    echo "Invalid $label range '$in' (expected start:end)" >&2
    exit 2
  fi
  local start="${in%%:*}"
  local end="${in##*:}"
  if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ ]]; then
    echo "Invalid $label range '$in' (non-numeric)" >&2
    exit 2
  fi
  if (( start > end )); then
    echo "Invalid $label range '$in' (start > end)" >&2
    exit 2
  fi
  echo "$start $end"
}

read -r V0_START V0_END < <(parse_range "$V0_RANGE" v0)
read -r V1_START V1_END < <(parse_range "$V1_RANGE" v1)
read -r V2_START V2_END < <(parse_range "$V2_RANGE" v2)

mkdir -p "$OUT_DIR" "$WORK_DIR"

build_fixture() {
  local label="$1"
  local checkpoint="$2"
  local start="$3"
  local end="$4"
  local out="$5"
  local tmp="$6"

  rm -rf "$tmp"
  mkdir -p "$tmp"

  echo "== Building $label fixture ($start..=$end) =="
  "$BENCH_BIN" sol fixture build \
    --source-checkpoint "$checkpoint" \
    --kernel "$KERNEL" \
    --start-height "$start" \
    --end-height "$end" \
    --output "$out" \
    --chunk-size 8 \
    --work-dir "$tmp"
}

build_fixture v0 "$CHECKPOINT_V01" "$V0_START" "$V0_END" "$OUT_DIR/v0-100.soltest" "$WORK_DIR/v0"
build_fixture v1 "$CHECKPOINT_V01" "$V1_START" "$V1_END" "$OUT_DIR/v1-100.soltest" "$WORK_DIR/v1"
build_fixture v2 "$CHECKPOINT_V2" "$V2_START" "$V2_END" "$OUT_DIR/v2-100.soltest" "$WORK_DIR/v2"

echo
for f in "$OUT_DIR"/v0-100.soltest "$OUT_DIR"/v1-100.soltest "$OUT_DIR"/v2-100.soltest; do
  echo "== Inspect $f =="
  "$BENCH_BIN" sol fixture inspect --fixture "$f" | rg -N "Format version|Derived checkpoint height|Archive range|Chunk size"
  ls -lh "$f"
  echo
done
