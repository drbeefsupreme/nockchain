#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Build a deterministic nockchain-bench Docker image with provenance labels.

This script avoids COPY-path ambiguity by staging exactly one validated binary
as `nockchain-bench` in an isolated build context.

Usage:
  scripts/build_nockchain_bench_image.sh --image <name:tag> [options]

Options:
  --image <name:tag>           Docker image reference to build (required)
  --binary <path>              Path to nockchain-bench binary
                               (default: target/release/nockchain-bench)
  --metadata-out <path>        Write key/value build metadata file
  --label <key=value>          Additional Docker label (repeatable)
  --base-image <image>         Base image (default: ubuntu:24.04)
  --push                       Push image after build
  -h, --help                   Show this help

Environment:
  SOURCE_DATE_EPOCH            Optional Unix epoch used for
                               org.opencontainers.image.created
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

IMAGE_REF=""
BINARY_PATH="$REPO_ROOT/target/release/nockchain-bench"
METADATA_OUT=""
BASE_IMAGE="ubuntu:24.04"
PUSH_IMAGE="false"
EXTRA_LABELS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE_REF="${2:-}"; shift 2 ;;
    --binary) BINARY_PATH="${2:-}"; shift 2 ;;
    --metadata-out) METADATA_OUT="${2:-}"; shift 2 ;;
    --label) EXTRA_LABELS+=("${2:-}"); shift 2 ;;
    --base-image) BASE_IMAGE="${2:-}"; shift 2 ;;
    --push) PUSH_IMAGE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$IMAGE_REF" ]] || {
  echo "Missing required --image argument" >&2
  usage
  exit 2
}

command -v docker >/dev/null 2>&1 || {
  echo "docker is required but not found in PATH" >&2
  exit 1
}

abs_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "$REPO_ROOT/$p"
  fi
}

BINARY_ABS="$(abs_path "$BINARY_PATH")"
[[ -f "$BINARY_ABS" ]] || {
  echo "Binary not found: $BINARY_ABS" >&2
  exit 1
}
[[ -x "$BINARY_ABS" ]] || {
  echo "Binary is not executable: $BINARY_ABS" >&2
  exit 1
}

BINARY_BASENAME="$(basename "$BINARY_ABS")"
if [[ "$BINARY_BASENAME" != "nockchain-bench" ]]; then
  echo "Expected binary basename 'nockchain-bench', got '$BINARY_BASENAME'" >&2
  echo "Refusing ambiguous artifact input." >&2
  exit 1
fi

for label in "${EXTRA_LABELS[@]}"; do
  [[ "$label" == *=* ]] || {
    echo "Invalid --label value '$label' (expected key=value)" >&2
    exit 2
  }
done

BIN_SHA256="$(sha256sum "$BINARY_ABS" | awk '{print $1}')"
BIN_SIZE_BYTES="$(stat -c '%s' "$BINARY_ABS")"

GIT_REV="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="false"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null || true)" ]]; then
  GIT_DIRTY="true"
fi
GIT_SOURCE="$(git -C "$REPO_ROOT" config --get remote.origin.url 2>/dev/null || echo unknown)"

if [[ -n "${SOURCE_DATE_EPOCH:-}" ]]; then
  CREATED_UTC="$(date -u -d "@$SOURCE_DATE_EPOCH" '+%Y-%m-%dT%H:%M:%SZ')"
else
  CREATED_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
fi

TMP_DIR="$(mktemp -d -t nockchain-bench-image-XXXXXX)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

cp "$BINARY_ABS" "$TMP_DIR/nockchain-bench"

cat > "$TMP_DIR/Dockerfile" <<EOF
FROM ${BASE_IMAGE}
RUN apt-get update \\
    && apt-get install -y --no-install-recommends ca-certificates libssl3 \\
    && rm -rf /var/lib/apt/lists/*
COPY nockchain-bench /usr/local/bin/nockchain-bench
ENTRYPOINT ["/usr/local/bin/nockchain-bench"]
EOF

label_args=(
  "--label" "org.opencontainers.image.title=nockchain-bench"
  "--label" "org.opencontainers.image.description=Benchmarking and memory profiling tool for Nockchain"
  "--label" "org.opencontainers.image.source=$GIT_SOURCE"
  "--label" "org.opencontainers.image.revision=$GIT_REV"
  "--label" "org.opencontainers.image.created=$CREATED_UTC"
  "--label" "io.nockchain.bench.binary.path=$BINARY_ABS"
  "--label" "io.nockchain.bench.binary.sha256=$BIN_SHA256"
  "--label" "io.nockchain.bench.binary.size_bytes=$BIN_SIZE_BYTES"
  "--label" "io.nockchain.bench.git.dirty=$GIT_DIRTY"
  "--label" "io.nockchain.bench.build.script=scripts/build_nockchain_bench_image.sh"
)

for label in "${EXTRA_LABELS[@]}"; do
  label_args+=("--label" "$label")
done

echo "Building image: $IMAGE_REF"
DOCKER_BUILDKIT=1 docker build \
  --tag "$IMAGE_REF" \
  "${label_args[@]}" \
  "$TMP_DIR"

if [[ "$PUSH_IMAGE" == "true" ]]; then
  echo "Pushing image: $IMAGE_REF"
  docker push "$IMAGE_REF"
fi

echo "Built image: $IMAGE_REF"
echo "  binary_sha256=$BIN_SHA256"
echo "  binary_size_bytes=$BIN_SIZE_BYTES"
echo "  git_revision=$GIT_REV"
echo "  git_dirty=$GIT_DIRTY"
echo "  created=$CREATED_UTC"

if [[ -n "$METADATA_OUT" ]]; then
  META_ABS="$(abs_path "$METADATA_OUT")"
  mkdir -p "$(dirname "$META_ABS")"
  cat > "$META_ABS" <<EOF
image_ref=$IMAGE_REF
binary_path=$BINARY_ABS
binary_sha256=$BIN_SHA256
binary_size_bytes=$BIN_SIZE_BYTES
git_revision=$GIT_REV
git_dirty=$GIT_DIRTY
source=$GIT_SOURCE
created=$CREATED_UTC
base_image=$BASE_IMAGE
EOF
  echo "Wrote metadata: $META_ABS"
fi
