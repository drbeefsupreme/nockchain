#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# sol_baseline_run.sh — Single deterministic baseline entrypoint
#
# Reads versioned TOML config, validates the working tree, invokes the
# benchmark matrix, collects provenance, and writes a canonical manifest.
#
# Used by both local runs and CI (via sol_baseline_ci.sh wrapper).
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
CONFIG="$REPO_ROOT/benchmarks/baseline/sol-baseline.toml"
PROFILE="quick"
ALLOW_DIRTY=false
VERBOSE=false
QUIET=false
PASSES_OVERRIDE=""
OUTPUT_ROOT_OVERRIDE=""

# Branch-bin and docker-image accumulation
BRANCH_BIN_ARGS=()
DOCKER_IMAGE_ARGS=()

usage() {
  cat <<'USAGE'
Run a reproducible SOL baseline benchmark with full provenance.

Usage:
  scripts/sol_baseline_run.sh [OPTIONS] --branch-bin <name=path>

Required:
  --branch-bin <name=path>    Branch binary specification (repeatable)

Options:
  --config <path>             TOML config file (default: benchmarks/baseline/sol-baseline.toml)
  --profile <quick|full>      Config profile to resolve (default: quick)
  --docker-image <name=image> Docker image specification (repeatable)
  --output-root <dir>         Override config output root
  --passes <N>                Override config passes count
  --verbose                   Increase verbosity
  --quiet                     Suppress non-error output
  -h, --help                  Show this help

Output:
  Creates run directory: <output-root>/<timestamp>_<commit-sha>/
    data/          Raw benchmark results
    meta/
      manifest.json       Canonical provenance manifest
      config-snapshot.toml  Resolved config copy
    logs/
      run.log             Combined stderr/timing

  Updates latest symlink to most recent run directory.

Examples:
  # Quick local run
  scripts/sol_baseline_run.sh \
    --profile quick \
    --branch-bin master=target/release/nockchain-bench

  # Full trusted baseline
  scripts/sol_baseline_run.sh \
    --profile full \
    --branch-bin master=target/release/nockchain-bench \
    --branch-bin feature=target/release/nockchain-bench
USAGE
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)       CONFIG="$2"; shift 2 ;;
    --profile)      PROFILE="$2"; shift 2 ;;
    --branch-bin)   BRANCH_BIN_ARGS+=("--branch-bin" "$2"); shift 2 ;;
    --docker-image) DOCKER_IMAGE_ARGS+=("--docker-image" "$2"); shift 2 ;;
    --output-root)  OUTPUT_ROOT_OVERRIDE="$2"; shift 2 ;;
    --passes)       PASSES_OVERRIDE="$2"; shift 2 ;;
    --verbose)      VERBOSE=true; shift ;;
    --quiet)        QUIET=false; shift ;;
    --allow-dirty)  ALLOW_DIRTY=true; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

# Require at least one branch-bin
if [[ ${#BRANCH_BIN_ARGS[@]} -eq 0 ]]; then
  echo "ERROR: At least one --branch-bin is required." >&2
  usage
  exit 2
fi

log() {
  if [[ "$QUIET" != "true" ]]; then
    echo "[sol-baseline] $*"
  fi
}

log_verbose() {
  if [[ "$VERBOSE" == "true" ]]; then
    echo "[sol-baseline] $*"
  fi
}

##############################################################################
# Step 1: Dirty tree guard
##############################################################################
if [[ "$ALLOW_DIRTY" != "true" ]]; then
  if ! git -C "$REPO_ROOT" diff --quiet || ! git -C "$REPO_ROOT" diff --cached --quiet; then
    echo "ERROR: Working tree is dirty. Commit or stash changes before benchmarking." >&2
    echo "This ensures reproducible baselines tied to specific commits." >&2
    echo "" >&2
    echo "If running in CI, use --allow-dirty flag." >&2
    exit 1
  fi
fi

##############################################################################
# Step 2: Load config via config-dump
##############################################################################
log "Loading config: $CONFIG (profile: $PROFILE)"

# Build nockchain-bench if binary doesn't exist
BENCH_BIN="$REPO_ROOT/target/release/nockchain-bench"
if [[ ! -x "$BENCH_BIN" ]]; then
  log "Building nockchain-bench (release)..."
  cargo build --release -p nockchain-bench --manifest-path "$REPO_ROOT/Cargo.toml" 2>&1 | \
    if [[ "$VERBOSE" == "true" ]]; then cat; else tail -3; fi
fi

# Dump resolved config as shell variables
CONFIG_DUMP=$("$BENCH_BIN" config-dump --config "$CONFIG" --profile "$PROFILE")
eval "$CONFIG_DUMP"

# Apply CLI overrides
if [[ -n "$PASSES_OVERRIDE" ]]; then
  PASSES="$PASSES_OVERRIDE"
fi
if [[ -n "$OUTPUT_ROOT_OVERRIDE" ]]; then
  OUTPUT_ROOT="$OUTPUT_ROOT_OVERRIDE"
fi

log_verbose "Resolved config: PASSES=$PASSES FIXTURES_DIR=$FIXTURES_DIR OUTPUT_ROOT=$OUTPUT_ROOT"

##############################################################################
# Step 3: Create run directory
##############################################################################
# Keep an ISO-8601 timestamp for machine-readable metadata.
TIMESTAMP_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# Filesystem-safe variant for run directory naming.
TIMESTAMP_SAFE="${TIMESTAMP_ISO//:/-}"
COMMIT_SHA=$(git -C "$REPO_ROOT" rev-parse HEAD)
COMMIT_SHORT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
GIT_BRANCH=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)

RUN_DIR="${OUTPUT_ROOT}/${TIMESTAMP_SAFE}_${COMMIT_SHORT}"
mkdir -p "$RUN_DIR"/{data,meta,logs}

log "Run directory: $RUN_DIR"

##############################################################################
# Step 4: Snapshot config to meta/
##############################################################################
cp "$CONFIG" "$RUN_DIR/meta/config-snapshot.toml"

##############################################################################
# Step 5: Run benchmark matrix
##############################################################################
log "Running benchmark matrix (passes=$PASSES, envs=$ENVS)..."

"$SCRIPT_DIR/sol_bench_matrix.sh" \
  --fixtures-dir "$FIXTURES_DIR" \
  "${BRANCH_BIN_ARGS[@]}" \
  "${DOCKER_IMAGE_ARGS[@]}" \
  --passes "$PASSES" \
  --envs "$ENVS" \
  --docker-memory "$DOCKER_MEMORY" \
  --enable-checkpointing "$ENABLE_CHECKPOINTING" \
  --output-root "$RUN_DIR/data" \
  2> >(tee "$RUN_DIR/logs/run.log" >&2) || {
  BENCH_EXIT=$?
  log "WARNING: Benchmark matrix exited with code $BENCH_EXIT"
  log "Check $RUN_DIR/logs/run.log for details"
}

##############################################################################
# Step 6: Collect provenance and write manifest
##############################################################################
log "Collecting provenance..."

RUSTC_VERSION=$(rustc --version 2>/dev/null || echo "unknown")
CARGO_VERSION=$(cargo --version 2>/dev/null || echo "unknown")
OS_INFO=$(uname -srm 2>/dev/null || echo "unknown")
KERNEL=$(uname -r 2>/dev/null || echo "unknown")
CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*: *//' || echo "unknown")
CPU_CORES=$(nproc 2>/dev/null || echo "1")
CPU_FREQ_RAW=$(lscpu 2>/dev/null | grep "CPU max MHz" | sed 's/.*: *//' | cut -d. -f1 || echo "")
RAM_BYTES=$(free -b 2>/dev/null | awk '/^Mem:/{print $2}' || echo "0")
BENCH_VERSION=$("$BENCH_BIN" --version 2>/dev/null | awk '{print $2}' || echo "unknown")
CONFIG_SHA=$(sha256sum "$CONFIG" | cut -d' ' -f1)

# Build cpu_frequency_mhz as JSON (nullable)
if [[ -n "$CPU_FREQ_RAW" && "$CPU_FREQ_RAW" != "0" ]]; then
  CPU_FREQ_JSON="$CPU_FREQ_RAW"
else
  CPU_FREQ_JSON="null"
fi

# Build benchmark_config from resolved values
BENCHMARK_CONFIG=$(jq -n \
  --arg profile "$PROFILE" \
  --argjson passes "$PASSES" \
  --arg fixtures_dir "$FIXTURES_DIR" \
  --arg enable_checkpointing "$ENABLE_CHECKPOINTING" \
  --arg envs "$ENVS" \
  --arg docker_memory "$DOCKER_MEMORY" \
  '{
    profile: $profile,
    passes: $passes,
    fixtures_dir: $fixtures_dir,
    enable_checkpointing: ($enable_checkpointing == "true"),
    envs: $envs,
    docker_memory: $docker_memory
  }')

# Write manifest.json
jq -n \
  --arg schema_version "1" \
  --arg timestamp "$TIMESTAMP_ISO" \
  --arg git_commit "$COMMIT_SHA" \
  --arg git_branch "$GIT_BRANCH" \
  --argjson benchmark_config "$BENCHMARK_CONFIG" \
  --arg config_sha256 "$CONFIG_SHA" \
  --arg os "$OS_INFO" \
  --arg kernel "$KERNEL" \
  --arg cpu_model "$CPU_MODEL" \
  --argjson cpu_cores "$CPU_CORES" \
  --argjson cpu_frequency_mhz "$CPU_FREQ_JSON" \
  --argjson ram_bytes "$RAM_BYTES" \
  --arg rustc "$RUSTC_VERSION" \
  --arg cargo "$CARGO_VERSION" \
  --arg nockchain_bench "$BENCH_VERSION" \
  '{
    schema_version: $schema_version,
    timestamp: $timestamp,
    git_commit: $git_commit,
    git_branch: $git_branch,
    benchmark_config: $benchmark_config,
    config_sha256: $config_sha256,
    environment: {
      os: $os,
      kernel: $kernel,
      cpu_model: $cpu_model,
      cpu_cores: $cpu_cores,
      cpu_frequency_mhz: $cpu_frequency_mhz,
      ram_bytes: $ram_bytes,
      active_cgroups: null
    },
    tool_versions: {
      rustc: $rustc,
      cargo: $cargo,
      nockchain_bench: $nockchain_bench
    }
  }' > "$RUN_DIR/meta/manifest.json"

log_verbose "Manifest written: $RUN_DIR/meta/manifest.json"

##############################################################################
# Step 7: Update latest symlink (atomic via temp + mv)
##############################################################################
LATEST_LINK="${OUTPUT_ROOT}/latest"
LATEST_TMP="${OUTPUT_ROOT}/.latest-tmp-$$"
# Use a sibling-relative target so "latest" resolves correctly even when
# RUN_DIR was provided as a repo-relative path.
RUN_DIR_NAME="$(basename "$RUN_DIR")"
ln -sfn "$RUN_DIR_NAME" "$LATEST_TMP"
mv -Tf "$LATEST_TMP" "$LATEST_LINK" 2>/dev/null || ln -sfn "$RUN_DIR_NAME" "$LATEST_LINK"

##############################################################################
# Step 8: Print summary
##############################################################################
log "============================================"
log "Baseline run complete"
log "============================================"
log "  Run directory:  $RUN_DIR"
log "  Manifest:       $RUN_DIR/meta/manifest.json"
log "  Config snapshot: $RUN_DIR/meta/config-snapshot.toml"
log "  Logs:           $RUN_DIR/logs/run.log"
log "  Latest:         $LATEST_LINK"
log ""
log "  Git commit:     $COMMIT_SHORT ($GIT_BRANCH)"
log "  Profile:        $PROFILE"
log "  Passes:         $PASSES"
log "  Config SHA-256: ${CONFIG_SHA:0:16}..."
log "============================================"
