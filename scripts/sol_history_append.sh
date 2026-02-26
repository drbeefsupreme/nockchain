#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# sol_history_append.sh — Per-run JSON generation and history index append
#
# Generates a per-run JSON file from a baseline run's manifest.json and
# combined_summary.tsv, then appends an entry to history/index.json.
#
# Usage:
#   sol_history_append.sh MANIFEST_JSON SUMMARY_TSV PUBLISH_DIR [EXISTING_INDEX]
#
# Arguments:
#   MANIFEST_JSON    Path to meta/manifest.json from the baseline run
#   SUMMARY_TSV      Path to data/combined_summary.tsv from the baseline run
#   PUBLISH_DIR      Empty directory where output files will be written
#   EXISTING_INDEX   (optional) Existing index.json content as JSON string,
#                    defaults to '[]'
#
# Output (in PUBLISH_DIR):
#   {RUN_ID}.json    Per-run JSON with provenance + 4 metric medians
#   index.json       Updated history index with new run entry appended
#
# The script only writes new/updated files to PUBLISH_DIR. When used with
# peaceiris/actions-gh-pages keep_files: true, existing history files are
# preserved on gh-pages.
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------
if [[ $# -lt 3 ]]; then
  echo "ERROR: Required arguments missing." >&2
  echo "Usage: $0 MANIFEST_JSON SUMMARY_TSV PUBLISH_DIR [EXISTING_INDEX]" >&2
  exit 2
fi

MANIFEST_JSON="$1"
SUMMARY_TSV="$2"
PUBLISH_DIR="$3"
EXISTING_INDEX="${4:-[]}"

if [[ ! -f "$MANIFEST_JSON" ]]; then
  echo "ERROR: manifest.json not found: $MANIFEST_JSON" >&2
  exit 1
fi

if [[ ! -f "$SUMMARY_TSV" ]]; then
  echo "ERROR: combined_summary.tsv not found: $SUMMARY_TSV" >&2
  exit 1
fi

mkdir -p "$PUBLISH_DIR"

# --------------------------------------------------------------------------
# Step 1: Read provenance from manifest
# --------------------------------------------------------------------------
TIMESTAMP=$(jq -r '.timestamp' "$MANIFEST_JSON")
GIT_COMMIT=$(jq -r '.git_commit' "$MANIFEST_JSON")
GIT_BRANCH=$(jq -r '.git_branch' "$MANIFEST_JSON")
PROFILE=$(jq -r '.benchmark_config.profile' "$MANIFEST_JSON")
CPU_MODEL=$(jq -r '.environment.cpu_model' "$MANIFEST_JSON")
CPU_CORES=$(jq -r '.environment.cpu_cores' "$MANIFEST_JSON")
RAM_BYTES=$(jq -r '.environment.ram_bytes' "$MANIFEST_JSON")

# --------------------------------------------------------------------------
# Step 2: Compute RUN_ID
# Format: {timestamp-with-colons-replaced}_{first7-of-commit}
# Example: 2026-02-26T08-00-00Z_abc1234
# --------------------------------------------------------------------------
TIMESTAMP_SAFE="${TIMESTAMP//:/-}"
COMMIT_SHORT="${GIT_COMMIT:0:7}"
RUN_ID="${TIMESTAMP_SAFE}_${COMMIT_SHORT}"

echo "[sol-history] Run ID: $RUN_ID"

# --------------------------------------------------------------------------
# Step 3: Compute metric medians from TSV using column-name lookup
# Target metrics: throughput_blocks_s, init_time_s, avg_per_block_ms, peak_rss_mib
# --------------------------------------------------------------------------

# Helper: compute median for a named column from the TSV
# Usage: compute_median FILENAME COL_NAME
compute_median() {
  local tsv_file="$1"
  local col_name="$2"

  awk -v col="$col_name" '
  BEGIN { FS="\t" }
  NR==1 {
    # Find the column index from header
    col_idx = -1
    for (i=1; i<=NF; i++) {
      if ($i == col) {
        col_idx = i
        break
      }
    }
    if (col_idx == -1) {
      print "ERROR: column not found: " col > "/dev/stderr"
      exit 1
    }
    next
  }
  NR>1 {
    # Skip rows with empty values
    if ($col_idx != "") {
      values[n++] = $col_idx + 0
    }
  }
  END {
    if (n == 0) {
      print "0"
      exit 0
    }
    # Sort ascending
    for (i=0; i<n-1; i++) {
      for (j=i+1; j<n; j++) {
        if (values[i] > values[j]) {
          tmp = values[i]; values[i] = values[j]; values[j] = tmp
        }
      }
    }
    # Median
    if (n % 2 == 1) {
      print values[int(n/2)]
    } else {
      print (values[n/2 - 1] + values[n/2]) / 2
    }
  }
  ' "$tsv_file"
}

# Count data rows (excluding header)
PASS_COUNT=$(awk 'NR>1 && NF>0' "$SUMMARY_TSV" | wc -l)

MEDIAN_THROUGHPUT=$(compute_median "$SUMMARY_TSV" "throughput_blocks_s")
MEDIAN_INIT_TIME=$(compute_median "$SUMMARY_TSV" "init_time_s")
MEDIAN_AVG_PER_BLOCK=$(compute_median "$SUMMARY_TSV" "avg_per_block_ms")
MEDIAN_PEAK_RSS=$(compute_median "$SUMMARY_TSV" "peak_rss_mib")

echo "[sol-history] Medians: throughput=${MEDIAN_THROUGHPUT} init=${MEDIAN_INIT_TIME} avg_block=${MEDIAN_AVG_PER_BLOCK} peak_rss=${MEDIAN_PEAK_RSS}"

# --------------------------------------------------------------------------
# Step 4: Build per-run JSON
# --------------------------------------------------------------------------
RUN_JSON_PATH="${PUBLISH_DIR}/${RUN_ID}.json"

jq -n \
  --arg run_id "$RUN_ID" \
  --arg timestamp "$TIMESTAMP" \
  --arg git_commit "$GIT_COMMIT" \
  --arg git_branch "$GIT_BRANCH" \
  --arg profile "$PROFILE" \
  --argjson passes "$PASS_COUNT" \
  --argjson throughput_blocks_s "$MEDIAN_THROUGHPUT" \
  --argjson init_time_s "$MEDIAN_INIT_TIME" \
  --argjson avg_per_block_ms "$MEDIAN_AVG_PER_BLOCK" \
  --argjson peak_rss_mib "$MEDIAN_PEAK_RSS" \
  --arg cpu_model "$CPU_MODEL" \
  --argjson cpu_cores "$CPU_CORES" \
  --argjson ram_bytes "$RAM_BYTES" \
  '{
    run_id: $run_id,
    timestamp: $timestamp,
    git_commit: $git_commit,
    git_branch: $git_branch,
    profile: $profile,
    passes: $passes,
    metrics: {
      throughput_blocks_s_median: $throughput_blocks_s,
      init_time_s_median: $init_time_s,
      avg_per_block_ms_median: $avg_per_block_ms,
      peak_rss_mib_median: $peak_rss_mib
    },
    environment: {
      cpu_model: $cpu_model,
      cpu_cores: $cpu_cores,
      ram_bytes: $ram_bytes
    }
  }' > "$RUN_JSON_PATH"

echo "[sol-history] Written: $RUN_JSON_PATH"

# --------------------------------------------------------------------------
# Step 5: Update history/index.json
# Append new index entry (is_active_baseline defaults to false)
# --------------------------------------------------------------------------
INDEX_JSON_PATH="${PUBLISH_DIR}/index.json"

NEW_ENTRY=$(jq -n \
  --arg run_id "$RUN_ID" \
  --arg filename "${RUN_ID}.json" \
  --arg timestamp "$TIMESTAMP" \
  --arg git_commit "$GIT_COMMIT" \
  --arg git_branch "$GIT_BRANCH" \
  --arg profile "$PROFILE" \
  '{
    run_id: $run_id,
    filename: $filename,
    timestamp: $timestamp,
    git_commit: $git_commit,
    git_branch: $git_branch,
    profile: $profile,
    is_active_baseline: false
  }')

# Append new entry to existing index
echo "$EXISTING_INDEX" | jq --argjson entry "$NEW_ENTRY" '. + [$entry]' > "$INDEX_JSON_PATH"

echo "[sol-history] Index updated: $INDEX_JSON_PATH ($(jq 'length' "$INDEX_JSON_PATH") entries)"
echo "[sol-history] Done."
