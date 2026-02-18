#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run sequential SOL benchmark matrix with memory profiling and summary TSV.

Usage:
  scripts/sol_bench_matrix.sh \
    --fixtures-dir <dir-with-v0-100.soltest,v1-100.soltest,v2-100.soltest> \
    --branch-bin <name=/path/to/nockchain-bench> [--branch-bin ...] \
    [--docker-image <name=image>] [--docker-image ...] \
    [--passes 2] [--envs native,docker] \
    [--docker-memory 16g] \
    [--output-root <dir>] \
    [--enable-checkpointing false]

Examples:
  scripts/sol_bench_matrix.sh \
    --fixtures-dir /shared/nockchain-ext4-bench/artifacts/fixtures-legacy \
    --branch-bin master=/shared/nockchain-ext4-bench/master/target/release/nockchain-bench \
    --branch-bin streaming=/shared/nockchain-ext4-bench/streaming/target/release/nockchain-bench \
    --branch-bin btree=/shared/nockchain-ext4-bench/btree/target/release/nockchain-bench \
    --docker-image master=nockbench-master:latest \
    --docker-image streaming=nockbench-streaming:latest \
    --docker-image btree=nockbench-btree:latest \
    --passes 2 --envs native,docker --docker-memory 16g \
    --output-root /shared/nockchain-ext4-bench/artifacts/runs
USAGE
}

FIXTURES_DIR=""
PASSES=2
ENVS_CSV="native,docker"
DOCKER_MEMORY="16g"
OUTPUT_ROOT="./bench-artifacts/sol-matrix"
ENABLE_CHECKPOINTING="false"

declare -A BRANCH_BIN
declare -A BRANCH_IMAGE
BRANCH_ORDER=()

add_branch_bin() {
  local spec="$1"
  [[ "$spec" == *=* ]] || { echo "Invalid --branch-bin '$spec'" >&2; exit 2; }
  local name="${spec%%=*}"
  local path="${spec#*=}"
  if [[ -z "$name" || -z "$path" ]]; then
    echo "Invalid --branch-bin '$spec'" >&2
    exit 2
  fi
  if [[ -z "${BRANCH_BIN[$name]:-}" ]]; then
    BRANCH_ORDER+=("$name")
  fi
  BRANCH_BIN[$name]="$path"
}

add_branch_image() {
  local spec="$1"
  [[ "$spec" == *=* ]] || { echo "Invalid --docker-image '$spec'" >&2; exit 2; }
  local name="${spec%%=*}"
  local image="${spec#*=}"
  if [[ -z "$name" || -z "$image" ]]; then
    echo "Invalid --docker-image '$spec'" >&2
    exit 2
  fi
  BRANCH_IMAGE[$name]="$image"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fixtures-dir) FIXTURES_DIR="$2"; shift 2 ;;
    --branch-bin) add_branch_bin "$2"; shift 2 ;;
    --docker-image) add_branch_image "$2"; shift 2 ;;
    --passes) PASSES="$2"; shift 2 ;;
    --envs) ENVS_CSV="$2"; shift 2 ;;
    --docker-memory) DOCKER_MEMORY="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --enable-checkpointing) ENABLE_CHECKPOINTING="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$FIXTURES_DIR" ]] || { echo "--fixtures-dir is required" >&2; usage; exit 2; }
[[ -d "$FIXTURES_DIR" ]] || { echo "Fixture dir not found: $FIXTURES_DIR" >&2; exit 2; }
[[ ${#BRANCH_ORDER[@]} -gt 0 ]] || { echo "At least one --branch-bin is required" >&2; exit 2; }
[[ "$PASSES" =~ ^[0-9]+$ ]] || { echo "--passes must be numeric" >&2; exit 2; }
(( PASSES >= 1 )) || { echo "--passes must be >= 1" >&2; exit 2; }

for name in "${BRANCH_ORDER[@]}"; do
  [[ -x "${BRANCH_BIN[$name]}" ]] || { echo "Branch binary missing/executable: $name=${BRANCH_BIN[$name]}" >&2; exit 2; }
done

FIXTURE_IDS=(v0 v1 v2)
fixture_path() {
  echo "$FIXTURES_DIR/$1-100.soltest"
}

for id in "${FIXTURE_IDS[@]}"; do
  fp="$(fixture_path "$id")"
  [[ -f "$fp" ]] || { echo "Missing fixture: $fp" >&2; exit 2; }
done

IFS=',' read -r -a ENVS <<< "$ENVS_CSV"
for env in "${ENVS[@]}"; do
  if [[ "$env" != "native" && "$env" != "docker" ]]; then
    echo "Invalid env '$env' in --envs (supported: native,docker)" >&2
    exit 2
  fi
  if [[ "$env" == "docker" ]]; then
    for name in "${BRANCH_ORDER[@]}"; do
      [[ -n "${BRANCH_IMAGE[$name]:-}" ]] || { echo "Missing --docker-image for branch '$name'" >&2; exit 2; }
    done
  fi
done

RUN_ROOT="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)-sol-matrix"
mkdir -p "$RUN_ROOT"
SUMMARY_TSV="$RUN_ROOT/combined_summary.tsv"
cat > "$SUMMARY_TSV" <<'TSV'
pass	env	branch	fixture	blocks_poked	failed_pokes	init_time_s	total_poke_time_s	avg_per_block_ms	throughput_blocks_s	checkpoints	samples	gc_events	fault_bursts	peak_rss_mib	p95_rss_mib	peak_pma_rss_mib	peak_pma_size_mib	peak_nockstack_rss_mib	peak_anon_mib	peak_file_mib	minor_faults_delta	major_faults_delta	wall_clock	max_rss_kb	exit_status	profile_path	bench_log	time_log
TSV

num_or_zero() {
  local profile="$1"; local expr="$2"
  jq -r "$expr // 0" "$profile" 2>/dev/null || echo 0
}

sample_max_mib() {
  local profile="$1"; local field="$2"
  jq -r "((.memory_profile.samples // []) | if length > 0 then (map(.$field // 0) | max) / 1024 else 0 end)" "$profile" 2>/dev/null || echo 0
}

fault_delta() {
  local profile="$1"; local field="$2"
  jq -r "((.memory_profile.samples // []) as \$s | if (\$s | length) > 1 then ((\$s[-1].$field // 0) - (\$s[0].$field // 0)) else 0 end)" "$profile" 2>/dev/null || echo 0
}

read_time_field() {
  local time_file="$1"; local needle="$2"
  awk -F': ' -v n="$needle" 'index($0, n) > 0 {print $2}' "$time_file" | tail -n1 | tr -d '\r'
}

run_native() {
  local branch="$1" fixture_id="$2" run_dir="$3"
  local profile="$run_dir/profile.json"
  local log="$run_dir/bench.log"
  local time_log="$run_dir/time.txt"
  local fixture="$(fixture_path "$fixture_id")"

  set +e
  /usr/bin/time -v "${BRANCH_BIN[$branch]}" sol bench \
    --fixture "$fixture" \
    --enable-checkpointing="$ENABLE_CHECKPOINTING" \
    --profile-memory \
    --profile-output "$profile" \
    >"$log" 2>"$time_log"
  local exit_code=$?
  set -e
  echo "$exit_code"
}

run_docker() {
  local branch="$1" fixture_id="$2" run_dir="$3"
  local profile="$run_dir/profile.json"
  local log="$run_dir/bench.log"
  local time_log="$run_dir/time.txt"
  local fixture="/bench/fixtures/$fixture_id-100.soltest"
  local image="${BRANCH_IMAGE[$branch]}"
  local cname="sol-matrix-$(date +%s)-$RANDOM"

  docker rm -f "$cname" >/dev/null 2>&1 || true
  if ! docker create --name "$cname" --memory="$DOCKER_MEMORY" "$image" \
      sol bench --fixture "$fixture" --enable-checkpointing="$ENABLE_CHECKPOINTING" \
      --profile-memory --profile-output /tmp/profile.json >/dev/null; then
    echo "docker create failed" >"$log"
    : >"$time_log"
    echo 125
    return
  fi

  set +e
  /usr/bin/time -v docker start -a "$cname" >"$log" 2>"$time_log"
  local exit_code=$?
  set -e

  docker cp "$cname:/tmp/profile.json" "$profile" >/dev/null 2>&1 || true
  docker rm -f "$cname" >/dev/null 2>&1 || true
  echo "$exit_code"
}

append_summary_row() {
  local pass="$1" env="$2" branch="$3" fixture_id="$4" run_dir="$5" exit_code="$6"
  local profile="$run_dir/profile.json"
  local time_log="$run_dir/time.txt"
  local log="$run_dir/bench.log"

  local blocks_poked=0 failed_pokes=0 init_time_s=0 total_poke_time_s=0 avg_per_block_ms=0 throughput_blocks_s=0
  local checkpoints=0 samples=0 gc_events=0 fault_bursts=0 peak_rss_mib=0 p95_rss_mib=0
  local peak_pma_rss_mib=0 peak_pma_size_mib=0 peak_nockstack_rss_mib=0 peak_anon_mib=0 peak_file_mib=0
  local minor_faults_delta=0 major_faults_delta=0

  if [[ -f "$profile" ]]; then
    blocks_poked=$(num_or_zero "$profile" '.blocks_poked')
    failed_pokes=$(num_or_zero "$profile" '.failed_pokes')
    init_time_s=$(num_or_zero "$profile" '.init_time_secs')
    total_poke_time_s=$(num_or_zero "$profile" '.total_poke_time_secs')
    throughput_blocks_s=$(num_or_zero "$profile" '.blocks_per_second')
    checkpoints=$(num_or_zero "$profile" '.checkpoint_count')
    samples=$(jq -r '((.memory_profile.samples // []) | length)' "$profile" 2>/dev/null || echo 0)
    gc_events=$(jq -r '((.memory_profile.gc_events // []) | length)' "$profile" 2>/dev/null || echo 0)
    fault_bursts=$(jq -r '((.memory_profile.page_fault_bursts // []) | length)' "$profile" 2>/dev/null || echo 0)
    peak_rss_mib=$(num_or_zero "$profile" '.memory_profile.scorecard.peak_rss_mib')
    p95_rss_mib=$(num_or_zero "$profile" '.memory_profile.scorecard.p95_rss_mib')

    peak_pma_rss_mib=$(sample_max_mib "$profile" 'pma_rss_kb')
    peak_pma_size_mib=$(sample_max_mib "$profile" 'pma_size_kb')
    peak_nockstack_rss_mib=$(sample_max_mib "$profile" 'nockstack_rss_kb')
    peak_anon_mib=$(sample_max_mib "$profile" 'rss_anon_kb')
    peak_file_mib=$(sample_max_mib "$profile" 'rss_file_kb')
    minor_faults_delta=$(fault_delta "$profile" 'minor_faults')
    major_faults_delta=$(fault_delta "$profile" 'major_faults')
    avg_per_block_ms=$(jq -r 'if (.blocks_poked // 0) > 0 then ((.total_poke_time_secs // 0) / (.blocks_poked) * 1000) else 0 end' "$profile" 2>/dev/null || echo 0)
  fi

  local wall_clock="" max_rss_kb=""
  if [[ -f "$time_log" ]]; then
    wall_clock=$(read_time_field "$time_log" 'Elapsed (wall clock) time')
    max_rss_kb=$(read_time_field "$time_log" 'Maximum resident set size')
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$pass" "$env" "$branch" "$fixture_id" \
    "$blocks_poked" "$failed_pokes" "$init_time_s" "$total_poke_time_s" "$avg_per_block_ms" "$throughput_blocks_s" \
    "$checkpoints" "$samples" "$gc_events" "$fault_bursts" "$peak_rss_mib" "$p95_rss_mib" \
    "$peak_pma_rss_mib" "$peak_pma_size_mib" "$peak_nockstack_rss_mib" "$peak_anon_mib" "$peak_file_mib" \
    "$minor_faults_delta" "$major_faults_delta" "$wall_clock" "$max_rss_kb" "$exit_code" \
    "$profile" "$log" "$time_log" \
    >>"$SUMMARY_TSV"
}

total_runs=$(( PASSES * ${#ENVS[@]} * ${#BRANCH_ORDER[@]} * ${#FIXTURE_IDS[@]} ))
run_counter=0

echo "Run root: $RUN_ROOT"
echo "Summary TSV: $SUMMARY_TSV"
echo "Total runs: $total_runs"

for ((pass=1; pass<=PASSES; pass++)); do
  for env in "${ENVS[@]}"; do
    for branch in "${BRANCH_ORDER[@]}"; do
      for fixture_id in "${FIXTURE_IDS[@]}"; do
        run_counter=$((run_counter + 1))
        run_dir="$RUN_ROOT/runs/pass${pass}/${env}/${branch}/${fixture_id}"
        mkdir -p "$run_dir"
        echo "[$run_counter/$total_runs] pass=$pass env=$env branch=$branch fixture=$fixture_id"

        if [[ "$env" == "native" ]]; then
          exit_code=$(run_native "$branch" "$fixture_id" "$run_dir")
        else
          exit_code=$(run_docker "$branch" "$fixture_id" "$run_dir")
        fi

        append_summary_row "$pass" "$env" "$branch" "$fixture_id" "$run_dir" "$exit_code"
        echo "  exit_status=$exit_code"
      done
    done
  done
done

echo "Completed matrix run"
echo "Summary TSV: $SUMMARY_TSV"
