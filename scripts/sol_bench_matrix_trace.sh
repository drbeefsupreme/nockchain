#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ROOT="${ROOT:-/shared/nockchain-ext4-bench}"
FIX_DIR="${FIX_DIR:-$ROOT/artifacts/fixtures-legacy}"
RUN_BASE="${RUN_BASE:-/tmp/sol-trace-runs}"
RUN_TAG="${RUN_TAG:-sol-100x2-tracy-perf}"
RUN_ROOT="${RUN_BASE}/$(date +%Y%m%d_%H%M%S)-${RUN_TAG}"
PASSES_CSV="${PASSES_CSV:-1,2}"
ENVS_CSV="${ENVS_CSV:-native,docker}"
FIXTURE_FILE_V0="${FIXTURE_FILE_V0:-v0-100.soltest}"
FIXTURE_FILE_V1="${FIXTURE_FILE_V1:-v1-100.soltest}"
FIXTURE_FILE_V2="${FIXTURE_FILE_V2:-v2-100.soltest}"
FIXTURE_SPECS="${FIXTURE_SPECS:-v0=${FIXTURE_FILE_V0},v1=${FIXTURE_FILE_V1},v2=${FIXTURE_FILE_V2}}"

DOCKER_MEMORY="${DOCKER_MEMORY:-16g}"
PROFILE_MEMORY="${PROFILE_MEMORY:-false}"
PERF_NATIVE="${PERF_NATIVE:-true}"
PERF_DOCKER="${PERF_DOCKER:-true}"
PERF_FREQ="${PERF_FREQ:-199}"
# Use user-space-only cycles to avoid unresolved kernel stacks.
PERF_EVENT="${PERF_EVENT:-cycles:u}"
PERF_CALL_GRAPH="${PERF_CALL_GRAPH:-fp}"
TRACY_CAPTURE_NATIVE="${TRACY_CAPTURE_NATIVE:-true}"
TRACY_CAPTURE_DOCKER="${TRACY_CAPTURE_DOCKER:-true}"
TRACY_CAPTURE_PRELAUNCH_SEC="${TRACY_CAPTURE_PRELAUNCH_SEC:-1}"
TRACY_CAPTURE_TIMEOUT_SEC="${TRACY_CAPTURE_TIMEOUT_SEC:-45}"
TRACY_CAPTURE_SECONDS="${TRACY_CAPTURE_SECONDS:-120}"
TRACY_PORT="${TRACY_PORT:-8086}"
TRACY_CAPTURE_MASTER_BIN="${TRACY_CAPTURE_MASTER_BIN:-/tmp/tracy-capture-0.13.1}"
TRACY_CAPTURE_LEGACY_BIN="${TRACY_CAPTURE_LEGACY_BIN:-/tmp/tracy-capture-0.12.2}"
DOCKER_USE_HOST_NETWORK="${DOCKER_USE_HOST_NETWORK:-true}"
NOCK_TRACING="${NOCK_TRACING:-false}"
NOCK_TRACE_MODE="${NOCK_TRACE_MODE:-}"

if [[ -z "$NOCK_TRACE_MODE" && "$NOCK_TRACING" == "true" ]]; then
  NOCK_TRACE_MODE="tracing"
fi

if [[ -n "$NOCK_TRACE_MODE" && "$NOCK_TRACE_MODE" != "tracing" ]]; then
  echo "Unsupported NOCK_TRACE_MODE='$NOCK_TRACE_MODE' (expected 'tracing')" >&2
  exit 1
fi
SOL_GUARD_POST_RUN="${SOL_GUARD_POST_RUN:-false}"
SOL_GUARD_CONTRACT="${SOL_GUARD_CONTRACT:-}"
SOL_GUARD_BASELINE_SUMMARY="${SOL_GUARD_BASELINE_SUMMARY:-}"
SOL_GUARD_STRICT="${SOL_GUARD_STRICT:-false}"

mkdir -p "$RUN_ROOT"

SUMMARY_TSV="$RUN_ROOT/combined_summary.tsv"
cat > "$SUMMARY_TSV" <<'TSV'
pass	env	branch	fixture	blocks_poked	failed_pokes	init_time_s	total_poke_time_s	avg_per_block_ms	throughput_blocks_s	checkpoints	samples	gc_events	fault_bursts	peak_rss_mib	p95_rss_mib	peak_pma_rss_mib	peak_pma_size_mib	peak_nockstack_rss_mib	peak_anon_mib	peak_file_mib	minor_faults_delta	major_faults_delta	wall_clock	max_rss_kb	exit_status	profile_path	bench_log	time_log	perf_samples	perf_unique_stacks	perf_summary_json	perf_flamegraph_svg	perf_report_txt	perf_folded	perf_data	tracy_trace	tracy_capture_log	tracy_size_bytes	tracy_frames	tracy_time_span_s	tracy_zones
TSV

BRANCH_IDS=(master streaming btree)
ENVS=()
IFS=',' read -r -a PASSES <<< "$PASSES_CSV"
IFS=',' read -r -a ENVS <<< "$ENVS_CSV"
FIXTURE_IDS=()
declare -A FIXTURE_FILE_BY_ID=()

[[ ${#ENVS[@]} -gt 0 ]] || { echo "ENVS_CSV is empty" >&2; exit 1; }
for env_name in "${ENVS[@]}"; do
  case "$env_name" in
    native|docker) ;;
    *) echo "Unsupported env '$env_name' in ENVS_CSV (expected native,docker entries)" >&2; exit 1 ;;
  esac
done

parse_fixture_specs() {
  local specs=()
  local spec
  IFS=',' read -r -a specs <<< "$FIXTURE_SPECS"
  [[ ${#specs[@]} -gt 0 ]] || { echo "FIXTURE_SPECS is empty" >&2; exit 1; }

  for spec in "${specs[@]}"; do
    [[ "$spec" == *=* ]] || { echo "Invalid fixture spec '$spec' (expected id=file)" >&2; exit 1; }
    local id="${spec%%=*}"
    local file="${spec#*=}"
    [[ -n "$id" && -n "$file" ]] || {
      echo "Invalid fixture spec '$spec' (empty id or file)" >&2
      exit 1
    }
    if [[ -n "${FIXTURE_FILE_BY_ID[$id]:-}" ]]; then
      echo "Duplicate fixture id in FIXTURE_SPECS: $id" >&2
      exit 1
    fi
    FIXTURE_IDS+=("$id")
    FIXTURE_FILE_BY_ID["$id"]="$file"
  done
}

parse_fixture_specs

fixture_file() {
  local id="$1"
  local file="${FIXTURE_FILE_BY_ID[$id]:-}"
  [[ -n "$file" ]] || { echo "unknown fixture id: $id" >&2; exit 1; }
  echo "$file"
}

branch_label() {
  case "$1" in
    master) echo "master" ;;
    streaming) echo "bump PMA" ;;
    btree) echo "btree" ;;
    *) echo "$1" ;;
  esac
}

branch_bin() {
  echo "$ROOT/$1/target/release/nockchain-bench"
}

branch_image() {
  echo "nockbench-$1:latest"
}

branch_uses_pma() {
  case "$1" in
    streaming|btree) return 0 ;;
    *) return 1 ;;
  esac
}

branch_tracy_capture_bin() {
  case "$1" in
    master) echo "$TRACY_CAPTURE_MASTER_BIN" ;;
    streaming|btree) echo "$TRACY_CAPTURE_LEGACY_BIN" ;;
    *) echo "$TRACY_CAPTURE_LEGACY_BIN" ;;
  esac
}

num_or_zero() {
  local profile="$1"
  local expr="$2"
  jq -r "$expr // 0" "$profile" 2>/dev/null || echo 0
}

sample_max_mib() {
  local profile="$1"
  local field="$2"
  jq -r "((.memory_profile.samples // []) | if length > 0 then (map(.$field // 0) | max) / 1024 else 0 end)" "$profile" 2>/dev/null || echo 0
}

fault_delta() {
  local profile="$1"
  local field="$2"
  jq -r "((.memory_profile.samples // []) as \$s | if (\$s | length) > 1 then ((\$s[-1].$field // 0) - (\$s[0].$field // 0)) else 0 end)" "$profile" 2>/dev/null || echo 0
}

read_time_field() {
  local time_file="$1"
  local needle="$2"
  awk -F': ' -v n="$needle" 'index($0, n) > 0 {print $2}' "$time_file" | tail -n1 | tr -d '\r'
}

read_bench_field() {
  local log_file="$1"
  local needle="$2"
  awk -F': *' -v n="$needle" '$1 == n {print $2}' "$log_file" | tail -n1 | tr -d '\r'
}

perf_postprocess() {
  local perf_data="$1"
  local perf_script="$2"
  local perf_folded="$3"
  local perf_svg="$4"
  local perf_summary="$5"
  local perf_report="$6"
  local label="$7"

  if [[ ! -s "$perf_data" ]]; then
    return 0
  fi

  perf script -i "$perf_data" >"$perf_script" 2>"$perf_script.err" || true
  if [[ -s "$perf_script" ]]; then
    python3 "$SCRIPT_DIR/perf_script_to_folded.py" --input "$perf_script" --output "$perf_folded"
  fi
  if [[ -s "$perf_folded" ]]; then
    python3 "$SCRIPT_DIR/perf_folded_summary.py" --input "$perf_folded" --output "$perf_summary"
    python3 "$SCRIPT_DIR/folded_to_flamegraph_svg.py" \
      --input "$perf_folded" \
      --output "$perf_svg" \
      --title "SOL perf flamegraph: $label"
  fi
  perf report -i "$perf_data" --stdio --no-children --sort comm,dso,symbol >"$perf_report" 2>&1 || true
}

run_native() {
  local branch="$1"
  local fixture_id="$2"
  local run_dir="$3"

  local fixture="$FIX_DIR/$(fixture_file "$fixture_id")"
  local profile="$run_dir/profile.json"
  local log="$run_dir/bench.log"
  local time_log="$run_dir/time.txt"
  local bin
  bin="$(branch_bin "$branch")"

  local perf_data="$run_dir/perf.data"
  local perf_script="$run_dir/perf.script"
  local perf_folded="$run_dir/perf.folded"
  local perf_svg="$run_dir/perf-flamegraph.svg"
  local perf_summary="$run_dir/perf-summary.json"
  local perf_report="$run_dir/perf-report.txt"
  local tracy_file="$run_dir/trace.tracy"
  local tracy_log="$run_dir/trace-capture.log"
  local tracy_pid=""

  local cmd=(
    "$bin" sol bench
    --fixture "$fixture"
    --enable-checkpointing=false
  )
  if [[ -n "$NOCK_TRACE_MODE" ]]; then
    cmd+=(--trace "$NOCK_TRACE_MODE")
  fi
  if [[ "$PROFILE_MEMORY" == "true" ]]; then
    cmd+=(--profile-memory --profile-output "$profile")
  fi

  if [[ "$TRACY_CAPTURE_NATIVE" == "true" ]]; then
    local tracy_capture_bin
    tracy_capture_bin="$(branch_tracy_capture_bin "$branch")"
    if [[ -x "$tracy_capture_bin" ]]; then
      timeout "${TRACY_CAPTURE_TIMEOUT_SEC}s" \
        "$tracy_capture_bin" \
        -a 127.0.0.1 \
        -p "$TRACY_PORT" \
        -o "$tracy_file" \
        -f \
        -s "$TRACY_CAPTURE_SECONDS" \
        >"$tracy_log" 2>&1 &
      tracy_pid=$!
      sleep "$TRACY_CAPTURE_PRELAUNCH_SEC"
    else
      echo "missing tracy capture binary: $tracy_capture_bin" >"$tracy_log"
    fi
  fi

  set +e
  local env_vars=(TRACY_NO_INVARIANT_CHECK=1)
  if branch_uses_pma "$branch"; then
    env_vars+=(NOCK_PMA_PERSIST=1)
  fi
  if [[ "$NOCK_TRACING" == "true" ]]; then
    env_vars+=(TRACY_ONLY_NOCKCODE=1)
  fi
  if [[ "$PERF_NATIVE" == "true" ]]; then
    /usr/bin/time -v \
      env "${env_vars[@]}" \
      perf record --quiet -e "$PERF_EVENT" -F "$PERF_FREQ" -g --call-graph "$PERF_CALL_GRAPH" -o "$perf_data" -- \
      "${cmd[@]}" >"$log" 2>"$time_log"
  else
    /usr/bin/time -v \
      env "${env_vars[@]}" \
      "${cmd[@]}" >"$log" 2>"$time_log"
  fi
  local exit_code=$?
  set -e

  if [[ "$PERF_NATIVE" == "true" ]]; then
    perf_postprocess \
      "$perf_data" \
      "$perf_script" \
      "$perf_folded" \
      "$perf_svg" \
      "$perf_summary" \
      "$perf_report" \
      "native/${branch}/${fixture_id}"
  fi

  if [[ -n "$tracy_pid" ]]; then
    local waited=0
    while kill -0 "$tracy_pid" 2>/dev/null && [[ "$waited" -lt 8 ]]; do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 "$tracy_pid" 2>/dev/null; then
      kill "$tracy_pid" >/dev/null 2>&1 || true
    fi
    wait "$tracy_pid" >/dev/null 2>&1 || true
  fi

  echo "$exit_code"
}

run_docker() {
  local branch="$1"
  local fixture_id="$2"
  local run_dir="$3"

  local fixture="/bench/fixtures/$(fixture_file "$fixture_id")"
  local profile="$run_dir/profile.json"
  local log="$run_dir/bench.log"
  local time_log="$run_dir/time.txt"
  local perf_data="$run_dir/perf.data"
  local perf_script="$run_dir/perf.script"
  local perf_folded="$run_dir/perf.folded"
  local perf_svg="$run_dir/perf-flamegraph.svg"
  local perf_summary="$run_dir/perf-summary.json"
  local perf_report="$run_dir/perf-report.txt"
  local perf_record_log="$run_dir/perf-record.log"
  local tracy_file="$run_dir/trace.tracy"
  local tracy_log="$run_dir/trace-capture.log"
  local image
  image="$(branch_image "$branch")"
  local cname="sol100trace-$(date +%s)-$RANDOM"
  local tracy_pid=""
  local perf_pid=""
  local container_pid="0"

  local docker_cmd=(
    sol bench
    --fixture "$fixture"
    --enable-checkpointing=false
  )
  if [[ -n "$NOCK_TRACE_MODE" ]]; then
    docker_cmd+=(--trace "$NOCK_TRACE_MODE")
  fi
  if [[ "$PROFILE_MEMORY" == "true" ]]; then
    docker_cmd+=(--profile-memory --profile-output /tmp/profile.json)
  fi

  local create_args=(
    --name "$cname"
    --memory="$DOCKER_MEMORY"
    --memory-swap="$DOCKER_MEMORY"
    --memory-swappiness=0
    --user "$(id -u):$(id -g)"
    --workdir /tmp
    --volume "$FIX_DIR:/bench/fixtures:ro"
    --env TRACY_NO_INVARIANT_CHECK=1
  )
  if [[ "$DOCKER_USE_HOST_NETWORK" == "true" ]]; then
    create_args+=(--network host)
  fi
  if branch_uses_pma "$branch"; then
    create_args+=(--env NOCK_PMA_PERSIST=1)
  fi
  if [[ "$NOCK_TRACING" == "true" ]]; then
    create_args+=(--env TRACY_ONLY_NOCKCODE=1)
  fi

  docker rm -f "$cname" >/dev/null 2>&1 || true
  if ! docker create \
    "${create_args[@]}" \
    "$image" \
    "${docker_cmd[@]}" >/dev/null; then
    echo "docker create failed" >"$log"
    : >"$time_log"
    echo 125
    return
  fi

  if [[ "$TRACY_CAPTURE_DOCKER" == "true" ]]; then
    local tracy_capture_bin
    tracy_capture_bin="$(branch_tracy_capture_bin "$branch")"
    if [[ -x "$tracy_capture_bin" ]]; then
      timeout "${TRACY_CAPTURE_TIMEOUT_SEC}s" \
        "$tracy_capture_bin" \
        -a 127.0.0.1 \
        -p "$TRACY_PORT" \
        -o "$tracy_file" \
        -f \
        -s "$TRACY_CAPTURE_SECONDS" \
        >"$tracy_log" 2>&1 &
      tracy_pid=$!
      sleep "$TRACY_CAPTURE_PRELAUNCH_SEC"
    else
      echo "missing tracy capture binary: $tracy_capture_bin" >"$tracy_log"
    fi
  fi

  set +e

  /usr/bin/time -v docker start -a "$cname" >"$log" 2>"$time_log" &
  local start_pid=$!

  if [[ "$PERF_DOCKER" == "true" ]]; then
    local tries=0
    while [[ "$tries" -lt 200 ]]; do
      container_pid=$(docker inspect -f '{{.State.Pid}}' "$cname" 2>/dev/null || echo 0)
      local running
      running=$(docker inspect -f '{{.State.Running}}' "$cname" 2>/dev/null || echo false)
      if [[ "$container_pid" =~ ^[0-9]+$ ]] && [[ "$container_pid" -gt 0 ]] && [[ "$running" == "true" ]]; then
        break
      fi
      if ! kill -0 "$start_pid" 2>/dev/null; then
        break
      fi
      tries=$((tries + 1))
      sleep 0.05
    done

    if [[ "$container_pid" =~ ^[0-9]+$ ]] && [[ "$container_pid" -gt 0 ]]; then
      perf record --quiet -e "$PERF_EVENT" -F "$PERF_FREQ" -g --call-graph "$PERF_CALL_GRAPH" -o "$perf_data" -p "$container_pid" >"$perf_record_log" 2>&1 &
      perf_pid=$!
    else
      echo "unable to attach perf to container pid for $cname" >"$perf_record_log"
    fi
  fi

  wait "$start_pid"
  local exit_code=$?

  if [[ -n "$perf_pid" ]]; then
    if kill -0 "$perf_pid" 2>/dev/null; then
      kill -INT "$perf_pid" >/dev/null 2>&1 || true
    fi
    wait "$perf_pid" >/dev/null 2>&1 || true
  fi
  set -e

  if [[ "$PERF_DOCKER" == "true" ]]; then
    perf_postprocess \
      "$perf_data" \
      "$perf_script" \
      "$perf_folded" \
      "$perf_svg" \
      "$perf_summary" \
      "$perf_report" \
      "docker/${branch}/${fixture_id}"
  fi

  if [[ -n "$tracy_pid" ]]; then
    local waited=0
    while kill -0 "$tracy_pid" 2>/dev/null && [[ "$waited" -lt 8 ]]; do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 "$tracy_pid" 2>/dev/null; then
      kill "$tracy_pid" >/dev/null 2>&1 || true
    fi
    wait "$tracy_pid" >/dev/null 2>&1 || true
  fi

  if [[ "$PROFILE_MEMORY" == "true" ]]; then
    docker cp "$cname:/tmp/profile.json" "$profile" >/dev/null 2>&1 || true
  fi
  docker rm -f "$cname" >/dev/null 2>&1 || true

  echo "$exit_code"
}

collect_and_append() {
  local pass="$1"
  local env="$2"
  local branch="$3"
  local fixture_id="$4"
  local run_dir="$5"
  local exit_code="$6"

  local profile="$run_dir/profile.json"
  local time_log="$run_dir/time.txt"
  local log="$run_dir/bench.log"
  local perf_summary="$run_dir/perf-summary.json"
  local perf_svg="$run_dir/perf-flamegraph.svg"
  local perf_report="$run_dir/perf-report.txt"
  local perf_folded="$run_dir/perf.folded"
  local perf_data="$run_dir/perf.data"
  local tracy_trace="$run_dir/trace.tracy"
  local tracy_capture_log="$run_dir/trace-capture.log"

  local blocks_poked=0
  local failed_pokes=0
  local init_time_s=0
  local total_poke_time_s=0
  local avg_per_block_ms=0
  local throughput_blocks_s=0
  local checkpoints=0
  local samples=0
  local gc_events=0
  local fault_bursts=0
  local peak_rss_mib=0
  local p95_rss_mib=0
  local peak_pma_rss_mib=0
  local peak_pma_size_mib=0
  local peak_nockstack_rss_mib=0
  local peak_anon_mib=0
  local peak_file_mib=0
  local minor_faults_delta=0
  local major_faults_delta=0
  local perf_samples=0
  local perf_unique_stacks=0
  local tracy_size_bytes=0
  local tracy_frames=0
  local tracy_time_span_s=0
  local tracy_zones=0

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
  elif [[ -f "$log" ]]; then
    blocks_poked=$(read_bench_field "$log" 'Blocks poked' | awk '{print $1}')
    failed_pokes=$(read_bench_field "$log" 'Failed pokes' | awk '{print $1}')
    init_time_s=$(read_bench_field "$log" 'Init time' | sed -E 's/s$//')
    total_poke_time_s=$(read_bench_field "$log" 'Total poke time' | sed -E 's/s$//')
    avg_per_block_ms=$(read_bench_field "$log" 'Avg per block' | sed -E 's/ms$//')
    throughput_blocks_s=$(read_bench_field "$log" 'Throughput' | awk '{print $1}')
    checkpoints=$(read_bench_field "$log" 'Checkpoints' | awk '{print $1}')
  fi

  if [[ -f "$perf_summary" ]]; then
    perf_samples=$(jq -r '.sample_count // 0' "$perf_summary" 2>/dev/null || echo 0)
    perf_unique_stacks=$(jq -r '.unique_stacks // 0' "$perf_summary" 2>/dev/null || echo 0)
  fi

  if [[ -f "$tracy_trace" ]]; then
    tracy_size_bytes=$(stat -c '%s' "$tracy_trace" 2>/dev/null || echo 0)
  fi
  if [[ -f "$tracy_capture_log" ]]; then
    tracy_frames=$(awk -F': ' '$1 == "Frames" {print $2}' "$tracy_capture_log" | tail -n1 | tr -d ',' | tr -d '\r')
    tracy_time_span_s=$(awk -F': ' '$1 == "Time span" {print $2}' "$tracy_capture_log" | tail -n1 | awk '{print $1}' | tr -d '\r')
    tracy_zones=$(awk -F': ' '$1 == "Zones" {print $2}' "$tracy_capture_log" | tail -n1 | tr -d ',' | tr -d '\r')
    tracy_frames="${tracy_frames:-0}"
    tracy_time_span_s="${tracy_time_span_s:-0}"
    tracy_zones="${tracy_zones:-0}"
  fi

  local wall_clock=""
  local max_rss_kb=""
  if [[ -f "$time_log" ]]; then
    wall_clock=$(read_time_field "$time_log" 'Elapsed (wall clock) time')
    max_rss_kb=$(read_time_field "$time_log" 'Maximum resident set size')
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$pass" "$env" "$(branch_label "$branch")" "$fixture_id" \
    "$blocks_poked" "$failed_pokes" "$init_time_s" "$total_poke_time_s" "$avg_per_block_ms" "$throughput_blocks_s" \
    "$checkpoints" "$samples" "$gc_events" "$fault_bursts" "$peak_rss_mib" "$p95_rss_mib" \
    "$peak_pma_rss_mib" "$peak_pma_size_mib" "$peak_nockstack_rss_mib" "$peak_anon_mib" "$peak_file_mib" \
    "$minor_faults_delta" "$major_faults_delta" "$wall_clock" "$max_rss_kb" "$exit_code" \
    "$profile" "$log" "$time_log" \
    "$perf_samples" "$perf_unique_stacks" "$perf_summary" "$perf_svg" "$perf_report" "$perf_folded" "$perf_data" \
    "$tracy_trace" "$tracy_capture_log" "$tracy_size_bytes" "$tracy_frames" "$tracy_time_span_s" "$tracy_zones" \
    >>"$SUMMARY_TSV"
}

run_counter=0
total_runs=$(( ${#PASSES[@]} * ${#ENVS[@]} * ${#BRANCH_IDS[@]} * ${#FIXTURE_IDS[@]} ))

for fixture_id in "${FIXTURE_IDS[@]}"; do
  fixture_path="$FIX_DIR/$(fixture_file "$fixture_id")"
  [[ -f "$fixture_path" ]] || { echo "Missing fixture file: $fixture_path" >&2; exit 1; }
done

echo "Run root: $RUN_ROOT"
echo "Summary TSV: $SUMMARY_TSV"

if [[ "$SOL_GUARD_POST_RUN" == "true" ]]; then
  if [[ -z "$SOL_GUARD_CONTRACT" ]]; then
    echo "SOL guard post-run requested but SOL_GUARD_CONTRACT is empty; skipping." >&2
  else
    echo "Running optional SOL guard checks..."
    guard_fail=0
    for env in "${ENVS[@]}"; do
      for branch in "${BRANCH_IDS[@]}"; do
        for fixture_id in "${FIXTURE_IDS[@]}"; do
          cmd=(
            "$SCRIPT_DIR/sol_guard_ci.sh"
            --candidate-summary "$SUMMARY_TSV"
            --contract "$SOL_GUARD_CONTRACT"
            --env "$env"
            --branch "$branch"
            --fixture "$fixture_id"
            --output-json "$RUN_ROOT/guard-${env}-${branch}-${fixture_id}.json"
            --output-md "$RUN_ROOT/guard-${env}-${branch}-${fixture_id}.md"
          )
          if [[ -n "$SOL_GUARD_BASELINE_SUMMARY" ]]; then
            cmd+=(--baseline-summary "$SOL_GUARD_BASELINE_SUMMARY")
          fi
          if [[ "$SOL_GUARD_STRICT" == "true" ]]; then
            cmd+=(--strict)
          fi

          set +e
          "${cmd[@]}"
          rc=$?
          set -e
          if [[ $rc -ne 0 ]]; then
            echo "SOL guard failed for env=$env branch=$branch fixture=$fixture_id (exit=$rc)" >&2
            guard_fail=1
          fi
        done
      done
    done
    if [[ $guard_fail -ne 0 ]]; then
      echo "One or more SOL guard checks failed" >&2
      exit 2
    fi
  fi
fi
echo "Total runs: $total_runs"
echo "Profile memory: $PROFILE_MEMORY"
echo "Perf native: $PERF_NATIVE | Perf docker: $PERF_DOCKER (event=$PERF_EVENT, freq=$PERF_FREQ, callgraph=$PERF_CALL_GRAPH)"
echo "Tracy native: $TRACY_CAPTURE_NATIVE | Tracy docker: $TRACY_CAPTURE_DOCKER (master=$TRACY_CAPTURE_MASTER_BIN, legacy=$TRACY_CAPTURE_LEGACY_BIN, timeout=${TRACY_CAPTURE_TIMEOUT_SEC}s)"
echo "Nock tracing: $NOCK_TRACING (TRACY_ONLY_NOCKCODE=1)"
echo "Trace arg: ${NOCK_TRACE_MODE:-off}"
echo "Docker host network: $DOCKER_USE_HOST_NETWORK"
echo "Passes: $PASSES_CSV"
echo "Envs: $ENVS_CSV"
printf 'Fixtures:'
for fixture_id in "${FIXTURE_IDS[@]}"; do
  printf ' %s=%s' "$fixture_id" "$(fixture_file "$fixture_id")"
done
printf '\n'

for pass in "${PASSES[@]}"; do
  for env in "${ENVS[@]}"; do
    for branch in "${BRANCH_IDS[@]}"; do
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

        collect_and_append "$pass" "$env" "$branch" "$fixture_id" "$run_dir" "$exit_code"
        echo "  exit_status=$exit_code"
      done
    done
  done
done

echo "Summary TSV: $SUMMARY_TSV"
