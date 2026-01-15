#!/bin/bash
# ram_comparison_experiment.sh
# Compare RAM usage between checkpoint mode and PMA-persist mode
#
# Usage: bash scripts/ram_comparison_experiment.sh
#
# Prerequisites:
#   1. Docker image built: make docker-nockchain-build
#   2. Docker network exists (created automatically)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/experiment-data}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nockchain-local}"
DOCKER_NETWORK="${DOCKER_NETWORK:-nockchain-metrics}"
DOCKER_MEM="${DOCKER_MEM:-16g}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-120}"  # seconds
MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"          # seconds
PHASE2_DURATION="${PHASE2_DURATION:-300}"          # seconds to run phase 2

# Data subdirectories (nockchain creates these)
CHECKPOINTS_DIR="${EXPERIMENT_DIR}/checkpoints"
PMA_DIR="${EXPERIMENT_DIR}/pma"

# Output files
CHECKPOINT_RAM_LOG="${EXPERIMENT_DIR}/checkpoint-ram.log"
PMA_PERSIST_RAM_LOG="${EXPERIMENT_DIR}/pma-persist-ram.log"

echo "=============================================="
echo "Nockchain RAM Comparison Experiment"
echo "=============================================="
echo "Docker image:        $DOCKER_IMAGE"
echo "Memory limit:        $DOCKER_MEM"
echo "Checkpoint interval: ${CHECKPOINT_INTERVAL}s"
echo "Phase 2 duration:    ${PHASE2_DURATION}s"
echo "Experiment dir:      $EXPERIMENT_DIR"
echo "Checkpoints dir:     $CHECKPOINTS_DIR"
echo "PMA dir:             $PMA_DIR"
echo ""
echo "NOTE: Mining is enabled (--mine) so events occur"
echo "      and checkpoints actually get triggered."
echo "=============================================="
echo ""

# Ensure docker network exists
docker network inspect "$DOCKER_NETWORK" >/dev/null 2>&1 || \
    docker network create "$DOCKER_NETWORK"

# Clean up any existing containers (don't fail if they don't exist)
docker rm -f nockchain-checkpoint nockchain-pma 2>/dev/null || true

# === PHASE 1: Run with checkpointing until first checkpoint ===
echo "=== PHASE 1: Checkpoint mode ==="
echo "Running nockchain (fakenet) with checkpointing enabled..."
echo "Will wait for first checkpoint file to appear."
echo ""

# Create fresh data directory
rm -rf "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR"

# Clear log file
> "$CHECKPOINT_RAM_LOG"

# Generate a mining address if not provided
if [ -z "$MINING_PKH" ]; then
    # Use a dummy address for fakenet mining
    MINING_PKH="9yPePjfWAdUnzaQKyxcRXKRa5PpUzKKEwtpECBZsUYt9Jd7egSDEWoV"
    echo "Using default MINING_PKH: $MINING_PKH"
fi

# Start container in background (no --rm so we can see logs if it fails)
# NOTE: We enable mining (--mine) so that events happen and checkpoints trigger
docker run -d --name nockchain-checkpoint \
    --network "$DOCKER_NETWORK" \
    --memory "$DOCKER_MEM" \
    -e TRACY_NO_INVARIANT_CHECK=1 \
    -e RUST_BACKTRACE=1 \
    -e RUST_LOG=info,nockchain=info \
    -e MINIMAL_LOG_FORMAT=true \
    -v "${EXPERIMENT_DIR}:/data/.data.nockchain" \
    "$DOCKER_IMAGE" \
    --fakenet \
    --new \
    --mine \
    --mining-pkh "$MINING_PKH" \
    --num-threads 1 \
    --save-interval "$CHECKPOINT_INTERVAL" \
    --data-dir /data/.data.nockchain \
    --bind /ip4/0.0.0.0/udp/30000/quic-v1

echo "Container started. Monitoring RAM usage..."
echo "Waiting for checkpoint file in ${CHECKPOINTS_DIR}..."
echo ""

# Monitor RAM until checkpoint appears
CHECKPOINT_FOUND=0
SECONDS_WAITED=0
MAX_WAIT=$((CHECKPOINT_INTERVAL * 3))  # Wait up to 3x the checkpoint interval

while [ $CHECKPOINT_FOUND -eq 0 ] && [ $SECONDS_WAITED -lt $MAX_WAIT ]; do
    if ! docker ps | grep -q nockchain-checkpoint; then
        echo ""
        echo "ERROR: Container stopped unexpectedly!"
        echo "=== Container logs ==="
        docker logs nockchain-checkpoint 2>&1 | tail -50
        echo "======================"
        docker rm -f nockchain-checkpoint 2>/dev/null || true
        exit 1
    fi

    # Log RAM usage with timestamp
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    RAM_USAGE=$(docker stats nockchain-checkpoint --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "N/A")
    echo "$TIMESTAMP $RAM_USAGE" >> "$CHECKPOINT_RAM_LOG"
    echo "  [$TIMESTAMP] RAM: $RAM_USAGE"

    # Check for checkpoint files in the correct subdirectory
    if [ -f "${CHECKPOINTS_DIR}/0.chkjam" ] || [ -f "${CHECKPOINTS_DIR}/1.chkjam" ]; then
        echo ""
        echo "Checkpoint file detected!"
        CHECKPOINT_FOUND=1

        # Continue monitoring for a bit after checkpoint to capture post-checkpoint RAM
        echo "Monitoring for 30 more seconds after checkpoint..."
        for i in {1..6}; do
            sleep 5
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            RAM_USAGE=$(docker stats nockchain-checkpoint --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "N/A")
            echo "$TIMESTAMP $RAM_USAGE" >> "$CHECKPOINT_RAM_LOG"
            echo "  [$TIMESTAMP] RAM: $RAM_USAGE"
        done
    else
        sleep "$MONITOR_INTERVAL"
        SECONDS_WAITED=$((SECONDS_WAITED + MONITOR_INTERVAL))
    fi
done

if [ $CHECKPOINT_FOUND -eq 0 ]; then
    echo ""
    echo "WARNING: No checkpoint file found after ${MAX_WAIT}s"
    echo "=== Container logs (last 30 lines) ==="
    docker logs nockchain-checkpoint 2>&1 | tail -30
    echo "======================================="
    echo "Continuing anyway..."
fi

# Stop container
echo ""
echo "Stopping checkpoint mode container..."
docker stop nockchain-checkpoint 2>/dev/null || true
docker rm -f nockchain-checkpoint 2>/dev/null || true

# Show phase 1 results
echo ""
echo "Phase 1 complete. RAM log saved to: $CHECKPOINT_RAM_LOG"
echo ""
echo "Files in experiment dir:"
ls -laR "$EXPERIMENT_DIR" 2>/dev/null || echo "(empty)"
echo ""

# Brief pause between phases
sleep 5

# === PHASE 2: Run with PMA persistence (no checkpointing) ===
echo "=== PHASE 2: PMA persist mode ==="
echo "Running nockchain with PMA persistence (no checkpoints)..."
echo "Will run for ${PHASE2_DURATION} seconds."
echo ""

# Verify pma.meta exists in the correct location
if [ ! -f "${PMA_DIR}/pma.meta" ]; then
    echo "WARNING: ${PMA_DIR}/pma.meta not found."
    echo "The checkpoint mode should have created this file."
    echo "PMA persist mode may not work correctly."
    echo ""
    echo "Contents of PMA dir:"
    ls -la "${PMA_DIR}" 2>/dev/null || echo "(directory does not exist)"
    echo ""
fi

# Clear log file
> "$PMA_PERSIST_RAM_LOG"

# Start container with PMA persist mode (no --new flag, reuse existing data)
# Also enable mining so activity continues (fair comparison)
docker run -d --name nockchain-pma \
    --network "$DOCKER_NETWORK" \
    --memory "$DOCKER_MEM" \
    -e TRACY_NO_INVARIANT_CHECK=1 \
    -e RUST_BACKTRACE=1 \
    -e RUST_LOG=info,nockchain=info \
    -e MINIMAL_LOG_FORMAT=true \
    -e NOCK_PMA_PERSIST=1 \
    -v "${EXPERIMENT_DIR}:/data/.data.nockchain" \
    "$DOCKER_IMAGE" \
    --fakenet \
    --pma-persist \
    --mine \
    --mining-pkh "$MINING_PKH" \
    --num-threads 1 \
    --save-interval 0 \
    --data-dir /data/.data.nockchain \
    --bind /ip4/0.0.0.0/udp/30001/quic-v1

echo "Container started. Monitoring RAM usage for ${PHASE2_DURATION}s..."
echo ""

# Monitor RAM for the specified duration
SECONDS_ELAPSED=0
while [ $SECONDS_ELAPSED -lt $PHASE2_DURATION ]; do
    if ! docker ps | grep -q nockchain-pma; then
        echo ""
        echo "WARNING: Container stopped unexpectedly at ${SECONDS_ELAPSED}s"
        echo "=== Container logs ==="
        docker logs nockchain-pma 2>&1 | tail -30
        echo "======================"
        break
    fi

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    RAM_USAGE=$(docker stats nockchain-pma --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "N/A")
    echo "$TIMESTAMP $RAM_USAGE" >> "$PMA_PERSIST_RAM_LOG"
    echo "  [$TIMESTAMP] RAM: $RAM_USAGE"

    sleep "$MONITOR_INTERVAL"
    SECONDS_ELAPSED=$((SECONDS_ELAPSED + MONITOR_INTERVAL))
done

# Stop container
echo ""
echo "Stopping PMA persist mode container..."
docker stop nockchain-pma 2>/dev/null || true
docker rm -f nockchain-pma 2>/dev/null || true

echo ""
echo "Phase 2 complete. RAM log saved to: $PMA_PERSIST_RAM_LOG"
echo ""

# === RESULTS ===
echo "=============================================="
echo "RESULTS"
echo "=============================================="
echo ""

echo "Checkpoint mode RAM readings (last 10):"
cat "$CHECKPOINT_RAM_LOG" | tail -10
echo ""

echo "PMA persist mode RAM readings (last 10):"
cat "$PMA_PERSIST_RAM_LOG" | tail -10
echo ""

# Find peak values
echo "----------------------------------------------"
echo "PEAK RAM USAGE:"
echo "----------------------------------------------"

# Parse RAM values - format is like "1.234GiB / 16GiB"
# Extract the used portion and find max
CHECKPOINT_PEAK=$(awk '{print $3}' "$CHECKPOINT_RAM_LOG" 2>/dev/null | grep -oE '[0-9.]+[GMK]iB' | sort -h | tail -1 || echo "N/A")
PMA_PEAK=$(awk '{print $3}' "$PMA_PERSIST_RAM_LOG" 2>/dev/null | grep -oE '[0-9.]+[GMK]iB' | sort -h | tail -1 || echo "N/A")

echo "Checkpoint mode peak: $CHECKPOINT_PEAK"
echo "PMA persist mode peak: $PMA_PEAK"
echo ""
echo "Full logs available at:"
echo "  $CHECKPOINT_RAM_LOG"
echo "  $PMA_PERSIST_RAM_LOG"
echo ""
echo "Data directory contents:"
ls -laR "$EXPERIMENT_DIR" 2>/dev/null | head -30
echo ""
echo "=============================================="
