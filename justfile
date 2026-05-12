# nockchain-bench workflow recipes
# Usage: just --list

# --- Configuration ---

# Path to the PMA worktree / checkout
pma-dir := ".worktrees/pma-bench-run"

# Default fixture for quick-bench runs
default-fixture := "fixtures/first-100-v2-derived-checkpoint-no-mempool.soltest"

# Default block count for quick-bench
default-blocks := "10"

# --- Master (native) workflows ---

# Build nockchain-bench for master (no PMA feature)
bench-build:
    cargo build -p nockchain-bench --release

# Run master bench tests
bench-test:
    cargo test -p nockchain-bench --release

# Quick-bench on master
bench-quick fixture=default-fixture blocks=default-blocks:
    cargo build -p nockchain-bench --release
    ./target/release/nockchain-bench sol quick-bench \
        --fixture {{ fixture }} \
        --blocks {{ blocks }}

# --- PMA workflows ---

# Transplant nockchain-bench into the PMA worktree and build
pma-sync:
    uv run --project scripts/bench_sync \
        scripts/bench_sync/pma_bench_sync.py \
        --target-dir {{ pma-dir }} \
        --force \
        --allow-dirty-source

# Transplant dry-run (preview what would happen)
pma-sync-dry:
    uv run --project scripts/bench_sync \
        scripts/bench_sync/pma_bench_sync.py \
        --target-dir {{ pma-dir }} \
        --force \
        --allow-dirty-source \
        --dry-run

# Build nockchain-bench in the PMA worktree (skip transplant)
pma-build:
    cargo build -p nockchain-bench --release --features pma-runtime-compat \
        --manifest-path {{ pma-dir }}/Cargo.toml

# Run PMA bench tests
pma-test:
    cargo test -p nockchain-bench --release --features pma-runtime-compat \
        --manifest-path {{ pma-dir }}/Cargo.toml

# Run PMA runtime_compat tests only
pma-test-compat:
    cargo test -p nockchain-bench --release --features pma-runtime-compat \
        --manifest-path {{ pma-dir }}/Cargo.toml \
        runtime_compat::

# Quick-bench on PMA (transplant + build + run)
pma-quick fixture=default-fixture blocks=default-blocks:
    just pma-sync
    {{ pma-dir }}/target/release/nockchain-bench sol quick-bench \
        --fixture {{ fixture }} \
        --blocks {{ blocks }} \
        --checkpoint-every-blocks 0

# Quick-bench on PMA with memory profiling
pma-quick-mem fixture=default-fixture blocks=default-blocks interval="500" output="/tmp/pma-quick-bench-memory.json":
    just pma-sync
    {{ pma-dir }}/target/release/nockchain-bench sol quick-bench \
        --fixture {{ fixture }} \
        --blocks {{ blocks }} \
        --checkpoint-every-blocks 0 \
        --profile-memory \
        --profile-interval-ms {{ interval }} \
        --profile-output {{ output }}

# Full PMA verification sequence (from handoff doc)
pma-verify fixture=default-fixture:
    just pma-sync
    cargo test -p nockapp --release \
        --manifest-path {{ pma-dir }}/Cargo.toml
    cargo test -p nockchain-bench --release --features pma-runtime-compat \
        --manifest-path {{ pma-dir }}/Cargo.toml \
        runtime_compat::
    cargo test -p nockchain-bench --release --features pma-runtime-compat \
        --manifest-path {{ pma-dir }}/Cargo.toml \
        test_pma_checkpoint_cadence_guard_rejects_nonzero_cadence
    cargo test -p nockchain-bench --release --features pma-runtime-compat \
        --manifest-path {{ pma-dir }}/Cargo.toml \
        test_pma_checkpoint_cadence_guard_allows_zero_cadence
    cargo test -p nockchain-bench --release --features pma-runtime-compat \
        --manifest-path {{ pma-dir }}/Cargo.toml \
        test_pma_init_nockapp_rejects_prefer_existing_checkpoint
    {{ pma-dir }}/target/release/nockchain-bench sol quick-bench \
        --fixture {{ fixture }} \
        --blocks 10 \
        --checkpoint-every-blocks 0

# --- Both targets ---

# Build and test on both master and PMA
both-test:
    just bench-test
    just pma-sync
    just pma-test

# Quick-bench on both master and PMA (regression comparison)
both-quick fixture=default-fixture blocks=default-blocks:
    just bench-quick {{ fixture }} {{ blocks }}
    just pma-quick {{ fixture }} {{ blocks }}

# --- Utilities ---

# Show PMA worktree branch and sync stamp
pma-status:
    @echo "=== PMA worktree ==="
    @git -C {{ pma-dir }} branch --show-current 2>/dev/null || echo "(not found)"
    @echo ""
    @echo "=== Sync stamp ==="
    @cat {{ pma-dir }}/.pma-bench-sync-stamp 2>/dev/null || echo "(no stamp)"

# List available fixtures
fixtures:
    @ls -1 fixtures/*.soltest 2>/dev/null || echo "No fixtures found in fixtures/"
