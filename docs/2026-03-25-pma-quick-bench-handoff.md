# PMA Quick-Bench Handoff

This note explains how to run the bench-side PMA compatibility work against a
PMA checkout, including the required PMA helper, the bench transplant step, and
the verified `quick-bench` commands.

## 1. PMA-side prerequisite

The PMA branch must contain a 15-line helper function found in the following location:

- branch: `jon/pma-branch-PmaConfig-nc-bench-shim`
- helper: `PmaConfig::for_nc_bench_shim(...)`
- file: `crates/nockapp/src/kernel/form.rs`

If your PMA branch does not already contain that helper, bring it over first.
The bench-side compatibility code depends on that constructor.

## 2. Bench branch with the PMA compatibility changes

The bench-side compatibility work lives on.

- branch: `nockchain-bench-pma-compat`

## 3. How to transplant `nockchain-bench`

Use:

- script: `scripts/bench_sync/pma_bench_sync.py`

Found in `nockchain-bench-pma-compat`

You need to transplant the `nockchain-bench` crate onto the PMA branch you would
like to benchmark. To do so:

```bash
cd /path/to/nockchain-bench-checkout
git branch --show-current
# expected: nockchain-bench-pma-compat

cd /path/to/pma-checkout
git branch --show-current
# expected: a branch that already contains PmaConfig::for_nc_bench_shim(...)

cd /path/to/nockchain-bench-checkout
uv run --project scripts/bench_sync \
  scripts/bench_sync/pma_bench_sync.py \
  --target-dir /path/to/pma-checkout \
  --force \
  --allow-dirty-source
```

What the script does:

- replaces `crates/nockchain-bench` in the target checkout
- patches the target workspace manifest if needed
- builds `cargo build -p nockchain-bench --release --features pma-runtime-compat`
- writes a `.pma-bench-sync-stamp`
- prints a placeholder `quick-bench` command

## 5. Quick-bench command after transplant

Verified smoke command:

```bash
/path/to/pma-checkout/target/release/nockchain-bench sol quick-bench \
  --fixture /path/to/fixtures/first-100-v2-derived-checkpoint-no-mempool.soltest \
  --blocks 10 \
  --checkpoint-every-blocks 0
```

Verified PMA memory-sampling command:

```bash
/path/to/pma-checkout/target/release/nockchain-bench sol quick-bench \
  --fixture /path/to/fixtures/first-100-v2-derived-checkpoint-no-mempool.soltest \
  --blocks 10 \
  --checkpoint-every-blocks 0 \
  --profile-memory \
  --profile-interval-ms 500 \
  --profile-output /tmp/pma-quick-bench-memory.json
```

## 6. Quick-bench settings summary

- `--fixture <path>`: required `.soltest` fixture
- `--blocks N`: replay the first `N` accepted blocks from the fixture archive
- `--blocks 0`: replay all accepted blocks in the fixture archive window
- `--checkpoint-every-blocks 0`: required for PMA right now (i.e. no checkpointing allowed)
- `--profile-memory`: enable process RSS/page-fault timeline sampling
- `--profile-interval-ms <ms>`: memory sampling interval in milliseconds
- `--profile-output <path>`: write benchmark + memory profile JSON
- `--enable-checkpointing true|false`: available, defaults to `true`
- `--cpu-profiler samply`: optional extra CPU-profile replay pass

## 7. Current PMA limitations

These are still intentionally unsupported under `pma-runtime-compat`:

- `--checkpoint-every-blocks > 0`
- replay with `prefer_existing_checkpoint = true`
- `boot::setup()`-based PMA boot
- PMA data-dir / event-log / snapshot boot-source behavior

## 8. Recommended verification sequence

From the PMA checkout:

```bash
cargo test -p nockapp --release
cargo test -p nockchain-bench --release --features pma-runtime-compat runtime_compat::
cargo test -p nockchain-bench --release --features pma-runtime-compat \
  test_pma_checkpoint_cadence_guard_rejects_nonzero_cadence
cargo test -p nockchain-bench --release --features pma-runtime-compat \
  test_pma_checkpoint_cadence_guard_allows_zero_cadence
cargo test -p nockchain-bench --release --features pma-runtime-compat \
  test_pma_init_nockapp_rejects_prefer_existing_checkpoint
cargo build -p nockchain-bench --release --features pma-runtime-compat
```

Then run the smoke `quick-bench` command with `--checkpoint-every-blocks 0`.

## 9. Extra context

- `sol quick-bench` is for fast inner-loop investigation, not trusted benchmark
  publication.
- The helper tests in `runtime_compat.rs` are the durable evidence that
  `replay-pma/0.pma` and `replay-pma/1.pma` are recreated fresh for each run.

## 10. Reference

For general `nockchain-bench` usage, including how to build fixtures from
checkpoints, see:

- `crates/nockchain-bench/README.md`
