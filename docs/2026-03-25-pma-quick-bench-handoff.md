# PMA Bench Handoff

This note explains how to validate bench-side PMA replay work against a PMA
checkout after transplant. It covers the required PMA helper contract, the
bench-sync flow, the current `sol quick-orchestrate` cold-peek surface, and the
release verification commands that belong in the transplanted PMA checkout.

## 1. PMA prerequisite

The bench-side PMA compatibility path still depends on the PMA helper
`PmaConfig::for_nc_bench_shim(...)` in:

- file: `crates/nockapp/src/kernel/form.rs`

Default PMA line today:

- branch: `bitemyapp/bump-pma-post-throughput-elas-sr-fsync-hrtb-closure`

Compatibility rule:

- any PMA branch that already carries `PmaConfig::for_nc_bench_shim(...)` is a
  valid transplant target
- the named branch above is the default recommendation, not an exclusivity rule

Workspace-local verification note:

- in this workspace, use
  `/shared/nockchain/.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure`
  on branch `pma-post-throughput-elas-sr-fsync-hrtb-closure-exact` for cold-peek
  PMA-side verification unless the upstream line has gained the helper and been
  re-verified

If your PMA target does not already contain `PmaConfig::for_nc_bench_shim(...)`,
bring that helper over first. The bench-side `pma-runtime-compat` code calls
into it directly.

## 2. Transplant `nockchain-bench` into the PMA checkout

From the bench checkout that contains the `nockchain-bench` changes you want to
validate, run:

```bash
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

Do not change `scripts/bench_sync/pma_bench_sync.py` for cold-peek handoff
work. `pma-runtime-compat` remains the only required feature.

## 3. Where feature-gated verification runs

Anything that uses `--features pma-runtime-compat` belongs in the transplanted
PMA checkout, not in the standalone bench checkout. That includes:

- PMA helper verification
- `sol quick-bench` PMA replay validation
- `sol quick-orchestrate` cold-step validation
- all `cargo test` and `cargo build` commands that enable
  `pma-runtime-compat`

The standalone bench checkout can still author and test non-feature-gated code,
but it is not the authoritative acceptance environment for PMA replay.

## 4. `sol quick-orchestrate` cold-peek surface

After transplant, `sol quick-orchestrate` supports four plan step types:

- `poke_archive_block`
- `peek_height`
- `force_cold`
- `peek_height_cold`

Cold-step gating:

- cold steps require `--features pma-runtime-compat`
- Linux performs verified cold eviction
- non-Linux still compiles under `pma-runtime-compat`, but cold execution
  degrades instead of claiming verified cold residency

Cold-step JSON fields:

- `label`: optional step label
- `tolerance_pages`: optional cold-residency tolerance, defaults to `0`
- `max_attempts`: optional retry budget, defaults to `3`

`peek_height_cold` is sugar for the common "force cold, then immediately peek"
case and emits one fused result. `force_cold` remains the primitive when you
want explicit composition.

Adjacency rule:

- only the immediately adjacent peek after `force_cold` is verifiably cold
- a `peek_height` label starting with `cold-` must be adjacent to a qualifying
  `force_cold` step or plan validation rejects it
- ambiguous interleavings warn before boot rather than silently claiming cold
  semantics

Example plan:

```json
{
  "checkpoint": "/path/to/0.chkjam",
  "kernel": "/path/to/dumb.jam",
  "steps": [
    {
      "type": "peek_height",
      "height": 100,
      "label": "warm-100"
    },
    {
      "type": "peek_height_cold",
      "height": 100,
      "label": "cold-100"
    },
    {
      "type": "force_cold",
      "label": "prep-101",
      "tolerance_pages": 0,
      "max_attempts": 3
    },
    {
      "type": "peek_height",
      "height": 101,
      "label": "cold-101"
    }
  ]
}
```

Example command:

```bash
/path/to/pma-checkout/target/release/nockchain-bench sol quick-orchestrate \
  --plan /path/to/plan.json \
  --cold-mode strict \
  --profile-output /tmp/pma-quick-orchestrate.json
```

`--cold-mode` behavior:

- `strict` is the default and aborts the run if a cold step cannot verify cold
  residency within the retry budget
- `soft` continues and records `cold_verified=false` plus the cold-step
  metadata so residue cases can still be inspected

Under `pma-runtime-compat`, `--fsync on|off` still controls PMA-backed runs.
Cold-prep honors that setting: `fsync=on` pre-syncs before cold eviction and
`fsync=off` leaves that writeback cost in the cold-prep path.

## 5. `sol quick-bench` still works after transplant

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

## 6. Release verification after transplant

Run these from the transplanted PMA checkout:

```bash
cargo test -p nockapp --release for_nc_bench_shim
cargo test -p nockchain-bench --release --features pma-runtime-compat cold_init_
cargo test -p nockchain-bench --release --features pma-runtime-compat quick_orchestrate_
cargo test -p nockchain-bench --release --features pma-runtime-compat force_cold_
cargo build -p nockchain-bench --release --features pma-runtime-compat
```

If you are using the saved local PMA verification worktree in this repository,
the equivalent build is:

```bash
cargo build -p nockchain-bench --release --features pma-runtime-compat \
  --manifest-path /shared/nockchain/.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure/Cargo.toml
```

Checkpoint-backed ignored cold smokes remain intentionally manual and
environment-dependent. Only run them on demand from the transplanted PMA
checkout when you have the checkpoint fixture and cgroup setup available.

## 7. Current PMA limitations

These are still intentionally unsupported under `pma-runtime-compat`:

- `--checkpoint-every-blocks > 0`
- replay with `prefer_existing_checkpoint = true`
- `boot::setup()`-based PMA boot
- PMA data-dir / event-log / snapshot boot-source behavior

## 8. Reference

For general `nockchain-bench` usage, including fixture creation and trusted
`sol bench` guidance, see:

- `crates/nockchain-bench/README.md`
