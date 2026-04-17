# AGENTS.md

## Default Build Mode

- Use release builds and release binaries unless the task explicitly says otherwise.
- Prefer `/shared/nockchain/target/release/nockchain-bench` over `cargo run` for long-running SOL extraction, fixture, and harness commands.

## Docker Harness Notes

- For trusted Docker-backed SOL runs, use `nockchain-bench sol bench`, not `sol quick-bench`.
- Trusted `sol bench` requires at least `--measured-runs 3`.
- The `--output` directory must exist and be empty before starting a trusted run.
- Docker execution requires all of:
  - `--image-tag`
  - `--memory-limit`
  - `--work-dir-mode`
- A known-good local image from this workspace is `nockchain-bench:local`.

## Docker Environment

- This machine uses Docker Desktop with context `desktop-linux`.
- The Docker Desktop socket for this user is `unix:///home/drbeefsupreme/.docker/desktop/docker.sock`.
- The `docker` CLI may work while in-process Docker clients fail if they cannot access that socket.
- If a harness command reports that Docker is unavailable, run it outside the sandbox or set `DOCKER_HOST=unix:///home/drbeefsupreme/.docker/desktop/docker.sock` for the process that launches the harness.

## PMA Verification Branch

- For cold-peek PMA-side verification against the exact PMA line, use the local branch `pma-post-throughput-elas-sr-fsync-hrtb-closure-exact` in worktree `/shared/nockchain/.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure`.
- This branch exists because upstream `nockchain/bitemyapp/bump-pma-post-throughput-elas-sr-fsync-hrtb-closure` did not yet contain `PmaConfig::for_nc_bench_shim(...)`, while `nockchain-bench --features pma-runtime-compat` depends on that helper.
- The saved branch adds the shim plus a focused PMA-side test, and it is the branch that passed:
  - `cargo test -p nockapp --release --manifest-path /shared/nockchain/.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure/Cargo.toml for_nc_bench_shim`
  - `cargo check -p nockchain-bench --release --features pma-runtime-compat --manifest-path /shared/nockchain/.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure/Cargo.toml`
- Prefer this saved branch for future PMA transplant/checkpoint verification unless the upstream exact branch gains the shim and is re-verified.

## Verified Docker Bench Example

```bash
/shared/nockchain/target/release/nockchain-bench sol bench \
  --fixture /shared/nockchain/fixtures/first-100-derived-checkpoint-no-mempool.soltest \
  --output /shared/nockchain/tmp/docker-bench-smoke \
  --image-tag nockchain-bench:local \
  --memory-limit 8g \
  --work-dir-mode docker-tmpfs \
  --warmup-runs 0 \
  --measured-runs 3 \
  --cooldown-secs 0
```

## Known Local Quirks

- `gnort` may log a background Datadog metrics client panic (`Operation not permitted`) during local runs. In this workspace, that did not prevent successful extraction, fixture build, native quick-bench, or Docker trusted bench completion.
- Full-range `sol extract` progress can appear quiet after the first chunk when invoked with a huge `--end-height`; the extractor writes the `.solarch` only at the end.
