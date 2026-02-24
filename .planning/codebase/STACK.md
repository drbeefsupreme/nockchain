# Technology Stack

**Analysis Date:** 2026-02-24

## Languages

**Primary:**
- Rust (Edition 2021, nightly toolchain) - core runtime, network, CLI, gRPC, bridge, and tooling across workspace crates in `Cargo.toml`, `crates/nockchain/Cargo.toml`, `crates/bridge/Cargo.toml`, `crates/nockapp/Cargo.toml`, and `crates/nockup/Cargo.toml`.

**Secondary:**
- Hoon (compiled to `.jam` assets) - kernel/app logic under `hoon/` and built via `Makefile` targets such as `assets/dumb.jam`, `assets/miner.jam`, `assets/wal.jam`, `assets/peek.jam`, `assets/bridge.jam`.
- Solidity (`solc 0.8.30`) - bridge contracts in `crates/bridge/contracts/MessageInbox.sol` and `crates/bridge/contracts/Nock.sol` configured by `crates/bridge/contracts/foundry.toml`.
- Nix - reproducible dev shell and toolchain bootstrap in `flake.nix`.
- Shell scripting - operational scripts under `scripts/` and contract automation in `crates/bridge/contracts/Makefile`.

## Runtime

**Environment:**
- Rust nightly toolchain pinned to `nightly-2025-11-26` with `miri` component in `rust-toolchain.toml`.

**Package Manager:**
- Cargo (workspace-based) defined in `Cargo.toml` and crate manifests under `crates/**/Cargo.toml`.
- Lockfile: present (`Cargo.lock`).

## Frameworks

**Core:**
- Tokio (`tokio`) - async runtime and concurrency in `Cargo.toml` and used across `crates/nockchain/src/main.rs`, `crates/bridge/src/main.rs`, `crates/nockapp/src/drivers/http/http.rs`.
- libp2p (git-pinned) - P2P networking in `crates/nockchain/Cargo.toml` and `crates/nockchain-libp2p-io/Cargo.toml`.
- Axum + axum-server - HTTP server/runtime in `crates/nockapp/Cargo.toml` and `crates/nockapp/src/drivers/http/http.rs`.
- Tonic/prost - gRPC services and clients in `crates/nockapp-grpc/Cargo.toml`, `crates/nockapp-grpc-proto/Cargo.toml`, `crates/nockchain-wallet/src/connection.rs`.

**Testing:**
- Cargo test harness - unit/integration testing via `Makefile` (`cargo test --release`) and crate-level tests in `crates/**/tests`.
- Proptest/Quickcheck - property testing dependencies in `Cargo.toml` and crates like `crates/noun-serde/Cargo.toml` and `crates/nockchain-libp2p-io/Cargo.toml`.
- Foundry (forge) - Solidity tests in `crates/bridge/contracts/test/*.t.sol` with config in `crates/bridge/contracts/foundry.toml`.

**Build/Dev:**
- Cargo + Make - primary build/install workflow in `Makefile`.
- Hoon compiler (`hoonc`) - builds kernel artifacts via `Makefile` and `crates/hoonc`.
- Nix flake shell - tool bootstrap and cross-platform build env in `flake.nix`.
- Bazelisk support in Nix shell (`flake.nix`) for optional build workflows.

## Key Dependencies

**Critical:**
- `libp2p` (git revision) - decentralized peer networking for node communications in `Cargo.toml` and `crates/nockchain-libp2p-io/Cargo.toml`.
- `tonic`/`tonic-prost` - RPC boundary between node, wallet, explorer, and bridge ingress in `crates/nockapp-grpc/Cargo.toml`, `crates/nockchain-wallet/Cargo.toml`, `crates/bridge/Cargo.toml`.
- `alloy` and `op-alloy` - EVM/Base chain connectivity and signing in `crates/bridge/Cargo.toml` and `crates/bridge/src/ethereum.rs`.
- `nockvm`/`zkvm-jetpack` - execution/prover substrate used by node and API in `crates/nockchain/Cargo.toml`, `crates/nockchain-api/Cargo.toml`.

**Infrastructure:**
- `diesel` + `deadpool-diesel` + `libsqlite3-sys` - bridge deposit log persistence in SQLite in `crates/bridge/Cargo.toml` and `crates/bridge/src/deposit_log.rs`.
- `opentelemetry`, `opentelemetry-otlp`, `tracing-*` - telemetry and tracing in `crates/nockapp/Cargo.toml` and `crates/nockapp/src/observability.rs`.
- `instant-acme`, `rustls`, `rcgen` - automatic HTTPS certificate management in `crates/nockapp/Cargo.toml` and `crates/nockapp/src/drivers/http/http.rs`.
- `reqwest` - HTTP client flows for `nockup` downloads/registry lookups in `crates/nockup/Cargo.toml` and `crates/nockup/src/commands/common.rs`.

## Configuration

**Environment:**
- Runtime env is loaded from `.env` for Make workflows (`Makefile`) with defaults in `.env_example`.
- Bridge runtime config is TOML-based (`crates/bridge/bridge-conf.example.toml`) and loaded at runtime from `bridge-conf.toml` in `crates/bridge/src/config.rs`.
- Contract deployment uses env-driven Foundry/Tenderly config from `crates/bridge/contracts/.env.template` and `crates/bridge/contracts/DEPLOYMENT.md`.
- Nockapp HTTP/observability knobs are env-driven (`crates/nockapp/src/drivers/http/http.rs`, `crates/nockapp/src/observability.rs`).

**Build:**
- Workspace/build profiles and dependency versions in `Cargo.toml`.
- Rust formatting settings in `rustfmt.toml`.
- Nightly pin in `rust-toolchain.toml`.
- Nix shell toolchain in `flake.nix`.
- Solidity build settings in `crates/bridge/contracts/foundry.toml`.

## Platform Requirements

**Development:**
- Rust nightly + Cargo from `rust-toolchain.toml`/`Cargo.toml`.
- Protobuf compiler required by gRPC build steps (documented in `README.md` and CI in `.github/workflows/release.yml`, `.gitlab-ci.yml`).
- Clang/LLVM toolchain for native dependencies (documented in `README.md` and encoded in `flake.nix`).
- Optional Nix development shell for reproducible setup (`flake.nix`).

**Production:**
- Binary-distributed deployments via GitHub Releases workflow in `.github/workflows/release.yml`.
- Node workloads run as native binaries (`nockchain`, `nockchain-wallet`, `nockchain-api`, `bridge`) with host-managed state directories (for example `.data.*` paths and `~/.nockapp` usage in `crates/nockapp/src/lib.rs`).

---

*Stack analysis: 2026-02-24*
