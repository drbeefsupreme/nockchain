# Technology Stack

**Analysis Date:** 2026-03-03

## Languages

**Primary:**
- Rust (edition 2021) - main application and benchmark crates across `crates/*`

**Secondary:**
- Bash - benchmark orchestration and matrix runs in `scripts/`
- Python - report/post-processing helpers in `scripts/`
- Hoon tooling - kernel asset build workflow referenced by repository build tooling

## Runtime

**Environment:**
- Native Linux runtime for benchmark/profiling paths (`/proc`, `perf`-based paths)
- Docker runtime for containerized benchmark scenarios

**Package Manager:**
- Cargo workspace (`Cargo.toml` at repo root)
- Lockfile: `Cargo.lock` present

## Frameworks

**Core:**
- `nockchain`, `nockapp`, `nockvm` ecosystem crates
- `tokio` async runtime and `clap` CLI for `nockchain-bench`

**Testing:**
- `cargo test` for unit/integration tests
- selective `#[ignore]` long-running tests (Docker/full checkpoint paths)

**Build/Dev:**
- pinned Rust toolchain in `rust-toolchain.toml`
- workspace formatting via `rustfmt.toml`

## Key Dependencies

**Critical:**
- `bollard` - Docker API integration for benchmark runners
- `arrow` / `parquet` - structured benchmark artifact output
- `serde` / `serde_json` / `toml` / `bincode` - configuration and artifact serialization
- `tracing` / `tracing-tracy` - tracing and profiler integration
- workspace crates (`nockvm`, `nockapp`, `nockchain-types`, `nockchain-math`) - runtime domain integration

## Configuration

**Environment:**
- CLI flags and config files (`sol-baseline.toml`, benchmark matrix scripts)
- benchmark artifacts and fixture paths passed at runtime

**Build:**
- workspace manifests in root `Cargo.toml` and crate-level manifests
- rust formatting/linting conventions from `rustfmt.toml` and crate attributes

## Platform Requirements

**Development:**
- Linux/macOS development supported, Linux strongly preferred for profiling fidelity
- Docker required for containerized scenario runs

**Production/Benchmark Execution:**
- benchmark pipelines rely on Docker + host filesystem mounts
- optional `perf` tooling required for native profiling workflows

---
*Stack analysis: 2026-03-03*
*Update after major dependency changes*
