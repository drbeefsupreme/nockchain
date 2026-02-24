# Coding Conventions

**Analysis Date:** 2026-02-24

## Naming Patterns

**Files:**
- Use `snake_case` for Rust source files and modules (examples: `crates/nockup/src/git_fetcher.rs`, `crates/nockchain-libp2p-io/src/key_fair_queue.rs`, `crates/nockchain-types/src/tx_engine/v0/note.rs`).

**Functions:**
- Use `snake_case` function names for both public and private APIs (examples: `crates/nockchain/src/lib.rs` has `gen_keypair`, `load_keypair`, `init_with_kernel`; `crates/nockchain-bench/src/speed_of_light/guard/ingest.rs` has `parse_combined_summary_tsv`).

**Variables:**
- Use `snake_case` locals and fields, with descriptive domain names (`initial_peer_multiaddrs`, `base_confirmation_depth`, `checkpoint_count`) as seen in `crates/nockchain/src/lib.rs` and `crates/bridge/src/config.rs`.

**Types:**
- Use `PascalCase` for structs/enums/errors (`DriverInitSignals`, `NockchainAPIConfig`, `BridgeConfigToml`, `IngestError`) as seen in `crates/nockchain/src/lib.rs`, `crates/bridge/src/config.rs`, and `crates/nockchain-bench/src/speed_of_light/guard/ingest.rs`.

## Code Style

**Formatting:**
- Tool used: `rustfmt` via `cargo fmt` (`Makefile`, `.github/workflows/rust-format.yml`).
- Key settings are centralized in `rustfmt.toml`: `max_width = 100`, `imports_granularity = "Module"`, `group_imports = "StdExternalCrate"`, and `reorder_imports = true`.

**Linting:**
- Formatting is enforced in CI with `.github/workflows/rust-format.yml`.
- Clippy is used pragmatically with crate-level `#![allow(...)]` for architectural constraints in `crates/nockchain/src/lib.rs`, `crates/nockchain-wallet/src/main.rs`, and `crates/nockchain-libp2p-io/src/lib.rs`.

## Import Organization

**Order:**
1. Standard library imports first (for example `use std::...` in `crates/nockup/src/main.rs`).
2. External crate imports second (`clap`, `serde`, `tracing`, `tokio`) in files such as `crates/bridge/src/config.rs` and `crates/nockchain/src/lib.rs`.
3. Internal crate imports last (`use crate::...` / sibling modules) as seen in `crates/nockchain/src/lib.rs` and `crates/nockup/src/main.rs`.

**Path Aliases:**
- Not applicable; Rust module paths and crate names are used directly (examples: `nockapp::...`, `nockchain_types::...`, `crate::...` in `crates/nockchain-wallet/src/main.rs`).

## Error Handling

**Patterns:**
- Use typed domain errors with `thiserror` for library boundaries (`IngestError` in `crates/nockchain-bench/src/speed_of_light/guard/ingest.rs`).
- Use `Result<T, Box<dyn std::error::Error>>` for CLI/test command boundaries when error diversity is wide (`crates/nockchain-bench/src/main.rs`, `crates/nockchain-types/tests/raw_tx_from_jam_v0.rs`).
- Convert/annotate lower-level errors with `map_err` and contextual strings at IO/config boundaries (`crates/bridge/src/config.rs`, `crates/nockchain-wallet/src/main.rs`).
- Fail fast with `expect`/`panic!` in explicit invariant paths and test-style assertions (`crates/nockchain/src/lib.rs`, `crates/nockapp/tests/integration.rs`).

## Logging

**Framework:** `tracing`

**Patterns:**
- Use structured operational logs in runtime paths (`tracing::info!`, `tracing::warn!`, `tracing::debug!`) as seen in `crates/bridge/src/config.rs`, `crates/bridge/src/signing.rs`, and `crates/nockapp-grpc/src/services/public_nockchain/v2/server.rs`.
- Use `#[tracing::instrument]` on key flows (`crates/nockchain/src/lib.rs`, `crates/nockapp/tests/integration.rs`).

## Comments

**When to Comment:**
- Use comments for architectural intent and operational caveats, not line-by-line narration (examples: crate-level allow rationale in `crates/bridge/src/lib.rs`, driver-init flow notes in `crates/nockchain/src/lib.rs`, slow-test caveat in `crates/hoonc/tests/build.rs`).

**JSDoc/TSDoc:**
- Not applicable; Rustdoc (`///`) and module docs (`//!`) are used instead (`crates/nockapp/src/lib.rs`, `crates/nockchain/src/lib.rs`, `crates/bridge/tests/failover_tests.rs`).

## Function Design

**Size:**
- Keep utility functions focused and single-purpose (`parse_runs_manifest` in `crates/nockchain-bench/src/speed_of_light/guard/ingest.rs`), but command/runtime orchestration functions can be large when they coordinate multiple subsystems (`init_with_kernel` in `crates/nockchain/src/lib.rs`).

**Parameters:**
- Prefer explicit typed parameters and config structs over untyped maps (`BridgeConfigToml` conversion in `crates/bridge/src/config.rs`, command enums in `crates/nockup/src/main.rs`).

**Return Values:**
- Return `Result` from IO/parse/runtime functions and propagate with `?` (`crates/nockup/src/manifest.rs`, `crates/bridge/src/config.rs`, `crates/nockchain-types/tests/balance_from_peek_v0.rs`).

## Module Design

**Exports:**
- Use crate roots to declare module surface with `pub mod ...` and controlled re-exports where useful (`crates/nockup/src/lib.rs`, `crates/nockapp/src/lib.rs`, `crates/bridge/src/lib.rs`).

**Barrel Files:**
- Rust-style module barrels are used (`lib.rs`, `mod.rs`) rather than language-level alias barrels; keep declarations centralized in `crates/nockup/src/commands/mod.rs` and crate roots.

---

*Convention analysis: 2026-02-24*
