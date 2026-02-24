# Testing Patterns

**Analysis Date:** 2026-02-24

## Test Framework

**Runner:**
- Rust built-in test harness via `cargo test` across workspace crates (`Makefile`, `.gitlab-ci.yml`).
- Config: Not detected as a standalone custom file (no `nextest.toml`, no `cargo-llvm-cov` config found in repository root).

**Assertion Library:**
- Standard Rust assertions (`assert!`, `assert_eq!`, `matches!`) used broadly in `crates/bridge/tests/config_tests.rs`, `crates/nockchain-types/tests/raw_tx_from_jam_v0.rs`, and `crates/nockchain-bench/tests/sol_guard_cli.rs`.

**Run Commands:**
```bash
cargo test --release                    # Run all tests (see `Makefile` and `.gitlab-ci.yml`)
cargo test -p nockchain-bench sol_guard # Run focused bench guard tests (`Makefile`)
Not detected                            # Watch mode command is not defined in repo scripts/config
```

## Test File Organization

**Location:**
- Use both integration tests under `tests/` and in-file unit tests under `#[cfg(test)] mod tests`.
- Examples: `crates/bridge/tests/*.rs`, `crates/nockchain-bench/tests/*.rs`, and `crates/nockchain-bench/src/speed_of_light/guard/ingest.rs`.

**Naming:**
- Integration test files use behavior-focused snake_case names (`failover_tests.rs`, `config_tests.rs`, `sol_guard_cli.rs`).
- Unit test functions use `test_*` naming (`crates/bridge/tests/failover_tests.rs`, `crates/nockchain-libp2p-io/src/cbor_tests.rs`).

**Structure:**
```text
crates/<crate>/
├── src/**/*.rs         # Production code with optional #[cfg(test)] modules
└── tests/**/*.rs       # Integration tests and harnesses
```

## Test Structure

**Suite Organization:**
```typescript
// Rust pattern used across crates
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_name() {
        // arrange
        // act
        // assert
    }

    #[tokio::test]
    async fn test_async_case() {
        // async arrange/act/assert
    }
}
```

**Patterns:**
- Setup pattern: temp directories/files and generated fixtures (`tempfile::TempDir`, `tempfile::tempdir`) in `crates/bridge/tests/config_tests.rs` and `crates/nockchain-bench/tests/sol_guard_cli.rs`.
- Teardown pattern: rely on RAII cleanup for temp resources, plus explicit cleanup where needed (`crates/hoonc/tests/build.rs`).
- Assertion pattern: explicit scenario checks plus structural/value invariants (`crates/nockchain-types/tests/balance_from_peek_v0.rs`, `crates/bridge/tests/failover_tests.rs`).

## Mocking

**Framework:**
- Custom/manual mocks and harnesses; no `mockall`-style mocking framework detected.

**Patterns:**
```typescript
// Rust pattern from `crates/bridge/tests/test_harness.rs`
pub struct MockBaseContract {
    submissions: Arc<Mutex<Vec<Submission>>>,
    processed: Arc<Mutex<HashMap<DepositId, TxHash>>>,
}

let verify_fn = |_hash: &[u8; 32], _sig: &[u8]| Some(signer_address);
```

**What to Mock:**
- External/blockchain/network behavior through deterministic in-memory fakes (`MockBaseContract`, `TestCluster`) in `crates/bridge/tests/test_harness.rs`.
- CLI process behavior through subprocess execution and exit code assertions in `crates/nockchain-bench/tests/sol_guard_cli.rs`.

**What NOT to Mock:**
- Core encode/decode and data structure roundtrips; test actual serialization logic directly (`crates/nockchain-types/tests/*.rs`, `crates/noun-serde/tests/serde.rs`).

## Fixtures and Factories

**Test Data:**
```typescript
// Rust fixture patterns used in repo
const RAW_TX_JAM: &[u8] = include_bytes!("../jams/v0/raw-tx.jam");
let summary = fixture_path("combined_summary.tsv");
let temp = tempfile::tempdir().expect("tempdir");
```

**Location:**
- Embedded binary fixtures with `include_bytes!` in `crates/nockchain-types/tests/raw_tx_from_jam_v0.rs` and `crates/nockchain-types/tests/balance_from_peek_v0.rs`.
- File fixtures under `tests/fixtures` in `crates/nockchain-bench/tests/fixtures/guard/`.
- Guidance for adding more fixture coverage in `crates/nockchain-types/jams/README.md`.

## Coverage

**Requirements:** None enforced globally.

**View Coverage:**
```bash
Not detected
```

## Test Types

**Unit Tests:**
- Heavily used via `#[cfg(test)]` modules for parser/logic helpers and crate internals (`crates/nockchain-bench/src/speed_of_light/guard/ingest.rs`, `crates/nockchain-libp2p-io/src/cbor_tests.rs`).

**Integration Tests:**
- Widely used under `tests/` for multi-component behavior (`crates/bridge/tests/failover_tests.rs`, `crates/nockapp/tests/integration.rs`, `crates/hoonc/tests/build.rs`).

**E2E Tests:**
- Not used as a dedicated framework; closest equivalent is long-running/ignored integration flows (`#[ignore]` in `crates/nockapp/tests/integration.rs` and `crates/hoonc/tests/build.rs`).

## Common Patterns

**Async Testing:**
```typescript
#[tokio::test]
async fn test_proposer_offline_failover() {
    let mut cluster = TestCluster::new(5).await;
    cluster.trigger_deposit(deposit.clone()).await;
    cluster.wait_for_signatures(&deposit, 4).await;
}
```

**Error Testing:**
```typescript
#[test]
fn test_missing_confirmation_depths_fails_parse() {
    let result = BridgeConfigToml::from_file(&config_path);
    assert!(result.is_err());
}
```

---

*Testing analysis: 2026-02-24*
