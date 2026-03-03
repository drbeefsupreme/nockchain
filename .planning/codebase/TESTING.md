# Testing Patterns

**Analysis Date:** 2026-03-03

## Test Framework

**Runner:**
- `cargo test` for unit + integration coverage across crates
- targeted package runs (`cargo test -p nockchain-bench`) for bench-focused loops

**Run Commands:**
```bash
cargo test
cargo test -p nockchain-bench
cargo test -p nockchain-bench --test sol_guard_cli
```

## Test File Organization

**Location:**
- module-local tests in `src/**`
- integration tests in `crates/nockchain-bench/tests/`
- fixtures in `crates/nockchain-bench/tests/fixtures/guard/`

**Naming:**
- integration tests use descriptive scenario names (`sol_comparison.rs`, `sol_provenance_manifest.rs`)

## Test Structure

- arrange/act/assert patterns in unit tests
- CLI integration tests verify exit semantics, report artifacts, and contract behavior

## Mocking and Fixtures

- fixture-heavy approach for SOL artifact validation
- realistic end-to-end paths sometimes rely on large artifacts and are marked `#[ignore]`

## Coverage Signals

- strong unit/integration coverage in many bench modules
- high-cost paths (Docker/full-checkpoint) often gated behind ignored tests and optional CI jobs

## Test Types

**Unit:** parsing, config loading, contract evaluation, archive/provenance invariants

**Integration:** CLI command behavior, fixture crossover/regression paths, manifest/report outputs

**System/Perf:** script-driven matrix runs and baseline comparisons outside normal `cargo test`

## Current Gaps

- ignored tests reduce always-on safety for critical replay/checkpoint flows
- benchmark validity assertions can drift from real-world branch divergences without stronger gating

---
*Testing analysis: 2026-03-03*
*Update when test patterns change*
