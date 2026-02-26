---
phase: 01-reproducible-baseline-execution
plan: 01
subsystem: infra
tags: [toml, config, clap, sha2, baseline]

requires: []
provides:
  - "Versioned TOML config with quick/full profiles"
  - "Config loader with profile resolution and shell-friendly dump"
  - "config-dump CLI subcommand for Bash consumption"
  - "Config SHA-256 hashing for provenance"
affects: [01-03, phase-2, phase-3]

tech-stack:
  added: [sha2]
  patterns: ["TOML config with profile overlay", "config-dump for shell integration"]

key-files:
  created:
    - benchmarks/baseline/sol-baseline.toml
    - crates/nockchain-bench/src/speed_of_light/config.rs
    - crates/nockchain-bench/tests/sol_baseline_config.rs
  modified:
    - crates/nockchain-bench/src/speed_of_light/mod.rs
    - crates/nockchain-bench/src/main.rs
    - crates/nockchain-bench/Cargo.toml

key-decisions:
  - "config-dump as top-level CLI subcommand (not nested under sol)"
  - "Shell-friendly KEY=VALUE output for eval in Bash scripts"
  - "Profile overlay: defaults merged with profile-specific overrides"

patterns-established:
  - "TOML config with [defaults] + named profile sections for override"
  - "Rust config module with load_config/dump_shell_vars/config_sha256"

requirements-completed: [ORCH-03]

duration: 8min
completed: 2026-02-24
---

# Phase 1 Plan 01: Versioned Baseline Config Summary

**TOML config with quick/full profiles, Rust config loader with profile resolution, and config-dump CLI subcommand for Bash integration**

## Performance

- **Duration:** 8 min
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- TOML config at benchmarks/baseline/sol-baseline.toml with [defaults], [quick], [full] sections
- Config loader resolves profiles by overlaying optional overrides on defaults
- config-dump subcommand emits KEY=VALUE pairs for Bash eval
- SHA-256 hashing of raw config for provenance tracking
- 5 contract tests pass (file exists, profile resolution, shell dump format, hash determinism)

## Task Commits

1. **Task 1: Create TOML config with profiles and Rust config loader** - `462e17b` (feat)
2. **Task 2: Add config-dump subcommand and contract tests** - `462e17b` (same commit)

## Files Created/Modified
- `benchmarks/baseline/sol-baseline.toml` - Versioned config with [defaults], [quick], [full] profiles
- `crates/nockchain-bench/src/speed_of_light/config.rs` - Config loading, profile resolution, shell dump, SHA-256
- `crates/nockchain-bench/src/main.rs` - Added config-dump subcommand
- `crates/nockchain-bench/Cargo.toml` - Added sha2 workspace dependency
- `crates/nockchain-bench/tests/sol_baseline_config.rs` - 5 contract tests

## Decisions Made
- Used CARGO_MANIFEST_DIR env var in tests for reliable path resolution across workspace
- Shell quoting with single quotes for values containing special characters

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
- Integration tests needed CARGO_MANIFEST_DIR-based path resolution since cargo runs tests from crate directory, not workspace root. Fixed by using `env!("CARGO_MANIFEST_DIR")` to construct absolute paths.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Config contract ready for sol_baseline_run.sh consumption
- config-dump subcommand ready for eval in Bash scripts

---
*Phase: 01-reproducible-baseline-execution*
*Completed: 2026-02-24*
