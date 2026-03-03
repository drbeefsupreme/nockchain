# Coding Conventions

**Analysis Date:** 2026-03-03

## Naming Patterns

**Files:**
- Rust modules are snake_case file names
- test files use descriptive snake_case names under `tests/`

**Functions/Variables:**
- Rust idiomatic snake_case for functions and variables
- constants in SCREAMING_SNAKE_CASE

**Types:**
- structs/enums/traits in PascalCase
- error types use `Error` suffix and `thiserror` derives in many modules

## Code Style

**Formatting:**
- repository `rustfmt.toml` enforces width/import grouping conventions
- common use of derive-heavy declarations (`clap`, `serde`, `ValueEnum`)

**Linting:**
- clippy is used; some crates explicitly allow selective lints for practical reasons

## Import Organization

- grouped imports with rustfmt conventions
- module facades frequently re-export with `pub use` in `mod.rs`

## Error Handling

- domain modules use typed errors (`thiserror` enums)
- CLI boundaries normalize to top-level exit behavior and user-readable messages

## Logging

- `tracing` spans/events used across benchmark and runtime paths
- profiler integrations optionally enabled (`tracy`, `perf`)

## Comments

- comments are used for non-obvious behavior, artifact formats, and invariants
- avoid redundant comments that restate obvious code

## Function and Module Design

- benchmark features are organized by subsystem modules (`runner`, `scenario`, `speed_of_light`)
- entrypoint performs dispatch; heavy logic lives in subsystem modules

## Practical Guidance

- follow existing module ownership boundaries rather than adding cross-cutting utility sprawl
- preserve artifact compatibility and explicit CLI semantics when modifying command paths

---
*Convention analysis: 2026-03-03*
*Update when patterns change*
