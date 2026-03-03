# Phase 1 Scope Boundary Contract

## Objective

Define strict acceptance criteria for Phase 1 incompatibility findings so entries can be accepted or rejected without ad-hoc interpretation.

## In Scope

- Rust code only under `crates/nockchain-bench/src/**`.
- Direct references from bench Rust code to runtime interfaces, modules, types, symbols, config keys, and CLI/runtime bindings.
- Explicit behavior assumptions encoded in Rust bench logic, even when no single symbol name captures the dependency.
- Runtime-path incompatibilities recorded in the runtime section of the canonical findings ledger.
- Test-only incompatibilities recorded in a dedicated test-only findings section (never merged into runtime-path findings).

## Out of Scope

- non-Rust artifacts as primary evidence input, including shell output, generated reports, CI logs, dashboards, and ad-hoc notes.
- Findings sourced only from transitive runtime internals not directly referenced or assumed by bench Rust code.
- Non-bench crate audits outside `crates/nockchain-bench/src/**`.
- Narrative-only claims lacking a concrete file path and symbol/API reference (or explicit behavior-assumption reference).

## Module Boundary Inventory (Initial)

This inventory defines the initial Phase 1 analysis surface rooted in `crates/nockchain-bench/src/**`:

- `main.rs` (CLI surface and command wiring)
- `runner/**`
- `scenario/**`
- `sampler/**`
- `output/**`
- `speed_of_light/**` (including `bench`, `extractor`, `fixture`, `checkpoint`, `guard`, `compat`, `types`)

Runtime interface tracing is allowed only when starting from these bench modules and following direct references or behavior assumptions expressed in Rust code.

## Scope Gate Checklist

Every finding MUST pass all Scope Gate checks before entering the canonical ledger:

- [ ] Evidence source is Rust code only inside `crates/nockchain-bench/src/**`.
- [ ] Finding is rooted in direct references or explicit behavior assumptions.
- [ ] Finding documents whether it is runtime-path or test-only.
- [ ] Evidence does not rely on non-Rust artifacts as primary inclusion basis.
- [ ] Record includes concrete `file_path` and `symbol_or_api` (or a precise behavior-assumption locator).

If any Scope Gate item fails, reject the finding until evidence is corrected.

