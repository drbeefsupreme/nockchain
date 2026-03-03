# Nockchain Bench Reset And Master Graft

## What This Is

This project restores trust in `nockchain-bench` by rebuilding an apples-to-apples benchmark baseline against `nockchain/master`. We will audit the current `nockchain-bench` implementation, identify every dependency on features absent from `nockchain/master`, and produce a clean graft strategy onto a fresh master-based branch. The primary user is the maintainer/operator responsible for benchmark validity and performance comparisons.

## Core Value

Benchmark outputs must reflect `nockchain` runtime behavior, not branch-specific harness cruft.

## Requirements

### Validated

- ✓ `nockchain-bench` provides executable benchmark flows for container and SOL replay paths — existing
- ✓ `nockchain-bench` produces structured artifacts and guard/comparison outputs — existing
- ✓ repository contains branch/matrix automation and historical benchmark artifacts for forensic comparison — existing

### Active

- [ ] Build a complete feature-dependency inventory of `nockchain-bench` against `nockchain/master` (all missing symbols, APIs, data contracts, and behavior assumptions).
- [ ] Trace historical provenance from `bitemyapp/ag2-opt-persistence-madvise-checkpoint-chaff-pma-gc-checkpoint-streaming` to explain how non-master features entered the bench crate.
- [ ] Classify each divergence (remove, replace with master equivalent, feature-gate, or isolate as optional extension).
- [ ] Produce a concrete graft plan to transplant `nockchain-bench` onto a new `nockchain/master` branch with zero references to absent features.
- [ ] Define verification criteria proving apples-to-apples comparisons across branches for SOL benchmarks.

### Out of Scope

- GitHub Actions redesign or CI pipeline overhaul — explicitly excluded for this project per user direction.
- New benchmark feature invention unrelated to restoring comparability — avoid expanding scope before baseline validity is re-established.
- Runtime feature development inside `nockchain/master` to satisfy bench assumptions — benchmark tooling must adapt to master, not the other way around.

## Context

- Current branch includes a `nockchain-bench` crate that diverged from upstream `nockchain/master` roughly three months ago.
- Prior attempts included both one-size-fits-many-branches bench variants and bespoke branch-specific variants; neither produced trustworthy comparisons.
- PMA-related behavior is a known example of branch-only coupling; additional couplings may include concepts like NounSpaces or other branch-specific runtime assumptions.
- Historical branch `bitemyapp/ag2-opt-persistence-madvise-checkpoint-chaff-pma-gc-checkpoint-streaming` is a likely source of divergence and must be used for provenance.
- Existing codebase map exists in `.planning/codebase/` and should be treated as baseline context for planning.

## Constraints

- **Scope**: Focus analysis on `nockchain-bench` and directly referenced runtime interfaces — avoid broad refactors outside benchmark graft goals.
- **Compatibility**: Target base must be `nockchain/master` as it exists at analysis time.
- **Method**: Differences must be evidence-backed (file path, symbol, commit provenance) and reproducible.
- **Quality**: Final graft plan must minimize optional behavior and remove dead/branch-specific cruft.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Treat this as a brownfield recovery effort instead of incremental tuning | Benchmark trust is currently compromised; we need a reset to known-good ground | — Pending |
| Use `nockchain/master` as the canonical target baseline | User objective is clean master graft and apples-to-apples comparisons | — Pending |
| Exclude GitHub Actions changes from current scope | Keep effort focused on bench/runtime compatibility and validity first | — Pending |

---
*Last updated: 2026-03-03 after initialization*
