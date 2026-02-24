---
phase: 01-reproducible-baseline-execution
plan: 02
subsystem: infra
tags: [provenance, json, serde, guard, manifest]

requires: []
provides:
  - "RunProvenance struct for canonical manifest schema"
  - "Strict manifest validation (rejects incomplete artifacts)"
  - "JSON manifest writer with pre-write validation"
  - "Reference manifest fixture for testing"
affects: [01-03, phase-2, phase-3]

tech-stack:
  added: []
  patterns: ["Provenance model in guard/ module", "Strict validation before write"]

key-files:
  created:
    - crates/nockchain-bench/src/speed_of_light/guard/provenance.rs
    - crates/nockchain-bench/tests/sol_provenance_manifest.rs
    - crates/nockchain-bench/tests/fixtures/guard/run-manifest.json
  modified:
    - crates/nockchain-bench/src/speed_of_light/guard/mod.rs

key-decisions:
  - "Strict validation: all required fields checked, any failure rejects entire manifest"
  - "write_manifest validates before writing to prevent partial artifacts on disk"
  - "Nullable fields (cpu_frequency_mhz, active_cgroups) use Option<T>"

patterns-established:
  - "Provenance as a first-class type in guard/ module"
  - "Validate-before-write pattern for artifact integrity"

requirements-completed: [DATA-01, DATA-02]

duration: 6min
completed: 2026-02-24
---

# Phase 1 Plan 02: Provenance Manifest Summary

**RunProvenance model with strict validation and canonical JSON manifest writer in the guard module**

## Performance

- **Duration:** 6 min
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- RunProvenance struct captures all provenance fields: git commit/branch, config, environment fingerprint, tool versions
- validate_manifest strictly checks 9 field constraints (schema version, commit format, branch, timestamp, SHA, OS, CPU, RAM, rustc)
- write_manifest validates before writing -- no incomplete artifacts on disk
- Reference fixture at tests/fixtures/guard/run-manifest.json
- 8 contract tests pass (valid manifest, 5 invalid field variants, write success, write rejection)

## Task Commits

1. **Task 1: Create RunProvenance model and validation logic** - `11bd3b0` (feat)
2. **Task 2: Create test fixture and contract tests** - `11bd3b0` (same commit)

## Files Created/Modified
- `crates/nockchain-bench/src/speed_of_light/guard/provenance.rs` - Provenance model, validation, JSON writer
- `crates/nockchain-bench/src/speed_of_light/guard/mod.rs` - Re-exports for provenance types
- `crates/nockchain-bench/tests/fixtures/guard/run-manifest.json` - Reference manifest fixture
- `crates/nockchain-bench/tests/sol_provenance_manifest.rs` - 8 contract tests

## Decisions Made
- cpu_frequency_mhz and active_cgroups as Option<T> since not available on all systems
- Validation returns Vec<String> of all errors (not just first) for comprehensive feedback

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Provenance schema ready for manifest.json generation in sol_baseline_run.sh
- Guard module patterns available for Phase 2 comparison logic

---
*Phase: 01-reproducible-baseline-execution*
*Completed: 2026-02-24*
