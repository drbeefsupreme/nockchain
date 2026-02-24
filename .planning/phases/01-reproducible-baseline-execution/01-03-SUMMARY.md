---
phase: 01-reproducible-baseline-execution
plan: 03
subsystem: infra
tags: [bash, ci, github-actions, workflow-dispatch, provenance]

requires:
  - phase: 01-01
    provides: "Config loader and config-dump subcommand"
  - phase: 01-02
    provides: "RunProvenance schema for manifest structure"
provides:
  - "Single-command local baseline runner"
  - "CI wrapper with --allow-dirty"
  - "GitHub Actions workflow_dispatch workflow"
  - "Canonical run directory layout: data/meta/logs"
affects: [phase-2, phase-3]

tech-stack:
  added: []
  patterns: ["Bash wrapper calling Rust CLI for config", "jq for JSON manifest generation", "Atomic symlink update"]

key-files:
  created:
    - scripts/sol_baseline_run.sh
    - scripts/sol_baseline_ci.sh
    - .github/workflows/sol-baseline.yml
  modified: []

key-decisions:
  - "Runner script builds nockchain-bench if binary missing"
  - "Provenance collected in Bash (not Rust) for shell-level environment access"
  - "Atomic symlink update via temp + mv pattern"
  - "CI wrapper is a one-liner exec with --allow-dirty prepended"

patterns-established:
  - "sol_baseline_run.sh as canonical entrypoint for all baseline runs"
  - "Run directory naming: {timestamp}_{sha}/ with data/meta/logs structure"
  - "manifest.json in meta/ for provenance, config-snapshot.toml for config"

requirements-completed: [ORCH-01, ORCH-02]

duration: 5min
completed: 2026-02-24
---

# Phase 1 Plan 03: Local Runner and CI Parity Summary

**Single-command baseline runner with dirty tree guard, provenance manifest, and GitHub Actions workflow_dispatch CI workflow**

## Performance

- **Duration:** 5 min
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- sol_baseline_run.sh: single deterministic entrypoint with --config, --profile, --branch-bin flags
- Dirty tree guard blocks runs on uncommitted changes (--allow-dirty escape hatch for CI)
- Config loading via nockchain-bench config-dump with CLI override support
- Run directory with {timestamp}_{sha}/ naming and data/meta/logs structure
- Provenance manifest generation via jq with complete environment fingerprint
- Atomic latest symlink update
- sol_baseline_ci.sh: thin wrapper that adds --allow-dirty and forwards all args
- sol-baseline.yml: workflow_dispatch GitHub Actions workflow with profile selection

## Task Commits

1. **Task 1: Create local baseline runner script** - `8ec2f5d` (feat)
2. **Task 2: Create CI wrapper and GitHub Actions workflow** - `8ec2f5d` (same commit)

## Files Created/Modified
- `scripts/sol_baseline_run.sh` - Single-command baseline entrypoint (executable)
- `scripts/sol_baseline_ci.sh` - CI wrapper with --allow-dirty (executable)
- `.github/workflows/sol-baseline.yml` - workflow_dispatch CI workflow

## Decisions Made
- Provenance collected via shell commands with fallbacks for CI environments
- jq used for JSON generation (proper escaping, no string concatenation)
- Benchmark matrix errors captured but don't halt manifest writing

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 1 deliverables complete: local + CI baseline runs produce canonical artifacts
- Ready for Phase 2: regression comparison using baseline artifacts

---
*Phase: 01-reproducible-baseline-execution*
*Completed: 2026-02-24*
