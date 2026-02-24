# Phase 1: Reproducible Baseline Execution - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish a single local/CI workflow that produces canonical benchmark artifacts with full provenance. Maintainers can run one scripted command locally or in CI to generate baseline benchmark runs for `nockchain-bench`, with versioned configuration controlling warmup, sample size, and iteration parameters. Creating comparison logic, regression gates, and scheduled automation are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Artifact layout & naming
- Run directories use combined naming: `{ISO-timestamp}_{commit-sha-prefix}/` (e.g., `2026-02-24T15-30-00Z_abc1234/`)
- Artifacts live in top-level `bench-artifacts/` directory (gitignored), consistent with existing repo convention
- Internal run structure grouped by concern: `data/` (raw results), `meta/` (provenance, config snapshot), `logs/` (stderr, timing)
- A `latest` symlink points to the most recent run directory for quick access by scripts and inspection

### Configuration defaults & format
- Two profiles in a single TOML file with sections: `[defaults]`, `[quick]`, and `[full]`
  - `quick` profile: low warmup, small sample size — fast local iteration
  - `full` profile: higher warmup, larger sample — trusted baseline generation
- TOML format (Rust ecosystem standard, supports comments)
- CLI flags can override individual config values (precedence: config file < CLI flag)
- Profile selection via `--profile=quick|full` flag

### Provenance depth
- Comprehensive environment fingerprint: git commit, branch, rustc version, cargo version, OS name/version, CPU model, RAM size, CPU core count, CPU frequency, kernel version, active cgroups/limits, filtered env vars
- Strict validation: manifest must have all required fields or the run fails — no incomplete artifacts
- Manifest includes both SHA-256 hash of resolved config AND full embedded config for comparison and inspection
- Dirty working tree blocks runs entirely — forces clean commits for reproducibility

### Local vs CI parity
- Same core script (`sol_baseline_run.sh`) used by both local and CI; CI adds a thin wrapper for artifact upload and env setup
- CI uploads the full run directory (data/, meta/, logs/) as a GitHub Actions artifact
- Same default verbosity for both environments; `--verbose` and `--quiet` flags to adjust
- CI workflow uses `workflow_dispatch` only (manual trigger) — scheduled runs are Phase 3 scope

### Claude's Discretion
- Exact TOML key names and structure within profile sections
- Loading skeleton and error message formatting
- Log file format and rotation
- Specific cgroups/env var filtering logic
- Temp file and cleanup handling

</decisions>

<specifics>
## Specific Ideas

- Existing `scripts/sol_bench_matrix.sh` stays as the benchmark execution engine; new work layers a configuration-first entrypoint on top
- Provenance guard module lives in `crates/nockchain-bench/src/speed_of_light/guard/` — validates required fields before accepting artifacts
- The "block dirty runs" policy means local development requires committing (or stashing) before benchmarking — this is intentional for reproducibility

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-reproducible-baseline-execution*
*Context gathered: 2026-02-24*
