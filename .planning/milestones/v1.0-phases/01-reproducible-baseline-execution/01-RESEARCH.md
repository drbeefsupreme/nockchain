# Phase 1: Reproducible Baseline Execution - Research

**Researched:** 2026-02-24
**Domain:** Benchmark orchestration, provenance, Bash/Rust/CI integration
**Confidence:** HIGH

## Summary

Phase 1 layers a configuration-first baseline entrypoint on top of the existing `scripts/sol_bench_matrix.sh` benchmark engine. The existing codebase already has strong foundations: `toml` parsing in Cargo dependencies, a mature `guard/` module with models, contracts, ingestion and reporting, and a Bash script with full CLI argument parsing. The work is primarily additive: a TOML config file, a wrapper script, a provenance manifest writer, and a CI workflow.

The key architectural insight is that `sol_bench_matrix.sh` stays as-is (the execution engine) while a new `sol_baseline_run.sh` script becomes the single deterministic entrypoint that reads config, validates the working tree, collects provenance, invokes the matrix script, and writes the canonical manifest. CI calls the same script with a thin wrapper for artifact upload.

**Primary recommendation:** Build from the outside in: config contract first (TOML profiles), then the wrapper script, then provenance emission, then CI parity. Each layer is independently testable.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Run directories use combined naming: `{ISO-timestamp}_{commit-sha-prefix}/` (e.g., `2026-02-24T15-30-00Z_abc1234/`)
- Artifacts live in top-level `bench-artifacts/` directory (gitignored), consistent with existing repo convention
- Internal run structure grouped by concern: `data/` (raw results), `meta/` (provenance, config snapshot), `logs/` (stderr, timing)
- A `latest` symlink points to the most recent run directory for quick access by scripts and inspection
- Two profiles in a single TOML file with sections: `[defaults]`, `[quick]`, and `[full]`
  - `quick` profile: low warmup, small sample size -- fast local iteration
  - `full` profile: higher warmup, larger sample -- trusted baseline generation
- TOML format (Rust ecosystem standard, supports comments)
- CLI flags can override individual config values (precedence: config file < CLI flag)
- Profile selection via `--profile=quick|full` flag
- Comprehensive environment fingerprint: git commit, branch, rustc version, cargo version, OS name/version, CPU model, RAM size, CPU core count, CPU frequency, kernel version, active cgroups/limits, filtered env vars
- Strict validation: manifest must have all required fields or the run fails -- no incomplete artifacts
- Manifest includes both SHA-256 hash of resolved config AND full embedded config for comparison and inspection
- Dirty working tree blocks runs entirely -- forces clean commits for reproducibility
- Same core script (`sol_baseline_run.sh`) used by both local and CI; CI adds a thin wrapper for artifact upload and env setup
- CI uploads the full run directory (data/, meta/, logs/) as a GitHub Actions artifact
- Same default verbosity for both environments; `--verbose` and `--quiet` flags to adjust
- CI workflow uses `workflow_dispatch` only (manual trigger) -- scheduled runs are Phase 3 scope
- Existing `scripts/sol_bench_matrix.sh` stays as the benchmark execution engine; new work layers a configuration-first entrypoint on top
- Provenance guard module lives in `crates/nockchain-bench/src/speed_of_light/guard/` -- validates required fields before accepting artifacts
- The "block dirty runs" policy means local development requires committing (or stashing) before benchmarking -- this is intentional for reproducibility

### Claude's Discretion
- Exact TOML key names and structure within profile sections
- Loading skeleton and error message formatting
- Log file format and rotation
- Specific cgroups/env var filtering logic
- Temp file and cleanup handling

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ORCH-01 | Single scripted command to generate baseline benchmark runs locally | Wrapper script `sol_baseline_run.sh` pattern; calls existing `sol_bench_matrix.sh` with config-derived arguments |
| ORCH-02 | Same workflow in CI with equivalent configuration semantics | Thin CI wrapper `sol_baseline_ci.sh` + `workflow_dispatch` GitHub Actions workflow; same core script |
| ORCH-03 | Configurable warmup, sample size, iteration controls via versioned config | TOML config with `[defaults]`, `[quick]`, `[full]` sections; `--profile` flag + CLI overrides |
| DATA-01 | Persist each run as machine-readable canonical artifact | JSON manifest in `meta/manifest.json` within run directory; validated by guard module |
| DATA-02 | Captured provenance: commit SHA, branch, config, environment fingerprint, tool versions | `RunProvenance` struct in guard/provenance.rs; strict field validation; SHA-256 config hash |
</phase_requirements>

## Standard Stack

### Core
| Library/Tool | Version | Purpose | Why Standard |
|-------------|---------|---------|--------------|
| toml (Rust) | 0.8 | Config file parsing | Already in Cargo.toml; Rust ecosystem standard for config |
| serde + serde_json | 1.0 | Manifest serialization | Already in dependencies; canonical JSON output |
| clap | 4 | CLI argument parsing | Already in nockchain-bench; derive macros for type-safe args |
| sha2 (Rust) | 0.10 | Config content hashing | Standard Rust SHA-256; or use `sha256sum` in Bash |
| Bash 4+ | system | Script orchestration | Existing pattern in scripts/; `set -euo pipefail` convention |
| GitHub Actions | v4 | CI workflow | Repository already uses GH Actions |

### Supporting
| Library/Tool | Version | Purpose | When to Use |
|-------------|---------|---------|-------------|
| tempfile | 3.10 | Temporary file handling in tests | Already in dev-dependencies |
| chrono | 0.4 | ISO timestamp generation | Already in dependencies |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| TOML config | JSON/YAML config | TOML is already in deps, supports comments, Rust-native |
| Bash wrapper | Rust CLI entrypoint | Bash matches existing script patterns; lower friction |
| sha256sum (shell) | sha2 crate (Rust) | Shell is simpler for manifest; Rust if guard validates in-process |

## Architecture Patterns

### Recommended Project Structure
```
scripts/
  sol_baseline_run.sh       # NEW: Single entrypoint (local + CI)
  sol_baseline_ci.sh        # NEW: Thin CI wrapper (artifact upload, env)
  sol_bench_matrix.sh       # EXISTING: Benchmark execution engine (untouched)
benchmarks/baseline/
  sol-baseline.toml         # NEW: Versioned config with profiles
crates/nockchain-bench/src/speed_of_light/guard/
  provenance.rs             # NEW: RunProvenance struct + validation + writer
  mod.rs                    # MODIFIED: re-export provenance
bench-artifacts/            # EXISTING: gitignored output root
  {timestamp}_{sha}/
    data/                   # Raw benchmark results
    meta/
      manifest.json         # Canonical provenance manifest
      config-snapshot.toml  # Resolved config copy
    logs/
      run.log               # Combined stderr/timing
    latest -> ../latest-run # Symlink
.github/workflows/
  sol-baseline.yml          # NEW: workflow_dispatch CI workflow
```

### Pattern 1: Config Resolution Chain
**What:** TOML defaults -> profile override -> CLI flag override
**When to use:** Every baseline run
**Example:**
```bash
# In sol_baseline_run.sh
PROFILE="${PROFILE:-quick}"
# Parse TOML, resolve profile section on top of defaults
# Apply any --passes, --warmup CLI overrides on top
```

### Pattern 2: Dirty Tree Guard
**What:** Check `git diff --quiet && git diff --cached --quiet` before any benchmark work
**When to use:** First thing in sol_baseline_run.sh
**Example:**
```bash
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: Working tree is dirty. Commit or stash changes first." >&2
  exit 1
fi
```

### Pattern 3: Provenance Collection in Bash
**What:** Collect environment fingerprint as JSON from shell commands
**When to use:** After benchmark run, before manifest write
**Example:**
```bash
GIT_COMMIT=$(git rev-parse HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
RUSTC_VERSION=$(rustc --version)
CARGO_VERSION=$(cargo --version)
OS_INFO=$(uname -srm)
CPU_MODEL=$(lscpu | grep "Model name" | sed 's/.*: *//')
```

### Anti-Patterns to Avoid
- **Modifying sol_bench_matrix.sh internals:** Layer on top, don't refactor the engine
- **Optional provenance fields:** All fields are required or the run fails
- **Parsing TOML in Bash:** Use a minimal parser or generate shell vars from Rust; don't write a full TOML parser in Bash

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| TOML parsing | Bash TOML parser | `toml` crate or simple key extraction | TOML has complex types (arrays, inline tables) |
| SHA-256 hashing | Custom hash | `sha256sum` (shell) or `sha2` crate | Standard, verified implementations |
| JSON serialization | String concatenation | `jq` (shell) or `serde_json` (Rust) | Proper escaping, nested structures |
| ISO 8601 timestamps | Manual formatting | `date -u +%Y-%m-%dT%H-%M-%SZ` | Timezone handling, format consistency |

**Key insight:** The Bash scripts should delegate structured data operations to established tools (jq, sha256sum) or to Rust code. Shell is for orchestration, not data manipulation.

## Common Pitfalls

### Pitfall 1: TOML Config Parsed in Pure Bash
**What goes wrong:** Bash has no native TOML parser; hand-rolled parsing breaks on edge cases
**Why it happens:** Desire to avoid Rust compilation for config reading
**How to avoid:** Use a small Rust helper binary or `nockchain-bench --dump-config` subcommand that reads TOML and emits shell-friendly key=value pairs
**Warning signs:** Regex-based TOML parsing, quoting issues in config values

### Pitfall 2: Race Condition on latest Symlink
**What goes wrong:** Concurrent runs overwrite `latest` symlink
**Why it happens:** `ln -sf` is not atomic on all filesystems
**How to avoid:** Use `ln -sfn` with a temporary name then `mv` for atomic swap
**Warning signs:** Broken symlinks, wrong `latest` target after parallel runs

### Pitfall 3: Incomplete Provenance on CI
**What goes wrong:** CI environment lacks commands like `lscpu` or has different `uname` flags
**Why it happens:** CI runners use minimal container images
**How to avoid:** Wrap each fingerprint collection in a fallback: `$(lscpu 2>/dev/null | grep "Model name" | ... || echo "unknown")`
**Warning signs:** Empty or "command not found" in manifest fields

### Pitfall 4: Git Dirty Check Fails in CI
**What goes wrong:** CI checkout may have modified files (e.g., LFS, submodule init)
**Why it happens:** `actions/checkout` with certain settings modifies working tree
**How to avoid:** Run dirty check after checkout, allow `--allow-dirty` override in CI wrapper only (not in core script)
**Warning signs:** CI runs always fail dirty check

## Code Examples

### TOML Config Structure
```toml
[defaults]
fixtures_dir = "bench-artifacts/fixtures"
output_root = "bench-artifacts/sol-baseline"
passes = 2
enable_checkpointing = false
envs = "native"

[quick]
passes = 1

[full]
passes = 5
envs = "native,docker"
docker_memory = "16g"
```

### Manifest JSON Structure
```json
{
  "schema_version": "1",
  "timestamp": "2026-02-24T15:30:00Z",
  "git_commit": "abc1234def5678...",
  "git_branch": "main",
  "benchmark_config": {
    "profile": "full",
    "passes": 5,
    "fixtures_dir": "bench-artifacts/fixtures",
    "enable_checkpointing": false
  },
  "config_sha256": "e3b0c44298fc1c14...",
  "environment": {
    "os": "Linux 6.x x86_64",
    "kernel": "6.17.0-14-generic",
    "cpu_model": "...",
    "cpu_cores": 16,
    "ram_bytes": 34359738368,
    "rustc_version": "rustc 1.82.0",
    "cargo_version": "cargo 1.82.0"
  },
  "tool_versions": {
    "nockchain_bench": "0.1.0",
    "nockvm": "..."
  }
}
```

### GitHub Actions Workflow Structure
```yaml
name: SOL Baseline
on:
  workflow_dispatch:
    inputs:
      profile:
        description: 'Benchmark profile'
        required: false
        default: 'full'
        type: choice
        options: [quick, full]

jobs:
  baseline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
      - name: Build nockchain-bench
        run: cargo build --release -p nockchain-bench
      - name: Run baseline
        run: scripts/sol_baseline_ci.sh --profile ${{ inputs.profile }}
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: sol-baseline-${{ github.sha }}
          path: bench-artifacts/sol-baseline/latest/
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|-------------|------------------|--------------|--------|
| Ad-hoc benchmark runs | `sol_bench_matrix.sh` with CLI args | Pre-project | Standardized execution |
| No provenance tracking | Guard module with contracts | Pre-project | Statistical validation exists |
| Manual parameter passing | Config-file-first with profiles | This phase | Reproducible, versioned runs |

## Open Questions

1. **TOML Parsing Strategy for Bash**
   - What we know: `toml` crate is in Cargo.toml; Bash cannot natively parse TOML
   - What's unclear: Whether to add a `nockchain-bench config-dump` subcommand or use a lightweight approach
   - Recommendation: Add a `--dump-config` flag to nockchain-bench that reads TOML and emits `KEY=VALUE` lines for Bash sourcing. This keeps the Rust TOML parser as single source of truth.

2. **Fixture Availability in CI**
   - What we know: Benchmarks need `.soltest` fixture files; they may be large
   - What's unclear: Whether CI will have pre-built fixtures or needs to build them
   - Recommendation: For Phase 1, assume fixtures are available (built by `sol_build_fixtures.sh` as a prerequisite step). Document the dependency.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `scripts/sol_bench_matrix.sh` -- full CLI interface, argument parsing, execution flow
- Codebase analysis: `crates/nockchain-bench/Cargo.toml` -- dependency versions (toml 0.8, serde 1.0, clap 4, chrono 0.4)
- Codebase analysis: `crates/nockchain-bench/src/speed_of_light/guard/` -- existing model, contract, baseline, report modules
- CONTEXT.md: User-locked decisions on artifact layout, config format, provenance depth, CI parity

### Secondary (MEDIUM confidence)
- GitHub Actions `upload-artifact@v4` and `workflow_dispatch` -- standard patterns, well-documented

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in project dependencies
- Architecture: HIGH - builds on existing patterns in codebase
- Pitfalls: HIGH - based on direct codebase analysis and known CI behaviors

**Research date:** 2026-02-24
**Valid until:** 2026-03-24 (stable domain, no fast-moving dependencies)
