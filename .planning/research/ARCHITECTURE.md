# Architecture Research

**Domain:** Reproducible benchmark baseline + regression-analysis pipeline (CI + local + static history publication)
**Researched:** 2026-02-24
**Confidence:** MEDIUM

## Standard Architecture

### System Overview

```text
┌──────────────────────────────────────────────────────────────────────┐
│                         Execution Surfaces                           │
├──────────────────────────────────────────────────────────────────────┤
│  Local CLI                                                          │
│  scripts/bench/run.sh  scripts/bench/analyze.sh  scripts/bench/publish.sh
│                                   │                                  │
│  GitHub Actions (push/schedule/workflow_dispatch)                   │
│  .github/workflows/bench-baseline.yml                               │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ emits raw benchmark outputs + metadata
┌──────────────────────────────▼───────────────────────────────────────┐
│                      Normalization + Analysis                        │
├──────────────────────────────────────────────────────────────────────┤
│  Adapter/Parser: benchmark stdout/json -> canonical schema           │
│  Regression Engine: compare candidate vs baseline/history             │
│  Verdict Engine: pass/warn/fail thresholds                            │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ writes versioned artifacts
┌──────────────────────────────▼───────────────────────────────────────┐
│                           Data Plane                                 │
├──────────────────────────────────────────────────────────────────────┤
│  Run Artifacts (immutable, per run):                                 │
│    bench-artifacts/runs/<run-id>/{raw,normalized,analysis,manifest}.json
│  Baseline Registry (rolling):                                        │
│    bench-artifacts/baselines/<benchmark>/<testbed>.json              │
│  Publication Bundle (static):                                        │
│    bench-artifacts/site/{index.html,data/*.json}                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ deploy artifact
┌──────────────────────────────▼───────────────────────────────────────┐
│                    Static History Publication                         │
├──────────────────────────────────────────────────────────────────────┤
│  GitHub Pages custom workflow: build -> upload-pages-artifact ->      │
│  deploy-pages                                                         │
│  Public output: longitudinal charts + downloadable JSON               │
└──────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Benchmark Runner | Execute benchmarks in deterministic mode for CI and local | Shell wrapper calling `cargo bench --locked` with pinned toolchain/env |
| Environment Fingerprinter | Capture run context for attribution | Script records commit SHA, branch, runner/host, CPU info, OS, benchmark config |
| Result Adapter | Convert harness output into canonical JSON | Rust or Python normalizer writing schema-stable payloads |
| Baseline Store | Persist latest accepted baseline per benchmark/testbed | Versioned JSON files in repo artifact directory |
| Regression Analyzer | Compare candidate runs to baseline/history with statistical thresholds | Existing `nockchain-bench` statistical engine + threshold policy file |
| Verdict Gate | Emit pass/warn/fail for CI status checks | CI step that exits non-zero on hard regression |
| Artifact Publisher | Upload immutable run package for traceability | `actions/upload-artifact` + digest validation |
| History Publisher | Build static history site and deploy to Pages | `actions/configure-pages`, `actions/upload-pages-artifact`, `actions/deploy-pages` |

## Recommended Project Structure

```text
.github/
└── workflows/
    ├── bench-baseline.yml          # Orchestrates collect->analyze->publish
    ├── bench-reusable.yml          # Reusable workflow for benchmark execution
    └── pages-deploy.yml            # Optional split deployment workflow

scripts/
└── bench/
    ├── run.sh                      # Deterministic benchmark execution entrypoint
    ├── normalize.py                # Raw output -> canonical JSON
    ├── analyze.py                  # Baseline comparison + regression verdict
    ├── publish_site.py             # Build static site payload from history
    └── env_fingerprint.sh          # Capture metadata for attribution

bench-artifacts/
├── schema/
│   ├── run-manifest.schema.json    # Contract for run metadata
│   └── metric-series.schema.json   # Contract for historical series
├── baselines/                      # Current accepted baselines
├── runs/                           # Immutable per-run snapshots
└── site/                           # GitHub Pages-ready static payload

docs/
└── benchmarking/
    ├── methodology.md              # Reproducibility protocol
    ├── threshold-policy.md         # Warn/fail criteria
    └── operating-playbook.md       # Local and CI invocation paths
```

### Structure Rationale

- **`scripts/bench/` boundary:** keeps benchmark orchestration independent from app/runtime code; easier local reproduction and CI parity.
- **`bench-artifacts/` boundary:** separates mutable baselines from immutable run history to prevent accidental data rewrite.
- **`schema/` contracts:** avoids silent format drift between collector/analyzer/publisher.
- **`workflows/` split:** keeps reusable benchmark execution logic distinct from publication/deployment concerns.

## Architectural Patterns

### Pattern 1: Canonical Metric Envelope

**What:** Always convert benchmark output into one stable internal schema before analysis/publication.
**When to use:** Immediately after benchmark execution, regardless of benchmark harness format.
**Trade-offs:** Extra transform step, but major reduction in downstream coupling.

**Example:**
```json
{
  "run_id": "2026-02-24T12:00:00Z-main-a1b2c3d",
  "commit": "a1b2c3d",
  "testbed": "gh-ubuntu-8core",
  "benchmarks": {
    "txn_apply": {
      "latency_ns": { "value": 12345.6, "lower": 12200.0, "upper": 12500.0 }
    }
  }
}
```

### Pattern 2: Two-Track Persistence (Immutable Runs + Mutable Baseline)

**What:** Store every run as immutable history, but maintain a separately curated baseline pointer per benchmark/testbed.
**When to use:** Any regression pipeline where baseline updates are policy-controlled (not automatic on every run).
**Trade-offs:** Slightly more storage and logic, but enables auditability and rollback.

### Pattern 3: Split Build and Deploy Jobs with Explicit `needs`

**What:** Build analysis/site artifact in one job, deploy in a separate job that depends on it.
**When to use:** GitHub Pages deployment and any CI pipeline requiring traceable artifacts.
**Trade-offs:** More YAML/jobs, but clear failure boundaries and cleaner retry semantics.

## Data Flow

### Request Flow

```text
[Trigger: local command | push | schedule | workflow_dispatch]
    ↓
[Runner] -> [Benchmark Execution] -> [Raw Output + Env Metadata]
    ↓
[Normalizer] -> [Canonical Run JSON]
    ↓
[Regression Analyzer] -> [Verdict + Delta Report]
    ↓
[Artifact Writer] -> [runs/<run-id>/...]
    ↓
[Site Builder] -> [site/data/*.json + index]
    ↓
[Pages Deploy] -> [Published static history]
```

### Key Data Flows

1. **Collection flow:** benchmark harness output and fingerprint data are merged into a canonical run manifest.
2. **Decision flow:** analyzer reads `baselines/<benchmark>/<testbed>.json` and emits warn/fail/pass with statistical deltas.
3. **Publication flow:** history builder reads immutable `runs/` snapshots and writes compact series files for static Pages rendering.

## Build Order and Dependencies (Roadmap-Critical)

1. **Define schema + contracts first**
   - Dependency: none.
   - Output: canonical metric schema, run manifest schema.
   - Why first: all later components depend on stable data interfaces.

2. **Implement deterministic runner + env fingerprinting**
   - Depends on: schema definitions.
   - Output: reproducible local/CI run command that emits raw data + metadata.
   - Notes: enforce pinned toolchain and `--locked` dependency resolution.

3. **Build normalizer (raw -> canonical)**
   - Depends on: runner output format + schema.
   - Output: canonical run JSON and validation checks.

4. **Build baseline registry + update policy**
   - Depends on: canonical run JSON.
   - Output: baseline storage layout and promotion rules (manual/controlled auto-promotion).

5. **Integrate regression analyzer + verdict gate**
   - Depends on: canonical runs + baseline registry.
   - Output: CI pass/warn/fail signals and machine-readable delta report.

6. **Add CI workflow orchestration (collect -> analyze -> artifact)**
   - Depends on: steps 2-5.
   - Output: workflow with explicit job dependencies, artifacts, and status checks.

7. **Add static history site builder**
   - Depends on: immutable run storage and analyzer outputs.
   - Output: compact historical JSON + static pages content.

8. **Add Pages deployment workflow**
   - Depends on: site builder outputs.
   - Output: custom Pages workflow (`configure-pages` -> `upload-pages-artifact` -> `deploy-pages`).

9. **Harden for scale and governance**
   - Depends on: end-to-end pipeline.
   - Output: retention policy, chart downsampling, branch protections, baseline update approvals.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| GitHub Actions | Workflow jobs with `needs` and artifacts | Use artifacts for cross-job transfer and debug reproducibility |
| GitHub Pages | Custom Actions workflow deployment | Enforce `pages: write` + `id-token: write`, split build/deploy jobs |
| Optional external benchmark backend (Bencher, etc.) | CLI/API push from analyzer output | Useful later if richer alerting/comparison UX is needed |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Runner -> Normalizer | File contract (`raw/*.json`, `manifest.json`) | Keep strict schema validation gate |
| Normalizer -> Analyzer | Canonical run JSON | No analyzer access to raw harness output |
| Analyzer -> Publisher | Delta report + metric series snapshot | Publisher should not re-run stats logic |
| CI Jobs -> Pages Deploy | Uploaded artifact only | Avoid rebuilding at deploy stage |

## Anti-Patterns

### Anti-Pattern 1: Analyzer Coupled to Raw Harness Output

**What people do:** Parse benchmark tool output directly inside regression logic.
**Why it's wrong:** Any harness output drift breaks both ingestion and analysis.
**Do this instead:** Enforce adapter layer and analyze only canonical schema.

### Anti-Pattern 2: Using Ephemeral CI Cache as Source of Truth

**What people do:** Keep baseline/history only in cache artifacts.
**Why it's wrong:** Cache eviction or key changes silently destroy historical continuity.
**Do this instead:** Keep immutable run snapshots + published static history as durable truth.

### Anti-Pattern 3: Auto-Promoting Every Green Run to Baseline

**What people do:** Replace baseline on each successful run.
**Why it's wrong:** Baseline drifts with noise and hides slow regressions.
**Do this instead:** Separate candidate evaluation from explicit baseline promotion policy.

## Sources

- https://docs.github.com/en/actions/tutorials/store-and-share-data (artifacts, cross-job data, digest validation) — HIGH
- https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-jobs (job dependency with `needs`) — HIGH
- https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages (Pages build/deploy workflow requirements) — HIGH
- https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits (publication size/throughput constraints) — HIGH
- https://doc.rust-lang.org/cargo/commands/cargo-bench.html (`cargo bench`, `--locked`, deterministic CI usage notes) — HIGH
- https://doc.rust-lang.org/cargo/commands/cargo-build.html (`--locked`/`--frozen` deterministic dependency behavior) — HIGH
- https://rust-lang.github.io/rustup/overrides.html (pinned `rust-toolchain.toml` behavior) — HIGH
- https://bencher.dev/docs/reference/bencher-metric-format/ (canonical benchmark metric schema pattern) — MEDIUM
- https://bencher.dev/docs/explanation/thresholds/ (threshold model dimensions: branch/testbed/measure) — MEDIUM
- https://github.com/benchmark-action/github-action-benchmark (community pattern: CI benchmark + regression alerts + Pages history) — LOW

---
*Architecture research for: benchmark baseline and regression-analysis systems*
*Researched: 2026-02-24*
