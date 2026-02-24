# Stack Research

**Domain:** Reproducible benchmark baseline generation and longitudinal performance analysis in a Rust monorepo
**Researched:** 2026-02-24
**Confidence:** MEDIUM-HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Rust toolchain pinning (`rust-toolchain.toml`) | stable channel pinned per repo (e.g. `1.89.x`) | Reproducible compiler + stdlib behavior across local/CI baseline runs | Rustup officially supports repo-pinned toolchains; pinning removes silent drift from compiler changes, which otherwise contaminates longitudinal benchmark trends. |
| `criterion` | `0.8.x` | Statistical microbenchmark execution and regression comparison | `criterion` remains the standard Rust benchmarking crate and explicitly provides statistically grounded change detection and rich outputs suitable for machine ingestion. |
| `iai-callgrind` | `0.16.x` | Deterministic instruction/cache-event benchmark lane for CI noise control | `iai-callgrind` is purpose-built for high consistency in virtualized CI and reports run-to-run deltas, making it the practical companion to wall-time benchmarking. |
| GitHub Actions + GitHub Pages custom workflow (`actions/configure-pages@v5`, `actions/upload-pages-artifact@v4`, `actions/deploy-pages@v4`) | current major tags | CI orchestration + static publication of historical reports | This is GitHub’s official and current Pages deployment path; it supports split build/deploy jobs and is stable for static historical benchmark hosting. |
| `benchmark-action/github-action-benchmark` | `v1` (latest `1.20.7` as of 2025-09) | Turn benchmark JSON into longitudinal charts + regression alerts | Mature OSS action (3k+ dependents) with built-in history updates and charting on Pages, reducing custom dashboard code for a greenfield benchmark infra. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `serde` + `serde_json` | `1.x` | Canonical, versioned benchmark result schema serialization | Always; make benchmark outputs schema-first so every run is diffable and publishable without parser rewrites. |
| `chrono` | `0.4.x` | UTC timestamping for run metadata and trend windows | Always; include ISO8601 timestamps for slicing historical windows and auditability. |
| `sysinfo` | `0.3x` | Capture machine metadata (CPU model, memory, core counts) into run manifests | Use in every persisted baseline run; environment metadata is required to attribute regressions correctly. |
| `flate2` + `zstd` (optional) | `1.x` / `0.13.x` | Compress historical artifacts for Pages size and transfer efficiency | Use once data volume grows; keep both raw JSON and compressed archives for easy debugging plus efficient storage. |
| `jsonschema` (optional validation step) | `0.2x` | Enforce benchmark schema compatibility in CI | Use when multiple tools emit benchmark output and you need strict contract checks before publish/deploy. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `dtolnay/rust-toolchain` action | Deterministic Rust install in CI | Pin by full commit SHA for supply-chain hygiene; aligns CI with `rust-toolchain.toml`. |
| `Swatinem/rust-cache@v2` | Fast Rust dependency/build caching without fragile custom cache keys | Good defaults for Cargo projects; use `save-if` to cache only trusted branches. |
| Self-hosted GitHub runner pool (Linux) | Stable benchmark hardware profile | GitHub docs confirm self-hosted gives hardware control; use this for benchmark jobs, keep generic CI on hosted runners. |
| `nextest` | Fast, stable non-benchmark test lane | Keep correctness tests separate from perf lanes; avoids benchmark wall-time budget pressure. |

## Installation

```bash
# Benchmark/statistics core
cargo add --dev criterion iai-callgrind

# Result schema + metadata capture
cargo add serde serde_json chrono sysinfo

# Optional data-volume hardening
cargo add flate2 zstd

# Optional schema contract enforcement
cargo add --dev jsonschema

# CI runner prerequisite for iai-callgrind lane (Linux)
sudo apt-get update && sudo apt-get install -y valgrind
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `criterion` + `iai-callgrind` dual-lane | `criterion` only | Acceptable for early prototypes if you do not gate on CI reproducibility yet. Not ideal for long-lived regression detection in noisy shared runners. |
| GitHub Pages custom workflow + official Pages actions | Direct `gh-pages` branch pushes only | Use only for ultra-simple repos. Custom workflow is cleaner for protected environments and explicit deploy permissions in 2025+ Actions best practices. |
| `github-action-benchmark` for trend UI | Build bespoke dashboard immediately | Use custom UI only if you need domain-specific visuals early; otherwise it is avoidable maintenance during baseline bootstrapping. |
| Self-hosted benchmark runners | GitHub-hosted `ubuntu-latest` only | Use hosted-only if team cannot maintain hardware. Expect higher variance and looser alert thresholds. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `cargo-criterion` as primary pipeline backbone | Latest release is from 2021; ecosystem and dependencies are stale for a new 2025/2026 baseline platform. | Run `criterion` directly and emit structured JSON artifacts in your own benchmark runner crate. |
| Benchmark gating solely on GitHub-hosted shared VMs | Runner image and host variability increase noise; GitHub docs and benchmark-action caveats both indicate environment stability matters for perf signal quality. | Dedicated self-hosted runners for benchmark workflows; keep hosted runners for regular CI checks. |
| Storing longitudinal history only in cache/artifacts | Caches are ephemeral/evictable and not a durable historical source of truth. | Persist canonical benchmark history in Pages-published data files (or dedicated data repo). |
| Building dashboard-first before schema and metadata contracts | UI-first pipelines often force later rewrites when schema evolves. | Lock JSON schema + metadata contract first, then render with `github-action-benchmark` or a thin static viewer. |

## Stack Patterns by Variant

**If you need fastest time-to-value (greenfield MVP):**
- Use `criterion` + `github-action-benchmark` + GitHub Pages
- Because this gives baseline generation, historical charts, and alerting with minimal custom code

**If you need strict CI reproducibility for PR gating:**
- Add `iai-callgrind` lane on dedicated self-hosted Linux runners
- Because instruction/cache-event signals are more stable than wall-clock metrics on shared infrastructure

**If you already have a custom stats engine in-repo:**
- Keep the engine; add an adapter that emits `customSmallerIsBetter` JSON for `github-action-benchmark`
- Because you preserve domain logic while standardizing historical publication and alert workflows

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `criterion@0.8.x` | Rust stable toolchains; `cargo bench` profile | Prefer pinned stable toolchain in `rust-toolchain.toml` to avoid trend drift from compiler upgrades. |
| `iai-callgrind@0.16.x` | Linux runners with Valgrind installed | Best suited for Linux CI perf lanes; treat as deterministic signal lane, not user-facing latency proxy. |
| `actions/upload-pages-artifact@v4` | `actions/deploy-pages@v4` | Official GitHub Pages workflow pair; requires proper `pages: write` + `id-token: write` permissions. |
| `benchmark-action/github-action-benchmark@v1` | GitHub Pages publication (`gh-pages`) | Supports Rust `cargo bench` and custom JSON data; pin exact minor in production workflows. |

## Sources

- https://rust-lang.github.io/rustup/overrides.html - Rust toolchain pinning and override precedence (official)
- https://doc.rust-lang.org/cargo/reference/profiles.html - Benchmark/release profile behavior and tuning knobs (official)
- https://docs.rs/criterion/latest/criterion/ - Current criterion crate version/capabilities (`0.8.2`)
- https://docs.rs/iai-callgrind/latest/iai_callgrind/ - Deterministic CI-oriented benchmark framework details (`0.16.1`)
- https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages - Official Pages workflow pattern and required permissions
- https://github.com/actions/upload-pages-artifact - Current Pages artifact action major (`v4`, Aug 2025 release)
- https://github.com/benchmark-action/github-action-benchmark - Continuous benchmark action, Pages charting, regression alerts (`v1.20.7`, Sep 2025)
- https://docs.github.com/en/actions/reference/runners/github-hosted-runners - Hosted runner characteristics and `-latest` caveats
- https://docs.github.com/en/actions/concepts/runners/self-hosted-runners - Self-hosted runner control model
- https://docs.rs/cargo-criterion/latest/cargo_criterion/ - Staleness check (latest `1.1.0`, 2021-07)
- https://github.com/Swatinem/rust-cache - Rust cache action usage and current release cadence (`v2.8.2`, Nov 2025)
- https://github.com/dtolnay/rust-toolchain - Common Rust toolchain setup action and pinning guidance

---
*Stack research for: benchmark baseline/statistical performance analysis infrastructure in a Rust monorepo*
*Researched: 2026-02-24*
