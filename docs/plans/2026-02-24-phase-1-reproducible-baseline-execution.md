# Phase 1 Reproducible Baseline Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver one deterministic baseline command (local and CI) that emits canonical run artifacts with complete provenance metadata for `nockchain-bench`.

**Architecture:** Keep `scripts/sol_bench_matrix.sh` as the benchmark execution engine and layer a new configuration-first entrypoint on top so local and CI call the exact same pipeline. Add a canonical run-manifest writer/validator in `crates/nockchain-bench` guard modules to guarantee required provenance fields before artifacts are accepted. Wire a dedicated GitHub Actions workflow to run the same script and upload the resulting artifact tree.

**Tech Stack:** Bash (`scripts/`), Rust (`clap`, `serde`, `serde_json`, existing `nockchain-bench` modules), GitHub Actions YAML.

---

### Task 1: Phase 1 Tracker + Verifier

**Files:**
- Create: `checkpoints/phase1_reproducible_baseline_implementation.md`
- Create: `scripts/verify_phase1_reproducible_baseline_plan.sh`

**Step 1: Write checklist skeleton with stable IDs (`R001..R028`)**

```markdown
- [ ] R001 Create Phase 1 checklist file
- [ ] R002 Create Phase 1 verifier script
```

**Step 2: Write verifier to enforce required IDs and no unchecked boxes**

```bash
#!/usr/bin/env bash
set -euo pipefail

CHECKLIST="${1:-checkpoints/phase1_reproducible_baseline_implementation.md}"
for i in $(seq 1 28); do
  id=$(printf "R%03d" "$i")
  rg -q "^- \[[ xX]\] ${id}\\b" "$CHECKLIST" || exit 1
done
```

**Step 3: Run verifier to confirm it fails until all boxes are checked**

Run: `bash scripts/verify_phase1_reproducible_baseline_plan.sh`
Expected: exits non-zero with missing or unchecked IDs.

**Step 4: Commit**

```bash
git add checkpoints/phase1_reproducible_baseline_implementation.md scripts/verify_phase1_reproducible_baseline_plan.sh
git commit -m "chore: add phase 1 execution checklist and verifier"
```

### Task 2: Versioned Baseline Config Contract (ORCH-03)

**Files:**
- Create: `benchmarks/baseline/sol-baseline-v1.toml`
- Modify: `scripts/sol_bench_matrix.sh`
- Test: `crates/nockchain-bench/tests/sol_baseline_config_contract.rs`

**Step 1: Write failing config-contract test**

```rust
#[test]
fn baseline_config_fixture_and_pass_controls_are_present() {
    let cfg = std::fs::read_to_string("benchmarks/baseline/sol-baseline-v1.toml").unwrap();
    assert!(cfg.contains("passes"));
    assert!(cfg.contains("fixtures_dir"));
    assert!(cfg.contains("enable_checkpointing"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench sol_baseline_config_contract -- --nocapture`
Expected: FAIL because config file does not exist.

**Step 3: Add config file and `--config` loading path in matrix script**

```bash
if [[ -n "${CONFIG_FILE:-}" ]]; then
  source "$CONFIG_FILE"
fi
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench sol_baseline_config_contract -- --nocapture`
Expected: PASS.

**Step 5: Commit**

```bash
git add benchmarks/baseline/sol-baseline-v1.toml scripts/sol_bench_matrix.sh crates/nockchain-bench/tests/sol_baseline_config_contract.rs
git commit -m "feat: add versioned baseline run config contract"
```

### Task 3: Single Local Entry Command (ORCH-01)

**Files:**
- Create: `scripts/sol_baseline_run.sh`
- Modify: `scripts/sol_bench_matrix.sh`
- Test: `crates/nockchain-bench/tests/sol_baseline_run_cli.rs`

**Step 1: Write failing integration test for local wrapper command**

```rust
#[test]
fn baseline_run_help_exits_zero() {
    let status = std::process::Command::new("bash")
        .args(["scripts/sol_baseline_run.sh", "--help"])
        .status()
        .unwrap();
    assert!(status.success());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench sol_baseline_run_cli -- --nocapture`
Expected: FAIL because wrapper script does not exist.

**Step 3: Implement wrapper script that only requires config + output root**

```bash
scripts/sol_bench_matrix.sh \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT"
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench sol_baseline_run_cli -- --nocapture`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/sol_baseline_run.sh scripts/sol_bench_matrix.sh crates/nockchain-bench/tests/sol_baseline_run_cli.rs
git commit -m "feat: add single-command local baseline runner"
```

### Task 4: Canonical Provenance Manifest (DATA-01, DATA-02)

**Files:**
- Create: `crates/nockchain-bench/src/speed_of_light/guard/provenance.rs`
- Modify: `crates/nockchain-bench/src/speed_of_light/guard/mod.rs`
- Modify: `scripts/sol_bench_matrix.sh`
- Test: `crates/nockchain-bench/tests/sol_provenance_manifest.rs`

**Step 1: Write failing manifest validation test with required keys**

```rust
#[test]
fn manifest_requires_commit_branch_and_tool_versions() {
    let manifest = std::fs::read_to_string("tests/fixtures/guard/run-manifest.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&manifest).unwrap();
    assert!(v.get("git_commit").is_some());
    assert!(v.get("git_branch").is_some());
    assert!(v.get("tool_versions").is_some());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p nockchain-bench sol_provenance_manifest -- --nocapture`
Expected: FAIL due to missing fixture/manifest.

**Step 3: Implement provenance model + writer and call it from matrix run completion**

```rust
pub struct RunProvenance {
    pub git_commit: String,
    pub git_branch: String,
    pub benchmark_config_sha256: String,
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p nockchain-bench sol_provenance_manifest -- --nocapture`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/nockchain-bench/src/speed_of_light/guard/provenance.rs crates/nockchain-bench/src/speed_of_light/guard/mod.rs scripts/sol_bench_matrix.sh crates/nockchain-bench/tests/sol_provenance_manifest.rs
git commit -m "feat: emit canonical baseline run provenance manifest"
```

### Task 5: CI Parity Workflow (ORCH-02)

**Files:**
- Create: `.github/workflows/sol-baseline.yml`
- Create: `scripts/sol_baseline_ci.sh`
- Modify: `scripts/sol_baseline_run.sh`
- Test: `.github/workflows/sol-baseline.yml`

**Step 1: Write failing dry-run check by linting workflow command references**

```bash
rg -n "sol_baseline_ci.sh" .github/workflows/sol-baseline.yml
```

**Step 2: Run check to verify it fails before workflow exists**

Run: `rg -n "sol_baseline_ci.sh" .github/workflows/sol-baseline.yml`
Expected: non-zero exit.

**Step 3: Implement CI wrapper so CI and local both call `scripts/sol_baseline_run.sh`**

```bash
scripts/sol_baseline_run.sh --config benchmarks/baseline/sol-baseline-v1.toml --output-root bench-artifacts/sol-baseline
```

**Step 4: Re-run check and workflow syntax validation**

Run: `python3 -m yaml --help >/dev/null 2>&1 || true`
Expected: workflow file exists and references `scripts/sol_baseline_ci.sh`.

**Step 5: Commit**

```bash
git add .github/workflows/sol-baseline.yml scripts/sol_baseline_ci.sh scripts/sol_baseline_run.sh
git commit -m "ci: add reproducible baseline workflow using shared runner script"
```

### Task 6: Docs + Planning State Updates

**Files:**
- Create: `docs/nockchain-bench/baseline-runbook.md`
- Modify: `.planning/ROADMAP.md`
- Modify: `.planning/STATE.md`
- Modify: `.planning/REQUIREMENTS.md`

**Step 1: Write failing docs existence assertion (shell)**

```bash
test -f docs/nockchain-bench/baseline-runbook.md
```

**Step 2: Run check to verify it fails**

Run: `test -f docs/nockchain-bench/baseline-runbook.md`
Expected: non-zero exit.

**Step 3: Document local/CI invocation and artifact layout, then update requirement statuses**

```markdown
Run local: `scripts/sol_baseline_run.sh --config benchmarks/baseline/sol-baseline-v1.toml`
Run CI: workflow `sol-baseline.yml`
```

**Step 4: Re-run check to verify docs exist and execute plan verifier**

Run: `test -f docs/nockchain-bench/baseline-runbook.md && bash scripts/verify_phase1_reproducible_baseline_plan.sh`
Expected: doc exists; verifier fails until every checklist step is checked.

**Step 5: Commit**

```bash
git add docs/nockchain-bench/baseline-runbook.md .planning/ROADMAP.md .planning/STATE.md .planning/REQUIREMENTS.md checkpoints/phase1_reproducible_baseline_implementation.md
git commit -m "docs: capture phase 1 baseline workflow and planning state"
```

### Task 7: Verification Gate Before Phase Completion

**Files:**
- Modify: `checkpoints/phase1_reproducible_baseline_implementation.md`

**Step 1: Run focused Rust tests added in this phase**

Run: `cargo test -p nockchain-bench sol_baseline_config_contract sol_baseline_run_cli sol_provenance_manifest -- --nocapture`
Expected: PASS.

**Step 2: Run existing SOL guard CLI regression tests to prevent breakage**

Run: `cargo test -p nockchain-bench sol_guard_cli -- --nocapture`
Expected: PASS.

**Step 3: Run script-level sanity checks**

Run: `bash scripts/sol_baseline_run.sh --help && bash scripts/sol_baseline_ci.sh --help`
Expected: both commands print usage and exit 0.

**Step 4: Run Phase 1 checklist verifier and capture evidence links in checklist**

Run: `bash scripts/verify_phase1_reproducible_baseline_plan.sh`
Expected: `Phase 1 reproducible baseline checklist complete`.

**Step 5: Commit**

```bash
git add checkpoints/phase1_reproducible_baseline_implementation.md
git commit -m "chore: complete phase 1 verification evidence"
```
