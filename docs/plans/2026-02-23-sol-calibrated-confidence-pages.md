# SOL Calibrated Confidence + Pages Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace heuristic confidence with calibrated confidence diagnostics (Brier, reliability, ECE), split legacy runs into an archive, and make the main SOL pages index focused on new calibration-era runs with machine-readable data feeds.

**Architecture:** Keep `scripts/publish_sol_trace_run.py` as the single publisher entrypoint. Add a calibration subsystem that derives tuple-level raw probabilities from robust z-scores, resolves labels from subsequent runs, applies isotonic calibration, computes Brier/reliability/ECE, and emits both human-facing HTML and LLM-oriented JSON/TSV feeds. Extend manifest metadata with a `calibration_start_run_id` pivot so historical runs are preserved as archive without retroactive calibration.

**Tech Stack:** Python 3 stdlib (`json`, `math`, `csv`, `pathlib`), static HTML/CSS/vanilla JS, existing GH Pages artifact layout.

---

### Task 1: Planning + Execution Tracker

**Files:**
- Create: `checkpoints/sol_calibration_pages_implementation.md`
- Create: `scripts/verify_sol_calibration_pages_plan.sh`

**Step 1: Create checklist with stable IDs (`P001..P030`) and concrete evidence targets**

```markdown
- [ ] P001 ...
- [ ] P002 ...
```

**Step 2: Add verifier script that checks missing IDs and unchecked boxes**

Run: `bash scripts/verify_sol_calibration_pages_plan.sh`
Expected: fails until all boxes are checked.

**Step 3: Mark P001/P002 with evidence once files exist**

Run: `bash scripts/verify_sol_calibration_pages_plan.sh`
Expected: still fails on remaining unchecked steps.

### Task 2: Calibrated Confidence Core

**Files:**
- Modify: `scripts/publish_sol_trace_run.py`

**Step 1: Add probability/stat helpers**
- `normal_cdf`, two-sided tail from z-score, clamp helpers.
- Brier score, reliability bins, ECE calculations.

**Step 2: Add isotonic calibration implementation (PAV)**
- Fit monotonic map from `(p_raw, y)` pairs.
- Apply map to future probabilities.
- Include empty-data and tiny-data fallbacks.

**Step 3: Replace heuristic confidence pipeline**
- Preserve classification labels.
- Compute `raw_change_probability` from robust throughput z-score.
- Compute class confidence as calibrated probability of predicted class.

**Step 4: Keep backward-compatible fields while adding new fields**
- Keep `confidence` and add metadata: `confidence_model`, `raw_change_probability`, `calibrated_change_probability`, `calibration_status`.

### Task 3: Labeling + Calibration Dataset Assembly

**Files:**
- Modify: `scripts/publish_sol_trace_run.py`

**Step 1: Build run history records for calibration era from `runs-manifest.json` + run causal JSON files**

**Step 2: Define outcome labels using future runs only (no reruns)**
- For each tuple prediction, label when enough subsequent runs exist.
- Keep unresolved tuples with `label_status: pending`.

**Step 3: Compute calibration metrics**
- Global and per-run resolved counts.
- Brier score, ECE, reliability bins.

**Step 4: Attach calibration summary back to current run metadata**

### Task 4: LLM-Ready Artifact Feeds

**Files:**
- Modify: `scripts/publish_sol_trace_run.py`

**Step 1: Emit calibration feed JSON**
- `sol-runs/calibration-feed.json` with model spec, run summaries, tuple rows.

**Step 2: Emit calibration feed TSV**
- `sol-runs/calibration-feed.tsv` for easy ingestion by scripts/LLMs.

**Step 3: Add per-run calibration artifact link**
- `runs/<run_id>/calibration-eval.json`.

### Task 5: Pages Split (Main vs Archive)

**Files:**
- Modify: `scripts/publish_sol_trace_run.py`

**Step 1: Add manifest pivot**
- Introduce `calibration_start_run_id` on first publish after feature rollout.

**Step 2: Render main index (`sol-runs/index.html`) for calibration-era runs only**
- Cards for Brier/ECE/resolved labels.
- Reliability curve graphic.
- Tuple explorer mode A: fixture across runs.
- Tuple explorer mode B: branch across fixtures.

**Step 3: Render archive page (`sol-runs/archive.html`) for legacy runs**
- Preserve access to all previous reports and links.

**Step 4: Add cross-links and machine data links in both pages**

### Task 6: Report Page Explainer + Field Updates

**Files:**
- Modify: `scripts/publish_sol_trace_run.py`

**Step 1: Update causal panel notes and labels to calibrated language**
- Explicitly define confidence semantics.

**Step 2: Include metrics summary and feed links in generated markdown/html report artifacts**

### Task 7: Verification + Smoke Publish

**Files:**
- Modify: `checkpoints/sol_calibration_pages_implementation.md`

**Step 1: Python syntax check**
Run: `python3 -m py_compile scripts/publish_sol_trace_run.py`
Expected: no output, exit 0.

**Step 2: Run publish script on existing run root (no benchmark rerun)**
Run: `python3 scripts/publish_sol_trace_run.py --run-root /shared/nockchain-ext4-bench/artifacts/runs/20260223_112036-sol-guard-refresh --pages-root /shared/Dropbox/zorp/agents/nockchain-bench-opt --title "..." --scope "..."`
Expected: publish completes and rewrites manifests/pages.

**Step 3: Validate generated artifacts**
- `jq` checks for calibration metrics fields.
- `rg` checks for explainer and chart elements in generated HTML.

**Step 4: Complete checklist evidence and run verifier**
Run: `bash scripts/verify_sol_calibration_pages_plan.sh`
Expected: `SOL calibration/pages checklist complete`.
