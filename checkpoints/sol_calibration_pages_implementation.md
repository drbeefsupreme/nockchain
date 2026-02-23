# SOL Calibration + Pages Split Implementation Checklist

This checklist tracks replacement of heuristic confidence with calibrated evaluation
metrics and the SOL pages split into calibration-era index + legacy archive.

## Checklist

- [x] P001 Create this tracked checklist file.
  Evidence: Added `checkpoints/sol_calibration_pages_implementation.md`.
- [x] P002 Add a verifier script for checklist completeness.
  Evidence: Added `scripts/verify_sol_calibration_pages_plan.sh`.
- [x] P003 Add probability/stat helpers in publisher (`normal_cdf`, tail prob, clamp).
  Evidence: Added `clamp(...)`, `clamp_prob(...)`, `normal_cdf(...)`, and `two_sided_tail_prob_from_z(...)` in `scripts/publish_sol_trace_run.py`.
- [x] P004 Add Brier score computation helper.
  Evidence: Added `brier_score(...)` in `scripts/publish_sol_trace_run.py`.
- [x] P005 Add reliability-bin builder for calibration curves.
  Evidence: Added `build_reliability_bins(...)` in `scripts/publish_sol_trace_run.py`.
- [x] P006 Add ECE computation from reliability bins.
  Evidence: Added `expected_calibration_error(...)` in `scripts/publish_sol_trace_run.py`.
- [x] P007 Add isotonic calibration fit (PAV) implementation.
  Evidence: Added `fit_isotonic_pav(...)` in `scripts/publish_sol_trace_run.py`.
- [x] P008 Add isotonic apply helper for new probabilities.
  Evidence: Added `apply_isotonic_pav(...)` in `scripts/publish_sol_trace_run.py`.
- [x] P009 Add raw change probability derived from robust throughput z-score.
  Evidence: Added `raw_change_probability_from_eval(...)` and wired into causal record generation.
- [x] P010 Replace heuristic confidence assignment with calibrated confidence assignment.
  Evidence: Removed weighted heuristic path and now derive class confidence from calibrated/non-calibrated `p(change)`.
- [x] P011 Preserve backward-compatible `confidence` while adding explicit probability fields.
  Evidence: Causal records now include `confidence`, `raw_change_probability`, `calibrated_change_probability`, `confidence_model`, and `calibration_status`.
- [x] P012 Add historical calibration snapshot loader from existing run artifacts.
  Evidence: Added `load_causal_records_for_run(...)` and calibration-era run ingestion in `main()`.
- [x] P013 Add future-outcome label resolver without reruns.
  Evidence: Added `resolve_prediction_labels(...)` with fixed future window and status fields.
- [x] P014 Build resolved calibration sample set and unresolved sample set.
  Evidence: Prediction rows now carry `label`, `label_status`, `label_resolved_at_run_id`, and pending/resolved handling.
- [x] P015 Compute global calibration metrics (Brier, ECE, reliability bins).
  Evidence: Added `calibration_metric_summary(...)` and global raw/calibrated metrics in `calibration-feed.json`.
- [x] P016 Compute per-run calibration metrics rollup.
  Evidence: Added per-run summaries in `apply_online_isotonic_calibration(...)` and emitted to feed/run artifacts.
- [x] P017 Emit per-run `calibration-eval.json` artifact.
  Evidence: Publisher now writes `runs/<run_id>/calibration-eval.json`.
- [x] P018 Emit top-level `sol-runs/calibration-feed.json` artifact.
  Evidence: Publisher now writes `docs/nockchain-bench/sol-runs/calibration-feed.json`.
- [x] P019 Emit top-level `sol-runs/calibration-feed.tsv` artifact.
  Evidence: Added `write_calibration_feed_tsv(...)` and write in `main()`.
- [x] P020 Add manifest field `calibration_start_run_id` and initialize on rollout publish.
  Evidence: `main()` now sets/persists `calibration_start_run_id` in `runs-manifest.json`.
- [x] P021 Split run lists into calibration-era runs and legacy archive runs.
  Evidence: Added `calibration_runs`/`archive_runs` partitioning in `main()`.
- [x] P022 Render new main calibration index page at `sol-runs/index.html`.
  Evidence: Added `render_sol_runs_calibration_index(...)` and write step in `main()`.
- [x] P023 Render legacy archive page at `sol-runs/archive.html`.
  Evidence: Added `render_sol_runs_archive_index(...)` and write step in `main()`.
- [x] P024 Add reliability-curve visual element in main index.
  Evidence: `sol-runs/index.html` now renders canvas chart `#reliabilityChart`.
- [x] P025 Add tuple explorer visuals (fixture history and branch snapshot) in main index.
  Evidence: `sol-runs/index.html` now includes mode switch + trend chart `#trendChart` with both views.
- [x] P026 Add in-depth explainer section for confidence/Brier/ECE/reliability semantics.
  Evidence: Added `Confidence Model Explainer` section in `render_sol_runs_calibration_index(...)`.
- [x] P027 Add LLM-consumable data feed links and schema hints in main index.
  Evidence: Added links to `calibration-feed.json/tsv` and embedded schema snippet panel.
- [x] P028 Update run report causal section text for calibrated confidence semantics.
  Evidence: Updated causal header/notes and markdown columns in `render_html(...)` and `render_md(...)`.
- [x] P029 Run syntax and smoke publish verification commands successfully.
  Evidence: Ran `python3 -m py_compile scripts/publish_sol_trace_run.py` and publish smoke run against `/shared/nockchain-ext4-bench/artifacts/runs/20260223_112036-sol-guard-refresh`.
- [x] P030 Run checklist verifier to completion.
  Evidence: Ran `bash scripts/verify_sol_calibration_pages_plan.sh` after completing this checklist.
