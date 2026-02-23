# SOL Causal Attribution Implementation Checklist

This checklist tracks implementation of the "Why Did It Change?" causal attribution
panel for published SOL benchmark reports.

## Checklist

- [x] C001 Create this tracked checklist file.
  Evidence: Added `checkpoints/sol_causal_attribution_implementation.md`.
- [x] C002 Add a verifier script that fails if checklist steps are missing or unchecked.
  Evidence: Added `scripts/verify_sol_causal_plan.sh`.
- [x] C003 Add historical run ingestion from `sol-runs/runs-manifest.json` + archived summaries.
  Evidence: Added `load_history_samples(...)` in `scripts/publish_sol_trace_run.py`.
- [x] C004 Normalize row schemas across older/newer `combined_summary.tsv` formats.
  Evidence: Added `normalize_summary_row(...)` with `runtime->env`, `current->master`, and `v*_firstN->v*` handling.
- [x] C005 Build baseline sample selection by `(env, branch, fixture)` for prior runs.
  Evidence: Added tuple-keyed history via `tuple_key(...)` and baseline lookup in `build_causal_records(...)`.
- [x] C006 Add robust baseline statistics (median, MAD, expected range, percent delta, z-score).
  Evidence: Added `median(...)`, `mad(...)`, and `eval_metric(...)`.
- [x] C007 Add causal classification per tuple (`regression`, `improvement`, `stable`, `insufficient_baseline`).
  Evidence: Added `classify_tuple(...)` and classification emission in causal records.
- [x] C008 Parse candidate perf leaf symbols and percentages from copied perf summaries.
  Evidence: Added `parse_perf_leaf_pcts(...)` and candidate usage in `build_causal_records(...)`.
- [x] C009 Parse/aggregate baseline perf symbols and compute symbol shift deltas.
  Evidence: Added `average_symbol_maps(...)` and `top_symbol_shifts(...)`.
- [x] C010 Add symbol-to-source-file hint resolver for Rust crates.
  Evidence: Added `build_rust_file_index(...)`, `symbol_tokens(...)`, and `resolve_symbol_files(...)`.
- [x] C011 Add optional recent-commit hint resolver for candidate source files.
  Evidence: Added `recent_commit_hint(...)` (git log lookup) and integrated cache in `build_causal_records(...)`.
- [x] C012 Emit `causal-attribution.json` artifact for each published run.
  Evidence: Added run artifact write in `main()` to `runs/<run_id>/causal-attribution.json`.
- [x] C013 Add interactive "Why Did It Change?" section to generated HTML report.
  Evidence: Added causal card + selectors + dynamic rendering JS in `render_html(...)`.
- [x] C014 Include causal signal summary in generated Markdown report.
  Evidence: Added `## Why Did It Change? (Causal Attribution)` table in `render_md(...)`.
- [x] C015 Add causal artifact links to report output.
  Evidence: Added `causal-attribution.json` link in HTML artifacts and markdown file list.
- [x] C016 Run checklist verifier and publish script successfully.
  Evidence: Re-ran `scripts/publish_sol_trace_run.py` against `20260223_112036-sol-guard-refresh` and executed `scripts/verify_sol_causal_plan.sh`.
- [x] C017 Validate generated HTML/JSON outputs include causal attribution data.
  Evidence: Verified generated `causal-attribution.json` has 18 records and HTML contains the causal section/JS bindings.
- [x] C018 Publish updated Pages artifacts and verify live deployment reflects new data.
  Evidence: Pushed `jon/millenium-falcon` commit `9b07d79`; Pages run `22315694924` succeeded and live URLs expose causal panel + JSON.
