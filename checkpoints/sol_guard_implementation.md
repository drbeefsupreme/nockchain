# SOL Guard Implementation Checklist

This checklist is the execution tracker for implementing `nockchain-bench sol guard`.
Each item must be checked with evidence before final completion.

## Checklist

- [x] G001 Create this tracked checklist file.
  Evidence: Added `checkpoints/sol_guard_implementation.md`.
- [x] G002 Add `scripts/verify_sol_guard_plan.sh` that fails if any checklist item is unchecked.
  Evidence: Added `scripts/verify_sol_guard_plan.sh`.
- [x] G003 Add a command entrypoint to run checklist verification + guard tests in one command.
  Evidence: Added `sol-guard-verify` target in `Makefile`.
- [x] G004 Define canonical metric names mapped from `combined_summary.tsv` columns.
  Evidence: Added `checkpoints/sol_guard_spec.md` canonical metric table.
- [x] G005 Define baseline key tuple + sparse fallback tuple.
  Evidence: Added key policy in `checkpoints/sol_guard_spec.md` and `BaselineKey` in `crates/nockchain-bench/src/speed_of_light/guard/model.rs`.
- [x] G006 Define baseline eligibility policy (window size and max age).
  Evidence: Added policy in `checkpoints/sol_guard_spec.md` and `BaselinePolicy` defaults in `crates/nockchain-bench/src/speed_of_light/guard/model.rs`.
- [x] G007 Define contract schema (`.toml`) for floors/ceilings/budgets/severity.
  Evidence: Added schema docs in `checkpoints/sol_guard_spec.md` and sample `crates/nockchain-bench/tests/fixtures/guard/contract.toml`.
- [x] G008 Define output report schema (`guard-report.json` and `guard-report.md`).
  Evidence: Added report schema docs in `checkpoints/sol_guard_spec.md` and typed report model in `crates/nockchain-bench/src/speed_of_light/guard/model.rs`.
- [x] G009 Define exit code contract for guard outcomes.
  Evidence: Added exit code constants in `crates/nockchain-bench/src/speed_of_light/guard/mod.rs`.
- [x] G010 Add `speed_of_light/guard` module skeleton files.
  Evidence: Added `crates/nockchain-bench/src/speed_of_light/guard/`.
- [x] G011 Export the `guard` module from `speed_of_light/mod.rs`.
  Evidence: Updated `crates/nockchain-bench/src/speed_of_light/mod.rs`.
- [x] G012 Add required dependencies to `crates/nockchain-bench/Cargo.toml`.
  Evidence: Added `toml` dependency to `crates/nockchain-bench/Cargo.toml`.
- [x] G013 Add guard fixtures under `crates/nockchain-bench/tests/fixtures/guard/`.
  Evidence: Added fixture files under `crates/nockchain-bench/tests/fixtures/guard/`.
- [x] G014 Implement TSV ingestion for `combined_summary.tsv`.
  Evidence: Added `parse_combined_summary_tsv` in `crates/nockchain-bench/src/speed_of_light/guard/ingest.rs`.
- [x] G015 Implement manifest/run history ingestion for baselines.
  Evidence: Added `parse_runs_manifest` in `crates/nockchain-bench/src/speed_of_light/guard/ingest.rs`.
- [x] G016 Implement artifact resolver for per-row files.
  Evidence: Added `resolve_row_artifacts` in `crates/nockchain-bench/src/speed_of_light/guard/ingest.rs`.
- [x] G017 Implement folded-stack parser and symbol aggregation.
  Evidence: Added `parse_folded_symbol_totals` and symbol shift support in `crates/nockchain-bench/src/speed_of_light/guard/autopsy.rs`.
- [x] G018 Implement memory/checkpoint profile field parsing.
  Evidence: Added `parse_profile_metrics` and profile metric types in `crates/nockchain-bench/src/speed_of_light/guard/ingest.rs`.
- [x] G019 Add ingestion parser unit tests for malformed/missing data.
  Evidence: Added ingest tests in `crates/nockchain-bench/src/speed_of_light/guard/ingest.rs`.
- [x] G020 Implement baseline row selection logic.
  Evidence: Added `select_baseline_rows` in `crates/nockchain-bench/src/speed_of_light/guard/baseline.rs`.
- [x] G021 Implement robust center/spread stats (median and MAD).
  Evidence: Added `median` and `mad` in `crates/nockchain-bench/src/speed_of_light/guard/stats.rs`.
- [x] G022 Implement deterministic bootstrap confidence intervals.
  Evidence: Added `bootstrap_median_ci` in `crates/nockchain-bench/src/speed_of_light/guard/stats.rs`.
- [x] G023 Implement small-sample fallback behavior and reason codes.
  Evidence: Added insufficient-baseline handling and reason text in `crates/nockchain-bench/src/speed_of_light/guard/contract.rs`.
- [x] G024 Add stats unit tests for outliers, ties, and tiny samples.
  Evidence: Added stats tests in `crates/nockchain-bench/src/speed_of_light/guard/stats.rs`.
- [x] G025 Implement contract evaluator for throughput floors and latency ceilings.
  Evidence: Added rule evaluation in `crates/nockchain-bench/src/speed_of_light/guard/contract.rs`.
- [x] G026 Implement evaluator for memory/fault/checkpoint budgets.
  Evidence: Added absolute/relative ceiling checks for memory/fault metrics in `crates/nockchain-bench/src/speed_of_light/guard/contract.rs`.
- [x] G027 Implement weighted verdict aggregation and rule traces.
  Evidence: Added weighted fail/warn rollup and per-rule reason traces in `crates/nockchain-bench/src/speed_of_light/guard/contract.rs`.
- [x] G028 Add contract evaluator unit tests.
  Evidence: Added contract tests in `crates/nockchain-bench/src/speed_of_light/guard/contract.rs`.
- [x] G029 Implement autopsy ranking by normalized deviation.
  Evidence: Added `rank_metric_failures` in `crates/nockchain-bench/src/speed_of_light/guard/autopsy.rs`.
- [x] G030 Implement stack-shift detector from folded stacks.
  Evidence: Added `detect_stack_shifts` in `crates/nockchain-bench/src/speed_of_light/guard/autopsy.rs`.
- [x] G031 Implement memory/checkpoint anomaly detectors.
  Evidence: Added `detect_profile_anomalies` in `crates/nockchain-bench/src/speed_of_light/guard/autopsy.rs`.
- [x] G032 Implement autopsy hint rendering output.
  Evidence: Added `build_basic_hints` in `crates/nockchain-bench/src/speed_of_light/guard/autopsy.rs`.
- [x] G033 Add `SolCommands::Guard` CLI variant with arguments.
  Evidence: Added `Guard` variant to `SolCommands` in `crates/nockchain-bench/src/main.rs`.
- [x] G034 Wire guard command dispatch to `cmd_sol_guard`.
  Evidence: Added guard dispatch in `crates/nockchain-bench/src/main.rs`.
- [x] G035 Implement JSON and Markdown report writers.
  Evidence: Added `write_json`, `write_markdown`, and renderer in `crates/nockchain-bench/src/speed_of_light/guard/report.rs`.
- [x] G036 Implement terminal summary and deterministic exit behavior.
  Evidence: Added `cmd_sol_guard` summary output and `CliExit` code mapping in `crates/nockchain-bench/src/main.rs`.
- [x] G037 Add CLI integration tests for pass/fail/insufficient-baseline exit codes.
  Evidence: Added `crates/nockchain-bench/tests/sol_guard_cli.rs`.
- [x] G038 Add `scripts/sol_guard_ci.sh` for CI guard execution.
  Evidence: Added executable `scripts/sol_guard_ci.sh`.
- [x] G039 Add optional post-run guard hook to `scripts/sol_bench_matrix_trace.sh`.
  Evidence: Added optional hook controlled by `SOL_GUARD_POST_RUN` in `scripts/sol_bench_matrix_trace.sh`.
- [x] G040 Add optional guard verdict section to `scripts/publish_sol_trace_run.py`.
  Evidence: Added guard record ingestion + HTML/Markdown guard sections in `scripts/publish_sol_trace_run.py`.
- [x] G041 Update `README.md` with guard usage and CI examples.
  Evidence: Added “SOL Performance Guard (nockchain-bench)” section in `README.md`.
- [x] G042 Run full validation for guard implementation and tests.
  Evidence: Ran `cargo check -p nockchain-bench`, `cargo test -p nockchain-bench guard`, `cargo test -p nockchain-bench --test sol_guard_cli`, `python3 -m py_compile scripts/publish_sol_trace_run.py`, and CLI smoke checks for exit codes 2/3.
- [x] G043 Mark all steps complete with final evidence.
  Evidence: Verified all `G001..G045` checkpoints are now marked complete with linked implementation/test artifacts.
- [x] G044 Final determinism/error/backward-compatibility audit.
  Evidence: Verified deterministic bootstrap seed use, explicit guard exit-code mapping, and non-invasive optional hooks (`SOL_GUARD_POST_RUN`) without changing existing default matrix behavior.
- [x] G045 Ship.
  Evidence: Feature is implemented, validated, and staged for commit on this branch.
