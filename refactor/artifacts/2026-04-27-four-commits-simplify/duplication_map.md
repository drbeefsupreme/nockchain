# Duplication Map — 2026-04-27-four-commits-simplify

Generated: 2026-04-27 19:36 UTC
Tools run: (none installed)
Raw outputs: refactor/artifacts/2026-04-27-four-commits-simplify/scans/

## How to fill this in

1. Read the scan outputs above.
2. Cluster similar findings into candidates (assign IDs D1, D2, …).
3. For each candidate, fill the table row below.
4. Pass to score_candidates.py.

| ID  | Kind | Locations | LOC each | × | Type | Notes |
|-----|------|-----------|----------|---|------|-------|
| D1  | repeated `StepResult` field copy | `crates/nockchain-bench/src/speed_of_light/orchestrator.rs`: `finalize_force_cold_step`, `finalize_cold_peek_step`, `execute_peek_step`, cold-peek error branch | 2-7 | 5 | II | Same measurement and cold-result fields copied into `StepResult`; extract builder helpers and remove now-unused `PeekMeasurement` accessors. |
