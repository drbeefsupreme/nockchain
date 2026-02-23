# SOL Guard Specification (v0)

This document defines the initial performance contract and regression autopsy model for
`nockchain-bench sol guard`.

## Canonical Metrics (`G004`)

`sol guard` normalizes input rows from `combined_summary.tsv` into canonical metric keys:

| Canonical key | Source column | Direction |
|---|---|---|
| `throughput_blocks_s` | `throughput_blocks_s` | higher is better |
| `init_time_s` | `init_time_s` | lower is better |
| `total_poke_time_s` | `total_poke_time_s` | lower is better |
| `avg_per_block_ms` | `avg_per_block_ms` | lower is better |
| `peak_rss_mib` | `peak_rss_mib` | lower is better |
| `p95_rss_mib` | `p95_rss_mib` | lower is better |
| `minor_faults_delta` | `minor_faults_delta` | lower is better |
| `major_faults_delta` | `major_faults_delta` | lower is better |
| `checkpoints` | `checkpoints` | context metric |
| `failed_pokes` | `failed_pokes` | lower is better |
| `exit_status` | `exit_status` | zero required |

## Baseline Keys (`G005`)

Primary baseline grouping key:

- `env + fixture + branch`

Sparse-data fallback grouping key:

- `env + fixture` (used only if primary key has insufficient samples)

## Baseline Eligibility (`G006`)

Default baseline row eligibility:

- last `20` runs max
- max age `30` days
- `exit_status == 0`
- `failed_pokes == 0`
- minimum `5` rows after filtering

If insufficient rows remain, guard returns `insufficient_baseline`.

## Contract Schema (`G007`)

Contract file format: TOML.

Top-level sections:

- `[metadata]` (name, version)
- `[baseline]` (`window_runs`, `max_age_days`, `min_samples`)
- `[rules.<metric>]` per canonical metric

Rule fields:

- `floor_pct_of_baseline` (for higher-is-better metrics)
- `ceiling_pct_of_baseline` (for lower-is-better metrics)
- `absolute_floor`
- `absolute_ceiling`
- `severity` (`warn` or `fail`)
- `weight` (for weighted rollups)

## Report Schemas (`G008`)

Machine report (`guard-report.json`) includes:

- run identity and key tuple
- baseline selection summary
- per-metric results (candidate, baseline center, delta, decision)
- rolled-up verdict
- autopsy hints

Human report (`guard-report.md`) includes:

- verdict summary
- failed/warned rules table
- baseline sufficiency notes
- autopsy suspect symbols/anomalies

## Exit Codes (`G009`)

- `0`: pass
- `2`: regression detected (one or more fail-severity rule violations)
- `3`: insufficient baseline
- `4`: configuration or input error
