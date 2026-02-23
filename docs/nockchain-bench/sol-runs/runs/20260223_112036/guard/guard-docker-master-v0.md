# SOL Guard Report

- Run: `20260217_183413` (`docker` / `master` / `v0`)
- Verdict: `fail`
- Baseline samples: `10`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 2.4200 | 3.0600 | -20.92 | Warn | yes | within contract (baseline median CI [2.4900, 3.2150]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 24.3000 | 71.8700 | -66.19 | Fail | no | candidate 24.3000 < floor 68.2765 (95.0% of baseline) |

## Autopsy

- throughput_blocks_s regression: candidate 24.3000 < floor 68.2765 (95.0% of baseline)
