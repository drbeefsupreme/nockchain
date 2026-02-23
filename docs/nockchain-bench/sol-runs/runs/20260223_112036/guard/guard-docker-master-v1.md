# SOL Guard Report

- Run: `20260217_183413` (`docker` / `master` / `v1`)
- Verdict: `fail`
- Baseline samples: `10`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 2.4200 | 3.0950 | -21.81 | Warn | yes | within contract (baseline median CI [2.5300, 3.2050]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 23.9800 | 72.7900 | -67.06 | Fail | no | candidate 23.9800 < floor 69.1505 (95.0% of baseline) |

## Autopsy

- throughput_blocks_s regression: candidate 23.9800 < floor 69.1505 (95.0% of baseline)
