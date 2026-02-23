# SOL Guard Report

- Run: `20260217_183413` (`native` / `btree` / `v1`)
- Verdict: `pass`
- Baseline samples: `5`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.7300 | 3.6500 | 2.19 | Warn | yes | within contract (baseline median CI [3.6100, 3.7000]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 67.9100 | 68.6100 | -1.02 | Fail | yes | within contract (baseline median CI [67.3600, 69.6200]) |

## Autopsy

- Top stack shifts vs baseline
