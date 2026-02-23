# SOL Guard Report

- Run: `20260217_183413` (`native` / `master` / `v1`)
- Verdict: `pass`
- Baseline samples: `6`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.0400 | 2.9950 | 1.50 | Warn | yes | within contract (baseline median CI [2.9500, 3.0350]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 22.7500 | 23.5200 | -3.27 | Fail | yes | within contract (baseline median CI [22.2200, 23.9200]) |

## Autopsy

- Top stack shifts vs baseline
