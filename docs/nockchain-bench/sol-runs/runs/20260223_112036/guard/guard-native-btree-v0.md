# SOL Guard Report

- Run: `20260217_183413` (`native` / `btree` / `v0`)
- Verdict: `pass`
- Baseline samples: `5`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.6700 | 3.7000 | -0.81 | Warn | yes | within contract (baseline median CI [3.6400, 3.8700]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 68.4000 | 69.0500 | -0.94 | Fail | yes | within contract (baseline median CI [67.4400, 69.9600]) |

## Autopsy

- Top stack shifts vs baseline
