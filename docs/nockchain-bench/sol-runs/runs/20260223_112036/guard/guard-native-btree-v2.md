# SOL Guard Report

- Run: `20260217_183413` (`native` / `btree` / `v2`)
- Verdict: `pass`
- Baseline samples: `5`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.7100 | 3.6700 | 1.09 | Warn | yes | within contract (baseline median CI [3.5700, 3.7700]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 67.0300 | 68.7800 | -2.54 | Fail | yes | within contract (baseline median CI [67.6700, 70.2600]) |

## Autopsy

- Top stack shifts vs baseline
