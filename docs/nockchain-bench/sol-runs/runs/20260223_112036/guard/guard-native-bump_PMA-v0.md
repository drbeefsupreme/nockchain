# SOL Guard Report

- Run: `20260217_183413` (`native` / `bump PMA` / `v0`)
- Verdict: `pass`
- Baseline samples: `5`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.6900 | 3.6500 | 1.10 | Warn | yes | within contract (baseline median CI [3.4500, 3.6900]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 71.2200 | 67.7100 | 5.18 | Fail | yes | within contract (baseline median CI [66.5800, 68.7700]) |

## Autopsy

- Top stack shifts vs baseline
