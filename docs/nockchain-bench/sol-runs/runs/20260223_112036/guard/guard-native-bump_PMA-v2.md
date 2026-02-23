# SOL Guard Report

- Run: `20260217_183413` (`native` / `bump PMA` / `v2`)
- Verdict: `pass`
- Baseline samples: `5`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.6600 | 3.6300 | 0.83 | Warn | yes | within contract (baseline median CI [3.4700, 3.7500]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 71.0600 | 68.7100 | 3.42 | Fail | yes | within contract (baseline median CI [67.8900, 68.8900]) |

## Autopsy

- Top stack shifts vs baseline
