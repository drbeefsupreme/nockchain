# SOL Guard Report

- Run: `20260217_183413` (`native` / `bump PMA` / `v1`)
- Verdict: `pass`
- Baseline samples: `5`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.6100 | 3.5900 | 0.56 | Warn | yes | within contract (baseline median CI [3.5100, 3.7800]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 70.6300 | 68.1400 | 3.65 | Fail | yes | within contract (baseline median CI [66.4500, 69.0200]) |

## Autopsy

- Top stack shifts vs baseline
