# SOL Guard Report

- Run: `20260217_183413` (`native` / `master` / `v2`)
- Verdict: `pass`
- Baseline samples: `6`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.1000 | 2.9450 | 5.26 | Warn | yes | within contract (baseline median CI [2.8800, 3.0200]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 22.1500 | 23.2300 | -4.65 | Fail | yes | within contract (baseline median CI [22.0900, 23.9550]) |

## Autopsy

- Top stack shifts vs baseline
