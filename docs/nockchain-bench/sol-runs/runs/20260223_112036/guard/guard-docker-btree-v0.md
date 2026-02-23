# SOL Guard Report

- Run: `20260217_183413` (`docker` / `btree` / `v0`)
- Verdict: `pass`
- Baseline samples: `10`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.1100 | 3.0600 | 1.63 | Warn | yes | within contract (baseline median CI [2.4900, 3.2150]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 70.2400 | 71.8700 | -2.27 | Fail | yes | within contract (baseline median CI [25.5000, 74.8850]) |

## Autopsy

- none
