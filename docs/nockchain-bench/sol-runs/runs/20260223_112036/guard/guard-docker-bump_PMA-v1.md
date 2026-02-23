# SOL Guard Report

- Run: `20260217_183413` (`docker` / `bump PMA` / `v1`)
- Verdict: `pass`
- Baseline samples: `10`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.0300 | 3.0950 | -2.10 | Warn | yes | within contract (baseline median CI [2.5300, 3.2050]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 74.5900 | 72.7900 | 2.47 | Fail | yes | within contract (baseline median CI [25.6200, 75.5450]) |

## Autopsy

- none
