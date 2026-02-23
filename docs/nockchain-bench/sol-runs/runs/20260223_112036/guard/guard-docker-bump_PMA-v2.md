# SOL Guard Report

- Run: `20260217_183413` (`docker` / `bump PMA` / `v2`)
- Verdict: `pass`
- Baseline samples: `10`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.0400 | 3.0650 | -0.82 | Warn | yes | within contract (baseline median CI [2.5200, 3.1900]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 75.5100 | 73.3000 | 3.02 | Fail | yes | within contract (baseline median CI [25.6900, 75.1850]) |

## Autopsy

- none
