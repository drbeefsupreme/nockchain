# SOL Guard Report

- Run: `20260217_183413` (`docker` / `btree` / `v2`)
- Verdict: `fail`
- Baseline samples: `10`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.0800 | 3.0650 | 0.49 | Warn | yes | within contract (baseline median CI [2.5200, 3.1900]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 68.1300 | 73.3000 | -7.05 | Fail | no | candidate 68.1300 < floor 69.6350 (95.0% of baseline) |

## Autopsy

- throughput_blocks_s regression: candidate 68.1300 < floor 69.6350 (95.0% of baseline)
