# SOL Guard Report

- Run: `20260217_183413` (`native` / `master` / `v0`)
- Verdict: `fail`
- Baseline samples: `6`

## Metrics

| metric | candidate | baseline median | delta % | severity | passed | reason |
|---|---:|---:|---:|---|---|---|
| InitTimeS | 3.0400 | 3.0150 | 0.83 | Warn | yes | within contract (baseline median CI [2.9300, 3.0450]) |
| MajorFaultsDelta | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| PeakRssMib | 0.0000 | 0.0000 | - | Fail | yes | within contract (baseline median CI [0.0000, 0.0000]) |
| ThroughputBlocksS | 22.3200 | 23.5800 | -5.34 | Fail | no | candidate 22.3200 < floor 22.4010 (95.0% of baseline) |

## Autopsy

- throughput_blocks_s regression: candidate 22.3200 < floor 22.4010 (95.0% of baseline)
- Top stack shifts vs baseline
