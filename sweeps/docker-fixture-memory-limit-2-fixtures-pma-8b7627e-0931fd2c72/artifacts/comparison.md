# SOL Sweep Comparison

| Case | Axes | Verdict | Throughput Median | Notes |
| --- | --- | --- | --- | --- |
| case-000-fixture_fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest-memory_limit_1g | fixture=fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest, memory_limit=1g | Valid | 42.78 | - |
| case-001-fixture_fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest-memory_limit_2g | fixture=fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest, memory_limit=2g | Valid | 44.02 | - |
| case-002-fixture_fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest-memory_limit_4g | fixture=fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest, memory_limit=4g | Valid | 45.29 | - |
| case-003-fixture_fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest-memory_limit_8g | fixture=fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest, memory_limit=8g | Valid | 45.16 | - |
| case-004-fixture_fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest-memory_limit_16g | fixture=fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest, memory_limit=16g | Valid | 45.02 | - |
| case-005-fixture_fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest-memory_limit_32g | fixture=fixtures-first-1000-v2-derived-checkpoint-no-mempool-soltest, memory_limit=32g | Partial | 44.55 | throughput CV 0.399 exceeded threshold 0.10 |
| case-007-fixture_fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest-memory_limit_2g | fixture=fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest, memory_limit=2g | Valid | 41.44 | - |
| case-008-fixture_fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest-memory_limit_4g | fixture=fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest, memory_limit=4g | Valid | 43.15 | - |
| case-009-fixture_fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest-memory_limit_8g | fixture=fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest, memory_limit=8g | Valid | 43.29 | - |
| case-010-fixture_fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest-memory_limit_16g | fixture=fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest, memory_limit=16g | Partial | 43.00 | throughput CV 0.344 exceeded threshold 0.10 |
| case-011-fixture_fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest-memory_limit_32g | fixture=fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest, memory_limit=32g | Valid | 42.50 | - |
| case-006-fixture_fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest-memory_limit_1g | fixture=fixtures-first-1000-v2-full-checkpoint-no-mempool-soltest, memory_limit=1g | Invalid | - | Command failure: docker exec nockchain-bench-1441225-1775152940050 nockchain-bench sol run-once --resolved-case /bench/input/resolved_case.json --run-dir /bench/output/runs/run-0 --run-id run-0 failed:  |
