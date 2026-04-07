# SOL Sweep Comparison

| Case | Axes | Verdict | Throughput Median | Notes |
| --- | --- | --- | --- | --- |
| case-000-fixture_fixtures-51000-51099-derived-checkpoint-no-mempool-soltest-work_dir_mode_dockervolume | fixture=fixtures-51000-51099-derived-checkpoint-no-mempool-soltest, work_dir_mode=dockervolume | Valid | 6.65 | - |
| case-001-fixture_fixtures-51000-51099-derived-checkpoint-no-mempool-soltest-work_dir_mode_dockertmpfs | fixture=fixtures-51000-51099-derived-checkpoint-no-mempool-soltest, work_dir_mode=dockertmpfs | Valid | 6.61 | - |
| case-002-fixture_fixtures-51000-51099-derived-checkpoint-no-mempool-soltest-work_dir_mode_hostbind | fixture=fixtures-51000-51099-derived-checkpoint-no-mempool-soltest, work_dir_mode=hostbind | Valid | 6.62 | - |
| case-003-fixture_fixtures-51000-51099-full-checkpoint-no-mempool-soltest-work_dir_mode_dockervolume | fixture=fixtures-51000-51099-full-checkpoint-no-mempool-soltest, work_dir_mode=dockervolume | Valid | 6.59 | - |
| case-004-fixture_fixtures-51000-51099-full-checkpoint-no-mempool-soltest-work_dir_mode_dockertmpfs | fixture=fixtures-51000-51099-full-checkpoint-no-mempool-soltest, work_dir_mode=dockertmpfs | Valid | 6.59 | - |
| case-005-fixture_fixtures-51000-51099-full-checkpoint-no-mempool-soltest-work_dir_mode_hostbind | fixture=fixtures-51000-51099-full-checkpoint-no-mempool-soltest, work_dir_mode=hostbind | Valid | 6.53 | - |
| case-006-fixture_fixtures-first-100-v2-derived-checkpoint-no-mempool-soltest-work_dir_mode_dockervolume | fixture=fixtures-first-100-v2-derived-checkpoint-no-mempool-soltest, work_dir_mode=dockervolume | Valid | 6.58 | - |
| case-007-fixture_fixtures-first-100-v2-derived-checkpoint-no-mempool-soltest-work_dir_mode_dockertmpfs | fixture=fixtures-first-100-v2-derived-checkpoint-no-mempool-soltest, work_dir_mode=dockertmpfs | Valid | 6.50 | - |
| case-008-fixture_fixtures-first-100-v2-derived-checkpoint-no-mempool-soltest-work_dir_mode_hostbind | fixture=fixtures-first-100-v2-derived-checkpoint-no-mempool-soltest, work_dir_mode=hostbind | Valid | 6.60 | - |
| case-009-fixture_fixtures-first-100-v2-full-checkpoint-no-mempool-soltest-work_dir_mode_dockervolume | fixture=fixtures-first-100-v2-full-checkpoint-no-mempool-soltest, work_dir_mode=dockervolume | Valid | 6.56 | - |
| case-010-fixture_fixtures-first-100-v2-full-checkpoint-no-mempool-soltest-work_dir_mode_dockertmpfs | fixture=fixtures-first-100-v2-full-checkpoint-no-mempool-soltest, work_dir_mode=dockertmpfs | Valid | 6.48 | - |
| case-011-fixture_fixtures-first-100-v2-full-checkpoint-no-mempool-soltest-work_dir_mode_hostbind | fixture=fixtures-first-100-v2-full-checkpoint-no-mempool-soltest, work_dir_mode=hostbind | Partial | 6.60 | throughput CV 0.216 exceeded threshold 0.10 |
