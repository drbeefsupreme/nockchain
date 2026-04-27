# Cold Peeking Simplification Dashboard

## Summary

| Metric | Before | After | Delta | Direction |
|---|---:|---:|---:|---|
| `cold_peek/cgroup.rs` LOC | 856 | 800 | -56 | down |
| `cold_peek/mod.rs` LOC | 121 | 111 | -10 | down |
| Net source LOC | 977 | 911 | -66 | down |
| Current branch quick-orchestrate tests | 13 | 13 | 0 | same |
| Current branch force-cold filter tests | 4 | 4 | 0 | same |
| PMA `cold_init_` tests | 2 | 2 | 0 | same |
| PMA `quick_orchestrate_` tests | 15 | 15 | 0 | same |
| PMA `force_cold_` tests | 8 passed, 2 ignored | 8 passed, 2 ignored | 0 | same |

## Notes

- One accepted candidate: hoist duplicated cold-peek data/error types into shared `cold_peek/mod.rs`.
- Standalone current checkout still cannot compile `pma-runtime-compat`, matching the handoff note that feature-gated verification belongs in the PMA checkout.
- PMA checkpoint-backed cold smoke was attempted but blocked by host cgroup delegation: `memory` is not delegated in `cgroup.subtree_control`.

