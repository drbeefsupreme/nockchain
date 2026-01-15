# Instructions

      CHECKPOINT_INTERVAL=60 \…)
  ⎿  ==============================================
     Nockchain RAM Comparison Experiment
     ==============================================
     … +161 lines (ctrl+o to expand)

● This time it worked. Here are the complete results:

  Results Summary
  ┌──────────────┬─────────────────┬─────────────────────┐
  │    Metric    │ Checkpoint Mode │  PMA Persist Mode   │
  ├──────────────┼─────────────────┼─────────────────────┤
  │ Peak RAM     │ 3.057 GiB       │ 1.837 GiB           │
  ├──────────────┼─────────────────┼─────────────────────┤
  │ Steady State │ ~3.06 GiB       │ ~1.83 GiB           │
  ├──────────────┼─────────────────┼─────────────────────┤
  │ RAM Savings  │ —               │ 40% less (~1.2 GiB) │
  └──────────────┴─────────────────┴─────────────────────┘
  Data Files Created

  ~/nockchain-experiment/
  ├── checkpoints/
  │   ├── 0.chkjam         (3.9 MB) - Checkpoint slot 0
  │   └── 1.chkjam         (3.9 MB) - Checkpoint slot 1
  ├── pma/
  │   ├── pma.meta         (98 bytes) - PMA metadata
  │   └── pma.mmap         (16 GB sparse file) - Memory-mapped arena
  ├── checkpoint-ram.log
  └── pma-persist-ram.log

  Key Observations

  1. Checkpoint Mode RAM Growth

  0s:   598 MiB  (startup)
  6s:   2.0 GiB  (kernel loaded, mining started)
  12s:  2.4 GiB  (state accumulating)
  18s:  2.7 GiB
  24s:  3.0 GiB  (checkpoint triggered around here)
  30s+: 3.06 GiB (stabilized after checkpoint)

  The checkpoint was created very quickly (~10 seconds) because fakenet mining is fast. Both checkpoint files exist (3.9 MB each) showing the alternating save worked.

  2. PMA Persist Mode - Flat RAM

  0s:   600 MiB  (startup)
  6s:   1.8 GiB  (kernel loaded)
  12s+: 1.83 GiB (completely stable for 2 minutes)

  RAM stays flat because after each poke, state is copied to PMA and the NockStack is reset.

  3. PMA File Size

  The pma.mmap file is 16 GB but that's a sparse file - it doesn't actually use 16 GB on disk:

● Bash(du -h ~/nockchain-experiment/pma/pma.mmap && du --apparent-size -h ~/nockchain-experiment/pma/pma.mmap)
  ⎿  150M	/home/drbeefsupreme/nockchain-experiment/pma/pma.mmap   
     17G	/home/drbeefsupreme/nockchain-experiment/pma/pma.mmap

● The PMA file is a sparse file:
  - Apparent size: 17 GB (virtual address space)
  - Actual disk usage: 150 MB (only allocated pages)

  Conclusions

  1. PMA persist mode uses ~40% less RAM than checkpoint mode for the same workload
  2. Checkpoint mode accumulates state in the NockStack between saves, causing RAM growth
  3. PMA persist mode stays flat because the NockStack is reset after each event
  4. Both persistence mechanisms work - checkpoints created, PMA metadata written
  5. Sparse files mean the 16 GB PMA arena doesn't actually consume 16 GB disk

  This validates that PMA persistence is more RAM-efficient for long-running nodes, at the cost of more frequent disk I/O (every poke vs. every 60-120 seconds).
