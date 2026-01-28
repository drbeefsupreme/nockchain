  ---
  nockchain-bench Crate: Design Proposal

  Context from Documentation Review

  Current state:
  - Tracy is integrated and provides detailed allocation tracking
  - compare_pma_mem.rs already does /proc/smaps parsing and memory bucketing
  - PMA persistence mode is working and shows ~40-85% RSS reduction vs checkpointing
  - The checkpoint RAM spike problem is well-documented in CHECKPOINT-HOT-PAGES.md

  Planned features that need benchmarking:
  - Streaming checkpoints (jam directly to disk, skip NounSlab copies)
  - COW replicas (Phase 5 of PMA-NOW.md)
  - PMA GC (multiple options in PMA-GC-IDEAS.md)
  - Dynamic NockStack growth (Option 1 in DYNAMIC-NOCKSTACK.md)

  ---
  Core Design Philosophy

  Don't duplicate what Tracy does well:
  - Allocation-level tracking → Tracy
  - Per-function memory attribution → Tracy
  - Timeline visualization → Tracy Profiler GUI

  Fill gaps Tracy doesn't cover:
  - Automated regression testing (no GUI)
  - Process-level memory bucketing (NockStack vs PMA vs heap)
  - A/B comparison across configurations
  - Time-series correlation with checkpoint/poke events
  - Docker orchestration for reproducible environments

  ---
  Proposed Architecture

  crates/nockchain-bench/
  ├── src/
  │   ├── lib.rs              # Public API for programmatic use
  │   ├── main.rs             # CLI: `nockchain-bench run --scenario mining`
  │   │
  │   ├── config.rs           # Configuration types
  │   │   - NockchainConfig { checkpoint_mode, pma_persist, stack_size, ... }
  │   │   - SamplerConfig { interval_ms, buckets, page_faults, ... }
  │   │   - ScenarioConfig { blocks_to_mine, sync_target, ... }
  │   │
  │   ├── runner/
  │   │   ├── mod.rs          # Runner trait
  │   │   ├── docker.rs       # bollard-based Docker runner
  │   │   └── process.rs      # Native subprocess runner
  │   │
  │   ├── sampler/
  │   │   ├── mod.rs          # Sampler trait + time-series collector
  │   │   ├── smaps.rs        # /proc/<pid>/smaps parsing (port from compare_pma_mem.rs)
  │   │   ├── buckets.rs      # NockStack/PMA/heap attribution logic
  │   │   └── stat.rs         # /proc/<pid>/stat for page faults
  │   │
  │   ├── events/
  │   │   ├── mod.rs          # Event correlation
  │   │   ├── log_watcher.rs  # Parse checkpoint/poke events from logs
  │   │   └── markers.rs      # Event markers (CheckpointStart, CheckpointEnd, PokeComplete, etc.)
  │   │
  │   ├── scenario/
  │   │   ├── mod.rs          # Scenario trait
  │   │   ├── mining.rs       # MineBlocks { count, difficulty }
  │   │   ├── checkpoint.rs   # ForceCheckpoint, WaitForCheckpoint
  │   │   └── sync.rs         # SyncFromPeer (future)
  │   │
  │   ├── compare.rs          # A/B comparison logic
  │   │   - Compare two BenchResult sets
  │   │   - Statistical tests (Mann-Whitney, t-test)
  │   │   - Regression detection
  │   │
  │   └── output/
  │       ├── mod.rs
  │       ├── parquet.rs      # Export to Parquet for Polars/DuckDB analysis
  │       ├── json.rs         # JSON metadata + summary
  │       └── report.rs       # Terminal report (like compare_pma_mem.rs output)
  │
  └── tests/
      └── scenarios.rs        # Test that scenarios run correctly

  ---
  Key Features

  1. Memory Bucketing (port from compare_pma_mem.rs)

  struct MemorySample {
      timestamp_ms: u64,

      // From /proc/<pid>/status
      vm_rss_kb: u64,
      vm_size_kb: u64,
      rss_anon_kb: u64,
      rss_file_kb: u64,

      // Bucketed from /proc/<pid>/smaps
      nockstack_rss_kb: u64,      // Single large anon mapping matching stack size
      pma_rss_kb: u64,            // File-backed .mmap mappings
      pma_size_kb: u64,           // Total PMA mapped size
      heap_other_anon_kb: u64,    // [heap] + remaining anon (jam buffers, slabs)

      // From /proc/<pid>/stat
      minor_faults: u64,
      major_faults: u64,

      // Derived
      pma_rss_ratio: f64,         // pma_rss / pma_size (paging effectiveness)
  }

  2. Event Correlation

  enum BenchEvent {
      CheckpointStarted { event_num: u64 },
      CheckpointCompleted { event_num: u64, duration_ms: u64, size_bytes: u64 },
      PokeCompleted { event_num: u64, duration_ms: u64 },
      BlockMined { height: u64, hash: String },
      PmaPreserveCompleted { copied_bytes: u64, duration_ms: u64 },
  }

  struct TimeSeries {
      samples: Vec<MemorySample>,
      events: Vec<(u64, BenchEvent)>,  // (timestamp_ms, event)
  }

  This lets you overlay memory graphs with events to see exactly what causes spikes.

  3. A/B Testing

  struct BenchComparison {
      baseline: BenchResult,
      variant: BenchResult,

      // Statistical comparison
      rss_peak_delta_pct: f64,
      rss_p99_delta_pct: f64,
      checkpoint_spike_delta_pct: f64,  // Peak during checkpoint vs steady state
      pma_rss_ratio_improvement: f64,

      // Regression verdict
      is_regression: bool,
      confidence: f64,
  }

  4. Streaming Checkpoint Testing (anticipating future work)

  enum CheckpointMode {
      Periodic { interval_secs: u64 },       // Current: full slab copy + jam
      PmaPersist,                            // Current: PMA + pma.meta, no checkpoint files
      StreamingJam { chunk_size_bytes: u64 }, // Future: jam directly to disk
      IncrementalPma { delta_only: bool },   // Future: only persist changed PMA pages
  }

  // Metrics specific to checkpoint testing
  struct CheckpointMetrics {
      peak_rss_during_checkpoint: u64,
      steady_state_rss: u64,
      checkpoint_spike_ratio: f64,  // peak / steady (should approach 1.0 for streaming)

      // From CHECKPOINT-HOT-PAGES.md hypotheses
      anon_spike_kb: u64,           // H1: slab + jam buffer allocations
      pma_residency_spike: f64,     // H2: PMA pages faulted in during checkpoint
      private_dirty_spike_kb: u64,  // H3: dirty pages during checkpoint
  }

  5. COW Replica Testing (anticipating Phase 5)

  struct ReplicaMetrics {
      leader_rss_kb: u64,
      replica_rss_kb: Vec<u64>,  // Per-replica
      shared_pma_pages: u64,     // Pages shared via COW
      private_cow_pages: u64,    // Pages copy-on-written
      replica_lag_events: Vec<u64>,
  }

  // Scenario for testing replicas
  struct ReplicaScenario {
      leader_config: NockchainConfig,
      replica_count: usize,
      peek_load_rps: f64,        // Peek requests per second to replicas
      poke_load_rps: f64,        // Poke requests to leader
  }

  ---
  CLI Interface

  # Run a scenario and output results
  nockchain-bench run \
      --scenario mining \
      --blocks 100 \
      --config checkpoint \
      --sample-interval 100ms \
      --output results/checkpoint-mining.parquet

  # A/B comparison
  nockchain-bench compare \
      --baseline results/checkpoint-mining.parquet \
      --variant results/pma-persist-mining.parquet \
      --output results/comparison.json

  # Quick regression check (returns exit code)
  nockchain-bench regress \
      --baseline results/baseline.parquet \
      --current results/current.parquet \
      --threshold 5%  # Fail if RSS increased >5%

  # Time-series export for external analysis
  nockchain-bench export \
      --input results/checkpoint-mining.parquet \
      --format csv \
      --output results/timeseries.csv

  ---
  Integration with Tracy

  Tracy remains the tool for detailed profiling. nockchain-bench complements it:
  ┌───────────────────────────────────────────────────────┬─────────────────┐
  │                       Use Case                        │      Tool       │
  ├───────────────────────────────────────────────────────┼─────────────────┤
  │ "Where are allocations happening during checkpoint?"  │ Tracy           │
  ├───────────────────────────────────────────────────────┼─────────────────┤
  │ "Did streaming checkpoint reduce peak RSS?"           │ nockchain-bench │
  ├───────────────────────────────────────────────────────┼─────────────────┤
  │ "Which function is allocating the most?"              │ Tracy           │
  ├───────────────────────────────────────────────────────┼─────────────────┤
  │ "Is the new code a regression vs baseline?"           │ nockchain-bench │
  ├───────────────────────────────────────────────────────┼─────────────────┤
  │ "What's the memory timeline during a 100-block mine?" │ nockchain-bench │
  ├───────────────────────────────────────────────────────┼─────────────────┤
  │ "Why is this specific poke slow?"                     │ Tracy           │
  └───────────────────────────────────────────────────────┴─────────────────┘
  For detailed profiling sessions, you'd:
  1. Run nockchain-bench to identify when the spike happens
  2. Connect Tracy to the node during that phase
  3. Use Tracy to identify where allocations are happening

  ---
  Data Analysis Workflow

  # Example analysis with Polars (or Pandas)
  import polars as pl

  # Load time-series data
  df = pl.read_parquet("results/checkpoint-mining.parquet")

  # Find checkpoint events
  checkpoints = df.filter(pl.col("event_type") == "CheckpointStarted")

  # Calculate spike ratio
  df = df.with_columns([
      (pl.col("vm_rss_kb") / pl.col("steady_state_rss_kb")).alias("spike_ratio")
  ])

  # Plot with matplotlib/plotly
  # X: timestamp, Y: vm_rss_kb, vertical lines at checkpoint events

  ---
  Anticipated Future Needs

  1. Streaming Checkpoints:
    - Measure heap_other_anon_kb during checkpoint (should be much lower)
    - Measure checkpoint duration vs size
    - Compare backref map memory with/without streaming
  2. COW Replicas:
    - Measure per-replica RSS
    - Track page fault rate per replica
    - Measure replica lag vs memory pressure
  3. PMA GC:
    - Measure PMA fragmentation (alloc_offset vs live_bytes)
    - Track GC pause times
    - Measure RSS before/after GC
  4. Dynamic NockStack:
    - Track NockStack growth events
    - Measure page commit latency
    - Verify RSS stays bounded after stack shrinks

  ---
  Implementation Priority

  1. Port compare_pma_mem.rs logic → sampler/smaps.rs, sampler/buckets.rs
  2. Basic Docker runner → runner/docker.rs using bollard
  3. Mining scenario → scenario/mining.rs
  4. Time-series output → output/parquet.rs
  5. CLI skeleton → main.rs
  6. A/B comparison → compare.rs
  7. Event correlation → events/log_watcher.rs

  This gives you a working tool for the checkpoint vs PMA-persist comparison you just ran, but automated and with richer data.

  ---
  Would you like me to start scaffolding this crate, or would you prefer to discuss/refine the design further first?
