# Architecture

**Analysis Date:** 2026-03-03

## Pattern Overview

**Overall:** Rust workspace monorepo with a benchmark harness layered over runtime crates.

**Key Characteristics:**
- multi-crate workspace with shared domain/runtime crates
- CLI-first benchmark orchestration (`nockchain-bench`)
- dual benchmark modes: online/container scenario and offline SOL replay
- artifact-driven regression/guard workflows

## Layers

**Benchmark CLI Layer:**
- Purpose: parse commands and dispatch benchmark flows
- Contains: command enums, CLI parsing, output routing
- Depends on: scenario/runner, SOL modules, output modules
- Used by: `crates/nockchain-bench/src/main.rs`

**Execution Layer:**
- Purpose: execute benchmarks and collect runtime metrics
- Contains: `runner`, `scenario`, `sampler`
- Depends on: Docker API, system process/proc inspection
- Used by: CLI `run/compare/analyze` paths

**SOL Replay Layer:**
- Purpose: deterministic replay/extraction/checkpoint/guard workflows
- Contains: `speed_of_light/*` modules (`bench`, `extractor`, `fixture`, `checkpoint`, `guard`)
- Depends on: runtime crates (`nockapp`, `nockvm`, `nockchain-types`)
- Used by: CLI `sol *` subcommands

**Runtime Integration Layer:**
- Purpose: shared node/runtime behavior consumed by bench and node binaries
- Contains: `nockchain`, `nockapp`, `nockvm`, libp2p/grpc crates
- Used by: both production node paths and benchmark harnesses

## Data Flow

**SOL Bench Flow:**
1. User runs `nockchain-bench sol bench ...`
2. CLI resolves fixture/archive/checkpoint inputs
3. bench runner initializes runtime context from kernel/checkpoint
4. replay loop pokes archived blocks
5. profiling/metrics are sampled and summarized
6. results emitted to report files (TSV/JSON/Parquet)

**Container Scenario Flow:**
1. User runs `nockchain-bench run|compare|analyze`
2. scenario config is built
3. runner launches/attaches Docker container
4. sampler/event parser captures memory+log signals
5. summary/report artifacts are produced

## Entry Points

- `crates/nockchain-bench/src/main.rs` - benchmark binary entry
- `crates/nockchain/src/main.rs` - node runtime binary entry
- `scripts/sol_bench_matrix_trace.sh` - branch/matrix benchmark orchestration

## Error Handling

**Strategy:** typed domain errors internally (`thiserror`) with top-level CLI normalization and explicit exit behavior.

## Cross-Cutting Concerns

**Logging/Tracing:** `tracing` + optional Tracy integration.

**Validation:** command/config/contract validation across SOL guard and artifact parsers.

---
*Architecture analysis: 2026-03-03*
*Update when major patterns change*
