# External Integrations

**Analysis Date:** 2026-03-03

## APIs & External Services

**Container Runtime:**
- Docker Engine API - benchmark container lifecycle and stats collection
  - Client: `bollard`
  - Auth: Unix socket access to local Docker daemon
  - Paths: `crates/nockchain-bench/src/runner/docker.rs`

**Network Stack:**
- libp2p network protocols - runtime node networking behavior
  - Integration: internal crates (`nockchain-libp2p-io`, `nockchain`)
  - Transport: QUIC/DNS/libp2p stack

**Artifact and Report Tooling:**
- Local file-based benchmark artifact contracts (`.solarch`, `.soltest`, `.chkjam`, TSV/Parquet)
- CI workflows consume benchmark CLI outputs for baseline/regression checks

## Data Storage

**Databases:**
- No external DB is required for `nockchain-bench` core workflows

**File Storage:**
- Local filesystem artifacts in repo and temp dirs
- Docker bind mounts for fixtures/checkpoints

**Caching:**
- No dedicated cache backend in benchmark flows

## Authentication & Identity

**Auth Provider:**
- None required for local benchmark execution

## Monitoring & Observability

**Tracing/Profiling:**
- Tracy integration for profiling traces
- Linux `perf` integration in matrix scripts
- `/proc` parsing for process memory/page-fault attribution

## CI/CD & Deployment

**CI Pipeline:**
- GitHub Actions workflows run benchmark regression/baseline orchestration
- Secrets are managed via CI environment when needed (not captured in docs)

## Environment Configuration

**Development:**
- Requires Docker daemon and filesystem access to fixture/checkpoint paths
- Optional `perf` and tracing tools for deep profiling

**Production/Benchmark Runs:**
- environment-specific runner settings encoded in scripts and benchmark config files

## Webhooks & Callbacks

**Incoming/Outgoing:**
- No webhook-style external callbacks in `nockchain-bench` runtime paths

---
*Integration audit: 2026-03-03*
*Update when adding/removing external services*
