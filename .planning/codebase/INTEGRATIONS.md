# External Integrations

**Analysis Date:** 2026-02-24

## APIs & External Services

**Blockchain Networks:**
- Base (Optimism stack) - bridge monitors and submits deposit/withdrawal flows over WebSocket RPC.
  - SDK/Client: `alloy`, `op-alloy` in `crates/bridge/Cargo.toml`.
  - Auth: configured with private key material in bridge config (`my_eth_key` in `crates/bridge/bridge-conf.example.toml`) and deployment envs in `crates/bridge/contracts/.env.template`.

**Package/Artifact Distribution:**
- GitHub Releases API and release assets - `nockup` downloads manifests and binaries from `nockchain/nockchain` releases.
  - SDK/Client: `reqwest` in `crates/nockup/src/commands/common.rs`.
  - Auth: unauthenticated GET for public endpoints (no API key required in code path).

**Registry Fetching:**
- GitHub raw content - package registry TOML fetched from raw GitHub URL.
  - SDK/Client: `reqwest::blocking` in `crates/nockup/src/resolver/registry.rs`.
  - Auth: none detected.

**Certificate Authority / TLS Automation:**
- Let's Encrypt ACME flow for HTTP driver TLS provisioning.
  - SDK/Client: `instant-acme` via ACME manager path in `crates/nockapp/src/drivers/http/http.rs` and dependency in `crates/nockapp/Cargo.toml`.
  - Auth: `ACME_EMAIL`, `ACME_CACHE_DIR`, and domain controls from env in `crates/nockapp/src/drivers/http/http.rs`.

**Observability Backend:**
- Datadog OTLP endpoint (or compatible OTLP collector) for traces.
  - SDK/Client: `opentelemetry-otlp` + `tracing-opentelemetry` in `crates/nockapp/Cargo.toml`.
  - Auth: host/port and service metadata from env (`DD_AGENT_HOST`, `DD_OTLP_GRPC_PORT`, `DD_SERVICE`, `DD_VERSION`, `DD_ENV`) in `crates/nockapp/src/observability.rs`.

**Sync/Automation Services:**
- GitHub Actions and GitLab CI pipelines for release, formatting, and sync automation in `.github/workflows/release.yml`, `.github/workflows/sync-to-nockchain.yml`, `.github/workflows/create-sync-pr.yml`, and `.gitlab-ci.yml`.
  - SDK/Client: GitHub Actions runners + `gh` CLI usage in workflow scripts.
  - Auth: repository secrets (for example `NOCKCHAIN_SYNC_TOKEN`, `GITHUB_TOKEN`, `GITHUB_NOCKCHAIN_STAGING_TOKEN`) referenced in workflow files.

## Data Storage

**Databases:**
- SQLite (local file-backed) for bridge deposit queue/log state.
  - Connection: filesystem path-based DSN, not centralized DB URL (`deposit-log.sqlite` / `deposit-queue.sqlite` paths in `crates/bridge/src/deposit_log.rs` and `crates/bridge/src/main.rs`).
  - Client: `diesel` with `deadpool-diesel` in `crates/bridge/Cargo.toml`.

**File Storage:**
- Local filesystem only for checkpoints, cache, logs, and generated assets (`.data.*`, `~/.nockapp`, `assets/*.jam`, `~/.nockup`) in `crates/nockapp/src/lib.rs`, `Makefile`, and `crates/nockup/src/commands/common.rs`.

**Caching:**
- In-process memory caches for HTTP responses and gRPC explorer data in `crates/nockapp/src/drivers/http/http.rs` and `crates/nockapp-grpc/src/services/public_nockchain/v2/server.rs`.
- On-disk cache for `nockup` manifests/binaries in user cache directories via `crates/nockup/src/cache.rs` and `crates/nockup/src/commands/common.rs`.

## Authentication & Identity

**Auth Provider:**
- Custom cryptographic identity model (no hosted IdP detected).
  - Implementation: wallet/node keypairs and bridge node keys configured via CLI/config/env (`MINING_PKH` in `.env_example`, bridge keys in `crates/bridge/bridge-conf.example.toml`, wallet endpoint options in `crates/nockchain-wallet/src/connection.rs`).

## Monitoring & Observability

**Error Tracking:**
- Not detected as a dedicated SaaS error tracker.

**Logs:**
- Structured tracing/logging through `tracing` ecosystem with env-driven filters (`RUST_LOG`, minimal format toggle) in `Makefile`, `.env_example`, and `crates/nockapp/src/kernel/boot.rs`.
- Bridge file-log rotation and retention in `crates/bridge/src/main.rs` and `crates/bridge/src/tui/mod.rs`.

## CI/CD & Deployment

**Hosting:**
- Self-hosted binary execution is primary runtime model (native binaries built and installed via Cargo/Make in `Makefile`, runtime entry points in `crates/nockchain/src/main.rs`, `crates/nockchain-api/src/main.rs`, `crates/bridge/src/main.rs`).
- Release artifact hosting on GitHub Releases in `.github/workflows/release.yml`.

**CI Pipeline:**
- GitHub Actions workflows for formatting, releases, and repo sync in `.github/workflows/*.yml`.
- GitLab CI pipeline for kernel builds/tests and staging sync in `.gitlab-ci.yml`.

## Environment Configuration

**Required env vars:**
- Core runtime examples: `RUST_LOG`, `MINIMAL_LOG_FORMAT`, `MINING_PKH` from `.env_example` and `Makefile`.
- HTTP/HTTPS driver: `HTTPS_DOMAIN`, `WEB_DIR`, `ACME_EMAIL`, `ACME_CACHE_DIR`, `EXPIRE_CACHE` from `crates/nockapp/src/drivers/http/http.rs`.
- Observability: `DD_AGENT_HOST`, `DD_OTLP_GRPC_PORT`, `DD_SERVICE`, `DD_VERSION`, `DD_ENV`, `OTEL_TRACES_SAMPLE_RATE` from `crates/nockapp/src/observability.rs`.
- Bridge contracts/deploy: `TENDERLY_RPC_URL`, `TENDERLY_PRIVATE_KEY`, `BRIDGE_NODE_0..4`, `INBOX_PRIVATE_KEY`, `TENDERLY_ACCESS_KEY` from `crates/bridge/contracts/.env.template` and `crates/bridge/contracts/DEPLOYMENT.md`.

**Secrets location:**
- `.env` files and CI secret stores are used; `.env` exists at repo root and `.env.template`/examples exist for bridge contracts (`.env` presence at repository root, templates in `crates/bridge/contracts/.env.template`).

## Webhooks & Callbacks

**Incoming:**
- ACME HTTP-01 callback route `/.well-known/acme-challenge/{token}` in `crates/nockapp/src/drivers/http/http.rs`.
- gRPC ingress server for bridge coordination in `crates/bridge/src/ingress.rs`.

**Outgoing:**
- Outbound WebSocket RPC calls to Base endpoint (`base_ws_url`) in `crates/bridge/src/ethereum.rs` and `crates/bridge/bridge-conf.example.toml`.
- Outbound HTTPS requests to GitHub APIs and release URLs from `crates/nockup/src/commands/common.rs` and `crates/nockup/src/resolver/registry.rs`.

---

*Integration audit: 2026-02-24*
