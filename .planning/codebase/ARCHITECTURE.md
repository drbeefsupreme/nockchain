# Architecture

**Analysis Date:** 2026-02-24

## Pattern Overview

**Overall:** Rust workspace with modular runtime + driver architecture around a Nock kernel.

**Key Characteristics:**
- Use crate boundaries as architecture boundaries: runtime core in `crates/nockapp/src/`, product composition in app crates like `crates/nockchain/src/` and `crates/bridge/src/`.
- Use I/O drivers as integration units; attach drivers with `add_io_driver` in `crates/nockapp/src/nockapp/mod.rs` and feature crates (`crates/nockchain/src/lib.rs`, `crates/bridge/src/main.rs`, `crates/nockchain-wallet/src/main.rs`).
- Use message-based kernel interaction (`poke`/`peek`) instead of direct domain mutation, via `NockAppHandle` in `crates/nockapp/src/nockapp/driver.rs`.

## Layers

**Workspace Composition Layer:**
- Purpose: Define all buildable units and shared dependency graph.
- Location: `Cargo.toml`.
- Contains: Workspace members, workspace dependency versions, profiles.
- Depends on: Cargo workspace metadata only.
- Used by: Every crate under `crates/`.

**Kernel Runtime Layer (`nockapp`):**
- Purpose: Host the kernel, schedule effects/actions, checkpoint state, and run lifecycle.
- Location: `crates/nockapp/src/nockapp/mod.rs`, `crates/nockapp/src/nockapp/driver.rs`, `crates/nockapp/src/kernel/boot.rs`.
- Contains: `NockApp`, `NockAppHandle`, `IOAction`, run loop, signal handling, save/checkpoint logic.
- Depends on: Kernel form/loading (`crates/nockapp/src/kernel/`), noun utilities (`crates/nockapp/src/noun/`), tokio/channels.
- Used by: `crates/nockchain/src/main.rs`, `crates/nockchain-api/src/main.rs`, `crates/nockchain-wallet/src/main.rs`, `crates/bridge/src/main.rs`, `crates/nockchain-peek/src/main.rs`.

**Protocol/Domain Layer:**
- Purpose: Encode chain/math/protocol data and transformations shared by services.
- Location: `crates/nockchain-types/src/`, `crates/nockchain-math/src/`, `crates/zkvm-jetpack/src/`.
- Contains: Tx engine versions (`crates/nockchain-types/src/tx_engine/v0/`, `crates/nockchain-types/src/tx_engine/v1/`), common primitives (`crates/nockchain-types/src/tx_engine/common/`), jets/hot-state helpers (`crates/zkvm-jetpack/src/hot.rs`).
- Depends on: Noun serialization and math utilities.
- Used by: Runtime adapters, gRPC services, wallet, node, bridge.

**Network/Adapter Layer:**
- Purpose: Translate kernel effects and requests into network protocols.
- Location: `crates/nockchain-libp2p-io/src/`, `crates/nockapp-grpc/src/services/`.
- Contains: libp2p swarm driver (`crates/nockchain-libp2p-io/src/driver.rs`), gRPC server/listener drivers (`crates/nockapp-grpc/src/services/public_nockchain/v1/driver.rs`, `crates/nockapp-grpc/src/services/private_nockapp/driver.rs`).
- Depends on: `nockapp` handle/wires, protocol types.
- Used by: Application composition layer (`crates/nockchain/src/lib.rs`, `crates/bridge/src/main.rs`, `crates/nockchain-wallet/src/main.rs`).

**Application Composition Layer:**
- Purpose: Compose kernels + drivers + CLI config into runnable binaries.
- Location: `crates/nockchain/src/main.rs`, `crates/nockchain/src/lib.rs`, `crates/bridge/src/main.rs`, `crates/nockchain-wallet/src/main.rs`, `crates/nockchain-api/src/main.rs`.
- Contains: CLI parse/validation, driver assembly, mode toggles (fakenet/public/private gRPC), app startup.
- Depends on: Runtime layer + adapter layer + kernel crates under `crates/kernels/`.
- Used by: End users and scripts in `scripts/`.

## Data Flow

**Node Runtime Flow (`nockchain`):**

1. Parse/validate CLI in `crates/nockchain/src/main.rs` and `crates/nockchain/src/config.rs`.
2. Boot kernel and runtime via `boot::setup` and `init_with_kernel` in `crates/nockchain/src/lib.rs`.
3. Register drivers (`mining`, `libp2p`, optional public gRPC, private gRPC, exit) in `crates/nockchain/src/lib.rs`.
4. Drivers submit `IOAction::Poke`/`IOAction::Peek` through `NockAppHandle` (`crates/nockapp/src/nockapp/driver.rs`).
5. Runtime loop executes kernel actions and broadcasts effects to drivers in `crates/nockapp/src/nockapp/mod.rs`.

**Bridge Flow:**

1. Load config and initialize runtime state in `crates/bridge/src/main.rs` and `crates/bridge/src/config.rs`.
2. Start bridge NockApp instance and install runtime driver in `crates/bridge/src/main.rs` and `crates/bridge/src/runtime.rs`.
3. Spawn ingress/network/watcher loops (`crates/bridge/src/ingress.rs`, `crates/bridge/src/nockchain.rs`, `crates/bridge/src/ethereum.rs`).
4. Persist/validate deterministic deposit queue through `crates/bridge/src/deposit_log.rs`.
5. Aggregate signatures/posting via `crates/bridge/src/proposal_cache.rs`, `crates/bridge/src/proposer.rs`, `crates/bridge/src/runtime.rs`.

**State Management:**
- Persist kernel state using checkpoint/saver mechanisms in `crates/nockapp/src/nockapp/mod.rs` and `crates/nockapp/src/nockapp/save.rs`.
- Store runtime data per app under data directories from `default_data_dir` and `system_data_dir` in `crates/nockapp/src/lib.rs`.
- Keep long-lived adapter state in driver-local structs (for example `P2PState` in `crates/nockchain-libp2p-io/src/p2p_state.rs`).

## Key Abstractions

**NockApp / NockAppHandle:**
- Purpose: Runtime container and message API between drivers and kernel.
- Examples: `crates/nockapp/src/nockapp/mod.rs`, `crates/nockapp/src/nockapp/driver.rs`.
- Pattern: Channel-based action dispatch + effect broadcast.

**IO Driver Function (`IODriverFn`):**
- Purpose: Pluggable async unit for side effects and protocol bridging.
- Examples: `crates/nockchain-libp2p-io/src/driver.rs`, `crates/nockapp-grpc/src/services/public_nockchain/v1/driver.rs`, `crates/nockapp-grpc/src/services/private_nockapp/driver.rs`.
- Pattern: `make_driver` closure captures config, then loops on `next_effect` and issues `poke`/`peek`.

**Wire Abstraction:**
- Purpose: Tag and route action causes between drivers and kernel.
- Examples: `crates/nockapp/src/nockapp/wire.rs`, `crates/nockchain/src/mining.rs`, `crates/nockchain-libp2p-io/src/driver.rs`.
- Pattern: Per-driver wire enums implementing `Wire` with static source/version and structured tags.

**Kernel Packaging via `kernels/*` crates:**
- Purpose: Provide compiled kernel jam bytes as dependency artifacts.
- Examples: `crates/kernels/dumb/src/lib.rs`, `crates/kernels/miner/src/lib.rs`, `crates/kernels/wallet/src/lib.rs`, `crates/kernels/bridge/src/lib.rs`.
- Pattern: Composition crates import `KERNEL` constants and pass them into `boot::setup`.

## Entry Points

**Main Node Binary (`nockchain`):**
- Location: `crates/nockchain/src/main.rs`.
- Triggers: Running `nockchain` binary.
- Responsibilities: Parse CLI, set tracing, initialize composed drivers, run `NockApp`.

**API Node Binary (`nockchain-api`):**
- Location: `crates/nockchain-api/src/main.rs`.
- Triggers: Running `nockchain-api` binary.
- Responsibilities: Start node runtime with API-focused defaults and gRPC config.

**Bridge Binary:**
- Location: `crates/bridge/src/main.rs`.
- Triggers: Running `bridge` binary.
- Responsibilities: Load bridge config, orchestrate signer/watcher/ingress/runtime tasks, run bridge drivers.

**Wallet CLI:**
- Location: `crates/nockchain-wallet/src/main.rs`.
- Triggers: Running `nockchain-wallet` binary.
- Responsibilities: Parse wallet command, optionally sync over gRPC, issue kernel commands via one-punch/file/markdown drivers.

**Explorer TUI:**
- Location: `crates/nockchain-explorer-tui/src/main.rs`.
- Triggers: Running `nockchain-explorer-tui` binary.
- Responsibilities: Connect to public gRPC services and render interactive chain explorer views.

## Error Handling

**Strategy:** Type-driven errors at crate boundaries, with runtime fallback to task shutdown/exit signals.

**Patterns:**
- Convert transport/protocol failures into crate-local error enums (`crates/bridge/src/errors.rs`, `crates/nockapp-grpc/src/error.rs`) and bubble with `?`.
- Use explicit process/runtime exit signaling through `NockAppExit` and `NockAppError::Exit` in `crates/nockapp/src/nockapp/mod.rs`.

## Cross-Cutting Concerns

**Logging:** Structured tracing with boot-time setup in `crates/nockapp/src/kernel/boot.rs`; bridge adds rotating file logs in `crates/bridge/src/tui.rs` and `crates/bridge/src/main.rs`.
**Validation:** CLI/config validation in per-app config modules (`crates/nockchain/src/config.rs`, `crates/bridge/src/config.rs`) before driver startup.
**Authentication:** No centralized auth layer inside runtime; trust boundaries are enforced by binding choices and endpoint role separation (`public` vs `private` gRPC drivers in `crates/nockapp-grpc/src/services/`).

---

*Architecture analysis: 2026-02-24*
