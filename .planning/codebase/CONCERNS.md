# Codebase Concerns

**Analysis Date:** 2026-02-24

## Tech Debt

**Bridge ingress authorization and network hardening:**
- Issue: Signature acceptance and stop-broadcast handling are intentionally permissive; signer authorization is still marked TODO and stop broadcasts are accepted after lossy decode.
- Files: `crates/bridge/src/ingress.rs`, `crates/bridge/src/config.rs`
- Impact: A peer can drive noisy invalid signature traffic and trigger stop behavior without a strict authorization gate.
- Fix approach: Enforce allowlisted signer/node checks before accepting signatures or stop requests; reject malformed hashes instead of lossy normalization.

**Networking layer coupling in libp2p driver:**
- Issue: The driver declares a placeholder wire and documents that libp2p handling is entangled with unrelated nockchain pokes.
- Files: `crates/nockchain-libp2p-io/src/driver.rs`
- Impact: Cross-cutting coupling increases regression risk when changing networking or kernel I/O behavior.
- Fix approach: Split wire types and request/effect handlers by responsibility (gossip, request/response, timer, control), then enforce explicit interface boundaries.

**Legacy pubkey mining compatibility path still active:**
- Issue: Mining still injects hardcoded v0 compatibility config while newer PKH paths are used.
- Files: `crates/nockchain/src/mining.rs`, `crates/nockchain/src/lib.rs`, `crates/nockchain/src/setup.rs`
- Impact: Dual-path mining behavior raises maintenance burden and can hide edge-case misconfiguration.
- Fix approach: Remove v0 pubkey fallback after migration checkpoint and keep only the PKH-based mining configuration path.

## Known Bugs

**Set jet does not deduplicate identical elements:**
- Symptoms: Insertion path for set operations is explicitly flagged as incorrect for duplicate handling.
- Files: `crates/nockvm/rust/nockvm/src/jets/set.rs`
- Trigger: Calling `jet_put` with values already present in set-shaped trees.
- Workaround: Avoid relying on jet-level dedup correctness for affected paths; validate dedup invariants in higher-level logic/tests.

**Mink stack-trace behavior mismatch in tests:**
- Symptoms: Two mink tests remain ignored with documented stack-trace format mismatch.
- Files: `crates/nockvm/rust/nockvm/src/jets/nock.rs`
- Trigger: Running ignored tests around `test_mink_zapzap` and `test_mink_trace`.
- Workaround: Keep tests ignored and validate nearby behavior through passing mink tests until trace formatting is aligned.

## Security Considerations

**Bridge stop endpoint trust model is weak:**
- Risk: `broadcast_stop` accepts peer-provided stop messages and always returns accepted, even when hash inputs are malformed and decoded lossy.
- Files: `crates/bridge/src/ingress.rs`
- Current mitigation: Logging plus optional last-block parsing.
- Recommendations: Add sender authentication/authorization, strict hash-length validation, replay protection, and reject-on-parse-failure semantics.

**Experimental/unaudited software posture:**
- Risk: Project-level statement declares the software experimental and unaudited.
- Files: `README.md`
- Current mitigation: Explicit warning in repository documentation.
- Recommendations: Track threat model and formal audit scope per subsystem (`bridge`, `nockvm`, `wallet`, `libp2p`) before production-critical usage.

## Performance Bottlenecks

**Linear note lookup in wallet Hoon path:**
- Problem: Note lookup by hash iterates over full note map with TODO comment saying it is "way too slow".
- Files: `hoon/apps/wallet/lib/utils.hoon`
- Cause: O(n) scan via `find-name-by-hash` against tapped map entries.
- Improvement path: Maintain reverse index `hash -> note-name` at write-time and perform direct map lookup.

**Known memory issues in bigint implementation:**
- Problem: Multiple `FIXME` markers call out memory leaks and invalid assumptions in `ibig` internals.
- Files: `crates/nockvm/rust/ibig/src/num_traits.rs`, `crates/nockvm/rust/ibig/src/buffer.rs`, `crates/nockvm/rust/ibig/src/ubig.rs`
- Cause: Current power/vec assumptions in bigint internals.
- Improvement path: Audit allocator behavior end-to-end, replace leak-prone operations, and gate risky paths behind targeted property tests/benchmarks.

## Fragile Areas

**Unsafe memory stack operations with incomplete bounds checks:**
- Files: `crates/nockvm/rust/nockvm/src/mem.rs`
- Why fragile: Core slot-pointer methods are marked TODO for missing simple bounds checks while operating inside unsafe memory primitives.
- Safe modification: Add explicit checked arithmetic and offset guards first, then refactor callsites gradually behind tests.
- Test coverage: Gaps exist where ignored/disabled tests are still present in nearby low-level subsystems (`crates/nockvm/rust/nockvm/src/serialization.rs`, `crates/nockvm/rust/nockvm/src/jets/nock.rs`).

**Checkpoint serialization path contains documented footgun:**
- Files: `crates/nockapp/src/kernel/form.rs`
- Why fragile: Comment marks cold-state noun copying semantics as a footgun in checkpoint creation.
- Safe modification: Isolate cold-state encoding/copy rules into a dedicated API with ownership guarantees and invariants tested at boundary.
- Test coverage: No focused regression test file is colocated with this footgun marker.

## Scaling Limits

**Bridge signer topology is fixed by defaults:**
- Current capacity: Defaults assume `min_signers=3` and `total_signers=5`.
- Files: `hoon/apps/bridge/types.hoon`, `crates/bridge/src/config.rs`
- Limit: Coordination, quorum, and config semantics are tuned for a small fixed validator set.
- Scaling path: Introduce explicit dynamic signer-set management and runtime reconfiguration workflow.

**Bridge activation thresholds still partially hardcoded/TODO:**
- Current capacity: Start-height and acceptance cutoffs rely on defaults with TODO markers for final cutoff values.
- Files: `hoon/apps/bridge/types.hoon`, `crates/bridge/src/config.rs`
- Limit: Misaligned network activation parameters can block intended deposits or accept too early.
- Scaling path: Move activation constants to audited release config and validate via startup invariants.

## Dependencies at Risk

**Critical dependencies pinned to git revisions:**
- Risk: Workspace relies on git-sourced dependencies rather than crates.io releases for key tooling/runtime components.
- Files: `Cargo.toml`
- Impact: Upstream force-push/repo churn or API drift can break reproducibility and upgrade planning.
- Migration plan: Mirror/fork critical git dependencies under controlled ownership or migrate to stable released versions where available.

## Missing Critical Features

**Bridge withdrawals are intentionally incomplete:**
- Problem: Withdrawal handling is explicitly deferred; several code paths no-op, stop, or carry TODO placeholders.
- Files: `hoon/apps/bridge/base.hoon`, `hoon/apps/bridge/nock.hoon`, `hoon/apps/bridge/types.hoon`
- Blocks: End-to-end withdraw proposal/execution lifecycle and full settlement handling.

**Protocol/version negotiation for gossip remains TODO:**
- Problem: Gossip handling skips strict version negotiation and compatibility rejection.
- Files: `crates/nockchain-libp2p-io/src/driver.rs`
- Blocks: Safe rolling upgrades and strict cross-version behavior guarantees.

## Test Coverage Gaps

**Wallet hot-path command tests are mostly ignored/stubbed:**
- What's not tested: Import keys, spend format flows, draft transaction flows, and show-tx hot path behavior.
- Files: `crates/nockchain-wallet/src/main.rs`
- Risk: CLI and transaction workflows can regress without CI detection.
- Priority: High

**Bridge gRPC pagination scenario has unfinished test:**
- What's not tested: Balance-by-first-name cache behavior across subsequent pages remains incomplete.
- Files: `crates/nockapp-grpc/src/services/public_nockchain/v2/server.rs`
- Risk: Regressions in pagination/caching semantics can slip into production APIs.
- Priority: Medium

**Core runtime stress/trace tests are ignored:**
- What's not tested: Long-running sync poke/peek tracing and some low-level nondeterministic/trace error cases.
- Files: `crates/nockapp/tests/integration.rs`, `crates/nockvm/rust/nockvm/src/serialization.rs`, `crates/nockvm/rust/nockvm/src/jets/nock.rs`
- Risk: Runtime edge-case failures remain latent under load or malformed input.
- Priority: Medium

---

*Concerns audit: 2026-02-24*
