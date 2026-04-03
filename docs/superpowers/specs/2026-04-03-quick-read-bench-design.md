# Spec: `sol quick-read-bench` v1

## Status

Drafted and approved in-session on 2026-04-03 for initial investigation and
planning. This document defines the first `quick`-style read benchmark for
`nockchain-bench`. It does not authorize implementation beyond this scoped v1
design.

## 1. Purpose

Add a new `nockchain-bench` modality that measures read pressure against a
prepared source node.

Unlike `sol quick-bench`, which replays archived blocks inside one local
process, `sol quick-read-bench` measures a running `nockchain` node serving
direct page reads. The benchmark should answer:

- while serving a controlled stream of direct `%heavy-n` requests
- from a prepared on-disk state at nontrivial height
- how much read pressure does the source node incur
- and at what request/latency profile

This is intended first as a one-off operator tool for local investigation. The
shape of its requested inputs and outputs should remain compatible with a later
trusted harness and sweep integration.

## 2. v1 Summary

`sol quick-read-bench` launches one source node, waits for its private gRPC
endpoint to become ready, generates direct `%heavy-n` requests over a declared
height range in ascending order, discards returned pages after minimal success
validation, samples source-side process evidence, and emits machine-readable
artifacts summarizing the run.

There is no peer B process in v1.

The load generator is internal to `nockchain-bench`; it does not shell out to
`nockchain-peek`.

## 3. Scope

### 3.1 In Scope

- a new quick-style command named `sol quick-read-bench`
- launching a prepared source `nockchain` process under bench control
- probing private gRPC readiness before measurement starts
- generating direct `%heavy-n` requests in ascending-height order
- configurable request concurrency
- configurable target height range
- machine-readable run artifacts
- source-process read-pressure measurement using bench-side evidence only
- branch-agnostic execution so the same command can be run separately on master
  and PMA checkouts

### 3.2 Out of Scope

- trusted `sol bench` orchestration
- `sol sweep` integration
- reproducible cache-cold or cache-eviction protocol
- a second full `nockchain` node acting as peer B
- public explorer/API read paths such as `GetBlocks` or `GetBlockDetails`
- descending or random request schedules in v1
- node-internal instrumentation requirements
- automatic cross-branch comparison in one command

## 4. Why Direct `%heavy-n`

Three candidate designs were considered:

1. synthetic direct file reads
2. public API load generation
3. direct `%heavy-n` request generation

v1 chooses direct `%heavy-n` request generation because it stays closest to the
real low-level page-serving path while avoiding extra explorer/API caching,
decoding, and serialization costs. It also avoids entangling the benchmark with
write speed or sync behavior from a second node.

## 5. Operator Model

The operator provides a prepared source state for node A. `nockchain-bench`
owns the source node process lifecycle for the benchmark run.

The expected flow is:

1. start source node A from a prepared data directory
2. wait until private gRPC is reachable and responsive
3. issue a controlled stream of `%heavy-n` peeks over a requested height range
4. discard returned page payloads after minimal validation
5. sample source-process metrics during the run
6. stop the source process
7. write JSON artifacts summarizing the benchmark

This makes setup noise and measured work separable. The prepared source state
is an operator input, not something the benchmark provisions for itself.

## 6. Command Surface

v1 should introduce a new command under `sol`:

```text
nockchain-bench sol quick-read-bench ...
```

The exact CLI spelling can evolve during implementation, but the command should
cover the following requested inputs.

### 6.1 Required Inputs

- source binary path, or an explicit "use current workspace binary" default
- source data directory
- source gRPC address
- `start_height`
- `end_height`

### 6.2 Important Optional Inputs

- source startup arguments needed to bind locally and avoid default peers
- concurrency
- request-count limit, if v1 supports stopping before the full range is
  exhausted
- output path for JSON artifacts
- memory profiling flags and interval
- startup timeout / readiness timeout

### 6.3 Recommended Defaults

- schedule fixed to `ascending`
- one pass over the inclusive `[start_height, end_height]` range
- modest concurrency default suitable for local debugging
- machine-readable output always enabled, even when a console summary is also
  printed

## 7. Requested Workload Model

The implementation should keep the runtime schedule narrow in v1 but the
requested-case schema extensible.

### 7.1 v1 Workload

v1 workload fields should include:

- `start_height`
- `end_height`
- `schedule`
- `concurrency`

For v1:

- `schedule` only accepts `ascending`
- the generator walks the declared range in increasing height order
- each height is requested once unless an explicit future request-budget field
  is added

### 7.2 Future-Proofing Requirement

Even though v1 only supports `ascending`, the schema should still model
schedule as an enum rather than baking ascending order into unrelated fields.
This keeps the output compatible with later `descending` and `random` modes
without redesigning artifacts.

Likely future workload axes:

- schedule
- range width
- start position
- concurrency
- request budget
- random seed

## 8. Runtime Architecture

v1 consists of three logical components.

### 8.1 Source Launcher

Responsible for:

- building the source process command
- setting local bind addresses and peer-isolation options
- spawning the process
- capturing stdout/stderr
- stopping the process at the end of the run

The launcher should treat source startup as setup, not measured work.

### 8.2 gRPC Readiness Gate

Responsible for:

- repeatedly probing the configured private gRPC endpoint
- failing fast on timeout
- only allowing measurement to begin once the source process is ready

This gate must make benchmark startup failures distinguishable from failures
that happen after the measured phase begins.

### 8.3 Direct `%heavy-n` Load Generator

Responsible for:

- constructing direct `%heavy-n` peek requests
- dispatching them with configured concurrency
- recording request timing and outcome
- performing only minimal response validation
- discarding response payloads rather than accumulating them

The load generator exists to maximize read demand while minimizing client-side
extra work.

## 9. Measurement Model

### 9.1 Primary Metrics

The headline result is A-side read pressure, not a second node's sync speed.

Primary evidence should include:

- source-process minor page-fault delta
- source-process major page-fault delta
- source-process I/O counters such as `read_bytes` where available
- wall-clock benchmark duration

If the platform cannot provide one of these counters, the artifact should omit
it explicitly or mark it unavailable rather than silently fabricating a zero.

### 9.2 Supporting Metrics

Supporting service metrics should include:

- requests attempted
- requests succeeded
- requests failed
- request latency summary: min, p50, p90, p99, max
- achieved requests/sec
- achieved pages/sec

These help explain read pressure but are not the primary score.

### 9.3 Optional Memory Profile

If enabled, the command should also collect a memory timeline comparable in
spirit to existing quick benchmark memory profiling:

- RSS over time
- any other already-available bench-side memory samples

This remains secondary evidence for the read benchmark.

## 10. Artifact Model

Even though this is a quick command, v1 should emit machine-readable artifacts
instead of relying only on terminal output.

Recommended artifact set:

- `requested_case.json`
- `summary.json`
- `source_stdout.log`
- `source_stderr.log`
- raw process-counter samples or snapshots used to compute deltas
- optional memory-profile output when profiling is enabled

### 10.1 Requested Case

The requested case should capture:

- benchmark kind
- source binary identity request
- source data directory path
- source gRPC address
- startup arguments
- `start_height`
- `end_height`
- `schedule`
- `concurrency`
- profiling options

This should be modeled as a read-benchmark request, not folded into the
existing replay `RequestedCase`, because the semantics are materially different.

### 10.2 Summary

The summary should capture:

- measured duration
- aggregate request counts
- latency summary
- primary read-pressure counters
- benchmark validity
- environment/context fields needed for later manual comparison

### 10.3 Context Fields

The output should record enough identity to compare master and PMA runs later,
including:

- source git commit / binary identity
- source runtime flavor when known
- source data-dir identity supplied by the operator
- exact requested height range
- concurrency
- schedule enum

## 11. Validity And Failure Semantics

v1 should distinguish setup failure from measured-run degradation.

### 11.1 Pre-Measurement Failure

These should fail the benchmark before measurement begins:

- source process spawn failure
- source readiness timeout
- source exits before readiness

### 11.2 In-Run Failure

These should be preserved in artifacts:

- individual request failures
- source process exits during the run
- malformed responses
- measurement counter collection failures

The summary should include:

- failure counts
- representative error strings or samples
- a validity flag rather than silently presenting partial work as clean success

If failure rate exceeds an implementation-defined threshold, the run should be
marked invalid.

## 12. Cross-Branch Comparison Workflow

v1 remains branch-agnostic.

The intended comparison flow is:

1. run `sol quick-read-bench` in one checkout or build of master
2. run the same command with the same prepared source-state class in the PMA
   checkout after bench transplant
3. compare the two artifact sets externally

The command itself should not know about both branches simultaneously.

## 13. PMA Relationship

This design is deliberately compatible with later PMA comparison work.

The command should not assume replay fixtures or `quick-bench` checkpoint
cadence semantics. It should instead treat the source node as an external
runtime with a prepared data directory. That lets the same benchmark family
apply to:

- master source nodes
- PMA source nodes
- future runtime variants with different storage behavior

without changing the benchmark's conceptual model.

## 14. Path To Trusted Harness And Sweep

v1 does not add sweep support, but its data model should not block it.

Future trusted integration will likely need:

- a read-benchmark requested case distinct from replay cases
- trusted orchestration over repeated runs
- warmup/measured repetition policy
- verdict rules for partial or degraded runs
- sweep axes over range, concurrency, schedule, and runtime flavor

The important v1 design rule is:

- keep the command quick-only for now
- but serialize the request and summary in a way that can later slot into a
  trusted harness without a schema reset

## 15. Explicit Non-Goals For v1

The implementation should not try to solve these in the first cut:

- guaranteed cold-page conditioning
- host-level cache dropping
- cache-thrashing orchestration
- second-node sync simulation
- public explorer API benchmarking
- mixed workload schedules
- automatic comparison reports
- sweep execution

## 16. Open Questions Deferred

These are intentionally deferred beyond the approved v1:

- exact CLI naming for source-binary defaults vs explicit binary paths
- exact process-counter portability guarantees across host environments
- exact validity threshold for request-failure rate
- whether future trusted mode should use duration-based or count-based request
  budgets
- whether future schedule variants should include seeded random order

## 17. Acceptance Criteria For v1 Design

The design is satisfied if an implementation can:

- launch one prepared source node under bench control
- verify gRPC readiness before measurement
- issue direct `%heavy-n` requests over an ascending height range
- discard results after minimal validation
- capture source-side read-pressure counters
- emit machine-readable artifacts for later master vs PMA comparison

This is the complete approved scope for `sol quick-read-bench` v1.
