# Bench Tracing Design

## Summary

Add opt-in replay tracing controls to `nockchain-bench` for SOL replay
workflows. The crate should support:

- Nock interpreter tracing via `--nock-tracing off|on`
- Tracy export control via `--tracy off|all|nockcode`
- optional Nock trace filters:
  - `--nock-tracing-keyword-filter`
  - `--nock-tracing-interval-filter`

These settings apply to replay-oriented commands only:

- `sol quick-bench`
- `sol bench`
- hidden `sol run-once`
- `sol sweep`

The key design decision is that both Nock tracing and Tracy are
invocation-global. For a sweep, they apply to the whole invocation and may be
configured in `base`, but they are not valid sweep axes and are not part of
per-case benchmark identity.

## Goals

- Make Tracy usable from `nockchain-bench` replay commands.
- Make Nock interpreter tracing usable from `nockchain-bench` replay commands.
- Keep sweep comparisons coherent by requiring one tracing configuration for the
  whole matrix.
- Record the active tracing configuration in trusted artifacts so operators can
  tell exactly how a run was produced.
- Improve Tracy readability by enabling Rust demangling whenever Tracy support
  is compiled in.

## Non-Goals

- Making Tracy mode a per-case or per-axis sweep setting.
- Making Nock tracing a per-case or per-axis sweep setting.
- Adding tracing controls to non-replay flows such as `sol extract`,
  `sol fixture build`, or checkpoint derivation commands outside the replay
  path.
- Adding a separate user-facing demangling flag.
- Changing the semantics of existing memory profiling flags such as
  `--profile-memory`.

## Current Situation

Today `nockchain-bench` boots a `NockApp` through
`speed_of_light::kernel_utils::init_nockapp(...)`, but it hardcodes
`TraceOpts::default()`, so Nock tracing is always disabled in bench replay.

Separately, `nockchain-bench` does not initialize a Tracy subscriber, so Tracy
is not available from the bench binary even though `nockchain` supports it.

The trusted sweep model currently stores only per-case benchmark data in
`RequestedCase` / `ResolvedCase`. There is no durable slot for invocation-global
tracing configuration today.

## Terminology

- `nock tracing`: the special Nock/Hoon interpreter span stream produced by
  `nockvm` when tracing is enabled
- `Tracy`: the profiler sink/subscriber that consumes tracing spans and, when
  permitted by the OS, native stack samples

These are related but distinct:

- `--nock-tracing` controls whether the special interpreter spans are created
- `--tracy` controls whether and how spans are exported to Tracy

## Configuration Model

### Invocation-global tracing config

Introduce an explicit invocation-global tracing model, separate from
`RequestedCase` and `ResolvedCase`.

Recommended shape:

- `nock_tracing: bool`
- `nock_tracing_keyword_filter: Option<String>`
- `nock_tracing_interval_filter: Option<usize>`
- `tracy: TracyMode`

Where `TracyMode` is:

- `off`
- `all`
- `nockcode`

This config applies to:

- the whole `sol quick-bench` invocation
- the whole `sol bench` invocation
- the whole `sol sweep` invocation
- each hidden `sol run-once` worker process

### Why it is not part of `RequestedCase`

Tracing is intentionally not a benchmark dimension for this feature. The user
requirement is that a sweep runs either with tracing enabled for the whole
matrix or not at all. That means tracing should not live in case identity or
comparison invariants alongside `threads`, `blocks`, `profile_memory`, and
other benchmark knobs.

## User-Facing CLI

### `sol quick-bench`

Add:

- `--nock-tracing off|on`
- `--nock-tracing-keyword-filter <csv>`
- `--nock-tracing-interval-filter <n>`
- `--tracy off|all|nockcode`

Defaults:

- `--nock-tracing off`
- `--tracy off`

Validation:

- `--nock-tracing-keyword-filter` requires `--nock-tracing on`
- `--nock-tracing-interval-filter` requires `--nock-tracing on`
- `--nock-tracing-interval-filter` must be positive

### `sol bench`

Add the same flags and defaults as `sol quick-bench`.

### Hidden `sol run-once`

Do not add ad hoc tracing CLI flags here. `sol run-once` remains
machine-oriented and receives tracing settings through a machine-readable
runtime config file.

## Sweep Configuration

### `base`

Allow these new fields in sweep `base`:

- `nock_tracing`: `"off"` or `"on"`
- `nock_tracing_keyword_filter`: string, optional
- `nock_tracing_interval_filter`: integer, optional
- `tracy`: `"off"`, `"all"`, or `"nockcode"`

Defaults:

- `nock_tracing = "off"`
- `tracy = "off"`

Validation:

- `nock_tracing_keyword_filter` requires `nock_tracing = "on"`
- `nock_tracing_interval_filter` requires `nock_tracing = "on"`
- `nock_tracing_interval_filter` must be positive

### `axes`

Reject the following axis names:

- `nock_tracing`
- `nock_tracing_keyword_filter`
- `nock_tracing_interval_filter`
- `tracy`

Error message should clearly state that tracing configuration is invocation-wide
for a sweep and may only be set in `base`, not in `axes`.

### Parsing model

Because the current sweep model collapses `base` directly into `RequestedCase`,
the spec-style `base` parser must be extended to produce two outputs:

- `base_case: RequestedCase`
- `tracing: InvocationTracingConfig`

That can be represented by introducing a new sweep-level structure rather than
trying to force tracing into `RequestedCase`.

## Machine Transport

### Hidden runtime config

Introduce a hidden machine-readable runtime config file, for example:

- `input/runtime_config.json`

Recommended contents:

- `nock_tracing`
- `nock_tracing_keyword_filter`
- `nock_tracing_interval_filter`
- `tracy`

### Hidden `sol run-once` contract

Extend hidden `sol run-once` to accept an additional machine-only input:

- `--runtime-config <path>`

This allows both native subprocess execution, if added later, and current
Docker replay workers to receive the same invocation-global tracing config
without turning it into a case-level field.

### Docker transport

Trusted Docker runs currently write only `input/resolved_case.json` before
invoking `docker exec ... nockchain-bench sol run-once ...`.

The Docker path must also write:

- `input/runtime_config.json`

and invoke:

- `nockchain-bench sol run-once --resolved-case ... --runtime-config ...`

Without this second channel, Tracy and Nock tracing do not reliably reach the
actual replay worker process in Docker mode.

## Runtime Behavior

### Nock tracing

When `nock_tracing` is enabled, replay boot should construct
`nockapp::kernel::boot::TraceOpts` with:

- tracing mode enabled
- optional keyword filter
- optional interval filter

This should be threaded into the replay `NockApp` boot path instead of using
`TraceOpts::default()`.

Filter semantics must match current `nockapp` behavior exactly:

- keyword filters are split on commas
- tokens are not trimmed automatically
- empty tokens are not silently normalized away unless implementation work
  deliberately changes shared behavior across both binaries
- when both keyword and interval filters are supplied, they are composed with
  logical OR, not AND

Bench should follow existing `nockapp` semantics unless there is a deliberate,
separate decision to change `nockapp` itself.

### Tracy

`nockchain-bench` should initialize a tracing subscriber from the invocation
runtime config when `tracy != off`.

Modes:

- `off`: do not install Tracy
- `all`: install Tracy and export all spans observed by the subscriber
- `nockcode`: install Tracy with a filter that forwards only `target ==
  "nockcode"`

Interaction with Nock tracing:

- `--nock-tracing off --tracy all`
  - Tracy sees ordinary Rust spans only
- `--nock-tracing on --tracy all`
  - Tracy sees ordinary Rust spans plus Nock spans
- `--nock-tracing on --tracy nockcode`
  - Tracy sees only Nock spans
- `--nock-tracing on --tracy off`
  - Nock tracing still exists internally, but no Tracy sink is active

The last combination is allowed for semantic consistency and future-proofing,
even if it is not usually useful in practice.

### Backend-specific behavior

Tracy process scope differs by backend and must be documented explicitly:

- Native quick-bench / bench / sweep:
  - one `nockchain-bench` process owns the invocation-global Tracy setup
- Docker trusted replay:
  - replay runs happen in hidden `sol run-once` worker processes inside the
    container, so Tracy capture is effectively per worker process, even though
    the configuration is still invocation-global from the operator’s point of
    view

The spec should not describe Tracy as one uniform process lifetime across all
backends.

## Demangling

Enable Rust-aware Tracy demangling whenever Tracy support is compiled in.

This is an explicit implementation requirement, not an incidental default.

Rationale:

- demangling improves native stack/sample readability in Tracy
- it does not affect Nock span naming, which is already human-readable
- it does not need to vary by case, command, or matrix
- a separate CLI flag would add complexity with little operator value

No user-facing demangling flag should be added.

## Provenance And Artifacts

### Requested/resolved case

Do not store invocation-global tracing settings in `requested_case.json` or
`resolved_case.json`.

Those files remain benchmark-case records.

### Runtime config artifact

Persist the invocation-global tracing config in a dedicated artifact, for
example:

- `runtime_config.json`

at the trusted run root.

### Provenance

Tracing facts must be recorded in provenance, not optionally but always.

Recommended fields:

- `tracing.nock_tracing`
- `tracing.nock_tracing_keyword_filter`
- `tracing.nock_tracing_interval_filter`
- `tracing.tracy_mode`
- `tracing.tracy_compiled`
- `tracing.demangling_enabled`

This is required so operators can distinguish:

- tracing disabled
- tracing enabled
- Tracy compiled out
- Tracy enabled with demangling

Tracing provenance is for operator visibility only. It does not affect validity
or benchmark identity unless a later design explicitly decides otherwise.

## Comparison Semantics

Tracing configuration must not silently vary within a sweep.

Because tracing-related fields are forbidden in `axes`, all expanded cases
inherit the same invocation-global tracing config from `base`.

Even so, the comparison layer still needs explicit changes:

- tracing-related axis names must be rejected during parse/expand
- sweep comparison code must not assume no new invariant work is needed
- any case-level fields added in the course of implementation must be wired into
  invariant enforcement deliberately rather than relying on serialization alone

Because the intended model is fully invocation-global tracing, the cleanest
outcome is to keep tracing out of `RequestedCase` entirely and record it through
runtime config plus provenance.

## Validation Rules

- `nock_tracing = off` with either filter set is invalid
- `nock_tracing = on` with no filters is valid
- `nock_tracing_interval_filter` must be positive
- tracing-related fields in sweep `axes` are invalid
- `tracy` must be one of `off`, `all`, `nockcode`

## Testing

### CLI parsing

Add tests for:

- `sol quick-bench` parsing all new flags
- `sol bench` parsing all new flags
- hidden `sol run-once` parsing the machine-only runtime config path

### Sweep parsing and expansion

Add tests for:

- `base` accepting the new tracing fields
- `axes` rejecting each tracing-related axis name
- parsed sweep config splitting into `base_case` plus invocation-global tracing
  config

### Runtime config transport

Add tests for:

- hidden runtime config JSON serialization/deserialization
- Docker prepare path writing `input/runtime_config.json`
- Docker replay invocation passing `--runtime-config`

### Runtime wiring

Add focused tests around replay boot to verify:

- `nock_tracing = off` maps to disabled/default `TraceOpts`
- `nock_tracing = on` maps to enabled `TraceOpts`
- keyword and interval filters are forwarded correctly
- combined keyword+interval filters preserve current OR semantics

### Provenance

Add tests for:

- provenance always recording tracing facts
- trusted artifacts containing `runtime_config.json`

### Documentation

Update:

- `crates/nockchain-bench/README.md`
- `crates/nockchain-bench/specs/bench-harness-spec.md`
- CLI help text in `crates/nockchain-bench/src/main.rs`

## Recommended Implementation Notes

- Reuse `nockapp`'s existing trace option model internally rather than creating
  a second Nock trace representation for replay boot.
- Introduce a distinct invocation-global tracing model rather than overloading
  `RequestedCase`.
- Keep Tracy bootstrap code in one place inside `nockchain-bench` startup so
  quick/trusted/sweep invocations all follow the same runtime-config path.
- Do not make Tracy or Nock tracing part of sweep case naming.
