# SOL Derisking Context (nockchain-bench)

This file captures the agreed findings and the implementation order.
Re-read this file whenever context is compacted.

## Ranked Findings

1. Version-locked decode path likely causes `Expected cell, found atom`.
   - `speed_of_light/types.rs` decodes page/tx as fixed v0 layouts.
   - Used by extractor during archive creation, so v1/v2 shapes can fail.

2. Archive extraction is monolithic in-memory and can spike RSS badly.
   - `ArchiveWriter` accumulates all jam blobs before one final flush.

3. Mempool replay currently does full `writer.to_bytes()` roundtrip.
   - Causes transient large allocations and memory spikes.

4. `.soltest` fixture packaging is also monolithic and duplicates buffers.
   - Reads full checkpoint/archive/kernel and serializes one big payload.

5. Deserialization is largely unbounded.
   - Risk of huge allocations on malformed/truncated inputs.

6. Reader path copies full sections after loading full file.
   - Additional avoidable memory pressure.

7. Archive integrity/invariant checks can be stronger.
   - Missing explicit limits/integrity verification in key areas.

8. Unsafe noun-root access surface can be narrowed in bench crate.
   - Lower risk than above, but safety boundary can be tightened.

## Implementation Plan (strict order)

1. Make extraction/archive path version-agnostic and shallow.
2. Stream archive writing to avoid monolithic in-memory blobs.
3. Remove mempool replay `to_bytes()` roundtrip.
4. Stream/section `.soltest` fixture format.
5. Add strict deserialization/size limits.
6. Reduce archive reader memory copies (borrowed/mmap-like).
7. Add stronger archive invariants/integrity checks.
8. Narrow unsafe root access with safe helpers.

## Process Constraints

- Do not move to N+1 until N has tests passing.
- Add tests for each change.
- Make a separate commit per change.
