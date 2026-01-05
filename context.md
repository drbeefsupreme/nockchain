# Instructions

We recently finished most of Phase 1 of Milestone 2 in nock-pma.md, see below.

Our next step is to wire the PMA into the Nockchain event loop. That is, we want
it so that when the Nockchain kernel receives a poke, the Nockvm processes the
poke, ends with a top frame, and the relevant data in that top frame is copied
into the PMA via the PmaCopy trait. Left behind in the NockStack are pointers to
the PMA. I believe these are then to be copied to the other side of the
NockStack, polarity is flipped, and its ready for the next event, but I'm not
entirely sure. You'll need to figure that out.

I'd like for you to present 2-3 options on how to wire things up and we'll
evaluate them from there. Ultimately, we want Nockchain to be able to run one
event without falling over (though probably two will be needed to really test
it). Really though, we should think about it as a NockApp that adds in the PMA
as a feature, not something specific to Nockchain.

Below you will find some relevant files.


## Phase 2 Options: Wiring PMA into the Event Loop

These are the options the LLM considered for integrating the PMA into the Nockchain
event loop. This branch tries to implement Option 1 as a minimal change to just
try things out.

### Current Event Loop Flow (in `Serf`)

1. `do_poke(job)` executes the poke via `self.soft()` → `self.slam()`
2. On success: `event_update(eve+1, new_arvo)` updates state
3. `stack.preserve(&mut fec)` preserves effects to other side
4. `preserve_event_update_leftovers()`:
   - Preserves `warm`, `test_jets`, `hot`, `cache`, `cold`, `arvo`
   - Calls `flip_top_frame(0)` to flip polarity
   - Calls `retag_survivors()` to convert stack pointers → offset form

**Goal**: Replace/augment the preservation step so survivors go to the PMA instead of (or in addition to) being preserved on the NockStack.

### Option 1: Replace `preserve` with `copy_to_pma` + Stack Reset

**Approach**: After event processing, copy all survivors to the PMA using `PmaCopy`, then reset the NockStack entirely.

```rust
// In Serf, add:
pma: Pma,  // owned or Arc<Mutex<Pma>>

// Replace preserve_event_update_leftovers with:
pub unsafe fn persist_event_to_pma(&mut self) {
    let stack = &mut self.context.stack;
    let pma = &mut self.pma;
    let _guard = pma.install();  // Install PMA arena for offset resolution

    // Copy survivors to PMA (converts to offset form automatically)
    self.context.warm.copy_to_pma(stack, pma);
    self.context.hot.copy_to_pma(stack, pma);
    self.context.cache.copy_to_pma(stack, pma);
    self.context.cold.copy_to_pma(stack, pma);
    self.arvo.copy_to_pma(stack, pma);

    // Reset NockStack completely - no preserve/flip needed
    // since all live data is now in PMA
    stack.reset();
}
```

**Pros**:
- Clean separation: NockStack is purely ephemeral working memory
- Simpler stack logic (no flip/retag dance needed)
- Natural persistence: PMA automatically contains the durable state

**Cons**:
- All surviving data goes to PMA every event (PMA grows faster)
- Bigger change surface
- Need to implement `PmaCopy` for `Warm`, `Hot`, `Cold`, `Cache`

### Option 2: Layer PMA Persistence on Top of Existing Preserve

**Approach**: Keep the existing preserve/flip mechanism, but after `retag_survivors()`, copy the already-offset-form data to the PMA.

```rust
pub unsafe fn preserve_event_update_leftovers(&mut self) {
    let stack = &mut self.context.stack;

    // Existing preserve calls
    stack.preserve(&mut self.context.warm);
    stack.preserve(&mut self.context.hot);
    stack.preserve(&mut self.context.cache);
    stack.preserve(&mut self.context.cold);
    stack.preserve(&mut self.arvo);
    stack.flip_top_frame(0);
    self.retag_survivors();  // Now in offset form in NockStack's arena

    // NEW: Copy to PMA for durable persistence
    if let Some(pma) = &mut self.pma {
        let _guard = pma.install();
        self.arvo.copy_to_pma(stack, pma);
        self.context.cold.copy_to_pma(stack, pma);
        // ... etc
    }
}
```

**Pros**:
- Minimal disruption to existing flow
- Existing tests still work
- PMA is optional - can be `Option<Pma>`

**Cons**:
- Duplicate work: preserve to stack, then copy to PMA
- Two representations exist simultaneously (stack offsets + PMA offsets)
- More complex mental model

### Option 3: NockApp Trait Extension with Optional PMA

**Approach**: Create a trait that NockApps can implement to opt into PMA persistence. PMA becomes a feature, not a requirement.

```rust
// In nockapp crate:
pub trait Persistent {
    fn persist_state(&mut self, pma: &mut Pma);
    fn restore_state(&mut self, pma: &Pma);
}

// Serf becomes generic over persistence strategy
pub struct Serf<P: Persistence = NoPersistence> {
    // ... existing fields
    persistence: P,
}

pub trait Persistence {
    fn after_event(&mut self, serf: &mut SerfCore);
}

pub struct PmaPersistence {
    pma: Pma,
}

impl Persistence for PmaPersistence {
    fn after_event(&mut self, serf: &mut SerfCore) {
        unsafe {
            let _guard = self.pma.install();
            serf.arvo.copy_to_pma(&serf.context.stack, &mut self.pma);
            // ... copy other survivors
        }
    }
}
```

**Pros**:
- PMA is opt-in per NockApp
- Clean abstraction boundary
- Easy to test with/without persistence
- Nockchain can enable PMA, other apps don't have to

**Cons**:
- More abstraction overhead
- Needs Serf refactoring to separate core logic

### Decision: Start with Option 1

We're implementing Option 1 first as a test to see if things work. Baby steps done with steppe wisdom. Once Option 1 is working, we can consider abstracting over NockApps (Option 3).


------------------

## crates/nockvm/docs/nock-pma.md
````
# Current status

The live nockvm still runs on the contiguous arena defined in `open/crates/nockvm/rust/nockvm/src/mem.rs`. That module owns the `Memory` abstraction (wrapping either `memmap2::MmapMut` or `malloc`) and the `NockStack`, a single slab that tracks `frame_offset`, `stack_offset`, and `alloc_offset` as word counts off of a base pointer. Every noun the VM manipulates lives inside that slab until it is explicitly copied into a PMA image.

- `Memory::allocate` chooses mmap vs. malloc and hands back a base pointer that stays immutable for the life of the VM; all in-VM pointers today are literal `base + offset` derivations performed via `derive_ptr()`, not tagged offsets.

- `NockStack` models both stack frames and bump-allocation via its `west`/`east` orientation flag, the `AllocationType` enum, and the `pc` (pre-copy) bit that gates when frame flipping or preservation can occur. The reserved slots at the bottom of each frame cache the previous frame/stack/alloc pointers so that copying collectors and frame pops can restore provenance without chasing raw pointers.

- `open/crates/nockvm/rust/nockvm/src/noun.rs` consumes the stack API through `NounAllocator`, layering
the tag scheme (direct vs. indirect atoms vs. cells) and the forwarding-pointer rules that keep
structural sharing intact while slabs are copied between frames or into the PMA. Helper modules such
as `jets.rs` and `flog.rs` lean on `Preserve`/`preserve_with` from `mem.rs` to ensure nouns stay pinned
during host callbacks.

- We have not yet switched the runtime over to offset-tagged references: any noun reloaded from a
persisted PMA still has to be patched up by rerunning `derive_ptr()` with the process-local base pointer.

## A Young System's Programmer's Primer

1. Read [https://doc.rust-lang.org/nomicon/](https://doc.rust-lang.org/nomicon/) and [https://blog.regehr.org/archives/213](https://blog.regehr.org/archives/213)
2. Meditate on the most vivid possible meaning of the "nasal demons" metaphor for undefined behavior and let it put the fear of God in you
3. Miri is enabled on every test except those it absolutely cannot be made to work for (hi tokio, ffi). If the test executes too slowly in Miri, your test is too slow. Make it faster or more "targeted" to the code coverage you need from Miri.

Relevant history:

## Epoch History

1. 2025-03-28: PR #1167, titled “Offsets, not aliasing” and authored by Chris Allen (`@bitemyapp`).
  - This branch (commit e4adb5a8c, 2025‑03‑28) is where the NockStack struct stopped storing live frame/stack/alloc raw pointers and instead began recording `frame_offset`, `stack_offset`, and `alloc_offset` word counts from the slab’s base pointer. The change also introduced `derive_ptr()`/`frame_pointer()` helpers so every access reconstructs a pointer from the base plus offset, and `MemoryState` now snapshots offsets instead of raw pointers (see history at commit e4adb5a8c affecting `open/crates/nockvm/rust/nockvm/src/mem.rs`)
2. 2025‑05‑19 · commit 00d288b1 · PR #1554 “Incremental hierarchy for hoonc”
  - Focused on reducing allocator overhead when running hoonc by (a) allowing builds to short-circuit
  OOM checks via a `no_check_oom` feature, (b) dropping the expensive assert_no_alloc::permit_alloc
  scaffolding around pointer validity checks, and (c) rewriting prev_alloc_offset() to use a single
  wrapping_sub instead of branching on the base pointer. Also simplified frame_pop’s null-pointer
  panic to avoid heap allocations. Net effect: the stack allocator became leaner and more predictable
  under hoonc's incremental compile workload.
3. 2025‑05‑26 · commit a61d3289 · PR #1664 “Least space metric”
  - Added the `least_space` field to `NockStack`, threaded it through initialization, resets, and frame
  flips, and updated both `west`/`east` allocation paths to maintain a running low-water mark. Exposed
  a `least_space()` accessor so the runtime could export a gauge of minimum free words/bytes, enabling
  Slam telemetry to flag stacks that are close to exhaustion.
4. 2025‑06‑27 · commit d01347fd · branch “test jets vs hoon” (squash merge)
  - Extended the `Preserve` trait with a trivial implementation for `()`, which let the jets test harness
  reuse preservation APIs without manufacturing dummy nouns. Small change, but it marked the first
  divergence where preservation logic needed to tolerate no-op placeholders.
5. 2025‑07‑01 · commit 0013e50e · branch “Fix rust formatting in open/”
  - Pure rustfmt/rust-analyzer cleanup of `mem.rs`: reordered use statements into the standard blocks
  (std → crates → local) so the file complied with workspace formatting rules. No behavioral changes,
  but it stabilized future diffs for readability.
6. 2025‑07‑24 · commit b6ebdc7a · branch “Tracing backends integration”
  - Touched `frame_pop` and the debugging walkers to move from format!-style placeholders to the new
  Rust inline formatting (`{ptr:p}`). This kept panic/log strings allocation-free and aligned with the
  tracing backend expectations while keeping the underlying mechanics unchanged.
7. 2025‑09‑24 · commit 68b40a80 · branch “gRPC public API / light wallet”
  - Updated the `NounAllocator` for `NockStack` impl so that callers using the allocator through the trait could invoke a new `equals()` hook. Under the hood it forwards to `crate::unifying_equality::unifying_equality`, ensuring components like the light-wallet gRPC service can compare nouns without downcasting to `NockStack`.
8. 2025‑10‑06 · commit c809688f · branch “hoonc benchmarking and prewarm best result”
  - Largest post-introduction refactor:
    * Marked `word_size_of` as #[inline] and pulled in `Vec` to support a heap-based worklist.
    * Promoted `frame_push` to pub fn and inlined pop/top helpers for tighter hot-path codegen.
    * Replaced the heavyweight NockStack::copy method (which reused the lightweight stack as a
    worklist) with a new noun_preserve free function that uses a Vec<(Noun, *mut Noun)>. The new
    routine bails early when the root is already direct, already forwarded, or outside the current
    frame, dramatically reducing the amount of stack flipping during hoonc prewarm. Preservation
    invariants (assert_acyclic, assert_no_forwarding_pointers, assert_no_junior_pointers) still bookend
    the operation, but the worklist logic now lives entirely off-stack, improving determinism when the
    allocator is hot.
    * There was recently a weird interaction between the `axis` vs. `axis.form` issue and hoonc's prewarm bootstrap, at time of writing I'm not totally clear on how the dust settled there but I don't remember Logan saying prewarm in-and-of-itself was implicated, just flagging a risk.

This foregoing history exemplifies the recent substantive architecture epochs for `mem.rs`:
- initial migration from sword
- partial offset-ification of NockStack indexes into the slab (cf. `e4adb5a8c`)
- perf hardening for hoonc
- observability of stack pressure
- and the more recent noun-preservation rewrite that decouples maintenance work from the lightweight stack. - Subsequent commits are mostly ergonomic or integration tweaks layered on that foundation.

# Tooling, debugging, profiling

Make sure your build and test/validation entrypoints you use to iterate on your work are batch-executable (meaning: not daemonized/persistent) Makefile entrypoints that "just work" out of the box with no additional steps required before-hand to make them complete successfully.

- Memory safety, segfaults, use-after-frees, etc.:

  * ASAN on Linux (ask `@bitemyapp` how but the `nada` Makefile has breadcrumbs of me doing this)
  * guard malloc on macOS ^^

- Memory leaks:

  * For Linux, I recommend `bytehound`. Same suggestions as the previous, there are breadcrumbs but ask me how. You'll probably need to flip the `Memory` type to using an ephemeral malloc if the problem you are diagnosing implicates the slab. You will think you have alternatives to `bytehound` for diagnosing leaks on Linux and I will be very surprised if that's true. You'll probably waste your time trying to find something better, I was not successful after many hours. If you find something nicer or better maintained please let me know.

  * macOS: Just use `cargo-instruments` and XCode. Frankly easier than Linux but might be slightly less informative/clear/precise than `bytehound` depending on your circumstances. Seems to work great for Mitchell Hashimoto across the board, idk man. I need to spend more time on it.

- Performance:
  * Cheap and cheerful, works for Linux and macOS, samples native runtime stacks: `samply record make run-my-benchmark-or-whatever`, you'll have to clear detritus threads from Cargo in the Firefox Profiler tab that spawns if the benchmark wasn't already built but the actual benchmark threads should be in there somewhere regardless.
  * _There is no legacy tracing JSON profiling for Nock in NockVM_. If there is, I simply forgot to merge the branch deleting it. If it exists, delete it. Please don't use it. Tracy subsumes any need for this and it was wasting bytes and developer time.
  * There, however, is _tracing for Nock in NockVM_: use tracy. [watch my youtube video](https://www.youtube.com/watch?v=Z1UA0SzZd6Q)
  * Linux only unified nock + native stack profiling: tracy profiler again. make sure you align the locally compiled version of the tracy profiler GUI and the library version in the Cargo project. Look at Nada's makefile.
    - `macOS` works with Tracy fine, client and server-side but you're going to get the nock traces by default, native (20 khz!) stack samples in Tracy only work on Windows and Linux. No, I don't know why they refuse to support it on macOS. Because they definitely could. They just choose not to. This has a solution: _use Docker_ (do I need to say it again? Look at the Makefile targets and Dockerfile I wrote for this)
    - I would strongly encourage you to take advantage of `tracy`'s `ondemand` mode (look at the Cargo features specified for `tracing-tracy`) so that you aren't eating the profiling overhead when the nockapp first boots and loads the slab, but I won't blame you if that's more faffing about than you have patience for.

# Writing tests gooder

This is all speaking to Rust norms and structural conventions. I don't care what Uncle Bob thinks an integration test is. Don't tell me, I don't want to know either.

## Unit tests live in the library modules

## Integration tests live in separate binaries

You see all the Rust test cases in `tests/` sub-directories? That's what makes them an integration test. Importantly, _you can have multiple test cases in a single integration test binary_. Too many integration test binaries increase linker surface area, please don't exacerbate that.

Reasons you'd use an integration test:
# Milestone 1: Offset-addressing

You need to have a pointer representation that can be used from the Nock code which addresses other objects as offsets from a static (not constant) base pointer (base address).

Some of this work happened in Chris's earlier offset branch, but it isn't complete and we're still leaning heavily on `derive_ptr()` because Chris didn't want to add a new tag bit or churn the rest of the runtime. That time has passed and we need to rip the bandaid off and finish the other 80% of the work now to set the stage for position-independent addressing for a persistent mmap slab.

We're going to mmap the PMA, let the system decide where the map the PMA, let the system decide the base address.

The base address is universal and singular to the PMA slab used for the nockvm instance.

The NockVM runtime will still be using direct pointers constructed using base address + offset arithmetic to dereference Noun nodes in the PMA. However, the PMA itself will work purely in terms of position-independent addressing which is all offset based so that if you reload the mmap-based PMA slab from disk all the offsets are still valid and simply recalculated in terms of the new base address you got from the virtual memory subsystem of your platform.

We will need to use pointer tags to distinguish between pointers and offsets.

On a read you branch on the pointer tag bits and you variously:

- strip the tag bits, and dereference the resulting pointer
- strip the tag bits, add the offset to a base address, cast that into a pointer, and dereference that pointer

Discriminant is a single bit in the tag. Signifies whether the reference is in the PMA or in the nockstack noun slab. There's a separate discriminator bit already in the Noun representation for distinguishing Atoms, IndirectAtoms, and Cells. For our purposes, we care about values vs. references.

After you've established whether the tag bits signify whether a value is a direct pointer to a noun entrypoint or a PMA offset, you now need to to distinguish whether the

## Milestone 1 discriminant bits hypothetical diff

### Current status / before milestone 1

Before (current master) ― four discriminants only: direct atom, indirect atom, cell, forwarding pointer.

All allocated variants currently hold raw pointers produced by `NockStack::derive_ptr()` (`open/crates/nockvm/rust/nockvm/src/mem.rs`).

```rust
/// Mirrors the actual constants in noun.rs.
const DIRECT_MASK: u64 = !(u64::MAX >> 1); // 0x8000_0000_0000_0000
const DIRECT_TAG: u64 = 0;

const INDIRECT_MASK: u64 = !(u64::MAX >> 2); // 0xC000_0000_0000_0000
const INDIRECT_TAG: u64 = u64::MAX & DIRECT_MASK; // pattern 10xx...

const CELL_MASK: u64 = !(u64::MAX >> 3); // 0xE000_0000_0000_0000
const CELL_TAG: u64 = u64::MAX & INDIRECT_MASK; // pattern 110x...

const FORWARDING_MASK: u64 = CELL_MASK;
const FORWARDING_TAG: u64 = u64::MAX & CELL_MASK; // pattern 111x...

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Noun {
    raw: u64,
}

impl Noun {
    #[inline]
    fn tag_bits(self) -> u64 {
        match self.raw & DIRECT_MASK {
            DIRECT_TAG => DIRECT_TAG,
            _ => self.raw & CELL_MASK, // covers indirect/cell/forwarding
        }
    }

    #[inline]
    fn payload_bits(self) -> u64 {
        match self.tag_bits() {
            DIRECT_TAG => self.raw,                         // value <= DIRECT_MAX
            INDIRECT_TAG => self.raw & !INDIRECT_MASK,      // pointer to IndirectAtom
            CELL_TAG => self.raw & !CELL_MASK,              // pointer to CellMemory
            FORWARDING_TAG => self.raw & !FORWARDING_MASK,  // pointer to Allocated
            _ => unreachable!(),
        }
    }
}
```

This is exactly what the current noun representation does: a single u64 word whose top three bits
distinguish values, indirect atoms, cells, or transient forwarding pointers; every allocated case stores
a literal pointer to stack memory.

#### Sidebar about discriminant bits

Can we please just use a safer library for doing this instead of doing it by hand? There's no performance or clarity downside unless they're doing something dumb. It's an unforced error to keep doing this raw when we're in-progress on changing the design anyway.

Short‑list after surveying the current ecosystem (no code, just tradeoffs):

#### Before (just type tags: direct vs indirect vs cell vs forwarding)

- `bitflags` (1.4+): still the cleanest zero‑cost way to name the masks and expose helper methods. Gives
  you a readable bitflags! { struct NounTag: u64 { … } } and keeps the rest of the code close to what we
  already have, just without hand‑rolled constants.
- `bitfield-struct`: generates getters/setters for named bit ranges in a `repr(transparent)` wrapper. Useful
  if you want a tidy `struct NounBits { #[bits = 1] kind: u8, … }` but don’t want a macro DSL as heavy as
  `modular-bitfield`.
- If you want to stick with enums, `strum` + `num_enum::TryFromPrimitive` can encode the three tag states
  into an enum without rolling your own match ladder; it’s still zero cost once optimized. (in...theory. in practice I use bit-masking in some places so an enum could give me heartburn later.)

#### After (type tags + “stack pointer vs PMA offset” location bit)

- `bitflags` still works here, but pairing it with `bytemuck::TransparentWrapper` lets you define a
  `TaggedPtr(u64)` newtype and safely reinterpret between masks/payloads and raw words, which makes the
  pointer/offset split less error‑prone.
- `bitfield-struct` or `modular-bitfield` both shine once you have two orthogonal fields (kind + location).
  They emit getters that return plain integers, so you branch on location() without remembering which bit
  it lives in, and the generated code is just a few shifts/masks.
- For the pointer/offset arm specifically, tagged-pointer (crate) can encode the “pointer with spare
  high bits” case; you would still keep your own offset handling, but it gives you a typed wrapper with
  compile‑time guarantees that the high bit is reserved for tagging.
- If you’d rather treat the tag word as a mini struct, packed_struct lets you declare
  `#[packed_struct(bit_numbering = "msb0")] struct NounBits { #[packed_field(bits="0:0")] location: bool,
  … }` and the derive does the rest. Slightly heavier macro, but great when you need to document the
  layout inline.

Bottom line: for the current “before” layout, `bitflags` (optionally with a thin newtype) keeps things
minimal. Once you add the location bit in Milestone 1, stepping up to a `bitfield` derive (`bitfield-struct`, `modular-bitfield`, or `packed_struct`) or a purpose-built tagged-pointer wrapper gives you clearer semantics without runtime cost, and you can choose whichever macro style fits your tolerance for abstraction.

#### Validating which bitfield/bitflag crate to use

Assuming they're not messing up the target representation (make note of any applications of `repr` in `noun.rs` in the git history) or outright buggy, it should come down to perf/unforced overhead.

Get some basic operations (like the case discrimination helpers, chewing through IndirectAtoms of cords, etc.) lifted into a `criterion` benchmark harness, implement all the variations of the same minimal target representation with these verbs attached, and horse-race them with the benchmarks.

If the benchmarks confuse you grab `@bitemyapp` as he will greatly enjoy being confused with you. I'm not expecting them to be different unless the underlying representations are different.

Oh yeah, and write tests that verify the exact bit representation of the noun values for each tag-bit discriminant/scenario/type.

### Discriminant bits / Noun repr after Milestone 1, hypothetical diff

After (Milestone 1) ― same value/allocated/forwarding taxonomy, but add a location bit so we can
distinguish direct stack pointers from PMA-relative offsets. Reads first branch on the location bit, then
interpret the payload as either a raw pointer (stack slab) or a word offset to be rebased through the PMA
base pointer supplied by `NockStack`.

The distinction is between the nursery (not persistent, will get thrown away on a stack flip if not permanently allocated) and the persistent non-nursery part of the area (it survived nursery generation on a stack flip/preserve because it was permanently allocated). This distinction exists in the previous system but there was no "persistence" entailed in surviving the nursery and reaching the slab permanently.

```rust
const LOCATION_BIT: u64 = 1 << 60; // next free bit above CELL_MASK
const VALUE_MASK: u64 = !(DIRECT_MASK | LOCATION_BIT);

#[derive(Clone, Copy)]
enum PtrKind {
    StackPtr(*mut u8),
    PmaOffset(u32), // word index inside mmap’d PMA slab
}

impl Noun {
    #[inline]
    fn pointer_descriptor(self) -> Option<(u64 /* tag */, PtrKind)> {
        let tag = self.tag_bits();
        if tag == DIRECT_TAG {
            return None;
        }

        let payload = self.raw & VALUE_MASK;
        let ptr = if self.raw & LOCATION_BIT == 0 {
            PtrKind::StackPtr(payload as *mut u8)
        } else {
            PtrKind::PmaOffset(payload as u32)
        };

        Some((tag, ptr))
    }

    #[inline]
    fn resolve_cell<'a>(&self, base: *const u8) -> Option<&'a CellMemory> {
        let (tag, descriptor) = self.pointer_descriptor()?;
        if tag != CELL_TAG {
            return None;
        }

        match descriptor {
            PtrKind::StackPtr(ptr) => Some(unsafe { &*(ptr as *const CellMemory) }),
            PtrKind::PmaOffset(words) => {
                let ptr = unsafe { base.add((words as usize) << 3) } as *const CellMemory;
                Some(unsafe { &*ptr })
            }
        }
    }
}
```

This “after” block is the hypothetical partial diff you can paste into docs/nock-pma.md: it keeps the
exact bit patterns the runtime already relies on, but demonstrates how Milestone 1 splits the allocated
payload into “stack pointer” vs “offset into PMA,” which is the key new discriminant the doc needs to
communicate.


# Milestone 2: Persistence

Using mmap to persist to disk. We will be assuming only a single reader/writer
for now (Milestone 5 is concurrent reads).

This consists of two phases:

## Phase 1

Phase 1 is to separate out the NockStack from the arena.

```

  ┌──────────────────────┐    ┌──────────────────────┐
  │      NockStack       │    │         PMA          │
  │(ephemeral, anon mmap)│    │  (persistent, file)  │
  │                      │    │                      │
  │ [frames][stk→ ←alloc]│    │ [bump-allocated      │
  │                      │    │  nouns in offset     │
  │ Cleared after each   │    │  form]               │
  │ event                │    │                      │
  │                      │    │ Loaded at boot,      │
  │ Stack-pointer form   │    │ persisted to disk    │
  │ only                 │    │                      │
  └──────────────────────┘    └──────────────────────┘
           │                            ▲
           │   evacuate_to_pma()        │
           └────────────────────────────┘
```

We need to push the persistent arena to a memory slab that is bump-allocated at
the page level. As things stand now, NockStack lives in an anonymous mmap.

Currently, at the end of every event, NockVM is left with a single stack frame,
the top frame, and a bunch of data to be preserved - the kernel, jet states, and
cache. `preserve()` gets called on all of these, which copies them to the other
side of the memory arena, where then any Nouns that are in stack-pointer form
are retagged into offset form.

This step is to be replaced with a new copying step, into a file-backed mmap
called the persistent memory arena (PMA).

Phase 1 will be complete when data is being copied into the PMA at the
conclusion of each event, and NockStack works with references into the PMA for a
single writer/reader.

### Phase 1 spec

Here is a more detailed spec for phase 1:

The central struct for the PMA. `alloc_offset` uses `usize` for now since there
is only one reader/writer, but we will move to `AtomicUsize` when multiplayer
gets enabled.
```rust
/// The Persistent Memory Arena
///
/// A bump-allocated memory region for storing nouns in offset form.
/// The PMA is backed by a file (in future milestones) and persists across
/// program restarts.
///
/// "Bump-allocated" means allocation simply increments the `alloc_offset`
/// pointer by the requested size—there is no free list, no compaction, and
/// no mechanism to reclaim memory once allocated. This makes allocation
/// extremely fast (just a pointer bump) but means the PMA grows monotonically
/// until explicitly reset.
///
/// When a Noun that lives in the PMA needs to be modified, the workflow is:
/// 1. The Noun is read from the PMA (already in offset form)
/// 2. Modifications happen in the NockStack (ephemeral working memory)
/// 3. The modified Noun is copied back to the PMA via `copy_to_pma()`
///
/// Step 3 only allocates space for the Allocated subtrees that changed. For
/// example, if `[2 3]` becomes `[4 3]`:
/// - The Cell is Allocated, so a NEW cell is allocated in the PMA with head=4,
///   tail=3 with new DirectAtoms for the 4 and 3 since they are not Allocated.
/// - The old `[2 3]` cell remains in the PMA, untouched but now unreachable
///
/// For Allocated structures, unchanged subtrees that are already in PMA (offset
/// form) are reused without copying. If `[[1 2] 3]` becomes `[[1 2] 4]`:
/// - A NEW outer cell is allocated with tail=4
/// - The head still points to the existing `[1 2]` in PMA (no copy needed)
/// - Only the old outer cell becomes garbage; `[1 2]` is shared
///
/// This copy allocates fresh space in the PMA for the new version—the old
/// version is not overwritten or freed, it simply becomes unreachable garbage.
/// Garbage collection (Milestone 4) will eventually reclaim this dead space.
///
/// Currently Pma is only suitable for a single reader/writer. In the future,
/// `alloc_offset` will be changed to `AtomicUsize` to allow multiple readers.
pub struct Pma {
    /// The underlying arena for memory management and pointer resolution
    arena: Arc<Arena>,
    /// Current allocation offset in words (bump pointer)
    alloc_offset: usize,
    /// Path to the backing file (for future file-backed persistence)
    path: PathBuf,
}
```

As the `Pma` is a place where `Noun`s get allocated, it ought to implement
`NounAllocator`:
```rust
impl NounAllocator for Pma { ... }
```

There is a `PmaError` enum for `Result` types coming out of the PMA.
```rust
#[derive(Debug, Error)]
pub enum PmaError {
    #[error("PMA is full, cannot allocate {requested} words (available: {available})")]
    OutOfMemory { requested: usize, available: usize },

    #[error("PMA not installed in thread-local storage")]
    NotInstalled,

    #[error("Failed to create arena: {0}")]
    ArenaError(#[from] NewStackError),
}
```

Everything that lives in a NockStack that we'd like to live in the PMA
implements the `PmaCopy` trait:
```rust
pub trait PmaCopy {
    /// Copy this value into the PMA.
    ///
    /// For nouns, this evacuates allocated data (indirect atoms, cells) to the PMA
    /// and converts pointers to offset form. Direct atoms are unchanged since they
    /// fit in a single word.
    ///
    /// # Safety
    /// The caller must ensure that the stack's arena is installed in thread-local storage.
    unsafe fn copy_to_pma(&mut self, stack: &NockStack, pma: &mut Pma);

    /// Assert that this value is fully contained within the PMA.
    ///
    /// For nouns, this verifies that all allocated data (indirect atoms, cells)
    /// resides in the PMA. Direct atoms always pass since they have no allocations.
    ///
    /// # Panics
    /// Panics if any part of this value is not in the PMA.
    fn assert_in_pma(&self, pma: &Pma);
}
```

`PmaCopy` is implemented for the following types:
```rust
// nouns
impl PmaCopy for Noun { ... } // Calls copy_noun_to_pma below
// The rest of the Noun types probably just call .as_noun().copy_to_pma()
impl PmaCopy for Atom { ... }
impl PmaCopy for IndirectAtom { ... }
impl PmaCopy for DirectAtom { ... }
impl PmaCopy for Allocated { ... }
impl PmaCopy for Cell { ... }
// cache
impl<T: Copy + PmaCopy> PmaCopy for Hamt<T> { ... }
// jet state
impl PmaCopy for Warm { ... }
impl PmaCopy for WarmEntry { ... }
impl PmaCopy for Hot { ... }
impl PmaCopy for Batteres { ... }
impl PmaCopy for BatteriesList { ... }
impl PmaCopy for NounList { ... }
impl PmaCopy for Cold { ... }
```
I'm not sure about this one, but `Retag` is implemented for it so I've done it.
```rust
impl PmaCopy for () { ... } // Ctrl-F d01347fd for why this is implemented for Preserve. It also implements
                            // Retag which makes me think it probably will be.
```

The main function to accomplish copying to the PMA for `Nouns`. Something like
this:
```rust
impl PmaCopy for Noun {
    /// Copy a noun and all its allocated substructure to the PMA.
    ///
    /// Uses a worklist algorithm to avoid stack overflow on deep structures.
    /// Structural sharing is preserved via forwarding pointers: if the same
    /// substructure is referenced multiple times, it's only copied once.
    ///
    /// # Algorithm
    /// 1. Push (noun, destination_ptr) onto worklist
    /// 2. Pop and process each item:
    ///    - Direct atoms: write directly to destination
    ///    - Already in PMA (offset form): write directly to destination
    ///    - Has forwarding pointer: write forwarded offset-form to destination
    ///    - Indirect atom: copy to PMA, set forwarding pointer, write offset-form
    ///    - Cell: copy metadata to PMA, set forwarding pointer, queue head/tail
    ///
    /// # Safety
    /// - The PMA arena should be installed for reading evacuated nouns afterward
    /// - Source nouns will have forwarding pointers set (corrupting the stack data)
...
}
```

#### Tests
Summary of tests implemented:

```rust
    // Verifies bump allocation returns sequential offsets and correctly tracks free space.
    fn test_pma_allocation() { ... }
    // Verifies offset-to-pointer and pointer-to-offset conversions are inverses of each other.
    fn test_pma_offset_round_trip() { ... }
    // Verifies reset() clears the allocation pointer and reset_to() sets it to a specific offset.
    fn test_pma_reset() { ... }
    // Verifies thread-local PMA installation, access via with_current(), and cleanup via clear.
    fn test_pma_thread_local() { ... }
    // Verifies direct atoms are unchanged by evacuation since they fit in a single word.
    fn test_evacuate_direct_atom() { ... }
    // Verifies indirect atoms (too large for direct representation) are copied to PMA and converted to offset form.
    fn test_evacuate_indirect_atom() { ... }
    // Verifies a simple cell with direct atom contents is evacuated and readable from PMA.
    fn test_evacuate_simple_cell() { ... }
    // Verifies nested cell structures are fully evacuated with all sub-cells in offset form.
    fn test_evacuate_nested_cells() { ... }
    // Verifies cells containing indirect atoms have both the cell and atoms correctly evacuated.
    fn test_evacuate_with_indirect_atoms() { ... }
    // Verifies structural sharing is preserved: [x x] evacuates x only once, with both refs pointing to same PMA location.
    fn test_evacuate_shared_structure() { ... }
    // Verifies sharing is preserved across separate evacuate calls via forwarding pointers left in stack memory.
    fn test_evacuate_multiple_nouns_preserves_sharing() { ... }
    // Verifies evacuating an already-evacuated noun is a no-op that allocates nothing.
    fn test_evacuate_already_evacuated() { ... }
    // Verifies deeply nested structures are fully evacuated and traversable after evacuation.
    fn test_evacuate_deep_tree() { ... }
    // Verifies contains_ptr correctly identifies pointers inside vs outside the PMA memory region.
    fn test_pma_contains_ptr() { ... }
    // Verifies allocation fails gracefully when PMA is full, rolling back the failed allocation.
    fn test_pma_out_of_memory() { ... }
    // checks that allocating in PMA bumps the alloc ptr
    fn test_persistent_arena_allocation_is_monotonic() { ... }
    // checks NockStack is empty after moving noun to PMA,
    fn test_pma_preserve_moves_noun_and_resets_stack() { ... }
    // does a HAMT preserve work?
    fn test_preserve_hamt_round_trip()  { ... }
    // HAMT evacuate with Cells as values and IndirectAtoms as keys
    fn test_evacuate_hamt_complex_nouns() { ... }
    // jet state round trip tests
    fn test_evacuate_warm_round_trip() { ... }
    fn test_evacuate_warm_entry_round_trip() { ... }
    fn test_evacuate_hot_round_trip() { ... }
    fn test_evacuate_batteries_round_trip() { ... }
    fn test_evacuate_batteries_list_round_trip() { ... }
    fn test_evacuate_noun_list_round_trip() { ... }
    fn test_evacuate_cold_round_trip() { ... }
```
Tests not yet implemented:

##### Memory alignment and layout:
- `test_evacuate_indirect_atom_alignment` - Verifies indirect atoms of various sizes (1, 2, 3, 7, 8, 9 words)
 are properly aligned in PMA and readable without alignment faults.
- `test_evacuate_cell_memory_layout` - Verifies CellMemory fields (metadata, head, tail) are at correct
offsets after evacuation by reading each field independently.

##### Forwarding pointer edge cases:
- `test_forwarding_pointer_diamond_sharing` - Verifies diamond-shaped DAGs (A→B, A→C, B→D, C→D) preserve all
sharing and D is only copied once.
- `test_forwarding_pointer_wide_sharing` - Verifies a single noun referenced by many (e.g., 100) different
cells is only copied once.
- `test_forwarding_pointer_not_leaked_to_pma` - Verifies no forwarding pointers remain in PMA memory after
evacuation completes (they should only exist transiently in stack memory).

##### Boundary and edge cases:
- `test_evacuate_maximum_depth_tree` - Verifies evacuation handles very deep trees (e.g., 1000 levels)
without stack overflow in the worklist loop.
- `test_evacuate_large_indirect_atom` - Verifies indirect atoms near the maximum representable size evacuate
correctly.
- `test_evacuate_single_word_indirect_atom` - Verifies the smallest possible indirect atom (just over
DIRECT_MAX) evacuates correctly.
- `test_evacuate_mixed_pma_stack_noun` - Verifies a cell where head is already in PMA and tail is on stack
evacuates correctly (only tail gets copied).

##### Use-after-evacuation (Miri should catch these):
- `test_stack_memory_not_accessed_after_evacuation` - After evacuation, verify that reading the evacuated
noun uses PMA memory, not stack memory (may need to poison/zero stack to detect).
- `test_evacuate_then_pop_frame_then_read` - Evacuate a noun, pop the stack frame that contained it, then
read from the PMA copy - should work without accessing freed memory.

##### Concurrent allocation:
For now we are assuming the PMA has only a single writer/reader, so we won't
implement these tests, but they are listed for future reference.
- `test_concurrent_pma_allocation` - Spawns multiple threads that allocate from PMA simultaneously, verifies
no overlapping allocations and total allocated equals sum of individual allocations.
- `test_concurrent_allocation_under_pressure` - Multiple threads racing to allocate when PMA is nearly full,
verifies OOM errors are returned correctly without corruption.

##### Idempotency and repeated operations:
- `test_evacuate_same_noun_twice_same_call` - Passes the same noun pointer twice in succession; second call
should be pure no-op.
- `test_evacuate_after_pma_reset` - Evacuate, reset PMA, evacuate same structure again - verifies clean
re-evacuation without confusion from old data.

##### Memory initialization:
- `test_evacuated_metadata_initialized` - Verifies cell metadata is properly copied (not uninitialized) by
checking mug cache bits after evacuation.
- `test_evacuated_indirect_atom_padding_zeroed` - For indirect atoms that don't fill their last word
completely, verify padding bytes are deterministic.

##### Invalid input detection (debug assertions):
- `test_evacuate_rejects_cyclic_structure` - Verifies the assert_acyclic! macro fires when given a cyclic
noun (if we can construct one).
- `test_evacuate_rejects_existing_forwarding_pointer` - Verifies `assert_no_forwarding_pointers!` fires when
given a noun with pre-existing forwarding pointers.

##### Arena switching correctness:
- `test_read_pma_noun_with_wrong_arena_installed` - Verifies reading an evacuated noun with the stack arena
(not PMA arena) installed produces incorrect/detectable results or panics.
- `test_arena_switch_mid_traversal` - Verifies that traversing a noun tree requires consistent arena
installation throughout.

## Phase 2

Once we have successfully separated out NockStack from the PMA, we need to
actually implement the ability to load the PMA from disk and make use of it in
ordinary operation of the NockVM.

# Milestone 3: Mutation and freeing

# Milestone 4: Garbage collection

# Milestone 5: Concurrent reads
````

## crates/nockvm/rust/nockvm/src/pma.rs
```
//! Persistent Memory Arena (PMA)
//!
//! The PMA is a file-backed memory region for storing long-lived Nouns.
//! It uses bump allocation and stores nouns in offset form.

use std::path::PathBuf;
use std::ptr::copy_nonoverlapping;
use std::sync::Arc;

use either::Either::{Left, Right};
use thiserror::Error;

use crate::ext::noun_equality;
use crate::mem::{word_size_of, Arena, NewStackError, NockStack};
use crate::noun::{Atom, Cell, CellMemory, IndirectAtom, Noun, NounAllocator};

/// Errors that can occur during PMA operations
#[derive(Debug, Error)]
pub enum PmaError {
    #[error("PMA is full, cannot allocate {requested} words (available: {available})")]
    OutOfMemory { requested: usize, available: usize },

    #[error("PMA not installed in thread-local storage")]
    NotInstalled,

    #[error("Failed to create arena: {0}")]
    ArenaError(#[from] NewStackError),
}

/// The Persistent Memory Arena
///
/// A bump-allocated memory region for storing nouns in offset form.
/// The PMA is backed by a file (in future milestones) and persists across
/// program restarts.
///
/// "Bump-allocated" means allocation simply increments the `alloc_offset`
/// pointer by the requested size—there is no free list, no compaction, and
/// no mechanism to reclaim memory once allocated. This makes allocation
/// extremely fast (just a pointer bump) but means the PMA grows monotonically
/// until explicitly reset.
///
/// When a Noun that lives in the PMA needs to be modified, the workflow is:
/// 1. The Noun is read from the PMA (already in offset form)
/// 2. Modifications happen in the NockStack (ephemeral working memory)
/// 3. The modified Noun is copied back to the PMA via `copy_to_pma()`
///
/// Step 3 only allocates space for the Allocated subtrees that changed. For
/// example, if `[2 3]` becomes `[4 3]`:
/// - The Cell is Allocated, so a NEW cell is allocated in the PMA with head=4,
///   tail=3 with new DirectAtoms for the 4 and 3 since they are not Allocated.
/// - The old `[2 3]` cell remains in the PMA, untouched but now unreachable
///
/// For Allocated structures, unchanged subtrees that are already in PMA (offset
/// form) are reused without copying. If `[[1 2] 3]` becomes `[[1 2] 4]`:
/// - A NEW outer cell is allocated with tail=4
/// - The head still points to the existing `[1 2]` in PMA (no copy needed)
/// - Only the old outer cell becomes garbage; `[1 2]` is shared
///
/// This copy allocates fresh space in the PMA for the new version—the old
/// version is not overwritten or freed, it simply becomes unreachable garbage.
/// Garbage collection (Milestone 4) will eventually reclaim this dead space.
///
/// Currently Pma is only suitable for a single reader/writer. In the future,
/// `alloc_offset` will be changed to `AtomicUsize` to allow multiple readers.
///
/// For more information, see nock-pma.md.
pub struct Pma {
    /// The underlying arena for memory management and pointer resolution
    arena: Arc<Arena>,
    /// Current allocation offset in words (bump pointer)
    alloc_offset: usize,
    /// Path to the backing file (for future file-backed persistence)
    path: PathBuf,
}

impl Pma {
    /// Create a new PMA with the given size in words
    pub fn new(size_words: usize, path: PathBuf) -> Result<Self, PmaError> {
        let arena = Arena::allocate(size_words)?;
        Ok(Self {
            arena,
            alloc_offset: 0,
            path,
        })
    }

    /// Get the underlying arena
    pub fn arena(&self) -> &Arc<Arena> {
        &self.arena
    }

    /// Install the PMA's arena in thread-local storage.
    ///
    /// Returns a guard that automatically clears the thread-local when dropped.
    /// This allows `Arena::with_current()` to access the PMA's arena.
    pub fn install(&self) -> PmaInstallGuard {
        Arena::set_thread_local(&self.arena);
        PmaInstallGuard { _private: () }
    }

    /// Get the current allocation offset in words
    pub fn alloc_offset(&self) -> usize {
        self.alloc_offset
    }

    /// Get the total size of the PMA in words
    pub fn size_words(&self) -> usize {
        self.arena.words()
    }

    /// Get the number of free words remaining
    pub fn free_words(&self) -> usize {
        self.size_words().saturating_sub(self.alloc_offset())
    }

    /// Convert a pointer within the PMA to an offset in words
    pub fn offset_from_ptr(&self, ptr: *const u8) -> u32 {
        self.arena.offset_from_ptr(ptr)
    }

    /// Convert an offset in words to a pointer
    pub fn ptr_from_offset(&self, offset_words: u32) -> *mut u8 {
        self.arena.ptr_from_offset(offset_words)
    }

    /// Check if a pointer is within the PMA's memory region
    pub fn contains_ptr(&self, ptr: *const u8) -> bool {
        let base = self.arena.base_ptr() as usize;
        let end = base + self.arena.len_bytes();
        let ptr_addr = ptr as usize;
        ptr_addr >= base && ptr_addr < end
    }

    /// Reset the allocation pointer to zero
    pub fn reset(&mut self) {
        self.alloc_offset = 0;
    }

    /// Reset the allocation pointer to a specific offset
    ///
    /// # Panics
    /// Panics if `offset` is greater than the PMA size.
    pub fn reset_to(&mut self, offset: usize) {
        assert!(
            offset <= self.size_words(),
            "reset_to offset {} exceeds PMA size {}",
            offset,
            self.size_words()
        );
        self.alloc_offset = offset;
    }

    /// Check if an allocation of `words` would exceed available space.
    ///
    /// # Panics
    /// Panics with `PmaError::OutOfMemory` if there isn't enough space.
    pub fn alloc_would_oom(&self, words: usize) {
        if words > self.free_words() {
            panic!(
                "{}",
                PmaError::OutOfMemory {
                    requested: words,
                    available: self.free_words(),
                }
            );
        }
    }

    /// Allocate `words` from the PMA, returning a pointer to the allocation.
    ///
    /// # Panics
    /// Panics if there isn't enough space in the PMA.
    unsafe fn raw_alloc(&mut self, words: usize) -> *mut u64 {
        self.alloc_would_oom(words);
        let ptr = self.arena.ptr_from_offset(self.alloc_offset as u32) as *mut u64;
        self.alloc_offset += words;
        ptr
    }
}

/// RAII guard for PMA arena installation.
///
/// When this guard is dropped, it automatically clears the thread-local arena.
/// This ensures the arena is only installed for the lifetime of the guard.
///
/// Note: Using `()` makes this a zero-sized type. If we need the ability to
/// "disarm" the guard (skip cleanup on drop), we could switch to a `bool` field
/// like `ReplicaInstallGuard` uses. See `ReplicaInstallGuard` in mem.rs for comparison.
pub struct PmaInstallGuard {
    /// Private field to prevent construction outside of Pma::install()
    _private: (),
}

impl Drop for PmaInstallGuard {
    fn drop(&mut self) {
        Arena::clear_thread_local();
    }
}

impl ibig::Stack for Pma {
    unsafe fn alloc_layout(&mut self, layout: std::alloc::Layout) -> *mut u64 {
        // Convert bytes to words, rounding up
        let words = (layout.size() + 7) >> 3;
        self.raw_alloc(words)
    }
}

impl NounAllocator for Pma {
    unsafe fn alloc_indirect(&mut self, words: usize) -> *mut u64 {
        self.raw_alloc(words + 2)
    }

    unsafe fn alloc_cell(&mut self) -> *mut CellMemory {
        self.raw_alloc(word_size_of::<CellMemory>()) as *mut CellMemory
    }

    unsafe fn alloc_struct<T>(&mut self, count: usize) -> *mut T {
        self.raw_alloc(word_size_of::<T>() * count) as *mut T
    }

    unsafe fn equals(&mut self, a: *mut Noun, b: *mut Noun) -> bool {
        let a = &*a;
        let b = &*b;
        noun_equality(a, b)
    }
}

/// Trait for types that can be copied into the PMA.
///
/// This is used to evacuate nouns from the NockStack to the PMA for persistence.
pub trait PmaCopy {
    /// Copy this value into the PMA.
    ///
    /// For nouns, this evacuates allocated data (indirect atoms, cells) to the PMA
    /// and converts pointers to offset form. Direct atoms are unchanged since they
    /// fit in a single word.
    ///
    /// # Safety
    /// The caller must ensure that the stack's arena is installed in thread-local storage.
    unsafe fn copy_to_pma(&mut self, stack: &NockStack, pma: &mut Pma);

    /// Assert that this value is fully contained within the PMA.
    ///
    /// For nouns, this verifies that all allocated data (indirect atoms, cells)
    /// resides in the PMA. Direct atoms always pass since they have no allocations.
    ///
    /// # Panics
    /// Panics if any part of this value is not in the PMA.
    fn assert_in_pma(&self, pma: &Pma);
}

impl PmaCopy for () {
    unsafe fn copy_to_pma(&mut self, _stack: &NockStack, _pma: &mut Pma) {}

    fn assert_in_pma(&self, _pma: &Pma) {}
}

impl PmaCopy for Atom {
    unsafe fn copy_to_pma(&mut self, stack: &NockStack, pma: &mut Pma) {
        let mut noun = self.as_noun();
        noun.copy_to_pma(stack, pma);
        *self = noun.as_atom().expect("Atom remains atom after copy_to_pma");
    }

    fn assert_in_pma(&self, pma: &Pma) {
        self.as_noun().assert_in_pma(pma);
    }
}

impl PmaCopy for Noun {
    /// Copy a noun and all its allocated substructure to the PMA.
    ///
    /// Uses a worklist algorithm to avoid stack overflow on deep structures.
    /// Structural sharing is preserved via forwarding pointers: if the same
    /// substructure is referenced multiple times, it's only copied once.
    ///
    /// # Algorithm
    /// 1. Push (noun, destination_ptr) onto worklist
    /// 2. Pop and process each item:
    ///    - Direct atoms: write directly to destination
    ///    - Already in PMA (offset form): write directly to destination
    ///    - Has forwarding pointer: write forwarded offset-form to destination
    ///    - Indirect atom: copy to PMA, set forwarding pointer, write offset-form
    ///    - Cell: copy metadata to PMA, set forwarding pointer, queue head/tail
    ///
    /// # Safety
    /// - The PMA arena should be installed for reading evacuated nouns afterward
    /// - Source nouns will have forwarding pointers set (corrupting the stack data)
    unsafe fn copy_to_pma(&mut self, _stack: &NockStack, pma: &mut Pma) {
        // Direct atoms fit in a single word and don't need evacuation
        if self.is_direct() {
            return;
        }

        // Already in offset form (already in PMA) - nothing to do
        if !self.is_stack_allocated() {
            return;
        }

        // Clone the Arc to avoid borrow conflicts during mutation
        //TODO not sure this is right
        let arena = Arc::clone(pma.arena());

        // Worklist of (source noun, destination pointer)
        // Destination pointers are either the root noun or fields within PMA cells
        let mut work: Vec<(Noun, *mut Noun)> = Vec::with_capacity(32);
        work.push((*self, self as *mut Noun));

        while let Some((noun, dest_ptr)) = work.pop() {
            match noun.as_either_direct_allocated() {
                Left(_direct) => {
                    // Direct atoms are copied as-is (no allocation needed)
                    *dest_ptr = noun;
                }
                Right(allocated) => {
                    // Check for forwarding pointer (already evacuated, structural sharing)
                    if let Some(forwarded) = allocated.forwarding_pointer_with_arena(&arena) {
                        // Convert forwarded pointer to offset form
                        let pma_ptr = forwarded.to_raw_pointer_with_arena(&arena);
                        let offset = pma.offset_from_ptr(pma_ptr as *const u8);
                        if allocated.is_indirect() {
                            *dest_ptr = IndirectAtom::from_offset_words(offset).as_noun();
                        } else {
                            *dest_ptr = Cell::from_offset_words(offset).as_noun();
                        }
                        continue;
                    }

                    // Already in offset form (already in PMA)
                    if !noun.is_stack_allocated() {
                        *dest_ptr = noun;
                        continue;
                    }

                    match allocated.as_either() {
                        Left(mut indirect) => {
                            // Get size and source pointer before allocating
                            let raw_size = indirect.raw_size_with_arena(&arena);
                            let src_ptr = indirect.to_raw_pointer_with_arena(&arena);

                            // Allocate in PMA
                            let pma_ptr = pma.raw_alloc(raw_size);

                            // Copy all data (metadata + size + data words)
                            copy_nonoverlapping(src_ptr, pma_ptr, raw_size);

                            // Set forwarding pointer in source for structural sharing
                            indirect.set_forwarding_pointer_with_arena(pma_ptr, &arena);

                            // Write offset-form noun to destination
                            let offset = pma.offset_from_ptr(pma_ptr as *const u8);
                            *dest_ptr = IndirectAtom::from_offset_words(offset).as_noun();
                        }
                        Right(mut cell) => {
                            // Get source cell pointer
                            let src_cell = cell.to_raw_pointer_with_arena(&arena);

                            // Allocate cell in PMA
                            let pma_ptr = pma.raw_alloc(word_size_of::<CellMemory>());
                            let pma_cell = pma_ptr as *mut CellMemory;

                            // Copy metadata
                            (*pma_cell).metadata = (*src_cell).metadata;

                            // Get head and tail BEFORE setting forwarding pointer
                            // (forwarding pointer overwrites head field)
                            let head = (*src_cell).head;
                            let tail = (*src_cell).tail;

                            // Set forwarding pointer in source for structural sharing
                            cell.set_forwarding_pointer_with_arena(pma_cell, &arena);

                            // Queue head and tail for processing
                            // Destinations are the head/tail slots in the PMA cell
                            work.push((tail, &mut (*pma_cell).tail));
                            work.push((head, &mut (*pma_cell).head));

                            // Write offset-form cell to destination
                            let offset = pma.offset_from_ptr(pma_ptr as *const u8);
                            *dest_ptr = Cell::from_offset_words(offset).as_noun();
                        }
                    }
                }
            }
        }
    }

    /// Assert that this noun and all its substructure is in the PMA.
    ///
    /// # Panics
    /// Panics if any allocated part of the noun is stack-allocated rather than
    /// in offset form (PMA).
    ///
    /// # Note
    /// The PMA arena must be installed before calling this for cells, as it needs
    /// to resolve cell head/tail pointers.
    fn assert_in_pma(&self, pma: &Pma) {
        // Direct atoms have no allocations, so they're trivially "in" the PMA
        if self.is_direct() {
            return;
        }

        // Check that allocated nouns are in offset form (not stack-allocated)
        assert!(
            !self.is_stack_allocated(),
            "Noun is stack-allocated, not in PMA"
        );

        // For cells, recursively check head and tail
        if self.is_cell() {
            let cell = self.as_cell().expect("checked is_cell");
            // Arena must be installed by caller for head()/tail() to work
            cell.head().assert_in_pma(pma);
            cell.tail().assert_in_pma(pma);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamt::Hamt;
    use crate::jets::cold::NounListMem;
    use crate::mem::{word_size_of, Arena, NockStack};
    use crate::noun::{D, DIRECT_MAX};
    use ibig::Stack;
    use std::alloc::Layout;
    use std::sync::Arc;

    /// Helper to create a test PMA with a given size
    fn test_pma(size_words: usize) -> Pma {
        Pma::new(size_words, PathBuf::from("/tmp/test_pma")).expect("Failed to create test PMA")
    }

    /// Verifies bump allocation returns sequential offsets and correctly tracks free space.
    ///
    /// This test exercises:
    /// - Pma::new creates a valid PMA
    /// - alloc_offset() starts at 0
    /// - free_words() equals size initially
    /// - NounAllocator::alloc_indirect bumps the offset correctly
    /// - NounAllocator::alloc_cell allocates CellMemory
    /// - NounAllocator::alloc_struct allocates arbitrary structs
    /// - Sequential allocations don't overlap
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_allocation() {
        let mut pma = test_pma(1000);

        // Initial state: nothing allocated yet
        assert_eq!(pma.alloc_offset(), 0, "Initial alloc_offset should be 0");
        assert_eq!(pma.free_words(), 1000, "Initial free_words should equal size");

        // First allocation: alloc_indirect(10) allocates 10 + 2 = 12 words (data + metadata + size)
        let ptr1 = unsafe { pma.alloc_indirect(10) };
        assert!(!ptr1.is_null(), "First allocation should return non-null pointer");
        assert_eq!(pma.alloc_offset(), 12, "After alloc_indirect(10), offset should be 12");
        assert_eq!(pma.free_words(), 988, "After alloc_indirect(10), free should be 988");

        // Second allocation: alloc_indirect(20) allocates 20 + 2 = 22 words
        let ptr2 = unsafe { pma.alloc_indirect(20) };
        assert!(!ptr2.is_null(), "Second allocation should return non-null pointer");
        assert_eq!(pma.alloc_offset(), 34, "After second alloc, offset should be 34");
        assert_eq!(pma.free_words(), 966, "After second alloc, free should be 966");

        // Third allocation: alloc_cell allocates word_size_of::<CellMemory>() words
        let ptr3 = unsafe { pma.alloc_cell() };
        assert!(!ptr3.is_null(), "Cell allocation should return non-null pointer");
        let cell_words = word_size_of::<CellMemory>();
        let offset_after_cell = 34 + cell_words;
        assert_eq!(
            pma.alloc_offset(),
            offset_after_cell,
            "After cell alloc, offset should increase by CellMemory size"
        );

        // Fourth allocation: alloc_struct for NounListMem
        let struct_words = word_size_of::<NounListMem>();
        let ptr4: *mut NounListMem = unsafe { pma.alloc_struct(1) };
        assert!(!ptr4.is_null(), "Struct allocation should return non-null pointer");
        let offset_after_struct = offset_after_cell + struct_words;
        assert_eq!(
            pma.alloc_offset(),
            offset_after_struct,
            "After struct alloc, offset should increase by struct size in words"
        );

        // Fifth allocation: alloc_struct with count > 1 (allocate array of 3 NounListMem)
        let ptr5: *mut NounListMem = unsafe { pma.alloc_struct(3) };
        assert!(!ptr5.is_null(), "Array struct allocation should return non-null pointer");
        let offset_after_array = offset_after_struct + (struct_words * 3);
        assert_eq!(
            pma.alloc_offset(),
            offset_after_array,
            "After array alloc, offset should increase by struct_size * count"
        );

        // Sixth allocation: alloc_layout for ibig::Stack trait (allocate 8 u64s)
        let layout_words = 8usize;
        let layout = Layout::array::<u64>(layout_words).expect("valid layout");
        let ptr6 = unsafe { pma.alloc_layout(layout) };
        assert!(!ptr6.is_null(), "Layout allocation should return non-null pointer");
        assert_eq!(
            pma.alloc_offset(),
            offset_after_array + layout_words,
            "After layout alloc, offset should increase by layout size in words"
        );

        // Verify all allocations are sequential and non-overlapping
        // For a bump allocator, each pointer should be at or after the end of the previous allocation
        let ptr1_end = unsafe { ptr1.add(12) }; // 12 words for alloc_indirect(10)
        let ptr2_end = unsafe { ptr2.add(22) }; // 22 words for alloc_indirect(20)
        let ptr3_end = unsafe { (ptr3 as *mut u64).add(cell_words) };
        let ptr4_end = unsafe { (ptr4 as *mut u64).add(struct_words) };
        let ptr5_end = unsafe { (ptr5 as *mut u64).add(struct_words * 3) };

        assert!(
            ptr2 >= ptr1_end,
            "ptr2 should start at or after ptr1's end"
        );
        assert!(
            ptr3 as *mut u64 >= ptr2_end,
            "ptr3 should start at or after ptr2's end"
        );
        assert!(
            ptr4 as *mut u64 >= ptr3_end,
            "ptr4 should start at or after ptr3's end"
        );
        assert!(
            ptr5 as *mut u64 >= ptr4_end,
            "ptr5 should start at or after ptr4's end"
        );
        assert!(
            ptr6 >= ptr5_end,
            "ptr6 should start at or after ptr5's end"
        );
    }

    /// Verifies offset-to-pointer and pointer-to-offset conversions are inverses.
    ///
    /// This test exercises:
    /// - ptr_from_offset converts word offset to pointer
    /// - offset_from_ptr converts pointer back to word offset
    /// - Round-trip: offset -> ptr -> offset gives same offset
    /// - Round-trip: ptr -> offset -> ptr gives same ptr
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_offset_round_trip() {
        let mut pma = test_pma(1000);

        // Test with offset 0 (base of PMA)
        let ptr_at_0 = pma.ptr_from_offset(0);
        let offset_from_0 = pma.offset_from_ptr(ptr_at_0);
        assert_eq!(offset_from_0, 0, "Offset at base should be 0");

        // Test with a known offset
        let test_offset: u32 = 42;
        let ptr = pma.ptr_from_offset(test_offset);
        let recovered_offset = pma.offset_from_ptr(ptr);
        assert_eq!(
            recovered_offset, test_offset,
            "Round-trip offset -> ptr -> offset should return same offset"
        );

        // Test with pointer from an allocation
        let alloc_ptr = unsafe { pma.alloc_indirect(10) };
        let alloc_offset = pma.offset_from_ptr(alloc_ptr as *const u8);
        let recovered_ptr = pma.ptr_from_offset(alloc_offset);
        assert_eq!(
            recovered_ptr, alloc_ptr as *mut u8,
            "Round-trip ptr -> offset -> ptr should return same pointer"
        );

        // Test multiple allocations have distinct offsets
        let ptr1 = unsafe { pma.alloc_indirect(5) };
        let ptr2 = unsafe { pma.alloc_indirect(5) };
        let offset1 = pma.offset_from_ptr(ptr1 as *const u8);
        let offset2 = pma.offset_from_ptr(ptr2 as *const u8);
        assert_ne!(
            offset1, offset2,
            "Different allocations should have different offsets"
        );

        // Verify the offsets differ by the expected amount (5 + 2 = 7 words)
        assert_eq!(
            offset2 - offset1,
            7,
            "Second allocation offset should be 7 words after first"
        );
    }

    /// Verifies contains_ptr correctly identifies pointers inside vs outside the PMA.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_contains_ptr() {
        let mut pma = test_pma(1000);

        // Get base pointer and compute some test pointers
        let base = pma.arena().base_ptr();
        let len_bytes = pma.arena().len_bytes();

        // Base pointer should be in PMA
        assert!(pma.contains_ptr(base), "Base pointer should be in PMA");

        // Pointer at offset 0 should be in PMA
        let ptr_at_0 = pma.ptr_from_offset(0);
        assert!(pma.contains_ptr(ptr_at_0), "Pointer at offset 0 should be in PMA");

        // Pointer in the middle should be in PMA
        let middle_offset = 500u32;
        let ptr_middle = pma.ptr_from_offset(middle_offset);
        assert!(pma.contains_ptr(ptr_middle), "Pointer in middle should be in PMA");

        // Last valid byte should be in PMA
        let last_byte = unsafe { base.add(len_bytes - 1) };
        assert!(pma.contains_ptr(last_byte), "Last byte should be in PMA");

        // Pointer just past the end should NOT be in PMA
        let past_end = unsafe { base.add(len_bytes) };
        assert!(!pma.contains_ptr(past_end), "Pointer past end should not be in PMA");

        // Pointer well past the end should NOT be in PMA
        let way_past_end = unsafe { base.add(len_bytes + 1000) };
        assert!(!pma.contains_ptr(way_past_end), "Pointer way past end should not be in PMA");

        // Pointer before the base should NOT be in PMA (if base > 0)
        if base as usize > 0 {
            let before_base = unsafe { base.sub(1) };
            assert!(!pma.contains_ptr(before_base), "Pointer before base should not be in PMA");
        }

        // Null pointer should NOT be in PMA
        assert!(!pma.contains_ptr(std::ptr::null()), "Null pointer should not be in PMA");

        // Allocated pointer should be in PMA
        let alloc_ptr = unsafe { pma.alloc_indirect(10) };
        assert!(pma.contains_ptr(alloc_ptr as *const u8), "Allocated pointer should be in PMA");
    }

    /// Verifies allocation fails gracefully when PMA is full.
    ///
    /// This test exercises:
    /// - alloc_would_oom() does not panic when there's space
    /// - alloc_would_oom() panics when there isn't enough space
    /// - Exact-fit allocations succeed
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_out_of_memory() {
        use std::panic::{catch_unwind, AssertUnwindSafe};

        let mut pma = test_pma(100); // Small PMA: 100 words

        // alloc_would_oom should not panic when there's space
        pma.alloc_would_oom(50); // Should not panic
        pma.alloc_would_oom(100); // Should not panic (exact fit)

        // alloc_would_oom should panic when there isn't space
        let result = catch_unwind(AssertUnwindSafe(|| {
            pma.alloc_would_oom(101);
        }));
        assert!(result.is_err(), "alloc_would_oom(101) should panic with 100 free");

        // Allocate some space
        unsafe { pma.alloc_indirect(10) }; // 12 words (10 + 2 for metadata/size)
        assert_eq!(pma.alloc_offset(), 12);
        assert_eq!(pma.free_words(), 88);

        // alloc_would_oom should reflect remaining space
        pma.alloc_would_oom(88); // Should not panic
        let result = catch_unwind(AssertUnwindSafe(|| {
            pma.alloc_would_oom(89);
        }));
        assert!(result.is_err(), "alloc_would_oom(89) should panic with 88 free");

        // Fill the rest
        unsafe { pma.alloc_struct::<u64>(88) };
        assert_eq!(pma.alloc_offset(), 100);
        assert_eq!(pma.free_words(), 0);

        // alloc_would_oom should panic for any non-zero allocation when full
        let result = catch_unwind(AssertUnwindSafe(|| {
            pma.alloc_would_oom(1);
        }));
        assert!(result.is_err(), "alloc_would_oom(1) should panic when full");

        // But 0 words should not panic
        pma.alloc_would_oom(0); // Should not panic

        // Reset and verify we can allocate again
        pma.reset();
        assert_eq!(pma.free_words(), 100);
        pma.alloc_would_oom(100); // Should not panic after reset

        // Verify exact-fit allocation works
        unsafe { pma.alloc_struct::<u64>(100) };
        assert_eq!(pma.alloc_offset(), 100);
        assert_eq!(pma.free_words(), 0);
    }

    /// Verifies reset() and reset_to() correctly manage the allocation pointer.
    ///
    /// This test exercises:
    /// - reset() sets alloc_offset back to 0
    /// - reset_to(offset) sets alloc_offset to a specific value
    /// - After reset, free_words equals size again
    /// - Allocations after reset start from the reset point
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_reset() {
        let mut pma = test_pma(1000);

        // Allocate some space
        unsafe { pma.alloc_indirect(10) }; // 12 words
        unsafe { pma.alloc_indirect(20) }; // 22 words
        assert_eq!(pma.alloc_offset(), 34);
        assert_eq!(pma.free_words(), 966);

        // Reset to zero
        pma.reset();
        assert_eq!(pma.alloc_offset(), 0, "reset() should set offset to 0");
        assert_eq!(pma.free_words(), 1000, "reset() should restore all free space");

        // Allocations after reset should start from 0
        let ptr_after_reset = unsafe { pma.alloc_indirect(5) }; // 7 words
        assert_eq!(pma.alloc_offset(), 7);
        let offset_after_reset = pma.offset_from_ptr(ptr_after_reset as *const u8);
        assert_eq!(offset_after_reset, 0, "First allocation after reset should be at offset 0");

        // Allocate more to create a checkpoint
        unsafe { pma.alloc_indirect(10) }; // 12 more words
        let checkpoint = pma.alloc_offset();
        assert_eq!(checkpoint, 19); // 7 + 12

        // Allocate even more
        unsafe { pma.alloc_indirect(30) }; // 32 more words
        assert_eq!(pma.alloc_offset(), 51); // 19 + 32

        // Reset to checkpoint
        pma.reset_to(checkpoint);
        assert_eq!(pma.alloc_offset(), 19, "reset_to() should set offset to checkpoint");
        assert_eq!(pma.free_words(), 981, "reset_to() should restore free space from checkpoint");

        // Next allocation should start at the checkpoint
        let ptr_after_reset_to = unsafe { pma.alloc_indirect(3) }; // 5 words
        let offset_after_reset_to = pma.offset_from_ptr(ptr_after_reset_to as *const u8);
        assert_eq!(offset_after_reset_to, 19, "Allocation after reset_to should start at checkpoint");
        assert_eq!(pma.alloc_offset(), 24); // 19 + 5
    }

    /// Verifies reset_to panics when given an offset outside the PMA bounds.
    #[test]
    #[should_panic(expected = "reset_to offset")]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_reset_to_out_of_bounds() {
        let mut pma = test_pma(1000);
        pma.reset_to(1001); // Should panic: offset exceeds PMA size
    }

    /// Verifies thread-local PMA installation, access via with_current(), and RAII cleanup.
    ///
    /// This test exercises:
    /// - pma.install() installs the PMA's arena in thread-local storage
    /// - Arena::with_current() can access the installed arena
    /// - The installed arena matches the PMA's arena
    /// - PmaInstallGuard automatically clears the arena when dropped
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_thread_local() {
        let pma = test_pma(1000);
        let pma_arena_ptr = Arc::as_ptr(pma.arena());

        {
            // Install the PMA's arena - guard will clear on drop
            let _guard = pma.install();

            // Verify we can access it via with_current and it's the same arena
            Arena::with_current(|arena| {
                let current_ptr = arena as *const Arena;
                assert_eq!(
                    current_ptr, pma_arena_ptr,
                    "Installed arena should match PMA's arena"
                );
            });
        } // _guard dropped here, arena should be cleared

        // Verify the guard cleared the thread-local by checking that
        // with_current would now panic (we don't call it here to avoid panic,
        // but we test this in test_pma_thread_local_not_installed)
    }

    /// Verifies Arena::with_current panics when no arena is installed.
    #[test]
    #[should_panic(expected = "Arena::with_current called without an installed Arena")]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_thread_local_not_installed() {
        // Ensure no arena is installed
        Arena::clear_thread_local();

        // This should panic
        Arena::with_current(|_arena| {});
    }

    /// Verifies PmaInstallGuard clears the arena when dropped.
    #[test]
    #[should_panic(expected = "Arena::with_current called without an installed Arena")]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_guard_clears_on_drop() {
        let pma = test_pma(1000);

        // Ensure no arena is installed initially
        Arena::clear_thread_local();

        {
            let _guard = pma.install();
            // Arena is installed here, with_current would work
        } // _guard dropped, arena should be cleared

        // This should panic because the guard cleared the arena
        Arena::with_current(|_arena| {});
    }

    /// Verifies direct atoms are unchanged by evacuation since they fit in a single word.
    ///
    /// Direct atoms don't require any allocation - they're just 64-bit values with
    /// the MSB = 0. Evacuation should leave them completely unchanged.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_direct_atom() {
        let stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);

        // Test several direct atom values
        let test_values: [u64; 5] = [0, 1, 42, 12345, DIRECT_MAX];

        for &val in &test_values {
            let mut noun = D(val);
            let original_raw = unsafe { noun.as_raw() };

            // Evacuate to PMA
            unsafe { noun.copy_to_pma(&stack, &mut pma) };

            // Direct atoms should be completely unchanged
            let after_raw = unsafe { noun.as_raw() };
            assert_eq!(
                original_raw, after_raw,
                "Direct atom {} should be unchanged after evacuation",
                val
            );

            // Verify it's still a direct atom
            assert!(noun.is_direct(), "Should still be a direct atom after evacuation");

            // Direct atoms should trivially pass assert_in_pma (no allocations to check)
            noun.assert_in_pma(&pma);
        }

        // PMA should have no allocations (direct atoms don't need space)
        assert_eq!(
            pma.alloc_offset(),
            0,
            "No allocations should be made for direct atoms"
        );
    }

    /// Verifies indirect atoms (too large for direct representation) are copied to PMA
    /// and converted to offset form.
    ///
    /// This test exercises:
    /// - Creating an indirect atom on the NockStack
    /// - Evacuating it to the PMA via copy_to_pma
    /// - Verifying the atom is now in offset form (LOCATION_BIT set)
    /// - Verifying the data can be read correctly via the PMA arena
    /// - Verifying PMA allocations were made
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_indirect_atom() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);

        // Create an indirect atom on the stack (value > DIRECT_MAX requires indirect storage)
        // We'll use a 2-word value to ensure it's indirect
        let data: [u64; 2] = [0xDEADBEEF_CAFEBABE, 0x12345678_9ABCDEF0];
        let indirect = unsafe { IndirectAtom::new_raw(&mut stack, 2, data.as_ptr()) };
        let mut noun = indirect.as_noun();

        // Verify it's an indirect atom on the stack
        assert!(noun.is_indirect(), "Should be an indirect atom");
        assert!(
            !noun.is_direct(),
            "Should not be a direct atom"
        );
        assert!(
            noun.is_stack_allocated(),
            "Should be stack-allocated before evacuation"
        );

        // Record the initial PMA offset
        let initial_offset = pma.alloc_offset();
        assert_eq!(initial_offset, 0, "PMA should start empty");

        // Install the PMA arena for pointer resolution after evacuation
        let _guard = pma.install();

        // Evacuate to PMA
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Verify PMA allocation was made
        // Indirect atom needs: metadata (1) + size (1) + data (2) = 4 words
        assert!(
            pma.alloc_offset() > initial_offset,
            "PMA should have allocations after evacuation"
        );
        assert_eq!(
            pma.alloc_offset(),
            4, // metadata + size + 2 data words
            "Indirect atom should allocate 4 words in PMA"
        );

        // Verify the noun is now in offset form (not stack-allocated)
        assert!(
            !noun.is_stack_allocated(),
            "Should be in offset form after evacuation"
        );
        assert!(noun.is_indirect(), "Should still be an indirect atom");

        // Verify data is readable and correct via PMA arena
        let atom = noun.as_atom().expect("Should be an atom");
        let read_indirect = atom.as_indirect().expect("Should be indirect");

        // Read the size - should be 2 words
        let size = read_indirect.size();
        assert_eq!(size, 2, "Indirect atom should have size 2");

        // Read the data back and verify it matches
        let data_ptr = read_indirect.data_pointer();
        let read_data = unsafe { std::slice::from_raw_parts(data_ptr, 2) };
        assert_eq!(
            read_data[0], data[0],
            "First data word should match"
        );
        assert_eq!(
            read_data[1], data[1],
            "Second data word should match"
        );

        // Verify assert_in_pma passes
        noun.assert_in_pma(&pma);
    }

    /// Verifies a simple cell with direct atom contents is evacuated and readable from PMA.
    ///
    /// This test exercises:
    /// - Creating a cell [head tail] on the NockStack
    /// - Evacuating it to the PMA
    /// - Verifying the cell is in offset form
    /// - Verifying head and tail are readable and correct
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_simple_cell() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);

        // Create a simple cell [42 123] with direct atoms
        let mut noun = Cell::new(&mut stack, D(42), D(123)).as_noun();

        // Verify it's a cell on the stack
        assert!(noun.is_cell(), "Should be a cell");
        assert!(noun.is_stack_allocated(), "Should be stack-allocated before evacuation");

        // Install PMA arena and evacuate
        let _guard = pma.install();
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Verify PMA allocation was made (CellMemory size)
        let cell_words = word_size_of::<CellMemory>();
        assert_eq!(
            pma.alloc_offset(),
            cell_words,
            "Cell should allocate {} words",
            cell_words
        );

        // Verify the noun is now in offset form
        assert!(!noun.is_stack_allocated(), "Should be in offset form after evacuation");
        assert!(noun.is_cell(), "Should still be a cell");

        // Read head and tail
        let cell = noun.as_cell().expect("Should be a cell");
        let head = cell.head();
        let tail = cell.tail();

        // Verify head and tail are correct direct atoms
        assert!(head.is_direct(), "Head should be direct");
        assert!(tail.is_direct(), "Tail should be direct");
        assert_eq!(
            head.as_direct().expect("head is direct").data(),
            42,
            "Head should be 42"
        );
        assert_eq!(
            tail.as_direct().expect("tail is direct").data(),
            123,
            "Tail should be 123"
        );

        // Verify assert_in_pma passes
        noun.assert_in_pma(&pma);
    }

    /// Verifies nested cell structures are fully evacuated with all sub-cells in offset form.
    ///
    /// This test exercises:
    /// - Creating nested cells [[1 2] [3 4]]
    /// - Evacuating the entire structure
    /// - Verifying all cells are in offset form
    /// - Verifying all values are readable
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_nested_cells() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);

        // Create nested cells: [[1 2] [3 4]]
        let left = Cell::new(&mut stack, D(1), D(2)).as_noun();
        let right = Cell::new(&mut stack, D(3), D(4)).as_noun();
        let mut noun = Cell::new(&mut stack, left, right).as_noun();

        // Verify structure before evacuation
        assert!(noun.is_cell(), "Root should be a cell");
        assert!(noun.is_stack_allocated(), "Root should be stack-allocated");

        // Install PMA arena and evacuate
        let _guard = pma.install();
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Should allocate 3 cells worth of space
        let cell_words = word_size_of::<CellMemory>();
        assert_eq!(
            pma.alloc_offset(),
            cell_words * 3,
            "Should allocate 3 cells"
        );

        // Verify root is in offset form
        assert!(!noun.is_stack_allocated(), "Root should be in offset form");

        // Navigate and verify structure
        let root = noun.as_cell().expect("root is cell");
        let left_cell = root.head().as_cell().expect("left is cell");
        let right_cell = root.tail().as_cell().expect("right is cell");

        // Verify left cell [1 2]
        assert!(!root.head().is_stack_allocated(), "Left should be in offset form");
        assert_eq!(left_cell.head().as_direct().expect("1").data(), 1);
        assert_eq!(left_cell.tail().as_direct().expect("2").data(), 2);

        // Verify right cell [3 4]
        assert!(!root.tail().is_stack_allocated(), "Right should be in offset form");
        assert_eq!(right_cell.head().as_direct().expect("3").data(), 3);
        assert_eq!(right_cell.tail().as_direct().expect("4").data(), 4);

        // Verify assert_in_pma passes for entire structure
        noun.assert_in_pma(&pma);
    }

    /// Verifies cells containing indirect atoms have both the cell and atoms correctly evacuated.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_cell_with_indirect_atoms() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);

        // Create indirect atoms
        let data1: [u64; 2] = [0xAAAAAAAA_BBBBBBBB, 0xCCCCCCCC_DDDDDDDD];
        let data2: [u64; 2] = [0x11111111_22222222, 0x33333333_44444444];
        let indirect1 = unsafe { IndirectAtom::new_raw(&mut stack, 2, data1.as_ptr()) };
        let indirect2 = unsafe { IndirectAtom::new_raw(&mut stack, 2, data2.as_ptr()) };

        // Create cell with indirect atoms
        let mut noun = Cell::new(&mut stack, indirect1.as_noun(), indirect2.as_noun()).as_noun();

        assert!(noun.is_stack_allocated(), "Should be stack-allocated");

        // Install PMA arena and evacuate
        let _guard = pma.install();
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Should allocate: 1 cell + 2 indirect atoms (4 words each)
        let cell_words = word_size_of::<CellMemory>();
        let indirect_words = 4; // metadata + size + 2 data words
        assert_eq!(
            pma.alloc_offset(),
            cell_words + indirect_words * 2,
            "Should allocate cell + 2 indirect atoms"
        );

        // Verify structure
        assert!(!noun.is_stack_allocated(), "Root should be in offset form");

        let cell = noun.as_cell().expect("is cell");
        let head = cell.head();
        let tail = cell.tail();

        // Verify head is indirect atom with correct data
        assert!(head.is_indirect(), "Head should be indirect");
        assert!(!head.is_stack_allocated(), "Head should be in offset form");
        let head_indirect = head.as_indirect().expect("head indirect");
        let head_data = unsafe { std::slice::from_raw_parts(head_indirect.data_pointer(), 2) };
        assert_eq!(head_data[0], data1[0]);
        assert_eq!(head_data[1], data1[1]);

        // Verify tail is indirect atom with correct data
        assert!(tail.is_indirect(), "Tail should be indirect");
        assert!(!tail.is_stack_allocated(), "Tail should be in offset form");
        let tail_indirect = tail.as_indirect().expect("tail indirect");
        let tail_data = unsafe { std::slice::from_raw_parts(tail_indirect.data_pointer(), 2) };
        assert_eq!(tail_data[0], data2[0]);
        assert_eq!(tail_data[1], data2[1]);

        noun.assert_in_pma(&pma);
    }

    /// Verifies structural sharing is preserved: [x x] evacuates x only once.
    ///
    /// When the same noun is referenced multiple times, the forwarding pointer
    /// mechanism ensures it's only copied once, and both references point to
    /// the same PMA location.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_shared_structure() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);

        // Create a shared subcell
        let shared = Cell::new(&mut stack, D(1), D(2)).as_noun();

        // Create [shared shared] - both head and tail point to same cell
        let mut noun = Cell::new(&mut stack, shared, shared).as_noun();

        // Install PMA arena and evacuate
        let _guard = pma.install();
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Should allocate only 2 cells: the root and the shared subcell (not 3!)
        let cell_words = word_size_of::<CellMemory>();
        assert_eq!(
            pma.alloc_offset(),
            cell_words * 2,
            "Should allocate only 2 cells due to sharing"
        );

        // Verify both head and tail point to the same PMA location
        let root = noun.as_cell().expect("is cell");
        let head_raw = unsafe { root.head().as_raw() };
        let tail_raw = unsafe { root.tail().as_raw() };
        assert_eq!(
            head_raw, tail_raw,
            "Head and tail should point to same location (sharing preserved)"
        );

        // Verify the shared cell is correct
        let shared_cell = root.head().as_cell().expect("shared is cell");
        assert_eq!(shared_cell.head().as_direct().expect("1").data(), 1);
        assert_eq!(shared_cell.tail().as_direct().expect("2").data(), 2);

        noun.assert_in_pma(&pma);
    }

    /// Verifies evacuating an already-evacuated noun is a no-op that allocates nothing.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_already_evacuated() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);

        // Create and evacuate a cell
        let mut noun = Cell::new(&mut stack, D(1), D(2)).as_noun();
        let _guard = pma.install();
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        let offset_after_first = pma.alloc_offset();
        assert!(offset_after_first > 0, "Should have allocated something");

        // Evacuate again - should be a no-op
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        assert_eq!(
            pma.alloc_offset(),
            offset_after_first,
            "Second evacuation should not allocate anything"
        );

        noun.assert_in_pma(&pma);
    }

    /// Verifies deeply nested structures are fully evacuated and traversable after evacuation.
    ///
    /// This test exercises the worklist algorithm's ability to handle deep trees
    /// without stack overflow (since we use iteration, not recursion).
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_deep_tree() {
        let mut stack = NockStack::new(1 << 14, 0); // Larger stack for deep nesting
        let mut pma = test_pma(10000);

        // Create a deeply nested structure: [1 [2 [3 [4 ... [999 1000]]]]]
        const DEPTH: u64 = 500;

        // Build from the inside out
        let mut noun = D(DEPTH);
        for i in (1..DEPTH).rev() {
            noun = Cell::new(&mut stack, D(i), noun).as_noun();
        }

        // Verify it's deeply nested and stack-allocated
        assert!(noun.is_cell(), "Root should be a cell");
        assert!(noun.is_stack_allocated(), "Should be stack-allocated");

        // Install PMA arena (needed for Cell::tail() even on stack-allocated nouns)
        let _guard = pma.install();

        // Count the depth before evacuation
        let mut depth_before = 0u64;
        let mut current = noun;
        while current.is_cell() {
            depth_before += 1;
            current = current.as_cell().unwrap().tail();
        }
        assert_eq!(depth_before, DEPTH - 1, "Should have correct depth before evacuation");

        // Evacuate
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Should allocate (DEPTH - 1) cells
        let cell_words = word_size_of::<CellMemory>();
        assert_eq!(
            pma.alloc_offset(),
            cell_words * (DEPTH as usize - 1),
            "Should allocate {} cells",
            DEPTH - 1
        );

        // Verify root is in offset form
        assert!(!noun.is_stack_allocated(), "Root should be in offset form");

        // Traverse the entire structure and verify values
        let mut current = noun;
        for expected in 1..DEPTH {
            assert!(current.is_cell(), "Should be cell at depth {}", expected);
            let cell = current.as_cell().expect("is cell");

            // Verify head value
            let head = cell.head();
            assert!(head.is_direct(), "Head at depth {} should be direct", expected);
            assert_eq!(
                head.as_direct().expect("direct").data(),
                expected,
                "Head at depth {} should be {}",
                expected,
                expected
            );

            // Verify this cell is in offset form
            assert!(
                !current.is_stack_allocated(),
                "Cell at depth {} should be in offset form",
                expected
            );

            current = cell.tail();
        }

        // Final element should be direct atom DEPTH
        assert!(current.is_direct(), "Leaf should be direct atom");
        assert_eq!(
            current.as_direct().expect("direct").data(),
            DEPTH,
            "Leaf should be {}",
            DEPTH
        );

        // Verify assert_in_pma passes for entire structure
        noun.assert_in_pma(&pma);
    }

    /// Verifies deeply nested structures with variable-sized indirect atoms are fully evacuated.
    ///
    /// Similar to test_evacuate_deep_tree, but each value is an IndirectAtom with
    /// data size varying from 2 to 10 words. This tests the evacuation of mixed
    /// cell/indirect-atom structures with variable allocation sizes.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_deep_tree_indirect_atoms() {
        let mut stack = NockStack::new(1 << 16, 0); // Larger stack for indirect atoms
        let mut pma = test_pma(100000); // Larger PMA for indirect atoms

        const DEPTH: usize = 200;

        // Helper to create an indirect atom with `word_count` words of data
        // Data pattern: first word is the index, remaining words are index + word_position
        let make_indirect = |stack: &mut NockStack, index: usize, word_count: usize| -> Noun {
            let mut data = vec![0u64; word_count];
            for (i, word) in data.iter_mut().enumerate() {
                *word = (index as u64) << 32 | (i as u64);
            }
            unsafe {
                IndirectAtom::new_raw(stack, word_count, data.as_ptr()).as_noun()
            }
        };

        // Helper to compute word count for index (varies 2-10)
        let word_count_for_index = |index: usize| -> usize { (index % 9) + 2 };

        // Build from inside out: [indirect_1 [indirect_2 [indirect_3 ... indirect_DEPTH]]]
        let mut noun = make_indirect(&mut stack, DEPTH, word_count_for_index(DEPTH));
        for i in (1..DEPTH).rev() {
            let head = make_indirect(&mut stack, i, word_count_for_index(i));
            noun = Cell::new(&mut stack, head, noun).as_noun();
        }

        // Verify structure before evacuation
        assert!(noun.is_cell(), "Root should be a cell");
        assert!(noun.is_stack_allocated(), "Should be stack-allocated");

        // Install PMA arena
        let _guard = pma.install();

        // Count expected allocations:
        // - (DEPTH - 1) cells
        // - DEPTH indirect atoms, each with (word_count + 2) words (metadata + size + data)
        let cell_words = word_size_of::<CellMemory>();
        let mut expected_indirect_words = 0usize;
        for i in 1..=DEPTH {
            expected_indirect_words += word_count_for_index(i) + 2; // +2 for metadata and size
        }
        let expected_total = (cell_words * (DEPTH - 1)) + expected_indirect_words;

        // Evacuate
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Verify allocation size
        assert_eq!(
            pma.alloc_offset(),
            expected_total,
            "Should allocate {} words total ({} cells + {} indirect atom words)",
            expected_total,
            DEPTH - 1,
            expected_indirect_words
        );

        // Verify root is in offset form
        assert!(!noun.is_stack_allocated(), "Root should be in offset form");

        // Traverse and verify all values
        let mut current = noun;
        for expected_index in 1..DEPTH {
            assert!(current.is_cell(), "Should be cell at depth {}", expected_index);
            let cell = current.as_cell().expect("is cell");

            // Verify head is an indirect atom with correct data
            let head = cell.head();
            assert!(head.is_indirect(), "Head at depth {} should be indirect", expected_index);
            assert!(
                !head.is_stack_allocated(),
                "Head at depth {} should be in offset form",
                expected_index
            );

            let head_indirect = head.as_indirect().expect("indirect");
            let expected_word_count = word_count_for_index(expected_index);
            assert_eq!(
                head_indirect.size(),
                expected_word_count,
                "Indirect atom at depth {} should have {} words",
                expected_index,
                expected_word_count
            );

            // Verify data pattern
            let data_ptr = head_indirect.data_pointer();
            for word_idx in 0..expected_word_count {
                let expected_value = (expected_index as u64) << 32 | (word_idx as u64);
                let actual_value = unsafe { *data_ptr.add(word_idx) };
                assert_eq!(
                    actual_value, expected_value,
                    "Data mismatch at depth {}, word {}",
                    expected_index, word_idx
                );
            }

            current = cell.tail();
        }

        // Final element should be indirect atom for index DEPTH
        assert!(current.is_indirect(), "Leaf should be indirect atom");
        assert!(!current.is_stack_allocated(), "Leaf should be in offset form");

        let leaf_indirect = current.as_indirect().expect("indirect");
        let expected_leaf_words = word_count_for_index(DEPTH);
        assert_eq!(
            leaf_indirect.size(),
            expected_leaf_words,
            "Leaf indirect atom should have {} words",
            expected_leaf_words
        );

        // Verify leaf data pattern
        let leaf_data_ptr = leaf_indirect.data_pointer();
        for word_idx in 0..expected_leaf_words {
            let expected_value = (DEPTH as u64) << 32 | (word_idx as u64);
            let actual_value = unsafe { *leaf_data_ptr.add(word_idx) };
            assert_eq!(
                actual_value, expected_value,
                "Leaf data mismatch at word {}",
                word_idx
            );
        }

        // Verify assert_in_pma passes for entire structure
        noun.assert_in_pma(&pma);
    }

    /// Verifies NounAllocator::equals works through the Pma interface.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_noun_allocator_equals() {
        let mut stack = NockStack::new(1 << 10, 0);
        stack.install_arena(); // Required for Cell::new to work
        let mut pma = test_pma(1000);

        let mut noun1 = Cell::new(&mut stack, D(1), D(2)).as_noun();
        let mut noun2 = Cell::new(&mut stack, D(1), D(2)).as_noun();
        let mut noun3 = Cell::new(&mut stack, D(1), D(3)).as_noun();

        // Test through NounAllocator trait
        assert!(
            unsafe { pma.equals(&mut noun1 as *mut Noun, &mut noun2 as *mut Noun) },
            "NounAllocator::equals should return true for equal nouns"
        );
        assert!(
            !unsafe { pma.equals(&mut noun1 as *mut Noun, &mut noun3 as *mut Noun) },
            "NounAllocator::equals should return false for unequal nouns"
        );
    }

    /// Verifies that a HAMT can be evacuated to PMA and lookups still work.
    ///
    /// This test exercises:
    /// - Creating a HAMT with multiple entries (direct atoms as keys/values)
    /// - Evacuating the entire HAMT structure to PMA
    /// - Verifying all entries are still retrievable via lookup
    /// - Verifying all internal pointers are in offset form (not stack-allocated)
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_hamt_round_trip() {
        let mut stack = NockStack::new(1 << 16, 0);
        let mut pma = test_pma(10000);
        let _guard = pma.install();

        // Create a HAMT with several entries
        let mut hamt: Hamt<Noun> = Hamt::new(&mut stack);

        // Insert 10 key-value pairs
        for i in 0u64..10 {
            let mut key = D(i);
            let value = D(i * 100);
            hamt = hamt.insert(&mut stack, &mut key, value);
        }

        // Verify lookups work before evacuation
        for i in 0u64..10 {
            let mut key = D(i);
            let result = hamt.lookup(&mut stack, &mut key);
            assert!(result.is_some(), "Lookup for key {} should succeed before evacuation", i);
            let value = result.unwrap();
            assert!(value.is_direct(), "Value should be direct atom");
            assert_eq!(
                value.as_direct().unwrap().data(),
                i * 100,
                "Value for key {} should be {}", i, i * 100
            );
        }

        // Evacuate the HAMT to PMA
        unsafe {
            hamt.copy_to_pma(&stack, &mut pma);
        }

        // Verify lookups still work after evacuation
        for i in 0u64..10 {
            let mut key = D(i);
            let result = hamt.lookup(&mut stack, &mut key);
            assert!(result.is_some(), "Lookup for key {} should succeed after evacuation", i);
            let value = result.unwrap();
            assert!(value.is_direct(), "Value should still be direct atom after evacuation");
            assert_eq!(
                value.as_direct().unwrap().data(),
                i * 100,
                "Value for key {} should still be {} after evacuation", i, i * 100
            );
        }

        // Verify internal structure is in PMA (offset form)
        // Iterate over the HAMT and check all nouns are not stack-allocated
        for entries in hamt.iter() {
            for (key, value) in entries {
                if !key.is_direct() {
                    assert!(
                        !key.is_stack_allocated(),
                        "HAMT key should be in offset form after evacuation"
                    );
                }
                if !value.is_direct() {
                    assert!(
                        !value.is_stack_allocated(),
                        "HAMT value should be in offset form after evacuation"
                    );
                }
            }
        }
    }

    /// Test that copy_to_pma correctly copies nouns to PMA and produces valid offset-form nouns.
    ///
    /// Note: copy_to_pma sets forwarding pointers in the source nouns, which corrupts
    /// them for normal use. This is by design for structural sharing. Therefore, we
    /// cannot compare source vs PMA copy directly. Instead, we verify the PMA copy
    /// contains the expected data.
    ///
    /// This test may look superfluous, but it helped debug test_evacuate_hamt_complex_nouns so
    /// that's why its in here.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_copy_to_pma_preserves_data() {
        use crate::noun::{Cell, IndirectAtom};

        let mut stack = NockStack::new(1 << 16, 0);
        stack.install_arena();
        let mut pma = test_pma(10000);
        let _guard = pma.install();

        // Test with indirect atom
        let data: [u64; 2] = [0xDEADBEEF_CAFEBABE, 0x12345678_9ABCDEF0];
        let stack_indirect =
            unsafe { IndirectAtom::new_raw(&mut stack, 2, data.as_ptr()) }.as_noun();

        // Copy to PMA
        let mut pma_indirect = stack_indirect;
        unsafe { pma_indirect.copy_to_pma(&stack, &mut pma) };

        // Verify the PMA copy is in offset form
        assert!(!pma_indirect.is_stack_allocated(), "PMA copy should be in offset form");

        // Verify the PMA copy contains correct data
        let pma_ia = pma_indirect.as_indirect().unwrap();
        let pma_size = pma_ia.size_with_arena(pma.arena());
        assert_eq!(pma_size, 2, "PMA indirect atom should have size 2");

        let pma_bytes = pma_ia.as_ne_bytes_with_arena(pma.arena());
        assert_eq!(pma_bytes.len(), 16, "PMA indirect should have 16 bytes of data");

        // Verify actual data values
        let pma_slice = pma_ia.as_slice_with_arena(pma.arena());
        assert_eq!(pma_slice[0], 0xDEADBEEF_CAFEBABE, "First word should match");
        assert_eq!(pma_slice[1], 0x12345678_9ABCDEF0, "Second word should match");

        // Test with cell containing direct atoms
        let stack_cell = Cell::new(&mut stack, D(42), D(99)).as_noun();
        let mut pma_cell = stack_cell;
        unsafe { pma_cell.copy_to_pma(&stack, &mut pma) };

        assert!(!pma_cell.is_stack_allocated(), "PMA cell should be in offset form");
        let cell = pma_cell.as_cell().unwrap();
        assert_eq!(
            cell.head().as_direct().unwrap().data(),
            42,
            "Cell head should be 42"
        );
        assert_eq!(
            cell.tail().as_direct().unwrap().data(),
            99,
            "Cell tail should be 99"
        );

        // Test with nested structure
        let inner = Cell::new(&mut stack, D(1), D(2)).as_noun();
        let stack_nested = Cell::new(&mut stack, inner, D(3)).as_noun();
        let mut pma_nested = stack_nested;
        unsafe { pma_nested.copy_to_pma(&stack, &mut pma) };

        assert!(!pma_nested.is_stack_allocated(), "PMA nested should be in offset form");
        let outer = pma_nested.as_cell().unwrap();
        assert_eq!(
            outer.tail().as_direct().unwrap().data(),
            3,
            "Outer tail should be 3"
        );
        let inner_cell = outer.head().as_cell().unwrap();
        assert_eq!(
            inner_cell.head().as_direct().unwrap().data(),
            1,
            "Inner head should be 1"
        );
        assert_eq!(
            inner_cell.tail().as_direct().unwrap().data(),
            2,
            "Inner tail should be 2"
        );
    }

    /// Test HAMT evacuation with complex noun types: Cells and IndirectAtoms.
    ///
    /// This test exercises:
    /// - HAMT with indirect atoms as keys (large numbers)
    /// - HAMT with cells as values (nested structures)
    /// - Deep cell nesting to test recursive evacuation
    /// - Structural equality verification using a reference copy on a separate stack
    ///
    /// Note: copy_to_pma sets forwarding pointers in source nouns, corrupting them.
    /// To verify values, we create a second NockStack with fresh copies of the same
    /// data and compare those against the PMA copy using noun_equality.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_hamt_complex_nouns() {
        use crate::ext::noun_equality;
        use crate::noun::{Cell, IndirectAtom};

        let mut stack = NockStack::new(1 << 16, 0);
        stack.install_arena();
        let mut pma = test_pma(100000);

        // Create a second stack with reference copies of keys/values for comparison
        // This stack won't be corrupted by forwarding pointers
        let mut ref_stack = NockStack::new(1 << 16, 0);

        // Install PMA arena - this must be the active arena when accessing PMA nouns
        let _guard = pma.install();

        let mut hamt: Hamt<Noun> = Hamt::new(&mut stack);

        // Store reference keys/values on the separate stack
        let mut ref_keys: Vec<Noun> = Vec::new();
        let mut ref_values: Vec<Noun> = Vec::new();

        // Insert entries with indirect atom keys and cell values
        for i in 0u64..5 {
            let key_data: [u64; 2] = [0xDEADBEEF_CAFEBABE + i, 0x12345678_9ABCDEF0 + i];

            // Create on main stack for HAMT
            let key_atom =
                unsafe { IndirectAtom::new_raw(&mut stack, 2, key_data.as_ptr()) }.as_noun();
            let inner = Cell::new(&mut stack, D(i + 100), D(i + 200)).as_noun();
            let value = Cell::new(&mut stack, D(i), inner).as_noun();

            // Create identical copies on reference stack
            let ref_key =
                unsafe { IndirectAtom::new_raw(&mut ref_stack, 2, key_data.as_ptr()) }.as_noun();
            let ref_inner = Cell::new(&mut ref_stack, D(i + 100), D(i + 200)).as_noun();
            let ref_value = Cell::new(&mut ref_stack, D(i), ref_inner).as_noun();
            ref_keys.push(ref_key);
            ref_values.push(ref_value);

            let mut key_copy = key_atom;
            hamt = hamt.insert(&mut stack, &mut key_copy, value);
        }

        // Insert entries with cell keys and indirect atom values
        for i in 5u64..10 {
            let val_data: [u64; 2] = [i * 1000, i * 2000];

            // Create on main stack for HAMT
            let key = Cell::new(&mut stack, D(i), D(i + 1)).as_noun();
            let value =
                unsafe { IndirectAtom::new_raw(&mut stack, 2, val_data.as_ptr()) }.as_noun();

            // Create identical copies on reference stack
            let ref_key = Cell::new(&mut ref_stack, D(i), D(i + 1)).as_noun();
            let ref_value =
                unsafe { IndirectAtom::new_raw(&mut ref_stack, 2, val_data.as_ptr()) }.as_noun();
            ref_keys.push(ref_key);
            ref_values.push(ref_value);

            let mut key_copy = key;
            hamt = hamt.insert(&mut stack, &mut key_copy, value);
        }

        // Insert entries with deeply nested cells
        for i in 10u64..12 {
            // Create on main stack for HAMT
            let ab = Cell::new(&mut stack, D(i), D(i + 1)).as_noun();
            let abc = Cell::new(&mut stack, ab, D(i + 2)).as_noun();
            let key = Cell::new(&mut stack, abc, D(i + 3)).as_noun();
            let zw = Cell::new(&mut stack, D(i + 10), D(i + 11)).as_noun();
            let yzw = Cell::new(&mut stack, D(i + 9), zw).as_noun();
            let value = Cell::new(&mut stack, D(i + 8), yzw).as_noun();

            // Create identical copies on reference stack
            let ref_ab = Cell::new(&mut ref_stack, D(i), D(i + 1)).as_noun();
            let ref_abc = Cell::new(&mut ref_stack, ref_ab, D(i + 2)).as_noun();
            let ref_key = Cell::new(&mut ref_stack, ref_abc, D(i + 3)).as_noun();
            let ref_zw = Cell::new(&mut ref_stack, D(i + 10), D(i + 11)).as_noun();
            let ref_yzw = Cell::new(&mut ref_stack, D(i + 9), ref_zw).as_noun();
            let ref_value = Cell::new(&mut ref_stack, D(i + 8), ref_yzw).as_noun();
            ref_keys.push(ref_key);
            ref_values.push(ref_value);

            let mut key_copy = key;
            hamt = hamt.insert(&mut stack, &mut key_copy, value);
        }

        // Count entries before evacuation
        let count_before: usize = hamt.iter().map(|entries| entries.len()).sum();
        assert_eq!(count_before, 12, "Should have 12 entries before evacuation");

        // Evacuate the HAMT to PMA
        unsafe {
            hamt.copy_to_pma(&stack, &mut pma);
        }

        // Count entries after evacuation
        let count_after: usize = hamt.iter().map(|entries| entries.len()).sum();
        assert_eq!(
            count_after, count_before,
            "Entry count should be preserved after evacuation"
        );

        // Re-install PMA arena (Cell::new on ref_stack may have changed thread-local arena)
        drop(_guard);
        let _guard = pma.install();

        // Verify all values match by comparing PMA nouns to reference stack nouns
        let mut found_count = 0;
        for entries in hamt.iter() {
            for (pma_key, pma_value) in entries {
                // Find matching reference key and verify value matches
                let mut found = false;
                for (idx, ref_key) in ref_keys.iter().enumerate() {
                    if noun_equality(pma_key, ref_key) {
                        assert!(
                            noun_equality(pma_value, &ref_values[idx]),
                            "Value for key {} should match reference after evacuation",
                            idx
                        );
                        found = true;
                        found_count += 1;
                        break;
                    }
                }
                assert!(found, "Every PMA key should match a reference key");
            }
        }
        assert_eq!(
            found_count,
            ref_keys.len(),
            "Should find all {} entries in HAMT after evacuation",
            ref_keys.len()
        );

        // Verify all nouns in the HAMT are in offset form
        for entries in hamt.iter() {
            for (key, value) in entries {
                verify_noun_not_stack_allocated(*key, "HAMT key");
                verify_noun_not_stack_allocated(*value, "HAMT value");
            }
        }

        // Verify the HAMT structure itself is in PMA
        hamt.assert_in_pma(&pma);
    }

    /// Helper to recursively verify a noun is not stack-allocated
    fn verify_noun_not_stack_allocated(noun: Noun, context: &str) {
        if noun.is_direct() {
            return;
        }

        assert!(
            !noun.is_stack_allocated(),
            "{} should be in offset form after evacuation",
            context
        );

        if let Ok(cell) = noun.as_cell() {
            verify_noun_not_stack_allocated(cell.head(), context);
            verify_noun_not_stack_allocated(cell.tail(), context);
        }
    }

    /// Verifies that PmaCopy for () is a no-op that allocates nothing.
    ///
    /// The unit type has no data, so copy_to_pma should not allocate anything
    /// and assert_in_pma should trivially pass.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_unit() {
        let stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);

        let mut unit = ();
        let initial_offset = pma.alloc_offset();

        // Copy to PMA - should be a no-op
        unsafe { unit.copy_to_pma(&stack, &mut pma) };

        // Verify no allocations were made
        assert_eq!(
            pma.alloc_offset(),
            initial_offset,
            "No allocations should be made for unit type"
        );

        // assert_in_pma should not panic
        unit.assert_in_pma(&pma);
    }
}
```

## crates/nockvm/rust/nockvm/src/noun.rs
```
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::{error, fmt, ptr};

use bitvec::prelude::{BitSlice, Lsb0};
use either::{Either, Left, Right};
use ibig::{Stack, UBig};
use intmap::IntMap;
use nockvm_macros::tas;
use static_assertions::assert_cfg;

use crate::mem::{word_size_of, Arena, NockStack};

crate::gdb!();

assert_cfg!(
    target_endian = "little",
    "nockvm will not execute correctly on non-little-endian systems"
);

/** Tag for a direct atom. */
pub(crate) const DIRECT_TAG: u64 = 0x0;

/** Tag mask for a direct atom. */
pub(crate) const DIRECT_MASK: u64 = !(u64::MAX >> 1);

/** Maximum value of a direct atom. Values higher than this must be represented by indirect atoms. */
pub const DIRECT_MAX: u64 = u64::MAX >> 1;

/** Tag for an indirect atom. */
pub(crate) const INDIRECT_TAG: u64 = u64::MAX & DIRECT_MASK;

/** Tag mask for an indirect atom. */
pub(crate) const INDIRECT_MASK: u64 = !(u64::MAX >> 2);

/** Tag for a cell. */
pub(crate) const CELL_TAG: u64 = u64::MAX & INDIRECT_MASK;

/** Tag mask for a cell. */
pub(crate) const CELL_MASK: u64 = !(u64::MAX >> 3);

const LOCATION_BIT: u64 = 1 << 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PtrLocation {
    Stack,
    Offset,
}

#[derive(Debug, Clone, Copy)]
struct TaggedPtr(u64);

impl TaggedPtr {
    #[inline(always)]
    fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[inline(always)]
    unsafe fn from_stack_ptr(ptr: *const u8, tag: u64) -> Self {
        debug_assert!(
            (ptr as usize) & 0x7 == 0,
            "Stack pointer {:p} not 8-byte aligned",
            ptr
        );
        Self(((ptr as u64) >> 3) | tag)
    }

    #[inline(always)]
    fn from_offset(words: u32, tag: u64) -> Self {
        debug_assert!(
            (words as u64) < LOCATION_BIT,
            "offset {} exceeds payload capacity",
            words
        );
        Self((words as u64) | LOCATION_BIT | tag)
    }

    #[inline(always)]
    fn location(self) -> PtrLocation {
        if self.0 & LOCATION_BIT == 0 {
            PtrLocation::Stack
        } else {
            PtrLocation::Offset
        }
    }

    #[inline(always)]
    fn payload(self, mask: u64) -> u64 {
        self.0 & !(mask | LOCATION_BIT)
    }

    fn resolve_const(self, mask: u64, arena: &Arena) -> *const u8 {
        match self.location() {
            PtrLocation::Stack => ((self.payload(mask)) << 3) as *const u8,
            PtrLocation::Offset => arena.ptr_from_offset(self.payload(mask) as u32) as *const u8,
        }
    }

    #[inline(always)]
    fn resolve_mut(self, mask: u64, arena: &Arena) -> *mut u8 {
        self.resolve_const(mask, arena) as *mut u8
    }

    #[inline(always)]
    fn raw(self) -> u64 {
        self.0
    }
}

/*  A note on forwarding pointers:
 *
 *  Forwarding pointers are only used temporarily during copies between NockStack frames and between
 *  the NockStack and the PMA. Since unifying equality checks can create structural sharing between
 *  Noun objects, forwarding pointers act as a signal that a Noun has already been copied to the
 *  "to" space. The old Noun object in the "from" space is given a forwarding pointer so that any
 *  future refernces to the same structure know that it has already been copied and that they should
 *  retain the structural sharing relationship by referencing the new copy in the "to" copy space.
 *
 *  The Nouns in the "from" space marked with forwarding pointers are dangling pointers after a copy
 *  operation. No code outside of the copying code checks for forwarding pointers. This invariant
 *  must be enforced in two ways:
 *      1. The current frame must be immediately popped after preserving data, when
 *          copying from a junior NockStack frame to a senior NockStack frame.
 *      2. All persistent derived state (e.g. Hot state, Warm state) must be preserved
 *          and the root NockStack frame flipped after saving data to the PMA.
 */

/** Tag for a forwarding pointer */
const FORWARDING_TAG: u64 = u64::MAX & CELL_MASK;

/** Tag mask for a forwarding pointer */
const FORWARDING_MASK: u64 = CELL_MASK;

/** Shorthand for 0's that actually are ~ **/
pub const SIG: Noun = D(0);

/** Loobeans */
pub const YES: Noun = D(0);
pub const NO: Noun = D(1);
pub const NONE: Noun = unsafe { DirectAtom::new_unchecked(tas!(b"MORMAGIC")).as_noun() };

#[cfg(feature = "check_acyclic")]
#[macro_export]
macro_rules! assert_acyclic {
    ( $x:expr ) => {
        assert_no_alloc::permit_alloc(|| {
            assert!(crate::noun::acyclic_noun($x));
        })
    };
}

#[cfg(not(feature = "check_acyclic"))]
#[macro_export]
macro_rules! assert_acyclic {
    ( $x:expr ) => {};
}

pub fn acyclic_noun(noun: Noun) -> bool {
    let mut seen = IntMap::new();
    acyclic_noun_go(noun, &mut seen)
}

fn acyclic_noun_go(noun: Noun, seen: &mut IntMap<u64, ()>) -> bool {
    match noun.as_either_atom_cell() {
        Left(_atom) => true,
        Right(cell) => {
            if seen.get(cell.0).is_some() {
                false
            } else {
                seen.insert(cell.0, ());
                if acyclic_noun_go(cell.head(), seen) {
                    if acyclic_noun_go(cell.tail(), seen) {
                        seen.remove(cell.0);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }
}

#[cfg(feature = "check_forwarding")]
#[macro_export]
macro_rules! assert_no_forwarding_pointers {
    ( $x:expr ) => {
        assert_no_alloc::permit_alloc(|| {
            assert!(crate::noun::no_forwarding_pointers($x));
        })
    };
}

#[cfg(not(feature = "check_forwarding"))]
#[macro_export]
macro_rules! assert_no_forwarding_pointers {
    ( $x:expr ) => {};
}

pub fn no_forwarding_pointers(noun: Noun) -> bool {
    let mut dbg_stack = Vec::new();
    dbg_stack.push(noun);

    while !dbg_stack.is_empty() {
        if let Some(noun) = dbg_stack.pop() {
            if unsafe { noun.raw & FORWARDING_MASK == FORWARDING_TAG } {
                return false;
            } else if let Ok(cell) = noun.as_cell() {
                dbg_stack.push(cell.tail());
                dbg_stack.push(cell.head());
            }
        } else {
            break;
        }
    }

    true
}

/** Test if a noun is a direct atom. */
fn is_direct_atom(noun: u64) -> bool {
    noun & DIRECT_MASK == DIRECT_TAG
}

/** Test if a noun is an indirect atom. */
fn is_indirect_atom(noun: u64) -> bool {
    noun & INDIRECT_MASK == INDIRECT_TAG
}

/** Test if a noun is a cell. */
fn is_cell(noun: u64) -> bool {
    noun & CELL_MASK == CELL_TAG
}

/** A noun-related error. */
#[derive(Debug, PartialEq)]
pub enum Error {
    /** Expected type [`Allocated`]. */
    NotAllocated,
    /** Expected type [`Atom`]. */
    NotAtom,
    /** Expected type [`Cell`]. */
    NotCell,
    /** Expected type [`DirectAtom`]. */
    NotDirectAtom,
    /** Expected type [`IndirectAtom`]. */
    NotIndirectAtom,
    /** The value can't be represented by the given type. */
    NotRepresentable,
}

impl error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotAllocated => f.write_str("not an allocated noun"),
            Error::NotAtom => f.write_str("not an atom"),
            Error::NotCell => f.write_str("not a cell"),
            Error::NotDirectAtom => f.write_str("not a direct atom"),
            Error::NotIndirectAtom => f.write_str("not an indirect atom"),
            Error::NotRepresentable => f.write_str("unrepresentable value"),
        }
    }
}

impl From<Error> for () {
    fn from(_: Error) -> Self {}
}

/** A [`Result`] that returns an [`Error`] on error. */
pub type Result<T> = std::result::Result<T, Error>;

/** A direct atom.
 *
 * Direct atoms represent an atom up to and including DIRECT_MAX as a machine word.
 */
#[derive(Copy, Clone)]
#[repr(C)]
#[repr(packed(8))]
pub struct DirectAtom(u64);

impl DirectAtom {
    /** Create a new direct atom, or panic if the value is greater than DIRECT_MAX */
    pub const fn new_panic(value: u64) -> Self {
        if value > DIRECT_MAX {
            panic!("Number is greater than DIRECT_MAX")
        } else {
            DirectAtom(value)
        }
    }

    /** Create a new direct atom, or return Err if the value is greater than DIRECT_MAX */
    pub const fn new(value: u64) -> Result<Self> {
        if value > DIRECT_MAX {
            Err(Error::NotRepresentable)
        } else {
            Ok(DirectAtom(value))
        }
    }

    /** Create a new direct atom. This is unsafe because the value is not checked.
     *
     * Attempting to create a direct atom with a value greater than DIRECT_MAX will
     * result in this value being interpreted by the runtime as a cell or indirect atom,
     * with corresponding memory accesses. Thus, this function is marked as unsafe.
     */
    pub const unsafe fn new_unchecked(value: u64) -> Self {
        DirectAtom(value)
    }

    pub fn bit_size(self) -> usize {
        (64 - self.0.leading_zeros()) as usize
    }

    pub fn as_atom(self) -> Atom {
        Atom { direct: self }
    }

    pub fn as_ubig<S: Stack>(self, _stack: &mut S) -> UBig {
        UBig::from(self.0)
    }

    pub const fn as_noun(self) -> Noun {
        Noun { direct: self }
    }

    pub fn data(self) -> u64 {
        self.0
    }

    pub fn as_bitslice(&self) -> &BitSlice<u64, Lsb0> {
        BitSlice::from_element(&self.0)
    }

    pub fn as_bitslice_mut(&mut self) -> &mut BitSlice<u64, Lsb0> {
        BitSlice::from_element_mut(&mut self.0)
    }

    pub fn as_ne_bytes(&self) -> &[u8] {
        let bytes: &[u8; 8] = unsafe { std::mem::transmute(&self.0) };
        &bytes[..]
    }

    /// Returns Vec<u8> under native-endian of the machine
    pub fn to_ne_bytes(&self) -> Vec<u8> {
        self.as_ne_bytes().to_vec()
    }

    pub fn to_be_bytes(&self) -> Vec<u8> {
        self.0.to_be_bytes().to_vec()
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
}

impl fmt::Debug for DirectAtom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            return write!(f, "0");
        }

        let mut null = false;
        let mut n = 0;
        let bytes = self.0.to_le_bytes();
        for byte in bytes.iter() {
            if *byte == 0 {
                null = true;
                continue;
            }
            if (null && *byte != 0) || *byte < 33 || *byte > 126 {
                return write!(f, "{}", self.0);
            }
            n += 1;
        }
        if n > 1 {
            write!(f, "%{}", unsafe {
                std::str::from_utf8_unchecked(&bytes[..n])
            })
        } else {
            write!(f, "{}", self.0)
        }
    }
}

#[allow(non_snake_case)]
pub const fn D(n: u64) -> Noun {
    DirectAtom::new_panic(n).as_noun()
}

#[allow(non_snake_case)]
pub fn T<A: NounAllocator>(allocator: &mut A, tup: &[Noun]) -> Noun {
    Cell::new_tuple(allocator, tup).as_noun()
}

/// Create $tape Noun from ASCII string
pub fn tape<A: NounAllocator>(allocator: &mut A, text: &str) -> Noun {
    //  XX: Needs unit tests
    let mut res = D(0);
    for c in text.bytes().rev() {
        res = T(allocator, &[D(c as u64), res])
    }
    res
}

/** An indirect atom.
 *
 *  Indirect atoms represent atoms above DIRECT_MAX as a tagged pointer to a memory buffer
 *  structured as:
 *
 *  - first word: metadata
 *  - second word: size in 64-bit words
 *  - remaining words: data
 *
 *  Indirect atoms are always stored in little-endian byte order
 */
#[derive(Copy, Clone)]
#[repr(C)]
#[repr(packed(8))]
pub struct IndirectAtom(u64);

impl IndirectAtom {
    /** Tag the pointer and type it as an indirect atom. */
    pub unsafe fn from_raw_pointer(ptr: *const u64) -> Self {
        IndirectAtom(TaggedPtr::from_stack_ptr(ptr as *const u8, INDIRECT_TAG).raw())
    }

    pub fn from_offset_words(words: u32) -> Self {
        IndirectAtom(TaggedPtr::from_offset(words, INDIRECT_TAG).raw())
    }

    /** Strip the tag from an indirect atom and return it as a mutable pointer to its memory buffer. */
    unsafe fn to_raw_pointer_mut_with_arena(&mut self, arena: &Arena) -> *mut u64 {
        TaggedPtr::from_raw(self.0).resolve_mut(INDIRECT_MASK, arena) as *mut u64
    }

    /** Strip the tag from an indirect atom and return it as a pointer to its memory buffer. */
    pub unsafe fn to_raw_pointer_with_arena(&self, arena: &Arena) -> *const u64 {
        TaggedPtr::from_raw(self.0).resolve_const(INDIRECT_MASK, arena) as *const u64
    }

    pub unsafe fn to_raw_pointer(&self) -> *const u64 {
        Arena::with_current(|arena| self.to_raw_pointer_with_arena(arena))
    }

    /// Get raw pointer for stack-pointer form atoms only
    pub unsafe fn to_raw_pointer_stack(&self) -> *const u64 {
        let tagged = TaggedPtr::from_raw(self.0);
        if tagged.location() == PtrLocation::Stack {
            ((tagged.payload(INDIRECT_MASK)) << 3) as *const u64
        } else {
            panic!("expected stack-pointer Noun, got offset instead");
        }
    }

    /// Get mutable raw pointer for stack-pointer form atoms only
    pub fn to_raw_pointer_mut_stack(&mut self) -> *mut u64 {
        let tagged = TaggedPtr::from_raw(self.0);
        if tagged.location() == PtrLocation::Stack {
            ((tagged.payload(INDIRECT_MASK)) << 3) as *mut u64
        } else {
            panic!("expected stack-pointer Noun, got offset instead");
        }
    }

    pub unsafe fn to_raw_pointer_mut(&mut self) -> *mut u64 {
        Arena::with_current(|arena| self.to_raw_pointer_mut_with_arena(arena))
    }

    pub unsafe fn set_forwarding_pointer_with_arena(&mut self, new_me: *const u64, arena: &Arena) {
        // This is OK because the size is stored as 64 bit words, not bytes.
        // Thus, a true size value will never be larger than U64::MAX >> 3, and so
        // any of the high bits set as an MSB
        *self.to_raw_pointer_mut_with_arena(arena).add(1) =
            TaggedPtr::from_stack_ptr(new_me as *const u8, FORWARDING_TAG).raw();
    }

    pub unsafe fn set_forwarding_pointer(&mut self, new_me: *const u64) {
        Arena::with_current(|arena| self.set_forwarding_pointer_with_arena(new_me, arena))
    }

    pub unsafe fn forwarding_pointer_with_arena(&self, arena: &Arena) -> Option<IndirectAtom> {
        let size_raw = *self.to_raw_pointer_with_arena(arena).add(1);
        if size_raw & FORWARDING_MASK == FORWARDING_TAG {
            let ptr =
                TaggedPtr::from_raw(size_raw).resolve_const(FORWARDING_MASK, arena) as *const u64;
            Some(Self::from_raw_pointer(ptr))
        } else {
            None
        }
    }

    pub unsafe fn forwarding_pointer(&self) -> Option<IndirectAtom> {
        Arena::with_current(|arena| self.forwarding_pointer_with_arena(arena))
    }

    /** Make an indirect atom by copying from other memory.
     *
     *  Note: size is in 64-bit words, not bytes.
     */
    pub unsafe fn new_raw<A: NounAllocator>(
        allocator: &mut A,
        size: usize,
        data: *const u64,
    ) -> Self {
        let (mut indirect, buffer) = Self::new_raw_mut(allocator, size);
        ptr::copy_nonoverlapping(data, buffer, size);
        // Use normalize_stack since new_raw_mut creates stack-pointer form atoms
        *(indirect.normalize_stack())
    }

    /** Make an indirect atom by copying from other memory.
     *
     *  Note: size is bytes, not words
     */
    pub unsafe fn new_raw_bytes<A: NounAllocator>(
        allocator: &mut A,
        size: usize,
        data: *const u8,
    ) -> Self {
        let (mut indirect, buffer) = Self::new_raw_mut_bytes(allocator, size);
        ptr::copy_nonoverlapping(data, buffer.as_mut_ptr(), size);
        // Use normalize_stack since new_raw_mut_bytes creates stack-pointer form atoms
        *(indirect.normalize_stack())
    }

    pub unsafe fn new_raw_bytes_ref<A: NounAllocator>(allocator: &mut A, data: &[u8]) -> Self {
        IndirectAtom::new_raw_bytes(allocator, data.len(), data.as_ptr())
    }

    /** Make an indirect atom that can be written into. Return the atom (which should not be used
     * until it is written and normalized) and a mutable pointer which is the data buffer for the
     * indirect atom, to be written into.
     */
    pub unsafe fn new_raw_mut<A: NounAllocator>(
        allocator: &mut A,
        size: usize,
    ) -> (Self, *mut u64) {
        debug_assert!(size > 0);
        let buffer = allocator.alloc_indirect(size);
        *buffer = 0;
        *buffer.add(1) = size as u64;
        (Self::from_raw_pointer(buffer), buffer.add(2))
    }

    /** Make an indirect atom that can be written into, and zero the whole data buffer.
     * Return the atom (which should not be used until it is written and normalized) and a mutable
     * pointer which is the data buffer for the indirect atom, to be written into.
     */
    pub unsafe fn new_raw_mut_zeroed<A: NounAllocator>(
        allocator: &mut A,
        size: usize,
    ) -> (Self, *mut u64) {
        let allocation = Self::new_raw_mut(allocator, size);
        ptr::write_bytes(allocation.1, 0, size);
        allocation
    }

    /** Make an indirect atom that can be written into as a bitslice. The constraints of
     * [new_raw_mut_zeroed] also apply here
     */
    pub unsafe fn new_raw_mut_bitslice<'a, A: NounAllocator>(
        allocator: &mut A,
        size: usize,
    ) -> (Self, &'a mut BitSlice<u64, Lsb0>) {
        let (noun, ptr) = Self::new_raw_mut_zeroed(allocator, size);
        (
            noun,
            BitSlice::from_slice_mut(from_raw_parts_mut(ptr, size)),
        )
    }

    /** Make an indirect atom that can be written into as a slice of bytes. The constraints of
     * [new_raw_mut_zeroed] also apply here
     *
     * Note: size is bytes, not words
     */
    pub unsafe fn new_raw_mut_bytes<'a, A: NounAllocator>(
        allocator: &mut A,
        size: usize,
    ) -> (Self, &'a mut [u8]) {
        let word_size = (size + 7) >> 3;
        let (noun, ptr) = Self::new_raw_mut_zeroed(allocator, word_size);
        (noun, from_raw_parts_mut(ptr as *mut u8, size))
    }

    /// Create an indirect atom backed by a fixed-size array
    pub unsafe fn new_raw_mut_bytearray<'a, const N: usize, A: NounAllocator>(
        allocator: &mut A,
    ) -> (Self, &'a mut [u8; N]) {
        let word_size = (std::mem::size_of::<[u8; N]>() + 7) >> 3;
        let (noun, ptr) = Self::new_raw_mut_zeroed(allocator, word_size);
        (noun, &mut *(ptr as *mut [u8; N]))
    }

    /** Size of an indirect atom in 64-bit words */
    pub fn size_with_arena(&self, arena: &Arena) -> usize {
        unsafe { *(self.to_raw_pointer_with_arena(arena).add(1)) as usize }
    }

    pub fn size(&self) -> usize {
        Arena::with_current(|arena| self.size_with_arena(arena))
    }

    /** Memory size of an indirect atom (including size + metadata fields) in 64-bit words */
    pub fn raw_size_with_arena(&self, arena: &Arena) -> usize {
        self.size_with_arena(arena) + 2
    }

    pub fn raw_size(&self) -> usize {
        Arena::with_current(|arena| self.raw_size_with_arena(arena))
    }

    pub fn bit_size_with_arena(&self, arena: &Arena) -> usize {
        unsafe {
            ((self.size_with_arena(arena) - 1) << 6) + 64
                - (*(self
                    .to_raw_pointer_with_arena(arena)
                    .add(2 + self.size_with_arena(arena) - 1)))
                .leading_zeros() as usize
        }
    }

    pub fn bit_size(&self) -> usize {
        Arena::with_current(|arena| self.bit_size_with_arena(arena))
    }

    /** Pointer to data for indirect atom */
    pub fn data_pointer_with_arena(&self, arena: &Arena) -> *const u64 {
        unsafe { self.to_raw_pointer_with_arena(arena).add(2) }
    }

    pub fn data_pointer(&self) -> *const u64 {
        Arena::with_current(|arena| self.data_pointer_with_arena(arena))
    }

    pub fn data_pointer_mut_with_arena(&mut self, arena: &Arena) -> *mut u64 {
        unsafe { self.to_raw_pointer_mut_with_arena(arena).add(2) }
    }

    pub fn data_pointer_mut(&mut self) -> *mut u64 {
        Arena::with_current(|arena| self.data_pointer_mut_with_arena(arena))
    }

    pub fn data_pointer_stack(&self) -> Option<*const u64> {
        let tagged = TaggedPtr::from_raw(self.0);
        if tagged.location() == PtrLocation::Stack {
            Some(((tagged.payload(INDIRECT_MASK)) << 3) as *const u64)
        } else {
            None
        }
    }

    pub fn as_slice_with_arena(&self, arena: &Arena) -> &[u64] {
        unsafe {
            from_raw_parts(
                self.data_pointer_with_arena(arena),
                self.size_with_arena(arena),
            )
        }
    }

    pub fn as_slice(&self) -> &[u64] {
        Arena::with_current(|arena| self.as_slice_with_arena(arena))
    }

    pub fn as_mut_slice_with_arena(&mut self, arena: &Arena) -> &mut [u64] {
        unsafe {
            from_raw_parts_mut(
                self.data_pointer_mut_with_arena(arena),
                self.size_with_arena(arena),
            )
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        Arena::with_current(|arena| self.as_mut_slice_with_arena(arena))
    }

    pub fn as_ne_bytes_with_arena(&self, arena: &Arena) -> &[u8] {
        unsafe {
            from_raw_parts(
                self.data_pointer_with_arena(arena) as *const u8,
                self.size_with_arena(arena) << 3,
            )
        }
    }

    pub fn as_ne_bytes(&self) -> &[u8] {
        Arena::with_current(|arena| self.as_ne_bytes_with_arena(arena))
    }

    pub fn to_ne_bytes_with_arena(&self, arena: &Arena) -> Vec<u8> {
        self.as_ne_bytes_with_arena(arena).to_vec()
    }

    pub fn to_ne_bytes(&self) -> Vec<u8> {
        Arena::with_current(|arena| self.to_ne_bytes_with_arena(arena))
    }

    #[allow(unused)]
    pub fn to_be_bytes_with_arena(&self, arena: &Arena) -> Vec<u8> {
        if self.size_with_arena(arena) == 1 {
            let num = unsafe { *(self.data_pointer_with_arena(arena)) };
            num.to_be_bytes().to_vec()
        } else {
            let mut bytes_ne = self.to_ne_bytes_with_arena(arena);
            #[cfg(target_endian = "little")]
            {
                bytes_ne.reverse()
            }
            bytes_ne
        }
    }

    #[allow(unused)]
    pub fn to_be_bytes(&self) -> Vec<u8> {
        Arena::with_current(|arena| self.to_be_bytes_with_arena(arena))
    }

    #[allow(unused)]
    pub fn to_le_bytes_with_arena(&self, arena: &Arena) -> Vec<u8> {
        if self.size_with_arena(arena) == 1 {
            let num = unsafe { *(self.data_pointer_with_arena(arena)) };
            num.to_le_bytes().to_vec()
        } else {
            let mut bytes_ne = self.to_ne_bytes_with_arena(arena);
            #[cfg(target_endian = "big")]
            {
                bytes_ne.reverse()
            }

            bytes_ne
        }
    }

    #[allow(unused)]
    pub fn to_le_bytes(&self) -> Vec<u8> {
        Arena::with_current(|arena| self.to_le_bytes_with_arena(arena))
    }

    /** BitSlice view on an indirect atom, with lifetime tied to reference to indirect atom. */
    pub fn as_bitslice_with_arena(&self, arena: &Arena) -> &BitSlice<u64, Lsb0> {
        BitSlice::from_slice(self.as_slice_with_arena(arena))
    }

    pub fn as_bitslice(&self) -> &BitSlice<u64, Lsb0> {
        Arena::with_current(|arena| self.as_bitslice_with_arena(arena))
    }

    pub fn as_bitslice_mut_with_arena(&mut self, arena: &Arena) -> &mut BitSlice<u64, Lsb0> {
        BitSlice::from_slice_mut(self.as_mut_slice_with_arena(arena))
    }

    pub fn as_bitslice_mut(&mut self) -> &mut BitSlice<u64, Lsb0> {
        Arena::with_current(|arena| self.as_bitslice_mut_with_arena(arena))
    }

    pub fn as_ubig_with_arena<S: Stack>(&self, stack: &mut S, arena: &Arena) -> UBig {
        let bytes_mem_repr = self.as_ne_bytes_with_arena(arena);

        #[cfg(target_endian = "little")]
        {
            UBig::from_le_bytes_stack(stack, bytes_mem_repr)
        }
        #[cfg(not(target_endian = "little"))]
        {
            UBig::from_be_bytes_stack(stack, bytes_mem_repr)
        }
    }

    pub fn as_ubig<S: Stack>(&self, stack: &mut S) -> UBig {
        Arena::with_current(|arena| self.as_ubig_with_arena(stack, arena))
    }

    pub unsafe fn as_u64(self) -> Result<u64> {
        if self.size() == 1 {
            Ok(*(self.data_pointer()))
        } else {
            Err(Error::NotRepresentable)
        }
    }

    /** Produce a SoftFloat-compatible ordered pair of 64-bit words */
    pub fn as_u64_pair(self) -> Result<[u64; 2]> {
        if self.size() <= 2 {
            let u128_array = &mut [0u64; 2];
            u128_array.copy_from_slice(&(self.as_slice()[0..2]));
            Ok(*u128_array)
        } else {
            Err(Error::NotRepresentable)
        }
    }

    /** Ensure that the size does not contain any trailing 0 words */
    pub unsafe fn normalize(&mut self) -> &Self {
        let mut index = self.size() - 1;
        let data = self.data_pointer();
        loop {
            if index == 0 || *(data.add(index)) != 0 {
                break;
            }
            index -= 1;
        }
        *(self.to_raw_pointer_mut().add(1)) = (index + 1) as u64;
        self
    }

    /// Normalize a stack-pointer form indirect atom (no arena needed).
    /// Panics if the atom is in offset form.
    pub unsafe fn normalize_stack(&mut self) -> &Self {
        let ptr = self
            .to_raw_pointer_mut_stack();
        let mut index = (*(ptr.add(1)) as usize) - 1; // size is at offset 1
        let data = ptr.add(2); // data starts at offset 2
        loop {
            if index == 0 || *(data.add(index)) != 0 {
                break;
            }
            index -= 1;
        }
        *(ptr.add(1)) = (index + 1) as u64;
        self
    }

    /** Normalize, but convert to direct atom if it will fit */
    pub unsafe fn normalize_as_atom(&mut self) -> Atom {
        self.normalize();
        if self.size() == 1 && *(self.data_pointer()) <= DIRECT_MAX {
            Atom {
                direct: DirectAtom(*(self.data_pointer())),
            }
        } else {
            Atom { indirect: *self }
        }
    }

    /// Normalize a stack-pointer form atom, converting to direct if it fits.
    /// Panics if the atom is in offset form.
    pub unsafe fn normalize_as_atom_stack(&mut self) -> Atom {
        self.normalize_stack();
        let ptr = self
            .to_raw_pointer_stack();
        let size = *(ptr.add(1)) as usize;
        let data = ptr.add(2);
        if size == 1 && *data <= DIRECT_MAX {
            Atom {
                direct: DirectAtom(*data),
            }
        } else {
            Atom { indirect: *self }
        }
    }

    pub fn as_atom(self) -> Atom {
        Atom { indirect: self }
    }

    pub fn as_allocated(self) -> Allocated {
        Allocated { indirect: self }
    }

    pub fn as_noun(self) -> Noun {
        Noun { indirect: self }
    }
}

// XX: Need a version that either:
//      a) allocates on the NockStack directly for creating a tape (or even a string?)
//      b) disables no-allocation, creates a string, utilitzes it (eprintf or generate tape), and then deallocates
impl fmt::Debug for IndirectAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x")?;
        let mut i = self.size() - 1;
        loop {
            write!(f, "_{:016x}", unsafe { *(self.data_pointer().add(i)) })?;
            if i == 0 {
                break;
            }
            i -= 1;
        }
        Ok(())
    }
}

/**
 * A cell.
 *
 * A cell is represented by a tagged pointer to a memory buffer with metadata, a word describing
 * the noun which is the cell's head, and a word describing a noun which is the cell's tail, each
 * at a fixed offset.
 */
#[derive(Copy, Clone)]
#[repr(C)]
#[repr(packed(8))]
pub struct Cell(u64);

impl Cell {
    pub unsafe fn from_raw_pointer(ptr: *const CellMemory) -> Self {
        Cell(TaggedPtr::from_stack_ptr(ptr as *const u8, CELL_TAG).raw())
    }

    pub fn from_offset_words(words: u32) -> Self {
        Cell(TaggedPtr::from_offset(words, CELL_TAG).raw())
    }

    pub unsafe fn to_raw_pointer_with_arena(&self, arena: &Arena) -> *const CellMemory {
        TaggedPtr::from_raw(self.0).resolve_const(CELL_MASK, arena) as *const CellMemory
    }

    pub unsafe fn to_raw_pointer(&self) -> *const CellMemory {
        Arena::with_current(|arena| self.to_raw_pointer_with_arena(arena))
    }

    pub unsafe fn to_raw_pointer_mut_with_arena(&mut self, arena: &Arena) -> *mut CellMemory {
        TaggedPtr::from_raw(self.0).resolve_mut(CELL_MASK, arena) as *mut CellMemory
    }

    pub unsafe fn to_raw_pointer_mut(&mut self) -> *mut CellMemory {
        Arena::with_current(|arena| self.to_raw_pointer_mut_with_arena(arena))
    }

    #[inline(always)]
    pub fn stack_memory_pointer(&self) -> Option<*const CellMemory> {
        let tagged = TaggedPtr::from_raw(self.0);
        if tagged.location() == PtrLocation::Stack {
            Some(((tagged.payload(CELL_MASK)) << 3) as *const CellMemory)
        } else {
            None
        }
    }

    pub unsafe fn head_as_mut_with_arena(mut self, arena: &Arena) -> *mut Noun {
        &mut (*self.to_raw_pointer_mut_with_arena(arena)).head as *mut Noun
    }

    pub unsafe fn head_as_mut(self) -> *mut Noun {
        Arena::with_current(|arena| self.head_as_mut_with_arena(arena))
    }

    pub unsafe fn tail_as_mut_with_arena(mut self, arena: &Arena) -> *mut Noun {
        &mut (*self.to_raw_pointer_mut_with_arena(arena)).tail as *mut Noun
    }

    pub unsafe fn tail_as_mut(self) -> *mut Noun {
        Arena::with_current(|arena| self.tail_as_mut_with_arena(arena))
    }

    pub unsafe fn set_forwarding_pointer_with_arena(
        &mut self,
        new_me: *const CellMemory,
        arena: &Arena,
    ) {
        (*self.to_raw_pointer_mut_with_arena(arena)).head = Noun {
            raw: TaggedPtr::from_stack_ptr(new_me as *const u8, FORWARDING_TAG).raw(),
        }
    }

    pub unsafe fn set_forwarding_pointer(&mut self, new_me: *const CellMemory) {
        Arena::with_current(|arena| self.set_forwarding_pointer_with_arena(new_me, arena))
    }

    pub unsafe fn forwarding_pointer_with_arena(&self, arena: &Arena) -> Option<Cell> {
        let head_raw = (*self.to_raw_pointer_with_arena(arena)).head.raw;
        if head_raw & FORWARDING_MASK == FORWARDING_TAG {
            let ptr = TaggedPtr::from_raw(head_raw).resolve_const(FORWARDING_MASK, arena)
                as *const CellMemory;
            Some(Self::from_raw_pointer(ptr))
        } else {
            None
        }
    }

    pub unsafe fn forwarding_pointer(&self) -> Option<Cell> {
        Arena::with_current(|arena| self.forwarding_pointer_with_arena(arena))
    }

    pub fn new<T: NounAllocator>(allocator: &mut T, head: Noun, tail: Noun) -> Cell {
        unsafe {
            let (cell, memory) = Self::new_raw_mut(allocator);
            (*memory).head = head;
            (*memory).tail = tail;
            cell
        }
    }

    pub fn new_tuple<A: NounAllocator>(allocator: &mut A, tup: &[Noun]) -> Cell {
        if tup.len() < 2 {
            panic!("Cannot create tuple with fewer than 2 elements");
        }

        let len = tup.len();
        let mut cell = Cell::new(allocator, tup[len - 2], tup[len - 1]);
        for i in (0..len - 2).rev() {
            cell = Cell::new(allocator, tup[i], cell.as_noun());
        }
        cell
    }

    pub unsafe fn new_raw_mut<A: NounAllocator>(allocator: &mut A) -> (Cell, *mut CellMemory) {
        let memory = allocator.alloc_cell();
        assert!(
            memory as usize % std::mem::align_of::<CellMemory>() == 0,
            "Memory is not aligned, {} {}",
            memory as usize,
            std::mem::align_of::<CellMemory>()
        );
        (*memory).metadata = 0;
        (Self::from_raw_pointer(memory), memory)
    }

    // TODO: idk about making these owned independently of their parent
    pub fn head_with_arena(&self, arena: &Arena) -> Noun {
        unsafe { (*(self.to_raw_pointer_with_arena(arena))).head }
    }

    pub fn head(&self) -> Noun {
        Arena::with_current(|arena| self.head_with_arena(arena))
    }

    // TODO: Ditto, etc.
    pub fn tail_with_arena(&self, arena: &Arena) -> Noun {
        unsafe { (*(self.to_raw_pointer_with_arena(arena))).tail }
    }

    pub fn tail(&self) -> Noun {
        Arena::with_current(|arena| self.tail_with_arena(arena))
    }

    pub fn head_ref_with_arena<'a>(&'a self, arena: &'a Arena) -> &'a Noun {
        unsafe {
            self.to_raw_pointer_with_arena(arena)
                .as_ref()
                .map(|cell| &cell.head)
                .unwrap_or_else(|| panic!("head_ref: invalid pointer"))
        }
    }

    pub fn head_ref(&self) -> &Noun {
        let ptr = Arena::with_current(|arena| unsafe {
            self.to_raw_pointer_with_arena(arena)
                .as_ref()
                .map(|cell| &cell.head as *const Noun)
                .unwrap_or_else(|| panic!("head_ref: invalid pointer"))
        });
        unsafe { &*ptr }
    }

    // TODO: Ditto, etc.
    pub fn tail_ref_with_arena<'a>(&'a self, arena: &'a Arena) -> &'a Noun {
        unsafe {
            self.to_raw_pointer_with_arena(arena)
                .as_ref()
                .map(|cell| &cell.tail)
                .unwrap_or_else(|| panic!("head_ref: invalid pointer"))
        }
    }

    pub fn tail_ref(&self) -> &Noun {
        let ptr = Arena::with_current(|arena| unsafe {
            self.to_raw_pointer_with_arena(arena)
                .as_ref()
                .map(|cell| &cell.tail as *const Noun)
                .unwrap_or_else(|| panic!("head_ref: invalid pointer"))
        });
        unsafe { &*ptr }
    }

    pub fn as_allocated(&self) -> Allocated {
        Allocated { cell: *self }
    }

    pub fn as_noun(&self) -> Noun {
        Noun { cell: *self }
    }
}

impl fmt::Debug for Cell {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        let cell = *self;
        write!(f, "{:?},", cell.head())?;
        write!(f, " {:?}]", unsafe { cell.tail().raw })?;
        Ok(())
    }
}

pub struct FullDebugCell<'a>(pub &'a Cell);

impl fmt::Debug for FullDebugCell<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn do_fmt(cell: &Cell, brackets: bool, f: &mut fmt::Formatter) -> fmt::Result {
            if brackets {
                write!(f, "[")?;
            }
            match cell.head().as_cell() {
                Ok(head_cell) => {
                    do_fmt(&head_cell, true, f)?;
                    write!(f, " ")?;
                }
                Err(_) => {
                    write!(f, "{:?} ", cell.head())?;
                }
            }
            match cell.tail().as_cell() {
                Ok(next_cell) => {
                    do_fmt(&next_cell, false, f)?;
                }
                Err(_) => {
                    write!(f, "{:?}", cell.tail())?;
                }
            }
            if brackets {
                write!(f, "]")?;
            }
            Ok(())
        }

        do_fmt(&*self.0, true, f)?;
        Ok(())
    }
}

// Render a path which is a linked-list of cells of of atoms (direct and indirect strings)
pub struct DebugPath<'a>(pub &'a Cell);

impl fmt::Debug for DebugPath<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        let mut cell = *self.0;
        loop {
            let head = cell.head().as_atom();
            match head {
                Ok(atom) => {
                    if atom.is_direct() {
                        write!(f, "{:?}", atom.as_direct())?;
                    } else if atom.is_indirect() {
                        write!(f, "{:?}", atom.as_indirect())?;
                    } else {
                        write!(f, "{atom:?}")?;
                    }
                }
                Err(_) => {
                    write!(f, "ERR, not atom")?;
                }
            }
            match cell.tail().as_cell() {
                Ok(next_cell) => {
                    write!(f, " ")?;
                    cell = next_cell;
                }
                Err(_) => {
                    write!(f, " {:?}]", cell.tail())?;
                    break;
                }
            }
        }
        Ok(())
    }
}

// Axis iteration helpers for direct axes (u64)
pub struct DirectAxisIterator {
    axis: u64,
    cursor: usize,
}

impl DirectAxisIterator {
    #[inline(always)]
    pub fn new(axis: u64) -> Option<Self> {
        if axis == 0 {
            None
        } else {
            let cursor = if axis == 1 {
                0
            } else {
                63 - axis.leading_zeros() as usize
            };
            Some(DirectAxisIterator { axis, cursor })
        }
    }

    #[inline(always)]
    pub fn next(&mut self) -> Option<bool> {
        if self.cursor == 0 {
            None
        } else {
            self.cursor -= 1;
            Some(((self.axis >> self.cursor) & 1) != 0)
        }
    }
}

// Axis iteration helpers for indirect axes (slice of u64)
pub struct IndirectAxisIterator<'a> {
    words: &'a [u64],
    cursor: usize,
}

impl<'a> IndirectAxisIterator<'a> {
    #[inline(always)]
    pub fn new(words: &'a [u64]) -> Option<Self> {
        if words.is_empty() {
            return None;
        }

        // Find highest bit in the axis
        let mut highest_word_idx = words.len() - 1;
        while highest_word_idx > 0 && words[highest_word_idx] == 0 {
            highest_word_idx -= 1;
        }

        let highest_word = words[highest_word_idx];
        if highest_word == 0 {
            return None;
        }

        let highest_bit_in_word = 63 - highest_word.leading_zeros() as usize;
        let cursor = (highest_word_idx << 6) + highest_bit_in_word;

        Some(IndirectAxisIterator { words, cursor })
    }

    #[inline(always)]
    pub fn next(&mut self) -> Option<bool> {
        if self.cursor == 0 {
            None
        } else {
            self.cursor -= 1;
            let word_idx = self.cursor >> 6;
            let bit_idx = self.cursor & 63;
            Some(((self.words[word_idx] >> bit_idx) & 1) != 0)
        }
    }
}

// Direct axis traversal without bitvec - for u64 axes
#[inline(always)]
fn slot_direct(cell: &Cell, axis: u64) -> Result<Noun> {
    if axis == 0 {
        return Err(Error::NotRepresentable);
    }
    if axis == 1 {
        return Ok(cell.as_noun());
    }

    let highest = 63 - axis.leading_zeros() as usize;
    let mut current = *cell;
    let mut noun = current.as_noun();

    for idx in (0..highest).rev() {
        let descend_tail = ((axis >> idx) & 1) != 0;
        let memory = unsafe { current.to_raw_pointer() };
        noun = unsafe {
            if descend_tail {
                (*memory).tail
            } else {
                (*memory).head
            }
        };

        if idx != 0 {
            if noun.is_cell() {
                current = unsafe { noun.cell };
            } else {
                return Err(Error::NotRepresentable);
            }
        }
    }

    Ok(noun)
}

impl Slots for Cell {}

// Indirect axis traversal - for large axes stored in word slices
#[inline(always)]
fn slot_indirect(cell: &Cell, words: &[u64]) -> Result<Noun> {
    if words.is_empty() {
        return Err(Error::NotRepresentable);
    }

    // Find highest bit in the axis
    let mut highest_word_idx = words.len() - 1;
    while highest_word_idx > 0 && words[highest_word_idx] == 0 {
        highest_word_idx -= 1;
    }

    let highest_word = words[highest_word_idx];
    if highest_word == 0 {
        return Err(Error::NotRepresentable);
    }

    let highest_bit_in_word = 63 - highest_word.leading_zeros() as usize;
    let highest = (highest_word_idx << 6) + highest_bit_in_word;

    if highest == 0 {
        return Ok(cell.as_noun());
    }

    let mut current = *cell;
    let mut noun = current.as_noun();
    let mut idx = highest;

    while idx != 0 {
        idx -= 1;
        let word_idx = idx >> 6;
        let bit_idx = idx & 63;
        let descend_tail = ((words[word_idx] >> bit_idx) & 1) != 0;

        let memory = unsafe { current.to_raw_pointer() };
        noun = unsafe {
            if descend_tail {
                (*memory).tail
            } else {
                (*memory).head
            }
        };

        if idx != 0 {
            if noun.is_cell() {
                current = unsafe { noun.cell };
            } else {
                return Err(Error::NotRepresentable);
            }
        }
    }

    Ok(noun)
}

impl private::RawSlots for Cell {
    #[inline(always)]
    fn raw_slot_direct(&self, axis: u64) -> Result<Noun> {
        slot_direct(self, axis)
    }

    #[inline(always)]
    fn raw_slot_indirect(&self, axis: &[u64]) -> Result<Noun> {
        slot_indirect(self, axis)
    }
}

/**
 * Memory representation of the contents of a cell
 */
#[derive(Copy, Clone)]
#[repr(C)]
#[repr(packed(8))]
pub struct CellMemory {
    pub metadata: u64,
    pub head: Noun,
    pub tail: Noun,
}

#[derive(Copy, Clone)]
#[repr(C)]
#[repr(packed(8))]
pub union Atom {
    pub(crate) raw: u64,
    direct: DirectAtom,
    indirect: IndirectAtom,
}

impl Atom {
    pub fn new<A: NounAllocator>(allocator: &mut A, value: u64) -> Atom {
        if value <= DIRECT_MAX {
            unsafe { DirectAtom::new_unchecked(value).as_atom() }
        } else {
            unsafe { IndirectAtom::new_raw(allocator, 1, &value).as_atom() }
        }
    }

    // to_le_bytes and new_raw are copies.  We should be able to do this completely without copies
    // if we integrate with ibig properly.
    pub fn from_ubig<A: NounAllocator>(allocator: &mut A, big: &UBig) -> Atom {
        let bit_size = big.bit_len();
        let buffer = big.to_le_bytes_stack();
        if bit_size < 64 {
            let mut value = 0u64;
            for i in (0..bit_size).step_by(8) {
                value |= (buffer[i / 8] as u64) << i;
            }
            unsafe { DirectAtom::new_unchecked(value).as_atom() }
        } else {
            let byte_size = (big.bit_len() + 7) >> 3;
            unsafe { IndirectAtom::new_raw_bytes(allocator, byte_size, buffer.as_ptr()).as_atom() }
        }
    }

    pub fn is_direct(&self) -> bool {
        unsafe { is_direct_atom(self.raw) }
    }

    pub fn is_indirect(&self) -> bool {
        unsafe { is_indirect_atom(self.raw) }
    }

    pub fn is_normalized(&self) -> bool {
        unsafe {
            if let Some(indirect) = self.indirect() {
                if (indirect.size() == 1 && *indirect.data_pointer() <= DIRECT_MAX)
                    || *indirect.data_pointer().add(indirect.size() - 1) == 0
                {
                    return false;
                }
            } // nothing to do for direct atom
        };

        true
    }

    pub fn as_direct(&self) -> Result<DirectAtom> {
        if self.is_direct() {
            unsafe { Ok(self.direct) }
        } else {
            Err(Error::NotDirectAtom)
        }
    }

    pub fn as_indirect(&self) -> Result<IndirectAtom> {
        if self.is_indirect() {
            unsafe { Ok(self.indirect) }
        } else {
            Err(Error::NotIndirectAtom)
        }
    }

    pub fn as_either(&self) -> Either<DirectAtom, IndirectAtom> {
        if self.is_indirect() {
            unsafe { Right(self.indirect) }
        } else {
            unsafe { Left(self.direct) }
        }
    }

    pub fn as_noun(self) -> Noun {
        Noun { atom: self }
    }

    /// Returns a slice of bytes in native-endian order. Currently, Sword only supports
    /// little-endian machines, so this will return little-endian.
    pub fn as_ne_bytes(&self) -> &[u8] {
        if self.is_direct() {
            unsafe { self.direct.as_ne_bytes() }
        } else {
            unsafe { self.indirect.as_ne_bytes() }
        }
    }

    /// Returns Vec<u8> in native-endian order
    pub fn to_ne_bytes(&self) -> Vec<u8> {
        if self.is_direct() {
            unsafe { self.direct.to_ne_bytes() }
        } else {
            unsafe { self.indirect.to_ne_bytes() }
        }
    }

    /// Returns Vec<u8> in big-endian order
    pub fn to_be_bytes(self) -> Vec<u8> {
        if self.is_direct() {
            unsafe { self.direct.to_be_bytes() }
        } else {
            unsafe { self.indirect.to_be_bytes() }
        }
    }

    /// Returns Vec<u8> in little-endian order
    pub fn to_le_bytes(self) -> Vec<u8> {
        if self.is_direct() {
            unsafe { self.direct.to_le_bytes() }
        } else {
            unsafe { self.indirect.to_le_bytes() }
        }
    }

    pub fn as_u64(self) -> Result<u64> {
        if self.is_direct() {
            Ok(unsafe { self.direct.data() })
        } else {
            unsafe { self.indirect.as_u64() }
        }
    }

    pub fn as_bool(self) -> Result<bool> {
        if self.is_direct() {
            Ok(unsafe { self.direct.data() == 0 })
        } else {
            Err(Error::NotRepresentable)
        }
    }

    /** Produce a SoftFloat-compatible ordered pair of 64-bit words */
    pub unsafe fn as_u64_pair(self) -> Result<[u64; 2]> {
        if self.is_direct() {
            let u128_array = &mut [0u64; 2];
            u128_array[0] = self.as_direct()?.data();
            u128_array[1] = 0x0_u64;
            Ok(*u128_array)
        } else {
            unsafe { self.indirect.as_u64_pair() }
        }
    }

    pub fn as_bitslice(&self) -> &BitSlice<u64, Lsb0> {
        if self.is_indirect() {
            unsafe { self.indirect.as_bitslice() }
        } else {
            unsafe { self.direct.as_bitslice() }
        }
    }

    pub fn as_bitslice_mut(&mut self) -> &mut BitSlice<u64, Lsb0> {
        if self.is_indirect() {
            unsafe { self.indirect.as_bitslice_mut() }
        } else {
            unsafe { self.direct.as_bitslice_mut() }
        }
    }

    pub fn as_ubig<S: Stack>(self, stack: &mut S) -> UBig {
        if self.is_indirect() {
            unsafe { self.indirect.as_ubig(stack) }
        } else {
            unsafe { self.direct.as_ubig(stack) }
        }
    }

    pub fn direct(&self) -> Option<DirectAtom> {
        if self.is_direct() {
            unsafe { Some(self.direct) }
        } else {
            None
        }
    }

    pub fn indirect(&self) -> Option<IndirectAtom> {
        if self.is_indirect() {
            unsafe { Some(self.indirect) }
        } else {
            None
        }
    }

    pub fn size(&self) -> usize {
        match self.as_either() {
            Left(_direct) => 1,
            Right(indirect) => indirect.size(),
        }
    }

    pub fn bit_size(&self) -> usize {
        match self.as_either() {
            Left(direct) => direct.bit_size(),
            Right(indirect) => indirect.bit_size(),
        }
    }

    pub fn data_pointer(&self) -> *const u64 {
        match self.as_either() {
            Left(_direct) => (self as *const Atom) as *const u64,
            Right(indirect) => indirect.data_pointer(),
        }
    }

    pub unsafe fn normalize(&mut self) -> Atom {
        if self.is_indirect() {
            self.indirect.normalize_as_atom()
        } else {
            *self
        }
    }

    /** Make an atom from a raw u64
     *
     * # Safety
     *
     * Note that the [u64] parameter is *not*, in general, the value of the atom!
     *
     * In particular, anything with the high bit set will be treated as a tagged pointer.
     * This method is only to be used to restore an atom from the raw [u64] representation
     * returned by [Noun::as_raw], and should only be used if we are sure the restored noun is in
     * fact an atom.
     */
    pub unsafe fn from_raw(raw: u64) -> Atom {
        Atom { raw }
    }
}

impl fmt::Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_noun().fmt(f)
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
#[repr(packed(8))]
pub union Allocated {
    raw: u64,
    indirect: IndirectAtom,
    cell: Cell,
}

impl Allocated {
    pub fn is_indirect(&self) -> bool {
        unsafe { is_indirect_atom(self.raw) }
    }

    pub fn is_cell(&self) -> bool {
        unsafe { is_cell(self.raw) }
    }

    pub unsafe fn to_raw_pointer_with_arena(&self, arena: &Arena) -> *const u64 {
        let tagged = TaggedPtr::from_raw(self.raw);
        if self.is_indirect() {
            tagged.resolve_const(INDIRECT_MASK, arena) as *const u64
        } else {
            tagged.resolve_const(CELL_MASK, arena) as *const u64
        }
    }

    pub unsafe fn to_raw_pointer(&self) -> *const u64 {
        Arena::with_current(|arena| self.to_raw_pointer_with_arena(arena))
    }

    pub unsafe fn to_raw_pointer_mut_with_arena(&mut self, arena: &Arena) -> *mut u64 {
        let tagged = TaggedPtr::from_raw(self.raw);
        if self.is_indirect() {
            tagged.resolve_mut(INDIRECT_MASK, arena) as *mut u64
        } else {
            tagged.resolve_mut(CELL_MASK, arena) as *mut u64
        }
    }

    pub unsafe fn to_raw_pointer_mut(&mut self) -> *mut u64 {
        Arena::with_current(|arena| self.to_raw_pointer_mut_with_arena(arena))
    }

    unsafe fn const_to_raw_pointer_mut_with_arena(self, arena: &Arena) -> *mut u64 {
        let tagged = TaggedPtr::from_raw(self.raw);
        if self.is_indirect() {
            tagged.resolve_mut(INDIRECT_MASK, arena) as *mut u64
        } else {
            tagged.resolve_mut(CELL_MASK, arena) as *mut u64
        }
    }

    unsafe fn const_to_raw_pointer_mut(self) -> *mut u64 {
        Arena::with_current(|arena| self.const_to_raw_pointer_mut_with_arena(arena))
    }

    pub unsafe fn forwarding_pointer_with_arena(&self, arena: &Arena) -> Option<Allocated> {
        match self.as_either() {
            Left(indirect) => indirect
                .forwarding_pointer_with_arena(arena)
                .map(|i| i.as_allocated()),
            Right(cell) => cell
                .forwarding_pointer_with_arena(arena)
                .map(|c| c.as_allocated()),
        }
    }

    pub unsafe fn forwarding_pointer(&self) -> Option<Allocated> {
        Arena::with_current(|arena| self.forwarding_pointer_with_arena(arena))
    }

    pub unsafe fn get_metadata_with_arena(&self, arena: &Arena) -> u64 {
        *(self.to_raw_pointer_with_arena(arena))
    }

    pub unsafe fn get_metadata(&self) -> u64 {
        Arena::with_current(|arena| self.get_metadata_with_arena(arena))
    }

    pub unsafe fn set_metadata_with_arena(&mut self, metadata: u64, arena: &Arena) {
        *(self.const_to_raw_pointer_mut_with_arena(arena)) = metadata;
    }

    pub unsafe fn set_metadata(&mut self, metadata: u64) {
        Arena::with_current(|arena| self.set_metadata_with_arena(metadata, arena))
    }

    pub fn as_either(&self) -> Either<IndirectAtom, Cell> {
        if self.is_indirect() {
            unsafe { Left(self.indirect) }
        } else {
            unsafe { Right(self.cell) }
        }
    }

    pub fn as_ref_either(&self) -> Either<&IndirectAtom, &Cell> {
        if self.is_indirect() {
            unsafe { Left(&self.indirect) }
        } else {
            unsafe { Right(&self.cell) }
        }
    }

    pub fn cell(&self) -> Option<Cell> {
        if self.is_cell() {
            unsafe { Some(self.cell) }
        } else {
            None
        }
    }

    pub fn as_noun(&self) -> Noun {
        Noun { allocated: *self }
    }

    pub fn get_cached_mug(self: Allocated) -> Option<u32> {
        unsafe {
            let bottom_metadata = self.get_metadata() as u32 & 0x7FFFFFFF; // magic number: LS 31 bits
            if bottom_metadata > 0 {
                Some(bottom_metadata)
            } else {
                None
            }
        }
    }
}

impl fmt::Debug for Allocated {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_noun().fmt(f)
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
#[repr(packed(8))]
pub union Noun {
    pub(crate) raw: u64,
    direct: DirectAtom,
    indirect: IndirectAtom,
    atom: Atom,
    cell: Cell,
    allocated: Allocated,
}

impl Noun {
    pub fn is_none(self) -> bool {
        unsafe { self.raw == u64::MAX }
    }

    pub fn is_direct(&self) -> bool {
        unsafe { is_direct_atom(self.raw) }
    }

    pub fn is_indirect(&self) -> bool {
        unsafe { is_indirect_atom(self.raw) }
    }

    pub fn is_atom(&self) -> bool {
        self.is_direct() || self.is_indirect()
    }

    pub fn is_allocated(&self) -> bool {
        self.is_indirect() || self.is_cell()
    }

    #[inline]
    pub fn is_stack_allocated(&self) -> bool {
        self.is_allocated() && unsafe { self.as_raw() & LOCATION_BIT == 0 }
    }

    pub fn is_cell(&self) -> bool {
        unsafe { is_cell(self.raw) }
    }

    pub fn as_direct(&self) -> Result<DirectAtom> {
        if self.is_direct() {
            unsafe { Ok(self.direct) }
        } else {
            Err(Error::NotDirectAtom)
        }
    }

    pub fn as_indirect(&self) -> Result<IndirectAtom> {
        if self.is_indirect() {
            unsafe { Ok(self.indirect) }
        } else {
            Err(Error::NotIndirectAtom)
        }
    }

    pub fn as_cell(&self) -> Result<Cell> {
        if self.is_cell() {
            unsafe { Ok(self.cell) }
        } else {
            Err(Error::NotCell)
        }
    }

    pub fn as_atom(&self) -> Result<Atom> {
        if self.is_atom() {
            unsafe { Ok(self.atom) }
        } else {
            Err(Error::NotAtom)
        }
    }

    pub fn as_allocated(&self) -> Result<Allocated> {
        if self.is_allocated() {
            unsafe { Ok(self.allocated) }
        } else {
            Err(Error::NotAllocated)
        }
    }

    pub fn as_either_atom_cell(&self) -> Either<Atom, Cell> {
        if self.is_cell() {
            unsafe { Right(self.cell) }
        } else {
            unsafe { Left(self.atom) }
        }
    }

    pub fn as_either_direct_allocated(self) -> Either<DirectAtom, Allocated> {
        if self.is_direct() {
            unsafe { Left(self.direct) }
        } else {
            unsafe { Right(self.allocated) }
        }
    }

    pub fn as_ref_either_direct_allocated(&self) -> Either<&DirectAtom, &Allocated> {
        if self.is_direct() {
            unsafe { Left(&self.direct) }
        } else {
            unsafe { Right(&self.allocated) }
        }
    }

    pub fn as_ref_mut_either_direct_allocated(
        &mut self,
    ) -> Either<&mut DirectAtom, &mut Allocated> {
        if self.is_direct() {
            unsafe { Left(&mut self.direct) }
        } else {
            unsafe { Right(&mut self.allocated) }
        }
    }

    pub fn atom(&self) -> Option<Atom> {
        if self.is_atom() {
            unsafe { Some(self.atom) }
        } else {
            None
        }
    }

    pub fn cell(&self) -> Option<Cell> {
        if self.is_cell() {
            unsafe { Some(self.cell) }
        } else {
            None
        }
    }

    pub fn direct(&self) -> Option<DirectAtom> {
        if self.is_direct() {
            unsafe { Some(self.direct) }
        } else {
            None
        }
    }

    pub fn indirect(&self) -> Option<IndirectAtom> {
        if self.is_indirect() {
            unsafe { Some(self.indirect) }
        } else {
            None
        }
    }

    pub fn allocated(&self) -> Option<Allocated> {
        if self.is_allocated() {
            unsafe { Some(self.allocated) }
        } else {
            None
        }
    }

    /** Are these the same noun */
    pub unsafe fn raw_equals(&self, other: &Noun) -> bool {
        self.raw == other.raw
    }

    pub unsafe fn as_raw(&self) -> u64 {
        self.raw
    }

    pub unsafe fn from_raw(raw: u64) -> Noun {
        Noun { raw }
    }

    /** Produce the total size of a noun, in words
     *
     * This counts the total size, see mass_frame() to count the size in the current frame.
     */
    pub fn mass(self) -> usize {
        unsafe {
            let res = self.mass_wind(&|_| true);
            self.mass_unwind(&|_| true);
            res
        }
    }

    /** Produce the size of a noun in the current frame, in words */
    pub fn mass_frame(self, stack: &NockStack) -> usize {
        unsafe {
            let res = self.mass_wind(&|p| stack.is_in_frame(p));
            self.mass_unwind(&|p| stack.is_in_frame(p));
            res
        }
    }

    /** Produce the total size of a noun, in words
     *
     * `inside` determines whether a pointer should be counted.  If it returns false, we also do
     * not recurse into that noun if it is a cell.  See mass_frame() for an example.
     *
     * This "winds up" the mass calculation, which includes setting the 32nd bit of the metadata to
     * mark nouns that have already been counted.
     *
     * This is unsafe because you *must* call mass_unwind() with the same `inside` function to
     * unmark the noun.  This is exposed so that you can count several the "mass difference" of a
     * series of nouns.  If you call this twice consecutively, the first result will be the mass of
     * the first noun, and the second will be the mass of the second noun minus the overlap with
     * the first noun.
     */
    pub unsafe fn mass_wind(self, inside: &impl Fn(*const u64) -> bool) -> usize {
        if let Ok(mut allocated) = self.as_allocated() {
            if inside(allocated.to_raw_pointer()) {
                if allocated.get_metadata() & (1 << 32) == 0 {
                    allocated.set_metadata(allocated.get_metadata() | (1 << 32));
                    match allocated.as_either() {
                        Left(indirect) => indirect.size() + 2,
                        Right(cell) => {
                            word_size_of::<CellMemory>()
                                + cell.head().mass_wind(inside)
                                + cell.tail().mass_wind(inside)
                        }
                    }
                } else {
                    0
                }
            } else {
                0
            }
        } else {
            0
        }
    }

    /** See mass_wind() */
    pub unsafe fn mass_unwind(self, inside: &impl Fn(*const u64) -> bool) {
        if let Ok(mut allocated) = self.as_allocated() {
            if inside(allocated.to_raw_pointer()) {
                allocated.set_metadata(allocated.get_metadata() & !(1 << 32));
                if let Right(cell) = allocated.as_either() {
                    cell.head().mass_unwind(inside);
                    cell.tail().mass_unwind(inside);
                }
            }
        }
    }
}

impl fmt::Debug for Noun {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            if self.is_direct() {
                write!(f, "{:?}", self.direct)
            } else if self.is_indirect() {
                write!(f, "{:?}", self.indirect)
            } else if self.is_cell() {
                write!(f, "{:?}", self.cell)
            } else if self.allocated.forwarding_pointer().is_some() {
                write!(
                    f,
                    "Noun::Forwarding({:?})",
                    self.allocated
                        .forwarding_pointer()
                        .unwrap_or_else(|| panic!(
                            "Panicked at {}:{} (git sha: {:?})",
                            file!(),
                            line!(),
                            option_env!("GIT_SHA")
                        ))
                )
            } else {
                write!(f, "Noun::Unknown({:x})", self.raw)
            }
        }
    }
}

impl Slots for Noun {}
impl private::RawSlots for Noun {
    #[inline(always)]
    fn raw_slot_direct(&self, axis: u64) -> Result<Noun> {
        match self.as_either_atom_cell() {
            Right(cell) => cell.raw_slot_direct(axis),
            Left(_atom) => {
                if axis == 1 {
                    Ok(*self)
                } else {
                    // Axis tried to descend through atom
                    Err(Error::NotCell)
                }
            }
        }
    }

    #[inline(always)]
    fn raw_slot_indirect(&self, axis: &[u64]) -> Result<Noun> {
        match self.as_either_atom_cell() {
            Right(cell) => cell.raw_slot_indirect(axis),
            Left(_atom) => {
                // Check if axis is 1 (all words are 0 except word[0] & 1 == 1)
                if axis.len() == 1 && axis[0] == 1 {
                    Ok(*self)
                } else if axis.is_empty() || (axis.len() == 1 && axis[0] == 0) {
                    Err(Error::NotRepresentable)
                } else {
                    // Axis tried to descend through atom
                    Err(Error::NotCell)
                }
            }
        }
    }
}

/**
 * An allocation object (probably a mem::NockStack) which can allocate a memory buffer sized to
 * a certain number of nouns
 */
pub trait NounAllocator: Sized + Stack {
    /** Allocate memory for some multiple of the size of a noun
     *
     * This should allocate *two more* `u64`s than `words` to make space for the size and metadata
     */
    unsafe fn alloc_indirect(&mut self, words: usize) -> *mut u64;

    /** Allocate memory for a cell */
    unsafe fn alloc_cell(&mut self) -> *mut CellMemory;

    /** Allocate space for a struct in a stack frame */
    unsafe fn alloc_struct<T>(&mut self, count: usize) -> *mut T;

    /** Check if two allocated nouns are equal **/
    unsafe fn equals(&mut self, a: *mut Noun, b: *mut Noun) -> bool;
}

/**
 * Implementing types allow component Nouns to be retreived by numeric axis
 */
pub trait Slots: private::RawSlots {
    /**
     * Retrieve component Noun at given axis, or fail with descriptive error
     */
    fn slot(&self, axis: u64) -> Result<Noun> {
        self.raw_slot_direct(axis)
    }

    /**
     * Retrieve component Noun at axis given as Atom, or fail with descriptive error
     */
    fn slot_atom(&self, atom: Atom) -> Result<Noun> {
        match atom.as_either() {
            Left(direct) => self.raw_slot_direct(direct.data()),
            Right(indirect) => self.raw_slot_indirect(indirect.as_slice()),
        }
    }
}

/**
 * Implementation methods that should not be made available to derived crates
 */
mod private {
    use crate::noun::{Noun, Result};

    /**
     * Implementation of the Slots trait
     */
    pub trait RawSlots {
        /**
         * Actual logic of retreiving Noun object at some axis (direct)
         */
        fn raw_slot_direct(&self, axis: u64) -> Result<Noun>;

        /**
         * Actual logic of retreiving Noun object at some axis (indirect)
         */
        fn raw_slot_indirect(&self, axis: &[u64]) -> Result<Noun>;
    }
}

#[cfg(test)]
mod tests {
    use crate::jets::util::test::init_context;
    use crate::noun::{Cell, Slots, D};

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_slot_direct_simple() {
        let mut context = init_context();
        let cell = Cell::new(&mut context.stack, D(1), D(2));

        // axis 1 returns the whole cell
        assert_eq!(
            unsafe { cell.slot(1).unwrap().raw_equals(&cell.as_noun()) },
            true
        );

        // axis 2 returns head
        assert_eq!(unsafe { cell.slot(2).unwrap().raw_equals(&D(1)) }, true);

        // axis 3 returns tail
        assert_eq!(unsafe { cell.slot(3).unwrap().raw_equals(&D(2)) }, true);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_slot_direct_nested() {
        let mut context = init_context();
        let inner = Cell::new(&mut context.stack, D(3), D(4));
        // cell = [1 [3 4]]
        let cell = Cell::new(&mut context.stack, D(1), inner.as_noun());

        // axis 6 = 110 binary = tail then head = head of tail = 3
        assert_eq!(unsafe { cell.slot(6).unwrap().raw_equals(&D(3)) }, true);

        // axis 7 = 111 binary = tail then tail = tail of tail = 4
        assert_eq!(unsafe { cell.slot(7).unwrap().raw_equals(&D(4)) }, true);

        // axis 4 = 100 binary = head then stop = should fail (head is atom)
        assert!(cell.slot(4).is_err());

        // cell2 = [[3 4] 2]
        let cell2 = Cell::new(&mut context.stack, inner.as_noun(), D(2));
        // axis 5 = 101 binary = head then tail = tail of head = 4
        assert_eq!(unsafe { cell2.slot(5).unwrap().raw_equals(&D(4)) }, true);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_slot_zero_axis() {
        let mut context = init_context();
        let cell = Cell::new(&mut context.stack, D(1), D(2));

        // axis 0 should fail
        assert!(cell.slot(0).is_err());
    }
}

#[cfg(test)]
mod test {
    use ibig::ubig;

    use crate::jets::util::test::init_context;
    use crate::noun::Atom;

    #[test]
    //  APOLOGIA: ibig/ubig ManuallyDrops Vec, we are aware, we plan on purging it
    #[cfg_attr(miri, ignore)]
    fn test_to_ne_bytes_direct() {
        let mut context = init_context();
        let big = ubig!(0x1234567890abcdefa0);
        let atom = Atom::from_ubig(&mut context.stack, &big);
        let bytes = atom.to_ne_bytes();
        #[cfg(target_endian = "little")]
        {
            assert_eq!(
                bytes,
                vec![
                    0xa0, 0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00
                ]
            );
        }
        #[cfg(target_endian = "big")]
        {
            assert_eq!(
                bytes,
                vec![
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab,
                    0xcd, 0xef, 0xa0
                ]
            );
        }
    }

    #[test]
    //  APOLOGIA: ibig/ubig ManuallyDrops Vec, we are aware, we plan on purging it
    #[cfg_attr(miri, ignore)]
    fn test_to_ne_bytes_indirect() {
        let mut context = init_context();
        let atom = Atom::new(&mut context.stack, 0x1234);
        let bytes = atom.to_ne_bytes();
        #[cfg(target_endian = "little")]
        {
            assert_eq!(bytes, vec![0x34, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        }
        #[cfg(target_endian = "big")]
        {
            assert_eq!(bytes, vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0x34]);
        }
    }

    #[test]
    //  APOLOGIA: ibig/ubig ManuallyDrops Vec, we are aware, we plan on purging it
    #[cfg_attr(miri, ignore)]
    fn test_to_x_bytes_direct() {
        let mut context = init_context();
        let atom = Atom::new(&mut context.stack, 0x1234);
        let bytes_le = atom.to_le_bytes();
        assert_eq!(
            bytes_le,
            vec![0x34, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );

        let bytes_be = atom.to_be_bytes();
        assert_eq!(
            bytes_be,
            vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0x34]
        );
    }

    #[test]
    //  APOLOGIA: ibig/ubig ManuallyDrops Vec, we are aware, we plan on purging it
    #[cfg_attr(miri, ignore)]
    fn test_to_le_bytes_indirect() {
        let mut context = init_context();
        let big = ubig!(0x1234567890abcd);
        let atom = Atom::from_ubig(&mut context.stack, &big);
        let bytes = atom.to_le_bytes();
        assert_eq!(bytes, vec![0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0x00]);
        //
        let big = ubig!(0x1234567890abcdefa0);
        let atom = Atom::from_ubig(&mut context.stack, &big);
        let bytes = atom.to_le_bytes();
        assert_eq!(
            bytes,
            vec![
                0xa0, 0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00
            ],
        );
    }

    #[test]
    //  APOLOGIA: ibig/ubig ManuallyDrops Vec, we are aware, we plan on purging it
    #[cfg_attr(miri, ignore)]
    fn test_to_be_bytes_indirect() {
        let mut context = init_context();
        let big = ubig!(0x34567890abcdef);
        let atom = Atom::from_ubig(&mut context.stack, &big);
        let bytes = atom.to_be_bytes();
        assert_eq!(bytes, vec![0x00, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd, 0xef]);
        //
        let big = ubig!(0x1234567890abcdefa0);
        let atom = Atom::from_ubig(&mut context.stack, &big);
        let bytes = atom.to_be_bytes();
        assert_eq!(
            bytes,
            vec![
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0x34, 0x56, 0x78, 0x90, 0xab, 0xcd,
                0xef, 0xa0
            ]
        );
    }
}
```

## crates/nockapp/src/kernel/form.rs
```
#![allow(dead_code)]
use std::any::Any;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use blake3::{Hash, Hasher};
use byteorder::{LittleEndian, WriteBytesExt};
use nockvm::hamt::Hamt;
use nockvm::interpreter::{self, interpret, Error, Mote, NockCancelToken};
use nockvm::jets::cold::{Cold, Nounable};
use nockvm::jets::hot::{HotEntry, URBIT_HOT_STATE};
use nockvm::jets::nock::util::mook;
use nockvm::mem::{NockStack, Retag};
use nockvm::mug::met3_usize;
use nockvm::noun::{Atom, Cell, DirectAtom, IndirectAtom, Noun, Slots, D, T};
use nockvm::trace::{path_to_cord, write_serf_trace_safe};
use nockvm_macros::tas;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Duration;
use tracing::{debug, warn};

use crate::kernel::boot::TraceOpts;
use crate::metrics::NockAppMetrics;
use crate::nockapp::wire::{wire_to_noun, WireRepr};
use crate::noun::slab::NounSlab;
use crate::noun::slam;
use crate::save::SaveableCheckpoint;
use crate::utils::{
    create_context, current_da, NOCK_STACK_SIZE, NOCK_STACK_SIZE_HUGE, NOCK_STACK_SIZE_LARGE,
    NOCK_STACK_SIZE_MEDIUM, NOCK_STACK_SIZE_SMALL, NOCK_STACK_SIZE_TINY,
};
use crate::{AtomExt, CrownError, IndirectAtomExt, NounExt, Result, ToBytesExt};

pub(crate) const STATE_AXIS: u64 = 6;
const LOAD_AXIS: u64 = 4;
const PEEK_AXIS: u64 = 22;
const POKE_AXIS: u64 = 23;

const SERF_FINISHED_INTERVAL: Duration = Duration::from_millis(100);
const SERF_THREAD_STACK_SIZE: usize = 256 * 1024 * 1024; // 8MB

pub struct LoadState {
    pub ker_hash: Hash,
    pub event_num: u64,
    pub kernel_state: NounSlab,
}

// Actions to request of the serf thread
pub enum SerfAction<C> {
    // Make a CheckPoint
    Checkpoint {
        result: oneshot::Sender<C>,
    },
    Import {
        state: LoadState,
        result: oneshot::Sender<Result<()>>,
    },
    Export {
        result: oneshot::Sender<Result<LoadState>>,
    },
    // Get the state noun of the kernel as a slab
    GetKernelStateSlab {
        result: oneshot::Sender<Result<NounSlab>>,
    },
    // Get the cold state as a NounSlab
    GetColdStateSlab {
        result: oneshot::Sender<NounSlab>,
    },
    // Run a peek
    Peek {
        ovo: NounSlab,
        result: oneshot::Sender<Result<NounSlab>>,
    },
    // Run a poke
    //
    // TODO: send back the event number after each poke
    Poke {
        wire: WireRepr,
        cause: NounSlab,
        result: oneshot::Sender<Result<NounSlab>>,
        result_ack: oneshot::Receiver<()>,
    },
    // Provide metrics
    ProvideMetrics {
        metrics: Arc<NockAppMetrics>,
        result: oneshot::Sender<()>,
    },
    // Stop the loop
    Stop,
}

pub struct SerfThread<C> {
    handle: Option<std::thread::JoinHandle<()>>,
    action_sender: mpsc::Sender<SerfAction<C>>,
    pub cancel_token: NockCancelToken,
    inhibit: Arc<AtomicBool>,
    pub event_number: Arc<AtomicU64>,
}

impl<C: SerfCheckpoint + Send + 'static> SerfThread<C> {
    pub async fn new(
        kernel_bytes: Vec<u8>,
        checkpoint: Option<C>,
        constant_hot_state: Vec<HotEntry>,
        nock_stack_size: usize,
        test_jets: Vec<NounSlab>,
        trace: TraceOpts,
    ) -> Result<Self> {
        let (action_sender, action_receiver) = mpsc::channel(1);
        let (event_number_sender, event_number_receiver) = oneshot::channel();
        let (cancel_token_sender, cancel_token_receiver) = oneshot::channel();
        let inhibit = Arc::new(AtomicBool::new(false));
        let inhibit_clone = inhibit.clone();
        let handle = std::thread::Builder::new()
            .name("serf".to_string())
            .stack_size(SERF_THREAD_STACK_SIZE)
            .spawn(move || {
                let stack = NockStack::new(nock_stack_size, 0);
                let serf = Serf::new(
                    stack, checkpoint, &kernel_bytes, &constant_hot_state, test_jets, trace,
                );
                event_number_sender
                    .send(serf.event_num.clone())
                    .expect("Could not send event number out of serf thread");
                cancel_token_sender
                    .send(serf.context.cancel_token())
                    .expect("Could not send cancel token out of serf thread");
                serf_loop(serf, action_receiver, inhibit_clone);
            })?;

        let event_number = event_number_receiver.await?;
        let cancel_token = cancel_token_receiver.await?;
        Ok(SerfThread {
            inhibit,
            handle: Some(handle),
            action_sender,
            event_number,
            cancel_token,
        })
    }
}

impl<C> SerfThread<C> {
    pub(crate) fn provide_metrics(
        &mut self,
        metrics: Arc<NockAppMetrics>,
    ) -> impl Future<Output = Result<()>> {
        let action_sender = self.action_sender.clone();
        let (result, result_recv) = oneshot::channel();
        async move {
            action_sender
                .send(SerfAction::ProvideMetrics { metrics, result })
                .await?;
            Ok(result_recv.await?)
        }
    }

    pub(crate) fn stop(&mut self) -> impl Future<Output = Result<()>> {
        let action_sender = self.action_sender.clone();
        let cancel_token = self.cancel_token.clone();
        let join_handle = self.handle.take().expect("Serf join handle already taken.");
        let tokio_join_handle = tokio::task::spawn_blocking(move || join_handle.join());
        self.inhibit.store(true, Ordering::SeqCst);
        async move {
            cancel_token.cancel();
            action_sender
                .send(SerfAction::Stop)
                .await
                .expect("Failed to send stop action");
            match tokio_join_handle.await {
                Ok(Ok(())) => Ok(()),
                Ok(Err(e)) => Err(CrownError::Unknown(format!("Serf thread panicked: {e:?}"))),
                Err(e) => Err(CrownError::JoinError(e)),
            }
        }
    }

    pub(crate) fn join(&mut self) -> Result<(), Box<dyn Any + Send + 'static>> {
        self.handle
            .take()
            .expect("Serf thread already joined")
            .join()
    }

    pub(crate) async fn get_kernel_state_slab(&self) -> Result<NounSlab> {
        let (result, result_fut) = oneshot::channel();
        self.action_sender
            .send(SerfAction::GetKernelStateSlab { result })
            .await?;
        result_fut.await?
    }

    pub(crate) async fn get_cold_state_slab(&self) -> Result<NounSlab> {
        let (result, result_fut) = oneshot::channel();
        self.action_sender
            .send(SerfAction::GetColdStateSlab { result })
            .await?;
        Ok(result_fut.await?)
    }

    pub(crate) fn peek(&self, ovo: NounSlab) -> impl Future<Output = Result<NounSlab>> {
        let (result, result_fut) = oneshot::channel();
        let action_sender = self.action_sender.clone();
        async move {
            action_sender.send(SerfAction::Peek { ovo, result }).await?;
            result_fut.await?
        }
    }

    // We are very carefully ensuring that the future does not contain the &self reference, to allow spawning a task without lifetime issues
    pub fn poke(&self, wire: WireRepr, cause: NounSlab) -> impl Future<Output = Result<NounSlab>> {
        let (result, result_fut) = oneshot::channel();
        let (result_ack_sender, result_ack) = oneshot::channel();
        let action_sender = self.action_sender.clone();
        async move {
            action_sender
                .send(SerfAction::Poke {
                    wire,
                    cause,
                    result,
                    result_ack,
                })
                .await?;
            let res = result_fut.await?;
            let _ = result_ack_sender.send(());
            res
        }
    }

    pub fn poke_timeout(
        &self,
        wire: WireRepr,
        cause: NounSlab,
        timeout: Duration,
    ) -> impl Future<Output = Result<NounSlab>> {
        let (result, result_fut) = oneshot::channel();
        let (result_ack_sender, result_ack) = oneshot::channel();
        let action_sender = self.action_sender.clone();
        let cancel = self.cancel_token.clone();
        let timer = tokio::time::sleep(timeout);
        let cancel_task = tokio::spawn(async move {
            timer.await;
            cancel.cancel();
        });
        async move {
            action_sender
                .send(SerfAction::Poke {
                    wire,
                    cause,
                    result,
                    result_ack,
                })
                .await?;
            let res = result_fut.await?;
            cancel_task.abort();
            let _ = cancel_task.await;
            let _ = result_ack_sender.send(());
            res
        }
    }

    pub(crate) fn poke_sync(&self, wire: WireRepr, cause: NounSlab) -> Result<NounSlab> {
        let (result, result_fut) = oneshot::channel();
        let (result_ack_sender, result_ack) = oneshot::channel();
        self.action_sender.blocking_send(SerfAction::Poke {
            wire,
            cause,
            result,
            result_ack,
        })?;
        let res = result_fut.blocking_recv()?;
        let _ = result_ack_sender.send(());
        res
    }

    pub(crate) fn peek_sync(&self, ovo: NounSlab) -> Result<NounSlab> {
        let (result, result_fut) = oneshot::channel();
        self.action_sender
            .blocking_send(SerfAction::Peek { ovo, result })?;
        result_fut.blocking_recv()?
    }

    pub(crate) fn checkpoint(&self) -> impl Future<Output = Result<C>> {
        let (result, result_fut) = oneshot::channel();
        let action_sender = self.action_sender.clone();
        async move {
            action_sender
                .send(SerfAction::Checkpoint { result })
                .await?;
            Ok(result_fut.await?)
        }
    }

    pub fn import(&self, state: LoadState) -> impl Future<Output = Result<()>> {
        let (result, result_fut) = oneshot::channel();
        let action_sender = self.action_sender.clone();
        async move {
            action_sender
                .send(SerfAction::Import { state, result })
                .await?;
            result_fut.await?
        }
    }

    pub fn export(&self) -> impl Future<Output = Result<LoadState>> {
        let (result, result_fut) = oneshot::channel();
        let action_sender = self.action_sender.clone();
        async move {
            action_sender.send(SerfAction::Export { result }).await?;
            result_fut.await?
        }
    }
}

fn serf_loop<C: SerfCheckpoint>(
    mut serf: Serf,
    mut action_receiver: mpsc::Receiver<SerfAction<C>>,
    inhibit: Arc<AtomicBool>,
) {
    loop {
        serf.context.install_arena();
        let start = std::time::Instant::now();
        let Some(action) = action_receiver.blocking_recv() else {
            break;
        };
        let recv_elapsed = start.elapsed();
        if let Some(nockapp_metrics) = &serf.metrics {
            nockapp_metrics
                .serf_loop_blocking_recv
                .add_timing(&recv_elapsed);
        };
        let action_start = std::time::Instant::now();
        match action {
            SerfAction::Stop => {
                break;
            }
            SerfAction::Export { result } => {
                let kernel_state_noun = serf.arvo.slot(STATE_AXIS);
                let kernel_state = kernel_state_noun.map_or_else(
                    |err| Err(CrownError::from(err)),
                    |noun| {
                        let mut slab = NounSlab::new();
                        slab.copy_into(noun);
                        Ok(slab)
                    },
                );
                let load_state = kernel_state.map(|kernel_state| LoadState {
                    kernel_state,
                    ker_hash: serf.ker_hash,
                    event_num: serf.event_num.load(Ordering::SeqCst),
                });
                let _ = result.send(load_state).inspect_err(|_err| {
                    debug!("Failed to send to dropped channel");
                });
            }
            SerfAction::Import { state, result } => {
                let state_noun = state.kernel_state.copy_to_stack(serf.stack());
                let arvo = serf.load(state_noun);
                match arvo {
                    Err(e) => {
                        let _ = result.send(Err(e)).map_err(|err| {
                            debug!("Tried to send to dropped channel: {:?}", err);
                        });
                    }
                    Ok(arvo) => {
                        if serf.ker_hash != state.ker_hash {
                            debug!(
                                "Importing state from kernel hash {} into kernel hash {}",
                                state.ker_hash, serf.ker_hash
                            );
                        }
                        unsafe {
                            serf.event_update(state.event_num, arvo);
                            serf.preserve_event_update_leftovers();
                        }
                        let _ = result.send(Ok(())).map_err(|err| {
                            debug!("Tried to send to dropped channel: {:?}", err);
                        });
                    }
                }
            }
            SerfAction::GetKernelStateSlab { result } => {
                let kernel_state_noun = serf.arvo.slot(STATE_AXIS);
                let kernel_state_slab = kernel_state_noun.map_or_else(
                    |err| Err(CrownError::from(err)),
                    |noun| {
                        let mut slab = NounSlab::new();
                        slab.copy_into(noun);
                        Ok(slab)
                    },
                );
                let _ = result.send(kernel_state_slab).inspect_err(|_e| {
                    debug!("Tried to send to dropped result channel");
                });
                let action_elapsed = action_start.elapsed();
                if let Some(nockapp_metrics) = &serf.metrics {
                    nockapp_metrics
                        .serf_loop_get_kernel_state_slab
                        .add_timing(&action_elapsed);
                };
            }
            SerfAction::GetColdStateSlab { result } => {
                let cold_state_noun = serf.context.cold.into_noun(serf.stack());
                let cold_state_slab = {
                    let mut slab = NounSlab::new();
                    slab.copy_into(cold_state_noun);
                    slab
                };
                let _ = result.send(cold_state_slab).inspect_err(|_e| {
                    debug!("Could not send cold state to dropped channel.");
                });
                let action_elapsed = action_start.elapsed();
                if let Some(nockapp_metrics) = &serf.metrics {
                    nockapp_metrics
                        .serf_loop_get_cold_state_slab
                        .add_timing(&action_elapsed);
                };
            }
            SerfAction::Checkpoint { result } => {
                let metrics_checkpoint = serf.metrics.clone();
                let checkpoint = create_checkpoint(&mut serf, &metrics_checkpoint);
                //result.send(checkpoint).expect("Could not send checkpoint");
                if result.send(checkpoint).is_err() {
                    debug!(
                        "Checkpoint receiver dropped before receiving result - likely timed out"
                    );
                };
                let action_elapsed = action_start.elapsed();
                if let Some(nockapp_metrics) = &serf.metrics {
                    nockapp_metrics
                        .serf_loop_checkpoint
                        .add_timing(&action_elapsed);
                };
            }
            SerfAction::Peek { ovo, result } => {
                if inhibit.load(Ordering::SeqCst) {
                    let _ = result
                        .send(Err(CrownError::Unknown("Serf stopping".to_string())))
                        .inspect_err(|_e| {
                            debug!("Tried to send inhibited peek state to dropped channel");
                        });
                } else {
                    let ovo_noun = ovo.copy_to_stack(serf.stack());
                    let noun_res = serf.peek(ovo_noun);
                    let noun_slab_res = noun_res.map(|noun| {
                        let mut slab = NounSlab::new();
                        slab.copy_into(noun);
                        slab
                    });
                    let _ = result.send(noun_slab_res).inspect_err(|_e| {
                        debug!("Tried to send peek state to dropped channel");
                    });
                };
                let action_elapsed = action_start.elapsed();
                if let Some(nockapp_metrics) = &serf.metrics {
                    nockapp_metrics.serf_loop_peek.add_timing(&action_elapsed);
                };
            }
            SerfAction::Poke {
                wire,
                cause,
                result,
                result_ack,
            } => {
                if inhibit.load(Ordering::SeqCst) {
                    let _ = result
                        .send(Err(CrownError::Unknown("Serf stopping".to_string())))
                        .inspect_err(|_e| {
                            debug!("Failed to send inihibited poke result from serf thread");
                        });
                } else {
                    let cause_noun = cause.copy_to_stack(serf.stack());
                    let noun_res = serf.poke(wire, cause_noun);
                    let noun_slab_res = noun_res.map(|noun| {
                        let mut slab = NounSlab::new();
                        slab.copy_into(noun);
                        slab
                    });
                    let _ = result.send(noun_slab_res).inspect_err(|_e| {
                        debug!("Failed to send poke result from serf thread");
                    });
                };
                let _ = result_ack.blocking_recv().inspect_err(|_e| {
                    debug!("Failed to receive result ack in serf thread");
                });
                let action_elapsed = action_start.elapsed();
                if let Some(nockapp_metrics) = &serf.metrics {
                    nockapp_metrics.serf_loop_poke.add_timing(&action_elapsed);
                };
            }
            SerfAction::ProvideMetrics { metrics, result } => {
                serf.metrics = Some(metrics);
                let _ = result.send(()).inspect_err(|_e| {
                    debug!("Failed to send metric-provision result from serf thread");
                });
                let action_elapsed = action_start.elapsed();
                if let Some(nockapp_metrics) = &serf.metrics {
                    nockapp_metrics
                        .serf_loop_provide_metrics
                        .add_timing(&action_elapsed);
                };
            }
        };
        let elapsed = start.elapsed();
        if let Some(nockapp_metrics) = &serf.metrics {
            nockapp_metrics.serf_loop_all.add_timing(&elapsed);
        };
    }
}

fn create_checkpoint<C: SerfCheckpoint>(
    serf: &mut Serf,
    metrics: &Option<Arc<NockAppMetrics>>,
) -> C {
    let ker_hash = serf.ker_hash;
    let event_num = serf.event_num.load(Ordering::SeqCst);
    let ker_state = serf.arvo.slot(STATE_AXIS).unwrap_or_else(|err| {
        panic!(
            "Panicked with {err:?} at {}:{} (git sha: {:?})",
            file!(),
            line!(),
            option_env!("GIT_SHA")
        )
    });
    let cold_state = serf.context.cold;

    C::new(
        serf.stack(),
        ker_hash,
        event_num,
        ker_state,
        cold_state,
        metrics,
    )
}

/// Represents a Sword kernel, containing a Serf and snapshot location.
pub struct Kernel<C> {
    /// The Serf managing the interface to the Sword.
    pub(crate) serf: SerfThread<C>,
}

impl<C: SerfCheckpoint + 'static> Kernel<C> {
    /// Loads a kernel with a custom hot state.
    ///
    /// # Arguments
    ///
    /// * `snap_dir` - Directory for storing snapshots.
    /// * `kernel` - Byte slice containing the kernel as a jammed noun.
    /// * `hot_state` - Custom hot state entries.
    /// * `trace` - Whether to enable tracing.
    ///
    /// # Returns
    ///
    /// A new `Kernel` instance.
    pub async fn load_with_hot_state(
        kernel: &[u8],
        checkpoint: Option<C>,
        hot_state: &[HotEntry],
        test_jets: Vec<NounSlab>,
        trace: TraceOpts,
    ) -> Result<Self> {
        let kernel_vec = Vec::from(kernel);
        let hot_state_vec = Vec::from(hot_state);
        let serf = SerfThread::new(
            kernel_vec, checkpoint, hot_state_vec, NOCK_STACK_SIZE, test_jets, trace,
        )
        .await?;
        Ok(Self { serf })
    }

    pub async fn load_with_hot_state_tiny(
        kernel: &[u8],
        checkpoint: Option<C>,
        hot_state: &[HotEntry],
        test_jets: Vec<NounSlab>,
        trace: TraceOpts,
    ) -> Result<Self> {
        let kernel_vec = Vec::from(kernel);
        let hot_state_vec = Vec::from(hot_state);
        let serf = SerfThread::new(
            kernel_vec, checkpoint, hot_state_vec, NOCK_STACK_SIZE_TINY, test_jets, trace,
        )
        .await?;
        Ok(Self { serf })
    }

    pub async fn load_with_hot_state_small(
        kernel: &[u8],
        checkpoint: Option<C>,
        hot_state: &[HotEntry],
        test_jets: Vec<NounSlab>,
        trace: TraceOpts,
    ) -> Result<Self> {
        let kernel_vec = Vec::from(kernel);
        let hot_state_vec = Vec::from(hot_state);
        let serf = SerfThread::new(
            kernel_vec, checkpoint, hot_state_vec, NOCK_STACK_SIZE_SMALL, test_jets, trace,
        )
        .await?;
        Ok(Self { serf })
    }

    pub async fn load_with_hot_state_medium(
        kernel: &[u8],
        checkpoint: Option<C>,
        hot_state: &[HotEntry],
        test_jets: Vec<NounSlab>,
        trace: TraceOpts,
    ) -> Result<Self> {
        let kernel_vec = Vec::from(kernel);
        let hot_state_vec = Vec::from(hot_state);
        let serf = SerfThread::new(
            kernel_vec, checkpoint, hot_state_vec, NOCK_STACK_SIZE_MEDIUM, test_jets, trace,
        )
        .await?;
        Ok(Self { serf })
    }

    pub async fn load_with_hot_state_large(
        kernel: &[u8],
        checkpoint: Option<C>,
        hot_state: &[HotEntry],
        test_jets: Vec<NounSlab>,
        trace: TraceOpts,
    ) -> Result<Self> {
        let kernel_vec = Vec::from(kernel);
        let hot_state_vec = Vec::from(hot_state);
        let serf = SerfThread::new(
            kernel_vec, checkpoint, hot_state_vec, NOCK_STACK_SIZE_LARGE, test_jets, trace,
        )
        .await?;
        Ok(Self { serf })
    }

    pub async fn load_with_hot_state_huge(
        kernel: &[u8],
        checkpoint: Option<C>,
        hot_state: &[HotEntry],
        test_jets: Vec<NounSlab>,
        trace: TraceOpts,
    ) -> Result<Self> {
        let kernel_vec = Vec::from(kernel);
        let hot_state_vec = Vec::from(hot_state);
        let serf = SerfThread::new(
            kernel_vec, checkpoint, hot_state_vec, NOCK_STACK_SIZE_HUGE, test_jets, trace,
        )
        .await?;
        Ok(Self { serf })
    }

    /// Loads a kernel with default hot state.
    ///
    /// # Arguments
    ///
    /// * `snap_dir` - Directory for storing snapshots.
    /// * `kernel` - Byte slice containing the kernel code.
    /// * `trace` - Whether to enable tracing.
    ///
    /// # Returns
    ///
    /// A new `Kernel` instance.
    pub async fn load(
        kernel: &[u8],
        checkpoint: Option<C>,
        test_jets: Vec<NounSlab>,
        trace: TraceOpts,
    ) -> Result<Self> {
        Self::load_with_hot_state(kernel, checkpoint, &Vec::new(), test_jets, trace).await
    }

    /// Produces a checkpoint of the kernel state.
    pub fn checkpoint(&self) -> impl Future<Output = Result<C>> {
        self.serf.checkpoint()
    }
}

impl<C> Kernel<C> {
    // We are very carefully ensuring the future does not contain the "self" reference to ensure no lifetime issues when spawning tasks
    pub fn poke(&self, wire: WireRepr, cause: NounSlab) -> impl Future<Output = Result<NounSlab>> {
        self.serf.poke(wire, cause)
    }

    pub fn poke_sync(&self, wire: WireRepr, cause: NounSlab) -> Result<NounSlab> {
        self.serf.poke_sync(wire, cause)
    }

    pub fn peek_sync(&self, ovo: NounSlab) -> Result<NounSlab> {
        self.serf.peek_sync(ovo)
    }

    pub fn poke_timeout(
        &self,
        wire: WireRepr,
        cause: NounSlab,
        timeout: Duration,
    ) -> impl Future<Output = Result<NounSlab>> {
        self.serf.poke_timeout(wire, cause, timeout)
    }

    // We are very carefully ensuring the future does not contain the "self" reference to ensure no lifetime issues when spawning tasks
    #[tracing::instrument(name = "crown::Kernel::peek", skip_all)]
    pub(crate) fn peek(&self, ovo: NounSlab) -> impl Future<Output = Result<NounSlab>> {
        self.serf.peek(ovo)
    }

    pub fn import(&self, state: LoadState) -> impl Future<Output = Result<()>> {
        self.serf.import(state)
    }

    pub fn export(&self) -> impl Future<Output = Result<LoadState>> {
        self.serf.export()
    }

    pub(crate) fn provide_metrics(
        &mut self,
        metrics: Arc<NockAppMetrics>,
    ) -> impl Future<Output = Result<()>> {
        self.serf.provide_metrics(metrics)
    }
}

/// Represents the Serf, which maintains context and provides an interface to
/// the Sword.
pub struct Serf {
    /// Hash of boot kernel
    pub ker_hash: Hash,
    /// The current Arvo state.
    pub arvo: Noun,
    /// The interpreter context.
    pub context: interpreter::Context,
    /// Cancellation
    pub cancel_token: NockCancelToken,
    /// The current event number.
    pub event_num: Arc<AtomicU64>,
    /// A metrics
    pub metrics: Option<Arc<NockAppMetrics>>,
}

impl Serf {
    /// Creates a new Serf instance.
    ///
    /// # Arguments
    ///
    /// * `stack` - The Nock stack.
    /// * `checkpoint` - Optional checkpoint to restore from.
    /// * `kernel_bytes` - Byte slice containing the kernel code.
    /// * `constant_hot_state` - Custom hot state entries.
    /// * `trace_info` - Optional nockvm tracing implementation.
    ///
    /// # Returns
    ///
    /// A new `Serf` instance.
    fn new<C: SerfCheckpoint>(
        mut stack: NockStack,
        checkpoint: Option<C>,
        kernel_bytes: &[u8],
        constant_hot_state: &[HotEntry],
        test_jets: Vec<NounSlab>,
        trace: TraceOpts,
    ) -> Self {
        let hot_state = [URBIT_HOT_STATE, constant_hot_state].concat();

        let mut hasher = Hasher::new();
        hasher.update(kernel_bytes);
        let ker_hash = hasher.finalize();

        let (maybe_state, cold, event_num_raw) = if let Some(c) = checkpoint {
            let saveable = c.load();

            let ker_state = saveable.state.copy_to_stack(&mut stack);
            let cold_noun = saveable.cold.copy_to_stack(&mut stack);
            let cold_vecs = Cold::from_noun(&mut stack, &cold_noun)
                .expect("Could not load cold state from snapshot");
            let cold = Cold::from_vecs(&mut stack, cold_vecs.0, cold_vecs.1, cold_vecs.2);
            if saveable.ker_hash != ker_hash {
                debug!(
                    "Loading snapshot from kernel {} into kernel {}",
                    saveable.ker_hash, ker_hash
                );
            }
            (Some(ker_state), cold, saveable.event_num)
        } else {
            (None, Cold::new(&mut stack), 0)
        };

        let event_num = Arc::new(AtomicU64::new(event_num_raw));

        let mut context = create_context(stack, &hot_state, cold, trace.into(), test_jets);
        let cancel_token = context.cancel_token();

        let mut arvo = {
            let kernel_trap = Noun::cue_bytes_slice(&mut context.stack, kernel_bytes)
                .expect("invalid kernel jam");
            let fol = T(&mut context.stack, &[D(9), D(2), D(0), D(1)]);

            if context.trace_info.is_some() {
                let start = Instant::now();
                let arvo = interpret(&mut context, kernel_trap, fol).unwrap_or_else(|err| {
                    panic!(
                        "Panicked with {err:?} at {}:{} (git sha: {:?})",
                        file!(),
                        line!(),
                        option_env!("GIT_SHA")
                    )
                });
                write_serf_trace_safe(&mut context, "boot", start);
                arvo
            } else {
                interpret(&mut context, kernel_trap, fol).unwrap_or_else(|err| {
                    panic!(
                        "Panicked with {err:?} at {}:{} (git sha: {:?})",
                        file!(),
                        line!(),
                        option_env!("GIT_SHA")
                    )
                })
            }
        };

        let mut serf = Self {
            ker_hash,
            arvo,
            context,
            event_num,
            cancel_token,
            metrics: None,
        };

        if let Some(kernel_state) = maybe_state {
            arvo = serf.load(kernel_state).expect("serf: load failed");
        }

        unsafe {
            serf.event_update(event_num_raw, arvo);
            serf.preserve_event_update_leftovers();
        }
        serf
    }

    /// Performs a peek operation on the Arvo state.
    ///
    /// # Arguments
    ///
    /// * `ovo` - The peek request noun.
    ///
    /// # Returns
    ///
    /// Result containing the peeked data or an error.
    #[tracing::instrument(skip_all)]
    pub fn peek(&mut self, ovo: Noun) -> Result<Noun> {
        if self.context.trace_info.is_some() {
            let trace_name = "peek";
            let start = Instant::now();
            let slam_res = self.slam(PEEK_AXIS, ovo);
            write_serf_trace_safe(&mut self.context, trace_name, start);

            slam_res
        } else {
            self.slam(PEEK_AXIS, ovo)
        }
    }

    /// Generates a goof (error) noun.
    ///
    /// # Arguments
    ///
    /// * `mote` - The error mote.
    /// * `traces` - Trace information.
    ///
    /// # Returns
    ///
    /// A noun representing the error.
    pub fn goof(&mut self, mote: Mote, traces: Noun) -> Noun {
        let tone = Cell::new(&mut self.context.stack, D(2), traces);
        let tang = mook(&mut self.context, tone, false)
            .expect("serf: goof: +mook crashed on bail")
            .tail();
        T(&mut self.context.stack, &[D(mote as u64), tang])
    }

    /// Performs a load operation on the Arvo state.
    ///
    /// # Arguments
    ///
    /// * `old` - The state to load.
    ///
    /// # Returns
    ///
    /// Result containing the loaded kernel or an error.
    pub fn load(&mut self, old: Noun) -> Result<Noun> {
        match self.soft(old, LOAD_AXIS, Some("load".to_string())) {
            Ok(res) => Ok(res),
            Err(goof) => {
                self.print_goof(goof);
                Err(CrownError::SerfLoadError)
            }
        }
    }

    pub fn print_goof(&mut self, goof: Noun) {
        let tang = goof
            .as_cell()
            .expect("print goof: expected goof to be a cell")
            .tail();
        tang.list_iter().for_each(|tank: Noun| {
            //  TODO: Slogger should be emitting Results in case of failure
            self.context.slogger.slog(&mut self.context.stack, 1, tank);
        });
    }

    /// Performs a poke operation on the Arvo state.
    ///
    /// # Arguments
    ///
    /// * `job` - The poke job noun.
    ///
    /// # Returns
    ///
    /// Result containing the poke response or an error.
    #[tracing::instrument(level = "info", skip_all)]
    pub fn do_poke(&mut self, job: Noun) -> Result<Noun> {
        match self.soft(job, POKE_AXIS, Some("poke".to_string())) {
            Ok(res) => {
                let cell = res.as_cell().expect("serf: poke: +slam returned atom");
                let mut fec = cell.head();
                let eve = self.event_num.load(Ordering::SeqCst);

                unsafe {
                    self.event_update(eve + 1, cell.tail());
                    self.stack().preserve(&mut fec);
                    self.preserve_event_update_leftovers();
                }
                Ok(fec)
            }
            Err(goof) => self.poke_swap(job, goof),
        }
    }

    /// Slams (applies) a gate at a specific axis of Arvo.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to slam.
    /// * `ovo` - The sample noun.
    ///
    /// # Returns
    ///
    /// Result containing the slammed result or an error.
    pub fn slam(&mut self, axis: u64, ovo: Noun) -> Result<Noun> {
        let arvo = self.arvo;
        slam(&mut self.context, arvo, axis, ovo, self.metrics.clone())
    }

    /// Performs a "soft" computation, handling errors gracefully.
    ///
    /// # Arguments
    ///
    /// * `ovo` - The input noun.
    /// * `axis` - The axis to slam.
    /// * `trace_name` - Optional name for tracing.
    ///
    /// # Returns
    ///
    /// Result containing the computed noun or an error noun.
    fn soft(&mut self, ovo: Noun, axis: u64, trace_name: Option<String>) -> Result<Noun, Noun> {
        let slam_res = if self.context.trace_info.is_some() {
            let start = Instant::now();
            let slam_res = self.slam(axis, ovo);
            write_serf_trace_safe(
                &mut self.context,
                trace_name.as_ref().unwrap_or_else(|| {
                    panic!(
                        "Panicked at {}:{} (git sha: {:?})",
                        file!(),
                        line!(),
                        option_env!("GIT_SHA")
                    )
                }),
                start,
            );

            slam_res
        } else {
            self.slam(axis, ovo)
        };

        match slam_res {
            Ok(res) => Ok(res),
            Err(error) => match error {
                CrownError::InterpreterError(e) => {
                    let (mote, traces) = match e.0 {
                        Error::Deterministic(mote, traces)
                        | Error::NonDeterministic(mote, traces) => (mote, traces),
                        Error::ScryBlocked(_) | Error::ScryCrashed(_) => {
                            panic!("serf: soft: .^ invalid outside of virtual Nock")
                        }
                    };

                    Err(self.goof(mote, traces))
                }
                _ => Err(D(0)),
            },
        }
    }

    /// Plays a list of events.
    ///
    /// # Arguments
    ///
    /// * `lit` - The list of events to play.
    ///
    /// # Returns
    ///
    /// Result containing the final Arvo state or an error.
    fn play_list(&mut self, mut lit: Noun) -> Result<Noun> {
        let mut eve = self.event_num.load(Ordering::SeqCst);
        while let Ok(cell) = lit.as_cell() {
            let ovo = cell.head();
            lit = cell.tail();
            let trace_name = if self.context.trace_info.is_some() {
                Some(format!("play [{}]", eve))
            } else {
                None
            };

            match self.soft(ovo, POKE_AXIS, trace_name) {
                Ok(res) => {
                    let arvo = res.as_cell()?.tail();
                    eve += 1;

                    unsafe {
                        self.event_update(eve, arvo);
                        self.context.stack.preserve(&mut lit);
                        self.preserve_event_update_leftovers();
                    }
                }
                Err(goof) => {
                    return Err(CrownError::KernelError(Some(goof)));
                }
            }
        }
        Ok(self.arvo)
    }

    /// Handles a poke error by swapping in a new event.
    ///
    /// # Arguments
    ///
    /// * `job` - The original poke job.
    /// * `goof` - The error noun.
    ///
    /// # Returns
    ///
    /// Result containing the new event or an error.
    fn poke_swap(&mut self, job: Noun, goof: Noun) -> Result<Noun> {
        let stack = &mut self.context.stack;
        self.context.cache = Hamt::<Noun>::new(stack);
        let job_cell = job.as_cell().expect("serf: poke: job not a cell");
        // job data is job without event_num
        let job_data = job_cell
            .tail()
            .as_cell()
            .expect("serf: poke: data not a cell");
        //  job input is job without event_num or wire
        let job_input = job_data.tail();
        let wire = T(stack, &[D(0), D(tas!(b"arvo")), D(0)]);
        let crud = DirectAtom::new_panic(tas!(b"crud"));
        let event_num = D(self.event_num.load(Ordering::SeqCst) + 1);

        let mut ovo = T(stack, &[event_num, wire, goof, job_input]);
        let trace_name = if self.context.trace_info.is_some() {
            Some(Self::poke_trace_name(
                &mut self.context.stack,
                wire,
                crud.as_atom(),
            ))
        } else {
            None
        };

        match self.soft(ovo, POKE_AXIS, trace_name) {
            Ok(res) => {
                let cell = res.as_cell().expect("serf: poke: crud +slam returned atom");
                let mut fec = cell.head();
                let eve = self.event_num.load(Ordering::SeqCst);

                unsafe {
                    self.event_update(eve + 1, cell.tail());
                    self.context.stack.preserve(&mut ovo);
                    self.context.stack.preserve(&mut fec);
                    self.preserve_event_update_leftovers();
                }
                Ok(fec)
            }
            Err(goof_crud) => Err(CrownError::KernelError(Some(goof_crud))),
        }
    }

    /// Generates a trace name for a poke operation.
    ///
    /// # Arguments
    ///
    /// * `stack` - The Nock stack.
    /// * `wire` - The wire noun.
    /// * `vent` - The vent atom.
    ///
    /// # Returns
    ///
    /// A string representing the trace name.
    fn poke_trace_name(stack: &mut NockStack, wire: Noun, vent: Atom) -> String {
        let wpc = path_to_cord(stack, wire);
        let wpc_len = met3_usize(wpc);
        let wpc_bytes = &wpc.as_ne_bytes()[0..wpc_len];
        let wpc_str = match std::str::from_utf8(wpc_bytes) {
            Ok(valid) => valid,
            Err(error) => {
                let (valid, _) = wpc_bytes.split_at(error.valid_up_to());
                unsafe { std::str::from_utf8_unchecked(valid) }
            }
        };

        let vc_len = met3_usize(vent);
        let vc_bytes = &vent.as_ne_bytes()[0..vc_len];
        let vc_str = match std::str::from_utf8(vc_bytes) {
            Ok(valid) => valid,
            Err(error) => {
                let (valid, _) = vc_bytes.split_at(error.valid_up_to());
                unsafe { std::str::from_utf8_unchecked(valid) }
            }
        };

        format!("poke [{} {}]", wpc_str, vc_str)
    }

    /// Performs a poke operation with a given cause.
    ///
    /// # Arguments
    ///
    /// * `wire` - The wire noun.
    /// * `cause` - The cause noun.
    ///
    /// # Returns
    ///
    /// Result containing the poke response or an error.
    #[tracing::instrument(level = "info", skip_all, fields(
        src = wire.source
    ))]
    pub fn poke(&mut self, wire: WireRepr, cause: Noun) -> Result<Noun> {
        let random_bytes = rand::random::<u64>();
        let bytes = random_bytes.as_bytes()?;
        let eny: Atom = Atom::from_bytes(&mut self.context.stack, &bytes);
        let our = <nockvm::noun::Atom as AtomExt>::from_value(&mut self.context.stack, 0)?; // Using 0 as default value
        let mut t_vec: Vec<u8> = vec![];
        t_vec.write_u128::<LittleEndian>(current_da().0)?;
        let now: Atom = <IndirectAtom as IndirectAtomExt>::from_bytes(
            &mut self.context.stack,
            t_vec.as_slice(),
        );

        let event_num = D(self.event_num.load(Ordering::SeqCst) + 1);
        let base_wire_noun = wire_to_noun(&mut self.context.stack, &wire);
        let wire = T(&mut self.context.stack, &[D(tas!(b"poke")), base_wire_noun]);
        let poke = T(
            &mut self.context.stack,
            &[event_num, wire, eny.as_noun(), our.as_noun(), now.as_noun(), cause],
        );

        self.do_poke(poke)
    }

    /// Updates the Serf's state after an event.
    ///
    /// # Arguments
    ///
    /// * `new_event_num` - The new event number.
    /// * `new_arvo` - The new Arvo state.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it modifies the Serf's state directly.
    #[tracing::instrument(level = "info", skip_all)]
    pub unsafe fn event_update(&mut self, new_event_num: u64, new_arvo: Noun) {
        self.arvo = new_arvo;
        self.event_num.store(new_event_num, Ordering::SeqCst);

        self.context.cache = Hamt::new(&mut self.context.stack);
        self.context.scry_stack = D(0);
    }

    /// Preserves leftovers after an event update.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it modifies the Serf's state directly.
    #[tracing::instrument(level = "info", skip_all)]
    pub unsafe fn preserve_event_update_leftovers(&mut self) {
        let stack = &mut self.context.stack;
        stack.preserve(&mut self.context.warm);
        stack.preserve(&mut self.context.test_jets);
        stack.preserve(&mut self.context.hot);
        stack.preserve(&mut self.context.cache);
        stack.preserve(&mut self.context.cold);
        stack.preserve(&mut self.arvo);
        stack.flip_top_frame(0);
        self.retag_survivors();
        #[cfg(debug_assertions)]
        self.debug_assert_offsets();
    }

    /// Returns a mutable reference to the Nock stack.
    ///
    /// # Returns
    ///
    /// A mutable reference to the `NockStack`.
    pub fn stack(&mut self) -> &mut NockStack {
        &mut self.context.stack
    }

    #[cfg(debug_assertions)]
    fn debug_assert_offsets(&mut self) {
        self.context.stack.install_arena();
        let mut work = vec![self.arvo, self.context.scry_stack];
        while let Some(noun) = work.pop() {
            if noun.is_stack_allocated() {
                panic!("serf: encountered stack pointer after preserve");
            }
            if let Ok(cell) = noun.as_cell() {
                work.push(cell.head());
                work.push(cell.tail());
            }
        }
    }

    fn retag_survivors(&mut self) {
        let stack = &self.context.stack;
        stack.install_arena();
        stack.retag_noun_tree(&mut self.arvo as *mut Noun);
        stack.retag_noun_tree(&mut self.context.scry_stack as *mut Noun);
        self.context.cache.retag(stack);
        self.context.hot.retag(stack);
        self.context.warm.retag(stack);
        self.context.cold.retag(stack);
        self.context.test_jets.retag(stack);
    }

    /// Creates a poke swap noun.
    ///
    /// # Arguments
    ///
    /// * `eve` - The event number.
    /// * `mug` - The mug value.
    /// * `ovo` - The original noun.
    /// * `fec` - The effect noun.
    ///
    /// # Returns
    ///
    /// A noun representing the poke swap.
    pub fn poke_bail(&mut self, eve: u64, mug: u64, ovo: Noun, fec: Noun) -> Noun {
        T(
            self.stack(),
            &[D(tas!(b"poke")), D(tas!(b"swap")), D(eve), D(mug), ovo, fec],
        )
    }

    /// Creates a poke bail noun.
    ///
    /// # Arguments
    ///
    /// * `lud` - The lud noun.
    ///
    /// # Returns
    ///
    /// A noun representing the poke bail.
    pub fn poke_bail_noun(&mut self, lud: Noun) -> Noun {
        T(self.stack(), &[D(tas!(b"poke")), D(tas!(b"bail")), lud])
    }
}

fn slot(noun: Noun, axis: u64) -> Result<Noun> {
    Ok(noun.slot(axis)?)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use nockvm::jets::cold::Cold;
    use nockvm::jets::hot::HotEntry;

    use super::*;

    fn dummy_serf() -> Serf {
        let mut stack = NockStack::new(1 << 18, 0);
        stack.install_arena();
        let cold = Cold::new(&mut stack);
        let hot_state: [HotEntry; 0] = [];
        let context = create_context(stack, &hot_state, cold, None, vec![]);
        let cancel_token = context.cancel_token();
        Serf {
            ker_hash: Hash::from([0; 32]),
            arvo: D(0),
            context,
            cancel_token,
            event_num: Arc::new(AtomicU64::new(0)),
            metrics: None,
        }
    }

    async fn setup_kernel(jam: &str) -> Kernel<SaveableCheckpoint> {
        let jam_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("assets")
            .join(jam);
        let jam_bytes =
            fs::read(jam_path).unwrap_or_else(|_| panic!("Failed to read {} file", jam));
        Kernel::load(&jam_bytes, None, vec![], TraceOpts::default())
            .await
            .expect("Could not load kernel")
    }

    // Convert this to an integration test and feed it the kernel.jam from Choo in CI/CD
    // https://www.youtube.com/watch?v=4m1EFMoRFvY
    // #[test]
    // #[cfg_attr(miri, ignore)]
    // fn test_kernel_boot() {
    //     let _ = setup_kernel("dumb.jam");
    // }

    // To test your own kernel, place a `kernel.jam` file in the `assets` directory
    // and uncomment the following test:
    //
    // #[test]
    // fn test_custom_kernel() {
    //     let (kernel, _temp_dir) = setup_kernel("kernel.jam");
    //     // Add your custom assertions here to test the kernel's behavior
    // }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn preserve_event_leftovers_retags_offsets() {
        let mut serf = dummy_serf();
        serf.context.stack.install_arena();
        let arvo = Cell::new(&mut serf.context.stack, D(1), D(2)).as_noun();
        assert!(arvo.is_stack_allocated());
        serf.arvo = arvo;
        unsafe {
            serf.preserve_event_update_leftovers();
        }
        assert!(
            !serf.arvo.is_stack_allocated(),
            "arvo should not retain stack pointers after preserve"
        );
        #[cfg(debug_assertions)]
        serf.debug_assert_offsets();
    }
}

pub trait SerfCheckpoint: Send {
    fn new(
        stack: &mut NockStack,
        ker_hash: Hash,
        event_num: u64,
        kernel_state: Noun,
        cold_state: Cold,
        metrics: &Option<Arc<NockAppMetrics>>,
    ) -> Self;

    fn load(self) -> SaveableCheckpoint;
}

impl SerfCheckpoint for SaveableCheckpoint {
    fn new(
        stack: &mut NockStack,
        ker_hash: Hash,
        event_num: u64,
        kernel_state: Noun,
        cold_state: Cold,
        metrics: &Option<Arc<NockAppMetrics>>,
    ) -> Self {
        let cold_noun_start = Instant::now();
        // Cold state has nouns in it which are *not* copied in into_noun
        // TODO: FIX THIS FOOTGUN
        let cold_stack_noun = cold_state.into_noun(stack);
        let mut cold_slab: NounSlab = NounSlab::new();
        let cold_copy = cold_slab.copy_into(cold_stack_noun);
        cold_slab.set_root(cold_copy);
        let cold_noun_elapsed = cold_noun_start.elapsed();

        let state_copy_start = Instant::now();
        let mut state_slab: NounSlab = NounSlab::new();
        let state_copy = state_slab.copy_into(kernel_state);
        state_slab.set_root(state_copy);
        let state_copy_elapsed = state_copy_start.elapsed();

        if let Some(metrics) = metrics {
            metrics
                .serf_loop_noun_encode_cold_state
                .add_timing(&cold_noun_elapsed);
            metrics
                .serf_loop_copy_state_noun
                .add_timing(&state_copy_elapsed);
        }
        Self {
            ker_hash,
            event_num,
            state: state_slab,
            cold: cold_slab,
        }
    }

    fn load(self) -> SaveableCheckpoint {
        self
    }
}
```

------------------
