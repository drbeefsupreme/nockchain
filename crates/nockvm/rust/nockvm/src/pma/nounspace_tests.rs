//! Test harness for NounSpace/NounHandle behavior testing.
//!
//! This module provides `NounSpaceTestBed`, a controlled environment for testing
//! how NounSpace handles various failure modes like epoch mismatches, PMA resets,
//! arena mismatches, and forwarding pointer corruption.

use std::panic::catch_unwind;
use std::sync::Arc;

use crate::mem::NockStack;
use crate::noun::{AllocLocation, Cell, IndirectAtom, Noun, NounSpace, D};
use crate::pma::{Pma, PmaCopy};

use super::test_pma_path;

/// Sentinel value used to poison freed/reset memory regions.
/// Chosen to be obviously invalid as a noun (high bits set in a pattern
/// that doesn't match any valid noun tag).
const POISON_WORD: u64 = 0xDEAD_BEEF_DEAD_BEEF;

/// A noun with metadata about its creation context.
///
/// This tracks factual information about when and where a noun was created,
/// allowing tests to reason about whether operations should succeed or fail.
#[derive(Debug, Clone, Copy)]
pub struct TrackedNoun {
    /// The noun itself
    pub noun: Noun,
    /// Where the noun was allocated (Stack, PmaPtr, or PmaOffset)
    pub location: AllocLocation,
    /// The stack epoch when this noun was created
    pub created_epoch: u64,
}

impl TrackedNoun {
    /// Create a new TrackedNoun with the given metadata
    pub fn new(noun: Noun, location: AllocLocation, created_epoch: u64) -> Self {
        Self {
            noun,
            location,
            created_epoch,
        }
    }
}

/// A controlled environment for testing NounSpace behavior.
///
/// Provides helpers for creating nouns, NounSpaces, and performing operations
/// that can invalidate noun references (stack flip, PMA reset, etc.).
pub struct NounSpaceTestBed {
    /// The NockStack for ephemeral allocations
    stack: NockStack,
    /// Optional PMA for persistent storage
    pma: Option<Pma>,
}

impl NounSpaceTestBed {
    /// Create a new test bed with the given stack and optional PMA sizes.
    ///
    /// # Arguments
    /// * `stack_words` - Size of the NockStack in 64-bit words
    /// * `pma_words` - Size of the PMA in words, or None for stack-only testing
    pub fn new(stack_words: usize, pma_words: Option<usize>) -> Self {
        let stack = NockStack::new(stack_words, 0);
        let pma = pma_words.map(|words| {
            let path = test_pma_path("nounspace_testbed");
            Pma::new(words, path).expect("Failed to create test PMA")
        });
        Self { stack, pma }
    }

    // =========================================================================
    // NounSpace Creation
    // =========================================================================

    /// Get a NounSpace for both stack and PMA (if present).
    ///
    /// This captures the current epoch, so it will be valid until the next
    /// stack flip or reset.
    pub fn full_space(&self) -> NounSpace {
        match &self.pma {
            Some(pma) => NounSpace::new(&self.stack, pma),
            None => NounSpace::stack_only(&self.stack),
        }
    }

    /// Get a NounSpace for stack only, even if PMA exists.
    ///
    /// Useful for testing arena mismatch detection when accessing PMA nouns.
    pub fn stack_space(&self) -> NounSpace {
        NounSpace::stack_only(&self.stack)
    }

    /// Get a NounSpace for PMA only.
    ///
    /// # Panics
    /// Panics if no PMA was configured.
    pub fn pma_space(&self) -> NounSpace {
        let pma = self.pma.as_ref().expect("pma_space called but no PMA configured");
        NounSpace::pma_only(pma)
    }

    /// Get a NounSpace with a specific epoch snapshot.
    ///
    /// This creates a NounSpace that will fail epoch validation if the stack's
    /// current epoch doesn't match the provided snapshot. Useful for testing
    /// that stale NounSpaces are properly detected.
    pub fn space_at_epoch(&self, epoch: u64) -> NounSpace {
        let pma_arena = self.pma.as_ref().map(|p| Arc::clone(p.arena()));
        NounSpace::with_epoch(&self.stack, pma_arena, epoch)
    }

    // =========================================================================
    // Noun Creation
    // =========================================================================

    /// Create a cell on the stack and track it.
    ///
    /// Returns a TrackedNoun with the cell's location and creation epoch.
    pub fn cell(&mut self, head: Noun, tail: Noun) -> TrackedNoun {
        let epoch = self.current_epoch();
        let cell = Cell::new(&mut self.stack, head, tail);
        TrackedNoun::new(cell.as_noun(), AllocLocation::Stack, epoch)
    }

    /// Create an indirect atom on the stack from the given data.
    ///
    /// The data slice is copied into a new indirect atom allocation.
    /// Returns a TrackedNoun with the atom's location and creation epoch.
    pub fn indirect_atom(&mut self, data: &[u64]) -> TrackedNoun {
        let epoch = self.current_epoch();
        let atom = unsafe {
            IndirectAtom::new_raw(&mut self.stack, data.len(), data.as_ptr())
        };
        TrackedNoun::new(atom.as_noun(), AllocLocation::Stack, epoch)
    }

    // =========================================================================
    // Dangerous Operations (can invalidate NounSpaces or noun data)
    // =========================================================================

    /// Flip the top stack frame.
    ///
    /// This increments the stack epoch, invalidating any NounSpaces created
    /// before the flip. Existing NounSpaces will panic on use due to epoch
    /// mismatch.
    pub fn flip(&mut self) {
        unsafe {
            self.stack.flip_top_frame(0);
        }
    }

    /// Reset the PMA allocation pointer to zero.
    ///
    /// **WARNING**: This does NOT increment any epoch counter! Existing
    /// NounSpaces that reference the PMA will still "work" but will access
    /// invalid/reused memory. This is a known gap in the safety model.
    ///
    /// # Panics
    /// Panics if no PMA was configured.
    pub fn reset_pma(&mut self) {
        let pma = self.pma.as_mut().expect("reset_pma called but no PMA configured");
        pma.reset();
    }

    /// Push a new stack frame with the given number of local slots.
    pub fn push_frame(&mut self, slots: usize) {
        self.stack.frame_push(slots);
    }

    /// Pop the current stack frame.
    ///
    /// **WARNING**: Any nouns allocated in the popped frame become invalid.
    pub fn pop_frame(&mut self) {
        unsafe {
            self.stack.frame_pop();
        }
    }

    // =========================================================================
    // Evacuation (copy to PMA)
    // =========================================================================

    /// Evacuate a noun from the stack to the PMA.
    ///
    /// This copies all allocated substructure to the PMA and converts the noun
    /// to offset form. The original stack data will have forwarding pointers
    /// set, corrupting it for direct access.
    ///
    /// # Panics
    /// Panics if no PMA was configured.
    pub fn evacuate(&mut self, noun: &mut Noun) {
        let pma = self.pma.as_mut().expect("evacuate called but no PMA configured");
        unsafe {
            noun.copy_to_pma(&self.stack, pma);
        }
    }

    // =========================================================================
    // Memory Poisoning
    // =========================================================================

    /// Fill the PMA data region with poison values.
    ///
    /// Call this after `reset_pma()` to make stale reads obvious.
    /// The poison pattern is chosen to be invalid as a noun.
    ///
    /// # Panics
    /// Panics if no PMA was configured.
    pub fn poison_pma(&mut self) {
        let pma = self.pma.as_ref().expect("poison_pma called but no PMA configured");
        let base = pma.arena().base_ptr() as *mut u64;
        let words = pma.size_words();
        unsafe {
            for i in 0..words {
                *base.add(i) = POISON_WORD;
            }
        }
    }

    /// Check if a value matches the poison pattern.
    pub fn is_poisoned(&self, value: u64) -> bool {
        value == POISON_WORD
    }

    /// Get the poison word value for manual checks.
    pub fn poison_word(&self) -> u64 {
        POISON_WORD
    }

    // =========================================================================
    // Queries
    // =========================================================================

    /// Get the current stack epoch.
    pub fn current_epoch(&self) -> u64 {
        self.stack.stack_epoch_snapshot()
    }

    /// Get the current PMA allocation offset in words.
    ///
    /// Returns None if no PMA is configured.
    pub fn pma_alloc_offset(&self) -> Option<usize> {
        self.pma.as_ref().map(|p| p.alloc_offset())
    }

    /// Get a reference to the underlying NockStack.
    pub fn stack(&self) -> &NockStack {
        &self.stack
    }

    /// Get a mutable reference to the underlying NockStack.
    pub fn stack_mut(&mut self) -> &mut NockStack {
        &mut self.stack
    }

    /// Get a reference to the underlying PMA, if configured.
    pub fn pma(&self) -> Option<&Pma> {
        self.pma.as_ref()
    }

    /// Get a mutable reference to the underlying PMA, if configured.
    pub fn pma_mut(&mut self) -> Option<&mut Pma> {
        self.pma.as_mut()
    }

    // =========================================================================
    // Assertions
    // =========================================================================

    /// Assert that a closure panics with a message containing the expected substring.
    ///
    /// Returns true if the closure panicked with a matching message.
    /// Returns false if the closure didn't panic or the message didn't match.
    ///
    /// # Example
    /// ```ignore
    /// let bed = NounSpaceTestBed::new(1024, None);
    /// let stale_space = bed.space_at_epoch(0);
    /// bed.flip();
    /// assert!(bed.assert_panics(
    ///     || stale_space.handle(some_noun).as_cell(),
    ///     "epoch"
    /// ));
    /// ```
    pub fn assert_panics<F, R>(&self, f: F, expected_msg: &str) -> bool
    where
        F: FnOnce() -> R + std::panic::UnwindSafe,
    {
        match catch_unwind(f) {
            Ok(_) => false, // Didn't panic
            Err(e) => {
                if let Some(msg) = e.downcast_ref::<String>() {
                    msg.contains(expected_msg)
                } else if let Some(msg) = e.downcast_ref::<&str>() {
                    msg.contains(expected_msg)
                } else {
                    // Panicked but couldn't extract message
                    false
                }
            }
        }
    }

    /// Assert that a closure panics (regardless of message).
    ///
    /// Returns true if the closure panicked, false otherwise.
    pub fn assert_panics_any<F, R>(&self, f: F) -> bool
    where
        F: FnOnce() -> R + std::panic::UnwindSafe,
    {
        catch_unwind(f).is_err()
    }
}

impl Drop for NounSpaceTestBed {
    fn drop(&mut self) {
        // Clean up PMA temp file if it exists
        if let Some(pma) = &self.pma {
            // The Pma doesn't expose its path, but the temp file will be cleaned
            // up by the OS eventually. For explicit cleanup, we'd need to track
            // the path ourselves.
            let _ = pma; // silence unused warning
        }
    }
}

// =============================================================================
// Stack Epoch Tests
// =============================================================================

/// Test that NounSpace detects epoch mismatch after a single stack flip.
///
/// This verifies the fundamental safety mechanism: a NounSpace captures the
/// stack's epoch at creation time, and any attempt to resolve pointers after
/// the stack has been flipped (incrementing the epoch) should panic.
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_nounspace_epoch_detects_single_flip() {
    let mut bed = NounSpaceTestBed::new(1 << 12, None);

    // Create a cell on the stack
    let tracked = bed.cell(D(1), D(2));
    let noun = tracked.noun;

    // Capture epoch and create a NounSpace before flipping
    let epoch_before = bed.current_epoch();
    let space_before_flip = bed.full_space();

    // Flip the stack - this should increment the epoch
    bed.flip();

    // Verify epoch actually changed
    let epoch_after = bed.current_epoch();
    assert_ne!(
        epoch_before, epoch_after,
        "flip should increment epoch (was {}, now {})",
        epoch_before, epoch_after
    );

    // Attempting to use the old NounSpace should panic with epoch message
    let panicked = bed.assert_panics(
        || {
            // Try to access the cell through the stale NounSpace
            // This should trigger epoch validation in classify_ptr -> assert_stack_epoch
            let cell = noun.as_cell().expect("noun is a cell");
            let _ = cell.head(&space_before_flip);
        },
        "epoch",
    );

    assert!(
        panicked,
        "Expected panic with 'epoch' message when using NounSpace after stack flip"
    );
}

/// Test that epoch increments correctly across multiple flips and all old NounSpaces are rejected.
///
/// This verifies that:
/// 1. Each flip increments the epoch by exactly 1
/// 2. NounSpaces from ANY previous epoch are rejected, not just the immediately prior one
/// 3. Only a NounSpace with the current epoch is accepted
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_nounspace_epoch_detects_multiple_flips() {
    let mut bed = NounSpaceTestBed::new(1 << 12, None);

    // Create a cell on the stack
    let tracked = bed.cell(D(1), D(2));
    let noun = tracked.noun;

    // Collect NounSpaces at each epoch
    let mut old_spaces: Vec<(u64, NounSpace)> = Vec::new();

    // Capture initial state
    let initial_epoch = bed.current_epoch();
    old_spaces.push((initial_epoch, bed.full_space()));

    // Flip multiple times, capturing state at each epoch
    const NUM_FLIPS: u64 = 5;
    for i in 1..=NUM_FLIPS {
        bed.flip();

        let current_epoch = bed.current_epoch();

        // Verify epoch incremented by exactly 1
        assert_eq!(
            current_epoch,
            initial_epoch + i,
            "After flip {}, epoch should be {} but was {}",
            i,
            initial_epoch + i,
            current_epoch
        );

        // Save this epoch's NounSpace (except the last one, which we'll test as valid)
        if i < NUM_FLIPS {
            old_spaces.push((current_epoch, bed.full_space()));
        }
    }

    let final_epoch = bed.current_epoch();
    assert_eq!(
        final_epoch,
        initial_epoch + NUM_FLIPS,
        "Final epoch should be initial + NUM_FLIPS"
    );

    // Verify ALL old NounSpaces are rejected
    for (epoch, old_space) in &old_spaces {
        let epoch_copy = *epoch;
        let panicked = bed.assert_panics(
            || {
                let cell = noun.as_cell().expect("noun is a cell");
                let _ = cell.head(old_space);
            },
            "epoch",
        );

        assert!(
            panicked,
            "NounSpace from epoch {} should panic when current epoch is {} (snapshot {})",
            epoch_copy,
            final_epoch,
            epoch_copy
        );
    }

    // Verify a fresh NounSpace with current epoch DOES work
    let current_space = bed.full_space();
    // This should NOT panic - if it does, the test will fail
    let cell = noun.as_cell().expect("noun is a cell");
    let head = cell.head(&current_space);

    // Note: The actual data may be garbage after flips since the memory was reused,
    // but the epoch check should pass. We're just verifying no panic occurs.
    // In a real scenario, you'd preserve data across flips if you need it.
    let _ = head; // We just care that we didn't panic
}

/// Test that CellHandle.head() and .tail() panic after stack flip.
///
/// This tests the NounHandle/CellHandle API specifically, verifying that
/// the handle's internal NounSpace reference properly validates epochs
/// when accessing cell contents via either head or tail.
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_nounhandle_head_tail_access_after_flip() {
    let mut bed = NounSpaceTestBed::new(1 << 12, None);

    // Create a cell on the stack
    let tracked = bed.cell(D(1), D(2));
    let noun = tracked.noun;

    // Create a NounSpace and get a CellHandle
    let space = bed.full_space();
    let cell_handle = space.handle(noun).as_cell().expect("noun is a cell");

    // Flip the stack - this invalidates the NounSpace inside cell_handle
    bed.flip();

    // Attempting to call .head() on the CellHandle should panic
    let head_panicked = bed.assert_panics(
        || {
            let _ = cell_handle.head();
        },
        "epoch",
    );

    assert!(
        head_panicked,
        "CellHandle.head() should panic with 'epoch' message after stack flip"
    );

    // Attempting to call .tail() on the CellHandle should also panic
    let tail_panicked = bed.assert_panics(
        || {
            let _ = cell_handle.tail();
        },
        "epoch",
    );

    assert!(
        tail_panicked,
        "CellHandle.tail() should panic with 'epoch' message after stack flip"
    );
}

/// Test that AtomHandle.data_pointer() panics after stack flip for indirect atoms.
///
/// Indirect atoms have their data stored in allocated memory on the stack.
/// Accessing that data through an AtomHandle after a flip should trigger
/// epoch validation and panic.
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_indirect_atom_data_access_after_flip() {
    let mut bed = NounSpaceTestBed::new(1 << 12, None);

    // Create an indirect atom on the stack (needs more than one u64 to be indirect)
    let data: [u64; 2] = [0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321];
    let tracked = bed.indirect_atom(&data);
    let noun = tracked.noun;

    // Create a NounSpace and get an AtomHandle
    let space = bed.full_space();
    let atom_handle = space.handle(noun).as_atom().expect("noun is an atom");

    // Verify it's actually indirect (not direct)
    assert!(
        atom_handle.is_indirect(),
        "Test requires an indirect atom, but got a direct atom"
    );

    // Flip the stack - this invalidates the NounSpace inside atom_handle
    bed.flip();

    // Attempting to call .data_pointer() on the AtomHandle should panic
    let panicked = bed.assert_panics(
        || {
            let _ = atom_handle.data_pointer();
        },
        "epoch",
    );

    assert!(
        panicked,
        "AtomHandle.data_pointer() should panic with 'epoch' message after stack flip"
    );
}

// =============================================================================
// Arena Mismatch Tests
// =============================================================================

/// Test that accessing a stack-allocated noun with a PMA-only NounSpace panics.
///
/// A PMA-only NounSpace has no stack arena configured, so it cannot resolve
/// pointers to stack-allocated nouns. This should panic with a message about
/// the pointer not being within any known arena.
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_stack_noun_with_pma_only_space() {
    let mut bed = NounSpaceTestBed::new(1 << 12, Some(1 << 10));

    // Create a cell on the stack
    let tracked = bed.cell(D(1), D(2));
    let noun = tracked.noun;

    // Get a PMA-only NounSpace (no stack arena)
    let pma_only_space = bed.pma_space();

    // Attempting to access the stack noun with PMA-only space should panic
    let panicked = bed.assert_panics(
        || {
            let cell = noun.as_cell().expect("noun is a cell");
            let _ = cell.head(&pma_only_space);
        },
        "arena",
    );

    assert!(
        panicked,
        "Accessing stack noun with PMA-only NounSpace should panic with 'arena' message"
    );
}

/// Test that traversing a structure that crosses arena boundaries panics mid-traversal.
///
/// This creates a stack cell that points to a PMA cell (via evacuation), then
/// attempts to traverse with a stack-only NounSpace. The first access (stack cell)
/// works, but following the pointer to the PMA cell fails.
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_traversal_crosses_unknown_arena() {
    let mut bed = NounSpaceTestBed::new(1 << 12, Some(1 << 10));

    // Create an inner cell and evacuate it to PMA
    let inner_tracked = bed.cell(D(100), D(200));
    let mut inner_noun = inner_tracked.noun;
    bed.evacuate(&mut inner_noun);
    // inner_noun is now in PMA offset form

    // Create an outer cell on the stack that points to the PMA cell
    let outer_tracked = bed.cell(inner_noun, D(999));
    let outer_noun = outer_tracked.noun;

    // Get a stack-only NounSpace (no PMA)
    let stack_only_space = bed.stack_space();

    // Accessing the outer cell's head should work initially (it's on the stack)
    // But the head value is a PMA offset-form noun, so resolving IT should fail
    let outer_cell = outer_noun.as_cell().expect("outer is a cell");

    // Getting head returns the raw noun (PMA offset form) - this works
    let head_noun = outer_cell.head(&stack_only_space);

    // But trying to USE that head noun (which is in PMA) should panic
    // because stack_only_space doesn't know about the PMA
    let panicked = bed.assert_panics(
        || {
            // head_noun is a PMA cell in offset form
            // Trying to access its contents requires resolving the PMA offset
            let head_cell = head_noun.as_cell().expect("head is a cell");
            let _ = head_cell.head(&stack_only_space);
        },
        "arena",
    );

    assert!(
        panicked,
        "Traversing into PMA with stack-only NounSpace should panic with 'arena' message"
    );
}

