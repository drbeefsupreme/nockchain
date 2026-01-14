//! Test harness for NounSpace/NounHandle behavior testing.
//!
//! This module provides `NounSpaceTestBed`, a controlled environment for testing
//! how NounSpace handles various failure modes like epoch mismatches, PMA resets,
//! arena mismatches, and forwarding pointer corruption.

use std::panic::catch_unwind;
use std::sync::Arc;

use crate::mem::NockStack;
use crate::noun::{AllocLocation, Cell, IndirectAtom, Noun, NounSpace};
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
