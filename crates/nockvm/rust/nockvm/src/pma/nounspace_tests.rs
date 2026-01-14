//! Tests for NounSpace safety mechanisms.
//!
//! These tests verify that NounSpace properly detects and rejects:
//! - Stale epoch references (after stack flip)
//! - Arena mismatches (accessing stack nouns with PMA-only space, etc.)

use std::panic::catch_unwind;

use crate::mem::NockStack;
use crate::noun::{Cell, IndirectAtom, Noun, NounSpace, D};
use crate::pma::{Pma, PmaCopy};

use super::test_pma_path;

/// Helper to assert that a closure panics with a message containing the expected substring.
fn assert_panics<F, R>(f: F, expected_msg: &str)
where
    F: FnOnce() -> R + std::panic::UnwindSafe,
{
    match catch_unwind(f) {
        Ok(_) => panic!("Expected panic with '{}' but closure completed normally", expected_msg),
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.as_str()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                *s
            } else {
                ""
            };
            assert!(
                msg.contains(expected_msg),
                "Expected panic message containing '{}', got: '{}'",
                expected_msg,
                msg
            );
        }
    }
}

// =============================================================================
// Stack Epoch Tests
// =============================================================================

/// Test that NounSpace detects epoch mismatch after a single stack flip.
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_nounspace_epoch_detects_single_flip() {
    let mut stack = NockStack::new(1 << 12, 0);

    // Create a cell on the stack
    let cell = Cell::new(&mut stack, D(1), D(2));
    let noun = cell.as_noun();

    // Capture epoch before flip
    let epoch_before = stack.stack_epoch_snapshot();
    let space_before_flip = NounSpace::stack_only(&stack);

    // Flip the stack - this increments the epoch
    unsafe { stack.flip_top_frame(0) };

    // Verify epoch changed
    let epoch_after = stack.stack_epoch_snapshot();
    assert_ne!(epoch_before, epoch_after, "flip should increment epoch");

    // Using the old NounSpace should panic
    assert_panics(
        || {
            let cell = noun.as_cell().expect("noun is a cell");
            let _ = cell.head(&space_before_flip);
        },
        "epoch",
    );
}

/// Test that epoch increments correctly across multiple flips and all old NounSpaces are rejected.
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_nounspace_epoch_detects_multiple_flips() {
    let mut stack = NockStack::new(1 << 12, 0);

    // Create a cell on the stack
    let cell = Cell::new(&mut stack, D(1), D(2));
    let noun = cell.as_noun();

    // Collect NounSpaces at each epoch
    let mut old_spaces: Vec<(u64, NounSpace)> = Vec::new();
    let initial_epoch = stack.stack_epoch_snapshot();
    old_spaces.push((initial_epoch, NounSpace::stack_only(&stack)));

    // Flip multiple times, saving NounSpaces
    const NUM_FLIPS: u64 = 5;
    for i in 1..=NUM_FLIPS {
        unsafe { stack.flip_top_frame(0) };

        let current_epoch = stack.stack_epoch_snapshot();
        assert_eq!(current_epoch, initial_epoch + i, "epoch should increment by 1 each flip");

        if i < NUM_FLIPS {
            old_spaces.push((current_epoch, NounSpace::stack_only(&stack)));
        }
    }

    // All old NounSpaces should be rejected
    for (epoch, old_space) in &old_spaces {
        let epoch_copy = *epoch;
        let result = catch_unwind(|| {
            let cell = noun.as_cell().expect("noun is a cell");
            let _ = cell.head(old_space);
        });
        assert!(
            result.is_err(),
            "NounSpace from epoch {} should panic when current epoch is {}",
            epoch_copy,
            stack.stack_epoch_snapshot()
        );
    }

    // Fresh NounSpace should work (though data may be garbage after flips)
    let current_space = NounSpace::stack_only(&stack);
    let cell = noun.as_cell().expect("noun is a cell");
    let _ = cell.head(&current_space); // Should not panic
}

/// Test that CellHandle.head() and .tail() panic after stack flip.
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_cellhandle_access_after_flip() {
    let mut stack = NockStack::new(1 << 12, 0);

    let cell = Cell::new(&mut stack, D(1), D(2));
    let noun = cell.as_noun();

    let space = NounSpace::stack_only(&stack);
    let cell_handle = space.handle(noun).as_cell().expect("noun is a cell");

    // Flip invalidates the NounSpace inside cell_handle
    unsafe { stack.flip_top_frame(0) };

    assert_panics(|| { let _ = cell_handle.head(); }, "epoch");
    assert_panics(|| { let _ = cell_handle.tail(); }, "epoch");
}

/// Test that AtomHandle.data_pointer() panics after stack flip for indirect atoms.
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_indirect_atom_access_after_flip() {
    let mut stack = NockStack::new(1 << 12, 0);

    // Create an indirect atom (needs more than one u64)
    let data: [u64; 2] = [0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321];
    let atom = unsafe { IndirectAtom::new_raw(&mut stack, data.len(), data.as_ptr()) };
    let noun = atom.as_noun();

    let space = NounSpace::stack_only(&stack);
    let atom_handle = space.handle(noun).as_atom().expect("noun is an atom");
    assert!(atom_handle.is_indirect(), "test requires indirect atom");

    unsafe { stack.flip_top_frame(0) };

    assert_panics(|| { let _ = atom_handle.data_pointer(); }, "epoch");
}

// =============================================================================
// Arena Mismatch Tests
// =============================================================================

/// Test that accessing a stack noun with a PMA-only NounSpace panics.
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_stack_noun_with_pma_only_space() {
    let mut stack = NockStack::new(1 << 12, 0);
    let pma = Pma::new(1 << 10, test_pma_path("pma_only_space")).expect("create PMA");

    let cell = Cell::new(&mut stack, D(1), D(2));
    let noun = cell.as_noun();

    // PMA-only NounSpace doesn't know about the stack
    let pma_only_space = NounSpace::pma_only(&pma);

    assert_panics(
        || {
            let cell = noun.as_cell().expect("noun is a cell");
            let _ = cell.head(&pma_only_space);
        },
        "arena",
    );
}

/// Test that traversing from stack into PMA fails with stack-only NounSpace.
#[test]
#[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
fn test_traversal_crosses_unknown_arena() {
    let mut stack = NockStack::new(1 << 12, 0);
    let mut pma = Pma::new(1 << 10, test_pma_path("cross_arena")).expect("create PMA");

    // Create inner cell and evacuate to PMA
    let inner_cell = Cell::new(&mut stack, D(100), D(200));
    let mut inner_noun = inner_cell.as_noun();
    unsafe { inner_noun.copy_to_pma(&stack, &mut pma) };

    // Create outer cell on stack pointing to PMA cell
    let outer_cell = Cell::new(&mut stack, inner_noun, D(999));
    let outer_noun = outer_cell.as_noun();

    // Stack-only NounSpace
    let stack_only_space = NounSpace::stack_only(&stack);

    // Can access outer cell (on stack)
    let outer = outer_noun.as_cell().expect("outer is a cell");
    let head_noun = outer.head(&stack_only_space);

    // But accessing the head's contents (in PMA) should fail
    assert_panics(
        || {
            let head_cell = head_noun.as_cell().expect("head is a cell");
            let _ = head_cell.head(&stack_only_space);
        },
        "arena",
    );
}
