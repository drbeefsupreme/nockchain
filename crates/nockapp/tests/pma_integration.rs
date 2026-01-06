//! Integration tests for PMA (Persistent Memory Arena) with real kernels.
//!
//! These tests verify that the PMA correctly persists kernel state across events.

use std::path::PathBuf;

use nockapp::kernel::form::Serf;
use nockvm::noun::D;
use nockvm::pma::Pma;
use tempfile::TempDir;

/// Load the dumb.jam kernel bytes
fn load_dumb_kernel() -> Vec<u8> {
    let kernel_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("assets")
        .join("dumb.jam");
    std::fs::read(&kernel_path)
        .unwrap_or_else(|e| panic!("Failed to read dumb.jam at {:?}: {}", kernel_path, e))
}

/// Test that booting a real kernel with PMA enabled works correctly.
///
/// This test verifies:
/// 1. A real kernel (dumb.jam) can be loaded and booted
/// 2. PMA can be enabled after boot
/// 3. persist_event_to_pma copies the kernel state to PMA
/// 4. The arvo state is in offset form after persistence
/// 5. The PMA data is readable via the installed arena
#[test]
#[cfg_attr(miri, ignore = "PMA uses memfd_create which is unsupported in Miri")]
fn test_pma_boot_dumb_kernel() {
    // Create temp directory for PMA
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let pma_path = temp_dir.path().join("pma");

    // Load kernel - use a large stack for dumb kernel
    let kernel_bytes = load_dumb_kernel();
    let stack_size = 1 << 27; // 128MB - dumb kernel needs more space

    // Create Serf with real kernel using new_for_pma_testing which does NOT
    // call preserve_event_update_leftovers. This keeps data in stack-pointer
    // form so copy_to_pma can properly copy it to PMA.
    let mut serf = Serf::new_for_pma_testing(&kernel_bytes, stack_size);

    // Verify arvo is a cell after kernel boot
    assert!(
        serf.arvo.is_cell(),
        "arvo should be a cell after kernel boot"
    );

    let initial_arvo = serf.arvo;

    // Create and enable PMA - use a reasonably sized PMA for dumb kernel
    // dumb.jam is ~19MB, so we need enough space for the booted state
    let pma = Pma::new(50_000_000, pma_path).expect("Failed to create PMA");
    serf.enable_pma(pma);

    // Persist to PMA
    unsafe {
        serf.persist_event_to_pma();
    }

    // Verify we can still read the arvo structure via the installed PMA arena
    let arvo_cell = serf.arvo.as_cell().expect("arvo should still be readable as cell");

    // The outer wrapper stores state as [%0 desk-hash internal]
    // Just verify we can traverse the structure
    let _head = arvo_cell.head();
    let _tail = arvo_cell.tail();

    // Verify the arvo changed (data was copied to PMA and offset updated)
    unsafe {
        assert!(
            serf.arvo.as_raw() != initial_arvo.as_raw(),
            "arvo should have different raw value after PMA copy (different offset)"
        );
    }

    // Verify all data is in PMA using assert_in_pma
    use nockvm::pma::PmaCopy;
    if let Some(ref pma) = serf.pma {
        serf.arvo.assert_in_pma(pma);
    }
}

/// Test that a second "event" (simulated state change) can be persisted after the first.
///
/// This verifies that after initial persistence:
/// 1. New data can be allocated on the stack
/// 2. References to existing PMA data work correctly
/// 3. A second persist_event_to_pma succeeds
#[test]
#[cfg_attr(miri, ignore = "PMA uses memfd_create which is unsupported in Miri")]
fn test_pma_multiple_persist_dumb_kernel() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let pma_path = temp_dir.path().join("pma");

    let kernel_bytes = load_dumb_kernel();
    let stack_size = 1 << 27;

    // Use new_for_pma_testing which does NOT call preserve_event_update_leftovers.
    // This keeps data in stack-pointer form so copy_to_pma can properly copy it.
    let mut serf = Serf::new_for_pma_testing(&kernel_bytes, stack_size);

    // Enable PMA and persist initial boot state
    let pma = Pma::new(100_000_000, pma_path).expect("Failed to create PMA");
    serf.enable_pma(pma);

    unsafe {
        serf.persist_event_to_pma();
    }

    // Verify first persist worked by checking data is in PMA
    use nockvm::pma::PmaCopy;
    let arvo_after_first = serf.arvo;
    if let Some(ref pma) = serf.pma {
        serf.arvo.assert_in_pma(pma);
    }

    // The arvo is now in offset form pointing to PMA.
    // In a real event, the kernel would compute new_arvo referencing parts
    // of the old arvo. We can simulate this by wrapping the existing arvo
    // in a new cell.
    //
    // Note: After persist_event_to_pma, the stack was reset, so we can
    // allocate new data on the (now empty) stack.
    let new_data = {
        let stack = &mut serf.context.stack;
        // Create a cell that references the existing PMA arvo
        // This simulates a kernel that produces new state while keeping
        // references to unchanged parts of the old state
        use nockvm::noun::Cell;
        Cell::new(stack, D(12345), arvo_after_first).as_noun()
    };

    // The new_data cell itself is on the stack (not yet copied to PMA),
    // while its tail (arvo_after_first) is already in PMA.

    // Replace arvo with our new structure (simulating event result)
    serf.arvo = new_data;

    // Persist again
    unsafe {
        serf.persist_event_to_pma();
    }

    // Verify second persist worked by checking data is in PMA
    if let Some(ref pma) = serf.pma {
        serf.arvo.assert_in_pma(pma);
    }

    // Verify we can traverse the persisted structure
    let cell = serf.arvo.as_cell().expect("should be cell");
    assert_eq!(
        cell.head().as_direct().expect("head should be direct").data(),
        12345,
        "head should be our marker value"
    );

    // The tail should be the original arvo (now also in PMA)
    assert!(
        cell.tail().as_cell().is_ok(),
        "tail (original arvo) should be readable"
    );

    println!("Successfully persisted multiple states to PMA");
}

/// Test that PMA assertion functions work correctly on persisted data.
///
/// This verifies that assert_in_pma correctly validates that all
/// data is in the PMA after persistence.
#[test]
#[cfg_attr(miri, ignore = "PMA uses memfd_create which is unsupported in Miri")]
fn test_pma_assert_in_pma_dumb_kernel() {
    use nockvm::pma::PmaCopy;

    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let pma_path = temp_dir.path().join("pma");

    let kernel_bytes = load_dumb_kernel();
    let stack_size = 1 << 27;

    // Use new_for_pma_testing which does NOT call preserve_event_update_leftovers.
    // This keeps data in stack-pointer form so copy_to_pma can properly copy it.
    let mut serf = Serf::new_for_pma_testing(&kernel_bytes, stack_size);

    let pma = Pma::new(50_000_000, pma_path).expect("Failed to create PMA");
    serf.enable_pma(pma);

    unsafe {
        serf.persist_event_to_pma();
    }

    // Now verify using assert_in_pma that all data is in PMA
    // This would panic if any pointers were still pointing to stack
    if let Some(ref pma) = serf.pma {
        serf.arvo.assert_in_pma(pma);
        serf.context.warm.assert_in_pma(pma);
        serf.context.hot.assert_in_pma(pma);
        serf.context.cold.assert_in_pma(pma);
        serf.context.cache.assert_in_pma(pma);
    } else {
        panic!("PMA should be set after enable_pma");
    }

    println!("All PMA assertions passed");
}
