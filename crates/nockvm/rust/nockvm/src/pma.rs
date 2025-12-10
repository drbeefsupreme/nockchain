//! Persistent Memory Arena (PMA)
//!
//! The PMA provides a file-backed memory region for long-lived Nouns.
//! After each event completes, surviving Nouns are evacuated from the
//! ephemeral NockStack to the PMA. The NockStack is then cleared.
//!
//! Key invariants:
//! - During computation: All newly allocated Nouns are in stack-pointer form (NockStack)
//! - After event completion: All surviving Nouns are in PMA-offset form; NockStack is empty
//! - At boot: Load PMA from disk; all Nouns start in PMA-offset form
//! - Offset form = PMA: There is no "NockStack offset form" — offset always means PMA

use std::cell::Cell as ThreadCell;
use std::path::{Path, PathBuf};
use std::ptr::{self, copy_nonoverlapping};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use either::Either;
use thiserror::Error;

use crate::mem::{word_size_of, Arena, NockStack};
use crate::noun::{Allocated, Cell, CellMemory, IndirectAtom, Noun, CELL_MASK, INDIRECT_MASK};
use crate::{assert_acyclic, assert_no_forwarding_pointers};

/// Error type for PMA operations
#[derive(Debug, Error)]
pub enum PmaError {
    #[error("PMA is full, cannot allocate {requested} words (available: {available})")]
    OutOfMemory { requested: usize, available: usize },
    #[error("PMA not installed in thread-local storage")]
    NotInstalled,
    #[error("Failed to create arena: {0}")]
    ArenaError(#[from] crate::mem::NewStackError),
}

/// Persistent Memory Arena
///
/// A file-backed memory region for storing Nouns that survive across events.
/// Uses bump allocation for simplicity.
///
/// Internally wraps an `Arena` so that offset-form nouns can be resolved
/// using the standard `Arena::with_current` mechanism.
#[derive(Debug)]
pub struct Pma {
    /// The underlying arena for memory management and thread-local resolution
    arena: Arc<Arena>,
    /// Current allocation offset in words (bump pointer)
    alloc_offset: AtomicU32,
    /// Path to the backing file (for future file-backed persistence)
    path: PathBuf,
}

// Pma needs to be Send + Sync for Arc<Pma>
unsafe impl Send for Pma {}
unsafe impl Sync for Pma {}

thread_local! {
    static CURRENT_PMA: ThreadCell<*const Pma> = ThreadCell::new(ptr::null());
}

impl Pma {
    /// Create a new PMA backed by an anonymous memory mapping (for testing).
    /// This is similar to how NockStack creates its arena.
    pub fn new_anonymous(size_words: usize) -> Result<Arc<Self>, PmaError> {
        let arena = Arena::allocate(size_words)?;

        Ok(Arc::new(Self {
            arena,
            alloc_offset: AtomicU32::new(0),
            path: PathBuf::from("<anonymous>"),
        }))
    }

    /// Create or open a PMA at the given path.
    ///
    /// If the file exists, it is opened and the existing contents are preserved.
    /// If the file does not exist, it is created with the given size.
    ///
    /// Note: For this initial implementation, we use an anonymous arena.
    /// True file-backed persistence will be added in a future phase.
    pub fn open(path: impl AsRef<Path>, size_words: usize) -> Result<Arc<Self>, PmaError> {
        let path = path.as_ref().to_path_buf();
        let arena = Arena::allocate(size_words)?;

        Ok(Arc::new(Self {
            arena,
            alloc_offset: AtomicU32::new(0),
            path,
        }))
    }

    /// Get the underlying arena (for thread-local installation)
    #[inline]
    pub fn arena(&self) -> &Arc<Arena> {
        &self.arena
    }

    /// Get the total size in words
    #[inline]
    pub fn size_words(&self) -> usize {
        self.arena.words()
    }

    /// Get the current allocation offset in words
    #[inline]
    pub fn alloc_offset(&self) -> u32 {
        self.alloc_offset.load(Ordering::Relaxed)
    }

    /// Get remaining free space in words
    #[inline]
    pub fn free_words(&self) -> usize {
        self.size_words() - self.alloc_offset() as usize
    }

    /// Get the base pointer
    #[inline]
    pub fn base_ptr(&self) -> *mut u8 {
        self.arena.base_ptr()
    }

    /// Convert a word offset to a pointer
    #[inline]
    pub fn ptr_from_offset(&self, offset_words: u32) -> *mut u8 {
        self.arena.ptr_from_offset(offset_words)
    }

    /// Convert a pointer to a word offset
    #[inline]
    pub fn offset_from_ptr(&self, ptr: *const u8) -> u32 {
        self.arena.offset_from_ptr(ptr)
    }

    /// Allocate `words` in the PMA, returns word offset.
    /// Returns error if out of memory.
    pub fn alloc(&self, words: usize) -> Result<u32, PmaError> {
        // Atomic bump allocation
        let old_offset = self.alloc_offset.fetch_add(words as u32, Ordering::Relaxed);
        let new_offset = old_offset + words as u32;

        if new_offset as usize > self.size_words() {
            // Rollback the allocation
            self.alloc_offset.fetch_sub(words as u32, Ordering::Relaxed);
            return Err(PmaError::OutOfMemory {
                requested: words,
                available: self.size_words() - old_offset as usize,
            });
        }

        Ok(old_offset)
    }

    /// Allocate `words` in the PMA, returns pointer.
    /// Returns error if out of memory.
    pub fn alloc_ptr(&self, words: usize) -> Result<*mut u64, PmaError> {
        let offset = self.alloc(words)?;
        Ok(self.ptr_from_offset(offset) as *mut u64)
    }

    /// Reset allocation pointer (for testing or re-initialization)
    pub fn reset(&self) {
        self.alloc_offset.store(0, Ordering::Relaxed);
    }

    /// Reset allocation pointer to a specific offset
    pub fn reset_to(&self, offset: u32) {
        debug_assert!(
            (offset as usize) <= self.size_words(),
            "reset offset {} exceeds size {}",
            offset,
            self.size_words()
        );
        self.alloc_offset.store(offset, Ordering::Relaxed);
    }

    /// Check if a pointer is within this PMA
    #[inline]
    pub fn contains_ptr(&self, ptr: *const u8) -> bool {
        let base = self.base_ptr() as usize;
        let end = base + (self.size_words() << 3);
        let ptr_usize = ptr as usize;
        ptr_usize >= base && ptr_usize < end
    }

    /// Install this PMA's arena as the thread-local Arena for offset resolution.
    /// This allows offset-form nouns to be resolved correctly.
    pub fn install_arena(&self) {
        Arena::set_thread_local(&self.arena);
    }

    /// Install this PMA as the thread-local PMA for offset resolution
    pub fn set_thread_local(pma: &Arc<Pma>) {
        let ptr = Arc::as_ptr(pma);
        CURRENT_PMA.with(|cell| cell.set(ptr));
    }

    /// Clear the thread-local PMA
    pub fn clear_thread_local() {
        CURRENT_PMA.with(|cell| cell.set(ptr::null()));
    }

    /// Check if a PMA is installed
    pub fn is_installed() -> bool {
        CURRENT_PMA.with(|cell| !cell.get().is_null())
    }

    /// Execute a closure with the current thread-local PMA
    pub fn with_current<F, R>(f: F) -> R
    where
        F: FnOnce(&Pma) -> R,
    {
        CURRENT_PMA.with(|cell| {
            let ptr = cell.get();
            if ptr.is_null() {
                panic!("Pma::with_current called without an installed Pma");
            }
            unsafe { f(&*ptr) }
        })
    }

    /// Try to execute a closure with the current thread-local PMA
    pub fn try_with_current<F, R>(f: F) -> Option<R>
    where
        F: FnOnce(&Pma) -> R,
    {
        CURRENT_PMA.with(|cell| {
            let ptr = cell.get();
            if ptr.is_null() {
                None
            } else {
                Some(unsafe { f(&*ptr) })
            }
        })
    }

    /// Get the path to the backing file
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// RAII guard for installing a PMA in thread-local storage
pub struct PmaInstallGuard {
    _private: (),
}

impl PmaInstallGuard {
    pub fn new(pma: &Arc<Pma>) -> Self {
        Pma::set_thread_local(pma);
        Self { _private: () }
    }
}

impl Drop for PmaInstallGuard {
    fn drop(&mut self) {
        Pma::clear_thread_local();
    }
}

/// Utility function to get the raw memory size of an IndirectAtom
fn indirect_raw_size(atom: IndirectAtom, pma: &Pma) -> usize {
    debug_assert!(atom.size_with_pma(pma) > 0);
    atom.size_with_pma(pma) + 2
}

/// Extension trait for IndirectAtom to work with PMA
pub trait IndirectAtomPmaExt {
    fn size_with_pma(&self, pma: &Pma) -> usize;
    fn to_raw_pointer_with_pma(&self, pma: &Pma) -> *const u64;
}

impl IndirectAtomPmaExt for IndirectAtom {
    fn size_with_pma(&self, pma: &Pma) -> usize {
        unsafe { *(self.to_raw_pointer_with_pma(pma).add(1)) as usize }
    }

    fn to_raw_pointer_with_pma(&self, pma: &Pma) -> *const u64 {
        use crate::noun::{TaggedPtr, LOCATION_BIT};
        let raw = unsafe { std::mem::transmute::<IndirectAtom, u64>(*self) };
        let tagged = TaggedPtr::from_raw(raw);

        // Check if this is offset form
        if raw & LOCATION_BIT != 0 {
            // Offset form - resolve via PMA
            pma.ptr_from_offset(tagged.payload(INDIRECT_MASK) as u32) as *const u64
        } else {
            // Stack pointer form - direct pointer
            (tagged.payload(INDIRECT_MASK) << 3) as *const u64
        }
    }
}

/// Extension trait for Cell to work with PMA
pub trait CellPmaExt {
    fn to_raw_pointer_with_pma(&self, pma: &Pma) -> *const CellMemory;
    fn head_with_pma(&self, pma: &Pma) -> Noun;
    fn tail_with_pma(&self, pma: &Pma) -> Noun;
}

impl CellPmaExt for Cell {
    fn to_raw_pointer_with_pma(&self, pma: &Pma) -> *const CellMemory {
        use crate::noun::{TaggedPtr, LOCATION_BIT};
        let raw = unsafe { std::mem::transmute::<Cell, u64>(*self) };
        let tagged = TaggedPtr::from_raw(raw);

        // Check if this is offset form
        if raw & LOCATION_BIT != 0 {
            // Offset form - resolve via PMA
            pma.ptr_from_offset(tagged.payload(CELL_MASK) as u32) as *const CellMemory
        } else {
            // Stack pointer form - direct pointer
            (tagged.payload(CELL_MASK) << 3) as *const CellMemory
        }
    }

    fn head_with_pma(&self, pma: &Pma) -> Noun {
        unsafe { (*self.to_raw_pointer_with_pma(pma)).head }
    }

    fn tail_with_pma(&self, pma: &Pma) -> Noun {
        unsafe { (*self.to_raw_pointer_with_pma(pma)).tail }
    }
}

/// Extension trait for Allocated to work with PMA
pub trait AllocatedPmaExt {
    fn to_raw_pointer_with_pma(&self, pma: &Pma) -> *const u64;
    fn is_pma_offset(&self) -> bool;
    fn forwarding_pointer_with_pma(&self, pma: &Pma) -> Option<Allocated>;
}

impl AllocatedPmaExt for Allocated {
    fn to_raw_pointer_with_pma(&self, pma: &Pma) -> *const u64 {
        use crate::noun::{TaggedPtr, LOCATION_BIT};
        let raw = unsafe { std::mem::transmute::<Allocated, u64>(*self) };
        let tagged = TaggedPtr::from_raw(raw);
        let mask = if self.is_indirect() {
            INDIRECT_MASK
        } else {
            CELL_MASK
        };

        // Check if this is offset form
        if raw & LOCATION_BIT != 0 {
            // Offset form - resolve via PMA
            pma.ptr_from_offset(tagged.payload(mask) as u32) as *const u64
        } else {
            // Stack pointer form - direct pointer
            (tagged.payload(mask) << 3) as *const u64
        }
    }

    fn is_pma_offset(&self) -> bool {
        use crate::noun::LOCATION_BIT;
        let raw = unsafe { std::mem::transmute::<Allocated, u64>(*self) };
        raw & LOCATION_BIT != 0
    }

    /// Get forwarding pointer if one exists.
    /// The forwarding pointer points to the NEW location (in PMA).
    /// We return it in offset form since that's what we want during evacuation.
    fn forwarding_pointer_with_pma(&self, pma: &Pma) -> Option<Allocated> {
        use crate::noun::FORWARDING_MASK;
        match self.as_either() {
            Either::Left(indirect) => {
                let size_raw = unsafe { *indirect.to_raw_pointer_with_pma(pma).add(1) };
                if size_raw & FORWARDING_MASK == crate::noun::FORWARDING_TAG {
                    use crate::noun::TaggedPtr;
                    // The forwarding pointer contains a stack pointer to the NEW location (in PMA memory)
                    let ptr = TaggedPtr::from_raw(size_raw).payload(FORWARDING_MASK);
                    let ptr = (ptr << 3) as *const u8;
                    // Convert to offset form since it's in PMA memory
                    let offset = pma.offset_from_ptr(ptr);
                    Some(IndirectAtom::from_offset_words(offset).as_allocated())
                } else {
                    None
                }
            }
            Either::Right(cell) => {
                let head_raw = unsafe { (*cell.to_raw_pointer_with_pma(pma)).head.raw };
                if head_raw & FORWARDING_MASK == crate::noun::FORWARDING_TAG {
                    use crate::noun::TaggedPtr;
                    // The forwarding pointer contains a stack pointer to the NEW location (in PMA memory)
                    let ptr = TaggedPtr::from_raw(head_raw).payload(FORWARDING_MASK);
                    let ptr = (ptr << 3) as *const u8;
                    // Convert to offset form since it's in PMA memory
                    let offset = pma.offset_from_ptr(ptr);
                    Some(Cell::from_offset_words(offset).as_allocated())
                } else {
                    None
                }
            }
        }
    }
}

/// Evacuate a noun from the NockStack to the PMA.
///
/// This function copies all allocated nouns reachable from `noun` that are
/// in stack-pointer form to the PMA, converting them to offset form.
/// Nouns already in offset form (already in PMA) are left as-is.
///
/// After this function returns, `noun` will point to the PMA copy.
///
/// # Safety
/// - The NockStack must have a valid frame
/// - The PMA must be installed as thread-local
/// - The noun must not contain forwarding pointers
pub unsafe fn evacuate_noun_to_pma(
    stack: &NockStack,
    pma: &Pma,
    noun: &mut Noun,
) -> Result<(), PmaError> {
    assert_acyclic!(*noun);
    assert_no_forwarding_pointers!(*noun);

    // Skip direct atoms - nothing to evacuate
    let root_allocated = match noun.as_either_direct_allocated() {
        Either::Left(_direct) => return Ok(()),
        Either::Right(allocated) => allocated,
    };

    // If already in PMA (offset form), nothing to do
    if root_allocated.is_pma_offset() {
        return Ok(());
    }

    // Check for forwarding pointer (already evacuated via another reference)
    if let Some(forwarded) = root_allocated.forwarding_pointer_with_pma(pma) {
        *noun = forwarded.as_noun();
        return Ok(());
    }

    // Not in current frame? Must be in PMA already (shouldn't happen with our invariants)
    if !stack.is_in_frame(root_allocated.to_raw_pointer_with_pma(pma)) {
        return Ok(());
    }

    // Worklist: (source noun, destination pointer)
    let mut work: Vec<(Noun, *mut Noun)> = Vec::with_capacity(32);
    work.push((*noun, noun as *mut Noun));

    while let Some((value, dest_ptr)) = work.pop() {
        match value.as_either_direct_allocated() {
            Either::Left(_direct) => {
                // Direct atoms are copied as-is
                *dest_ptr = value;
            }
            Either::Right(allocated) => {
                // Check for forwarding pointer
                if let Some(forwarded) = allocated.forwarding_pointer_with_pma(pma) {
                    *dest_ptr = forwarded.as_noun();
                    continue;
                }

                // Already in PMA?
                if allocated.is_pma_offset() {
                    *dest_ptr = value;
                    continue;
                }

                // Not in current frame? (already preserved elsewhere)
                if !stack.is_in_frame(allocated.to_raw_pointer_with_pma(pma)) {
                    *dest_ptr = value;
                    continue;
                }

                match allocated.as_either() {
                    Either::Left(mut indirect) => {
                        let size = indirect_raw_size(indirect, pma);

                        // Allocate in PMA
                        let pma_ptr = pma.alloc_ptr(size)?;

                        // Copy data (metadata + size + data words)
                        let src_ptr = indirect.to_raw_pointer_with_pma(pma);
                        copy_nonoverlapping(src_ptr, pma_ptr, size);

                        // Set forwarding pointer in source
                        indirect.set_forwarding_pointer(pma_ptr);

                        // Compute offset and create PMA-offset form noun
                        let offset = pma.offset_from_ptr(pma_ptr as *const u8);
                        *dest_ptr = IndirectAtom::from_offset_words(offset).as_noun();
                    }
                    Either::Right(mut cell) => {
                        // Allocate in PMA
                        let pma_ptr = pma.alloc_ptr(word_size_of::<CellMemory>())?;
                        let pma_cell = pma_ptr as *mut CellMemory;

                        // Copy metadata
                        let src_cell = cell.to_raw_pointer_with_pma(pma);
                        (*pma_cell).metadata = (*src_cell).metadata;

                        // Get head and tail before setting forwarding pointer
                        let head = cell.head_with_pma(pma);
                        let tail = cell.tail_with_pma(pma);

                        // Set forwarding pointer in source
                        cell.set_forwarding_pointer(pma_cell);

                        // Queue head and tail for processing
                        // Note: we write to the PMA cell's head/tail slots
                        work.push((tail, &mut (*pma_cell).tail));
                        work.push((head, &mut (*pma_cell).head));

                        // Compute offset and create PMA-offset form noun
                        let offset = pma.offset_from_ptr(pma_ptr as *const u8);
                        *dest_ptr = Cell::from_offset_words(offset).as_noun();
                    }
                }
            }
        }
    }

    assert_acyclic!(*noun);
    assert_no_forwarding_pointers!(*noun);

    Ok(())
}

/// Evacuate multiple nouns to the PMA.
///
/// This is more efficient than calling `evacuate_noun_to_pma` multiple times
/// because forwarding pointers are preserved across all evacuations, enabling
/// proper sharing.
///
/// # Safety
/// Same as `evacuate_noun_to_pma`
pub unsafe fn evacuate_nouns_to_pma(
    stack: &NockStack,
    pma: &Pma,
    nouns: &mut [&mut Noun],
) -> Result<(), PmaError> {
    for noun in nouns.iter_mut() {
        evacuate_noun_to_pma(stack, pma, *noun)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::NockStack;
    use crate::noun::{Atom, Cell, D, T, DIRECT_MAX};

    const TEST_PMA_SIZE: usize = 1 << 20; // 1M words = 8MB
    const TEST_STACK_SIZE: usize = 1 << 16; // 64K words

    fn make_test_stack() -> NockStack {
        NockStack::new(TEST_STACK_SIZE, 0)
    }

    fn make_test_pma() -> Arc<Pma> {
        Pma::new_anonymous(TEST_PMA_SIZE).expect("failed to create test PMA")
    }

    struct TestContext {
        stack: NockStack,
        pma: Arc<Pma>,
    }

    impl TestContext {
        fn new() -> Self {
            let stack = make_test_stack();
            let pma = make_test_pma();

            // Install stack's arena for allocation during tests
            stack.install_arena();
            Pma::set_thread_local(&pma);

            Self { stack, pma }
        }

        /// Switch to PMA arena for reading evacuated nouns
        fn install_pma_arena(&self) {
            self.pma.install_arena();
        }

        /// Switch back to stack arena for allocation
        fn install_stack_arena(&self) {
            self.stack.install_arena();
        }
    }

    impl Drop for TestContext {
        fn drop(&mut self) {
            Pma::clear_thread_local();
            Arena::clear_thread_local();
        }
    }

    // Helper to check if a noun subtree contains any stack-allocated pointers
    // Note: This requires the appropriate arena to be installed for the nouns being checked
    fn contains_stack_allocated(noun: Noun) -> bool {
        let mut work = vec![noun];
        while let Some(n) = work.pop() {
            if n.is_stack_allocated() {
                return true;
            }
            if let Ok(cell) = n.as_cell() {
                work.push(cell.head());
                work.push(cell.tail());
            }
        }
        false
    }

    // Helper to verify all allocated nouns are in offset form
    // Note: This requires the appropriate arena to be installed for the nouns being checked
    fn all_offset_form(noun: Noun) -> bool {
        !contains_stack_allocated(noun)
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_allocation() {
        let pma = make_test_pma();

        // Allocate some words
        let offset1 = pma.alloc(10).expect("alloc failed");
        assert_eq!(offset1, 0);

        let offset2 = pma.alloc(20).expect("alloc failed");
        assert_eq!(offset2, 10);

        let offset3 = pma.alloc(5).expect("alloc failed");
        assert_eq!(offset3, 30);

        // Check free space
        assert_eq!(pma.alloc_offset(), 35);
        assert_eq!(pma.free_words(), TEST_PMA_SIZE - 35);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_offset_round_trip() {
        let pma = make_test_pma();

        // Allocate and get pointer
        let offset = pma.alloc(10).expect("alloc failed");
        let ptr = pma.ptr_from_offset(offset);

        // Round-trip
        let offset2 = pma.offset_from_ptr(ptr);
        assert_eq!(offset, offset2);

        let ptr2 = pma.ptr_from_offset(offset2);
        assert_eq!(ptr, ptr2);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_reset() {
        let pma = make_test_pma();

        pma.alloc(100).expect("alloc failed");
        assert_eq!(pma.alloc_offset(), 100);

        pma.reset();
        assert_eq!(pma.alloc_offset(), 0);

        pma.alloc(50).expect("alloc failed");
        assert_eq!(pma.alloc_offset(), 50);

        pma.reset_to(25);
        assert_eq!(pma.alloc_offset(), 25);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_thread_local() {
        let pma = make_test_pma();

        assert!(!Pma::is_installed());

        Pma::set_thread_local(&pma);
        assert!(Pma::is_installed());

        Pma::with_current(|p| {
            assert_eq!(p.size_words(), TEST_PMA_SIZE);
        });

        Pma::clear_thread_local();
        assert!(!Pma::is_installed());
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_install_guard() {
        let pma = make_test_pma();

        assert!(!Pma::is_installed());

        {
            let _guard = PmaInstallGuard::new(&pma);
            assert!(Pma::is_installed());
        }

        assert!(!Pma::is_installed());
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_direct_atom() {
        let ctx = TestContext::new();
        let mut noun = D(42);

        unsafe {
            evacuate_noun_to_pma(&ctx.stack, &ctx.pma, &mut noun).expect("evacuation failed");
        }

        // Direct atoms are unchanged
        assert!(noun.is_direct());
        assert_eq!(noun.as_direct().unwrap().data(), 42);
        assert_eq!(ctx.pma.alloc_offset(), 0); // Nothing allocated
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_indirect_atom() {
        let mut ctx = TestContext::new();
        ctx.stack.frame_push(0);

        // Create an indirect atom (larger than DIRECT_MAX)
        let mut noun = Atom::new(&mut ctx.stack, DIRECT_MAX + 1).as_noun();

        // Verify it's stack-allocated before evacuation
        assert!(noun.is_stack_allocated(), "should be stack-allocated before");

        unsafe {
            evacuate_noun_to_pma(&ctx.stack, &ctx.pma, &mut noun).expect("evacuation failed");
        }

        // Verify it's now in offset form
        assert!(
            !noun.is_stack_allocated(),
            "should be offset form after evacuation"
        );

        // Switch to PMA arena to read back the value
        ctx.install_pma_arena();

        // Verify we can still read the value
        let atom = noun.as_atom().expect("should be atom");
        assert_eq!(atom.as_u64().expect("should fit in u64"), DIRECT_MAX + 1);

        // Verify PMA has allocation
        assert!(ctx.pma.alloc_offset() > 0, "PMA should have allocation");
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_simple_cell() {
        let mut ctx = TestContext::new();
        ctx.stack.frame_push(0);

        // Create a simple cell [5 7]
        let mut noun = Cell::new(&mut ctx.stack, D(5), D(7)).as_noun();

        assert!(noun.is_stack_allocated(), "should be stack-allocated before");

        unsafe {
            evacuate_noun_to_pma(&ctx.stack, &ctx.pma, &mut noun).expect("evacuation failed");
        }

        assert!(
            !noun.is_stack_allocated(),
            "should be offset form after evacuation"
        );

        // Switch to PMA arena to read back the structure
        ctx.install_pma_arena();

        // Verify structure
        let cell = noun.as_cell().expect("should be cell");
        assert_eq!(cell.head().as_direct().unwrap().data(), 5);
        assert_eq!(cell.tail().as_direct().unwrap().data(), 7);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_nested_cells() {
        let mut ctx = TestContext::new();
        ctx.stack.frame_push(0);

        // Create nested structure [[1 2] [3 4]]
        let left = T(&mut ctx.stack, &[D(1), D(2)]);
        let right = T(&mut ctx.stack, &[D(3), D(4)]);
        let mut noun = Cell::new(&mut ctx.stack, left, right).as_noun();

        assert!(
            contains_stack_allocated(noun),
            "should contain stack pointers before"
        );

        unsafe {
            evacuate_noun_to_pma(&ctx.stack, &ctx.pma, &mut noun).expect("evacuation failed");
        }

        // Switch to PMA arena to read back the structure
        ctx.install_pma_arena();

        assert!(all_offset_form(noun), "all should be offset form after");

        // Verify structure
        let cell = noun.as_cell().expect("cell");
        let left_cell = cell.head().as_cell().expect("left cell");
        let right_cell = cell.tail().as_cell().expect("right cell");

        assert_eq!(left_cell.head().as_direct().unwrap().data(), 1);
        assert_eq!(left_cell.tail().as_direct().unwrap().data(), 2);
        assert_eq!(right_cell.head().as_direct().unwrap().data(), 3);
        assert_eq!(right_cell.tail().as_direct().unwrap().data(), 4);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_with_indirect_atoms() {
        let mut ctx = TestContext::new();
        ctx.stack.frame_push(0);

        // Create structure with indirect atoms
        let big1 = Atom::new(&mut ctx.stack, DIRECT_MAX + 100).as_noun();
        let big2 = Atom::new(&mut ctx.stack, DIRECT_MAX + 200).as_noun();
        let mut noun = Cell::new(&mut ctx.stack, big1, big2).as_noun();

        assert!(
            contains_stack_allocated(noun),
            "should contain stack pointers before"
        );

        unsafe {
            evacuate_noun_to_pma(&ctx.stack, &ctx.pma, &mut noun).expect("evacuation failed");
        }

        // Switch to PMA arena to read back the structure
        ctx.install_pma_arena();

        assert!(all_offset_form(noun), "all should be offset form after");

        // Verify values
        let cell = noun.as_cell().expect("cell");
        let atom1 = cell.head().as_atom().expect("atom1");
        let atom2 = cell.tail().as_atom().expect("atom2");

        assert_eq!(atom1.as_u64().unwrap(), DIRECT_MAX + 100);
        assert_eq!(atom2.as_u64().unwrap(), DIRECT_MAX + 200);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_shared_structure() {
        let mut ctx = TestContext::new();
        ctx.stack.frame_push(0);

        // Create shared structure: let x = [1 2], create [x x]
        let shared = T(&mut ctx.stack, &[D(1), D(2)]);
        let mut noun = Cell::new(&mut ctx.stack, shared, shared).as_noun();

        let initial_alloc = ctx.pma.alloc_offset();

        unsafe {
            evacuate_noun_to_pma(&ctx.stack, &ctx.pma, &mut noun).expect("evacuation failed");
        }

        let final_alloc = ctx.pma.alloc_offset();

        // Switch to PMA arena to read back the structure
        ctx.install_pma_arena();

        // Verify structure
        let cell = noun.as_cell().expect("cell");

        // The head and tail should point to the same memory (sharing preserved)
        // We can check this by verifying the raw values are equal
        unsafe {
            assert_eq!(
                cell.head().as_raw(),
                cell.tail().as_raw(),
                "sharing should be preserved"
            );
        }

        // And verify we only allocated one copy of the shared cell
        // A CellMemory is 3 words (metadata + head + tail)
        // Plus the outer cell is another 3 words
        // Total: 6 words (without sharing it would be 9 words)
        let allocated = final_alloc - initial_alloc;
        assert!(
            allocated <= 6,
            "should preserve sharing, allocated {} words",
            allocated
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_multiple_nouns() {
        let mut ctx = TestContext::new();
        ctx.stack.frame_push(0);

        let mut noun1 = T(&mut ctx.stack, &[D(1), D(2)]);
        let mut noun2 = T(&mut ctx.stack, &[D(3), D(4)]);

        // Create a shared piece
        let shared = T(&mut ctx.stack, &[D(5), D(6)]);
        let mut noun3 = Cell::new(&mut ctx.stack, shared, D(7)).as_noun();
        let mut noun4 = Cell::new(&mut ctx.stack, shared, D(8)).as_noun();

        unsafe {
            evacuate_nouns_to_pma(
                &ctx.stack,
                &ctx.pma,
                &mut [&mut noun1, &mut noun2, &mut noun3, &mut noun4],
            )
            .expect("evacuation failed");
        }

        // Switch to PMA arena to read back the structures
        ctx.install_pma_arena();

        assert!(all_offset_form(noun1));
        assert!(all_offset_form(noun2));
        assert!(all_offset_form(noun3));
        assert!(all_offset_form(noun4));

        // Verify noun3 and noun4 share their head
        let cell3 = noun3.as_cell().expect("cell3");
        let cell4 = noun4.as_cell().expect("cell4");
        unsafe {
            assert_eq!(
                cell3.head().as_raw(),
                cell4.head().as_raw(),
                "sharing should be preserved across nouns"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_already_evacuated() {
        let mut ctx = TestContext::new();
        ctx.stack.frame_push(0);

        let mut noun = T(&mut ctx.stack, &[D(1), D(2)]);

        // First evacuation
        unsafe {
            evacuate_noun_to_pma(&ctx.stack, &ctx.pma, &mut noun).expect("evacuation failed");
        }

        let alloc_after_first = ctx.pma.alloc_offset();
        let raw_after_first = unsafe { noun.as_raw() };

        // Second evacuation should be a no-op
        unsafe {
            evacuate_noun_to_pma(&ctx.stack, &ctx.pma, &mut noun).expect("evacuation failed");
        }

        let alloc_after_second = ctx.pma.alloc_offset();
        let raw_after_second = unsafe { noun.as_raw() };

        assert_eq!(
            alloc_after_first, alloc_after_second,
            "second evacuation should not allocate"
        );
        assert_eq!(
            raw_after_first, raw_after_second,
            "noun should be unchanged"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_deep_tree() {
        let mut ctx = TestContext::new();
        ctx.stack.frame_push(0);

        // Build a deep tree: [[[[[1 2] 3] 4] 5] 6]
        let mut noun = T(&mut ctx.stack, &[D(1), D(2)]);
        for i in 3..=6 {
            noun = Cell::new(&mut ctx.stack, noun, D(i)).as_noun();
        }

        unsafe {
            evacuate_noun_to_pma(&ctx.stack, &ctx.pma, &mut noun).expect("evacuation failed");
        }

        // Switch to PMA arena to read back the structure
        ctx.install_pma_arena();

        assert!(all_offset_form(noun));

        // Verify we can traverse the whole structure
        let mut current = noun;
        for expected in (3..=6).rev() {
            let cell = current.as_cell().expect("cell");
            assert_eq!(cell.tail().as_direct().unwrap().data(), expected);
            current = cell.head();
        }
        let innermost = current.as_cell().expect("innermost");
        assert_eq!(innermost.head().as_direct().unwrap().data(), 1);
        assert_eq!(innermost.tail().as_direct().unwrap().data(), 2);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_contains_ptr() {
        let pma = make_test_pma();

        let offset = pma.alloc(10).expect("alloc failed");
        let ptr = pma.ptr_from_offset(offset);

        assert!(pma.contains_ptr(ptr));
        assert!(pma.contains_ptr(unsafe { ptr.add(9 * 8) })); // Last byte of allocation

        // Pointer outside PMA
        let outside: *const u8 = 0x1000 as *const u8;
        assert!(!pma.contains_ptr(outside));
    }

    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_pma_out_of_memory() {
        // Create a tiny PMA
        let pma = Pma::new_anonymous(100).expect("failed to create tiny PMA");

        // Allocate most of it
        pma.alloc(90).expect("first alloc should succeed");

        // Try to allocate more than available
        let result = pma.alloc(20);
        assert!(matches!(result, Err(PmaError::OutOfMemory { .. })));

        // Verify the failed allocation didn't change the offset
        assert_eq!(pma.alloc_offset(), 90);

        // Small allocation should still work
        pma.alloc(5).expect("small alloc should succeed");
        assert_eq!(pma.alloc_offset(), 95);
    }
}
