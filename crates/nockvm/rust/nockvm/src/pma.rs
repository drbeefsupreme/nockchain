//! Persistent Memory Arena (PMA)
//!
//! The PMA is a file-backed memory region for storing long-lived Nouns.
//! It uses bump allocation and stores nouns in offset form.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::ptr::copy_nonoverlapping;
use std::sync::Arc;
use std::time::Instant;

use either::Either::{Left, Right};
#[cfg(feature = "pma-assert")]
use intmap::IntMap;
use smallvec::SmallVec;
use thiserror::Error;
use tracing::info;

use crate::ext::noun_equality;
use crate::mem::{word_size_of, Arena, NewStackError, NockStack};
use crate::noun::{
    AllocLocation, Atom, Cell, CellMemory, IndirectAtom, Noun, NounAllocator, NounRepr, NounSpace,
};

const PMA_MAGIC: u64 = u64::from_le_bytes(*b"NOCKPMA1");
const PMA_VERSION: u64 = 1;

/// The metadata for the PMA is a trailer or footer because otherwise the base + offset pointer derivations would need
/// to account for the footer size. With this design it's just base pointer + offset.
#[repr(C)]
#[derive(Clone, Copy)]
struct PmaTrailer {
    magic: u64,
    version: u64,
    data_words: u64,
    alloc_offset: u64,
}

const PMA_TRAILER_BYTES: usize = std::mem::size_of::<PmaTrailer>();

impl PmaTrailer {
    fn to_bytes(self) -> [u8; PMA_TRAILER_BYTES] {
        let mut buf = [0u8; PMA_TRAILER_BYTES];
        buf[0..8].copy_from_slice(&self.magic.to_le_bytes());
        buf[8..16].copy_from_slice(&self.version.to_le_bytes());
        buf[16..24].copy_from_slice(&self.data_words.to_le_bytes());
        buf[24..32].copy_from_slice(&self.alloc_offset.to_le_bytes());
        buf
    }

    fn from_bytes(buf: [u8; PMA_TRAILER_BYTES]) -> Self {
        let magic = u64::from_le_bytes(buf[0..8].try_into().expect("magic slice"));
        let version = u64::from_le_bytes(buf[8..16].try_into().expect("version slice"));
        let data_words = u64::from_le_bytes(buf[16..24].try_into().expect("data_words slice"));
        let alloc_offset = u64::from_le_bytes(buf[24..32].try_into().expect("alloc_offset slice"));
        Self {
            magic,
            version,
            data_words,
            alloc_offset,
        }
    }
}

/// Errors that can occur during PMA operations
#[derive(Debug, Error)]
pub enum PmaError {
    #[error("PMA is full, cannot allocate {requested} words (available: {available})")]
    OutOfMemory { requested: usize, available: usize },

    #[error("Failed to create arena: {0}")]
    ArenaError(#[from] NewStackError),

    #[error("PMA metadata IO failed: {0}")]
    MetadataIo(#[from] std::io::Error),

    #[error("Invalid PMA metadata: {0}")]
    InvalidMetadata(String),
}

/// The Persistent Memory Arena
///
/// A bump-allocated memory region for storing nouns in offset form.
/// The PMA is backed by a file and can persist across program restarts.
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
        let arena = Arena::allocate_file(&path, size_words, PMA_TRAILER_BYTES)?;
        let pma = Self {
            arena,
            alloc_offset: 0,
            path,
        };
        pma.persist_metadata();
        Ok(pma)
    }

    /// Open an existing PMA file without truncating it.
    pub fn open(path: PathBuf) -> Result<Self, PmaError> {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;
        let file_len = file.metadata()?.len() as usize;
        if file_len < PMA_TRAILER_BYTES {
            return Err(PmaError::InvalidMetadata(format!(
                "file too small: {file_len} bytes"
            )));
        }
        let data_bytes = file_len - PMA_TRAILER_BYTES;
        if data_bytes % 8 != 0 {
            return Err(PmaError::InvalidMetadata(format!(
                "data region not word-aligned: {data_bytes} bytes"
            )));
        }
        let data_words = data_bytes >> 3;

        let mut trailer_bytes = [0u8; PMA_TRAILER_BYTES];
        file.seek(SeekFrom::End(-(PMA_TRAILER_BYTES as i64)))?;
        file.read_exact(&mut trailer_bytes)?;
        let trailer = PmaTrailer::from_bytes(trailer_bytes);

        if trailer.magic != PMA_MAGIC {
            return Err(PmaError::InvalidMetadata("bad PMA magic".to_string()));
        }
        if trailer.version != PMA_VERSION {
            return Err(PmaError::InvalidMetadata(format!(
                "unsupported PMA version {}",
                trailer.version
            )));
        }
        if trailer.data_words as usize != data_words {
            return Err(PmaError::InvalidMetadata(format!(
                "metadata data_words {} does not match file ({data_words})",
                trailer.data_words
            )));
        }
        if trailer.alloc_offset > trailer.data_words {
            return Err(PmaError::InvalidMetadata(format!(
                "alloc_offset {} exceeds data_words {}",
                trailer.alloc_offset, trailer.data_words
            )));
        }

        let arena = Arena::open_file(&path, data_words)?;
        let pma = Self {
            arena,
            alloc_offset: trailer.alloc_offset as usize,
            path,
        };
        pma.persist_metadata();
        Ok(pma)
    }

    /// Open an existing PMA file at a fixed base address.
    pub fn open_with_base(path: PathBuf, base: u64) -> Result<Self, PmaError> {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)?;
        let file_len = file.metadata()?.len() as usize;
        if file_len < PMA_TRAILER_BYTES {
            return Err(PmaError::InvalidMetadata(format!(
                "file too small: {file_len} bytes"
            )));
        }
        let data_bytes = file_len - PMA_TRAILER_BYTES;
        if data_bytes % 8 != 0 {
            return Err(PmaError::InvalidMetadata(format!(
                "data region not word-aligned: {data_bytes} bytes"
            )));
        }
        let data_words = data_bytes >> 3;

        let mut trailer_bytes = [0u8; PMA_TRAILER_BYTES];
        file.seek(SeekFrom::End(-(PMA_TRAILER_BYTES as i64)))?;
        file.read_exact(&mut trailer_bytes)?;
        let trailer = PmaTrailer::from_bytes(trailer_bytes);

        if trailer.magic != PMA_MAGIC {
            return Err(PmaError::InvalidMetadata("bad PMA magic".to_string()));
        }
        if trailer.version != PMA_VERSION {
            return Err(PmaError::InvalidMetadata(format!(
                "unsupported PMA version {}",
                trailer.version
            )));
        }
        if trailer.data_words as usize != data_words {
            return Err(PmaError::InvalidMetadata(format!(
                "metadata data_words {} does not match file ({data_words})",
                trailer.data_words
            )));
        }
        if trailer.alloc_offset > trailer.data_words {
            return Err(PmaError::InvalidMetadata(format!(
                "alloc_offset {} exceeds data_words {}",
                trailer.alloc_offset, trailer.data_words
            )));
        }

        let base_ptr = base as *mut u8;
        if base_ptr.is_null() {
            return Err(PmaError::InvalidMetadata("null PMA base".to_string()));
        }
        let arena = Arena::open_file_with_base(&path, data_words, base_ptr)?;
        let pma = Self {
            arena,
            alloc_offset: trailer.alloc_offset as usize,
            path,
        };
        pma.persist_metadata();
        Ok(pma)
    }

    /// Get the underlying arena
    pub fn arena(&self) -> &Arc<Arena> {
        &self.arena
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
        self.persist_metadata();
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
        self.persist_metadata();
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
        self.persist_metadata();
        ptr
    }

    pub fn persist_metadata(&self) {
        debug_assert!(
            self.arena.mapped_len_bytes() >= self.arena.len_bytes() + PMA_TRAILER_BYTES,
            "PMA arena mapping is too small for metadata trailer"
        );
        let trailer = PmaTrailer {
            magic: PMA_MAGIC,
            version: PMA_VERSION,
            data_words: self.arena.words() as u64,
            alloc_offset: self.alloc_offset as u64,
        };
        let bytes = trailer.to_bytes();
        let dst = unsafe { self.arena.base_ptr().add(self.arena.len_bytes()) };
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
        }
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
        let space = NounSpace::pma_only(self);
        noun_equality(a.in_space(&space), b.in_space(&space))
    }

    fn noun_space(&self) -> NounSpace {
        NounSpace::pma_only(self)
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
    /// The caller must ensure `stack` and `pma` describe the arenas that own the
    /// nouns being copied; pointer-form nouns are resolved via `NounSpace::new`.
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

    #[cfg(feature = "pma-assert")]
    fn assert_in_pma(&self, pma: &Pma) {
        self.as_noun().assert_in_pma(pma);
    }

    #[cfg(not(feature = "pma-assert"))]
    #[inline(always)]
    fn assert_in_pma(&self, _pma: &Pma) {}
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
    /// - Source nouns will have forwarding pointers set (corrupting the stack data)
    unsafe fn copy_to_pma(&mut self, stack: &NockStack, pma: &mut Pma) {
        if self.is_direct() {
            return;
        }

        let trace_noun = std::env::var_os("NOCK_PMA_TRACE_NOUN").is_some();
        let trace_start = Instant::now();
        let mut last_progress = trace_start;
        let mut steps = 0usize;

        let space = NounSpace::new(stack, &*pma);
        let root_repr = self.repr(&space);
        match root_repr {
            NounRepr::Indirect(AllocLocation::PmaOffset)
            | NounRepr::Cell(AllocLocation::PmaOffset) => {
                self.assert_in_pma(pma);
                return;
            }
            NounRepr::Indirect(AllocLocation::PmaPtr) | NounRepr::Cell(AllocLocation::PmaPtr) => {
                let offset_noun = {
                    let allocated = self.as_allocated().expect("repr said allocated");
                    let ptr = allocated.to_raw_pointer(&space);
                    assert!(
                        pma.contains_ptr(ptr as *const u8),
                        "noun claims PMA pointer but is outside PMA"
                    );
                    let offset = pma.offset_from_ptr(ptr as *const u8);
                    if allocated.is_indirect() {
                        IndirectAtom::from_offset_words(offset).as_noun()
                    } else {
                        Cell::from_offset_words(offset).as_noun()
                    }
                };
                *self = offset_noun;
                self.assert_in_pma(pma);
                return;
            }
            NounRepr::Forwarding(_) => {
                panic!("forwarding-pointer noun encountered during PMA copy");
            }
            _ => {}
        }

        let mut work: SmallVec<[(Noun, *mut Noun); 64]> = SmallVec::new();
        work.push((*self, self as *mut Noun));

        while let Some((noun, dest_ptr)) = work.pop() {
            steps += 1;
            if trace_noun && (steps & 0x3fff == 0) {
                let now = Instant::now();
                if now.duration_since(last_progress).as_millis() >= 2000 {
                    info!(
                        "pma-copy: noun progress: steps={}, elapsed_ms={}",
                        steps,
                        trace_start.elapsed().as_millis()
                    );
                    last_progress = now;
                }
            }
            match noun.as_either_direct_allocated() {
                Left(_direct) => {
                    *dest_ptr = noun;
                }
                Right(allocated) => {
                    let forwarded = allocated.forwarding_pointer(&space);
                    if let Some(forwarded) = forwarded {
                        let offset_noun = {
                            let ptr = forwarded.to_raw_pointer(&space);
                            assert!(
                                pma.contains_ptr(ptr as *const u8),
                                "forwarding pointer escapes PMA"
                            );
                            let offset = pma.offset_from_ptr(ptr as *const u8);
                            if forwarded.is_indirect() {
                                IndirectAtom::from_offset_words(offset).as_noun()
                            } else {
                                Cell::from_offset_words(offset).as_noun()
                            }
                        };
                        *dest_ptr = offset_noun;
                        continue;
                    }

                    let repr = noun.repr(&space);

                    match repr {
                        NounRepr::Indirect(AllocLocation::PmaOffset)
                        | NounRepr::Cell(AllocLocation::PmaOffset) => {
                            noun.assert_in_pma(pma);
                            *dest_ptr = noun;
                            continue;
                        }
                        NounRepr::Indirect(AllocLocation::PmaPtr)
                        | NounRepr::Cell(AllocLocation::PmaPtr) => {
                            let offset_noun = {
                                let ptr = allocated.to_raw_pointer(&space);
                                assert!(
                                    pma.contains_ptr(ptr as *const u8),
                                    "noun claims PMA pointer but is outside PMA"
                                );
                                let offset = pma.offset_from_ptr(ptr as *const u8);
                                if allocated.is_indirect() {
                                    IndirectAtom::from_offset_words(offset).as_noun()
                                } else {
                                    Cell::from_offset_words(offset).as_noun()
                                }
                            };
                            noun.assert_in_pma(pma);
                            *dest_ptr = offset_noun;
                            continue;
                        }
                        NounRepr::Forwarding(_) => {
                            panic!("forwarding-pointer noun encountered during PMA copy");
                        }
                        NounRepr::Direct => {
                            *dest_ptr = noun;
                            continue;
                        }
                        NounRepr::Indirect(AllocLocation::Stack)
                        | NounRepr::Cell(AllocLocation::Stack) => {}
                    }

                    match allocated.as_either() {
                        Left(mut indirect) => {
                            let (raw_size, src_ptr) =
                                { (indirect.raw_size(&space), indirect.to_raw_pointer(&space)) };

                            let pma_ptr = pma.raw_alloc(raw_size);
                            copy_nonoverlapping(src_ptr, pma_ptr, raw_size);

                            indirect.set_forwarding_pointer(pma_ptr, &space);

                            let offset = pma.offset_from_ptr(pma_ptr as *const u8);
                            *dest_ptr = IndirectAtom::from_offset_words(offset).as_noun();
                        }
                        Right(mut cell) => {
                            let (src_cell, head, tail) = {
                                let src_cell = cell.to_raw_pointer(&space);
                                let head = (*src_cell).head;
                                let tail = (*src_cell).tail;
                                (src_cell, head, tail)
                            };

                            let pma_ptr = pma.raw_alloc(word_size_of::<CellMemory>());
                            let pma_cell = pma_ptr as *mut CellMemory;
                            (*pma_cell).metadata = (*src_cell).metadata;

                            cell.set_forwarding_pointer(pma_cell, &space);

                            work.push((tail, &mut (*pma_cell).tail));
                            work.push((head, &mut (*pma_cell).head));

                            let offset = pma.offset_from_ptr(pma_ptr as *const u8);
                            *dest_ptr = Cell::from_offset_words(offset).as_noun();
                        }
                    }
                }
            }
        }

        if trace_noun {
            info!(
                "pma-copy: noun done: steps={}, elapsed_ms={}",
                steps,
                trace_start.elapsed().as_millis()
            );
        }
    }

    /// Assert that this noun and all its substructure is in the PMA.
    ///
    #[cfg(feature = "pma-assert")]
    fn assert_in_pma(&self, pma: &Pma) {
        if self.is_direct() {
            return;
        }

        let space = NounSpace::pma_only(pma);
        let mut seen = IntMap::new();
        let mut work = vec![*self];

        while let Some(noun) = work.pop() {
            if noun.is_direct() {
                continue;
            }

            match noun.repr(&space) {
                NounRepr::Indirect(AllocLocation::Stack) | NounRepr::Cell(AllocLocation::Stack) => {
                    panic!("noun is stack-allocated, not in PMA");
                }
                NounRepr::Forwarding(_) => {
                    panic!("forwarding pointer is not valid PMA state");
                }
                NounRepr::Indirect(_) | NounRepr::Direct => {}
                NounRepr::Cell(_) => {
                    let cell = noun.in_space(&space).as_cell().expect("checked is_cell");
                    let ptr = unsafe { cell.raw_pointer() } as usize as u64;
                    if seen.get(ptr).is_some() {
                        continue;
                    }
                    seen.insert(ptr, ());
                    work.push(cell.head().noun());
                    work.push(cell.tail().noun());
                }
            }
        }
    }

    #[cfg(not(feature = "pma-assert"))]
    #[inline(always)]
    fn assert_in_pma(&self, _pma: &Pma) {}
}

#[cfg(test)]
pub(crate) fn test_pma_path(label: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut path = std::env::temp_dir();
    path.push(format!("nockvm_pma_{label}_{pid}_{id}.mmap"));
    path
}

#[cfg(test)]
mod tests {
    use std::alloc::Layout;

    use ibig::Stack;

    use super::*;
    use crate::hamt::Hamt;
    use crate::jets::cold::NounListMem;
    use crate::mem::{word_size_of, NockStack};
    use crate::noun::{AllocLocation, D, DIRECT_MAX};

    /// Helper to create a test PMA with a given size
    fn test_pma(size_words: usize) -> Pma {
        let path = test_pma_path("pma");
        Pma::new(size_words, path).expect("Failed to create test PMA")
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
        assert_eq!(
            pma.free_words(),
            1000,
            "Initial free_words should equal size"
        );

        // First allocation: alloc_indirect(10) allocates 10 + 2 = 12 words (data + metadata + size)
        let ptr1 = unsafe { pma.alloc_indirect(10) };
        assert!(
            !ptr1.is_null(),
            "First allocation should return non-null pointer"
        );
        assert_eq!(
            pma.alloc_offset(),
            12,
            "After alloc_indirect(10), offset should be 12"
        );
        assert_eq!(
            pma.free_words(),
            988,
            "After alloc_indirect(10), free should be 988"
        );

        // Second allocation: alloc_indirect(20) allocates 20 + 2 = 22 words
        let ptr2 = unsafe { pma.alloc_indirect(20) };
        assert!(
            !ptr2.is_null(),
            "Second allocation should return non-null pointer"
        );
        assert_eq!(
            pma.alloc_offset(),
            34,
            "After second alloc, offset should be 34"
        );
        assert_eq!(
            pma.free_words(),
            966,
            "After second alloc, free should be 966"
        );

        // Third allocation: alloc_cell allocates word_size_of::<CellMemory>() words
        let ptr3 = unsafe { pma.alloc_cell() };
        assert!(
            !ptr3.is_null(),
            "Cell allocation should return non-null pointer"
        );
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
        assert!(
            !ptr4.is_null(),
            "Struct allocation should return non-null pointer"
        );
        let offset_after_struct = offset_after_cell + struct_words;
        assert_eq!(
            pma.alloc_offset(),
            offset_after_struct,
            "After struct alloc, offset should increase by struct size in words"
        );

        // Fifth allocation: alloc_struct with count > 1 (allocate array of 3 NounListMem)
        let ptr5: *mut NounListMem = unsafe { pma.alloc_struct(3) };
        assert!(
            !ptr5.is_null(),
            "Array struct allocation should return non-null pointer"
        );
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
        assert!(
            !ptr6.is_null(),
            "Layout allocation should return non-null pointer"
        );
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

        assert!(ptr2 >= ptr1_end, "ptr2 should start at or after ptr1's end");
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
        assert!(ptr6 >= ptr5_end, "ptr6 should start at or after ptr5's end");
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
        assert!(
            pma.contains_ptr(ptr_at_0),
            "Pointer at offset 0 should be in PMA"
        );

        // Pointer in the middle should be in PMA
        let middle_offset = 500u32;
        let ptr_middle = pma.ptr_from_offset(middle_offset);
        assert!(
            pma.contains_ptr(ptr_middle),
            "Pointer in middle should be in PMA"
        );

        // Last valid byte should be in PMA
        let last_byte = unsafe { base.add(len_bytes - 1) };
        assert!(pma.contains_ptr(last_byte), "Last byte should be in PMA");

        // Pointer just past the end should NOT be in PMA
        let past_end = unsafe { base.add(len_bytes) };
        assert!(
            !pma.contains_ptr(past_end),
            "Pointer past end should not be in PMA"
        );

        // Pointer well past the end should NOT be in PMA
        let way_past_end = unsafe { base.add(len_bytes + 1000) };
        assert!(
            !pma.contains_ptr(way_past_end),
            "Pointer way past end should not be in PMA"
        );

        // Pointer before the base should NOT be in PMA (if base > 0)
        if base as usize > 0 {
            let before_base = unsafe { base.sub(1) };
            assert!(
                !pma.contains_ptr(before_base),
                "Pointer before base should not be in PMA"
            );
        }

        // Null pointer should NOT be in PMA
        assert!(
            !pma.contains_ptr(std::ptr::null()),
            "Null pointer should not be in PMA"
        );

        // Allocated pointer should be in PMA
        let alloc_ptr = unsafe { pma.alloc_indirect(10) };
        assert!(
            pma.contains_ptr(alloc_ptr as *const u8),
            "Allocated pointer should be in PMA"
        );
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
        assert!(
            result.is_err(),
            "alloc_would_oom(101) should panic with 100 free"
        );

        // Allocate some space
        unsafe { pma.alloc_indirect(10) }; // 12 words (10 + 2 for metadata/size)
        assert_eq!(pma.alloc_offset(), 12);
        assert_eq!(pma.free_words(), 88);

        // alloc_would_oom should reflect remaining space
        pma.alloc_would_oom(88); // Should not panic
        let result = catch_unwind(AssertUnwindSafe(|| {
            pma.alloc_would_oom(89);
        }));
        assert!(
            result.is_err(),
            "alloc_would_oom(89) should panic with 88 free"
        );

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
        assert_eq!(
            pma.free_words(),
            1000,
            "reset() should restore all free space"
        );

        // Allocations after reset should start from 0
        let ptr_after_reset = unsafe { pma.alloc_indirect(5) }; // 7 words
        assert_eq!(pma.alloc_offset(), 7);
        let offset_after_reset = pma.offset_from_ptr(ptr_after_reset as *const u8);
        assert_eq!(
            offset_after_reset, 0,
            "First allocation after reset should be at offset 0"
        );

        // Allocate more to create a checkpoint
        unsafe { pma.alloc_indirect(10) }; // 12 more words
        let checkpoint = pma.alloc_offset();
        assert_eq!(checkpoint, 19); // 7 + 12

        // Allocate even more
        unsafe { pma.alloc_indirect(30) }; // 32 more words
        assert_eq!(pma.alloc_offset(), 51); // 19 + 32

        // Reset to checkpoint
        pma.reset_to(checkpoint);
        assert_eq!(
            pma.alloc_offset(),
            19,
            "reset_to() should set offset to checkpoint"
        );
        assert_eq!(
            pma.free_words(),
            981,
            "reset_to() should restore free space from checkpoint"
        );

        // Next allocation should start at the checkpoint
        let ptr_after_reset_to = unsafe { pma.alloc_indirect(3) }; // 5 words
        let offset_after_reset_to = pma.offset_from_ptr(ptr_after_reset_to as *const u8);
        assert_eq!(
            offset_after_reset_to, 19,
            "Allocation after reset_to should start at checkpoint"
        );
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

    #[test]
    fn test_pma_open_restores_alloc_offset() {
        let path = test_pma_path("open_restore");
        {
            let mut pma = Pma::new(1000, path.clone()).expect("Failed to create test PMA");
            unsafe { pma.alloc_indirect(10) };
            unsafe { pma.alloc_cell() };
            assert!(
                pma.alloc_offset() > 0,
                "Expected allocations to advance offset"
            );
        }

        let pma = Pma::open(path).expect("Failed to open PMA");
        assert!(
            pma.alloc_offset() > 0,
            "alloc_offset should be restored on open"
        );
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
            assert!(
                noun.is_direct(),
                "Should still be a direct atom after evacuation"
            );

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
        let space = NounSpace::new(&stack, &pma);

        // Create an indirect atom on the stack (value > DIRECT_MAX requires indirect storage)
        // We'll use a 2-word value to ensure it's indirect
        let data: [u64; 2] = [0xDEADBEEF_CAFEBABE, 0x12345678_9ABCDEF0];
        let indirect = unsafe { IndirectAtom::new_raw(&mut stack, 2, data.as_ptr()) };
        let mut noun = indirect.as_noun();

        // Verify it's an indirect atom on the stack
        assert!(noun.is_indirect(), "Should be an indirect atom");
        assert!(!noun.is_direct(), "Should not be a direct atom");
        assert!(
            matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be stack-allocated before evacuation"
        );

        // Record the initial PMA offset
        let initial_offset = pma.alloc_offset();
        assert_eq!(initial_offset, 0, "PMA should start empty");

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
            !matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be in offset form after evacuation"
        );
        assert!(noun.is_indirect(), "Should still be an indirect atom");

        // Verify data is readable and correct via PMA arena
        let atom = noun.as_atom().expect("Should be an atom");
        let read_indirect = atom.as_indirect().expect("Should be indirect");

        // Read the size - should be 2 words
        let read_handle = read_indirect.as_atom().in_space(&space);
        let size = read_handle.size();
        assert_eq!(size, 2, "Indirect atom should have size 2");

        // Read the data back and verify it matches
        let data_ptr = read_handle.data_pointer();
        let read_data = unsafe { std::slice::from_raw_parts(data_ptr, 2) };
        assert_eq!(read_data[0], data[0], "First data word should match");
        assert_eq!(read_data[1], data[1], "Second data word should match");

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
        let space = NounSpace::new(&stack, &pma);

        // Create a simple cell [42 123] with direct atoms
        let mut noun = Cell::new(&mut stack, D(42), D(123)).as_noun();

        // Verify it's a cell on the stack
        assert!(noun.is_cell(), "Should be a cell");
        assert!(
            matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be stack-allocated before evacuation"
        );

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
        assert!(
            !matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be in offset form after evacuation"
        );
        assert!(noun.is_cell(), "Should still be a cell");

        // Read head and tail
        let cell = noun.in_space(&space).as_cell().expect("Should be a cell");
        let head = cell.head().noun();
        let tail = cell.tail().noun();

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
        let space = NounSpace::new(&stack, &pma);

        // Create nested cells: [[1 2] [3 4]]
        let left = Cell::new(&mut stack, D(1), D(2)).as_noun();
        let right = Cell::new(&mut stack, D(3), D(4)).as_noun();
        let mut noun = Cell::new(&mut stack, left, right).as_noun();

        // Verify structure before evacuation
        assert!(noun.is_cell(), "Root should be a cell");
        assert!(
            matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Root should be stack-allocated"
        );

        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Should allocate 3 cells worth of space
        let cell_words = word_size_of::<CellMemory>();
        assert_eq!(
            pma.alloc_offset(),
            cell_words * 3,
            "Should allocate 3 cells"
        );

        // Verify root is in offset form
        assert!(
            !matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Root should be in offset form"
        );

        // Navigate and verify structure
        let root = noun.in_space(&space).as_cell().expect("root is cell");
        let left_cell = root.head().as_cell().expect("left is cell");
        let right_cell = root.tail().as_cell().expect("right is cell");

        // Verify left cell [1 2]
        assert!(
            !matches!(root.head().allocated_location(), Some(AllocLocation::Stack)),
            "Left should be in offset form"
        );
        assert_eq!(left_cell.head().noun().as_direct().expect("1").data(), 1);
        assert_eq!(left_cell.tail().noun().as_direct().expect("2").data(), 2);

        // Verify right cell [3 4]
        assert!(
            !matches!(root.tail().allocated_location(), Some(AllocLocation::Stack)),
            "Right should be in offset form"
        );
        assert_eq!(right_cell.head().noun().as_direct().expect("3").data(), 3);
        assert_eq!(right_cell.tail().noun().as_direct().expect("4").data(), 4);

        // Verify assert_in_pma passes for entire structure
        noun.assert_in_pma(&pma);
    }

    /// Verifies cells containing indirect atoms have both the cell and atoms correctly evacuated.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_cell_with_indirect_atoms() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);
        let space = NounSpace::new(&stack, &pma);

        // Create indirect atoms
        let data1: [u64; 2] = [0xAAAAAAAA_BBBBBBBB, 0xCCCCCCCC_DDDDDDDD];
        let data2: [u64; 2] = [0x11111111_22222222, 0x33333333_44444444];
        let indirect1 = unsafe { IndirectAtom::new_raw(&mut stack, 2, data1.as_ptr()) };
        let indirect2 = unsafe { IndirectAtom::new_raw(&mut stack, 2, data2.as_ptr()) };

        // Create cell with indirect atoms
        let mut noun = Cell::new(&mut stack, indirect1.as_noun(), indirect2.as_noun()).as_noun();

        assert!(
            matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be stack-allocated"
        );

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
        assert!(
            !matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Root should be in offset form"
        );

        let cell = noun.in_space(&space).as_cell().expect("is cell");
        let head = cell.head().noun();
        let tail = cell.tail().noun();

        // Verify head is indirect atom with correct data
        assert!(head.is_indirect(), "Head should be indirect");
        assert!(
            !matches!(
                head.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Head should be in offset form"
        );
        let head_indirect = head.as_indirect().expect("head indirect");
        let head_handle = head_indirect.as_atom().in_space(&space);
        let head_data = unsafe { std::slice::from_raw_parts(head_handle.data_pointer(), 2) };
        assert_eq!(head_data[0], data1[0]);
        assert_eq!(head_data[1], data1[1]);

        // Verify tail is indirect atom with correct data
        assert!(tail.is_indirect(), "Tail should be indirect");
        assert!(
            !matches!(
                tail.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Tail should be in offset form"
        );
        let tail_indirect = tail.as_indirect().expect("tail indirect");
        let tail_handle = tail_indirect.as_atom().in_space(&space);
        let tail_data = unsafe { std::slice::from_raw_parts(tail_handle.data_pointer(), 2) };
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
        let space = NounSpace::new(&stack, &pma);

        // Create a shared subcell
        let shared = Cell::new(&mut stack, D(1), D(2)).as_noun();

        // Create [shared shared] - both head and tail point to same cell
        let mut noun = Cell::new(&mut stack, shared, shared).as_noun();

        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Should allocate only 2 cells: the root and the shared subcell (not 3!)
        let cell_words = word_size_of::<CellMemory>();
        assert_eq!(
            pma.alloc_offset(),
            cell_words * 2,
            "Should allocate only 2 cells due to sharing"
        );

        // Verify both head and tail point to the same PMA location
        let root = noun.in_space(&space).as_cell().expect("is cell");
        let head_raw = unsafe { root.head().noun().as_raw() };
        let tail_raw = unsafe { root.tail().noun().as_raw() };
        assert_eq!(
            head_raw, tail_raw,
            "Head and tail should point to same location (sharing preserved)"
        );

        // Verify the shared cell is correct
        let shared_cell = root.head().as_cell().expect("shared is cell");
        assert_eq!(shared_cell.head().noun().as_direct().expect("1").data(), 1);
        assert_eq!(shared_cell.tail().noun().as_direct().expect("2").data(), 2);

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
        let space = NounSpace::new(&stack, &pma);

        // Create a deeply nested structure: [1 [2 [3 [4 ... [999 1000]]]]]
        const DEPTH: u64 = 500;

        // Build from the inside out
        let mut noun = D(DEPTH);
        for i in (1..DEPTH).rev() {
            noun = Cell::new(&mut stack, D(i), noun).as_noun();
        }

        // Verify it's deeply nested and stack-allocated
        assert!(noun.is_cell(), "Root should be a cell");
        assert!(
            matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be stack-allocated"
        );

        // Count the depth before evacuation
        let mut depth_before = 0u64;
        let mut current = noun;
        while current.is_cell() {
            depth_before += 1;
            current = current.in_space(&space).as_cell().unwrap().tail().noun();
        }
        assert_eq!(
            depth_before,
            DEPTH - 1,
            "Should have correct depth before evacuation"
        );

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
        assert!(
            !matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Root should be in offset form"
        );

        // Traverse the entire structure and verify values
        let mut current = noun;
        for expected in 1..DEPTH {
            assert!(current.is_cell(), "Should be cell at depth {}", expected);
            let cell = current.in_space(&space).as_cell().expect("is cell");

            // Verify head value
            let head = cell.head().noun();
            assert!(
                head.is_direct(),
                "Head at depth {} should be direct",
                expected
            );
            assert_eq!(
                head.as_direct().expect("direct").data(),
                expected,
                "Head at depth {} should be {}",
                expected,
                expected
            );

            // Verify this cell is in offset form
            assert!(
                !matches!(
                    current.in_space(&space).allocated_location(),
                    Some(AllocLocation::Stack)
                ),
                "Cell at depth {} should be in offset form",
                expected
            );

            current = cell.tail().noun();
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
        let space = NounSpace::new(&stack, &pma);

        const DEPTH: usize = 200;

        // Helper to create an indirect atom with `word_count` words of data
        // Data pattern: first word is the index, remaining words are index + word_position
        let make_indirect = |stack: &mut NockStack, index: usize, word_count: usize| -> Noun {
            let mut data = vec![0u64; word_count];
            for (i, word) in data.iter_mut().enumerate() {
                *word = (index as u64) << 32 | (i as u64);
            }
            unsafe { IndirectAtom::new_raw(stack, word_count, data.as_ptr()).as_noun() }
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
        assert!(
            matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be stack-allocated"
        );

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
        assert!(
            !matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Root should be in offset form"
        );

        // Traverse and verify all values
        let mut current = noun;
        for expected_index in 1..DEPTH {
            assert!(
                current.is_cell(),
                "Should be cell at depth {}",
                expected_index
            );
            let cell = current.in_space(&space).as_cell().expect("is cell");

            // Verify head is an indirect atom with correct data
            let head = cell.head().noun();
            assert!(
                head.is_indirect(),
                "Head at depth {} should be indirect",
                expected_index
            );
            assert!(
                !matches!(
                    head.in_space(&space).allocated_location(),
                    Some(AllocLocation::Stack)
                ),
                "Head at depth {} should be in offset form",
                expected_index
            );

            let head_indirect = head.as_indirect().expect("indirect");
            let head_handle = head_indirect.as_atom().in_space(&space);
            let expected_word_count = word_count_for_index(expected_index);
            assert_eq!(
                head_handle.size(),
                expected_word_count,
                "Indirect atom at depth {} should have {} words",
                expected_index,
                expected_word_count
            );

            // Verify data pattern
            let data_ptr = head_handle.data_pointer();
            for word_idx in 0..expected_word_count {
                let expected_value = (expected_index as u64) << 32 | (word_idx as u64);
                let actual_value = unsafe { *data_ptr.add(word_idx) };
                assert_eq!(
                    actual_value, expected_value,
                    "Data mismatch at depth {}, word {}",
                    expected_index, word_idx
                );
            }

            current = cell.tail().noun();
        }

        // Final element should be indirect atom for index DEPTH
        assert!(current.is_indirect(), "Leaf should be indirect atom");
        assert!(
            !matches!(
                current.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Leaf should be in offset form"
        );

        let leaf_indirect = current.as_indirect().expect("indirect");
        let leaf_handle = leaf_indirect.as_atom().in_space(&space);
        let expected_leaf_words = word_count_for_index(DEPTH);
        assert_eq!(
            leaf_handle.size(),
            expected_leaf_words,
            "Leaf indirect atom should have {} words",
            expected_leaf_words
        );

        // Verify leaf data pattern
        let leaf_data_ptr = leaf_handle.data_pointer();
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
        let mut pma = test_pma(1000);

        let mut noun1 = Cell::new(&mut stack, D(1), D(2)).as_noun();
        let mut noun2 = Cell::new(&mut stack, D(1), D(2)).as_noun();
        let mut noun3 = Cell::new(&mut stack, D(1), D(3)).as_noun();

        unsafe {
            noun1.copy_to_pma(&stack, &mut pma);
            noun2.copy_to_pma(&stack, &mut pma);
            noun3.copy_to_pma(&stack, &mut pma);
        }

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
        let space = NounSpace::new(&stack, &pma);

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
            assert!(
                result.is_some(),
                "Lookup for key {} should succeed before evacuation",
                i
            );
            let value = result.unwrap();
            assert!(value.is_direct(), "Value should be direct atom");
            assert_eq!(
                value.as_direct().unwrap().data(),
                i * 100,
                "Value for key {} should be {}",
                i,
                i * 100
            );
        }

        // Evacuate the HAMT to PMA
        unsafe {
            hamt.copy_to_pma(&stack, &mut pma);
        }

        // Verify entries are still present after evacuation
        let mut found = vec![false; 10];
        for entries in hamt.iter() {
            for (key, value) in entries {
                let key_direct = key.as_direct().expect("key should be direct");
                let value_direct = value.as_direct().expect("value should be direct");
                let idx = key_direct.data() as usize;
                assert!(
                    idx < found.len(),
                    "Key {} should be within expected range",
                    idx
                );
                assert_eq!(
                    value_direct.data(),
                    (idx as u64) * 100,
                    "Value for key {} should still be {} after evacuation",
                    idx,
                    (idx as u64) * 100
                );
                found[idx] = true;
            }
        }
        assert!(
            found.iter().all(|present| *present),
            "All keys should be present after evacuation"
        );

        // Verify internal structure is in PMA (offset form)
        // Iterate over the HAMT and check all nouns are not stack-allocated
        for entries in hamt.iter() {
            for (key, value) in entries {
                if !key.is_direct() {
                    assert!(
                        !matches!(
                            key.in_space(&space).allocated_location(),
                            Some(AllocLocation::Stack)
                        ),
                        "HAMT key should be in offset form after evacuation"
                    );
                }
                if !value.is_direct() {
                    assert!(
                        !matches!(
                            value.in_space(&space).allocated_location(),
                            Some(AllocLocation::Stack)
                        ),
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
        let mut pma = test_pma(10000);
        let space = NounSpace::new(&stack, &pma);

        // Test with indirect atom
        let data: [u64; 2] = [0xDEADBEEF_CAFEBABE, 0x12345678_9ABCDEF0];
        let stack_indirect =
            unsafe { IndirectAtom::new_raw(&mut stack, 2, data.as_ptr()) }.as_noun();

        // Copy to PMA
        let mut pma_indirect = stack_indirect;
        unsafe { pma_indirect.copy_to_pma(&stack, &mut pma) };

        // Verify the PMA copy is in offset form
        assert!(
            !matches!(
                pma_indirect.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "PMA copy should be in offset form"
        );

        // Verify the PMA copy contains correct data
        let pma_ia = pma_indirect.as_indirect().unwrap();
        let pma_handle = pma_ia.as_atom().in_space(&space);
        let pma_size = pma_handle.size();
        assert_eq!(pma_size, 2, "PMA indirect atom should have size 2");

        let pma_bytes = pma_handle.as_ne_bytes();
        assert_eq!(
            pma_bytes.len(),
            16,
            "PMA indirect should have 16 bytes of data"
        );

        // Verify actual data values
        let pma_slice =
            unsafe { std::slice::from_raw_parts(pma_handle.data_pointer(), pma_handle.size()) };
        assert_eq!(pma_slice[0], 0xDEADBEEF_CAFEBABE, "First word should match");
        assert_eq!(
            pma_slice[1], 0x12345678_9ABCDEF0,
            "Second word should match"
        );

        // Test with cell containing direct atoms
        let stack_cell = Cell::new(&mut stack, D(42), D(99)).as_noun();
        let mut pma_cell = stack_cell;
        unsafe { pma_cell.copy_to_pma(&stack, &mut pma) };

        assert!(
            !matches!(
                pma_cell.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "PMA cell should be in offset form"
        );
        let cell = pma_cell.in_space(&space).as_cell().unwrap();
        assert_eq!(
            cell.head().noun().as_direct().unwrap().data(),
            42,
            "Cell head should be 42"
        );
        assert_eq!(
            cell.tail().noun().as_direct().unwrap().data(),
            99,
            "Cell tail should be 99"
        );

        // Test with nested structure
        let inner = Cell::new(&mut stack, D(1), D(2)).as_noun();
        let stack_nested = Cell::new(&mut stack, inner, D(3)).as_noun();
        let mut pma_nested = stack_nested;
        unsafe { pma_nested.copy_to_pma(&stack, &mut pma) };

        assert!(
            !matches!(
                pma_nested.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "PMA nested should be in offset form"
        );
        let outer = pma_nested.in_space(&space).as_cell().unwrap();
        assert_eq!(
            outer.tail().noun().as_direct().unwrap().data(),
            3,
            "Outer tail should be 3"
        );
        let inner_cell = outer.head().as_cell().unwrap();
        assert_eq!(
            inner_cell.head().noun().as_direct().unwrap().data(),
            1,
            "Inner head should be 1"
        );
        assert_eq!(
            inner_cell.tail().noun().as_direct().unwrap().data(),
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
        let mut pma = test_pma(100000);

        // Create a second stack with reference copies of keys/values for comparison
        // This stack won't be corrupted by forwarding pointers
        let mut ref_stack = NockStack::new(1 << 16, 0);
        let space = NounSpace::new(&stack, &pma);
        let ref_space = NounSpace::new(&ref_stack, &pma);

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

        // Verify all values match by comparing PMA nouns to reference stack nouns
        let mut found_count = 0;
        for entries in hamt.iter() {
            for (pma_key, pma_value) in entries {
                // Find matching reference key and verify value matches
                let mut found = false;
                for (idx, ref_key) in ref_keys.iter().enumerate() {
                    if noun_equality(
                        (*pma_key).in_space(&ref_space),
                        (*ref_key).in_space(&ref_space),
                    ) {
                        assert!(
                            noun_equality(
                                (*pma_value).in_space(&ref_space),
                                ref_values[idx].in_space(&ref_space),
                            ),
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
                verify_noun_not_stack_allocated(*key, &space, "HAMT key");
                verify_noun_not_stack_allocated(*value, &space, "HAMT value");
            }
        }

        // Verify the HAMT structure itself is in PMA
        hamt.assert_in_pma(&pma);
    }

    /// Helper to recursively verify a noun is not stack-allocated
    fn verify_noun_not_stack_allocated(noun: Noun, space: &NounSpace, context: &str) {
        if noun.is_direct() {
            return;
        }

        let location = noun.in_space(space).allocated_location();
        assert!(
            !matches!(location, Some(AllocLocation::Stack)),
            "{} should be in offset form after evacuation",
            context
        );

        if let Ok(cell) = noun.in_space(space).as_cell() {
            verify_noun_not_stack_allocated(cell.head().noun(), space, context);
            verify_noun_not_stack_allocated(cell.tail().noun(), space, context);
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

    /// Verifies indirect atoms of various sizes (1, 2, 3, 7, 8, 9 words) are properly
    /// aligned in PMA and readable without alignment faults.
    ///
    /// This test exercises:
    /// - Creating indirect atoms of varying sizes on the NockStack
    /// - Evacuating each to the PMA
    /// - Verifying the PMA pointer is 8-byte aligned (required for u64 access)
    /// - Verifying all data words are readable without alignment faults
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_indirect_atom_alignment() {
        let mut stack = NockStack::new(1 << 14, 0);
        let mut pma = test_pma(10000);
        let space = NounSpace::new(&stack, &pma);

        // Test various sizes including edge cases
        let test_sizes: [usize; 6] = [1, 2, 3, 7, 8, 9];

        for &word_count in &test_sizes {
            // Reset PMA for each test to get fresh allocations
            pma.reset();

            // Create data pattern: each word contains the index
            let data: Vec<u64> = (0..word_count)
                .map(|i| 0xDEAD_0000_0000_0000u64 | i as u64)
                .collect();

            // Create indirect atom on stack
            let indirect = unsafe { IndirectAtom::new_raw(&mut stack, word_count, data.as_ptr()) };
            let mut noun = indirect.as_noun();

            // Verify it's stack-allocated before evacuation
            assert!(
                matches!(
                    noun.in_space(&space).allocated_location(),
                    Some(AllocLocation::Stack)
                ),
                "Size {}: should be stack-allocated before evacuation",
                word_count
            );

            // Evacuate to PMA
            unsafe { noun.copy_to_pma(&stack, &mut pma) };

            // Verify the noun is now in PMA (offset form)
            assert!(
                !matches!(
                    noun.in_space(&space).allocated_location(),
                    Some(AllocLocation::Stack)
                ),
                "Size {}: should be in offset form after evacuation",
                word_count
            );

            // Get the PMA pointer and verify alignment
            let read_indirect = noun.as_indirect().expect("Should be indirect");
            let handle = read_indirect.as_atom().in_space(&space);
            let data_ptr = handle.data_pointer();

            // Verify 8-byte alignment (required for u64 access)
            assert_eq!(
                (data_ptr as usize) % 8,
                0,
                "Size {}: data pointer {:p} should be 8-byte aligned",
                word_count,
                data_ptr
            );

            // Verify size is correct
            assert_eq!(
                handle.size(),
                word_count,
                "Size {}: indirect atom should have correct size",
                word_count
            );

            // Read all data words and verify they match (this would fault on misalignment)
            for i in 0..word_count {
                let read_value = unsafe { *data_ptr.add(i) };
                let expected = 0xDEAD_0000_0000_0000u64 | i as u64;
                assert_eq!(
                    read_value, expected,
                    "Size {}: word {} should match expected value",
                    word_count, i
                );
            }

            // Verify expected allocation size: metadata (1) + size (1) + data (word_count)
            let expected_alloc = word_count + 2;
            assert_eq!(
                pma.alloc_offset(),
                expected_alloc,
                "Size {}: should allocate {} words",
                word_count,
                expected_alloc
            );

            noun.assert_in_pma(&pma);
        }
    }

    /// Verifies CellMemory fields (metadata, head, tail) are at correct offsets after
    /// evacuation by reading each field independently.
    ///
    /// This test exercises:
    /// - CellMemory layout is preserved after evacuation to PMA
    /// - metadata field is at offset 0
    /// - head field is at offset 8 (sizeof u64)
    /// - tail field is at offset 16 (sizeof u64 + sizeof Noun)
    /// - Each field is independently readable with correct values
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_cell_memory_layout() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);
        let space = NounSpace::new(&stack, &pma);

        // Create a cell with known values: [42 123]
        let head_val = 42u64;
        let tail_val = 123u64;
        let mut noun = Cell::new(&mut stack, D(head_val), D(tail_val)).as_noun();

        // Evacuate to PMA
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Get the cell and its raw pointer
        let cell = noun.in_space(&space).as_cell().expect("Should be a cell");
        let cell_ptr = unsafe { cell.raw_pointer() };

        // Verify the pointer is valid and 8-byte aligned
        assert!(!cell_ptr.is_null(), "Cell pointer should not be null");
        assert_eq!(
            (cell_ptr as usize) % 8,
            0,
            "Cell pointer should be 8-byte aligned"
        );

        // Verify CellMemory layout by reading fields at expected offsets
        // CellMemory is #[repr(C)] #[repr(packed(8))]:
        //   offset 0: metadata (u64)
        //   offset 8: head (Noun, which is u64-sized)
        //   offset 16: tail (Noun, which is u64-sized)

        let base_ptr = cell_ptr as *const u8;

        // Read metadata at offset 0
        let metadata_ptr = base_ptr as *const u64;
        let metadata = unsafe { *metadata_ptr };
        // Metadata contains mug cache; we just verify it's readable
        // (value depends on implementation, but should not cause faults)
        let _ = metadata;

        // Read head at offset 8
        let head_ptr = unsafe { base_ptr.add(8) as *const u64 };
        let head_raw = unsafe { *head_ptr };
        let head_noun = unsafe { Noun::from_raw(head_raw) };
        assert!(head_noun.is_direct(), "Head should be a direct atom");
        assert_eq!(
            head_noun.as_direct().expect("head direct").data(),
            head_val,
            "Head value should be {}",
            head_val
        );

        // Read tail at offset 16
        let tail_ptr = unsafe { base_ptr.add(16) as *const u64 };
        let tail_raw = unsafe { *tail_ptr };
        let tail_noun = unsafe { Noun::from_raw(tail_raw) };
        assert!(tail_noun.is_direct(), "Tail should be a direct atom");
        assert_eq!(
            tail_noun.as_direct().expect("tail direct").data(),
            tail_val,
            "Tail value should be {}",
            tail_val
        );

        // Verify total size matches expected CellMemory size (3 words = 24 bytes)
        assert_eq!(
            std::mem::size_of::<CellMemory>(),
            24,
            "CellMemory should be 24 bytes (3 words)"
        );
        assert_eq!(
            word_size_of::<CellMemory>(),
            3,
            "CellMemory should be 3 words"
        );

        // Verify reading through the proper API gives same results
        assert_eq!(
            cell.head().noun().as_direct().expect("head").data(),
            head_val,
            "Head via API should match"
        );
        assert_eq!(
            cell.tail().noun().as_direct().expect("tail").data(),
            tail_val,
            "Tail via API should match"
        );

        noun.assert_in_pma(&pma);
    }

    /// Verifies diamond-shaped DAGs (A→B, A→C, B→D, C→D) preserve all sharing
    /// and D is only copied once.
    ///
    /// Structure:
    /// ```text
    ///       A
    ///      / \
    ///     B   C
    ///      \ /
    ///       D
    /// ```
    ///
    /// After evacuation, both paths A→B→D and A→C→D should point to the
    /// same location in the PMA.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_forwarding_pointer_diamond_sharing() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);
        let space = NounSpace::new(&stack, &pma);

        // Create the shared node D: [100 200]
        let d = Cell::new(&mut stack, D(100), D(200)).as_noun();

        // Create B: [1 D] - left branch containing D
        let b = Cell::new(&mut stack, D(1), d).as_noun();

        // Create C: [2 D] - right branch containing D (same D!)
        let c = Cell::new(&mut stack, D(2), d).as_noun();

        // Create A: [B C] - root containing both branches
        let mut a = Cell::new(&mut stack, b, c).as_noun();

        // Verify D is shared before evacuation (same raw pointer)
        let b_cell = b.in_space(&space).as_cell().expect("B is cell");
        let c_cell = c.in_space(&space).as_cell().expect("C is cell");
        let d_from_b = b_cell.tail().noun();
        let d_from_c = c_cell.tail().noun();
        assert_eq!(
            unsafe { d_from_b.as_raw() },
            unsafe { d_from_c.as_raw() },
            "D should be shared before evacuation"
        );

        // Record initial PMA state
        let initial_offset = pma.alloc_offset();

        // Evacuate A (which includes B, C, and D)
        unsafe { a.copy_to_pma(&stack, &mut pma) };

        // Calculate expected allocation:
        // - A: 1 cell (3 words)
        // - B: 1 cell (3 words)
        // - C: 1 cell (3 words)
        // - D: 1 cell (3 words) - should only be copied ONCE
        // Total: 4 cells = 12 words
        let cell_words = word_size_of::<CellMemory>();
        let expected_alloc = cell_words * 4;
        assert_eq!(
            pma.alloc_offset() - initial_offset,
            expected_alloc,
            "Should allocate exactly 4 cells (D copied only once)"
        );

        // Navigate to D through both paths and verify they point to same location
        let a_cell = a.in_space(&space).as_cell().expect("A is cell");
        let b_after = a_cell.head().as_cell().expect("B is cell");
        let c_after = a_cell.tail().as_cell().expect("C is cell");

        let d_via_b = b_after.tail().noun();
        let d_via_c = c_after.tail().noun();

        // Both paths should yield the same raw noun value (same PMA offset)
        assert_eq!(
            unsafe { d_via_b.as_raw() },
            unsafe { d_via_c.as_raw() },
            "D should be shared after evacuation (same PMA offset)"
        );

        // Verify D's contents are correct
        let d_cell = d_via_b.in_space(&space).as_cell().expect("D is cell");
        assert_eq!(
            d_cell.head().noun().as_direct().expect("100").data(),
            100,
            "D's head should be 100"
        );
        assert_eq!(
            d_cell.tail().noun().as_direct().expect("200").data(),
            200,
            "D's tail should be 200"
        );

        // Verify the entire structure is in PMA
        a.assert_in_pma(&pma);
    }

    /// Verifies a single noun referenced by many (e.g., 100) different cells
    /// is only copied once.
    ///
    /// Structure:
    /// ```text
    ///   [cell_0, cell_1, cell_2, ... cell_99]
    ///      |       |       |           |
    ///      +-------+-------+-----------+
    ///                      |
    ///                   shared
    /// ```
    ///
    /// All 100 cells reference the same shared noun. After evacuation,
    /// the shared noun should only be copied once to the PMA.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_forwarding_pointer_wide_sharing() {
        let mut stack = NockStack::new(1 << 14, 0);
        let mut pma = test_pma(100000);
        let space = NounSpace::new(&stack, &pma);

        const NUM_REFS: usize = 100;

        // Create the shared noun: [999 888]
        let shared = Cell::new(&mut stack, D(999), D(888)).as_noun();

        // Create 100 cells, each referencing the shared noun
        // Each cell is [index shared]
        let mut cells: Vec<Noun> = Vec::with_capacity(NUM_REFS);
        for i in 0..NUM_REFS {
            let cell = Cell::new(&mut stack, D(i as u64), shared).as_noun();
            cells.push(cell);
        }

        // Build a list structure to hold all cells: [cell_0 [cell_1 [cell_2 ... [cell_99 0]]]]
        let mut root = D(0); // nil terminator
        for cell in cells.iter().rev() {
            root = Cell::new(&mut stack, *cell, root).as_noun();
        }

        // Verify all cells reference the same shared noun before evacuation
        let shared_raw = unsafe { shared.as_raw() };
        for (i, cell) in cells.iter().enumerate() {
            let c = cell.in_space(&space).as_cell().expect("is cell");
            let tail_raw = unsafe { c.tail().noun().as_raw() };
            assert_eq!(
                tail_raw, shared_raw,
                "Cell {} should reference shared noun before evacuation",
                i
            );
        }

        // Record initial PMA state
        let initial_offset = pma.alloc_offset();

        // Evacuate the root (which includes all cells and the shared noun)
        unsafe { root.copy_to_pma(&stack, &mut pma) };

        // Calculate expected allocation:
        // - shared: 1 cell (only copied ONCE)
        // - cells[0..99]: 100 cells (each [index shared])
        // - list nodes: 100 cells (the cons cells forming the list)
        // Total: 1 + 100 + 100 = 201 cells
        let cell_words = word_size_of::<CellMemory>();
        let expected_alloc = cell_words * (1 + NUM_REFS + NUM_REFS);
        assert_eq!(
            pma.alloc_offset() - initial_offset,
            expected_alloc,
            "Should allocate exactly {} cells (shared copied only once)",
            1 + NUM_REFS + NUM_REFS
        );

        // Navigate through the list and verify all cells still reference the same shared noun
        let mut current = root;
        let mut first_shared_raw: Option<u64> = None;
        let mut count = 0;

        while current.is_cell() {
            let list_cell = current.in_space(&space).as_cell().expect("list cell");
            let item = list_cell.head().as_cell().expect("item cell");

            // Get the shared noun from this cell's tail
            let item_shared = item.tail().noun();
            let item_shared_raw = unsafe { item_shared.as_raw() };

            if let Some(first) = first_shared_raw {
                assert_eq!(
                    item_shared_raw, first,
                    "Cell {} should reference same shared noun as cell 0",
                    count
                );
            } else {
                first_shared_raw = Some(item_shared_raw);
            }

            // Verify the shared noun's contents
            let shared_cell = item_shared.in_space(&space).as_cell().expect("shared cell");
            assert_eq!(
                shared_cell.head().noun().as_direct().expect("999").data(),
                999,
                "Shared noun head should be 999"
            );
            assert_eq!(
                shared_cell.tail().noun().as_direct().expect("888").data(),
                888,
                "Shared noun tail should be 888"
            );

            current = list_cell.tail().noun();
            count += 1;
        }

        assert_eq!(count, NUM_REFS, "Should have traversed all {} cells", NUM_REFS);

        // Verify the entire structure is in PMA
        root.assert_in_pma(&pma);
    }

    /// Verifies no forwarding pointers remain in PMA memory after evacuation completes.
    ///
    /// Forwarding pointers are used transiently during evacuation to preserve
    /// structural sharing. They should only exist in stack memory and must never
    /// be written to the PMA. This test scans the entire allocated PMA region
    /// to verify no forwarding pointer tags are present.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_forwarding_pointer_not_leaked_to_pma() {
        use crate::noun::CELL_MASK;

        let mut stack = NockStack::new(1 << 14, 0);
        let mut pma = test_pma(10000);

        // Forwarding pointer tag: all top 3 bits set (111)
        const FORWARDING_MASK: u64 = CELL_MASK;
        let forwarding_tag: u64 = u64::MAX & CELL_MASK;

        // Create a complex structure with sharing to exercise forwarding pointers
        // This is similar to the diamond test but we'll verify the PMA contents

        // Shared nodes at different levels
        let shared_leaf = Cell::new(&mut stack, D(100), D(200)).as_noun();
        let shared_mid = Cell::new(&mut stack, D(1), shared_leaf).as_noun();

        // Create multiple references to shared nodes
        let branch_a = Cell::new(&mut stack, D(10), shared_mid).as_noun();
        let branch_b = Cell::new(&mut stack, D(20), shared_mid).as_noun();
        let branch_c = Cell::new(&mut stack, D(30), shared_leaf).as_noun();

        // Root containing all branches
        let ab = Cell::new(&mut stack, branch_a, branch_b).as_noun();
        let mut root = Cell::new(&mut stack, ab, branch_c).as_noun();

        // Also add an indirect atom to test that path
        let data: [u64; 3] = [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444, 0x5555_6666_7777_8888];
        let indirect = unsafe { IndirectAtom::new_raw(&mut stack, 3, data.as_ptr()) };
        root = Cell::new(&mut stack, root, indirect.as_noun()).as_noun();

        // Evacuate to PMA
        unsafe { root.copy_to_pma(&stack, &mut pma) };

        // Scan the entire allocated PMA region for forwarding pointers
        let alloc_words = pma.alloc_offset();
        let base_ptr = pma.arena().base_ptr() as *const u64;

        for word_idx in 0..alloc_words {
            let word = unsafe { *base_ptr.add(word_idx) };

            // Check if this word has the forwarding pointer tag
            // Forwarding tag is when top 3 bits are all 1s (111)
            if word & FORWARDING_MASK == forwarding_tag {
                // This could be a legitimate value, but let's be more careful:
                // A forwarding pointer would have the tag bits set AND point to
                // a valid address. Direct atoms with high values could also have
                // these bits set, so we need to distinguish.
                //
                // However, for safety, we check if it looks like a tagged pointer
                // by checking if the lower bits (after removing tag) could be
                // a valid pointer or offset.
                //
                // For our test data, we use small direct atoms (< DIRECT_MAX)
                // and known indirect atom data that won't have these tag bits.

                // The safest check: in our test, no legitimate data should have
                // the forwarding tag pattern, because:
                // 1. Direct atoms in our test are small (< 100, 200, etc.)
                // 2. Cells store nouns with different tag patterns
                // 3. Indirect atom data is controlled and doesn't have this pattern

                panic!(
                    "Found potential forwarding pointer at word {}: {:#018x}",
                    word_idx, word
                );
            }
        }

        // Verify the structure is still valid and readable
        root.assert_in_pma(&pma);
    }

    /// Verifies evacuation handles very deep trees (e.g., 1000 levels) without
    /// stack overflow in the worklist loop.
    ///
    /// The evacuation algorithm uses an iterative worklist rather than recursion,
    /// so it should handle arbitrarily deep structures without overflowing the
    /// Rust call stack.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_maximum_depth_tree() {
        let mut stack = NockStack::new(1 << 16, 0);
        let mut pma = test_pma(100000);
        let space = NounSpace::new(&stack, &pma);

        const DEPTH: u64 = 1000;

        // Build a deeply nested structure: [1 [2 [3 [4 ... [999 1000]]]]]
        let mut noun = D(DEPTH);
        for i in (1..DEPTH).rev() {
            noun = Cell::new(&mut stack, D(i), noun).as_noun();
        }

        // Verify it's deeply nested and stack-allocated
        assert!(noun.is_cell(), "Root should be a cell");
        assert!(
            matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be stack-allocated before evacuation"
        );

        // Evacuate - this should NOT cause a stack overflow
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Verify expected allocation: (DEPTH - 1) cells
        let cell_words = word_size_of::<CellMemory>();
        assert_eq!(
            pma.alloc_offset(),
            cell_words * (DEPTH as usize - 1),
            "Should allocate {} cells",
            DEPTH - 1
        );

        // Verify root is in offset form
        assert!(
            !matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Root should be in offset form after evacuation"
        );

        // Traverse the entire structure to verify correctness
        let mut current = noun;
        for expected in 1..DEPTH {
            assert!(current.is_cell(), "Should be cell at depth {}", expected);
            let cell = current.in_space(&space).as_cell().expect("is cell");

            let head = cell.head().noun();
            assert_eq!(
                head.as_direct().expect("direct").data(),
                expected,
                "Head at depth {} should be {}",
                expected,
                expected
            );

            current = cell.tail().noun();
        }

        // Final element should be DEPTH
        assert_eq!(
            current.as_direct().expect("direct").data(),
            DEPTH,
            "Leaf should be {}",
            DEPTH
        );

        noun.assert_in_pma(&pma);
    }

    /// Verifies indirect atoms near the maximum representable size evacuate correctly.
    ///
    /// Tests with a large indirect atom (1000 words = 8000 bytes) to ensure
    /// the evacuation handles large allocations properly.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_large_indirect_atom() {
        let mut stack = NockStack::new(1 << 16, 0);
        let mut pma = test_pma(100000);
        let space = NounSpace::new(&stack, &pma);

        const LARGE_SIZE: usize = 1000; // 1000 words = 8000 bytes

        // Create a large data array with a recognizable pattern
        let data: Vec<u64> = (0..LARGE_SIZE)
            .map(|i| 0xFEDCBA98_00000000u64 | (i as u64))
            .collect();

        // Create the large indirect atom
        let indirect = unsafe { IndirectAtom::new_raw(&mut stack, LARGE_SIZE, data.as_ptr()) };
        let mut noun = indirect.as_noun();

        // Verify it's stack-allocated
        assert!(noun.is_indirect(), "Should be indirect atom");
        assert!(
            matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be stack-allocated before evacuation"
        );

        // Evacuate to PMA
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Verify allocation size: metadata (1) + size (1) + data (LARGE_SIZE)
        let expected_alloc = LARGE_SIZE + 2;
        assert_eq!(
            pma.alloc_offset(),
            expected_alloc,
            "Should allocate {} words",
            expected_alloc
        );

        // Verify it's now in offset form
        assert!(
            !matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be in offset form after evacuation"
        );

        // Verify the data is correct
        let read_indirect = noun.as_indirect().expect("indirect");
        let handle = read_indirect.as_atom().in_space(&space);

        assert_eq!(handle.size(), LARGE_SIZE, "Size should be {}", LARGE_SIZE);

        let data_ptr = handle.data_pointer();
        for i in 0..LARGE_SIZE {
            let expected = 0xFEDCBA98_00000000u64 | (i as u64);
            let actual = unsafe { *data_ptr.add(i) };
            assert_eq!(
                actual, expected,
                "Word {} should match expected value",
                i
            );
        }

        noun.assert_in_pma(&pma);
    }

    /// Verifies the smallest possible indirect atom (just over DIRECT_MAX) evacuates correctly.
    ///
    /// DIRECT_MAX is the largest value that fits in a direct atom. Any value larger
    /// requires an indirect atom. This tests the boundary case.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_single_word_indirect_atom() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);
        let space = NounSpace::new(&stack, &pma);

        // DIRECT_MAX + 1 is the smallest value requiring indirect storage
        // DIRECT_MAX = 0x7FFF_FFFF_FFFF_FFFF (63 bits set)
        // DIRECT_MAX + 1 = 0x8000_0000_0000_0000 (bit 63 set)
        let value = DIRECT_MAX + 1;

        // Create a single-word indirect atom
        let indirect = unsafe { IndirectAtom::new_raw(&mut stack, 1, &value) };
        let mut noun = indirect.as_noun();

        // Verify it's an indirect atom (not direct)
        assert!(noun.is_indirect(), "Should be indirect atom");
        assert!(!noun.is_direct(), "Should not be direct atom");
        assert!(
            matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be stack-allocated before evacuation"
        );

        // Evacuate to PMA
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Verify allocation size: metadata (1) + size (1) + data (1) = 3 words
        assert_eq!(
            pma.alloc_offset(),
            3,
            "Single-word indirect atom should allocate 3 words"
        );

        // Verify it's now in offset form
        assert!(
            !matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be in offset form after evacuation"
        );

        // Verify the value is correct
        let read_indirect = noun.as_indirect().expect("indirect");
        let handle = read_indirect.as_atom().in_space(&space);

        assert_eq!(handle.size(), 1, "Size should be 1 word");

        let data_ptr = handle.data_pointer();
        let read_value = unsafe { *data_ptr };
        assert_eq!(
            read_value, value,
            "Value should be DIRECT_MAX + 1 = {:#x}",
            value
        );

        noun.assert_in_pma(&pma);
    }

    /// Verifies a cell where head is already in PMA and tail is on stack
    /// evacuates correctly (only tail gets copied).
    ///
    /// This tests the mixed-location case where part of a noun is already
    /// evacuated and part is still on the stack.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_mixed_pma_stack_noun() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);
        let space = NounSpace::new(&stack, &pma);

        // First, create and evacuate the head: [1 2]
        let mut head = Cell::new(&mut stack, D(1), D(2)).as_noun();
        unsafe { head.copy_to_pma(&stack, &mut pma) };

        // Record offset after evacuating head
        let offset_after_head = pma.alloc_offset();
        let cell_words = word_size_of::<CellMemory>();
        assert_eq!(offset_after_head, cell_words, "Head should allocate 1 cell");

        // Verify head is in PMA
        assert!(
            !matches!(
                head.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Head should be in PMA"
        );

        // Create a new tail on the stack: [3 4]
        let tail = Cell::new(&mut stack, D(3), D(4)).as_noun();

        // Verify tail is on stack
        assert!(
            matches!(
                tail.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Tail should be on stack"
        );

        // Create a cell with PMA head and stack tail
        let mut mixed = Cell::new(&mut stack, head, tail).as_noun();

        // Verify the mixed cell is on stack
        assert!(
            matches!(
                mixed.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Mixed cell should be on stack"
        );

        // Evacuate the mixed cell
        unsafe { mixed.copy_to_pma(&stack, &mut pma) };

        // Calculate expected allocation:
        // - Head was already in PMA (0 new words)
        // - Tail: 1 cell (3 words)
        // - Mixed cell itself: 1 cell (3 words)
        // Total new: 2 cells
        let expected_new_alloc = cell_words * 2;
        assert_eq!(
            pma.alloc_offset() - offset_after_head,
            expected_new_alloc,
            "Should only allocate tail and outer cell (head already in PMA)"
        );

        // Verify the entire structure is now in PMA
        assert!(
            !matches!(
                mixed.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Mixed cell should be in PMA after evacuation"
        );

        // Verify we can read through the structure
        let mixed_cell = mixed.in_space(&space).as_cell().expect("mixed is cell");

        // Check head [1 2]
        let head_cell = mixed_cell.head().as_cell().expect("head is cell");
        assert_eq!(
            head_cell.head().noun().as_direct().expect("1").data(),
            1,
            "Head's head should be 1"
        );
        assert_eq!(
            head_cell.tail().noun().as_direct().expect("2").data(),
            2,
            "Head's tail should be 2"
        );

        // Check tail [3 4]
        let tail_cell = mixed_cell.tail().as_cell().expect("tail is cell");
        assert_eq!(
            tail_cell.head().noun().as_direct().expect("3").data(),
            3,
            "Tail's head should be 3"
        );
        assert_eq!(
            tail_cell.tail().noun().as_direct().expect("4").data(),
            4,
            "Tail's tail should be 4"
        );

        mixed.assert_in_pma(&pma);
    }

    /// Verifies that evacuating the same noun twice in succession is a no-op
    /// the second time.
    ///
    /// After the first evacuation, the noun is already in PMA (offset form).
    /// A second evacuation should detect this and not allocate any new memory.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_same_noun_twice_same_call() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);
        let space = NounSpace::new(&stack, &pma);

        // Create a cell with nested structure
        let inner = Cell::new(&mut stack, D(1), D(2)).as_noun();
        let mut noun = Cell::new(&mut stack, inner, D(3)).as_noun();

        // First evacuation
        unsafe { noun.copy_to_pma(&stack, &mut pma) };
        let offset_after_first = pma.alloc_offset();

        // Verify it's in PMA
        assert!(
            !matches!(
                noun.in_space(&space).allocated_location(),
                Some(AllocLocation::Stack)
            ),
            "Should be in PMA after first evacuation"
        );

        // Second evacuation of the same noun
        unsafe { noun.copy_to_pma(&stack, &mut pma) };
        let offset_after_second = pma.alloc_offset();

        // Should be a no-op - no new allocations
        assert_eq!(
            offset_after_first, offset_after_second,
            "Second evacuation should not allocate anything"
        );

        // Verify data is still correct
        let cell = noun.in_space(&space).as_cell().expect("is cell");
        let inner_cell = cell.head().as_cell().expect("inner is cell");
        assert_eq!(
            inner_cell.head().noun().as_direct().expect("1").data(),
            1,
            "Inner head should be 1"
        );
        assert_eq!(
            inner_cell.tail().noun().as_direct().expect("2").data(),
            2,
            "Inner tail should be 2"
        );
        assert_eq!(
            cell.tail().noun().as_direct().expect("3").data(),
            3,
            "Outer tail should be 3"
        );

        noun.assert_in_pma(&pma);
    }

    /// Verifies that after resetting the PMA, we can evacuate the same structure
    /// again without confusion from old data.
    ///
    /// This tests that:
    /// 1. Reset properly clears the allocation state
    /// 2. Re-evacuation works correctly (no stale forwarding pointers)
    /// 3. The new copy is independent of the old (now-invalid) copy
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuate_after_pma_reset() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);

        // Create a structure on the stack
        let inner = Cell::new(&mut stack, D(100), D(200)).as_noun();
        let outer = Cell::new(&mut stack, inner, D(300)).as_noun();

        // We need fresh copies for each evacuation since copy_to_pma modifies
        // the source noun (sets forwarding pointers)

        // First evacuation
        let mut noun1 = outer;
        unsafe { noun1.copy_to_pma(&stack, &mut pma) };
        let offset_after_first = pma.alloc_offset();

        // Verify allocation happened
        let cell_words = word_size_of::<CellMemory>();
        assert_eq!(
            offset_after_first,
            cell_words * 2, // outer + inner cells
            "First evacuation should allocate 2 cells"
        );

        // Reset the PMA
        pma.reset();
        assert_eq!(pma.alloc_offset(), 0, "Reset should clear alloc_offset");

        // Create a fresh copy of the same structure (since original was modified)
        let inner2 = Cell::new(&mut stack, D(100), D(200)).as_noun();
        let outer2 = Cell::new(&mut stack, inner2, D(300)).as_noun();
        let mut noun2 = outer2;

        // Second evacuation after reset
        unsafe { noun2.copy_to_pma(&stack, &mut pma) };
        let offset_after_second = pma.alloc_offset();

        // Should allocate the same amount (starting from 0)
        assert_eq!(
            offset_after_second,
            cell_words * 2,
            "Second evacuation should allocate 2 cells (same as first)"
        );

        // Verify the new copy is correct
        let space = NounSpace::new(&stack, &pma);
        let cell = noun2.in_space(&space).as_cell().expect("is cell");
        let inner_cell = cell.head().as_cell().expect("inner is cell");

        assert_eq!(
            inner_cell.head().noun().as_direct().expect("100").data(),
            100,
            "Inner head should be 100"
        );
        assert_eq!(
            inner_cell.tail().noun().as_direct().expect("200").data(),
            200,
            "Inner tail should be 200"
        );
        assert_eq!(
            cell.tail().noun().as_direct().expect("300").data(),
            300,
            "Outer tail should be 300"
        );

        noun2.assert_in_pma(&pma);
    }

    /// Verifies cell metadata is properly copied (not uninitialized) by checking
    /// mug cache bits after evacuation.
    ///
    /// The metadata field contains a cached mug hash in its lower 31 bits.
    /// This test computes and caches the mug before evacuation, then verifies
    /// it's preserved after evacuation.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuated_metadata_initialized() {
        use crate::mug::mug_u32;

        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);

        // Create a cell structure
        let inner = Cell::new(&mut stack, D(42), D(99)).as_noun();
        let mut noun = Cell::new(&mut stack, inner, D(7)).as_noun();

        // Compute and cache the mug (this writes to metadata)
        let space_before = NounSpace::stack_only(&stack);
        let mug_before = mug_u32(&mut stack, noun);
        assert!(mug_before > 0, "Mug should be non-zero");

        // Verify mug is cached in metadata before evacuation
        let allocated_before = noun.as_allocated().expect("is allocated");
        let cached_mug_before = allocated_before.get_cached_mug(&space_before);
        assert_eq!(
            cached_mug_before,
            Some(mug_before),
            "Mug should be cached before evacuation"
        );

        // Evacuate to PMA
        unsafe { noun.copy_to_pma(&stack, &mut pma) };

        // Verify mug is still accessible after evacuation
        let space_after = NounSpace::new(&stack, &pma);
        let allocated_after = noun.as_allocated().expect("is allocated");
        let cached_mug_after = allocated_after.get_cached_mug(&space_after);

        assert_eq!(
            cached_mug_after,
            Some(mug_before),
            "Mug should be preserved after evacuation"
        );

        // Also verify the metadata word directly
        let cell = noun.in_space(&space_after).as_cell().expect("is cell");
        let cell_ptr = unsafe { cell.raw_pointer() };
        let metadata = unsafe { (*cell_ptr).metadata };

        // Lower 31 bits should contain the mug
        let mug_from_metadata = (metadata & 0x7FFFFFFF) as u32;
        assert_eq!(
            mug_from_metadata, mug_before,
            "Metadata lower 31 bits should contain mug"
        );

        noun.assert_in_pma(&pma);
    }

    /// Verifies that for indirect atoms that don't fill their last word completely,
    /// the padding bytes are deterministic (zeroed).
    ///
    /// This test creates an indirect atom with a byte count that doesn't evenly
    /// divide into 8-byte words, evacuates it, and verifies the padding bytes
    /// in the last word are zero.
    #[test]
    #[cfg_attr(miri, ignore = "memfd_create unsupported in Miri")]
    fn test_evacuated_indirect_atom_padding_zeroed() {
        let mut stack = NockStack::new(1 << 10, 0);
        let mut pma = test_pma(1000);
        let space = NounSpace::new(&stack, &pma);

        // Create indirect atoms with various non-word-aligned byte sizes
        // Test cases: 9 bytes (2 words, 1 byte used in second word)
        //             13 bytes (2 words, 5 bytes used in second word)
        //             1 byte that requires indirect (value > DIRECT_MAX needs special handling)

        // Test case 1: 9 bytes of data (requires 2 words, only 1 byte in second word)
        // We'll use new_raw_bytes which zeros the buffer
        let data_9: [u8; 9] = [0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77];
        let indirect_9 = unsafe { IndirectAtom::new_raw_bytes_ref(&mut stack, &data_9) };
        let mut noun_9 = indirect_9.as_noun();

        // Evacuate to PMA
        unsafe { noun_9.copy_to_pma(&stack, &mut pma) };

        // Verify padding in second word is zero
        let read_indirect = noun_9.as_indirect().expect("indirect");
        let handle = read_indirect.as_atom().in_space(&space);
        assert_eq!(handle.size(), 2, "Should be 2 words");

        let data_ptr = handle.data_pointer();
        let word_0 = unsafe { *data_ptr };
        let word_1 = unsafe { *data_ptr.add(1) };

        // First word should be fully populated: 0x8899AABBCCDDEEFF
        assert_eq!(
            word_0,
            u64::from_le_bytes([0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88]),
            "First word should match data"
        );

        // Second word: only lowest byte (0x77) should be set, rest should be zero
        assert_eq!(
            word_1,
            0x77,
            "Second word should have only lowest byte set, padding should be zero"
        );

        // Test case 2: 13 bytes of data (requires 2 words, 5 bytes in second word)
        pma.reset();
        let data_13: [u8; 13] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
        let indirect_13 = unsafe { IndirectAtom::new_raw_bytes_ref(&mut stack, &data_13) };
        let mut noun_13 = indirect_13.as_noun();

        unsafe { noun_13.copy_to_pma(&stack, &mut pma) };

        let read_indirect_13 = noun_13.as_indirect().expect("indirect");
        let handle_13 = read_indirect_13.as_atom().in_space(&space);
        assert_eq!(handle_13.size(), 2, "Should be 2 words");

        let data_ptr_13 = handle_13.data_pointer();
        let word_1_13 = unsafe { *data_ptr_13.add(1) };

        // Second word should have 5 bytes set (9, 10, 11, 12, 13), upper 3 bytes zero
        let expected_word_1 = u64::from_le_bytes([9, 10, 11, 12, 13, 0, 0, 0]);
        assert_eq!(
            word_1_13, expected_word_1,
            "Second word should have 5 bytes set with 3 bytes of zero padding"
        );

        // Verify the high 3 bytes are specifically zero
        assert_eq!(
            word_1_13 & 0xFFFFFF00_00000000,
            0,
            "Upper 3 bytes of second word should be zero"
        );

        noun_13.assert_in_pma(&pma);
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod paging_tests {
    use super::{test_pma_path, Pma};

    const SLAB_BYTES: usize = 64 * 1024 * 1024;
    const TOUCH_PAGES: usize = 64;

    #[test]
    #[cfg_attr(miri, ignore = "mincore/madvise unsupported in Miri")]
    fn pma_file_backed_pages_out_and_faults_lazily() {
        let words = SLAB_BYTES >> 3;
        let path = test_pma_path("paging");
        let pma = Pma::new(words, path).expect("failed to create PMA");
        let base = pma.arena().base_ptr();
        let len = pma.arena().len_bytes();
        let page = page_size();

        assert_eq!(len, SLAB_BYTES, "unexpected PMA length");
        assert_eq!(
            len % page,
            0,
            "PMA length must be page sized, len={len}, page={page}"
        );

        touch_entire_region(base, len, page);
        let resident_bitmap = mincore_bitmap(base, len);
        let initial_ratio = residency_ratio(&resident_bitmap);
        println!("[pma-paging] initial residency ratio {:.3}", initial_ratio);
        assert!(
            resident_bitmap.iter().all(|b| b & 1 == 1),
            "expected fully resident slab after touching every page"
        );

        drop_all_pages(base, len);
        let after_drop = mincore_bitmap(base, len);
        let post_drop_ratio = residency_ratio(&after_drop);
        println!(
            "[pma-paging] post-drop residency ratio {:.3}",
            post_drop_ratio
        );
        if post_drop_ratio > 0.9 {
            println!(
                "[pma-paging] paging did not drop pages; skipping remainder (ratio={post_drop_ratio:.3})"
            );
            return;
        }
        assert!(
            post_drop_ratio < 0.1,
            "expected paging to drop most pages, ratio={post_drop_ratio}"
        );

        let total_pages = len / page;
        let touched_pages = fault_sparse(base, len, page, TOUCH_PAGES);
        assert!(touched_pages > 0, "expected to fault at least one page");

        let post_fault = mincore_bitmap(base, len);
        let post_fault_ratio = residency_ratio(&post_fault);
        let expected_ratio = touched_pages as f64 / total_pages.max(1) as f64;
        println!(
            "[pma-paging] post-fault residency ratio {:.4} (expected {:.4}, touched {} pages)",
            post_fault_ratio, expected_ratio, touched_pages
        );
        assert!(
            post_fault_ratio >= expected_ratio * 0.5 && post_fault_ratio <= expected_ratio * 2.0,
            "faulted pages should roughly match touched subset (ratio {} expected {})",
            post_fault_ratio,
            expected_ratio
        );
    }

    fn touch_entire_region(ptr: *mut u8, len: usize, page: usize) {
        for offset in (0..len).step_by(page) {
            unsafe {
                std::ptr::write_volatile(ptr.add(offset), (offset / page % 255) as u8);
            }
        }
    }

    fn fault_sparse(ptr: *mut u8, len: usize, page: usize, desired_pages: usize) -> usize {
        let total_pages = len / page;
        if total_pages == 0 {
            return 0;
        }
        let touches = desired_pages.min(total_pages.max(1));
        let stride = (total_pages / touches).max(1);
        let mut touched = 0;
        let mut page_idx = 0;
        while touched < touches && page_idx < total_pages {
            unsafe {
                std::ptr::read_volatile(ptr.add(page_idx * page));
            }
            touched += 1;
            page_idx = page_idx.saturating_add(stride);
        }
        touched
    }

    fn drop_all_pages(ptr: *mut u8, len: usize) {
        #[cfg(target_os = "linux")]
        {
            let ret = unsafe { libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_PAGEOUT) };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                match err.raw_os_error() {
                    Some(libc::EINVAL) | Some(libc::ENOSYS) => {
                        let fallback = unsafe {
                            libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_DONTNEED)
                        };
                        if fallback != 0 {
                            panic!(
                                "madvise fallback failed: {}",
                                std::io::Error::last_os_error()
                            );
                        }
                    }
                    _ => panic!("madvise(MADV_PAGEOUT) failed: {err}"),
                }
            }
        }
        #[cfg(target_os = "macos")]
        {
            let ret = unsafe { libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_DONTNEED) };
            if ret != 0 {
                panic!(
                    "madvise(MADV_DONTNEED) failed: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    fn mincore_bitmap(ptr: *mut u8, len: usize) -> Vec<u8> {
        let page = page_size();
        assert_eq!(
            len % page,
            0,
            "mincore requires len to be page sized, len={len}, page={page}"
        );
        let pages = len / page;
        let mut vec = vec![0u8; pages];
        let ret = unsafe {
            libc::mincore(
                ptr as *mut libc::c_void,
                len,
                vec.as_mut_ptr() as *mut libc::c_uchar,
            )
        };
        if ret != 0 {
            panic!("mincore failed: {}", std::io::Error::last_os_error());
        }
        vec
    }

    fn residency_ratio(bitmap: &[u8]) -> f64 {
        if bitmap.is_empty() {
            return 0.0;
        }
        let resident = bitmap.iter().filter(|b| **b & 1 == 1).count();
        resident as f64 / bitmap.len() as f64
    }

    fn page_size() -> usize {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }
}
