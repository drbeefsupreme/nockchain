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
mod tests;

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod paging_tests;
