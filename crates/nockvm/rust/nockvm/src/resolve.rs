//! Resolution Closures for Noun pointer resolution
//!
//! This module provides a trait-based approach to resolving noun pointers,
//! replacing thread-local storage with explicit resolver parameters.
//!
//! # Design
//!
//! Nouns can exist in different arenas (NockStack, PMA, NounSlab) and can be
//! in different forms:
//! - **Pointer form**: The payload is a shifted raw pointer (for stack/slab)
//! - **Offset form**: The payload is a word offset from an arena base (for PMA)
//!
//! The `Resolve` trait abstracts over these differences, allowing operations
//! on nouns to work uniformly regardless of where the noun lives.
//!
//! # Resolver Types
//!
//! - [`StackResolver`]: For stack-only operations (pointer form, no PMA)
//! - [`PmaResolver`]: For PMA-only operations (offset form only)
//! - [`MixedResolver`]: For mixed stack + PMA operations (common case during interpretation)
//!
//! # Future: Stack Offset Form
//!
//! When the NockStack moves to offset form for relocatability/COW forking,
//! the arena bit in nouns will distinguish stack-offset from PMA-offset.
//! The resolver types are designed to accommodate this future change.

use crate::mem::Arena;
use crate::pma::Pma;

/// Trait for resolving tagged noun pointers to raw memory pointers.
///
/// Implementations provide the logic to convert a tagged 64-bit noun value
/// to a raw pointer, handling both pointer-form and offset-form nouns.
///
/// # Safety
///
/// The `resolve` method returns a raw pointer. Callers must ensure:
/// - The tagged value represents a valid allocated noun
/// - The returned pointer is only used while the underlying memory is valid
pub trait Resolve: Copy {
    /// Resolve a tagged pointer value to a raw pointer.
    ///
    /// # Arguments
    /// * `tagged` - The raw 64-bit noun value
    /// * `mask` - The tag mask for the noun type (INDIRECT_MASK or CELL_MASK)
    ///
    /// # Returns
    /// A raw pointer to the noun's data in memory.
    fn resolve(&self, tagged: u64, mask: u64) -> *const u8;

    /// Resolve a tagged pointer value to a mutable raw pointer.
    #[inline(always)]
    fn resolve_mut(&self, tagged: u64, mask: u64) -> *mut u8 {
        self.resolve(tagged, mask) as *mut u8
    }
}

// Location bit position (same as in noun.rs)
const LOCATION_BIT: u64 = 1 << 60;

/// Extract the payload from a tagged pointer.
#[inline(always)]
fn payload(tagged: u64, mask: u64) -> u64 {
    (tagged & !mask) & !(LOCATION_BIT)
}

/// Check if a tagged pointer is in offset form.
#[inline(always)]
fn is_offset_form(tagged: u64) -> bool {
    tagged & LOCATION_BIT != 0
}

/// Resolver for stack-only operations.
///
/// This resolver handles pointer-form nouns that live in the NockStack.
/// It's the most efficient resolver when you know all nouns are stack-allocated.
///
/// # Panics
/// Panics if given an offset-form noun (PMA reference).
#[derive(Copy, Clone, Debug)]
pub struct StackResolver;

impl Resolve for StackResolver {
    #[inline(always)]
    fn resolve(&self, tagged: u64, mask: u64) -> *const u8 {
        debug_assert!(
            !is_offset_form(tagged),
            "StackResolver cannot resolve offset-form nouns"
        );
        // Pointer form: payload is shifted pointer
        (payload(tagged, mask) << 3) as *const u8
    }
}

/// Resolver for PMA-only operations.
///
/// This resolver handles offset-form nouns that live in the PMA.
/// Use when reading from persistent storage without modification.
#[derive(Copy, Clone, Debug)]
pub struct PmaResolver {
    pma_base: *const u8,
}

impl PmaResolver {
    /// Create a new PMA-only resolver.
    pub fn new(pma_base: *const u8) -> Self {
        Self { pma_base }
    }

    /// Create from an Arena reference.
    pub fn from_arena(arena: &Arena) -> Self {
        Self {
            pma_base: arena.base_ptr(),
        }
    }

    /// Create from a Pma reference.
    pub fn from_pma(pma: &Pma) -> Self {
        Self {
            pma_base: pma.arena().base_ptr(),
        }
    }
}

impl Resolve for PmaResolver {
    #[inline(always)]
    fn resolve(&self, tagged: u64, mask: u64) -> *const u8 {
        debug_assert!(
            is_offset_form(tagged),
            "PmaResolver cannot resolve pointer-form nouns"
        );
        // Offset form: base + offset * 8
        let offset = payload(tagged, mask) as usize;
        unsafe { self.pma_base.add(offset << 3) }
    }
}

/// Resolver for mixed stack + PMA operations.
///
/// This is the most common resolver during interpretation, where nouns
/// may be a mix of:
/// - Stack-allocated cells/atoms (pointer form)
/// - PMA-persistent cells/atoms (offset form)
///
/// The LOCATION_BIT in each noun determines which form it uses.
#[derive(Copy, Clone, Debug)]
pub struct MixedResolver {
    /// Base pointer for PMA offset resolution.
    /// Stack pointers are self-resolving (the pointer is in the payload).
    pma_base: *const u8,
}

impl MixedResolver {
    /// Create a new mixed resolver with the given PMA base.
    ///
    /// Stack pointers don't need a base (they contain the raw pointer),
    /// so only the PMA base is needed.
    pub fn new(pma_base: *const u8) -> Self {
        Self { pma_base }
    }

    /// Create a mixed resolver from a Pma reference.
    pub fn from_pma(pma: &Pma) -> Self {
        Self {
            pma_base: pma.arena().base_ptr(),
        }
    }

    /// Create a mixed resolver from explicit arena references.
    ///
    /// The stack arena is not used (stack pointers are self-resolving),
    /// but accepted for API clarity when you have both arenas.
    pub fn from_arenas(_stack_arena: &Arena, pma_arena: &Arena) -> Self {
        Self {
            pma_base: pma_arena.base_ptr(),
        }
    }

    /// Create a stack-only resolver (no PMA support).
    ///
    /// This will panic if any offset-form nouns are encountered.
    pub fn stack_only() -> Self {
        Self {
            pma_base: std::ptr::null(),
        }
    }
}

impl Resolve for MixedResolver {
    #[inline(always)]
    fn resolve(&self, tagged: u64, mask: u64) -> *const u8 {
        if is_offset_form(tagged) {
            // Offset form: PMA base + offset * 8
            let offset = payload(tagged, mask) as usize;
            debug_assert!(
                !self.pma_base.is_null(),
                "MixedResolver has null PMA base but received offset-form noun"
            );
            unsafe { self.pma_base.add(offset << 3) }
        } else {
            // Pointer form: payload is shifted pointer
            (payload(tagged, mask) << 3) as *const u8
        }
    }
}

/// Legacy resolver that uses an Arena reference.
///
/// This provides compatibility with the existing `*_with_arena` methods
/// during migration from TLS to resolvers.
#[derive(Copy, Clone)]
pub struct ArenaResolver<'a> {
    arena: &'a Arena,
}

impl<'a> ArenaResolver<'a> {
    /// Create a resolver from an arena reference.
    pub fn new(arena: &'a Arena) -> Self {
        Self { arena }
    }
}

impl Resolve for ArenaResolver<'_> {
    #[inline(always)]
    fn resolve(&self, tagged: u64, mask: u64) -> *const u8 {
        if is_offset_form(tagged) {
            let offset = payload(tagged, mask) as usize;
            self.arena.ptr_from_offset(offset as u32)
        } else {
            (payload(tagged, mask) << 3) as *const u8
        }
    }
}

/// Resolver that wraps another resolver and validates pointer bounds.
///
/// Useful for debugging to catch invalid pointer resolutions.
#[derive(Copy, Clone)]
pub struct ValidatingResolver<R: Resolve> {
    inner: R,
    stack_start: *const u8,
    stack_end: *const u8,
    pma_start: *const u8,
    pma_end: *const u8,
}

impl<R: Resolve> ValidatingResolver<R> {
    /// Create a validating wrapper around another resolver.
    pub fn new(
        inner: R,
        stack_start: *const u8,
        stack_end: *const u8,
        pma_start: *const u8,
        pma_end: *const u8,
    ) -> Self {
        Self {
            inner,
            stack_start,
            stack_end,
            pma_start,
            pma_end,
        }
    }
}

impl<R: Resolve> Resolve for ValidatingResolver<R> {
    #[inline(always)]
    fn resolve(&self, tagged: u64, mask: u64) -> *const u8 {
        let ptr = self.inner.resolve(tagged, mask);

        // Validate pointer is in expected range
        let in_stack = ptr >= self.stack_start && ptr < self.stack_end;
        let in_pma = ptr >= self.pma_start && ptr < self.pma_end;

        debug_assert!(
            in_stack || in_pma,
            "Resolved pointer {:p} not in stack [{:p}..{:p}] or PMA [{:p}..{:p}]",
            ptr,
            self.stack_start,
            self.stack_end,
            self.pma_start,
            self.pma_end
        );

        ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_resolver() {
        let resolver = StackResolver;

        // Create a fake stack-pointer tagged value
        // Pointer form: ptr >> 3, no LOCATION_BIT
        let fake_ptr = 0x1000u64;
        let tagged = (fake_ptr >> 3) | (0b10u64 << 62); // indirect tag

        let resolved = resolver.resolve(tagged, crate::noun::INDIRECT_MASK);
        assert_eq!(resolved as u64, fake_ptr);
    }

    #[test]
    fn test_pma_resolver() {
        let base = 0x2000u64 as *const u8;
        let resolver = PmaResolver::new(base);

        // Create a fake offset-form tagged value
        // Offset form: offset in words, LOCATION_BIT set
        let offset_words = 10u64;
        let tagged = offset_words | LOCATION_BIT | (0b10u64 << 62); // indirect tag + location bit

        let resolved = resolver.resolve(tagged, crate::noun::INDIRECT_MASK);
        let expected = unsafe { base.add((offset_words << 3) as usize) };
        assert_eq!(resolved, expected);
    }

    #[test]
    fn test_mixed_resolver_pointer_form() {
        let pma_base = 0x2000u64 as *const u8;
        let resolver = MixedResolver::new(pma_base);

        // Pointer form (no LOCATION_BIT)
        let fake_ptr = 0x1000u64;
        let tagged = (fake_ptr >> 3) | (0b10u64 << 62);

        let resolved = resolver.resolve(tagged, crate::noun::INDIRECT_MASK);
        assert_eq!(resolved as u64, fake_ptr);
    }

    #[test]
    fn test_mixed_resolver_offset_form() {
        let pma_base = 0x2000u64 as *const u8;
        let resolver = MixedResolver::new(pma_base);

        // Offset form (LOCATION_BIT set)
        let offset_words = 10u64;
        let tagged = offset_words | LOCATION_BIT | (0b10u64 << 62);

        let resolved = resolver.resolve(tagged, crate::noun::INDIRECT_MASK);
        let expected = unsafe { pma_base.add((offset_words << 3) as usize) };
        assert_eq!(resolved, expected);
    }
}
