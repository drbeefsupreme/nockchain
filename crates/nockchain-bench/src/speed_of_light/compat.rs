use nockapp::noun::slab::NounSlab;
use nockvm::noun::{Noun, NounAllocator};

pub use nockvm::noun::NounSpace;

pub trait NounSlabCompatExt {
    fn bench_noun_space(&self) -> NounSpace;
    fn bench_root_noun(&self) -> Noun;
}

impl<J> NounSlabCompatExt for NounSlab<J> {
    fn bench_noun_space(&self) -> NounSpace {
        NounAllocator::noun_space(self)
    }

    fn bench_root_noun(&self) -> Noun {
        // SAFETY: Bench code only calls this after the slab root has been
        // initialized by `set_root` or `cue_into`.
        unsafe { *self.root() }
    }
}

#[cfg(test)]
mod tests {
    use nockapp::noun::slab::NockJammer;
    use nockvm::noun::D;

    use super::*;

    #[test]
    fn test_root_noun_returns_set_root() {
        let mut slab: NounSlab<NockJammer> = NounSlab::new();
        slab.set_root(D(42));
        let space = slab.bench_noun_space();
        let atom = slab
            .bench_root_noun()
            .in_space(&space)
            .as_atom()
            .expect("root should be atom")
            .as_u64()
            .expect("atom should fit u64");
        assert_eq!(atom, 42);
    }
}
