use nockapp::noun::slab::NounSlab;
use nockchain_math::structs::HoonMapIter;
use nockvm::noun::{Atom, Cell, Noun};

pub type NounSpace = ();

pub trait NounSlabCompatExt {
    fn noun_space(&self) -> NounSpace;
    fn root_noun(&self) -> Noun;
}

impl<J> NounSlabCompatExt for NounSlab<J> {
    fn noun_space(&self) -> NounSpace {}

    fn root_noun(&self) -> Noun {
        // SAFETY: Bench code only calls this after the slab root has been
        // initialized by `set_root` or `cue_into`.
        unsafe { *self.root() }
    }
}

pub struct NounSpaceView {
    noun: Noun,
}

impl NounSpaceView {
    pub fn as_atom(&self) -> Result<Atom, nockvm::noun::Error> {
        self.noun.as_atom()
    }

    pub fn as_cell(&self) -> Result<Cell, nockvm::noun::Error> {
        self.noun.as_cell()
    }

    pub fn noun(&self) -> Noun {
        self.noun
    }
}

pub trait NounCompatExt {
    fn in_space(&self, _space: &NounSpace) -> NounSpaceView;
    fn noun(&self) -> Noun;
}

impl NounCompatExt for Noun {
    fn in_space(&self, _space: &NounSpace) -> NounSpaceView {
        NounSpaceView { noun: *self }
    }

    fn noun(&self) -> Noun {
        *self
    }
}

pub trait HoonMapIterCompatExt {
    fn new(noun: Noun, _space: &NounSpace) -> Self;
}

impl HoonMapIterCompatExt for HoonMapIter {
    fn new(noun: Noun, _space: &NounSpace) -> Self {
        HoonMapIter::from(noun)
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
        let atom = slab
            .root_noun()
            .as_atom()
            .expect("root should be atom")
            .as_u64()
            .expect("atom should fit u64");
        assert_eq!(atom, 42);
    }
}
