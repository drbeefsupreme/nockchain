use nockapp::noun::slab::NounSlab;
use nockchain_math::structs::HoonMapIter;
use nockvm::noun::{Atom, Cell, Noun};

pub type NounSpace = ();

pub trait NounSlabCompatExt {
    fn noun_space(&self) -> NounSpace;
}

impl<J> NounSlabCompatExt for NounSlab<J> {
    fn noun_space(&self) -> NounSpace {}
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
