//! Private bench-local noun compatibility helpers for PMA runtime support.
//!
//! PMA noun access uses the bridge pattern:
//! `noun.in_space(space)` -> handle -> handle-side `as_cell()`, `head()`, `tail()`, `uncell()`.

use nockapp::noun::slab::NounSlab;
use nockchain_math::structs::{HoonList, HoonMapIter};
use nockvm::noun::Noun;
#[cfg(feature = "pma-runtime-compat")]
use nockvm::noun::NounAllocator;
#[cfg(feature = "pma-runtime-compat")]
pub(crate) use nockvm::noun::NounSpace;
use noun_serde::{NounDecode, NounDecodeError};

#[cfg(not(feature = "pma-runtime-compat"))]
pub(crate) struct NounSpace;

#[cfg(not(feature = "pma-runtime-compat"))]
static EMPTY_SPACE: NounSpace = NounSpace;

#[cfg(feature = "pma-runtime-compat")]
pub(crate) fn space_for_slab<J>(slab: &NounSlab<J>) -> NounSpace {
    slab.noun_space()
}

#[cfg(not(feature = "pma-runtime-compat"))]
pub(crate) fn space_for_slab<J>(_slab: &NounSlab<J>) -> NounSpace {
    EMPTY_SPACE
}

pub(crate) fn decode_with_space<T: NounDecode>(
    noun: &Noun,
    space: &NounSpace,
) -> Result<T, NounDecodeError> {
    #[cfg(feature = "pma-runtime-compat")]
    {
        T::from_noun(noun, space)
    }

    #[cfg(not(feature = "pma-runtime-compat"))]
    {
        let _ = space;
        T::from_noun(noun)
    }
}

pub(crate) fn atom_is_zero(noun: &Noun, space: &NounSpace) -> Result<bool, NounDecodeError> {
    #[cfg(feature = "pma-runtime-compat")]
    {
        Ok(noun.in_space(space).as_atom()?.as_u64()? == 0)
    }

    #[cfg(not(feature = "pma-runtime-compat"))]
    {
        let _ = space;
        Ok(noun.as_atom()?.as_u64()? == 0)
    }
}

pub(crate) fn hoon_list_items(noun: Noun, space: &NounSpace) -> Result<Vec<Noun>, NounDecodeError> {
    #[cfg(feature = "pma-runtime-compat")]
    {
        Ok(HoonList::try_from(noun, space)?.collect())
    }

    #[cfg(not(feature = "pma-runtime-compat"))]
    {
        let _ = space;
        Ok(HoonList::try_from(noun)?.collect())
    }
}

pub(crate) fn hoon_map_entries(noun: Noun, space: &NounSpace) -> Vec<Noun> {
    #[cfg(feature = "pma-runtime-compat")]
    {
        HoonMapIter::new(&noun.in_space(space))
            .map(|entry| entry.noun())
            .collect()
    }

    #[cfg(not(feature = "pma-runtime-compat"))]
    {
        let _ = space;
        HoonMapIter::from(noun).collect()
    }
}

pub(crate) fn noun_head(noun: Noun, space: &NounSpace) -> Result<Noun, NounDecodeError> {
    #[cfg(feature = "pma-runtime-compat")]
    {
        Ok(noun.in_space(space).as_cell()?.head().noun())
    }

    #[cfg(not(feature = "pma-runtime-compat"))]
    {
        let _ = space;
        Ok(noun.as_cell()?.head())
    }
}

pub(crate) fn noun_tail(noun: Noun, space: &NounSpace) -> Result<Noun, NounDecodeError> {
    #[cfg(feature = "pma-runtime-compat")]
    {
        Ok(noun.in_space(space).as_cell()?.tail().noun())
    }

    #[cfg(not(feature = "pma-runtime-compat"))]
    {
        let _ = space;
        Ok(noun.as_cell()?.tail())
    }
}
