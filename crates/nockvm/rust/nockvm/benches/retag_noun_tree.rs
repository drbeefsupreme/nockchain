use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use nockvm::mem::NockStack;
use nockvm::noun::{Cell, IndirectAtom, Noun, D};

const STACK_WORDS: usize = 1 << 24; // 128 MiB arena
const LEAF_COUNT: usize = 5_120;

fn make_stack() -> NockStack {
    let stack = NockStack::new(STACK_WORDS, 0);
    stack.install_arena();
    stack
}

fn small_indirect(stack: &mut NockStack, seed: u64) -> Noun {
    let mut buf = [0u8; 24];
    buf[..8].copy_from_slice(&seed.to_le_bytes());
    buf[8..16].copy_from_slice(&seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).to_le_bytes());
    buf[16..24].copy_from_slice(&(seed ^ 0x55aa_55aa_55aa_55aa).to_le_bytes());
    unsafe { IndirectAtom::new_raw_bytes(stack, buf.len(), buf.as_ptr()).as_noun() }
}

fn large_indirect(stack: &mut NockStack, seed: u64, words: usize) -> Noun {
    debug_assert!(words >= 5 && words <= 1000);
    let mut data = vec![0u64; words];
    for (idx, slot) in data.iter_mut().enumerate() {
        *slot = seed
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(idx as u64 ^ 0xfeed_face_dead_beef);
    }
    // Keep the tail non-zero so normalization does not shrink the atom.
    data[words - 1] |= 1;
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() << 3) };
    let atom = unsafe { IndirectAtom::new_raw_bytes(stack, bytes.len(), bytes.as_ptr()) };
    atom.as_noun()
}

fn build_tree(stack: &mut NockStack, mut leaves: Vec<Noun>) -> Noun {
    assert!(!leaves.is_empty());
    while leaves.len() > 1 {
        let mut next = Vec::with_capacity((leaves.len() + 1) / 2);
        for chunk in leaves.chunks(2) {
            if chunk.len() == 2 {
                let cell = Cell::new(stack, chunk[0], chunk[1]);
                next.push(cell.as_noun());
            } else {
                next.push(chunk[0]);
            }
        }
        leaves = next;
    }
    leaves.pop().expect("non-empty tree")
}

fn build_unique_direct(_stack: &mut NockStack) -> Vec<Noun> {
    (0..LEAF_COUNT).map(|i| D(i as u64 + 1)).collect()
}

fn build_unique_small_indirect(stack: &mut NockStack) -> Vec<Noun> {
    (0..LEAF_COUNT)
        .map(|i| small_indirect(stack, 0x1000_0000 + i as u64))
        .collect()
}

fn build_mixed_direct_small_indirect(stack: &mut NockStack) -> Vec<Noun> {
    (0..LEAF_COUNT)
        .map(|i| {
            if i % 2 == 0 {
                D(i as u64 + 1)
            } else {
                small_indirect(stack, 0x2000_0000 + i as u64)
            }
        })
        .collect()
}

fn build_shared_small_indirect(stack: &mut NockStack) -> Vec<Noun> {
    let shared = small_indirect(stack, 0x5eed_baa5);
    let uniques = LEAF_COUNT / 10;
    let mut leaves = Vec::with_capacity(LEAF_COUNT);
    for i in 0..LEAF_COUNT {
        if i % 10 == 0 && (i / 10) < uniques {
            leaves.push(small_indirect(
                stack,
                0x3000_0000 + (i / 10) as u64,
            ));
        } else {
            leaves.push(shared);
        }
    }
    leaves
}

fn build_unique_large_indirect(stack: &mut NockStack) -> Vec<Noun> {
    (0..LEAF_COUNT)
        .map(|i| {
            let size = 5 + (i % 996);
            large_indirect(stack, 0x4000_0000 + i as u64, size)
        })
        .collect()
}

fn build_shared_large_indirect(stack: &mut NockStack) -> Vec<Noun> {
    let shared = large_indirect(stack, 0x5aa5_5aa5, 768);
    let mut leaves = Vec::with_capacity(LEAF_COUNT);
    for i in 0..LEAF_COUNT {
        if i % 10 == 0 {
            let size = 5 + (i % 996);
            leaves.push(large_indirect(
                stack,
                0x5000_0000 + (i / 10) as u64,
                size,
            ));
        } else {
            leaves.push(shared);
        }
    }
    leaves
}

fn setup_case<F>(mut make_leaves: F) -> impl FnMut() -> (NockStack, Noun)
where
    F: FnMut(&mut NockStack) -> Vec<Noun>,
{
    move || {
        let mut stack = make_stack();
        let leaves = make_leaves(&mut stack);
        assert!(leaves.len() >= LEAF_COUNT);
        let root = build_tree(&mut stack, leaves);
        (stack, root)
    }
}

fn bench_retag_noun_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("retag_noun_tree");
    let mut cases: Vec<(&str, Box<dyn FnMut() -> (NockStack, Noun)>)> = vec![
        ("direct_unique", Box::new(setup_case(build_unique_direct))),
        (
            "small_indirect_unique",
            Box::new(setup_case(build_unique_small_indirect)),
        ),
        (
            "mixed_direct_small_indirect",
            Box::new(setup_case(build_mixed_direct_small_indirect)),
        ),
        (
            "small_indirect_shared",
            Box::new(setup_case(build_shared_small_indirect)),
        ),
        (
            "large_indirect_unique",
            Box::new(setup_case(build_unique_large_indirect)),
        ),
        (
            "large_indirect_shared",
            Box::new(setup_case(build_shared_large_indirect)),
        ),
    ];

    for (label, setup) in cases.iter_mut() {
        group.bench_function(BenchmarkId::new("retag", label), |b| {
            b.iter_batched(
                || setup(),
                |(stack, mut root)| {
                    // retag_noun_tree is only meaningful on stack-allocated nouns
                    stack.retag_noun_tree(&mut root as *mut Noun);
                    black_box(root);
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_millis(700))
        .warm_up_time(Duration::from_millis(200))
        .sample_size(20);
    targets = bench_retag_noun_tree
}
criterion_main!(benches);
