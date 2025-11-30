use std::{fs, path::PathBuf, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use lazy_static::lazy_static;
use nockvm::ext::NounExt;
use nockvm::mem::NockStack;
use nockvm::noun::{Cell, IndirectAtom, Noun, D};
use rand::{rngs::StdRng, Rng, SeedableRng};

const STACK_WORDS: usize = 1 << 24; // 128 MiB arena
const KERNEL_STACK_WORDS: usize = 1 << 27; // 1 GiB arena for kernel
const LEAF_COUNT: usize = 5_120;
const RANDOM_SEED_BASE: u64 = 0x5eed_ba11_0000_0000;
const DUMB_JAM_PATH: &str = "../../../../assets/dumb.jam";

#[derive(Clone, Copy)]
enum TreeShape {
    Balanced,
    RightAssoc,
    Random(u64),
}

fn make_stack() -> NockStack {
    let stack = NockStack::new(STACK_WORDS, 0);
    stack.install_arena();
    stack
}

fn make_kernel_stack() -> NockStack {
    let stack = NockStack::new(KERNEL_STACK_WORDS, 0);
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

fn build_tree_balanced(stack: &mut NockStack, mut leaves: Vec<Noun>) -> Noun {
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

fn build_tree_right_assoc(stack: &mut NockStack, leaves: Vec<Noun>) -> Noun {
    assert!(!leaves.is_empty());
    let mut iter = leaves.into_iter().rev();
    let mut acc = iter
        .next()
        .expect("at least one element when building right-associated tree");
    for leaf in iter {
        acc = Cell::new(stack, leaf, acc).as_noun();
    }
    acc
}

fn build_tree_random(stack: &mut NockStack, leaves: Vec<Noun>, seed: u64) -> Noun {
    assert!(!leaves.is_empty());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut nodes = leaves;
    while nodes.len() > 1 {
        let i = rng.gen_range(0..nodes.len());
        let a = nodes.swap_remove(i);
        let j = rng.gen_range(0..nodes.len());
        let b = nodes.swap_remove(j);
        let cell = Cell::new(stack, a, b).as_noun();
        nodes.push(cell);
    }
    nodes.pop().expect("non-empty tree")
}

fn build_tree_with_shape(stack: &mut NockStack, leaves: Vec<Noun>, shape: TreeShape) -> Noun {
    match shape {
        TreeShape::Balanced => build_tree_balanced(stack, leaves),
        TreeShape::RightAssoc => build_tree_right_assoc(stack, leaves),
        TreeShape::Random(seed) => build_tree_random(stack, leaves, seed),
    }
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

lazy_static! {
    static ref DUMB_JAM_BYTES: Vec<u8> = {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DUMB_JAM_PATH);
        fs::read(&path).unwrap_or_else(|e| panic!("failed to read {:?}: {}", path, e))
    };
}

fn load_kernel_dumb(stack: &mut NockStack) -> Noun {
    Noun::cue_bytes_slice(stack, &DUMB_JAM_BYTES).expect("cue dumb.jam")
}

fn setup_case<F>(mut make_leaves: F, shape: TreeShape) -> impl FnMut() -> (NockStack, Noun)
where
    F: FnMut(&mut NockStack) -> Vec<Noun>,
{
    move || {
        let mut stack = make_stack();
        let leaves = make_leaves(&mut stack);
        assert!(leaves.len() >= LEAF_COUNT);
        let root = build_tree_with_shape(&mut stack, leaves, shape);
        (stack, root)
    }
}

fn bench_retag_noun_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("retag_noun_tree");
    let kernel_only = std::env::var_os("NOCKCHAIN_KERNEL_ONLY").is_some();
    let skip_kernel = std::env::var_os("NOCKCHAIN_SKIP_KERNEL").is_some();
    let mut cases: Vec<(String, Box<dyn FnMut() -> (NockStack, Noun)>)> = Vec::new();

    if !kernel_only {
        let base_cases: Vec<(&str, fn(&mut NockStack) -> Vec<Noun>)> = vec![
            ("direct_unique", build_unique_direct),
            ("small_indirect_unique", build_unique_small_indirect),
            (
                "mixed_direct_small_indirect",
                build_mixed_direct_small_indirect,
            ),
            ("small_indirect_shared", build_shared_small_indirect),
            ("large_indirect_unique", build_unique_large_indirect),
            ("large_indirect_shared", build_shared_large_indirect),
        ];

        for (case_idx, (label, leaves_fn)) in base_cases.into_iter().enumerate() {
            let shapes = [
                (TreeShape::Balanced, "balanced"),
                (TreeShape::RightAssoc, "right_assoc"),
                (
                    TreeShape::Random(RANDOM_SEED_BASE ^ (case_idx as u64)),
                    "random",
                ),
            ];
            for (shape, suffix) in shapes {
                let name = format!("{label}_{suffix}");
                cases.push((name, Box::new(setup_case(leaves_fn, shape))));
            }
        }
    }

    // Real-world noun: Nockchain kernel from assets/dumb.jam (currently commented out; too slow for CI/local runs)
    // if !skip_kernel {
    //     cases.push((
    //         "kernel_dumb_jam".to_string(),
    //         Box::new(|| {
    //             let mut stack = make_kernel_stack();
    //             let root = load_kernel_dumb(&mut stack);
    //             (stack, root)
    //         }),
    //     ));
    // }

    for (label, setup) in cases.iter_mut() {
        let is_kernel = label == "kernel_dumb_jam";
        let name = label.clone();
        group.bench_function(BenchmarkId::new("retag", name), |b| {
            let batch_size = if is_kernel {
                BatchSize::NumIterations(1)
            } else {
                BatchSize::LargeInput
            };
            b.iter_batched(
                || setup(),
                |(stack, mut root)| {
                    stack.retag_noun_tree(&mut root as *mut Noun);
                    black_box(root);
                },
                batch_size,
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
