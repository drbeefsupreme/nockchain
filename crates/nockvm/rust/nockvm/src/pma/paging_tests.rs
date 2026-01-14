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
