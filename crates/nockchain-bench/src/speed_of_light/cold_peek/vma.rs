use std::path::{Path, PathBuf};
use std::{fs, io};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vma {
    pub start: usize,
    pub end: usize,
    pub perms: String,
    pub path: PathBuf,
}

impl Vma {
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_shared(&self) -> bool {
        self.perms.as_bytes().get(3) == Some(&b's')
    }
}

pub fn read_pma_vmas(work_dir: &Path) -> io::Result<Vec<Vma>> {
    let replay_dir = work_dir.join("replay-pma");
    let replay_dir = fs::canonicalize(&replay_dir).unwrap_or(replay_dir);
    let maps = fs::read_to_string("/proc/self/maps")?;
    parse_proc_maps(&maps, &replay_dir)
}

pub fn parse_proc_maps(contents: &str, replay_dir: &Path) -> io::Result<Vec<Vma>> {
    let mut out = Vec::new();
    for line in contents.lines() {
        let mut parts = line.split_whitespace();
        let Some(range) = parts.next() else {
            continue;
        };
        let Some(perms) = parts.next() else {
            continue;
        };
        let _offset = parts.next();
        let _dev = parts.next();
        let _inode = parts.next();
        let Some(path_str) = parts.next() else {
            continue;
        };

        let path = PathBuf::from(path_str);
        if !path.starts_with(replay_dir) {
            continue;
        }

        let (start_s, end_s) = range.split_once('-').ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("bad range: {range}"))
        })?;
        let start = usize::from_str_radix(start_s, 16)
            .map_err(|source| io::Error::new(io::ErrorKind::InvalidData, source))?;
        let end = usize::from_str_radix(end_s, 16)
            .map_err(|source| io::Error::new(io::ErrorKind::InvalidData, source))?;

        out.push(Vma {
            start,
            end,
            perms: perms.to_string(),
            path,
        });
    }
    Ok(out)
}

pub fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

pub fn reduce_mincore_bitmap(bitmap: &[u8]) -> (usize, usize) {
    let resident = bitmap.iter().filter(|byte| (**byte & 1) == 1).count();
    (resident, bitmap.len())
}

pub fn resident_pages(vma: &Vma) -> io::Result<(usize, usize)> {
    let ps = page_size();
    let total_pages = vma.len() / ps;
    if total_pages == 0 {
        return Ok((0, 0));
    }

    let mut bitmap = vec![0u8; total_pages];
    let ret = unsafe {
        libc::mincore(
            vma.start as *mut libc::c_void,
            vma.len(),
            bitmap.as_mut_ptr() as *mut _,
        )
    };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(reduce_mincore_bitmap(&bitmap))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_replay_pma_vmas() {
        let replay_dir = PathBuf::from("/tmp/work/replay-pma");
        let maps = "\
7f0a0000-7f0a1000 rw-s 00000000 00:00 0 /tmp/work/replay-pma/slab-0.bin\n\
7f0a1000-7f0a2000 rw-p 00000000 00:00 0 /tmp/work/replay-pma/slab-1.bin\n\
7f0a2000-7f0a3000 rw-s 00000000 00:00 0 /tmp/work/elsewhere.bin\n";

        let parsed = parse_proc_maps(maps, &replay_dir).expect("parse maps");

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].start, 0x7f0a0000);
        assert_eq!(parsed[0].end, 0x7f0a1000);
        assert!(parsed[0].is_shared());
        assert!(!parsed[1].is_shared());
    }

    #[test]
    fn mincore_bitmap_reduction_counts_only_low_bit() {
        let (resident, total) = reduce_mincore_bitmap(&[0, 1, 2, 3, 4, 5]);
        assert_eq!(resident, 3);
        assert_eq!(total, 6);
    }
}
