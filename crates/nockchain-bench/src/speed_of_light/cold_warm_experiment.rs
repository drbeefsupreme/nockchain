//! Minimal experiment: force-cold the PMA slab via `MADV_PAGEOUT`, then
//! compare the first-pass (cold) vs second-pass (warm) peek behavior.
//!
//! Linux-only. Requires kernel >= 6.15 for `MADV_PAGEOUT` to cover
//! `MAP_SHARED` file-backed mappings.

use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{fs, io};

use nockapp::nockapp::NockApp;
use tempfile::TempDir;

use super::kernel_utils::init_checkpoint_backed_nockapp;
use super::peek_bench::peek_height_result;

const DEFAULT_COLD_ATTEMPTS: usize = 3;

#[derive(Debug, Clone)]
pub struct Vma {
    pub start: usize,
    pub end: usize,
    pub perms: String,
    pub path: PathBuf,
}

impl Vma {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_shared(&self) -> bool {
        self.perms.as_bytes().get(3) == Some(&b's')
    }
}

pub fn read_pma_vmas(work_dir: &Path) -> io::Result<Vec<Vma>> {
    let replay_dir = work_dir.join("replay-pma");
    let replay_dir = fs::canonicalize(&replay_dir).unwrap_or(replay_dir);
    let maps = fs::read_to_string("/proc/self/maps")?;

    let mut out = Vec::new();
    for line in maps.lines() {
        let parts: Vec<&str> = line.splitn(6, ' ').collect();
        if parts.len() < 6 {
            continue;
        }
        let path_str = parts[5].trim_start();
        if path_str.is_empty() {
            continue;
        }
        let path = PathBuf::from(path_str);
        if !path.starts_with(&replay_dir) {
            continue;
        }

        let (start_s, end_s) = parts[0].split_once('-').ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad range: {}", parts[0]),
            )
        })?;
        let start = usize::from_str_radix(start_s, 16)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let end = usize::from_str_radix(end_s, 16)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        out.push(Vma {
            start,
            end,
            perms: parts[1].to_string(),
            path,
        });
    }
    Ok(out)
}

pub fn page_size() -> usize {
    // SAFETY: sysconf is a trivial libc call.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

pub fn resident_pages(vma: &Vma) -> io::Result<(usize, usize)> {
    let ps = page_size();
    let total = vma.len() / ps;
    if total == 0 {
        return Ok((0, 0));
    }
    let mut vec = vec![0u8; total];
    // SAFETY: vma range is taken from /proc/self/maps, so it is valid for the
    // process; vec has `total` bytes as required by mincore.
    let ret = unsafe {
        libc::mincore(
            vma.start as *mut libc::c_void,
            vma.len(),
            vec.as_mut_ptr() as *mut _,
        )
    };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    let resident = vec.iter().filter(|b| **b & 1 == 1).count();
    Ok((resident, total))
}

pub fn madv_pageout(vma: &Vma) -> io::Result<()> {
    // SAFETY: vma is a live file-backed mapping in this process.
    let ret = unsafe {
        libc::madvise(
            vma.start as *mut libc::c_void,
            vma.len(),
            libc::MADV_PAGEOUT,
        )
    };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct ColdAttempt {
    pub attempt: usize,
    pub resident: usize,
    pub total: usize,
}

pub fn own_cgroup_path() -> io::Result<PathBuf> {
    let contents = fs::read_to_string("/proc/self/cgroup")?;
    for line in contents.lines() {
        let parts: Vec<&str> = line.splitn(3, ':').collect();
        if parts.len() == 3 && parts[0] == "0" {
            let rel = parts[2].trim_start_matches('/');
            return Ok(PathBuf::from("/sys/fs/cgroup").join(rel));
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "no cgroup v2 (0::) entry in /proc/self/cgroup",
    ))
}

pub fn cgroup_memory_reclaim(bytes: u64, swappiness: Option<u32>) -> io::Result<PathBuf> {
    let cgroup = own_cgroup_path()?;
    let reclaim_path = cgroup.join("memory.reclaim");
    let cmd = match swappiness {
        Some(s) => format!("{} swappiness={}", bytes, s),
        None => bytes.to_string(),
    };
    fs::write(&reclaim_path, cmd.as_bytes())?;
    Ok(reclaim_path)
}

pub fn force_cold(vma: &Vma, max_attempts: usize) -> io::Result<Vec<ColdAttempt>> {
    let mut history = Vec::with_capacity(max_attempts);
    for attempt in 1..=max_attempts {
        madv_pageout(vma)?;
        let (resident, total) = resident_pages(vma)?;
        history.push(ColdAttempt {
            attempt,
            resident,
            total,
        });
        if resident == 0 {
            break;
        }
    }
    Ok(history)
}

#[derive(Debug, Clone, Copy)]
struct FaultCounters {
    minflt: u64,
    majflt: u64,
}

fn getrusage_self() -> Option<FaultCounters> {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes into the provided rusage struct on success.
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if ret != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    Some(FaultCounters {
        minflt: usage.ru_minflt as u64,
        majflt: usage.ru_majflt as u64,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct PeekRow {
    pub height: u64,
    pub duration_us: u64,
    pub minflt_delta: u64,
    pub majflt_delta: u64,
    pub success: bool,
}

pub async fn peek_measured(nockapp: &mut NockApp, height: u64) -> PeekRow {
    let before = getrusage_self();
    let started = Instant::now();
    let result = peek_height_result(nockapp, height).await;
    let duration_us = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
    let after = getrusage_self();

    let (minflt_delta, majflt_delta) = match (before, after) {
        (Some(b), Some(a)) => (
            a.minflt.saturating_sub(b.minflt),
            a.majflt.saturating_sub(b.majflt),
        ),
        _ => (0, 0),
    };

    PeekRow {
        height,
        duration_us,
        minflt_delta,
        majflt_delta,
        success: result.is_ok(),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PassSummary {
    pub count: usize,
    pub ok: usize,
    pub total_us: u64,
    pub total_minflt: u64,
    pub total_majflt: u64,
}

impl PassSummary {
    fn from_rows(rows: &[PeekRow]) -> Self {
        Self {
            count: rows.len(),
            ok: rows.iter().filter(|r| r.success).count(),
            total_us: rows.iter().map(|r| r.duration_us).sum(),
            total_minflt: rows.iter().map(|r| r.minflt_delta).sum(),
            total_majflt: rows.iter().map(|r| r.majflt_delta).sum(),
        }
    }
}

pub struct ExperimentResults {
    pub work_dir: PathBuf,
    pub vmas_before: Vec<(Vma, (usize, usize))>,
    pub cold_history: Vec<(PathBuf, Vec<ColdAttempt>)>,
    pub vmas_after_cold_force: Vec<(PathBuf, (usize, usize))>,
    pub vmas_after_cgroup_reclaim: Option<Vec<(PathBuf, (usize, usize))>>,
    pub cold_rows: Vec<PeekRow>,
    pub vmas_after_cold_pass: Vec<(PathBuf, (usize, usize))>,
    pub warm_rows: Vec<PeekRow>,
    pub vmas_after_warm_pass: Vec<(PathBuf, (usize, usize))>,
}

pub async fn run_experiment(
    checkpoint: &Path,
    kernel: &Path,
    start_height: u64,
    count: u64,
    fsync: bool,
    cgroup_reclaim_bytes: Option<u64>,
) -> Result<ExperimentResults, Box<dyn std::error::Error>> {
    let temp = TempDir::new()?;
    let work_dir = temp.path().to_path_buf();

    println!("Checkpoint: {}", checkpoint.display());
    println!("Kernel:     {}", kernel.display());
    println!("Work dir:   {}", work_dir.display());
    println!(
        "Range:      heights {}..={}",
        start_height,
        start_height + count - 1
    );
    println!();

    println!("--- Booting NockApp ---");
    let boot_started = Instant::now();
    let mut nockapp = init_checkpoint_backed_nockapp(checkpoint, kernel, &work_dir, fsync).await?;
    println!("Boot took {:.2}s", boot_started.elapsed().as_secs_f64());
    println!();

    let vmas = read_pma_vmas(&work_dir)?;
    if vmas.is_empty() {
        return Err("no PMA VMAs discovered under /proc/self/maps — is the \
                    pma-runtime-compat feature enabled and did the PMA actually mmap?"
            .into());
    }

    println!("--- PMA VMAs (post-boot) ---");
    let mut vmas_before = Vec::with_capacity(vmas.len());
    for vma in &vmas {
        let residency = resident_pages(vma)?;
        println!(
            "  {}  0x{:x}..0x{:x}  len={} MiB  perms={}  resident={}/{} pages",
            vma.path.display(),
            vma.start,
            vma.end,
            vma.len() / (1024 * 1024),
            vma.perms,
            residency.0,
            residency.1,
        );
        vmas_before.push((vma.clone(), residency));
    }
    println!();

    println!("--- Forcing cold (MADV_PAGEOUT + mincore verify) ---");
    let mut cold_history = Vec::with_capacity(vmas.len());
    let mut vmas_after_cold_force = Vec::with_capacity(vmas.len());
    for vma in &vmas {
        let history = force_cold(vma, DEFAULT_COLD_ATTEMPTS)?;
        let final_attempt = history.last().copied();
        for attempt in &history {
            println!(
                "  {}  attempt {}: resident={}/{}",
                vma.path.display(),
                attempt.attempt,
                attempt.resident,
                attempt.total,
            );
        }
        if let Some(final_attempt) = final_attempt {
            vmas_after_cold_force.push((
                vma.path.clone(),
                (final_attempt.resident, final_attempt.total),
            ));
        }
        cold_history.push((vma.path.clone(), history));
    }
    println!();

    let vmas_after_cgroup_reclaim = if let Some(bytes) = cgroup_reclaim_bytes {
        println!(
            "--- cgroup v2 memory.reclaim ({} bytes, swappiness=0) ---",
            bytes
        );
        match cgroup_memory_reclaim(bytes, Some(0)) {
            Ok(path) => {
                println!("  wrote to {}", path.display());
            }
            Err(e) => {
                println!(
                    "  reclaim write failed (raw_os_error={:?}): {}",
                    e.raw_os_error(),
                    e
                );
                if e.raw_os_error() == Some(libc::EAGAIN) {
                    println!("  (EAGAIN = kernel under-reclaimed; pages may still have moved)");
                }
            }
        }
        Some(collect_residencies(&vmas, "post cgroup-reclaim")?)
    } else {
        None
    };
    if vmas_after_cgroup_reclaim.is_some() {
        println!();
    }

    let end_height = start_height + count - 1;

    println!(
        "--- Cold peek pass (heights {}..={}) ---",
        start_height, end_height
    );
    println!("height,duration_us,minflt_delta,majflt_delta,success");
    let mut cold_rows = Vec::with_capacity(count as usize);
    for height in start_height..=end_height {
        let row = peek_measured(&mut nockapp, height).await;
        println!(
            "{},{},{},{},{}",
            row.height, row.duration_us, row.minflt_delta, row.majflt_delta, row.success
        );
        cold_rows.push(row);
    }
    println!();

    let vmas_after_cold_pass = collect_residencies(&vmas, "post cold-pass")?;
    println!();

    println!(
        "--- Warm peek pass (heights {}..={}) ---",
        start_height, end_height
    );
    println!("height,duration_us,minflt_delta,majflt_delta,success");
    let mut warm_rows = Vec::with_capacity(count as usize);
    for height in start_height..=end_height {
        let row = peek_measured(&mut nockapp, height).await;
        println!(
            "{},{},{},{},{}",
            row.height, row.duration_us, row.minflt_delta, row.majflt_delta, row.success
        );
        warm_rows.push(row);
    }
    println!();

    let vmas_after_warm_pass = collect_residencies(&vmas, "post warm-pass")?;
    println!();

    print_summary(&cold_rows, &warm_rows);

    Ok(ExperimentResults {
        work_dir,
        vmas_before,
        cold_history,
        vmas_after_cold_force,
        vmas_after_cgroup_reclaim,
        cold_rows,
        vmas_after_cold_pass,
        warm_rows,
        vmas_after_warm_pass,
    })
}

fn collect_residencies(vmas: &[Vma], label: &str) -> io::Result<Vec<(PathBuf, (usize, usize))>> {
    println!("--- PMA residency ({}) ---", label);
    let mut out = Vec::with_capacity(vmas.len());
    for vma in vmas {
        let residency = resident_pages(vma)?;
        println!(
            "  {}  resident={}/{} pages",
            vma.path.display(),
            residency.0,
            residency.1,
        );
        out.push((vma.path.clone(), residency));
    }
    Ok(out)
}

fn print_summary(cold: &[PeekRow], warm: &[PeekRow]) {
    let cold_sum = PassSummary::from_rows(cold);
    let warm_sum = PassSummary::from_rows(warm);
    println!("--- Summary ---");
    println!(
        "Cold pass: {} peeks ({} ok) total {:.3} ms, Σminflt={}, Σmajflt={}",
        cold_sum.count,
        cold_sum.ok,
        cold_sum.total_us as f64 / 1000.0,
        cold_sum.total_minflt,
        cold_sum.total_majflt,
    );
    println!(
        "Warm pass: {} peeks ({} ok) total {:.3} ms, Σminflt={}, Σmajflt={}",
        warm_sum.count,
        warm_sum.ok,
        warm_sum.total_us as f64 / 1000.0,
        warm_sum.total_minflt,
        warm_sum.total_majflt,
    );
    if warm_sum.total_us > 0 {
        let speedup = cold_sum.total_us as f64 / warm_sum.total_us as f64;
        println!("Cold/Warm wall-time ratio: {:.2}x", speedup);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vma_shared_flag_reads_fourth_perm_byte() {
        let vma = Vma {
            start: 0x1000,
            end: 0x2000,
            perms: "rw-s".to_string(),
            path: PathBuf::from("/tmp/a"),
        };
        assert!(vma.is_shared());
        assert_eq!(vma.len(), 0x1000);
    }

    #[test]
    fn vma_private_flag_rejects_shared() {
        let vma = Vma {
            start: 0,
            end: 1,
            perms: "rw-p".to_string(),
            path: PathBuf::from("/tmp/a"),
        };
        assert!(!vma.is_shared());
    }
}
