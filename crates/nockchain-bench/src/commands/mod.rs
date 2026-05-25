pub mod sol;

use std::io::Read;
use std::path::{Path, PathBuf};

use clap::ValueEnum;

#[derive(Clone, Debug, ValueEnum)]
pub enum CutoverVersion {
    V1,
    V2,
}

pub fn ensure_existing_file(path: &Path, label: &str) -> Result<(), Box<dyn std::error::Error>> {
    if path.exists() {
        return Ok(());
    }
    Err(format!("{label} file not found: {}", path.display()).into())
}

pub fn create_timestamped_subdir(
    base: &Path,
    prefix: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dir = base.join(format!(
        "{prefix}-{}-{}",
        std::process::id(),
        unix_time_millis()
    ));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

pub fn unix_time_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

pub fn print_heading(title: &str) {
    println!("=== {title} ===\n");
}

pub fn print_heading_with_leading_newline(title: &str) {
    println!("\n=== {title} ===\n");
}

pub fn on_or_off(enabled: bool) -> &'static str {
    if enabled {
        "on"
    } else {
        "off"
    }
}

pub fn included_or_off(enabled: bool) -> &'static str {
    if enabled {
        "included"
    } else {
        "off"
    }
}

pub fn all_or_number(value: u64) -> String {
    if value == 0 {
        "all".to_string()
    } else {
        value.to_string()
    }
}

pub fn blake3_hash_hex_for_file(path: &Path) -> Result<String, std::io::Error> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}
