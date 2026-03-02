use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use nockchain_bench::speed_of_light::{read_fixture_file, SolFixtureFile};

const FIXTURE_MAGIC: &[u8; 8] = b"SOLTEST\0";

#[derive(Debug, Parser)]
#[command(about = "Convert a SOL fixture (.soltest) to legacy v1 format")]
struct Args {
    /// Input fixture path (v1 or v2)
    #[arg(long)]
    input: PathBuf,

    /// Output fixture path (legacy v1)
    #[arg(long)]
    output: PathBuf,
}

fn write_legacy_fixture(path: &PathBuf, fixture: &SolFixtureFile) -> Result<(), String> {
    let payload =
        bincode::serialize(fixture).map_err(|e| format!("failed to serialize fixture: {e}"))?;
    let file = File::create(path)
        .map_err(|e| format!("failed to create output fixture '{}': {e}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(FIXTURE_MAGIC)
        .map_err(|e| format!("failed to write fixture magic: {e}"))?;
    writer
        .write_all(&1u16.to_le_bytes())
        .map_err(|e| format!("failed to write legacy header: {e}"))?;
    writer
        .write_all(&(payload.len() as u64).to_le_bytes())
        .map_err(|e| format!("failed to write legacy payload length: {e}"))?;
    writer
        .write_all(&payload)
        .map_err(|e| format!("failed to write legacy payload bytes: {e}"))?;
    writer
        .flush()
        .map_err(|e| format!("failed to flush output fixture: {e}"))?;
    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let fixture = read_fixture_file(&args.input)
        .map_err(|e| format!("failed to read input fixture '{}': {e}", args.input.display()))?;

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create output parent directory '{}': {e}",
                parent.display()
            )
        })?;
    }
    write_legacy_fixture(&args.output, &fixture)?;

    println!("Converted fixture to v1:");
    println!("  Input:  {}", args.input.display());
    println!("  Output: {}", args.output.display());
    Ok(())
}
