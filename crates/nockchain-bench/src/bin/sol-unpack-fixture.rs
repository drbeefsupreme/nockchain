use std::path::PathBuf;

use clap::Parser;
use nockchain_bench::speed_of_light::read_fixture_file;

#[derive(Debug, Parser)]
#[command(name = "sol-unpack-fixture")]
#[command(about = "Unpack a .soltest fixture into checkpoint/archive/kernel files")]
struct Args {
    /// Fixture (.soltest) to unpack
    #[arg(short, long)]
    fixture: PathBuf,

    /// Output directory for extracted files
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Filename for checkpoint output
    #[arg(long, default_value = "fixture.chkjam")]
    checkpoint_name: String,

    /// Filename for archive output
    #[arg(long, default_value = "fixture.solarch")]
    archive_name: String,

    /// Filename for kernel output
    #[arg(long, default_value = "fixture.jam")]
    kernel_name: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let fixture = read_fixture_file(&args.fixture)?;
    std::fs::create_dir_all(&args.output_dir)?;

    let checkpoint_path = args.output_dir.join(&args.checkpoint_name);
    let archive_path = args.output_dir.join(&args.archive_name);
    let kernel_path = args.output_dir.join(&args.kernel_name);

    std::fs::write(&checkpoint_path, &fixture.checkpoint_bytes)?;
    std::fs::write(&archive_path, &fixture.archive_bytes)?;
    std::fs::write(&kernel_path, &fixture.kernel_bytes)?;

    println!("Checkpoint: {}", checkpoint_path.display());
    println!("Archive:    {}", archive_path.display());
    println!("Kernel:     {}", kernel_path.display());
    println!(
        "Range:      {}..={}",
        fixture.manifest.archive_start_height.as_u64(),
        fixture.manifest.archive_end_height.as_u64()
    );

    Ok(())
}
