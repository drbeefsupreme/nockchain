use std::path::PathBuf;

use clap::Parser;
use nockchain_bench::speed_of_light::checkpoint::checkpoint_event_num;
use nockchain_bench::speed_of_light::{
    write_fixture_file, ArchiveReader, SolFixtureFile, SolFixtureManifest, SolHeight,
};

#[derive(Debug, Parser)]
#[command(name = "sol-pack-fixture")]
#[command(about = "Package checkpoint/archive/kernel files into a .soltest fixture")]
struct Args {
    /// Derived checkpoint file to embed
    #[arg(long)]
    checkpoint: PathBuf,

    /// Archive (.solarch) file to embed
    #[arg(long)]
    archive: PathBuf,

    /// Kernel jam file to embed
    #[arg(long)]
    kernel: PathBuf,

    /// Output fixture path
    #[arg(short, long)]
    output: PathBuf,

    /// Source checkpoint path to record in fixture metadata
    #[arg(long, default_value = "0.chkjam")]
    source_checkpoint_path: String,

    /// Optional explicit source checkpoint event number
    #[arg(long)]
    source_event_num: Option<u64>,

    /// Chunk size metadata to record in fixture manifest
    #[arg(long, default_value_t = 8)]
    chunk_size: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if !args.checkpoint.exists() {
        return Err(format!("Checkpoint not found: {}", args.checkpoint.display()).into());
    }
    if !args.archive.exists() {
        return Err(format!("Archive not found: {}", args.archive.display()).into());
    }
    if !args.kernel.exists() {
        return Err(format!("Kernel not found: {}", args.kernel.display()).into());
    }

    let checkpoint_bytes = std::fs::read(&args.checkpoint)?;
    let archive_bytes = std::fs::read(&args.archive)?;
    let kernel_bytes = std::fs::read(&args.kernel)?;

    let reader = ArchiveReader::from_file(&args.archive)?;
    let meta = reader.metadata();
    if meta.block_count == 0 {
        return Err("Archive contains zero blocks".into());
    }

    let archive_start = meta.min_height;
    let archive_end = meta.max_height;
    let derived_checkpoint_height = SolHeight(archive_start.as_u64().saturating_sub(1));

    let derived_checkpoint_event_num = checkpoint_event_num(&args.checkpoint)?;
    let source_checkpoint_event_num = match args.source_event_num {
        Some(value) => value,
        None => checkpoint_event_num(&args.source_checkpoint_path)?,
    };

    let fixture = SolFixtureFile {
        manifest: SolFixtureManifest {
            format_version: 1,
            source_checkpoint_path: args.source_checkpoint_path.clone(),
            source_checkpoint_event_num,
            derived_checkpoint_height,
            derived_checkpoint_event_num,
            archive_start_height: archive_start,
            archive_end_height: archive_end,
            include_mempool: meta.has_mempool,
            chunk_size: args.chunk_size,
            kernel_hash_hex: blake3::hash(&kernel_bytes).to_hex().to_string(),
            checkpoint_hash_hex: blake3::hash(&checkpoint_bytes).to_hex().to_string(),
            archive_hash_hex: blake3::hash(&archive_bytes).to_hex().to_string(),
        },
        checkpoint_bytes,
        archive_bytes,
        kernel_bytes,
    };

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_fixture_file(&args.output, &fixture)?;

    println!("Fixture: {}", args.output.display());
    println!(
        "Range:   {}..={}",
        archive_start.as_u64(),
        archive_end.as_u64()
    );
    println!("Blocks:  {}", meta.block_count);
    println!(
        "Derived checkpoint height/event: {} / {}",
        derived_checkpoint_height.as_u64(),
        derived_checkpoint_event_num
    );

    Ok(())
}
