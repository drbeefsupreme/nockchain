# nockchain-bench CLI Commands

This document lists every CLI command and option available in `nockchain-bench`, plus usage examples.

## Global

```
nockchain-bench [COMMAND] [OPTIONS]
```

Global flags (provided by Clap):
- `-h`, `--help`: show help
- `-V`, `--version`: show version

## Commands

### 1) `sample`
Sample memory usage of a running process.

```
nockchain-bench sample <pid|self> [--nockstack-size <bytes>]
```

Arguments:
- `pid`: process ID, or `self` to sample the current process

Options:
- `--nockstack-size <bytes>`: expected NockStack size in bytes (optional)

Example:
```
# sample current process
nockchain-bench sample self

# sample PID 12345 with known NockStack size
nockchain-bench sample 12345 --nockstack-size 17179869184
```

---

### 2) `run`
Run a mining benchmark scenario in Docker.

```
nockchain-bench run [OPTIONS]
```

Options:
- `-n`, `--name <string>`: scenario name (default: `benchmark`)
- `-m`, `--mode <checkpoint|pma-persist>`: persistence mode (default: `checkpoint`)
- `--save-interval <seconds>`: checkpoint save interval (default: `120`)
- `-d`, `--duration <seconds>`: duration to run (default: `300`)
- `--sample-interval <seconds>`: sample interval (default: `1`)
- `--image <string>`: Docker image (default: `nockchain-local:latest`)
- `--data-dir <path>`: data directory on host (default: `/tmp/nockchain-bench`)
- `--memory-limit <string>`: memory limit (default: `16g`)
- `--threads <u32>`: mining threads (default: `1`)
- `-o`, `--output <path>`: output directory (optional)
- `--format <text|json|parquet>`: output format (default: `text`)

Example:
```
# run a 5-minute checkpoint-mode scenario
nockchain-bench run --duration 300 --mode checkpoint --name bench-5m
```

---

### 3) `attach`
Attach to an existing container and collect stats.

```
nockchain-bench attach <container> [OPTIONS]
```

Arguments:
- `container`: container name or ID

Options:
- `-d`, `--duration <seconds>`: duration to collect stats (default: `60`)
- `--sample-interval <seconds>`: sample interval (default: `1`)
- `-o`, `--output <path>`: output directory (optional)
- `--format <text|json|parquet>`: output format (default: `text`)

Example:
```
nockchain-bench attach nockchain-node-1 --duration 120 --format json
```

---

### 4) `compare`
Run A/B comparison between checkpoint and PMA persist modes.

```
nockchain-bench compare [OPTIONS]
```

Options:
- `-d`, `--duration <seconds>`: duration per scenario (default: `300`)
- `--sample-interval <seconds>`: sample interval (default: `1`)
- `--save-interval <seconds>`: checkpoint save interval (default: `120`)
- `--image <string>`: Docker image (default: `nockchain-local:latest`)
- `--data-dir <path>`: base data directory (default: `/tmp/nockchain-bench`)
- `--memory-limit <string>`: memory limit (default: `16g`)
- `--threads <u32>`: mining threads (default: `1`)
- `-o`, `--output <path>`: output directory (optional)

Example:
```
nockchain-bench compare --duration 180 --memory-limit 8g
```

---

### 5) `analyze`
Analyze a container with event correlation.

```
nockchain-bench analyze <container> [OPTIONS]
```

Arguments:
- `container`: container name or ID

Options:
- `-d`, `--duration <seconds>`: duration to collect stats (default: `30`)
- `--sample-interval <seconds>`: sample interval (default: `1`)
- `--spike-threshold <pct>`: memory spike threshold percent (default: `5.0`)
- `--all-events`: show all events, not just significant ones

Example:
```
nockchain-bench analyze nockchain-node-1 --duration 45 --spike-threshold 3.0
```

---

## Speed-of-light Subcommands (`sol`)

```
nockchain-bench sol <extract|bench|checkpoint|inspect> [OPTIONS]
```

### 6) `sol extract`
Extract blocks from a checkpoint to a `.solarch` archive.

```
nockchain-bench sol extract [OPTIONS]
```

Options:
- `-n`, `--blocks <u64>`: number of blocks to extract (default: `1000`)
- `-c`, `--checkpoint <path>`: checkpoint path (default: `0.chkjam`)
- `-k`, `--kernel <path>`: kernel jam path (default: `assets/dumb.jam`)
- `-o`, `--output <path>`: output archive path (default: `blocks_<N>.solarch`)
- `--chunk-size <u64>`: chunk size for range queries (default: `8`)
- `--include-mempool`: include per-block mempool snapshots (default: off)

Example:
```
nockchain-bench sol extract -n 1000 -c 0.chkjam -k assets/dumb.jam

# include per-block mempool snapshots
nockchain-bench sol extract -n 1000 -c 0.chkjam -k assets/dumb.jam --include-mempool
```

Notes:
- Mempool snapshots store `tx_id` and `heard_at` for each height.
- The inspector below requires archives created with `--include-mempool`.

---

### 7) `sol bench`
Run speed-of-light benchmark (poke blocks as fast as possible).

```
nockchain-bench sol bench [OPTIONS]
```

Options:
- `-a`, `--archive <path>`: archive path (default: `blocks_1000.solarch`)
- `-k`, `--kernel <path>`: kernel jam path (default: `assets/dumb.jam`)
- `-n`, `--blocks <u64>`: blocks to benchmark, `0` = all (default: `0`)
- `--skip-genesis`: skip genesis block (default: off)
- `--proof-version <v0|v1|v2>`: filter by proof version (optional)
- `--checkpoint <path>`: load an existing checkpoint before benchmarking (optional)
- `--start-height <u64>`: start height override; defaults to checkpoint height + 1 if checkpoint provided

Examples:
```
# run full benchmark from archive
nockchain-bench sol bench -a blocks_full.solarch -k assets/dumb.jam

# run only v1 blocks
nockchain-bench sol bench --archive blocks_full.solarch --proof-version v1

# run from a checkpoint (start height defaults to checkpoint height + 1)
nockchain-bench sol bench --archive blocks_full.solarch --checkpoint checkpoint_at_v1_crossover.chkjam
```

---

### 8) `sol checkpoint`
Build a single checkpoint by replaying archive blocks up to a target height.

```
nockchain-bench sol checkpoint [OPTIONS]
```

Options:
- `-a`, `--archive <path>`: archive path (default: `blocks_1000.solarch`)
- `-k`, `--kernel <path>`: kernel jam path (default: `assets/dumb.jam`)
- `--checkpoint <path>`: existing checkpoint to start from (optional)
- `--target-height <u64>`: target block height (inclusive)
- `--cutover <v1|v2>`: cutover to build checkpoint for (mutually exclusive with `--target-height`)
- `--start-height <u64>`: start height override; defaults to checkpoint height + 1 if checkpoint provided
- `-o`, `--output <path>`: output checkpoint file (default: `checkpoint_at_vN_crossover.chkjam` or `checkpoint_at_height_<H>.chkjam`)
- `--work-dir <path>`: working directory for snapshot files (default: temp dir)

Examples:
```
# build v1 crossover checkpoint (height 6,749)
nockchain-bench sol checkpoint --archive blocks_full.solarch --cutover v1 --output checkpoint_at_v1_crossover.chkjam

# build v2 crossover checkpoint (height 11,999)
nockchain-bench sol checkpoint --archive blocks_full.solarch --cutover v2 --output checkpoint_at_v2_crossover.chkjam

# build checkpoint at explicit height 5000
nockchain-bench sol checkpoint --archive blocks_full.solarch --target-height 5000

# chain from an existing checkpoint and continue to v2 crossover
nockchain-bench sol checkpoint --archive blocks_full.solarch --checkpoint checkpoint_at_v1_crossover.chkjam --cutover v2 --output checkpoint_at_v2_crossover.chkjam
```

---

### 9) `sol inspect`
Inspect mempool snapshots for stale transactions (age >= retain).

```
nockchain-bench sol inspect [OPTIONS]
```

Options:
- `-a`, `--archive <path>`: archive path (default: `blocks_1000.solarch`)
- `--retain <u64>`: retention threshold in blocks (default: `20`)

Example:
```
# report transactions stale for 20+ blocks
nockchain-bench sol inspect --archive blocks_full.solarch --retain 20
```

Notes:
- Errors if the archive was created without mempool snapshots.
