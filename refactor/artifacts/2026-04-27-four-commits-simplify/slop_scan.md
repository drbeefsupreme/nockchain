# AI slop scan — 2026-04-27-four-commits-simplify

Generated 2026-04-27T19:38:52Z
Scope: `crates/nockchain-bench/src/speed_of_light`

(See references/VIBE-CODED-PATHOLOGIES.md for P1-P40 catalog.)


## P1 over-defensive try/catch (Python: ≥3 except Exception per file)

_none found_

## P1 over-defensive try/catch (TS: catch blocks per file)

_none found_

## P2 long nullish/optional chains (three+ `?.`)

_none found_

## P2 double-nullish coalescing

_none found_

## P3 orphaned _v2/_new/_old/_improved/_copy files

_none found_

## P4 utils/helpers/misc/common files > 500 LOC

_none found_

## P5 abstract Base/Abstract class hierarchy

_none found_

## P5 abstract class in Rust (rare idiom; often AI-generated)

_none found_

## P6 feature flags (review each for whether it is still toggling)

_none found_

## P7 re-export barrel files (`export * from`)

_none found_

## P8 pass-through wrappers (function whose sole body returns another call)

_none found_

## P9 functions with ≥5 optional parameters

_none found_

## P10 swallowed catch (empty or `return null`)

_none found_

## P10 Python: except ... : pass

_none found_

## P11 Step/Phase/TODO comments (per-file counts)

_none found_

## P12 many-import files (top 20)

_none found_

## P14 mocks (jest.mock, vi.mock, sinon.stub, __mocks__)

_none found_

## P15 TS `any` usage (per-file counts, top 20)

_none found_

## P16 *Error enums in Rust (often duplicate variants)

```
crates/nockchain-bench/src/speed_of_light/bench.rs:30:pub enum BenchError {
crates/nockchain-bench/src/speed_of_light/fixture.rs:63:pub enum FixtureError {
crates/nockchain-bench/src/speed_of_light/profiling.rs:241:pub enum MemorySamplerError {
crates/nockchain-bench/src/speed_of_light/poke.rs:60:pub enum PokeStepError {
crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs:23:pub enum CheckpointBuildError {
crates/nockchain-bench/src/speed_of_light/extractor.rs:65:pub enum ExtractorError {
crates/nockchain-bench/src/speed_of_light/start_height.rs:8:pub enum StartHeightError {
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:52:pub enum HarnessDockerError {
crates/nockchain-bench/src/speed_of_light/checkpoint.rs:10:pub enum CheckpointLoadError {
crates/nockchain-bench/src/speed_of_light/checkpoint.rs:22:pub enum CheckpointMetaError {
crates/nockchain-bench/src/speed_of_light/cold_peek/mod.rs:69:pub enum ColdStepError {
crates/nockchain-bench/src/speed_of_light/cold_peek/mod.rs:86:pub enum ColdInitError {
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:507:pub enum PlanValidationError {
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:643:pub enum PreRunError {
crates/nockchain-bench/src/speed_of_light/archive.rs:92:pub enum ArchiveError {
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:26:pub enum PeekBenchError {
crates/nockchain-bench/src/speed_of_light/harness/mod.rs:61:pub enum HarnessError {
crates/nockchain-bench/src/speed_of_light/mempool_inspector.rs:12:pub enum InspectorError {
crates/nockchain-bench/src/speed_of_light/kernel_utils.rs:27:pub enum KernelInitError {
crates/nockchain-bench/src/speed_of_light/kernel_utils.rs:39:pub enum CheckpointBackedInitError {
crates/nockchain-bench/src/speed_of_light/kernel_utils.rs:48:pub enum PeekChainError {
```

## P17 heavily drilled props (top 10 most-passed via JSX)

_none found_

## P18 everything hook (custom hook file with many useState/useEffect)

_none found_

## P19 N+1 pattern (await inside for loop)

_none found_

## P19 Python N+1 (for ... : await)

_none found_

## P20 config files (candidates for unification)

```
./.env
./.worktrees/pma-bench-shim-verify/.env_example
./.worktrees/pma-bench-shim-verify/docker-compose.metrics.yml
./.worktrees/bench-harness-samply-sweep-20260317/.env_example
./.worktrees/pma-smoke-quickread-20260414/.env_example
./.worktrees/pma-smoke-quickread-20260414/docker-compose.metrics.yml
./.worktrees/quick-read-bench-task1-red/.env_example
./.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure/.env_example
./.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure/docker-compose.metrics.yml
./.worktrees/pma-bench-fsync-target/.env_example
```

## P22 stringly-typed status/state comparisons

_none found_

## P22 Rust stringly-typed status/state comparisons

_none found_

## P23 reflex trim/lower/upper normalization

```
crates/nockchain-bench/src/speed_of_light/cold_peek/cgroup.rs:571:        if !contents.trim().is_empty() {
crates/nockchain-bench/src/speed_of_light/harness/mod.rs:107:    let text = text.trim();
crates/nockchain-bench/src/speed_of_light/harness/mod.rs:116:    let value = value.trim();
crates/nockchain-bench/src/speed_of_light/harness/provenance.rs:258:        .map(|output| !String::from_utf8_lossy(&output.stdout).trim().is_empty())
crates/nockchain-bench/src/speed_of_light/harness/provenance.rs:279:    let text = text.trim();
crates/nockchain-bench/src/speed_of_light/harness/provenance.rs:306:            return Some(model.trim().to_string());
crates/nockchain-bench/src/speed_of_light/harness/docker_image.rs:103:    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
crates/nockchain-bench/src/speed_of_light/harness/docker_image.rs:108:    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
crates/nockchain-bench/src/speed_of_light/harness/docker_image.rs:142:            String::from_utf8_lossy(&output.stderr).trim()
crates/nockchain-bench/src/speed_of_light/harness/docker_image.rs:145:    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
crates/nockchain-bench/src/speed_of_light/harness/docker_image.rs:177:    if entry.id.trim().is_empty() {
crates/nockchain-bench/src/speed_of_light/harness/docker_image.rs:191:                .find(|value| !value.trim().is_empty())
crates/nockchain-bench/src/speed_of_light/harness/docker_image.rs:213:    if requested_ref.trim().is_empty() {
crates/nockchain-bench/src/speed_of_light/harness/case.rs:145:        .filter(|profile| !profile.trim().is_empty())
crates/nockchain-bench/src/speed_of_light/harness/case.rs:257:        DockerImageSource::Provided { reference } if reference.trim().is_empty() => {
crates/nockchain-bench/src/speed_of_light/harness/case.rs:262:        DockerImageSource::AutoBuild { tag } if tag.trim().is_empty() => {
crates/nockchain-bench/src/speed_of_light/harness/case.rs:278:        .is_some_and(|cpuset| cpuset.trim().is_empty())
crates/nockchain-bench/src/speed_of_light/harness/artifacts.rs:107:            .filter(|line| !line.trim().is_empty())
crates/nockchain-bench/src/speed_of_light/harness/profiler.rs:161:    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
crates/nockchain-bench/src/speed_of_light/harness/profiler.rs:166:    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
crates/nockchain-bench/src/speed_of_light/harness/profiler.rs:576:                assert!(!message.trim().is_empty());
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:470:        .trim()
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:538:        .trim()
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:1184:            String::from_utf8_lossy(&output.stderr).trim()
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:1187:    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:1473:    pid.trim().parse::<u32>().ok()
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:1582:            String::from_utf8_lossy(&output.stderr).trim()
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:1585:    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:1651:    let stat = stat.trim();
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:1667:    let value = value.trim().to_lowercase();
```

## P24 testability wrappers / mutable deps seams

_none found_

## P25 docstrings/comments that may contradict implementation

_none found_

## P26 TypeScript type assertions

_none found_

## P27 addEventListener sites (audit for cleanup)

_none found_

## P28 timers (audit for clearTimeout/clearInterval cleanup)

_none found_

## P29 regex construction in functions/loops

_none found_

## P30 debug print/log leftovers

```
crates/nockchain-bench/src/speed_of_light/bench.rs:190:        println!("\n=== Benchmark Results ===\n");
crates/nockchain-bench/src/speed_of_light/bench.rs:191:        println!("Blocks poked:    {}", self.blocks_poked);
crates/nockchain-bench/src/speed_of_light/bench.rs:192:        println!("Failed pokes:    {}", self.failed_pokes);
crates/nockchain-bench/src/speed_of_light/bench.rs:193:        println!("Init time:       {:.2}s", self.init_time.as_secs_f64());
crates/nockchain-bench/src/speed_of_light/bench.rs:194:        println!(
crates/nockchain-bench/src/speed_of_light/bench.rs:198:        println!(
crates/nockchain-bench/src/speed_of_light/bench.rs:202:        println!("Throughput:      {:.2} blocks/s", self.blocks_per_second());
crates/nockchain-bench/src/speed_of_light/bench.rs:203:        println!("Checkpoints:     {}", self.checkpoint_count);
crates/nockchain-bench/src/speed_of_light/bench.rs:205:            println!("Avg checkpoint:  {:.2}s", avg.as_secs_f64());
crates/nockchain-bench/src/speed_of_light/bench.rs:209:            println!("\n=== Memory Profile ===\n");
crates/nockchain-bench/src/speed_of_light/bench.rs:210:            println!("Samples:         {}", profile.samples.len());
crates/nockchain-bench/src/speed_of_light/bench.rs:211:            println!("GC events:       {}", profile.gc_events.len());
crates/nockchain-bench/src/speed_of_light/bench.rs:212:            println!("Fault bursts:    {}", profile.page_fault_bursts.len());
crates/nockchain-bench/src/speed_of_light/bench.rs:213:            println!("Peak RSS:        {:.2} MiB", profile.scorecard.peak_rss_mib);
crates/nockchain-bench/src/speed_of_light/bench.rs:214:            println!("P95 RSS:         {:.2} MiB", profile.scorecard.p95_rss_mib);
crates/nockchain-bench/src/speed_of_light/bench.rs:216:                println!("Ckpt peak RSS:   {:.2} MiB", value);
crates/nockchain-bench/src/speed_of_light/bench.rs:219:                println!("Ckpt sec/GiB:    {:.2}", value);
crates/nockchain-bench/src/speed_of_light/bench.rs:222:                println!("GC pause p95:    {:.1} ms", value);
crates/nockchain-bench/src/speed_of_light/bench.rs:224:            println!(
crates/nockchain-bench/src/speed_of_light/checkpoint.rs:155:            println!("Loaded checkpoint at event_num: {}", loaded.event_num);
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:279:            println!("Dry run:         yes");
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:280:            println!("Init time:       {:.2}s", self.init_time_secs);
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:282:                println!(
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:290:        println!("Peeks attempted: {}", self.peeks_attempted);
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:291:        println!("Success peeks:   {}", self.success_peeks);
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:292:        println!("Missing peeks:   {}", self.missing_peeks);
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:293:        println!("Error peeks:     {}", self.error_peeks);
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:294:        println!("Init time:       {:.2}s", self.init_time_secs);
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:295:        println!("Total peek time: {:.2}s", self.total_peek_time_secs);
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:299:                println!(
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:309:            _ => println!("Latency:         unavailable"),
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:312:        println!("Throughput:      {:.2} peeks/s", self.peeks_per_second);
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:315:            println!(
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:319:            println!(
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:326:            println!(
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:394:        println!(
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:400:            println!("Dry run requested; setup completed without executing peeks.");
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:476:                println!(
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:902:        eprintln!(
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:948:            eprintln!("Warning: memory sample unavailable during {phase}: {error}");
crates/nockchain-bench/src/speed_of_light/extractor.rs:680:                println!("=== Initializing shared BlockExtractor ===");
crates/nockchain-bench/src/speed_of_light/extractor.rs:682:                println!("=== Shared BlockExtractor ready ===");
crates/nockchain-bench/src/speed_of_light/extractor.rs:851:        println!("Extractor initialized successfully");
crates/nockchain-bench/src/speed_of_light/extractor.rs:861:        println!("[TEST 02] About to call get_chain_height()");
crates/nockchain-bench/src/speed_of_light/extractor.rs:864:                println!("[TEST 02] Chain height: {}", height);
crates/nockchain-bench/src/speed_of_light/extractor.rs:865:                println!("[TEST 02] Tip hash: {}", hash.to_base58());
crates/nockchain-bench/src/speed_of_light/extractor.rs:872:                println!(
crates/nockchain-bench/src/speed_of_light/extractor.rs:875:                println!("[TEST 02] Archive extraction can still proceed via range peek");
crates/nockchain-bench/src/speed_of_light/extractor.rs:878:                println!(
crates/nockchain-bench/src/speed_of_light/extractor.rs:898:        println!("[TEST 03] Extracting 100 blocks to archive...");
crates/nockchain-bench/src/speed_of_light/extractor.rs:909:        println!("[TEST 03] Archive size: {} bytes", archive_bytes.len());
crates/nockchain-bench/src/speed_of_light/extractor.rs:914:        println!("[TEST 03] Archive metadata:");
crates/nockchain-bench/src/speed_of_light/extractor.rs:915:        println!("  block_count: {}", metadata.block_count);
crates/nockchain-bench/src/speed_of_light/extractor.rs:916:        println!("  total_tx_count: {}", metadata.total_tx_count);
crates/nockchain-bench/src/speed_of_light/extractor.rs:917:        println!(
crates/nockchain-bench/src/speed_of_light/extractor.rs:927:        println!("[TEST 03] ✓ Archive created and validated successfully");
crates/nockchain-bench/src/speed_of_light/extractor.rs:942:        println!("[TEST 04] Extracting blocks 0-15 to archive...");
crates/nockchain-bench/src/speed_of_light/extractor.rs:965:        println!("[TEST 04] Loading archive...");
crates/nockchain-bench/src/speed_of_light/extractor.rs:992:        println!("[TEST 04] ✓ Archive roundtrip verified for blocks 0-15");
crates/nockchain-bench/src/speed_of_light/extractor.rs:1004:        println!("[TEST 05] Extracting blocks 0-15 to archive with mempool snapshots...");
crates/nockchain-bench/src/speed_of_light/extractor.rs:1035:        println!("[TEST 05] ✓ Archive mempool replay verified for blocks 0-15");
crates/nockchain-bench/src/speed_of_light/archive.rs:640:/// println!("Archive has {} blocks", reader.block_count());
crates/nockchain-bench/src/speed_of_light/archive.rs:647:///     println!("Block {}: {} bytes", entry.height, jam_bytes.len());
crates/nockchain-bench/src/speed_of_light/harness/profiler.rs:203:    eprintln!(
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:339:        println!("Checkpoint: {}", self.checkpoint_path.display());
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:340:        println!("Kernel:     {}", self.kernel_path.display());
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:341:        println!("Boot time:  {:.3}s", self.init_time.as_secs_f64());
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:347:            println!(
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:356:                println!("  error={error}");
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:360:            println!("Final tip:  {} {}", final_tip.height, final_tip.hash);
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:395:            eprintln!("quick-orchestrate warning: {warning}");
```

## P31 JSON.stringify used as key/hash/memo identity

_none found_

## P32 money-like arithmetic (audit integer cents/decimal)

_none found_

## P33 local time / UTC drift candidates

_none found_

## P34 detailed internal errors exposed

_none found_

## P35 suspicious ambiguous imports

```
crates/nockchain-bench/src/speed_of_light/bench.rs:6:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/fixture.rs:10:use std::path::Path;
crates/nockchain-bench/src/speed_of_light/checkpoint_builder.rs:3:use std::path::PathBuf;
crates/nockchain-bench/src/speed_of_light/checkpoint.rs:3:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/checkpoint.rs:136:    use std::path::PathBuf;
crates/nockchain-bench/src/speed_of_light/archive.rs:22:use std::path::Path;
crates/nockchain-bench/src/speed_of_light/extractor.rs:3:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/cold_peek/vma.rs:1:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/cold_peek/cgroup.rs:2:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/cold_peek/mod.rs:2:use std::path::Path;
crates/nockchain-bench/src/speed_of_light/cold_peek/mod.rs:3:use std::path::PathBuf;
crates/nockchain-bench/src/speed_of_light/peek_bench.rs:1:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:3:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/orchestrator.rs:1196:    use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/harness/validate.rs:1:use std::path::Path;
crates/nockchain-bench/src/speed_of_light/harness/profiler.rs:1:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/harness/profiler.rs:447:    use std::path::PathBuf;
crates/nockchain-bench/src/speed_of_light/harness/docker_image.rs:2:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/harness/sweep.rs:2:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/harness/sweep.rs:1514:    use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/harness/provenance.rs:1:use std::path::PathBuf;
crates/nockchain-bench/src/speed_of_light/harness/provenance.rs:314:    use std::path::PathBuf;
crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs:1:use std::path::Path;
crates/nockchain-bench/src/speed_of_light/harness/orchestrate.rs:323:    use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/harness/execute.rs:1:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:2:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/harness/docker.rs:1696:    use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/harness/case.rs:2:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/harness/case.rs:359:    use std::path::PathBuf;
crates/nockchain-bench/src/speed_of_light/harness/artifacts.rs:2:use std::path::Path;
crates/nockchain-bench/src/speed_of_light/harness/artifacts.rs:148:    use std::path::PathBuf;
crates/nockchain-bench/src/speed_of_light/harness/native.rs:1:use std::path::Path;
crates/nockchain-bench/src/speed_of_light/harness/native.rs:223:    use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/harness/mod.rs:14:use std::path::PathBuf;
crates/nockchain-bench/src/speed_of_light/harness/mod.rs:126:    use std::path::PathBuf;
crates/nockchain-bench/src/speed_of_light/runtime_compat.rs:6:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/kernel_utils.rs:3:use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/kernel_utils.rs:372:    use std::path::{Path, PathBuf};
crates/nockchain-bench/src/speed_of_light/mod.rs:88:    use std::path::PathBuf;
```

## P36 infra/config surfaces that should not ride with refactor commits

```
./docker/nockchain-bench/Dockerfile
./.worktrees/pma-bench-shim-verify/Dockerfile
./.worktrees/pma-bench-shim-verify/Cargo.toml
./.worktrees/pma-bench-shim-verify/docker-compose.metrics.yml
./.worktrees/pma-bench-shim-verify/Cargo.lock
./.worktrees/bench-harness-samply-sweep-20260317/Cargo.toml
./.worktrees/bench-harness-samply-sweep-20260317/Cargo.lock
./.worktrees/pma-smoke-quickread-20260414/Dockerfile
./.worktrees/pma-smoke-quickread-20260414/Cargo.toml
./.worktrees/pma-smoke-quickread-20260414/docker-compose.metrics.yml
./.worktrees/pma-smoke-quickread-20260414/Cargo.lock
./.worktrees/quick-read-bench-task1-red/Cargo.toml
./.worktrees/quick-read-bench-task1-red/Cargo.lock
./.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure/Dockerfile
./.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure/Cargo.toml
./.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure/docker-compose.metrics.yml
./.worktrees/pma-post-throughput-elas-sr-fsync-hrtb-closure/Cargo.lock
./.worktrees/pma-bench-fsync-target/Dockerfile
./.worktrees/pma-bench-fsync-target/Cargo.toml
./.worktrees/pma-bench-fsync-target/docker-compose.metrics.yml
./.worktrees/pma-bench-fsync-target/Cargo.lock
./.worktrees/sweep-fsync-impl/Cargo.toml
./.worktrees/sweep-fsync-impl/Cargo.lock
./.worktrees/sweep-fsync-task1/Cargo.toml
./.worktrees/sweep-fsync-task1/Cargo.lock
./.worktrees/pma-post-throughput-elas-sr-fsync-hrtb/Dockerfile
./.worktrees/pma-post-throughput-elas-sr-fsync-hrtb/Cargo.toml
./.worktrees/pma-post-throughput-elas-sr-fsync-hrtb/docker-compose.metrics.yml
./.worktrees/pma-post-throughput-elas-sr-fsync-hrtb/Cargo.lock
./.worktrees/bench-pages-pma-context/Cargo.toml
./.worktrees/bench-pages-pma-context/Cargo.lock
./.worktrees/pma-bench-run/Dockerfile
./.worktrees/pma-bench-run/Cargo.toml
./.worktrees/pma-bench-run/docker-compose.metrics.yml
./.worktrees/pma-bench-run/Cargo.lock
./.worktrees/bench-harness-phase2-closeout/Cargo.toml
./.worktrees/bench-harness-phase2-closeout/Cargo.lock
./.github/workflows/create-sync-pr.yml
./.github/workflows/docs.yml
./.github/workflows/release.yml
./.github/workflows/rust-format.yml
./.github/workflows/sync-to-nockchain.yml
./Cargo.toml
./crates/nockapp-grpc/Cargo.toml
./crates/nockchain-libp2p-io/Cargo.toml
./crates/nockup/Cargo.toml
./crates/nockup/Cargo.lock
./crates/nockchain/Cargo.toml
./crates/nockchain-api/Cargo.toml
./crates/bridge/Cargo.toml
./crates/nockchain-math/Cargo.toml
./crates/raw-tx-checker/Cargo.toml
./crates/nockapp/Cargo.toml
./crates/equix-latency/Cargo.toml
./crates/nockchain-explorer-tui/Cargo.toml
./crates/hoon/Cargo.toml
./crates/zkvm-jetpack/Cargo.toml
./crates/nockchain-bench/Cargo.toml
./crates/hoonc/Cargo.toml
./crates/nockchain-peek/Cargo.toml
./crates/habit/Cargo.toml
./crates/nockapp-grpc-proto/Cargo.toml
./crates/noun-serde/Cargo.toml
./crates/nockchain-types/Cargo.toml
./crates/chaff/Cargo.toml
./crates/nockchain-wallet/Cargo.toml
./crates/noun-serde-derive/Cargo.toml
./Cargo.lock
./tmp/pma-sweep-clean-worktree-20260415/Dockerfile
./tmp/pma-sweep-clean-worktree-20260415/Cargo.toml
./tmp/pma-sweep-clean-worktree-20260415/docker-compose.metrics.yml
./tmp/pma-sweep-clean-worktree-20260415/Cargo.lock
```

## P37 unpinned dependency snippets

```
Cargo.toml:277:[profile.release.package."*"]
Cargo.toml:287:[profile.bytehound.package."*"]
```

## P38 wildcard/glob imports

_none found_

## P39 async functions returning Promise (audit for real await)

_none found_

## P40 await/then in nearby non-async contexts (manual audit)

_none found_

---

## Next steps

1. Review each section; confirm which hits are real vs. false positives.
2. File beads for accepted patterns (one per pathology class).
3. Proceed to `./scripts/dup_scan.sh` for structural duplication.
4. Score candidates via `./scripts/score_candidates.py`.
5. For each accepted candidate: fill isomorphism card, edit, verify, ledger.

Full P1-P40 pathology catalog: `references/VIBE-CODED-PATHOLOGIES.md`.
Attack order (cheap wins first): the "AI-slop refactor playbook" in that file.
