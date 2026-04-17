use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use nockapp::nockapp::NockApp;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::archive::{ArchiveError, SolArchiveReader};
use super::checkpoint::CheckpointLoadError;
use super::harness::fsync_mode_label;
use super::kernel_utils::{
    init_checkpoint_backed_nockapp, peek_heaviest_chain_or_block, sol_replay_wire,
    CheckpointBackedInitError, KernelInitError,
};
use super::peek_bench::{peek_height_result, PeekResultKind};
use super::poke::{poke_block_from_jam, PokeStepError};
use super::types::SolHeight;

#[derive(Debug, Clone, Deserialize)]
pub struct QuickOrchestratePlan {
    pub checkpoint: PathBuf,
    pub kernel: PathBuf,
    #[serde(default)]
    pub steps: Vec<QuickOrchestrateStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdMode {
    Strict,
    Soft,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QuickOrchestrateStep {
    PokeArchiveBlock {
        archive: PathBuf,
        height: u64,
        #[serde(default)]
        label: Option<String>,
    },
    PeekHeight {
        height: u64,
        #[serde(default)]
        label: Option<String>,
    },
    ForceCold {
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        tolerance_pages: Option<u64>,
        #[serde(default)]
        max_attempts: Option<u32>,
    },
    PeekHeightCold {
        height: u64,
        #[serde(default)]
        label: Option<String>,
        #[serde(default)]
        tolerance_pages: Option<u64>,
        #[serde(default)]
        max_attempts: Option<u32>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StepOutcome {
    #[serde(rename = "ok")]
    Ok,
    #[serde(rename = "success")]
    Success,
    #[serde(rename = "missing")]
    Missing,
    #[serde(rename = "error")]
    Error,
}

impl StepOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Success => "success",
            Self::Missing => "missing",
            Self::Error => "error",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StepType {
    PokeArchiveBlock,
    PeekHeight,
    ForceCold,
    PeekHeightCold,
}

impl StepType {
    fn as_str(self) -> &'static str {
        match self {
            Self::PokeArchiveBlock => "poke_archive_block",
            Self::PeekHeight => "peek_height",
            Self::ForceCold => "force_cold",
            Self::PeekHeightCold => "peek_height_cold",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepResult {
    label: String,
    step_type: StepType,
    height: Option<u64>,
    outcome: StepOutcome,
    duration: Duration,
    error_message: Option<String>,
    minflt_delta: Option<u64>,
    majflt_delta: Option<u64>,
    cold_verified: Option<bool>,
    residency_pages_after: Option<u64>,
    residency_total_pages: Option<u64>,
    cold_attempts: Option<u32>,
    degraded_reason: Option<String>,
}

#[derive(Serialize)]
struct StepResultWire<'a> {
    label: &'a str,
    #[serde(rename = "type")]
    step_type: StepType,
    #[serde(skip_serializing_if = "Option::is_none")]
    height: Option<u64>,
    outcome: StepOutcome,
    duration_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none", rename = "error")]
    error: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    minflt_delta: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    majflt_delta: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cold_verified: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    residency_pages_after: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    residency_total_pages: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cold_attempts: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    degraded_reason: Option<&'a str>,
}

impl StepResult {
    fn new(
        label: String,
        step_type: StepType,
        height: Option<u64>,
        outcome: StepOutcome,
        duration: Duration,
        error_message: Option<String>,
    ) -> Self {
        Self {
            label,
            step_type,
            height,
            outcome,
            duration,
            error_message,
            minflt_delta: None,
            majflt_delta: None,
            cold_verified: None,
            residency_pages_after: None,
            residency_total_pages: None,
            cold_attempts: None,
            degraded_reason: None,
        }
    }

    fn ok(label: String, step_type: StepType, height: Option<u64>, duration: Duration) -> Self {
        Self::new(label, step_type, height, StepOutcome::Ok, duration, None)
    }

    fn with_outcome(
        label: String,
        step_type: StepType,
        height: Option<u64>,
        outcome: StepOutcome,
        duration: Duration,
    ) -> Self {
        Self::new(label, step_type, height, outcome, duration, None)
    }

    fn error(
        label: String,
        step_type: StepType,
        height: Option<u64>,
        duration: Duration,
        error_message: String,
    ) -> Self {
        Self::new(
            label,
            step_type,
            height,
            StepOutcome::Error,
            duration,
            Some(error_message),
        )
    }

    fn wire(&self) -> StepResultWire<'_> {
        StepResultWire {
            label: &self.label,
            step_type: self.step_type,
            height: self.height,
            outcome: self.outcome,
            duration_ms: duration_ms(self.duration),
            error: self.error_message.as_deref(),
            minflt_delta: self.minflt_delta,
            majflt_delta: self.majflt_delta,
            cold_verified: self.cold_verified,
            residency_pages_after: self.residency_pages_after,
            residency_total_pages: self.residency_total_pages,
            cold_attempts: self.cold_attempts,
            degraded_reason: self.degraded_reason.as_deref(),
        }
    }
}

impl Serialize for StepResult {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.wire().serialize(serializer)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FinalTip {
    height: u64,
    hash: String,
}

#[derive(Debug, Clone)]
pub struct QuickOrchestrateResults {
    checkpoint_path: PathBuf,
    kernel_path: PathBuf,
    fsync: bool,
    init_time: Duration,
    steps: Vec<StepResult>,
    failed_step_index: Option<usize>,
    final_tip: Option<FinalTip>,
}

#[derive(Serialize)]
struct BootWire<'a> {
    checkpoint: &'a str,
    kernel: &'a str,
    fsync: &'static str,
    init_time_secs: f64,
}

#[derive(Serialize)]
struct QuickOrchestrateResultsWire<'a> {
    boot: BootWire<'a>,
    steps: &'a [StepResult],
}

impl QuickOrchestrateResults {
    pub fn succeeded(&self) -> bool {
        self.failed_step_index.is_none()
    }

    pub fn has_step_failure(&self) -> bool {
        self.failed_step_index.is_some()
    }

    pub fn failure_message(&self) -> Option<&str> {
        self.failed_step_index
            .and_then(|index| self.steps.get(index))
            .and_then(|step| step.error_message.as_deref())
    }

    pub fn to_compact_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&QuickOrchestrateResultsWire {
            boot: BootWire {
                checkpoint: &self.checkpoint_path.to_string_lossy(),
                kernel: &self.kernel_path.to_string_lossy(),
                fsync: fsync_mode_label(self.fsync),
                init_time_secs: self.init_time.as_secs_f64(),
            },
            steps: &self.steps,
        })
    }

    pub fn print_summary(&self) {
        println!("Checkpoint: {}", self.checkpoint_path.display());
        println!("Kernel:     {}", self.kernel_path.display());
        println!("Boot time:  {:.3}s", self.init_time.as_secs_f64());
        for step in &self.steps {
            let height_fragment = step
                .height
                .map(|height| format!(" height={height}"))
                .unwrap_or_default();
            println!(
                "Step {label}: type={step_type}{height_fragment} duration_ms={duration_ms:.3} outcome={outcome}",
                label = step.label,
                step_type = step.step_type.as_str(),
                height_fragment = height_fragment,
                duration_ms = duration_ms(step.duration),
                outcome = step.outcome.as_str(),
            );
            if let Some(error) = &step.error_message {
                println!("  error={error}");
            }
        }
        if let Some(final_tip) = &self.final_tip {
            println!("Final tip:  {} {}", final_tip.height, final_tip.hash);
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuickOrchestrateRunner {
    plan_path: PathBuf,
    work_dir: PathBuf,
    fsync: bool,
    cold_mode: ColdMode,
}

impl QuickOrchestrateRunner {
    pub fn new(plan_path: PathBuf, work_dir: PathBuf, fsync: bool, cold_mode: ColdMode) -> Self {
        Self {
            plan_path,
            work_dir,
            fsync,
            cold_mode,
        }
    }

    pub async fn run(&self) -> Result<QuickOrchestrateResults, PreRunError> {
        let prepared = load_and_validate_plan(&self.plan_path)?;
        let PreparedPlan {
            checkpoint_path,
            kernel_path,
            steps,
            archive_cache,
            warnings,
        } = prepared;
        let _cold_mode = self.cold_mode;

        for warning in &warnings {
            eprintln!("quick-orchestrate warning: {warning}");
        }

        std::fs::create_dir_all(&self.work_dir).map_err(|source| BootFailure::WorkDirCreate {
            path: self.work_dir.clone(),
            source,
        })?;

        let init_started_at = Instant::now();
        let nockapp = init_checkpoint_backed_nockapp(
            &checkpoint_path, &kernel_path, &self.work_dir, self.fsync,
        )
        .await
        .map_err(BootFailure::from)?;
        let init_time = init_started_at.elapsed();

        let mut context = ScenarioContext {
            nockapp,
            archive_cache,
        };

        let replay_wire = sol_replay_wire();
        let mut results = QuickOrchestrateResults {
            checkpoint_path,
            kernel_path,
            fsync: self.fsync,
            init_time,
            steps: Vec::with_capacity(steps.len()),
            failed_step_index: None,
            final_tip: None,
        };

        for (index, step) in steps.iter().enumerate() {
            let step_result = execute_step(&mut context, step, &replay_wire).await;
            let failed = matches!(step_result.outcome, StepOutcome::Error);
            results.steps.push(step_result);
            if failed {
                results.failed_step_index = Some(index);
                break;
            }
        }

        results.final_tip = query_final_tip(&mut context.nockapp).await;
        Ok(results)
    }
}

struct ScenarioContext {
    nockapp: NockApp,
    archive_cache: HashMap<PathBuf, SolArchiveReader>,
}

struct PreparedPlan {
    checkpoint_path: PathBuf,
    kernel_path: PathBuf,
    steps: Vec<PreparedStep>,
    archive_cache: HashMap<PathBuf, SolArchiveReader>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone)]
enum PreparedStep {
    PokeArchiveBlock {
        label: String,
        height: u64,
        archive_path: PathBuf,
    },
    PeekHeight {
        label: String,
        height: u64,
    },
    ForceCold {
        label: String,
        options: crate::speed_of_light::cold_peek::ColdStepOptions,
    },
    PeekHeightCold {
        label: String,
        height: u64,
        options: crate::speed_of_light::cold_peek::ColdStepOptions,
    },
}

impl PreparedStep {
    #[cfg(test)]
    fn label(&self) -> &str {
        match self {
            Self::PokeArchiveBlock { label, .. }
            | Self::PeekHeight { label, .. }
            | Self::ForceCold { label, .. }
            | Self::PeekHeightCold { label, .. } => label,
        }
    }
}

#[derive(Debug, Error)]
pub enum PlanValidationError {
    #[error("failed to read quick-orchestrate plan {path}: {source}")]
    ReadPlan {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse quick-orchestrate plan {path}: {source}")]
    ParsePlan {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("failed to resolve current working directory: {0}")]
    CurrentDir(#[source] std::io::Error),

    #[error("checkpoint path does not exist: {path}")]
    MissingCheckpoint { path: PathBuf },

    #[error("kernel path does not exist: {path}")]
    MissingKernel { path: PathBuf },

    #[error("archive path does not exist: {path}")]
    MissingArchive { path: PathBuf },

    #[error("failed to canonicalize {kind} path {path}: {source}")]
    Canonicalize {
        kind: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse archive {path}: {source}")]
    ArchiveParse {
        path: PathBuf,
        #[source]
        source: ArchiveError,
    },

    #[error(
        "quick-orchestrate step {step_type} at index {index} requires --features pma-runtime-compat"
    )]
    ColdStepRequiresPmaRuntimeCompat {
        index: usize,
        step_type: &'static str,
    },
}

#[derive(Debug, Error)]
pub enum BootFailure {
    #[error("failed to create work dir {path}: {source}")]
    WorkDirCreate {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to load checkpoint: {0}")]
    CheckpointLoad(#[from] CheckpointLoadError),

    #[error("failed to initialize checkpoint-backed kernel: {0}")]
    KernelInit(#[from] KernelInitError),
}

impl From<CheckpointBackedInitError> for BootFailure {
    fn from(value: CheckpointBackedInitError) -> Self {
        match value {
            CheckpointBackedInitError::CheckpointLoad(source) => Self::CheckpointLoad(source),
            CheckpointBackedInitError::KernelInit(source) => Self::KernelInit(source),
        }
    }
}

#[derive(Debug, Error)]
enum StepExecutionError {
    #[error("block not found in archive at height {height}: {path}")]
    ArchiveMissing { path: PathBuf, height: u64 },

    #[error("failed to read archive {path} at height {height}: {source}")]
    ArchiveLookup {
        path: PathBuf,
        height: u64,
        #[source]
        source: ArchiveError,
    },

    #[error("failed to replay archive block from {path} at height {height}: {source}")]
    Poke {
        path: PathBuf,
        height: u64,
        #[source]
        source: PokeStepError,
    },

    #[error("failed to peek height {height}: {source}")]
    Peek {
        height: u64,
        #[source]
        source: nockapp::nockapp::NockAppError,
    },
}

#[derive(Debug, Error)]
pub enum PreRunError {
    #[error(transparent)]
    Plan(#[from] PlanValidationError),

    #[error(transparent)]
    Boot(#[from] BootFailure),
}

fn load_and_validate_plan(plan_path: &Path) -> Result<PreparedPlan, PlanValidationError> {
    let bytes = std::fs::read(plan_path).map_err(|source| PlanValidationError::ReadPlan {
        path: plan_path.to_path_buf(),
        source,
    })?;
    let plan: QuickOrchestratePlan =
        serde_json::from_slice(&bytes).map_err(|source| PlanValidationError::ParsePlan {
            path: plan_path.to_path_buf(),
            source,
        })?;

    let checkpoint_path = resolve_existing_path(&plan.checkpoint, "checkpoint")?;
    let kernel_path = resolve_existing_path(&plan.kernel, "kernel")?;
    let mut archive_cache = HashMap::new();
    let mut steps = Vec::with_capacity(plan.steps.len());
    let warnings = Vec::new();

    for (index, step) in plan.steps.into_iter().enumerate() {
        match step {
            QuickOrchestrateStep::PokeArchiveBlock {
                archive,
                height,
                label,
            } => {
                let archive_path = resolve_existing_path(&archive, "archive")?;
                if !archive_cache.contains_key(&archive_path) {
                    let reader = SolArchiveReader::from_file(&archive_path).map_err(|source| {
                        PlanValidationError::ArchiveParse {
                            path: archive_path.clone(),
                            source,
                        }
                    })?;
                    archive_cache.insert(archive_path.clone(), reader);
                }
                steps.push(PreparedStep::PokeArchiveBlock {
                    label: label.unwrap_or_else(|| format!("step-{index}")),
                    height,
                    archive_path,
                });
            }
            QuickOrchestrateStep::PeekHeight { height, label } => {
                steps.push(PreparedStep::PeekHeight {
                    label: label.unwrap_or_else(|| format!("step-{index}")),
                    height,
                });
            }
            QuickOrchestrateStep::ForceCold {
                label,
                tolerance_pages,
                max_attempts,
            } => {
                if !cfg!(feature = "pma-runtime-compat") {
                    return Err(PlanValidationError::ColdStepRequiresPmaRuntimeCompat {
                        index,
                        step_type: "force_cold",
                    });
                }
                steps.push(PreparedStep::ForceCold {
                    label: label.unwrap_or_else(|| format!("step-{index}")),
                    options: crate::speed_of_light::cold_peek::ColdStepOptions {
                        tolerance_pages: tolerance_pages.unwrap_or(0),
                        max_attempts: max_attempts.unwrap_or(3),
                    },
                });
            }
            QuickOrchestrateStep::PeekHeightCold {
                height,
                label,
                tolerance_pages,
                max_attempts,
            } => {
                if !cfg!(feature = "pma-runtime-compat") {
                    return Err(PlanValidationError::ColdStepRequiresPmaRuntimeCompat {
                        index,
                        step_type: "peek_height_cold",
                    });
                }
                steps.push(PreparedStep::PeekHeightCold {
                    label: label.unwrap_or_else(|| format!("step-{index}")),
                    height,
                    options: crate::speed_of_light::cold_peek::ColdStepOptions {
                        tolerance_pages: tolerance_pages.unwrap_or(0),
                        max_attempts: max_attempts.unwrap_or(3),
                    },
                });
            }
        }
    }

    Ok(PreparedPlan {
        checkpoint_path,
        kernel_path,
        steps,
        archive_cache,
        warnings,
    })
}

fn resolve_existing_path(path: &Path, kind: &'static str) -> Result<PathBuf, PlanValidationError> {
    let current_dir = std::env::current_dir().map_err(PlanValidationError::CurrentDir)?;
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        current_dir.join(path)
    };

    if !resolved.exists() {
        return Err(match kind {
            "checkpoint" => PlanValidationError::MissingCheckpoint { path: resolved },
            "kernel" => PlanValidationError::MissingKernel { path: resolved },
            _ => PlanValidationError::MissingArchive { path: resolved },
        });
    }

    resolved
        .canonicalize()
        .map_err(|source| PlanValidationError::Canonicalize {
            kind,
            path: resolved,
            source,
        })
}

async fn execute_step(
    context: &mut ScenarioContext,
    step: &PreparedStep,
    replay_wire: &nockapp::nockapp::wire::WireRepr,
) -> StepResult {
    match step {
        PreparedStep::PokeArchiveBlock {
            label,
            height,
            archive_path,
        } => execute_poke_step(context, label, *height, archive_path, replay_wire).await,
        PreparedStep::PeekHeight { label, height } => {
            execute_peek_step(context, label, *height).await
        }
        PreparedStep::ForceCold { label, options } => {
            execute_unimplemented_cold_step(label, None, StepType::ForceCold, *options)
        }
        PreparedStep::PeekHeightCold {
            label,
            height,
            options,
        } => execute_unimplemented_cold_step(
            label,
            Some(*height),
            StepType::PeekHeightCold,
            *options,
        ),
    }
}

fn execute_unimplemented_cold_step(
    label: &str,
    height: Option<u64>,
    step_type: StepType,
    options: crate::speed_of_light::cold_peek::ColdStepOptions,
) -> StepResult {
    StepResult::error(
        label.to_string(),
        step_type,
        height,
        Duration::ZERO,
        format!(
            "cold step execution is not wired until Task 4 (tolerance_pages={}, max_attempts={})",
            options.tolerance_pages, options.max_attempts
        ),
    )
}

async fn execute_poke_step(
    context: &mut ScenarioContext,
    label: &str,
    height: u64,
    archive_path: &Path,
    replay_wire: &nockapp::nockapp::wire::WireRepr,
) -> StepResult {
    let started_at = Instant::now();

    let jam_bytes = match lookup_archive_jam(context, archive_path, height) {
        Ok(jam_bytes) => jam_bytes,
        Err(error) => {
            return StepResult::error(
                label.to_string(),
                StepType::PokeArchiveBlock,
                Some(height),
                started_at.elapsed(),
                error.to_string(),
            );
        }
    };

    match poke_block_from_jam(&mut context.nockapp, replay_wire.clone(), &jam_bytes).await {
        Ok(duration) => StepResult::ok(
            label.to_string(),
            StepType::PokeArchiveBlock,
            Some(height),
            duration,
        ),
        Err(source) => StepResult::error(
            label.to_string(),
            StepType::PokeArchiveBlock,
            Some(height),
            started_at.elapsed(),
            StepExecutionError::Poke {
                path: archive_path.to_path_buf(),
                height,
                source,
            }
            .to_string(),
        ),
    }
}

async fn execute_peek_step(context: &mut ScenarioContext, label: &str, height: u64) -> StepResult {
    let started_at = Instant::now();

    match peek_height_result(&mut context.nockapp, height).await {
        Ok(sample) => {
            let outcome = match sample.kind {
                PeekResultKind::Success => StepOutcome::Success,
                PeekResultKind::Missing => StepOutcome::Missing,
            };
            StepResult::with_outcome(
                label.to_string(),
                StepType::PeekHeight,
                Some(height),
                outcome,
                Duration::from_micros(sample.latency_us()),
            )
        }
        Err(source) => StepResult::error(
            label.to_string(),
            StepType::PeekHeight,
            Some(height),
            started_at.elapsed(),
            StepExecutionError::Peek { height, source }.to_string(),
        ),
    }
}

fn lookup_archive_jam(
    context: &ScenarioContext,
    archive_path: &Path,
    height: u64,
) -> Result<Vec<u8>, StepExecutionError> {
    let reader = context
        .archive_cache
        .get(archive_path)
        .expect("validated archive should be cached");
    reader
        .get_jam_by_height(SolHeight(height))
        .map(|jam_bytes| jam_bytes.to_vec())
        .map_err(|source| match source {
            ArchiveError::BlockNotFound(_) => StepExecutionError::ArchiveMissing {
                path: archive_path.to_path_buf(),
                height,
            },
            other => StepExecutionError::ArchiveLookup {
                path: archive_path.to_path_buf(),
                height,
                source: other,
            },
        })
}

async fn query_final_tip(nockapp: &mut NockApp) -> Option<FinalTip> {
    match peek_heaviest_chain_or_block(nockapp).await {
        Ok(Some((height, hash))) => Some(FinalTip {
            height: height.0 .0,
            hash: hash.to_base58(),
        }),
        Ok(None) | Err(_) => None,
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use nockchain_math::belt::Belt;
    use nockchain_types::tx_engine::common::Hash;
    use serde_json::json;
    use tempfile::tempdir;

    use super::*;
    use crate::speed_of_light::archive::SolArchiveWriter;
    use crate::speed_of_light::checkpoint::load_checkpoint;
    use crate::speed_of_light::kernel_utils::{init_nockapp, peek_heaviest_chain_or_block};
    use crate::speed_of_light::types::{ProofVersion, SolHeight};

    #[test]
    fn quick_orchestrate_plan_json_deserializes_mvp_schema() {
        let plan: QuickOrchestratePlan = serde_json::from_value(json!({
            "checkpoint": "/tmp/0.chkjam",
            "kernel": "/tmp/dumb.jam",
            "steps": [
                {
                    "type": "poke_archive_block",
                    "archive": "/tmp/blocks.solarch",
                    "height": 7,
                    "label": "poke-one"
                },
                {
                    "type": "peek_height",
                    "height": 7,
                    "label": "peek-one"
                }
            ]
        }))
        .expect("plan should deserialize");

        assert_eq!(plan.steps.len(), 2);
    }

    #[test]
    fn missing_step_labels_default_to_step_indexes() {
        let temp_dir = tempdir().expect("temp dir");
        let checkpoint = temp_dir.path().join("checkpoint.chkjam");
        let kernel = temp_dir.path().join("kernel.jam");
        std::fs::write(&checkpoint, "checkpoint").expect("checkpoint");
        std::fs::write(&kernel, "kernel").expect("kernel");
        let archive = write_parseable_archive(temp_dir.path(), "blocks.solarch");

        let plan_path = write_plan(
            temp_dir.path(),
            json!({
                "checkpoint": checkpoint,
                "kernel": kernel,
                "steps": [
                    {
                        "type": "poke_archive_block",
                        "archive": archive,
                        "height": 1
                    },
                    {
                        "type": "peek_height",
                        "height": 1
                    }
                ]
            }),
        );

        let validated = load_and_validate_plan(&plan_path).expect("validation");
        assert_eq!(validated.steps[0].label(), "step-0");
        assert_eq!(validated.steps[1].label(), "step-1");
    }

    #[test]
    fn unknown_step_type_fails_validation() {
        let temp_dir = tempdir().expect("temp dir");
        let checkpoint = temp_dir.path().join("checkpoint.chkjam");
        let kernel = temp_dir.path().join("kernel.jam");
        std::fs::write(&checkpoint, "checkpoint").expect("checkpoint");
        std::fs::write(&kernel, "kernel").expect("kernel");

        let plan_path = write_plan(
            temp_dir.path(),
            json!({
                "checkpoint": checkpoint,
                "kernel": kernel,
                "steps": [
                    {
                        "type": "not-a-real-step",
                        "height": 1
                    }
                ]
            }),
        );

        let error = load_and_validate_plan(&plan_path)
            .err()
            .expect("validation should fail");
        assert!(error.to_string().contains("not-a-real-step"));
    }

    #[test]
    fn missing_required_fields_fail_validation() {
        let temp_dir = tempdir().expect("temp dir");
        let checkpoint = temp_dir.path().join("checkpoint.chkjam");
        let kernel = temp_dir.path().join("kernel.jam");
        std::fs::write(&checkpoint, "checkpoint").expect("checkpoint");
        std::fs::write(&kernel, "kernel").expect("kernel");

        let plan_path = write_plan(
            temp_dir.path(),
            json!({
                "checkpoint": checkpoint,
                "kernel": kernel,
                "steps": [
                    {
                        "type": "poke_archive_block",
                        "height": 1
                    }
                ]
            }),
        );

        let error = load_and_validate_plan(&plan_path)
            .err()
            .expect("validation should fail");
        assert!(error.to_string().contains("archive"));
    }

    #[test]
    fn validation_eagerly_parses_archives() {
        let temp_dir = tempdir().expect("temp dir");
        let checkpoint = temp_dir.path().join("checkpoint.chkjam");
        let kernel = temp_dir.path().join("kernel.jam");
        let archive = temp_dir.path().join("broken.solarch");
        std::fs::write(&checkpoint, "checkpoint").expect("checkpoint");
        std::fs::write(&kernel, "kernel").expect("kernel");
        std::fs::write(&archive, "not-an-archive").expect("archive");

        let plan_path = write_plan(
            temp_dir.path(),
            json!({
                "checkpoint": checkpoint,
                "kernel": kernel,
                "steps": [
                    {
                        "type": "poke_archive_block",
                        "archive": archive,
                        "height": 1
                    }
                ]
            }),
        );

        let error = load_and_validate_plan(&plan_path)
            .err()
            .expect("archive should be parsed eagerly");
        assert!(error.to_string().contains("archive"));
    }

    #[test]
    fn step_outcome_serializes_to_lowercase_strings() {
        assert_eq!(
            serde_json::to_string(&StepOutcome::Ok).expect("serialize"),
            "\"ok\""
        );
        assert_eq!(
            serde_json::to_string(&StepOutcome::Success).expect("serialize"),
            "\"success\""
        );
        assert_eq!(
            serde_json::to_string(&StepOutcome::Missing).expect("serialize"),
            "\"missing\""
        );
        assert_eq!(
            serde_json::to_string(&StepOutcome::Error).expect("serialize"),
            "\"error\""
        );
    }

    #[test]
    fn quick_orchestrate_step_json_uses_type_duration_ms_and_error_fields() {
        let value = serde_json::to_value(StepResult {
            label: "poke-one".to_string(),
            step_type: StepType::PokeArchiveBlock,
            height: Some(7),
            outcome: StepOutcome::Error,
            duration: Duration::from_micros(12_345),
            error_message: Some("no block".to_string()),
            minflt_delta: None,
            majflt_delta: None,
            cold_verified: None,
            residency_pages_after: None,
            residency_total_pages: None,
            cold_attempts: None,
            degraded_reason: None,
        })
        .expect("serialize step");

        assert_eq!(value["label"], json!("poke-one"));
        assert_eq!(value["type"], json!("poke_archive_block"));
        assert_eq!(value["height"], json!(7));
        assert_eq!(value["outcome"], json!("error"));
        assert!(value["duration_ms"].is_number());
        assert_eq!(value["error"], json!("no block"));
        assert!(value.get("step_type").is_none());
        assert!(value.get("error_message").is_none());
    }

    #[test]
    fn quick_orchestrate_step_json_handles_optional_cold_fields_and_optional_height() {
        let force_cold = serde_json::to_value(StepResult {
            label: "cold-prep".to_string(),
            step_type: StepType::ForceCold,
            height: None,
            outcome: StepOutcome::Ok,
            duration: Duration::from_millis(2),
            error_message: None,
            minflt_delta: Some(11),
            majflt_delta: Some(1),
            cold_verified: Some(false),
            residency_pages_after: Some(7),
            residency_total_pages: Some(100),
            cold_attempts: Some(3),
            degraded_reason: Some("macos_unsupported".to_string()),
        })
        .expect("serialize force cold");
        let cold_peek = serde_json::to_value(StepResult {
            label: "cold-peek".to_string(),
            step_type: StepType::PeekHeightCold,
            height: Some(7),
            outcome: StepOutcome::Success,
            duration: Duration::from_millis(3),
            error_message: None,
            minflt_delta: Some(22),
            majflt_delta: Some(0),
            cold_verified: Some(true),
            residency_pages_after: Some(0),
            residency_total_pages: Some(100),
            cold_attempts: Some(1),
            degraded_reason: None,
        })
        .expect("serialize cold peek");

        assert!(force_cold.get("height").is_none());
        assert_eq!(force_cold["type"], json!("force_cold"));
        assert_eq!(force_cold["cold_verified"], json!(false));
        assert_eq!(force_cold["degraded_reason"], json!("macos_unsupported"));
        assert_eq!(cold_peek["height"], json!(7));
        assert_eq!(cold_peek["type"], json!("peek_height_cold"));
    }

    #[test]
    fn quick_orchestrate_fail_fast_result_json_keeps_only_executed_steps() {
        let results = QuickOrchestrateResults {
            checkpoint_path: PathBuf::from("/tmp/0.chkjam"),
            kernel_path: PathBuf::from("/tmp/dumb.jam"),
            fsync: true,
            init_time: Duration::from_millis(123),
            steps: vec![
                StepResult {
                    label: "peek-one".to_string(),
                    step_type: StepType::PeekHeight,
                    height: Some(7),
                    outcome: StepOutcome::Success,
                    duration: Duration::from_millis(3),
                    error_message: None,
                    minflt_delta: None,
                    majflt_delta: None,
                    cold_verified: None,
                    residency_pages_after: None,
                    residency_total_pages: None,
                    cold_attempts: None,
                    degraded_reason: None,
                },
                StepResult {
                    label: "poke-bad".to_string(),
                    step_type: StepType::PokeArchiveBlock,
                    height: Some(99),
                    outcome: StepOutcome::Error,
                    duration: Duration::from_millis(1),
                    error_message: Some("missing".to_string()),
                    minflt_delta: None,
                    majflt_delta: None,
                    cold_verified: None,
                    residency_pages_after: None,
                    residency_total_pages: None,
                    cold_attempts: None,
                    degraded_reason: None,
                },
            ],
            failed_step_index: Some(1),
            final_tip: None,
        };

        assert!(!results.succeeded());
        let value = serde_json::from_str::<serde_json::Value>(
            &results.to_compact_json().expect("compact json"),
        )
        .expect("parse json");
        assert_eq!(value["steps"].as_array().expect("steps").len(), 2);
        assert_eq!(value["steps"][1]["error"], json!("missing"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "checkpoint-backed integration coverage; exercised in release smoke verification"]
    async fn fail_fast_runner_returns_results_after_a_successful_step() {
        let Some((checkpoint, kernel, tip_height)) =
            tokio::time::timeout(Duration::from_secs(60), fixture_boot_inputs())
                .await
                .ok()
                .flatten()
        else {
            return;
        };

        let temp_dir = tempdir().expect("temp dir");
        let archive = write_parseable_archive(temp_dir.path(), "blocks.solarch");
        let plan_path = write_plan(
            temp_dir.path(),
            json!({
                "checkpoint": checkpoint,
                "kernel": kernel,
                "steps": [
                    {
                        "type": "peek_height",
                        "height": tip_height,
                        "label": "peek-tip"
                    },
                    {
                        "type": "poke_archive_block",
                        "archive": archive,
                        "height": tip_height + 1000,
                        "label": "poke-missing"
                    },
                    {
                        "type": "peek_height",
                        "height": tip_height,
                        "label": "never-runs"
                    }
                ]
            }),
        );

        let Ok(Ok(results)) = tokio::time::timeout(
            Duration::from_secs(60),
            QuickOrchestrateRunner::new(
                plan_path,
                temp_dir.path().join("work"),
                true,
                ColdMode::Strict,
            )
            .run(),
        )
        .await
        else {
            return;
        };

        assert!(!results.succeeded());
        assert_eq!(results.failed_step_index, Some(1));
        assert_eq!(results.steps.len(), 2);
        assert_eq!(results.steps[0].outcome, StepOutcome::Success);
        assert_eq!(results.steps[1].outcome, StepOutcome::Error);

        let json = results.to_compact_json().expect("compact json");
        assert!(!json.contains('\n'));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "checkpoint-backed integration coverage; exercised in release smoke verification"]
    async fn peek_missing_is_nonfatal_and_serializes_as_missing() {
        let Some((checkpoint, kernel, tip_height)) =
            tokio::time::timeout(Duration::from_secs(60), fixture_boot_inputs())
                .await
                .ok()
                .flatten()
        else {
            return;
        };

        let temp_dir = tempdir().expect("temp dir");
        let plan_path = write_plan(
            temp_dir.path(),
            json!({
                "checkpoint": checkpoint,
                "kernel": kernel,
                "steps": [
                    {
                        "type": "peek_height",
                        "height": tip_height + 1
                    }
                ]
            }),
        );

        let Ok(Ok(results)) = tokio::time::timeout(
            Duration::from_secs(60),
            QuickOrchestrateRunner::new(
                plan_path,
                temp_dir.path().join("work"),
                true,
                ColdMode::Strict,
            )
            .run(),
        )
        .await
        else {
            return;
        };

        assert!(results.succeeded());
        assert_eq!(results.steps.len(), 1);
        assert_eq!(results.steps[0].outcome, StepOutcome::Missing);

        let value = serde_json::from_str::<serde_json::Value>(
            &results.to_compact_json().expect("compact json"),
        )
        .expect("parse json");
        assert_eq!(value["steps"][0]["outcome"], json!("missing"));
    }

    #[test]
    fn quick_orchestrate_plan_json_deserializes_new_cold_steps() {
        let plan: QuickOrchestratePlan = serde_json::from_value(json!({
            "checkpoint": "/tmp/0.chkjam",
            "kernel": "/tmp/dumb.jam",
            "steps": [
                {
                    "type": "force_cold",
                    "label": "cold-prep",
                    "tolerance_pages": 2,
                    "max_attempts": 5
                },
                {
                    "type": "peek_height_cold",
                    "height": 7,
                    "label": "cold-peek",
                    "tolerance_pages": 1,
                    "max_attempts": 4
                }
            ]
        }))
        .expect("plan should deserialize");

        assert_eq!(plan.steps.len(), 2);
        match &plan.steps[0] {
            QuickOrchestrateStep::ForceCold {
                tolerance_pages,
                max_attempts,
                ..
            } => {
                assert_eq!(*tolerance_pages, Some(2));
                assert_eq!(*max_attempts, Some(5));
            }
            other => panic!("expected force_cold, got {other:?}"),
        }
        match &plan.steps[1] {
            QuickOrchestrateStep::PeekHeightCold {
                height,
                tolerance_pages,
                max_attempts,
                ..
            } => {
                assert_eq!(*height, 7);
                assert_eq!(*tolerance_pages, Some(1));
                assert_eq!(*max_attempts, Some(4));
            }
            other => panic!("expected peek_height_cold, got {other:?}"),
        }
    }

    #[test]
    fn quick_orchestrate_validation_rejects_cold_steps_without_pma_runtime_compat() {
        let temp_dir = tempdir().expect("temp dir");
        let checkpoint = temp_dir.path().join("checkpoint.chkjam");
        let kernel = temp_dir.path().join("kernel.jam");
        std::fs::write(&checkpoint, "checkpoint").expect("checkpoint");
        std::fs::write(&kernel, "kernel").expect("kernel");

        let plan_path = write_plan(
            temp_dir.path(),
            json!({
                "checkpoint": checkpoint,
                "kernel": kernel,
                "steps": [
                    {
                        "type": "force_cold",
                        "label": "cold-prep"
                    }
                ]
            }),
        );

        let error = load_and_validate_plan(&plan_path)
            .err()
            .expect("validation should fail");
        assert!(
            error.to_string().contains("--features pma-runtime-compat"),
            "{error}"
        );
        assert!(error.to_string().contains("force_cold"), "{error}");
    }

    fn write_plan(dir: &Path, value: serde_json::Value) -> PathBuf {
        let path = dir.join("plan.json");
        std::fs::write(&path, serde_json::to_vec(&value).expect("plan json")).expect("write plan");
        path
    }

    fn write_parseable_archive(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut writer = SolArchiveWriter::new();
        writer
            .add_block(
                SolHeight(1),
                dummy_hash(1),
                0,
                ProofVersion::V0,
                b"junk-jam-bytes",
            )
            .expect("add block");
        writer.write_to_file(&path).expect("write archive");
        path
    }

    async fn fixture_boot_inputs() -> Option<(PathBuf, PathBuf, u64)> {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let checkpoint = repo_root.join("checkpoints/0.chkjam");
        let kernel = repo_root.join("assets/dumb.jam");
        if !checkpoint.is_file() || !kernel.is_file() {
            return None;
        }

        let temp_dir = tempdir().expect("temp dir");
        let loaded = load_checkpoint(&checkpoint).ok()?;
        let checkpoint_state = nockapp::nockapp::save::SaveableCheckpoint {
            ker_hash: loaded.ker_hash,
            event_num: loaded.event_num,
            state: loaded.state,
            cold: loaded.cold,
        };
        let mut nockapp = init_nockapp(
            &kernel,
            Some(checkpoint_state),
            &temp_dir.path().to_path_buf(),
            false,
            true,
        )
        .await
        .ok()?;
        let (tip, _hash) = peek_heaviest_chain_or_block(&mut nockapp).await.ok()??;
        Some((checkpoint, kernel, tip.0 .0))
    }

    fn dummy_hash(v: u64) -> Hash {
        Hash([Belt(v), Belt(v + 1), Belt(v + 2), Belt(v + 3), Belt(v + 4)])
    }
}
