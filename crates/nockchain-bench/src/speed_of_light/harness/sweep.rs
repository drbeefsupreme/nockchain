use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::ValueEnum;
use futures::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time::sleep;

use super::artifacts::{write_json, write_schema_version, write_verdict};
use super::docker::execute_docker_trusted_run;
use super::native::execute_native_trusted_run;
use super::orchestrate::{prepare_output_root, TrustedRunResult};
use super::provenance::BackendRuntimeFacts;
use super::summary::{Validity, Verdict};
use super::{ExecutionRequest, HarnessError, RequestedCase, ResolvedCase, WorkDirMode};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AxisValue {
    Boolean(bool),
    Integer(i64),
    String(String),
}

impl AxisValue {
    fn slug_value(&self) -> String {
        match self {
            Self::Boolean(value) => value.to_string(),
            Self::Integer(value) => value.to_string(),
            Self::String(value) => sanitize_slug(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepMatrix {
    pub base_case: RequestedCase,
    pub axes: BTreeMap<String, Vec<AxisValue>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum CpuProfilerKind {
    Samply,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuProfilerConfig {
    pub kind: CpuProfilerKind,
    pub sample_rate_hz: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SweepOptions {
    pub allow_multi_axis: bool,
}

impl Default for SweepOptions {
    fn default() -> Self {
        Self {
            allow_multi_axis: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpandedCase {
    pub case_index: usize,
    pub case_id: String,
    pub axis_assignments: BTreeMap<String, AxisValue>,
    pub requested_case: RequestedCase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScheduleMode {
    Sequential,
    Interleaved,
    Randomized,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SweepSchedule {
    pub mode: ScheduleMode,
    pub case_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SweepRunOptions {
    pub allow_multi_axis: bool,
    pub schedule_mode: ScheduleMode,
    pub random_seed: Option<u64>,
    pub comparison_markdown: bool,
    pub allow_debug_benchmark: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_profiler: Option<CpuProfilerConfig>,
}

impl Default for SweepRunOptions {
    fn default() -> Self {
        Self {
            allow_multi_axis: false,
            schedule_mode: ScheduleMode::Sequential,
            random_seed: None,
            comparison_markdown: false,
            allow_debug_benchmark: false,
            cpu_profiler: None,
        }
    }
}

#[derive(Debug)]
pub struct SweepCaseRun {
    pub expanded_case: ExpandedCase,
    pub output_root: PathBuf,
    pub result: TrustedRunResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepCaseComparison {
    pub case_id: String,
    pub axis_assignments: BTreeMap<String, AxisValue>,
    pub output_root: PathBuf,
    pub resolved_case: ResolvedCase,
    pub summary: super::summary::RunSummary,
    pub verdict: Verdict,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepComparison {
    pub axis_names: Vec<String>,
    pub case_count: usize,
    pub cases: Vec<SweepCaseComparison>,
    pub invariant_violations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepResult {
    pub expanded_cases: Vec<ExpandedCase>,
    pub schedule: SweepSchedule,
    pub comparison: SweepComparison,
    pub verdict: Verdict,
}

pub trait SweepExecutor {
    fn execute_case<'a>(
        &'a mut self,
        requested_case: RequestedCase,
        output_root: &'a Path,
        allow_debug_benchmark: bool,
        cpu_profiler: Option<CpuProfilerConfig>,
    ) -> futures::future::BoxFuture<'a, Result<TrustedRunResult, HarnessError>>;
}

pub struct HarnessSweepExecutor;

impl SweepExecutor for HarnessSweepExecutor {
    fn execute_case<'a>(
        &'a mut self,
        requested_case: RequestedCase,
        output_root: &'a Path,
        allow_debug_benchmark: bool,
        cpu_profiler: Option<CpuProfilerConfig>,
    ) -> futures::future::BoxFuture<'a, Result<TrustedRunResult, HarnessError>> {
        async move {
            match requested_case.execution.clone() {
                ExecutionRequest::Native => execute_native_trusted_run(
                    requested_case, output_root, allow_debug_benchmark, cpu_profiler,
                )
                .await
                .map(|result| TrustedRunResult {
                    resolved: result.resolved,
                    provenance: result.provenance,
                    summary: result.summary,
                    verdict: result.verdict,
                }),
                ExecutionRequest::Docker { .. } => {
                    execute_docker_trusted_run(
                        requested_case, output_root, allow_debug_benchmark, cpu_profiler,
                    )
                    .await
                }
            }
        }
        .boxed()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SweepMatrixFile {
    Internal(SweepMatrix),
    Spec(SweepMatrixSpec),
}

impl SweepMatrixFile {
    pub fn into_matrix(self) -> Result<SweepMatrix, HarnessError> {
        match self {
            Self::Internal(matrix) => Ok(matrix),
            Self::Spec(spec) => spec.into_matrix(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepMatrixSpec {
    pub benchmark: String,
    pub base: SweepBaseCase,
    pub axes: BTreeMap<String, Vec<AxisValue>>,
}

impl SweepMatrixSpec {
    fn into_matrix(self) -> Result<SweepMatrix, HarnessError> {
        if self.benchmark != "sol-replay" {
            return Err(HarnessError::InvalidRequestedCase(format!(
                "unsupported sweep benchmark `{}`",
                self.benchmark
            )));
        }

        Ok(SweepMatrix {
            base_case: self.base.into_requested_case()?,
            axes: self.axes,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepBaseCase {
    pub fixture: PathBuf,
    #[serde(default)]
    pub blocks: u64,
    #[serde(default)]
    pub skip_genesis: bool,
    #[serde(default = "default_true")]
    pub enable_checkpointing: bool,
    #[serde(default)]
    pub checkpoint_every_blocks: u64,
    #[serde(default)]
    pub profile_memory: bool,
    #[serde(default = "default_profile_interval_ms")]
    pub profile_interval_ms: u64,
    #[serde(default = "default_threads")]
    pub threads: u32,
    #[serde(default = "default_warmup_runs")]
    pub warmup_runs: u32,
    #[serde(default = "default_measured_runs")]
    pub measured_runs: u32,
    #[serde(default = "default_cooldown_secs")]
    pub cooldown_secs: u64,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub mode: SweepModeInput,
}

impl SweepBaseCase {
    fn into_requested_case(self) -> Result<RequestedCase, HarnessError> {
        let mut requested = RequestedCase::native(self.fixture);
        requested.blocks = self.blocks;
        requested.skip_genesis = self.skip_genesis;
        requested.enable_checkpointing = self.enable_checkpointing;
        requested.checkpoint_every_blocks = self.checkpoint_every_blocks;
        requested.profile_memory = self.profile_memory;
        requested.profile_interval_ms = self.profile_interval_ms;
        requested.threads = self.threads;
        requested.warmup_runs = self.warmup_runs;
        requested.measured_runs = self.measured_runs;
        requested.cooldown_secs = self.cooldown_secs;
        requested.label = self.label;
        requested.execution = self.mode.into_execution_request()?;
        Ok(requested)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SweepModeInput {
    #[serde(default)]
    pub native: Option<Value>,
    #[serde(default)]
    pub docker: Option<SweepDockerModeInput>,
}

impl SweepModeInput {
    fn into_execution_request(self) -> Result<ExecutionRequest, HarnessError> {
        match (self.native.is_some(), self.docker) {
            (false, None) | (true, None) => Ok(ExecutionRequest::Native),
            (false, Some(docker)) => Ok(ExecutionRequest::Docker {
                image_tag: docker.image_tag.unwrap_or_default(),
                memory_limit: docker.memory_limit.unwrap_or_default(),
                cpuset: docker.cpuset,
                cpu_quota: docker.cpu_quota,
                cpu_period: docker.cpu_period,
                work_dir_mode: docker.work_dir_mode.unwrap_or(WorkDirMode::DockerTmpfs),
                allow_version_skew: docker.allow_version_skew,
            }),
            (true, Some(_)) => Err(HarnessError::InvalidRequestedCase(
                "sweep base mode must specify either native or docker, not both".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct SweepDockerModeInput {
    #[serde(default)]
    pub image_tag: Option<String>,
    #[serde(default)]
    pub memory_limit: Option<String>,
    #[serde(default)]
    pub cpuset: Option<String>,
    #[serde(default)]
    pub cpu_quota: Option<i64>,
    #[serde(default)]
    pub cpu_period: Option<i64>,
    #[serde(default)]
    pub work_dir_mode: Option<WorkDirMode>,
    #[serde(default)]
    pub allow_version_skew: bool,
}

pub fn parse_matrix_value(value: Value) -> Result<SweepMatrix, HarnessError> {
    serde_json::from_value::<SweepMatrixFile>(value)?.into_matrix()
}

pub fn expand_matrix(
    matrix: &SweepMatrix,
    options: &SweepOptions,
) -> Result<Vec<ExpandedCase>, HarnessError> {
    if matrix.axes.is_empty() {
        return Err(HarnessError::InvalidRequestedCase(
            "sweep matrix requires at least one axis".to_string(),
        ));
    }

    if matrix.axes.len() > 1 && !options.allow_multi_axis {
        return Err(HarnessError::InvalidRequestedCase(
            "multi-axis sweeps require --allow-multi-axis".to_string(),
        ));
    }

    let mut assignments = vec![BTreeMap::new()];
    for (axis_name, values) in &matrix.axes {
        if values.is_empty() {
            return Err(HarnessError::InvalidRequestedCase(format!(
                "sweep axis `{axis_name}` requires at least one value"
            )));
        }
        let mut next = Vec::new();
        for assignment in &assignments {
            for value in values {
                let mut assignment = assignment.clone();
                assignment.insert(axis_name.clone(), value.clone());
                next.push(assignment);
            }
        }
        assignments = next;
    }

    assignments
        .into_iter()
        .enumerate()
        .map(|(case_index, axis_assignments)| {
            let mut requested_case = matrix.base_case.clone();
            apply_axis_assignments(&mut requested_case, &axis_assignments)?;
            let case_slug = axis_assignments
                .iter()
                .map(|(axis, value)| format!("{}_{}", sanitize_slug(axis), value.slug_value()))
                .collect::<Vec<_>>()
                .join("-");
            Ok(ExpandedCase {
                case_index,
                case_id: format!("case-{case_index:03}-{case_slug}"),
                axis_assignments,
                requested_case,
            })
        })
        .collect()
}

pub fn build_schedule(
    expanded_cases: &[ExpandedCase],
    mode: ScheduleMode,
    seed: Option<u64>,
) -> Result<SweepSchedule, HarnessError> {
    let case_ids = match mode {
        ScheduleMode::Sequential => expanded_cases
            .iter()
            .map(|case| case.case_id.clone())
            .collect::<Vec<_>>(),
        ScheduleMode::Interleaved => {
            let mut cases = expanded_cases.to_vec();
            cases.sort_by_key(interleave_sort_key);
            cases.into_iter().map(|case| case.case_id).collect()
        }
        ScheduleMode::Randomized => {
            let mut case_ids = expanded_cases
                .iter()
                .map(|case| case.case_id.clone())
                .collect::<Vec<_>>();
            deterministic_shuffle(&mut case_ids, seed.unwrap_or(0));
            case_ids
        }
    };

    if case_ids.is_empty() {
        return Err(HarnessError::InvalidRequestedCase(
            "sweep schedule requires at least one expanded case".to_string(),
        ));
    }

    Ok(SweepSchedule { mode, case_ids })
}

fn apply_axis_assignments(
    requested_case: &mut RequestedCase,
    axis_assignments: &BTreeMap<String, AxisValue>,
) -> Result<(), HarnessError> {
    for (axis, value) in axis_assignments {
        if apply_general_axis(requested_case, axis, value)? {
            continue;
        }

        if is_docker_axis(axis) {
            apply_docker_axis(requested_case, axis, value)?;
            continue;
        }

        return Err(HarnessError::InvalidRequestedCase(format!(
            "unsupported sweep axis `{axis}`"
        )));
    }

    Ok(())
}

fn apply_general_axis(
    requested_case: &mut RequestedCase,
    axis: &str,
    value: &AxisValue,
) -> Result<bool, HarnessError> {
    match axis {
        "threads" => requested_case.threads = integer_to_u32(axis, value)?,
        "blocks" => requested_case.blocks = integer_to_u64(axis, value)?,
        "skip_genesis" => requested_case.skip_genesis = boolean_value(axis, value)?,
        "enable_checkpointing" => requested_case.enable_checkpointing = boolean_value(axis, value)?,
        "checkpoint_every_blocks" => {
            requested_case.checkpoint_every_blocks = integer_to_u64(axis, value)?
        }
        "profile_memory" => requested_case.profile_memory = boolean_value(axis, value)?,
        "profile_interval_ms" => requested_case.profile_interval_ms = integer_to_u64(axis, value)?,
        "warmup_runs" => requested_case.warmup_runs = integer_to_u32(axis, value)?,
        "measured_runs" => requested_case.measured_runs = integer_to_u32(axis, value)?,
        "cooldown_secs" => requested_case.cooldown_secs = integer_to_u64(axis, value)?,
        "fixture" => requested_case.fixture_path = path_value(axis, value)?,
        "label" => requested_case.label = Some(string_value(axis, value)?),
        _ => return Ok(false),
    }

    Ok(true)
}

fn integer_to_u32(axis: &str, value: &AxisValue) -> Result<u32, HarnessError> {
    let value = integer_value(axis, value)?;
    u32::try_from(value).map_err(|_| {
        HarnessError::InvalidRequestedCase(format!(
            "sweep axis `{axis}` requires a non-negative 32-bit integer"
        ))
    })
}

fn integer_to_u64(axis: &str, value: &AxisValue) -> Result<u64, HarnessError> {
    let value = integer_value(axis, value)?;
    u64::try_from(value).map_err(|_| {
        HarnessError::InvalidRequestedCase(format!(
            "sweep axis `{axis}` requires a non-negative 64-bit integer"
        ))
    })
}

fn integer_value(axis: &str, value: &AxisValue) -> Result<i64, HarnessError> {
    match value {
        AxisValue::Integer(value) => Ok(*value),
        _ => Err(HarnessError::InvalidRequestedCase(format!(
            "sweep axis `{axis}` requires an integer value"
        ))),
    }
}

fn string_value(_axis: &str, value: &AxisValue) -> Result<String, HarnessError> {
    match value {
        AxisValue::String(value) => Ok(value.clone()),
        AxisValue::Integer(value) => Ok(value.to_string()),
        AxisValue::Boolean(value) => Ok(value.to_string()),
    }
}

fn path_value(axis: &str, value: &AxisValue) -> Result<PathBuf, HarnessError> {
    Ok(PathBuf::from(string_value(axis, value)?))
}

fn boolean_value(axis: &str, value: &AxisValue) -> Result<bool, HarnessError> {
    match value {
        AxisValue::Boolean(value) => Ok(*value),
        _ => Err(HarnessError::InvalidRequestedCase(format!(
            "sweep axis `{axis}` requires a boolean value"
        ))),
    }
}

fn work_dir_mode_value(axis: &str, value: &AxisValue) -> Result<WorkDirMode, HarnessError> {
    let normalized = string_value(axis, value)?
        .replace(['_', '-'], "")
        .to_ascii_lowercase();
    match normalized.as_str() {
        "hostbind" => Ok(WorkDirMode::HostBind),
        "dockervolume" => Ok(WorkDirMode::DockerVolume),
        "dockertmpfs" => Ok(WorkDirMode::DockerTmpfs),
        _ => Err(HarnessError::InvalidRequestedCase(format!(
            "sweep axis `{axis}` requires a valid work dir mode"
        ))),
    }
}

fn is_docker_axis(axis: &str) -> bool {
    matches!(
        axis,
        "image_tag"
            | "memory_limit"
            | "cpuset"
            | "cpu_quota"
            | "cpu_period"
            | "work_dir_mode"
            | "allow_version_skew"
    )
}

fn apply_docker_axis(
    requested_case: &mut RequestedCase,
    axis: &str,
    value: &AxisValue,
) -> Result<(), HarnessError> {
    match &mut requested_case.execution {
        ExecutionRequest::Docker {
            image_tag,
            memory_limit,
            cpuset,
            cpu_quota,
            cpu_period,
            work_dir_mode,
            allow_version_skew,
        } => match axis {
            "image_tag" => *image_tag = string_value(axis, value)?,
            "memory_limit" => *memory_limit = string_value(axis, value)?,
            "cpuset" => *cpuset = Some(string_value(axis, value)?),
            "cpu_quota" => *cpu_quota = Some(integer_value(axis, value)?),
            "cpu_period" => *cpu_period = Some(integer_value(axis, value)?),
            "work_dir_mode" => *work_dir_mode = work_dir_mode_value(axis, value)?,
            "allow_version_skew" => *allow_version_skew = boolean_value(axis, value)?,
            other => {
                return Err(HarnessError::InvalidRequestedCase(format!(
                    "unsupported sweep axis `{other}`"
                )));
            }
        },
        ExecutionRequest::Native => {
            return Err(HarnessError::InvalidRequestedCase(format!(
                "sweep axis `{axis}` requires Docker execution"
            )));
        }
    }

    Ok(())
}

fn interleave_sort_key(expanded_case: &ExpandedCase) -> Vec<String> {
    let mut reversed = expanded_case
        .axis_assignments
        .iter()
        .rev()
        .map(|(axis, value)| format!("{axis}={}", value.slug_value()))
        .collect::<Vec<_>>();
    reversed.push(format!("{:06}", expanded_case.case_index));
    reversed
}

fn deterministic_shuffle<T>(values: &mut [T], seed: u64) {
    if values.len() < 2 {
        return;
    }

    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    for index in (1..values.len()).rev() {
        state = xorshift64(state);
        let swap_index = (state as usize) % (index + 1);
        values.swap(index, swap_index);
    }
}

fn xorshift64(mut state: u64) -> u64 {
    if state == 0 {
        state = 0x4d59_5df4_d0f3_3173;
    }
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

fn sanitize_slug(input: &str) -> String {
    let mut slug = String::with_capacity(input.len());
    let mut previous_separator = false;
    for ch in input.chars() {
        let normalized = if ch.is_ascii_alphanumeric() || ch == '_' {
            ch.to_ascii_lowercase()
        } else {
            '-'
        };
        if normalized == '-' {
            if !previous_separator {
                slug.push(normalized);
            }
            previous_separator = true;
        } else {
            slug.push(normalized);
            previous_separator = false;
        }
    }
    slug.trim_matches('-').to_string()
}

pub async fn execute_sweep<E: SweepExecutor>(
    matrix_json: &Value,
    matrix: SweepMatrix,
    output_root: &Path,
    options: &SweepRunOptions,
    executor: &mut E,
) -> Result<SweepResult, HarnessError> {
    validate_sweep_profiling_support(&matrix, options)?;
    prepare_output_root(output_root)?;
    std::fs::create_dir_all(output_root)?;
    write_schema_version(output_root)?;

    let expanded_cases = expand_matrix(
        &matrix,
        &SweepOptions {
            allow_multi_axis: options.allow_multi_axis,
        },
    )?;
    let schedule = build_schedule(&expanded_cases, options.schedule_mode, options.random_seed)?;

    write_json(output_root.join("matrix.json"), matrix_json)?;
    write_json(output_root.join("matrix_expanded.json"), &expanded_cases)?;
    write_json(output_root.join("schedule.json"), &schedule)?;

    let cases_root = output_root.join("cases");
    std::fs::create_dir_all(&cases_root)?;
    let expanded_by_id = expanded_cases
        .iter()
        .cloned()
        .map(|case| (case.case_id.clone(), case))
        .collect::<BTreeMap<_, _>>();

    let mut case_runs = Vec::with_capacity(schedule.case_ids.len());
    for (index, case_id) in schedule.case_ids.iter().enumerate() {
        let expanded_case = expanded_by_id.get(case_id).cloned().ok_or_else(|| {
            HarnessError::InvalidRequestedCase(format!("unknown scheduled case `{case_id}`"))
        })?;
        let case_output_root = cases_root.join(case_id);
        std::fs::create_dir_all(&case_output_root)?;
        let result = match executor
            .execute_case(
                expanded_case.requested_case.clone(),
                &case_output_root,
                options.allow_debug_benchmark,
                options.cpu_profiler.clone(),
            )
            .await
        {
            Ok(result) => result,
            Err(error) => {
                persist_failed_sweep_verdict(
                    output_root,
                    format!("case {case_id} failed: {error}"),
                )?;
                return Err(error);
            }
        };
        case_runs.push(SweepCaseRun {
            expanded_case: expanded_case.clone(),
            output_root: case_output_root,
            result,
        });

        if index + 1 < schedule.case_ids.len() && expanded_case.requested_case.cooldown_secs > 0 {
            sleep(Duration::from_secs(
                expanded_case.requested_case.cooldown_secs,
            ))
            .await;
        }
    }

    let comparison = build_comparison(&case_runs)?;
    write_json(output_root.join("comparison.json"), &comparison)?;
    if options.comparison_markdown {
        std::fs::write(
            output_root.join("comparison.md"),
            render_comparison_markdown(&comparison),
        )?;
    }

    let verdict = derive_sweep_verdict(&comparison);
    write_verdict(output_root, &verdict)?;

    Ok(SweepResult {
        expanded_cases,
        schedule,
        comparison,
        verdict,
    })
}

fn validate_sweep_profiling_support(
    _matrix: &SweepMatrix,
    _options: &SweepRunOptions,
) -> Result<(), HarnessError> {
    Ok(())
}

pub fn build_comparison(case_runs: &[SweepCaseRun]) -> Result<SweepComparison, HarnessError> {
    if case_runs.is_empty() {
        return Err(HarnessError::InvalidRequestedCase(
            "sweep comparison requires at least one case".to_string(),
        ));
    }

    let axis_names = case_runs[0]
        .expanded_case
        .axis_assignments
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    let axis_name_set = axis_names.iter().cloned().collect::<BTreeSet<_>>();
    let baseline = &case_runs[0].result;
    let mut invariant_violations = Vec::new();

    for case_run in case_runs.iter().skip(1) {
        let current = &case_run.result;
        macro_rules! compare_case_invariant {
            ($axis:literal, $field:literal, $left:expr, $right:expr) => {
                compare_invariant(
                    &mut invariant_violations, &axis_name_set, $axis, $field, &$left, &$right,
                    &case_run.expanded_case.case_id,
                );
            };
        }
        compare_case_invariant!(
            "enable_checkpointing", "enable_checkpointing",
            baseline.resolved.requested.enable_checkpointing,
            current.resolved.requested.enable_checkpointing
        );
        compare_case_invariant!(
            "fixture", "fixture_sha256_hex", baseline.resolved.fixture_sha256_hex,
            current.resolved.fixture_sha256_hex
        );
        compare_case_invariant!(
            "fixture", "fixture_manifest", baseline.resolved.fixture_manifest,
            current.resolved.fixture_manifest
        );
        compare_case_invariant!(
            "threads", "threads", baseline.resolved.requested.threads,
            current.resolved.requested.threads
        );
        compare_case_invariant!(
            "checkpoint_every_blocks", "checkpoint_every_blocks",
            baseline.resolved.requested.checkpoint_every_blocks,
            current.resolved.requested.checkpoint_every_blocks
        );
        compare_case_invariant!(
            "profile_memory", "profile_memory", baseline.resolved.requested.profile_memory,
            current.resolved.requested.profile_memory
        );
        compare_case_invariant!(
            "blocks", "blocks", baseline.resolved.requested.blocks,
            current.resolved.requested.blocks
        );
        compare_case_invariant!(
            "skip_genesis", "skip_genesis", baseline.resolved.requested.skip_genesis,
            current.resolved.requested.skip_genesis
        );
        compare_case_invariant!(
            "profile_interval_ms", "profile_interval_ms",
            baseline.resolved.requested.profile_interval_ms,
            current.resolved.requested.profile_interval_ms
        );
        compare_case_invariant!(
            "warmup_runs", "warmup_runs", baseline.resolved.requested.warmup_runs,
            current.resolved.requested.warmup_runs
        );
        compare_case_invariant!(
            "measured_runs", "measured_runs", baseline.resolved.requested.measured_runs,
            current.resolved.requested.measured_runs
        );
        compare_case_invariant!(
            "cooldown_secs", "cooldown_secs", baseline.resolved.requested.cooldown_secs,
            current.resolved.requested.cooldown_secs
        );
        compare_case_invariant!(
            "version", "binary.version", baseline.resolved.binary.version,
            current.resolved.binary.version
        );
        compare_case_invariant!(
            "git_commit", "binary.git_commit", baseline.resolved.binary.git_commit,
            current.resolved.binary.git_commit
        );
        compare_case_invariant!(
            "build_profile", "binary.build_profile", baseline.resolved.binary.build_profile,
            current.resolved.binary.build_profile
        );
        compare_case_invariant!(
            "git_commit",
            "provenance.git.commit",
            baseline
                .provenance
                .git
                .as_ref()
                .and_then(|git| git.commit.clone()),
            current
                .provenance
                .git
                .as_ref()
                .and_then(|git| git.commit.clone())
        );
        compare_case_invariant!(
            "git_dirty",
            "provenance.git.dirty",
            baseline.provenance.git.as_ref().map(|git| git.dirty),
            current.provenance.git.as_ref().map(|git| git.dirty)
        );
        compare_case_invariant!(
            "host_identity", "provenance.host", baseline.provenance.host, current.provenance.host
        );
        compare_resolved_docker_invariants(
            &mut invariant_violations,
            &axis_name_set,
            baseline.resolved.docker.as_ref(),
            current.resolved.docker.as_ref(),
            &case_run.expanded_case.case_id,
        );
        compare_backend_invariants(
            &mut invariant_violations, &axis_name_set, &baseline.provenance.backend,
            &current.provenance.backend, &case_run.expanded_case.case_id,
        );
    }

    let cases = case_runs
        .iter()
        .map(|case_run| SweepCaseComparison {
            case_id: case_run.expanded_case.case_id.clone(),
            axis_assignments: case_run.expanded_case.axis_assignments.clone(),
            output_root: case_run.output_root.clone(),
            resolved_case: case_run.result.resolved.clone(),
            summary: case_run.result.summary.clone(),
            verdict: case_run.result.verdict.clone(),
        })
        .collect::<Vec<_>>();

    Ok(SweepComparison {
        axis_names,
        case_count: cases.len(),
        cases,
        invariant_violations,
    })
}

pub fn derive_sweep_verdict(comparison: &SweepComparison) -> Verdict {
    let mut invalid_reasons = comparison.invariant_violations.clone();
    let mut partial_reasons = Vec::new();

    for case in &comparison.cases {
        match &case.verdict.validity {
            Validity::Valid => {}
            Validity::Partial { reasons } => {
                partial_reasons.extend(
                    reasons
                        .iter()
                        .map(|reason| format!("{}: {reason}", case.case_id)),
                );
            }
            Validity::Invalid { reasons } => {
                invalid_reasons.extend(
                    reasons
                        .iter()
                        .map(|reason| format!("{}: {reason}", case.case_id)),
                );
            }
        }
    }

    if !invalid_reasons.is_empty() {
        Verdict {
            validity: Validity::Invalid {
                reasons: invalid_reasons,
            },
        }
    } else if !partial_reasons.is_empty() {
        Verdict {
            validity: Validity::Partial {
                reasons: partial_reasons,
            },
        }
    } else {
        Verdict {
            validity: Validity::Valid,
        }
    }
}

fn compare_invariant<T: PartialEq>(
    invariant_violations: &mut Vec<String>,
    axis_names: &BTreeSet<String>,
    axis_name: &str,
    field_name: &str,
    baseline: &T,
    current: &T,
    case_id: &str,
) {
    compare_invariant_any_axis(
        invariant_violations,
        axis_names,
        &[axis_name],
        field_name,
        baseline,
        current,
        case_id,
    );
}

fn compare_invariant_any_axis<T: PartialEq>(
    invariant_violations: &mut Vec<String>,
    axis_names: &BTreeSet<String>,
    axis_names_to_ignore: &[&str],
    field_name: &str,
    baseline: &T,
    current: &T,
    case_id: &str,
) {
    if axis_names_to_ignore
        .iter()
        .any(|axis_name| axis_names.contains(*axis_name))
        || baseline == current
    {
        return;
    }
    invariant_violations.push(format!(
        "case {case_id} changed non-axis field `{field_name}`"
    ));
}

fn compare_resolved_docker_invariants(
    invariant_violations: &mut Vec<String>,
    axis_names: &BTreeSet<String>,
    baseline: Option<&crate::speed_of_light::harness::case::DockerResolvedConfig>,
    current: Option<&crate::speed_of_light::harness::case::DockerResolvedConfig>,
    case_id: &str,
) {
    match (baseline, current) {
        (None, None) => {}
        (Some(baseline), Some(current)) => {
            compare_invariant(
                invariant_violations, axis_names, "cpuset", "docker.cpuset", &baseline.cpuset,
                &current.cpuset, case_id,
            );
            compare_invariant(
                invariant_violations, axis_names, "cpu_quota", "docker.cpu_quota",
                &baseline.cpu_quota, &current.cpu_quota, case_id,
            );
            compare_invariant(
                invariant_violations, axis_names, "cpu_period", "docker.cpu_period",
                &baseline.cpu_period, &current.cpu_period, case_id,
            );
            compare_invariant(
                invariant_violations, axis_names, "work_dir_mode", "docker.work_dir_mode",
                &baseline.work_dir_mode, &current.work_dir_mode, case_id,
            );
            compare_invariant(
                invariant_violations, axis_names, "allow_version_skew",
                "docker.allow_version_skew", &baseline.allow_version_skew,
                &current.allow_version_skew, case_id,
            );
        }
        _ => invariant_violations.push(format!(
            "case {case_id} changed non-axis field `resolved.docker`"
        )),
    }
}

fn compare_backend_invariants(
    invariant_violations: &mut Vec<String>,
    axis_names: &BTreeSet<String>,
    baseline: &BackendRuntimeFacts,
    current: &BackendRuntimeFacts,
    case_id: &str,
) {
    match (baseline, current) {
        (BackendRuntimeFacts::Native, BackendRuntimeFacts::Native) => {}
        (
            BackendRuntimeFacts::Docker {
                host_binary: baseline_host_binary,
                container_binary: baseline_container_binary,
                image_digest: baseline_image_digest,
                realized_cpuset: baseline_cpuset,
                realized_cpu_max: baseline_cpu_max,
                ..
            },
            BackendRuntimeFacts::Docker {
                host_binary: current_host_binary,
                container_binary: current_container_binary,
                image_digest: current_image_digest,
                realized_cpuset: current_cpuset,
                realized_cpu_max: current_cpu_max,
                ..
            },
        ) => {
            compare_invariant_any_axis(
                invariant_violations,
                axis_names,
                &["image_tag"],
                "backend.image_digest",
                baseline_image_digest,
                current_image_digest,
                case_id,
            );
            compare_invariant_any_axis(
                invariant_violations,
                axis_names,
                &[],
                "backend.host_binary",
                baseline_host_binary,
                current_host_binary,
                case_id,
            );
            compare_invariant_any_axis(
                invariant_violations,
                axis_names,
                &[],
                "backend.container_binary",
                baseline_container_binary,
                current_container_binary,
                case_id,
            );
            compare_invariant(
                invariant_violations, axis_names, "cpuset", "backend.realized_cpuset",
                baseline_cpuset, current_cpuset, case_id,
            );
            compare_invariant_any_axis(
                invariant_violations,
                axis_names,
                &["cpu_quota", "cpu_period"],
                "backend.realized_cpu_max",
                baseline_cpu_max,
                current_cpu_max,
                case_id,
            );
        }
        _ => invariant_violations.push(format!(
            "case {case_id} changed non-axis field `execution_mode`"
        )),
    }
}

fn render_comparison_markdown(comparison: &SweepComparison) -> String {
    let mut output = String::from("# SOL Sweep Comparison\n\n");
    output.push_str("| Case | Axes | Verdict | Throughput Median |\n");
    output.push_str("| --- | --- | --- | --- |\n");
    for case in &comparison.cases {
        let axes = case
            .axis_assignments
            .iter()
            .map(|(axis, value)| format!("{axis}={}", value.slug_value()))
            .collect::<Vec<_>>()
            .join(", ");
        let verdict = match &case.verdict.validity {
            Validity::Valid => "Valid".to_string(),
            Validity::Partial { .. } => "Partial".to_string(),
            Validity::Invalid { .. } => "Invalid".to_string(),
        };
        let throughput = case
            .summary
            .throughput_blocks_per_second
            .as_ref()
            .map(|stats| format!("{:.2}", stats.median))
            .unwrap_or_else(|| "-".to_string());
        output.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            case.case_id, axes, verdict, throughput
        ));
    }
    output
}

fn persist_failed_sweep_verdict(output_root: &Path, reason: String) -> Result<(), HarnessError> {
    write_verdict(
        output_root,
        &Verdict {
            validity: Validity::Invalid {
                reasons: vec![reason],
            },
        },
    )
}

fn default_true() -> bool {
    true
}

fn default_profile_interval_ms() -> u64 {
    500
}

fn default_threads() -> u32 {
    1
}

fn default_warmup_runs() -> u32 {
    1
}

fn default_measured_runs() -> u32 {
    5
}

fn default_cooldown_secs() -> u64 {
    10
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    use futures::FutureExt;
    use tempfile::tempdir;

    use super::*;
    use crate::speed_of_light::fixture::SolFixtureManifest;
    use crate::speed_of_light::harness::case::{
        BinaryIdentity, ExecutionConfig, ExecutionRequest, ResolvedCase, WorkDirMode,
    };
    use crate::speed_of_light::harness::orchestrate::TrustedRunResult;
    use crate::speed_of_light::harness::provenance::{
        BackendRuntimeFacts, HostIdentity, Provenance,
    };
    use crate::speed_of_light::harness::summary::{RunSummary, Validity, ValueStats, Verdict};
    use crate::speed_of_light::harness::SCHEMA_VERSION;
    use crate::speed_of_light::types::SolHeight;

    struct FakeExecutor {
        seen_paths: Arc<Mutex<Vec<PathBuf>>>,
        seen_requested_cases: Arc<Mutex<Vec<RequestedCase>>>,
        seen_cpu_profilers: Arc<Mutex<Vec<Option<CpuProfilerConfig>>>>,
        results: Vec<Result<TrustedRunResult, HarnessError>>,
    }

    impl FakeExecutor {
        fn new(results: Vec<Result<TrustedRunResult, HarnessError>>) -> Self {
            Self {
                seen_paths: Arc::new(Mutex::new(Vec::new())),
                seen_requested_cases: Arc::new(Mutex::new(Vec::new())),
                seen_cpu_profilers: Arc::new(Mutex::new(Vec::new())),
                results,
            }
        }

        fn seen_paths(&self) -> Arc<Mutex<Vec<PathBuf>>> {
            Arc::clone(&self.seen_paths)
        }

        fn seen_requested_cases(&self) -> Arc<Mutex<Vec<RequestedCase>>> {
            Arc::clone(&self.seen_requested_cases)
        }

        fn seen_cpu_profilers(&self) -> Arc<Mutex<Vec<Option<CpuProfilerConfig>>>> {
            Arc::clone(&self.seen_cpu_profilers)
        }
    }

    impl SweepExecutor for FakeExecutor {
        fn execute_case<'a>(
            &'a mut self,
            requested_case: RequestedCase,
            output_root: &'a Path,
            _allow_debug_benchmark: bool,
            cpu_profiler: Option<CpuProfilerConfig>,
        ) -> futures::future::BoxFuture<'a, Result<TrustedRunResult, HarnessError>> {
            self.seen_paths
                .lock()
                .expect("seen paths lock")
                .push(output_root.to_path_buf());
            self.seen_requested_cases
                .lock()
                .expect("requested cases lock")
                .push(requested_case.clone());
            self.seen_cpu_profilers
                .lock()
                .expect("cpu profilers lock")
                .push(cpu_profiler.clone());
            let result = self.results.remove(0);
            async move { result }.boxed()
        }
    }

    fn fixture_manifest() -> SolFixtureManifest {
        SolFixtureManifest {
            format_version: 2,
            source_archive_path: "archive.solarch".to_string(),
            source_archive_event_num: 1,
            derived_checkpoint_height: SolHeight(10),
            derived_checkpoint_event_num: 10,
            archive_start_height: SolHeight(11),
            archive_end_height: SolHeight(20),
            include_mempool: false,
            chunk_size: 8,
            kernel_hash_hex: "kernel".to_string(),
            checkpoint_hash_hex: "checkpoint".to_string(),
            archive_hash_hex: "archive".to_string(),
        }
    }

    fn trusted_run_result(
        fixture_sha256_hex: &str,
        threads: u32,
        case_validity: Validity,
    ) -> TrustedRunResult {
        let requested = RequestedCase {
            threads,
            warmup_runs: 0,
            measured_runs: 3,
            cooldown_secs: 0,
            ..RequestedCase::native(PathBuf::from("fixture.soltest"))
        };
        let resolved = ResolvedCase {
            schema_version: SCHEMA_VERSION.to_string(),
            requested: requested.clone(),
            absolute_fixture_path: PathBuf::from("/tmp/fixture.soltest"),
            fixture_sha256_hex: fixture_sha256_hex.to_string(),
            fixture_manifest: fixture_manifest(),
            execution_config: ExecutionConfig::default(),
            binary: BinaryIdentity {
                version: "0.1.0".to_string(),
                build_profile: "release".to_string(),
                git_commit: Some("abc123".to_string()),
            },
            docker: Some(crate::speed_of_light::harness::case::DockerResolvedConfig {
                image_tag: "nockchain-bench:test".to_string(),
                requested_memory_limit_bytes: 4 * 1024 * 1024 * 1024,
                cpuset: Some("0-3".to_string()),
                cpu_quota: Some(200_000),
                cpu_period: Some(100_000),
                work_dir_mode: WorkDirMode::DockerTmpfs,
                allow_version_skew: false,
            }),
        };
        TrustedRunResult {
            resolved: resolved.clone(),
            provenance: Provenance {
                schema_version: SCHEMA_VERSION.to_string(),
                capture_timestamp_ms: 1,
                host: HostIdentity {
                    hostname: Some("host".to_string()),
                    os: "linux".to_string(),
                    arch: "x86_64".to_string(),
                    kernel: Some("6.0".to_string()),
                    cpu_count: 8,
                    total_memory_bytes: Some(32 * 1024 * 1024 * 1024),
                    cpu_model: Some("cpu".to_string()),
                },
                git: Some(crate::speed_of_light::harness::provenance::GitIdentity {
                    commit: Some("abc123".to_string()),
                    branch: Some("main".to_string()),
                    commit_date: Some("2026-03-11T00:00:00Z".to_string()),
                    dirty: false,
                }),
                backend: BackendRuntimeFacts::Docker {
                    host_binary: resolved.binary.clone(),
                    container_binary: resolved.binary.clone(),
                    image_tag: "nockchain-bench:test".to_string(),
                    image_digest: "sha256:digest".to_string(),
                    container_id: format!("container-{threads}"),
                    docker_engine_version: "28.0".to_string(),
                    docker_context: "desktop-linux".to_string(),
                    cgroup_version: "2".to_string(),
                    storage_driver: "overlay2".to_string(),
                    realized_memory_max: 4 * 1024 * 1024 * 1024,
                    realized_memory_current: 256 * 1024 * 1024,
                    realized_cpuset: Some("0-3".to_string()),
                    realized_cpu_max: Some("200000 100000".to_string()),
                },
                binary: resolved.binary.clone(),
                fixture_path: resolved.absolute_fixture_path.clone(),
                fixture_sha256_hex: resolved.fixture_sha256_hex.clone(),
                fixture_manifest: resolved.fixture_manifest.clone(),
            },
            summary: RunSummary {
                measured_runs_requested: 3,
                measured_runs_succeeded: 3,
                failed_runs: Vec::new(),
                throughput_blocks_per_second: Some(ValueStats {
                    median: 100.0 + threads as f64,
                    min: 90.0,
                    max: 110.0,
                    mad: 5.0,
                    stddev: 3.0,
                    cv: 0.03,
                    values: vec![90.0, 100.0 + threads as f64, 110.0],
                }),
                init_time_secs: None,
                total_replay_time_secs: None,
                average_block_time_ms: None,
                failed_pokes: None,
                checkpoint_count: None,
                average_checkpoint_time_secs: None,
                peak_process_rss_bytes: None,
                minor_faults_total: None,
                major_faults_total: None,
            },
            verdict: Verdict {
                validity: case_validity,
            },
        }
    }

    #[test]
    fn sweep_comparison_marks_non_axis_drift_invalid() {
        let expanded_cases = vec![
            ExpandedCase {
                case_index: 0,
                case_id: "case-000-threads_1".to_string(),
                axis_assignments: BTreeMap::from([("threads".to_string(), AxisValue::Integer(1))]),
                requested_case: RequestedCase::native(PathBuf::from("fixture.soltest")),
            },
            ExpandedCase {
                case_index: 1,
                case_id: "case-001-threads_2".to_string(),
                axis_assignments: BTreeMap::from([("threads".to_string(), AxisValue::Integer(2))]),
                requested_case: RequestedCase::native(PathBuf::from("fixture.soltest")),
            },
        ];
        let case_runs = vec![
            SweepCaseRun {
                expanded_case: expanded_cases[0].clone(),
                output_root: PathBuf::from("/tmp/cases/case-000-threads_1"),
                result: trusted_run_result("fixture-a", 1, Validity::Valid),
            },
            SweepCaseRun {
                expanded_case: expanded_cases[1].clone(),
                output_root: PathBuf::from("/tmp/cases/case-001-threads_2"),
                result: trusted_run_result("fixture-b", 2, Validity::Valid),
            },
        ];

        let comparison = build_comparison(&case_runs).expect("comparison");
        let verdict = derive_sweep_verdict(&comparison);

        assert_eq!(comparison.invariant_violations.len(), 1);
        assert!(comparison.invariant_violations[0].contains("fixture_sha256_hex"));
        assert!(matches!(verdict.validity, Validity::Invalid { .. }));
    }

    #[test]
    fn sweep_comparison_flags_missing_non_axis_invariants() {
        let expanded_cases = vec![
            ExpandedCase {
                case_index: 0,
                case_id: "case-000-threads_1".to_string(),
                axis_assignments: BTreeMap::from([("threads".to_string(), AxisValue::Integer(1))]),
                requested_case: RequestedCase::native(PathBuf::from("fixture.soltest")),
            },
            ExpandedCase {
                case_index: 1,
                case_id: "case-001-threads_2".to_string(),
                axis_assignments: BTreeMap::from([("threads".to_string(), AxisValue::Integer(2))]),
                requested_case: RequestedCase::native(PathBuf::from("fixture.soltest")),
            },
        ];
        let baseline = trusted_run_result("fixture-a", 1, Validity::Valid);
        let mut drifted = trusted_run_result("fixture-a", 2, Validity::Valid);
        drifted.resolved.requested.enable_checkpointing = false;
        drifted.resolved.binary.version = "0.2.0".to_string();
        drifted.resolved.binary.git_commit = Some("def456".to_string());
        drifted.provenance.binary.git_commit = Some("def456".to_string());
        drifted
            .resolved
            .docker
            .as_mut()
            .expect("docker config")
            .work_dir_mode = WorkDirMode::HostBind;
        drifted
            .resolved
            .docker
            .as_mut()
            .expect("docker config")
            .allow_version_skew = true;
        drifted.provenance.git.as_mut().expect("git identity").dirty = true;
        if let BackendRuntimeFacts::Docker {
            host_binary,
            container_binary,
            ..
        } = &mut drifted.provenance.backend
        {
            host_binary.version = "0.2.0".to_string();
            container_binary.version = "0.2.0".to_string();
        }

        let comparison = build_comparison(&[
            SweepCaseRun {
                expanded_case: expanded_cases[0].clone(),
                output_root: PathBuf::from("/tmp/cases/case-000-threads_1"),
                result: baseline,
            },
            SweepCaseRun {
                expanded_case: expanded_cases[1].clone(),
                output_root: PathBuf::from("/tmp/cases/case-001-threads_2"),
                result: drifted,
            },
        ])
        .expect("comparison");

        assert!(comparison
            .invariant_violations
            .iter()
            .any(|reason| reason.contains("enable_checkpointing")));
        assert!(comparison
            .invariant_violations
            .iter()
            .any(|reason| reason.contains("binary.version")));
        assert!(comparison
            .invariant_violations
            .iter()
            .any(|reason| reason.contains("binary.git_commit")));
        assert!(comparison
            .invariant_violations
            .iter()
            .any(|reason| reason.contains("provenance.git.dirty")));
        assert!(comparison
            .invariant_violations
            .iter()
            .any(|reason| reason.contains("docker.work_dir_mode")));
        assert!(comparison
            .invariant_violations
            .iter()
            .any(|reason| reason.contains("docker.allow_version_skew")));
        assert!(comparison
            .invariant_violations
            .iter()
            .any(|reason| reason.contains("backend.host_binary")));
        assert!(comparison
            .invariant_violations
            .iter()
            .any(|reason| reason.contains("backend.container_binary")));
    }

    #[test]
    fn sweep_comparison_allows_cpu_period_axis_to_change_realized_cpu_max() {
        let baseline = trusted_run_result("fixture-a", 1, Validity::Valid);
        let mut varied = trusted_run_result("fixture-a", 1, Validity::Valid);
        varied
            .resolved
            .docker
            .as_mut()
            .expect("docker config")
            .cpu_period = Some(50_000);
        if let BackendRuntimeFacts::Docker {
            realized_cpu_max, ..
        } = &mut varied.provenance.backend
        {
            *realized_cpu_max = Some("200000 50000".to_string());
        }

        let comparison = build_comparison(&[
            SweepCaseRun {
                expanded_case: ExpandedCase {
                    case_index: 0,
                    case_id: "case-000-cpu_period_100000".to_string(),
                    axis_assignments: BTreeMap::from([(
                        "cpu_period".to_string(),
                        AxisValue::Integer(100_000),
                    )]),
                    requested_case: RequestedCase::native(PathBuf::from("fixture.soltest")),
                },
                output_root: PathBuf::from("/tmp/cases/case-000-cpu_period_100000"),
                result: baseline,
            },
            SweepCaseRun {
                expanded_case: ExpandedCase {
                    case_index: 1,
                    case_id: "case-001-cpu_period_50000".to_string(),
                    axis_assignments: BTreeMap::from([(
                        "cpu_period".to_string(),
                        AxisValue::Integer(50_000),
                    )]),
                    requested_case: RequestedCase::native(PathBuf::from("fixture.soltest")),
                },
                output_root: PathBuf::from("/tmp/cases/case-001-cpu_period_50000"),
                result: varied,
            },
        ])
        .expect("comparison");

        assert!(!comparison
            .invariant_violations
            .iter()
            .any(|reason| reason.contains("backend.realized_cpu_max")));
    }

    #[tokio::test]
    async fn sweep_profiling_metadata() {
        let tempdir = tempdir().expect("tempdir");
        let output_root = tempdir.path().join("sweep");
        let matrix = SweepMatrix {
            base_case: RequestedCase {
                cooldown_secs: 0,
                ..RequestedCase::native(PathBuf::from("fixture.soltest"))
            },
            axes: BTreeMap::from([(
                "threads".to_string(),
                vec![AxisValue::Integer(1), AxisValue::Integer(2)],
            )]),
        };
        let matrix_json = serde_json::to_value(&matrix).expect("matrix json");
        let baseline_runs = vec![
            Ok(trusted_run_result("fixture-a", 1, Validity::Valid)),
            Ok(trusted_run_result("fixture-a", 2, Validity::Valid)),
        ];
        let profiled_runs = vec![
            Ok(trusted_run_result("fixture-a", 1, Validity::Valid)),
            Ok(trusted_run_result("fixture-a", 2, Validity::Valid)),
        ];

        let mut baseline_executor = FakeExecutor::new(baseline_runs);
        let baseline_requested_cases = baseline_executor.seen_requested_cases();
        let baseline = execute_sweep(
            &matrix_json,
            matrix.clone(),
            &output_root.join("baseline"),
            &SweepRunOptions {
                cpu_profiler: None,
                ..SweepRunOptions::default()
            },
            &mut baseline_executor,
        )
        .await
        .expect("baseline sweep");

        let mut profiled_executor = FakeExecutor::new(profiled_runs);
        let profiled_requested_cases = profiled_executor.seen_requested_cases();
        let profiled = execute_sweep(
            &matrix_json,
            matrix,
            &output_root.join("profiled"),
            &SweepRunOptions {
                cpu_profiler: Some(CpuProfilerConfig {
                    kind: CpuProfilerKind::Samply,
                    sample_rate_hz: 1000,
                }),
                ..SweepRunOptions::default()
            },
            &mut profiled_executor,
        )
        .await
        .expect("profiled sweep");

        assert_eq!(baseline.expanded_cases, profiled.expanded_cases);
        let baseline_seen = baseline_requested_cases
            .lock()
            .expect("baseline requested cases");
        let profiled_seen = profiled_requested_cases
            .lock()
            .expect("profiled requested cases");
        assert_eq!(baseline_seen.len(), baseline.expanded_cases.len());
        assert_eq!(profiled_seen.len(), baseline.expanded_cases.len());
        for ((baseline_case, baseline_requested), profiled_requested) in baseline
            .expanded_cases
            .iter()
            .zip(baseline_seen.iter())
            .zip(profiled_seen.iter())
        {
            assert_eq!(&baseline_case.requested_case, baseline_requested);
            assert_eq!(baseline_requested, profiled_requested);
        }
        assert_eq!(baseline.schedule, profiled.schedule);
        assert_eq!(
            baseline.comparison.axis_names,
            profiled.comparison.axis_names
        );
        assert_eq!(
            baseline.comparison.invariant_violations,
            profiled.comparison.invariant_violations
        );
        assert_eq!(
            baseline.comparison.case_count,
            profiled.comparison.case_count
        );
    }

    #[tokio::test]
    async fn sweep_passes_cpu_profiling_config_through_for_docker_cases() {
        let tempdir = tempdir().expect("tempdir");
        let output_root = tempdir.path().join("sweep");
        let matrix = SweepMatrix {
            base_case: RequestedCase {
                cooldown_secs: 0,
                execution: ExecutionRequest::Docker {
                    image_tag: "nockchain-bench:test".to_string(),
                    memory_limit: "4g".to_string(),
                    cpuset: None,
                    cpu_quota: None,
                    cpu_period: None,
                    work_dir_mode: WorkDirMode::DockerTmpfs,
                    allow_version_skew: false,
                },
                ..RequestedCase::native(PathBuf::from("fixture.soltest"))
            },
            axes: BTreeMap::from([("threads".to_string(), vec![AxisValue::Integer(1)])]),
        };
        let matrix_json = serde_json::to_value(&matrix).expect("matrix json");
        let profiler = CpuProfilerConfig {
            kind: CpuProfilerKind::Samply,
            sample_rate_hz: 1_000,
        };
        let mut executor = FakeExecutor::new(vec![Ok(trusted_run_result(
            "fixture-a",
            1,
            Validity::Valid,
        ))]);
        let seen_paths = executor.seen_paths();
        let seen_cpu_profilers = executor.seen_cpu_profilers();

        let result = execute_sweep(
            &matrix_json,
            matrix,
            &output_root,
            &SweepRunOptions {
                cpu_profiler: Some(profiler.clone()),
                ..SweepRunOptions::default()
            },
            &mut executor,
        )
        .await
        .expect("docker sweeps should accept cpu profiling");

        assert_eq!(result.comparison.case_count, 1);
        assert_eq!(seen_paths.lock().expect("seen paths").len(), 1);
        assert_eq!(
            seen_cpu_profilers
                .lock()
                .expect("seen cpu profilers")
                .as_slice(),
            &[Some(profiler)]
        );
    }

    #[test]
    fn sweep_parses_spec_style_matrix_json() {
        let value = serde_json::json!({
            "benchmark": "sol-replay",
            "base": {
                "fixture": "fixture.soltest",
                "threads": 4,
                "warmup_runs": 1,
                "measured_runs": 3,
                "cooldown_secs": 0,
                "mode": {
                    "docker": {
                        "image_tag": "nockchain-bench:test",
                        "work_dir_mode": "DockerTmpfs"
                    }
                }
            },
            "axes": {
                "memory_limit": ["4g", "8g"]
            }
        });

        let matrix = parse_matrix_value(value).expect("parse matrix");

        assert_eq!(matrix.base_case.threads, 4);
        assert_eq!(matrix.base_case.measured_runs, 3);
        match matrix.base_case.execution {
            ExecutionRequest::Docker {
                image_tag,
                memory_limit,
                work_dir_mode,
                ..
            } => {
                assert_eq!(image_tag, "nockchain-bench:test");
                assert_eq!(memory_limit, "");
                assert_eq!(work_dir_mode, WorkDirMode::DockerTmpfs);
            }
            _ => panic!("expected docker execution"),
        }
        assert_eq!(
            matrix.axes.get("memory_limit"),
            Some(&vec![
                AxisValue::String("4g".to_string()),
                AxisValue::String("8g".to_string()),
            ])
        );
    }

    #[test]
    fn sweep_expand_matrix_rejects_unknown_native_axis_as_unsupported() {
        let matrix = SweepMatrix {
            base_case: RequestedCase::native(PathBuf::from("fixture.soltest")),
            axes: BTreeMap::from([("bogus".to_string(), vec![AxisValue::Integer(1)])]),
        };

        let error = expand_matrix(&matrix, &SweepOptions::default()).expect_err("unknown axis");

        assert!(
            error.to_string().contains("unsupported sweep axis `bogus`"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn sweep_expand_matrix_rejects_docker_axis_for_native_execution() {
        let matrix = SweepMatrix {
            base_case: RequestedCase::native(PathBuf::from("fixture.soltest")),
            axes: BTreeMap::from([(
                "memory_limit".to_string(),
                vec![AxisValue::String("4g".to_string())],
            )]),
        };

        let error = expand_matrix(&matrix, &SweepOptions::default()).expect_err("docker-only axis");

        assert!(
            error
                .to_string()
                .contains("sweep axis `memory_limit` requires Docker execution"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn sweep_execution_writes_top_level_artifacts_and_case_outputs() {
        let tempdir = tempdir().expect("tempdir");
        let output_root = tempdir.path().join("sweep");
        let matrix = SweepMatrix {
            base_case: RequestedCase {
                measured_runs: 3,
                cooldown_secs: 0,
                execution: ExecutionRequest::Docker {
                    image_tag: "nockchain-bench:test".to_string(),
                    memory_limit: "4g".to_string(),
                    cpuset: Some("0-3".to_string()),
                    cpu_quota: Some(200_000),
                    cpu_period: Some(100_000),
                    work_dir_mode: WorkDirMode::DockerTmpfs,
                    allow_version_skew: false,
                },
                ..RequestedCase::native(PathBuf::from("fixture.soltest"))
            },
            axes: BTreeMap::from([(
                "threads".to_string(),
                vec![AxisValue::Integer(1), AxisValue::Integer(2)],
            )]),
        };
        let matrix_json = serde_json::to_value(&matrix).expect("matrix json");
        let mut executor = FakeExecutor::new(vec![
            Ok(trusted_run_result("fixture-a", 1, Validity::Valid)),
            Ok(trusted_run_result(
                "fixture-a",
                2,
                Validity::Partial {
                    reasons: vec!["throughput CV high".to_string()],
                },
            )),
        ]);
        let seen_paths = executor.seen_paths();

        let result = execute_sweep(
            &matrix_json,
            matrix,
            &output_root,
            &SweepRunOptions {
                schedule_mode: ScheduleMode::Sequential,
                comparison_markdown: true,
                ..SweepRunOptions::default()
            },
            &mut executor,
        )
        .await
        .expect("execute sweep");

        assert_eq!(result.comparison.cases.len(), 2);
        assert!(output_root.join("schema_version.txt").exists());
        assert!(output_root.join("matrix.json").exists());
        assert!(output_root.join("matrix_expanded.json").exists());
        assert!(output_root.join("schedule.json").exists());
        assert!(output_root.join("comparison.json").exists());
        assert!(output_root.join("comparison.md").exists());
        assert!(output_root.join("verdict.json").exists());

        let seen_paths = seen_paths.lock().expect("seen paths");
        assert_eq!(
            seen_paths.as_slice(),
            &[
                output_root.join("cases/case-000-threads_1"),
                output_root.join("cases/case-001-threads_2"),
            ]
        );
        assert!(matches!(result.verdict.validity, Validity::Partial { .. }));
    }

    #[tokio::test]
    async fn sweep_execution_failure_writes_top_level_verdict_before_returning_error() {
        let tempdir = tempdir().expect("tempdir");
        let output_root = tempdir.path().join("sweep");
        let matrix = SweepMatrix {
            base_case: RequestedCase {
                measured_runs: 3,
                cooldown_secs: 0,
                execution: ExecutionRequest::Docker {
                    image_tag: "nockchain-bench:test".to_string(),
                    memory_limit: "4g".to_string(),
                    cpuset: Some("0-3".to_string()),
                    cpu_quota: Some(200_000),
                    cpu_period: Some(100_000),
                    work_dir_mode: WorkDirMode::DockerTmpfs,
                    allow_version_skew: false,
                },
                ..RequestedCase::native(PathBuf::from("fixture.soltest"))
            },
            axes: BTreeMap::from([(
                "threads".to_string(),
                vec![AxisValue::Integer(1), AxisValue::Integer(2)],
            )]),
        };
        let matrix_json = serde_json::to_value(&matrix).expect("matrix json");
        let mut executor = FakeExecutor::new(vec![
            Ok(trusted_run_result("fixture-a", 1, Validity::Valid)),
            Err(HarnessError::CommandFailure(
                "second case failed".to_string(),
            )),
        ]);

        let error = execute_sweep(
            &matrix_json,
            matrix,
            &output_root,
            &SweepRunOptions {
                schedule_mode: ScheduleMode::Sequential,
                ..SweepRunOptions::default()
            },
            &mut executor,
        )
        .await
        .expect_err("sweep should fail");

        assert!(error.to_string().contains("second case failed"));
        let verdict: Verdict = serde_json::from_slice(
            &std::fs::read(output_root.join("verdict.json")).expect("verdict artifact"),
        )
        .expect("parse verdict");
        match verdict.validity {
            Validity::Invalid { reasons } => {
                assert!(reasons
                    .iter()
                    .any(|reason| reason.contains("case-001-threads_2")));
                assert!(reasons
                    .iter()
                    .any(|reason| reason.contains("second case failed")));
            }
            other => panic!("expected invalid verdict, got {other:?}"),
        }
    }
}
