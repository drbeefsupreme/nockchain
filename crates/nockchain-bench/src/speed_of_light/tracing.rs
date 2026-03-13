#[cfg(feature = "tracing-tracy")]
use std::sync::OnceLock;
use std::path::{Path, PathBuf};

use clap::ValueEnum;
use nockapp::kernel::boot::{TraceMode, TraceOpts};
use nockvm::trace::{
    CompositeTraceBackend, FileTraceBackend, FileTraceMetadata, IntervalFilter, KeywordFilter,
    TraceFilter, TraceInfo, TracingBackend,
};
use serde::{Deserialize, Serialize};

const DEFAULT_LOG_FILTER: &str = "info";

#[cfg(feature = "tracing-tracy")]
static TRACY_SUBSCRIBER_STATE: OnceLock<TracySubscriberState> = OnceLock::new();

#[cfg(feature = "tracing-tracy")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TracySubscriberState {
    Bench(TracyMode),
    External,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum NockTracingMode {
    #[default]
    Off,
    On,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum TracyMode {
    #[default]
    Off,
    All,
    Nockcode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InvocationTracingConfig {
    pub nock_tracing: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nock_tracing_keyword_filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nock_tracing_interval_filter: Option<usize>,
    #[serde(default)]
    pub tracy: TracyMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TracingProvenance {
    pub nock_tracing: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nock_tracing_keyword_filter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nock_tracing_interval_filter: Option<usize>,
    pub tracy_mode: TracyMode,
    pub tracy_compiled: bool,
    pub demangling_enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NockTracePaths {
    pub ndjson_path: PathBuf,
    pub metadata_path: PathBuf,
}

#[cfg(feature = "tracing-tracy")]
tracy_client::register_demangler!();

impl InvocationTracingConfig {
    pub fn new(
        nock_tracing: NockTracingMode,
        nock_tracing_keyword_filter: Option<String>,
        nock_tracing_interval_filter: Option<usize>,
        tracy: TracyMode,
    ) -> Result<Self, String> {
        let config = Self {
            nock_tracing: matches!(nock_tracing, NockTracingMode::On),
            nock_tracing_keyword_filter,
            nock_tracing_interval_filter,
            tracy,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.nock_tracing
            && (self.nock_tracing_keyword_filter.is_some()
                || self.nock_tracing_interval_filter.is_some())
        {
            return Err(
                "Nock tracing filters require --nock-tracing on (or nock_tracing = \"on\")"
                    .to_string(),
            );
        }

        if self.nock_tracing_interval_filter == Some(0) {
            return Err("nock tracing interval filter must be positive".to_string());
        }

        Ok(())
    }

    fn validate_for_context_with_compiled(
        &self,
        requires_local_replay: bool,
        tracy_compiled: bool,
    ) -> Result<(), String> {
        self.validate()?;

        if requires_local_replay && self.tracy != TracyMode::Off && !tracy_compiled {
            return Err(
                "this nockchain-bench build was compiled without Tracy support".to_string(),
            );
        }

        Ok(())
    }

    pub fn validate_for_local_replay(&self) -> Result<(), String> {
        self.validate_for_context_with_compiled(true, tracy_compiled())
    }

    pub fn to_trace_opts(&self) -> TraceOpts {
        TraceOpts {
            mode: self.nock_tracing.then_some(TraceMode::Tracing),
            keyword_filter: self.nock_tracing_keyword_filter.clone(),
            interval_filter: self.nock_tracing_interval_filter,
        }
    }

    pub fn nock_trace_paths_for_run(&self, run_dir: &Path) -> Option<NockTracePaths> {
        if !self.nock_tracing {
            return None;
        }

        Some(NockTracePaths {
            ndjson_path: run_dir.join("nock_trace.ndjson"),
            metadata_path: run_dir.join("nock_trace_meta.json"),
        })
    }

    pub fn to_trace_info(
        &self,
        nock_trace_paths: Option<&NockTracePaths>,
    ) -> Result<Option<TraceInfo>, String> {
        self.validate()?;

        if !self.nock_tracing {
            return Ok(None);
        }

        let keyword_values = self
            .nock_tracing_keyword_filter
            .clone()
            .map(|v| v.split(',').map(String::from).collect::<Vec<String>>());
        let keyword_filter = keyword_values
            .clone()
            .map(|keywords| KeywordFilter { keywords });
        let interval_filter = self
            .nock_tracing_interval_filter
            .map(|interval| IntervalFilter { interval, cnt: 0 });

        let filter = match (keyword_filter, interval_filter) {
            (Some(a), Some(b)) => Some(a.or(b).boxed()),
            (Some(a), None) => Some(a.boxed()),
            (None, Some(b)) => Some(b.boxed()),
            (None, None) => None,
        };

        let file_backend = if let Some(paths) = nock_trace_paths {
            Some(
                FileTraceBackend::new(
                    &paths.ndjson_path,
                    &paths.metadata_path,
                    FileTraceMetadata::new(
                        "tracing",
                        keyword_values,
                        self.nock_tracing_interval_filter,
                    ),
                )
                .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };

        let backend: Box<dyn nockvm::trace::TraceBackend> = if let Some(file_backend) = file_backend
        {
            Box::new(CompositeTraceBackend::new(
                Some(TracingBackend::new()),
                Some(file_backend),
            ))
        } else {
            Box::new(TracingBackend::new())
        };

        Ok(Some(TraceInfo { backend, filter }))
    }

    pub fn provenance(&self) -> TracingProvenance {
        TracingProvenance {
            nock_tracing: self.nock_tracing,
            nock_tracing_keyword_filter: self.nock_tracing_keyword_filter.clone(),
            nock_tracing_interval_filter: self.nock_tracing_interval_filter,
            tracy_mode: self.tracy,
            tracy_compiled: tracy_compiled(),
            demangling_enabled: demangling_enabled(),
        }
    }
}

pub fn tracy_compiled() -> bool {
    cfg!(feature = "tracing-tracy")
}

pub fn demangling_enabled() -> bool {
    cfg!(feature = "tracing-tracy")
}

pub fn init_tracing_subscriber(config: &InvocationTracingConfig) -> Result<(), String> {
    config.validate_for_local_replay()?;

    if config.tracy == TracyMode::Off {
        return Ok(());
    }

    #[cfg(feature = "tracing-tracy")]
    {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        use tracing_subscriber::{fmt, EnvFilter, Layer as _};

        if let Some(existing_state) = TRACY_SUBSCRIBER_STATE.get().copied() {
            return ensure_tracy_state_compatible(existing_state, config.tracy);
        }

        let tracy = tracing_tracy::TracyLayer::default();
        let env_filter = || {
            EnvFilter::new(
                std::env::var("RUST_LOG").unwrap_or_else(|_| DEFAULT_LOG_FILTER.to_string()),
            )
        };
        let fmt_layer = || fmt::layer().with_target(true).with_level(true);

        let init_result = match config.tracy {
            TracyMode::Off => Ok(()),
            TracyMode::All => tracing_subscriber::registry()
                .with(env_filter())
                .with(fmt_layer())
                .with(tracy)
                .try_init()
                .map_err(|err| err.to_string()),
            TracyMode::Nockcode => {
                let nockcode_filter =
                    tracing_subscriber::filter::filter_fn(|meta| meta.target() == "nockcode");
                tracing_subscriber::registry()
                    .with(env_filter())
                    .with(fmt_layer())
                    .with(tracy.with_filter(nockcode_filter))
                    .try_init()
                    .map_err(|err| err.to_string())
            }
        };

        match init_result {
            Ok(()) => {
                let _ = TRACY_SUBSCRIBER_STATE.set(TracySubscriberState::Bench(config.tracy));
                Ok(())
            }
            Err(error) => {
                if is_global_subscriber_already_set_error(&error) {
                    let _ = TRACY_SUBSCRIBER_STATE.set(TracySubscriberState::External);
                    Ok(())
                } else if let Some(existing_state) = TRACY_SUBSCRIBER_STATE.get().copied() {
                    ensure_tracy_state_compatible(existing_state, config.tracy)
                } else {
                    Err(error)
                }
            }
        }
    }

    #[cfg(not(feature = "tracing-tracy"))]
    {
        let _ = config;
        Err("this nockchain-bench build was compiled without Tracy support".to_string())
    }
}

#[cfg(feature = "tracing-tracy")]
fn ensure_tracy_state_compatible(
    existing: TracySubscriberState,
    requested: TracyMode,
) -> Result<(), String> {
    match existing {
        TracySubscriberState::Bench(existing_mode) => {
            if existing_mode == requested
                || matches!(
                    (existing_mode, requested),
                    (TracyMode::All, TracyMode::Nockcode)
                )
            {
                Ok(())
            } else {
                Err(format!(
                    "Tracy subscriber already initialized in {:?} mode and cannot satisfy {:?}",
                    existing_mode, requested
                ))
            }
        }
        TracySubscriberState::External => Ok(()),
    }
}

fn is_global_subscriber_already_set_error(error: &str) -> bool {
    error.contains("global default trace dispatcher has already been set")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invocation_tracing_config_rejects_filters_when_disabled() {
        let error = InvocationTracingConfig::new(
            NockTracingMode::Off,
            Some("foo".to_string()),
            None,
            TracyMode::Off,
        )
        .expect_err("filters should require tracing");
        assert!(error.contains("require"));
    }

    #[test]
    fn invocation_tracing_config_rejects_zero_interval() {
        let error =
            InvocationTracingConfig::new(NockTracingMode::On, None, Some(0), TracyMode::Off)
                .expect_err("zero interval should fail");
        assert!(error.contains("positive"));
    }

    #[test]
    fn invocation_tracing_config_maps_to_trace_opts() {
        let config = InvocationTracingConfig::new(
            NockTracingMode::On,
            Some("foo,bar".to_string()),
            Some(8),
            TracyMode::Nockcode,
        )
        .expect("valid tracing config");
        let trace_opts = config.to_trace_opts();

        assert!(trace_opts.mode.is_some());
        assert_eq!(trace_opts.keyword_filter.as_deref(), Some("foo,bar"));
        assert_eq!(trace_opts.interval_filter, Some(8));
    }

    #[test]
    fn invocation_tracing_config_builds_run_trace_paths_when_enabled() {
        let config =
            InvocationTracingConfig::new(NockTracingMode::On, None, Some(8), TracyMode::Off)
                .expect("valid config");

        let paths = config
            .nock_trace_paths_for_run(Path::new("/tmp/run-0"))
            .expect("paths");
        assert_eq!(
            paths.ndjson_path,
            PathBuf::from("/tmp/run-0/nock_trace.ndjson")
        );
        assert_eq!(
            paths.metadata_path,
            PathBuf::from("/tmp/run-0/nock_trace_meta.json")
        );
    }

    #[test]
    fn invocation_tracing_config_omits_run_trace_paths_when_disabled() {
        let config = InvocationTracingConfig::default();
        assert!(config
            .nock_trace_paths_for_run(Path::new("/tmp/run-0"))
            .is_none());
    }

    #[test]
    fn invocation_tracing_config_only_requires_tracy_support_for_local_replay() {
        let config = InvocationTracingConfig {
            nock_tracing: false,
            nock_tracing_keyword_filter: None,
            nock_tracing_interval_filter: None,
            tracy: TracyMode::All,
        };

        assert!(config
            .validate_for_context_with_compiled(false, false)
            .is_ok());
        assert!(config
            .validate_for_context_with_compiled(true, false)
            .is_err());
    }

    #[test]
    fn already_set_global_subscriber_errors_are_treated_as_nonfatal() {
        assert!(is_global_subscriber_already_set_error(
            "a global default trace dispatcher has already been set"
        ));
    }

    #[test]
    fn init_tracing_subscriber_is_idempotent_for_repeated_requests() {
        let config =
            InvocationTracingConfig::new(NockTracingMode::Off, None, None, TracyMode::Nockcode)
                .expect("valid config");

        init_tracing_subscriber(&config).expect("first init");
        init_tracing_subscriber(&config).expect("second init");
    }
}
