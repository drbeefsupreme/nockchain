use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::speed_of_light::{InvocationTracingConfig, TracyMode};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceArtifactFile {
    pub artifact: String,
    pub file_name: String,
    pub requested: bool,
    pub exists: bool,
    pub nonempty: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunTraceArtifacts {
    pub nock_tracing_requested: bool,
    pub tracy_requested: bool,
    pub complete: bool,
    pub artifacts: Vec<TraceArtifactFile>,
}

impl RunTraceArtifacts {
    pub fn for_run(run_dir: &Path, tracing: &InvocationTracingConfig) -> Option<Self> {
        let mut artifacts = Vec::new();

        if tracing.nock_tracing {
            artifacts.push(build_artifact(run_dir, "nock_trace", "nock_trace.ndjson"));
            artifacts.push(build_artifact(
                run_dir,
                "nock_trace_meta",
                "nock_trace_meta.json",
            ));
        }

        if tracing.tracy != TracyMode::Off {
            artifacts.push(build_artifact(
                run_dir,
                "tracy_capture",
                "tracy_capture.tracy",
            ));
        }

        if artifacts.is_empty() {
            return None;
        }

        let mut manifest = Self {
            nock_tracing_requested: tracing.nock_tracing,
            tracy_requested: tracing.tracy != TracyMode::Off,
            complete: false,
            artifacts,
        };
        manifest.refresh_from_disk(run_dir);
        Some(manifest)
    }

    pub fn refresh_from_disk(&mut self, run_dir: &Path) {
        for artifact in &mut self.artifacts {
            let path = run_dir.join(&artifact.file_name);
            artifact.size_bytes = std::fs::metadata(&path).ok().map(|metadata| metadata.len());
            artifact.exists = artifact.size_bytes.is_some();
            artifact.nonempty = artifact.size_bytes.map(|size| size > 0).unwrap_or(false);
        }
        self.complete = self.artifacts.iter().all(|artifact| artifact.nonempty);
    }

    pub fn is_requested(&self) -> bool {
        self.nock_tracing_requested || self.tracy_requested
    }
}

fn build_artifact(_run_dir: &Path, artifact: &str, file_name: &str) -> TraceArtifactFile {
    TraceArtifactFile {
        artifact: artifact.to_string(),
        file_name: file_name.to_string(),
        requested: true,
        exists: false,
        nonempty: false,
        size_bytes: None,
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn run_trace_artifacts_describes_missing_requested_files() {
        let tempdir = tempdir().expect("tempdir");
        let run_dir = tempdir.path().join("runs/run-0");
        std::fs::create_dir_all(&run_dir).expect("run dir");

        let artifacts = RunTraceArtifacts::for_run(
            &run_dir,
            &InvocationTracingConfig {
                nock_tracing: true,
                nock_tracing_keyword_filter: Some("foo".to_string()),
                nock_tracing_interval_filter: Some(8),
                tracy: TracyMode::Nockcode,
            },
        )
        .expect("trace artifacts should be requested");

        assert!(artifacts.is_requested());
        assert!(!artifacts.complete);
        assert_eq!(artifacts.artifacts.len(), 3);
        assert!(artifacts.artifacts.iter().all(|artifact| !artifact.exists));
    }

    #[test]
    fn run_trace_artifacts_marks_zero_byte_requested_file_incomplete() {
        let tempdir = tempdir().expect("tempdir");
        let run_dir = tempdir.path().join("runs/run-0");
        std::fs::create_dir_all(&run_dir).expect("run dir");
        std::fs::write(run_dir.join("nock_trace.ndjson"), b"").expect("empty trace file");
        std::fs::write(run_dir.join("nock_trace_meta.json"), b"{\"ok\":true}")
            .expect("meta file");

        let artifacts = RunTraceArtifacts::for_run(
            &run_dir,
            &InvocationTracingConfig {
                nock_tracing: true,
                nock_tracing_keyword_filter: None,
                nock_tracing_interval_filter: None,
                tracy: TracyMode::Off,
            },
        )
        .expect("trace artifacts should be requested");

        assert!(!artifacts.complete);
        assert_eq!(artifacts.artifacts.len(), 2);
        assert_eq!(artifacts.artifacts[0].artifact, "nock_trace");
        assert!(artifacts.artifacts[0].exists);
        assert!(!artifacts.artifacts[0].nonempty);
        assert_eq!(artifacts.artifacts[0].size_bytes, Some(0));
    }

    #[test]
    fn run_trace_artifacts_supports_tracy_only_requests() {
        let tempdir = tempdir().expect("tempdir");
        let run_dir = tempdir.path().join("runs/run-0");
        std::fs::create_dir_all(&run_dir).expect("run dir");
        std::fs::write(run_dir.join("tracy_capture.tracy"), b"trace-bytes")
            .expect("tracy capture");

        let artifacts = RunTraceArtifacts::for_run(
            &run_dir,
            &InvocationTracingConfig {
                nock_tracing: false,
                nock_tracing_keyword_filter: None,
                nock_tracing_interval_filter: None,
                tracy: TracyMode::Nockcode,
            },
        )
        .expect("trace artifacts should be requested");

        assert!(artifacts.complete);
        assert!(!artifacts.nock_tracing_requested);
        assert!(artifacts.tracy_requested);
        assert_eq!(artifacts.artifacts.len(), 1);
        assert_eq!(artifacts.artifacts[0].artifact, "tracy_capture");
        assert!(artifacts.artifacts[0].nonempty);
    }
}
