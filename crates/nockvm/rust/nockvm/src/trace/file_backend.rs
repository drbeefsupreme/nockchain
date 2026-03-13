use std::fs::File;
use std::io::{BufWriter, Error, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::*;

const MISSING_TRACE_HANDLE: u64 = u64::MAX;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileTraceMetadata {
    pub format: String,
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyword_filter: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval_filter: Option<usize>,
}

impl FileTraceMetadata {
    pub fn new(
        mode: impl Into<String>,
        keyword_filter: Option<Vec<String>>,
        interval_filter: Option<usize>,
    ) -> Self {
        Self {
            format: "nock-trace-v1".to_string(),
            mode: mode.into(),
            keyword_filter,
            interval_filter,
        }
    }
}

#[derive(Debug, Serialize)]
struct FileTraceRecord<'a> {
    path: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    chum: Option<&'a str>,
}

#[derive(Debug)]
struct FileTraceEntry {
    path: String,
    chum: Option<String>,
}

#[derive(Clone, Copy)]
struct FileTraceData {
    handle: u64,
}

pub struct FileTraceBackend {
    writer: BufWriter<File>,
    metadata_path: PathBuf,
    metadata: FileTraceMetadata,
    entries: Vec<FileTraceEntry>,
}

impl FileTraceBackend {
    pub fn new(
        trace_path: impl AsRef<Path>,
        metadata_path: impl AsRef<Path>,
        metadata: FileTraceMetadata,
    ) -> Result<Self, Error> {
        let trace_path = trace_path.as_ref();
        if let Some(parent) = trace_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let metadata_path = metadata_path.as_ref().to_path_buf();
        if let Some(parent) = metadata_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        Ok(Self {
            writer: BufWriter::new(File::create(trace_path)?),
            metadata_path,
            metadata,
            entries: Vec::new(),
        })
    }

    pub fn append_trace_data(&mut self, stack: &mut NockStack, path: Noun) -> u64 {
        let handle = self.entries.len() as u64;
        self.entries.push(FileTraceEntry {
            path: render_trace_path(stack, path),
            chum: render_trace_chum(path),
        });
        handle
    }

    pub fn write_trace_data(&mut self, handle: u64) -> Result<(), Error> {
        let Some(entry) = self.entries.get(handle as usize) else {
            return Err(Error::other(format!(
                "missing file trace handle {handle}"
            )));
        };

        let record = FileTraceRecord {
            path: &entry.path,
            chum: entry.chum.as_deref(),
        };
        serde_json::to_writer(&mut self.writer, &record)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()
    }
}

impl TraceBackend for FileTraceBackend {
    fn append_trace(&mut self, stack: &mut NockStack, path: Noun) {
        let handle = self.append_trace_data(stack, path);
        TraceStack::push_on_stack(stack, FileTraceData { handle });
    }

    unsafe fn write_nock_trace(
        &mut self,
        _: &mut NockStack,
        trace_stack: *const TraceStack,
    ) -> Result<(), Error> {
        let mut trace_stack = trace_stack as *const TraceStack<FileTraceData>;

        if trace_stack.is_null() {
            return Ok(());
        }

        loop {
            self.write_trace_data((&*trace_stack).handle)?;

            trace_stack = (&*trace_stack).next;
            if trace_stack.is_null() {
                break Ok(());
            }
        }
    }

    fn write_metadata(&mut self) -> Result<(), Error> {
        std::fs::write(
            &self.metadata_path,
            serde_json::to_vec_pretty(&self.metadata)?,
        )
    }
}

#[derive(Clone, Copy)]
struct CompositeTraceData {
    tracing_span_id: u64,
    file_handle: u64,
}

pub struct CompositeTraceBackend {
    tracing: Option<TracingBackend>,
    file: Option<FileTraceBackend>,
}

impl CompositeTraceBackend {
    pub fn new(tracing: Option<TracingBackend>, file: Option<FileTraceBackend>) -> Self {
        Self { tracing, file }
    }
}

impl TraceBackend for CompositeTraceBackend {
    fn append_trace(&mut self, stack: &mut NockStack, path: Noun) {
        let tracing_span_id = self
            .tracing
            .as_mut()
            .and_then(|backend| backend.append_trace_data(stack, path))
            .unwrap_or(MISSING_TRACE_HANDLE);
        let file_handle = self
            .file
            .as_mut()
            .map(|backend| backend.append_trace_data(stack, path))
            .unwrap_or(MISSING_TRACE_HANDLE);

        if tracing_span_id == MISSING_TRACE_HANDLE && file_handle == MISSING_TRACE_HANDLE {
            return;
        }

        TraceStack::push_on_stack(
            stack,
            CompositeTraceData {
                tracing_span_id,
                file_handle,
            },
        );
    }

    unsafe fn write_nock_trace(
        &mut self,
        _: &mut NockStack,
        trace_stack: *const TraceStack,
    ) -> Result<(), Error> {
        let mut trace_stack = trace_stack as *const TraceStack<CompositeTraceData>;

        if trace_stack.is_null() {
            return Ok(());
        }

        loop {
            let data = &*trace_stack;

            if data.tracing_span_id != MISSING_TRACE_HANDLE {
                if let Some(tracing) = self.tracing.as_mut() {
                    tracing.write_trace_data(data.tracing_span_id)?;
                }
            }

            if data.file_handle != MISSING_TRACE_HANDLE {
                if let Some(file) = self.file.as_mut() {
                    file.write_trace_data(data.file_handle)?;
                }
            }

            trace_stack = data.next;
            if trace_stack.is_null() {
                break Ok(());
            }
        }
    }

    fn write_metadata(&mut self) -> Result<(), Error> {
        if let Some(file) = self.file.as_mut() {
            file.write_metadata()?;
        }
        Ok(())
    }

    fn write_serf_trace(&mut self, name: &str, start: Instant) -> Result<(), Error> {
        if let Some(tracing) = self.tracing.as_mut() {
            tracing.write_serf_trace(name, start)?;
        }
        if let Some(file) = self.file.as_mut() {
            file.write_serf_trace(name, start)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::mem::NockStack;
    use crate::noun::{D, T};

    use super::*;

    #[test]
    fn file_trace_backend_writes_trace_and_metadata_files() {
        let root = std::env::temp_dir().join(format!(
            "nockvm-file-trace-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("unix time")
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root dir");
        let trace_path = root.join("nock_trace.ndjson");
        let metadata_path = root.join("nock_trace_meta.json");

        let mut backend = FileTraceBackend::new(
            &trace_path,
            &metadata_path,
            FileTraceMetadata::new("tracing", Some(vec!["foo".to_string()]), Some(8)),
        )
        .expect("backend");
        let mut stack = NockStack::new(1 << 20, 2);
        let path = T(&mut stack, &[D(42), D(0)]);
        backend.append_trace(&mut stack, path);

        let trace_stack = unsafe { *(stack.local_noun_pointer(1) as *const *const TraceStack) };
        unsafe {
            backend
                .write_nock_trace(&mut stack, trace_stack)
                .expect("write trace");
        }
        backend.write_metadata().expect("metadata");

        assert!(std::fs::metadata(&trace_path).expect("trace metadata").len() > 0);
        assert!(std::fs::metadata(&metadata_path).expect("meta metadata").len() > 0);

        let _ = std::fs::remove_dir_all(&root);
    }
}
