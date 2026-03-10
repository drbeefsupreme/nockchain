use std::path::Path;
use std::time::Instant;

use bollard::container::Stats;
use bollard::Docker;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HarnessDockerError {
    #[error("Docker API error: {0}")]
    Api(#[from] bollard::errors::Error),

    #[error("Docker not available: {0}")]
    NotAvailable(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContainerStats {
    pub timestamp_ms: u64,
    pub memory_usage_bytes: u64,
    pub memory_limit_bytes: u64,
    pub memory_percent: f64,
    pub memory_cache_bytes: u64,
    pub memory_rss_bytes: u64,
    pub cpu_percent: f64,
    pub minor_faults: Option<u64>,
    pub major_faults: Option<u64>,
}

impl ContainerStats {
    pub fn from_docker_stats(stats: &Stats, start_time: Instant) -> Self {
        use bollard::container::MemoryStatsStats;

        let memory_usage = stats.memory_stats.usage.unwrap_or(0);
        let memory_limit = stats.memory_stats.limit.unwrap_or(0);
        let (memory_cache, memory_rss) = stats
            .memory_stats
            .stats
            .as_ref()
            .map(|memory_stats| match memory_stats {
                MemoryStatsStats::V1(v1) => (v1.cache, v1.rss),
                MemoryStatsStats::V2(v2) => (v2.file, v2.anon),
            })
            .unwrap_or((0, memory_usage));

        let memory_percent = if memory_limit > 0 {
            (memory_usage as f64 / memory_limit as f64) * 100.0
        } else {
            0.0
        };

        Self {
            timestamp_ms: start_time.elapsed().as_millis() as u64,
            memory_usage_bytes: memory_usage,
            memory_limit_bytes: memory_limit,
            memory_percent,
            memory_cache_bytes: memory_cache,
            memory_rss_bytes: memory_rss,
            cpu_percent: calculate_cpu_percent(stats),
            minor_faults: None,
            major_faults: None,
        }
    }
}

pub async fn connect_docker() -> Result<Docker, HarnessDockerError> {
    let home = std::env::var("HOME").unwrap_or_default();
    let socket_paths = [
        "/var/run/docker.sock".to_string(),
        format!("{home}/.docker/desktop/docker.sock"),
        format!("{home}/.docker/run/docker.sock"),
    ];

    if let Ok(docker) = Docker::connect_with_local_defaults() {
        if docker.ping().await.is_ok() {
            return Ok(docker);
        }
    }

    for socket_path in socket_paths {
        if !Path::new(&socket_path).exists() {
            continue;
        }
        if let Ok(docker) =
            Docker::connect_with_unix(&socket_path, 120, bollard::API_DEFAULT_VERSION)
        {
            if docker.ping().await.is_ok() {
                return Ok(docker);
            }
        }
    }

    Err(HarnessDockerError::NotAvailable(
        "Cannot connect to Docker. Tried: default, /var/run/docker.sock, ~/.docker/desktop/docker.sock, ~/.docker/run/docker.sock"
            .to_string(),
    ))
}

pub fn parse_proc_stat_faults(stat: &str) -> Option<(u64, u64)> {
    let stat = stat.trim();
    if stat.is_empty() {
        return None;
    }

    let stat_after_comm = stat.rfind(')').map(|index| &stat[index + 1..]).unwrap_or(stat);
    let fields: Vec<&str> = stat_after_comm.split_whitespace().collect();
    let minflt = fields.get(7).and_then(|value| value.parse::<u64>().ok())?;
    let majflt = fields.get(9).and_then(|value| value.parse::<u64>().ok())?;
    Some((minflt, majflt))
}

pub fn parse_memory_limit(value: &str) -> i64 {
    let value = value.trim().to_lowercase();

    if let Some(num) = value.strip_suffix('g') {
        num.parse::<i64>().unwrap_or(0) * 1024 * 1024 * 1024
    } else if let Some(num) = value.strip_suffix('m') {
        num.parse::<i64>().unwrap_or(0) * 1024 * 1024
    } else if let Some(num) = value.strip_suffix('k') {
        num.parse::<i64>().unwrap_or(0) * 1024
    } else {
        value.parse::<i64>().unwrap_or(0)
    }
}

pub fn calculate_cpu_percent(stats: &Stats) -> f64 {
    let cpu_delta = stats.cpu_stats.cpu_usage.total_usage as i64
        - stats.precpu_stats.cpu_usage.total_usage as i64;
    let system_delta = stats.cpu_stats.system_cpu_usage.unwrap_or(0) as i64
        - stats.precpu_stats.system_cpu_usage.unwrap_or(0) as i64;
    let num_cpus = stats.cpu_stats.online_cpus.unwrap_or(1) as f64;

    if system_delta > 0 && cpu_delta > 0 {
        (cpu_delta as f64 / system_delta as f64) * num_cpus * 100.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_memory_limit() {
        assert_eq!(parse_memory_limit("16g"), 16 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_limit("512m"), 512 * 1024 * 1024);
        assert_eq!(parse_memory_limit("1024k"), 1024 * 1024);
        assert_eq!(parse_memory_limit("1073741824"), 1073741824);
        assert_eq!(parse_memory_limit("16G"), 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_proc_stat_faults() {
        let stat = "1 (nockchain) S 0 0 0 0 0 0 123 0 4 0 0 0 0 0 0 0 0 0 0 0 0";
        let parsed = parse_proc_stat_faults(stat).expect("expected parse");
        assert_eq!(parsed.0, 123);
        assert_eq!(parsed.1, 4);
        assert!(parse_proc_stat_faults("").is_none());
    }
}
