//! Sweep planning and aggregation helpers for PMA/chunk-size benchmark matrices.

use serde::{Deserialize, Serialize};

use crate::events::{EventType, LogEvent};
use crate::runner::ContainerStats;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SweepCase {
    pub candidate: String,
    pub chunk_size: u64,
    pub memory_limit: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepRunMetrics {
    pub case: SweepCase,
    pub run_index: u32,
    pub peak_rss_mib: f64,
    pub avg_rss_mib: f64,
    pub checkpoint_count: u64,
    pub checkpoint_avg_duration_s: Option<f64>,
    pub checkpoint_mib_per_s: Option<f64>,
    pub page_fault_bursts: Option<u64>,
    pub minor_faults_delta_total: Option<u64>,
    pub major_faults_delta_total: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepCaseSummary {
    pub case: SweepCase,
    pub runs: u32,
    pub peak_rss_mib_mean: f64,
    pub peak_rss_mib_stddev: f64,
    pub checkpoint_mib_per_s_mean: Option<f64>,
    pub checkpoint_mib_per_s_stddev: Option<f64>,
    pub checkpoint_avg_duration_s_mean: Option<f64>,
    pub page_fault_bursts_mean: Option<f64>,
}

pub fn build_sweep_cases(
    candidates: &[String],
    chunk_sizes: &[u64],
    memory_limits: &[String],
) -> Vec<SweepCase> {
    let mut cases = Vec::new();
    for candidate in candidates {
        for chunk_size in chunk_sizes {
            for memory_limit in memory_limits {
                cases.push(SweepCase {
                    candidate: candidate.clone(),
                    chunk_size: *chunk_size,
                    memory_limit: memory_limit.clone(),
                });
            }
        }
    }
    cases
}

pub fn checkpoint_durations_ms(events: &[LogEvent]) -> Vec<u64> {
    let mut sorted = events.to_vec();
    sorted.sort_by_key(|event| event.timestamp_ms);

    let mut durations = Vec::new();
    let mut pending_starts = Vec::new();
    for event in sorted {
        match event.event_type {
            EventType::CheckpointStarted => pending_starts.push(event.timestamp_ms),
            EventType::CheckpointCompleted => {
                if let Some(start_ms) = pending_starts.first().copied() {
                    pending_starts.remove(0);
                    durations.push(event.timestamp_ms.saturating_sub(start_ms));
                }
            }
            _ => {}
        }
    }

    durations
}

pub fn page_fault_bursts(
    samples: &[ContainerStats],
    minor_threshold: u64,
    major_threshold: u64,
) -> Option<(u64, u64, u64)> {
    if samples.len() < 2 {
        return Some((0, 0, 0));
    }
    if samples
        .iter()
        .any(|sample| sample.minor_faults.is_none() || sample.major_faults.is_none())
    {
        return None;
    }

    let mut burst_count = 0u64;
    let mut minor_total = 0u64;
    let mut major_total = 0u64;
    for pair in samples.windows(2) {
        let prev = &pair[0];
        let next = &pair[1];
        let minor_delta = next
            .minor_faults
            .unwrap_or_default()
            .saturating_sub(prev.minor_faults.unwrap_or_default());
        let major_delta = next
            .major_faults
            .unwrap_or_default()
            .saturating_sub(prev.major_faults.unwrap_or_default());
        minor_total += minor_delta;
        major_total += major_delta;
        if minor_delta >= minor_threshold || major_delta >= major_threshold {
            burst_count += 1;
        }
    }

    Some((burst_count, minor_total, major_total))
}

pub fn summarize_case_runs(case: &SweepCase, runs: &[SweepRunMetrics]) -> SweepCaseSummary {
    let peak_values: Vec<f64> = runs.iter().map(|run| run.peak_rss_mib).collect();
    let throughput_values: Vec<f64> = runs
        .iter()
        .filter_map(|run| run.checkpoint_mib_per_s)
        .collect();
    let checkpoint_duration_values: Vec<f64> = runs
        .iter()
        .filter_map(|run| run.checkpoint_avg_duration_s)
        .collect();
    let page_fault_burst_values: Vec<f64> = runs
        .iter()
        .filter_map(|run| run.page_fault_bursts.map(|value| value as f64))
        .collect();

    SweepCaseSummary {
        case: case.clone(),
        runs: runs.len() as u32,
        peak_rss_mib_mean: mean(&peak_values).unwrap_or(0.0),
        peak_rss_mib_stddev: stddev(&peak_values),
        checkpoint_mib_per_s_mean: mean(&throughput_values),
        checkpoint_mib_per_s_stddev: if throughput_values.is_empty() {
            None
        } else {
            Some(stddev(&throughput_values))
        },
        checkpoint_avg_duration_s_mean: mean(&checkpoint_duration_values),
        page_fault_bursts_mean: mean(&page_fault_burst_values),
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn stddev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = mean(values).unwrap_or(0.0);
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(ts: u64, minor: Option<u64>, major: Option<u64>) -> ContainerStats {
        ContainerStats {
            timestamp_ms: ts,
            memory_usage_bytes: 0,
            memory_limit_bytes: 0,
            memory_percent: 0.0,
            memory_cache_bytes: 0,
            memory_rss_bytes: 0,
            cpu_percent: 0.0,
            minor_faults: minor,
            major_faults: major,
        }
    }

    #[test]
    fn test_build_sweep_cases_cartesian() {
        let cases = build_sweep_cases(
            &["a".to_string(), "b".to_string()],
            &[64, 128],
            &["8g".to_string(), "16g".to_string()],
        );
        assert_eq!(cases.len(), 8);
    }

    #[test]
    fn test_checkpoint_durations_pairing() {
        let events = vec![
            LogEvent {
                timestamp_ms: 10,
                time_str: "00:00:10".to_string(),
                level: 'I',
                event_type: EventType::CheckpointStarted,
                raw_line: String::new(),
            },
            LogEvent {
                timestamp_ms: 50,
                time_str: "00:00:50".to_string(),
                level: 'I',
                event_type: EventType::CheckpointCompleted,
                raw_line: String::new(),
            },
            LogEvent {
                timestamp_ms: 80,
                time_str: "00:01:20".to_string(),
                level: 'I',
                event_type: EventType::CheckpointStarted,
                raw_line: String::new(),
            },
            LogEvent {
                timestamp_ms: 120,
                time_str: "00:02:00".to_string(),
                level: 'I',
                event_type: EventType::CheckpointCompleted,
                raw_line: String::new(),
            },
        ];
        let durations = checkpoint_durations_ms(&events);
        assert_eq!(durations, vec![40, 40]);
    }

    #[test]
    fn test_page_fault_bursts() {
        let samples = vec![
            sample(0, Some(100), Some(1)),
            sample(1000, Some(200), Some(1)),
            sample(2000, Some(1000), Some(3)),
        ];
        let (bursts, minor_total, major_total) =
            page_fault_bursts(&samples, 300, 1).expect("faults available");
        assert_eq!(bursts, 1);
        assert_eq!(minor_total, 900);
        assert_eq!(major_total, 2);
    }

    #[test]
    fn test_page_fault_bursts_unavailable_when_missing() {
        let samples = vec![sample(0, None, Some(1)), sample(1000, Some(20), Some(1))];
        assert!(page_fault_bursts(&samples, 1, 1).is_none());
    }

    #[test]
    fn test_summarize_case_runs() {
        let case = SweepCase {
            candidate: "alpha".to_string(),
            chunk_size: 64,
            memory_limit: "8g".to_string(),
        };
        let runs = vec![
            SweepRunMetrics {
                case: case.clone(),
                run_index: 0,
                peak_rss_mib: 100.0,
                avg_rss_mib: 90.0,
                checkpoint_count: 2,
                checkpoint_avg_duration_s: Some(3.0),
                checkpoint_mib_per_s: Some(50.0),
                page_fault_bursts: Some(2),
                minor_faults_delta_total: Some(100),
                major_faults_delta_total: Some(1),
            },
            SweepRunMetrics {
                case: case.clone(),
                run_index: 1,
                peak_rss_mib: 120.0,
                avg_rss_mib: 95.0,
                checkpoint_count: 2,
                checkpoint_avg_duration_s: Some(4.0),
                checkpoint_mib_per_s: Some(40.0),
                page_fault_bursts: Some(4),
                minor_faults_delta_total: Some(120),
                major_faults_delta_total: Some(2),
            },
        ];
        let summary = summarize_case_runs(&case, &runs);
        assert_eq!(summary.runs, 2);
        assert!((summary.peak_rss_mib_mean - 110.0).abs() < 1e-9);
        assert!(summary.peak_rss_mib_stddev > 0.0);
        assert_eq!(summary.checkpoint_mib_per_s_mean, Some(45.0));
        assert_eq!(summary.page_fault_bursts_mean, Some(3.0));
    }
}
