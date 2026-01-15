//! Log parsing and event correlation for Nockchain
//!
//! Extracts structured events from Nockchain log output and correlates
//! them with memory usage samples for analysis.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::runner::ContainerStats;

/// Type of event extracted from logs
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Block was validated
    BlockValidated {
        block_hash: String,
    },
    /// Block was accepted into the chain
    BlockAccepted {
        block_hash: String,
        height: u64,
    },
    /// New heaviest block
    NewHeaviestBlock {
        block_hash: String,
    },
    /// Checkpoint save started
    CheckpointStarted,
    /// Checkpoint save completed
    CheckpointCompleted,
    /// PMA sync/flush
    PmaSync,
    /// Mining started on new candidate
    MiningStarted,
    /// Block mined successfully
    BlockMined {
        block_hash: String,
    },
    /// Timer event
    Timer,
    /// Syncing/catching up
    Syncing {
        current_height: u64,
        target_height: u64,
    },
    /// Warning event
    Warning {
        message: String,
    },
    /// Error event
    Error {
        message: String,
    },
    /// Generic info event
    Info {
        source: String,
        message: String,
    },
}

impl EventType {
    /// Get a short label for this event type
    pub fn label(&self) -> &'static str {
        match self {
            EventType::BlockValidated { .. } => "block-validated",
            EventType::BlockAccepted { .. } => "block-accepted",
            EventType::NewHeaviestBlock { .. } => "new-heaviest",
            EventType::CheckpointStarted => "checkpoint-start",
            EventType::CheckpointCompleted => "checkpoint-done",
            EventType::PmaSync => "pma-sync",
            EventType::MiningStarted => "mining-start",
            EventType::BlockMined { .. } => "block-mined",
            EventType::Timer => "timer",
            EventType::Syncing { .. } => "syncing",
            EventType::Warning { .. } => "warning",
            EventType::Error { .. } => "error",
            EventType::Info { .. } => "info",
        }
    }

    /// Check if this is a significant event (worth highlighting)
    pub fn is_significant(&self) -> bool {
        matches!(
            self,
            EventType::BlockAccepted { .. }
                | EventType::NewHeaviestBlock { .. }
                | EventType::CheckpointStarted
                | EventType::CheckpointCompleted
                | EventType::PmaSync
                | EventType::BlockMined { .. }
                | EventType::Error { .. }
        )
    }
}

/// A parsed log event with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEvent {
    /// Timestamp in milliseconds (relative to some start time)
    pub timestamp_ms: u64,

    /// Time string from log (HH:MM:SS)
    pub time_str: String,

    /// Log level (I=Info, W=Warning, E=Error)
    pub level: char,

    /// Event type and details
    pub event_type: EventType,

    /// Raw log line (stripped of ANSI codes)
    pub raw_line: String,
}

impl LogEvent {
    /// Check if this is a significant event
    pub fn is_significant(&self) -> bool {
        self.event_type.is_significant()
    }
}

/// Parser for Nockchain log output
pub struct LogParser {
    /// Base time for relative timestamps (first event time)
    base_time: Option<(u8, u8, u8)>,
}

impl LogParser {
    /// Create a new log parser
    pub fn new() -> Self {
        Self { base_time: None }
    }

    /// Parse a single log line
    pub fn parse_line(&mut self, line: &str) -> Option<LogEvent> {
        // Strip ANSI escape codes
        let clean = strip_ansi_codes(line);

        // Parse format: "L (HH:MM:SS) source: message"
        // where L is I/W/E for log level
        let clean = clean.trim();
        if clean.len() < 12 {
            return None;
        }

        // Extract level
        let level = clean.chars().next()?;
        if !matches!(level, 'I' | 'W' | 'E') {
            return None;
        }

        // Extract time
        let time_start = clean.find('(')?;
        let time_end = clean.find(')')?;
        if time_end <= time_start + 1 {
            return None;
        }
        let time_str = &clean[time_start + 1..time_end];

        // Parse time components
        let parts: Vec<&str> = time_str.split(':').collect();
        if parts.len() != 3 {
            return None;
        }
        let hours: u8 = parts[0].parse().ok()?;
        let minutes: u8 = parts[1].parse().ok()?;
        let seconds: u8 = parts[2].parse().ok()?;

        // Calculate relative timestamp
        let timestamp_ms = self.calc_relative_time(hours, minutes, seconds);

        // Extract message (after time)
        let message = clean[time_end + 1..].trim();

        // Parse event type
        let event_type = self.parse_event_type(level, message);

        Some(LogEvent {
            timestamp_ms,
            time_str: time_str.to_string(),
            level,
            event_type,
            raw_line: clean.to_string(),
        })
    }

    /// Parse multiple log lines
    pub fn parse_lines(&mut self, lines: &[String]) -> Vec<LogEvent> {
        lines.iter().filter_map(|line| self.parse_line(line)).collect()
    }

    /// Calculate relative timestamp in milliseconds
    fn calc_relative_time(&mut self, hours: u8, minutes: u8, seconds: u8) -> u64 {
        if self.base_time.is_none() {
            self.base_time = Some((hours, minutes, seconds));
        }

        let (base_h, base_m, base_s) = self.base_time.unwrap();

        // Convert to total seconds
        let base_total = (base_h as u64) * 3600 + (base_m as u64) * 60 + (base_s as u64);
        let mut current_total = (hours as u64) * 3600 + (minutes as u64) * 60 + (seconds as u64);

        // Handle day wraparound
        if current_total < base_total {
            current_total += 24 * 3600;
        }

        (current_total - base_total) * 1000
    }

    /// Parse the event type from the message
    fn parse_event_type(&self, level: char, message: &str) -> EventType {
        // Check for warnings/errors first
        if level == 'W' {
            return EventType::Warning {
                message: message.to_string(),
            };
        }
        if level == 'E' {
            return EventType::Error {
                message: message.to_string(),
            };
        }

        // Parse info messages by source
        if let Some((source, rest)) = message.split_once(':') {
            let source = source.trim();
            let rest = rest.trim();

            match source {
                "validate-page-with-txs" => {
                    if let Some(hash) = rest.strip_prefix("Block validated:") {
                        return EventType::BlockValidated {
                            block_hash: hash.trim().to_string(),
                        };
                    }
                }
                "accept-block" => {
                    if rest.contains("added to validated blocks at") {
                        // Parse "block HASH added to validated blocks at HEIGHT"
                        if let Some(block_info) = rest.strip_prefix("block ") {
                            let parts: Vec<&str> = block_info.split(" added to validated blocks at ").collect();
                            if parts.len() == 2 {
                                let hash = parts[0].trim().to_string();
                                // Height might have "with proof version X" at the end
                                let height_str = parts[1].split_whitespace().next().unwrap_or("0");
                                let height = height_str.parse().unwrap_or(0);
                                return EventType::BlockAccepted {
                                    block_hash: hash,
                                    height,
                                };
                            }
                        }
                    } else if rest.contains("New heaviest block") {
                        return EventType::NewHeaviestBlock {
                            block_hash: String::new(),
                        };
                    }
                }
                "update-heaviest" => {
                    if rest.contains("is new heaviest block") {
                        // Parse "Block HASH is new heaviest block"
                        if let Some(hash_part) = rest.strip_prefix("Block ") {
                            let hash = hash_part
                                .split_whitespace()
                                .next()
                                .unwrap_or("")
                                .to_string();
                            return EventType::NewHeaviestBlock { block_hash: hash };
                        }
                    }
                }
                "handle-command" => {
                    if rest == "timer" {
                        return EventType::Timer;
                    }
                }
                "missing-parent-effects" => {
                    // Parse syncing info
                    if rest.contains("but we only have blocks up to height") {
                        // Try to extract heights
                        let current = extract_number_after(rest, "up to height ");
                        let target = extract_number_after(rest, "at height ");
                        return EventType::Syncing {
                            current_height: current.unwrap_or(0),
                            target_height: target.unwrap_or(0),
                        };
                    }
                }
                "checkpoint" | "save" => {
                    if rest.contains("start") || rest.contains("saving") {
                        return EventType::CheckpointStarted;
                    } else if rest.contains("complete") || rest.contains("done") || rest.contains("saved") {
                        return EventType::CheckpointCompleted;
                    }
                }
                "pma" => {
                    if rest.contains("sync") || rest.contains("flush") {
                        return EventType::PmaSync;
                    }
                }
                "miner" | "mining" => {
                    if rest.contains("start") || rest.contains("candidate") {
                        return EventType::MiningStarted;
                    } else if rest.contains("found") || rest.contains("mined") {
                        let hash = rest
                            .split_whitespace()
                            .find(|s| s.len() > 30)
                            .unwrap_or("")
                            .to_string();
                        return EventType::BlockMined { block_hash: hash };
                    }
                }
                _ => {}
            }

            // Default to generic info
            return EventType::Info {
                source: source.to_string(),
                message: rest.to_string(),
            };
        }

        // Fallback
        EventType::Info {
            source: "unknown".to_string(),
            message: message.to_string(),
        }
    }

    /// Reset the parser state (for a new run)
    pub fn reset(&mut self) {
        self.base_time = None;
    }
}

impl Default for LogParser {
    fn default() -> Self {
        Self::new()
    }
}

/// A container stats sample with correlated events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelatedSample {
    /// The stats sample
    pub stats: ContainerStats,

    /// Events that occurred near this sample's timestamp
    pub events: Vec<LogEvent>,

    /// Summary of significant events
    pub event_summary: Option<String>,
}

impl CorrelatedSample {
    /// Check if this sample has significant events
    pub fn has_significant_events(&self) -> bool {
        self.events.iter().any(|e| e.is_significant())
    }

    /// Get significant events only
    pub fn significant_events(&self) -> Vec<&LogEvent> {
        self.events.iter().filter(|e| e.is_significant()).collect()
    }
}

/// Correlates log events with stats samples
pub struct EventCorrelator {
    /// Window size in milliseconds for event correlation
    window_ms: u64,
}

impl EventCorrelator {
    /// Create a new correlator with default window (500ms)
    pub fn new() -> Self {
        Self { window_ms: 500 }
    }

    /// Set the correlation window size
    pub fn with_window_ms(mut self, window_ms: u64) -> Self {
        self.window_ms = window_ms;
        self
    }

    /// Correlate events with stats samples
    ///
    /// Each sample is annotated with events that occurred within `window_ms`
    /// of the sample's timestamp.
    pub fn correlate(
        &self,
        samples: &[ContainerStats],
        events: &[LogEvent],
    ) -> Vec<CorrelatedSample> {
        let mut results = Vec::with_capacity(samples.len());

        // Use a sliding window for efficiency
        let mut event_queue: VecDeque<&LogEvent> = VecDeque::new();
        let mut event_idx = 0;

        for sample in samples {
            let sample_time = sample.timestamp_ms;
            let window_start = sample_time.saturating_sub(self.window_ms);
            let window_end = sample_time + self.window_ms;

            // Remove events that are too old
            while let Some(front) = event_queue.front() {
                if front.timestamp_ms < window_start {
                    event_queue.pop_front();
                } else {
                    break;
                }
            }

            // Add events that fall within the window
            while event_idx < events.len() && events[event_idx].timestamp_ms <= window_end {
                if events[event_idx].timestamp_ms >= window_start {
                    event_queue.push_back(&events[event_idx]);
                }
                event_idx += 1;
            }

            // Collect events for this sample
            let sample_events: Vec<LogEvent> = event_queue
                .iter()
                .filter(|e| e.timestamp_ms >= window_start && e.timestamp_ms <= window_end)
                .cloned()
                .cloned()
                .collect();

            // Generate summary
            let summary = self.generate_summary(&sample_events);

            results.push(CorrelatedSample {
                stats: sample.clone(),
                events: sample_events,
                event_summary: summary,
            });
        }

        results
    }

    /// Generate a summary string for significant events
    fn generate_summary(&self, events: &[LogEvent]) -> Option<String> {
        let significant: Vec<&str> = events
            .iter()
            .filter(|e| e.is_significant())
            .map(|e| e.event_type.label())
            .collect();

        if significant.is_empty() {
            None
        } else {
            Some(significant.join(", "))
        }
    }

    /// Find memory spikes and correlate with events
    ///
    /// Returns samples where memory increased by more than `threshold_pct`
    /// from the previous sample.
    pub fn find_spikes<'a>(
        &self,
        correlated: &'a [CorrelatedSample],
        threshold_pct: f64,
    ) -> Vec<(usize, &'a CorrelatedSample, f64)> {
        let mut spikes = Vec::new();

        for i in 1..correlated.len() {
            let prev = correlated[i - 1].stats.memory_usage_bytes as f64;
            let curr = correlated[i].stats.memory_usage_bytes as f64;

            if prev > 0.0 {
                let change_pct = ((curr - prev) / prev) * 100.0;
                if change_pct > threshold_pct {
                    spikes.push((i, &correlated[i], change_pct));
                }
            }
        }

        spikes
    }
}

impl Default for EventCorrelator {
    fn default() -> Self {
        Self::new()
    }
}

/// Strip ANSI escape codes from a string
fn strip_ansi_codes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_escape = false;

    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if c == 'm' {
                in_escape = false;
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Extract a number after a prefix in a string
fn extract_number_after(s: &str, prefix: &str) -> Option<u64> {
    let start = s.find(prefix)? + prefix.len();
    let rest = &s[start..];
    let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    num_str.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_ansi_codes() {
        let input = "\x1b[32mI\x1b[0m \x1b[38;5;246m(23:33:51)\x1b[0m \x1b[32mtest message\x1b[0m";
        let output = strip_ansi_codes(input);
        assert_eq!(output, "I (23:33:51) test message");
    }

    #[test]
    fn test_parse_block_validated() {
        let mut parser = LogParser::new();
        let line = "I (12:34:56) validate-page-with-txs: Block validated: ABCD1234";
        let event = parser.parse_line(line).unwrap();

        assert_eq!(event.level, 'I');
        assert_eq!(event.time_str, "12:34:56");
        assert!(matches!(
            event.event_type,
            EventType::BlockValidated { block_hash } if block_hash == "ABCD1234"
        ));
    }

    #[test]
    fn test_parse_block_accepted() {
        let mut parser = LogParser::new();
        let line = "I (12:34:56) accept-block: block HASH123 added to validated blocks at 1234 with proof version 0";
        let event = parser.parse_line(line).unwrap();

        assert!(matches!(
            event.event_type,
            EventType::BlockAccepted { block_hash, height } if block_hash == "HASH123" && height == 1234
        ));
    }

    #[test]
    fn test_parse_new_heaviest() {
        let mut parser = LogParser::new();
        let line = "I (12:34:56) update-heaviest: Block NEWHASH is new heaviest block";
        let event = parser.parse_line(line).unwrap();

        assert!(matches!(
            event.event_type,
            EventType::NewHeaviestBlock { block_hash } if block_hash == "NEWHASH"
        ));
    }

    #[test]
    fn test_parse_warning() {
        let mut parser = LogParser::new();
        let line = "W (12:34:56) heard-block: Duplicate block";
        let event = parser.parse_line(line).unwrap();

        assert_eq!(event.level, 'W');
        assert!(matches!(event.event_type, EventType::Warning { .. }));
    }

    #[test]
    fn test_parse_timer() {
        let mut parser = LogParser::new();
        let line = "I (12:34:56) handle-command: timer";
        let event = parser.parse_line(line).unwrap();

        assert!(matches!(event.event_type, EventType::Timer));
    }

    #[test]
    fn test_relative_timestamps() {
        let mut parser = LogParser::new();

        let e1 = parser.parse_line("I (10:00:00) test: first").unwrap();
        let e2 = parser.parse_line("I (10:00:05) test: second").unwrap();
        let e3 = parser.parse_line("I (10:01:00) test: third").unwrap();

        assert_eq!(e1.timestamp_ms, 0);
        assert_eq!(e2.timestamp_ms, 5000);
        assert_eq!(e3.timestamp_ms, 60000);
    }

    #[test]
    fn test_correlate_events() {
        let samples = vec![
            ContainerStats {
                timestamp_ms: 0,
                memory_usage_bytes: 1000,
                memory_limit_bytes: 10000,
                memory_percent: 10.0,
                memory_cache_bytes: 100,
                memory_rss_bytes: 900,
                cpu_percent: 50.0,
            },
            ContainerStats {
                timestamp_ms: 1000,
                memory_usage_bytes: 2000,
                memory_limit_bytes: 10000,
                memory_percent: 20.0,
                memory_cache_bytes: 200,
                memory_rss_bytes: 1800,
                cpu_percent: 60.0,
            },
        ];

        let events = vec![
            LogEvent {
                timestamp_ms: 100,
                time_str: "10:00:00".to_string(),
                level: 'I',
                event_type: EventType::BlockAccepted {
                    block_hash: "hash1".to_string(),
                    height: 100,
                },
                raw_line: "test".to_string(),
            },
            LogEvent {
                timestamp_ms: 900,
                time_str: "10:00:00".to_string(),
                level: 'I',
                event_type: EventType::NewHeaviestBlock {
                    block_hash: "hash1".to_string(),
                },
                raw_line: "test".to_string(),
            },
        ];

        let correlator = EventCorrelator::new().with_window_ms(500);
        let correlated = correlator.correlate(&samples, &events);

        assert_eq!(correlated.len(), 2);
        // First event (100ms) should be in first sample's window (0 ± 500)
        assert!(correlated[0].events.iter().any(|e| matches!(
            &e.event_type,
            EventType::BlockAccepted { .. }
        )));
        // Second event (900ms) should be in second sample's window (1000 ± 500)
        assert!(correlated[1].events.iter().any(|e| matches!(
            &e.event_type,
            EventType::NewHeaviestBlock { .. }
        )));
    }

    #[test]
    fn test_find_spikes() {
        let samples = vec![
            CorrelatedSample {
                stats: ContainerStats {
                    timestamp_ms: 0,
                    memory_usage_bytes: 1000,
                    memory_limit_bytes: 10000,
                    memory_percent: 10.0,
                    memory_cache_bytes: 100,
                    memory_rss_bytes: 900,
                    cpu_percent: 50.0,
                },
                events: vec![],
                event_summary: None,
            },
            CorrelatedSample {
                stats: ContainerStats {
                    timestamp_ms: 1000,
                    memory_usage_bytes: 1200, // 20% increase
                    memory_limit_bytes: 10000,
                    memory_percent: 12.0,
                    memory_cache_bytes: 120,
                    memory_rss_bytes: 1080,
                    cpu_percent: 55.0,
                },
                events: vec![],
                event_summary: None,
            },
            CorrelatedSample {
                stats: ContainerStats {
                    timestamp_ms: 2000,
                    memory_usage_bytes: 1100, // decrease
                    memory_limit_bytes: 10000,
                    memory_percent: 11.0,
                    memory_cache_bytes: 110,
                    memory_rss_bytes: 990,
                    cpu_percent: 52.0,
                },
                events: vec![],
                event_summary: None,
            },
        ];

        let correlator = EventCorrelator::new();
        let spikes = correlator.find_spikes(&samples, 15.0); // >15% threshold

        assert_eq!(spikes.len(), 1);
        assert_eq!(spikes[0].0, 1); // Index of spike
        assert!((spikes[0].2 - 20.0).abs() < 0.1); // ~20% increase
    }

    #[test]
    fn test_event_type_labels() {
        assert_eq!(EventType::BlockValidated { block_hash: "".to_string() }.label(), "block-validated");
        assert_eq!(EventType::CheckpointStarted.label(), "checkpoint-start");
        assert_eq!(EventType::Timer.label(), "timer");
    }

    #[test]
    fn test_event_significance() {
        assert!(EventType::BlockAccepted { block_hash: "".to_string(), height: 0 }.is_significant());
        assert!(EventType::CheckpointCompleted.is_significant());
        assert!(!EventType::Timer.is_significant());
        assert!(!EventType::Info { source: "".to_string(), message: "".to_string() }.is_significant());
    }

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number_after("height 1234 blocks", "height "), Some(1234));
        assert_eq!(extract_number_after("no number here", "height "), None);
    }
}
