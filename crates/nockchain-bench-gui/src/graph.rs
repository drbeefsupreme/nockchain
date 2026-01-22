//! Time-series graph visualization using egui_plot
//!
//! Displays memory usage and other metrics over time with event markers.

use std::collections::HashMap;

use egui::{Color32, RichText, Ui};
use egui_plot::{Line, Plot, PlotPoints, VLine};
use uuid::Uuid;

use crate::config::MetricType;
use crate::storage::{DataSample, TestEvent, TestResult};

/// Computed metadata for an event (memory delta, duration, etc.)
#[derive(Debug, Clone, Default)]
pub struct EventMetadata {
    /// Memory usage before the event (in KiB)
    pub memory_before: Option<f64>,
    /// Memory usage after the event (in KiB)
    pub memory_after: Option<f64>,
    /// Memory delta (after - before) in KiB
    pub memory_delta: Option<f64>,
    /// Duration in milliseconds (for paired events like checkpoint start/done)
    pub duration_ms: Option<u64>,
    /// CPU usage around the event (percentage)
    pub cpu_around_event: Option<f64>,
    /// Whether there was a CPU spike (>20% increase from baseline)
    pub cpu_spike: bool,
    /// Block height (for block-related events)
    pub block_height: Option<u64>,
    /// Running count of this event type (e.g., "checkpoint #3")
    pub event_count: u32,
}

/// Compute metadata for all events
pub fn compute_event_metadata(
    events: &[&TestEvent],
    samples: &[DataSample],
    container_id: Option<Uuid>,
) -> Vec<EventMetadata> {
    let mut metadata = Vec::with_capacity(events.len());
    let mut event_counts: HashMap<String, u32> = HashMap::new();

    // Track checkpoint start times for duration calculation
    let mut checkpoint_starts: HashMap<Uuid, u64> = HashMap::new();

    // Calculate baseline CPU (average of first 10 samples)
    let baseline_cpu = calculate_baseline_cpu(samples, container_id);

    for event in events {
        let mut meta = EventMetadata::default();

        // Filter samples for this container if specified
        let container_samples: Vec<&DataSample> = if let Some(cid) = container_id {
            samples.iter().filter(|s| s.container_id == cid).collect()
        } else {
            samples.iter().filter(|s| s.container_id == event.container_id).collect()
        };

        // Event count
        let count = event_counts.entry(event.event_type.clone()).or_insert(0);
        *count += 1;
        meta.event_count = *count;

        // Memory before/after/delta
        if let Some((before, after)) = find_samples_around_event(event.timestamp_ms, &container_samples) {
            meta.memory_before = before.get(MetricType::ContainerMemory);
            meta.memory_after = after.get(MetricType::ContainerMemory);

            if let (Some(b), Some(a)) = (meta.memory_before, meta.memory_after) {
                meta.memory_delta = Some(a - b);
            }

            // CPU around event
            meta.cpu_around_event = after.get(MetricType::CpuPercent);

            // CPU spike detection
            if let (Some(baseline), Some(current)) = (baseline_cpu, meta.cpu_around_event) {
                meta.cpu_spike = current > baseline * 1.2; // 20% above baseline
            }
        }

        // Duration for checkpoint events
        if event.event_type == "checkpoint-start" {
            checkpoint_starts.insert(event.container_id, event.timestamp_ms);
        } else if event.event_type == "checkpoint-done" {
            if let Some(start_time) = checkpoint_starts.get(&event.container_id) {
                meta.duration_ms = Some(event.timestamp_ms.saturating_sub(*start_time));
            }
        }

        // Block height parsing
        meta.block_height = parse_block_height(&event.message);

        metadata.push(meta);
    }

    metadata
}

/// Find samples immediately before and after an event timestamp
fn find_samples_around_event<'a>(
    event_time_ms: u64,
    samples: &[&'a DataSample],
) -> Option<(&'a DataSample, &'a DataSample)> {
    // Find sample just before the event
    let before = samples.iter()
        .filter(|s| s.timestamp_ms <= event_time_ms)
        .max_by_key(|s| s.timestamp_ms)?;

    // Find sample just after the event (within 5 second window)
    let after = samples.iter()
        .filter(|s| s.timestamp_ms > event_time_ms && s.timestamp_ms <= event_time_ms + 5000)
        .min_by_key(|s| s.timestamp_ms)?;

    Some((before, after))
}

/// Calculate baseline CPU from first samples
fn calculate_baseline_cpu(samples: &[DataSample], container_id: Option<Uuid>) -> Option<f64> {
    let filtered: Vec<_> = if let Some(cid) = container_id {
        samples.iter().filter(|s| s.container_id == cid).take(10).collect()
    } else {
        samples.iter().take(10).collect()
    };

    if filtered.is_empty() {
        return None;
    }

    let sum: f64 = filtered.iter()
        .filter_map(|s| s.get(MetricType::CpuPercent))
        .sum();
    let count = filtered.iter()
        .filter(|s| s.get(MetricType::CpuPercent).is_some())
        .count();

    if count > 0 {
        Some(sum / count as f64)
    } else {
        None
    }
}

/// Parse block height from event message
fn parse_block_height(message: &str) -> Option<u64> {
    // Look for patterns like "at height 123" or "height: 123" or "block 123"
    let patterns = [
        "at height ", "height: ", "height ", "at block ", "block "
    ];

    for pattern in patterns {
        if let Some(pos) = message.to_lowercase().find(pattern) {
            let start = pos + pattern.len();
            let rest = &message[start..];
            let num_str: String = rest.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(height) = num_str.parse() {
                return Some(height);
            }
        }
    }

    None
}

/// Format memory value for display (KiB to human-readable)
fn format_memory(kib: f64) -> String {
    if kib.abs() >= 1024.0 * 1024.0 {
        format!("{:.1} GiB", kib / 1024.0 / 1024.0)
    } else if kib.abs() >= 1024.0 {
        format!("{:.1} MiB", kib / 1024.0)
    } else {
        format!("{:.0} KiB", kib)
    }
}

/// Format memory delta with sign
fn format_memory_delta(kib: f64) -> String {
    let sign = if kib >= 0.0 { "+" } else { "" };
    if kib.abs() >= 1024.0 * 1024.0 {
        format!("{}{:.1} GiB", sign, kib / 1024.0 / 1024.0)
    } else if kib.abs() >= 1024.0 {
        format!("{}{:.1} MiB", sign, kib / 1024.0)
    } else {
        format!("{}{:.0} KiB", sign, kib)
    }
}

/// Colors for different containers/lines
const LINE_COLORS: [Color32; 8] = [
    Color32::from_rgb(66, 133, 244),  // Blue
    Color32::from_rgb(234, 67, 53),   // Red
    Color32::from_rgb(52, 168, 83),   // Green
    Color32::from_rgb(251, 188, 4),   // Yellow
    Color32::from_rgb(156, 39, 176),  // Purple
    Color32::from_rgb(0, 188, 212),   // Cyan
    Color32::from_rgb(255, 152, 0),   // Orange
    Color32::from_rgb(121, 85, 72),   // Brown
];

/// Configuration for graph display
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Metrics to display
    pub metrics: Vec<MetricType>,

    /// Container IDs to display
    pub containers: Vec<Uuid>,

    /// Whether to show event markers
    pub show_events: bool,

    /// Whether to show only significant events
    pub significant_events_only: bool,

    /// Y-axis scale (None = auto)
    pub y_scale: Option<(f64, f64)>,

    /// X-axis range in milliseconds (None = auto)
    pub x_range: Option<(f64, f64)>,

    /// Whether to enable zoom/pan
    pub interactive: bool,

    /// Graph height in pixels
    pub height: f32,

    /// Index of currently highlighted event (for hover effect)
    pub highlighted_event: Option<usize>,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            metrics: vec![MetricType::VmRss],
            containers: Vec::new(),
            show_events: true,
            significant_events_only: true,
            y_scale: None,
            x_range: None,
            interactive: true,
            height: 300.0,
            highlighted_event: None,
        }
    }
}

impl GraphConfig {
    /// Create a config for a single metric
    pub fn single_metric(metric: MetricType) -> Self {
        Self {
            metrics: vec![metric],
            ..Default::default()
        }
    }

    /// Create a config for multiple metrics
    pub fn multi_metric(metrics: Vec<MetricType>) -> Self {
        Self {
            metrics,
            ..Default::default()
        }
    }

    /// Set the containers to display
    pub fn with_containers(mut self, containers: Vec<Uuid>) -> Self {
        self.containers = containers;
        self
    }

    /// Set whether to show events
    pub fn with_events(mut self, show: bool) -> Self {
        self.show_events = show;
        self
    }

    /// Set the graph height
    pub fn with_height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }
}

/// A point on the graph with metadata
#[derive(Debug, Clone)]
pub struct GraphPoint {
    pub x: f64,           // Time in seconds
    pub y: f64,           // Value
    pub label: String,    // Hover label
}

/// A line on the graph
#[derive(Debug, Clone)]
pub struct GraphLine {
    pub points: Vec<GraphPoint>,
    pub name: String,
    pub color: Color32,
}

/// An event marker on the graph
#[derive(Debug, Clone)]
pub struct EventMarker {
    pub x: f64,           // Time in seconds
    pub label: String,
    pub is_significant: bool,
}

/// Render a time-series graph for a test result
pub fn render_graph(ui: &mut Ui, result: &TestResult, config: &GraphConfig, highlighted_event: &mut Option<usize>) {
    if result.samples.is_empty() {
        ui.label("No data to display");
        return;
    }

    let containers: Vec<Uuid> = if config.containers.is_empty() {
        result.config.containers.iter().map(|c| c.id).collect()
    } else {
        config.containers.clone()
    };

    // Build lines for each container and metric
    let mut lines: Vec<GraphLine> = Vec::new();
    let mut color_idx = 0;

    for container_id in &containers {
        let container_name = result
            .config
            .containers
            .iter()
            .find(|c| c.id == *container_id)
            .map(|c| c.name.as_str())
            .unwrap_or("Unknown");

        let container_samples = result.samples_for_container(*container_id);

        for metric in &config.metrics {
            let points: Vec<GraphPoint> = container_samples
                .iter()
                .filter_map(|sample| {
                    sample.get(*metric).map(|value| {
                        let time_secs = sample.timestamp_ms as f64 / 1000.0;
                        let display_value = if metric.is_memory() {
                            value / 1024.0 // Convert KiB to MiB
                        } else {
                            value
                        };
                        GraphPoint {
                            x: time_secs,
                            y: display_value,
                            label: format!(
                                "{} - {}: {:.2} {}",
                                container_name,
                                metric.label(),
                                display_value,
                                if metric.is_memory() { "MiB" } else { metric.unit() }
                            ),
                        }
                    })
                })
                .collect();

            if !points.is_empty() {
                let name = if containers.len() > 1 || config.metrics.len() > 1 {
                    format!("{} - {}", container_name, metric.label())
                } else {
                    metric.label().to_string()
                };

                lines.push(GraphLine {
                    points,
                    name,
                    color: LINE_COLORS[color_idx % LINE_COLORS.len()],
                });
                color_idx += 1;
            }
        }
    }

    // Build event markers
    let significant_events: Vec<&TestEvent> = if config.show_events {
        result
            .events
            .iter()
            .filter(|e| !config.significant_events_only || e.is_significant)
            .collect()
    } else {
        Vec::new()
    };

    let event_markers: Vec<EventMarker> = significant_events
        .iter()
        .map(|e| EventMarker {
            x: e.timestamp_ms as f64 / 1000.0,
            label: format!("{}: {}", e.event_type, e.message),
            is_significant: e.is_significant,
        })
        .collect();

    // Render the plot with event highlighting
    render_plot_with_highlight(ui, &lines, &event_markers, config, *highlighted_event);

    // Render events panel if there are events
    if !significant_events.is_empty() {
        ui.add_space(10.0);
        render_events_panel(ui, &significant_events, &result.samples, highlighted_event, 100.0);
    }
}

/// Render a live-updating graph (simple version without events panel)
#[allow(dead_code)]
pub fn render_live_graph(
    ui: &mut Ui,
    samples: &[DataSample],
    events: &[TestEvent],
    containers: &[(Uuid, String)],
    metrics: &[MetricType],
    config: &GraphConfig,
) {
    if samples.is_empty() {
        ui.label("Waiting for data...");
        return;
    }

    let mut lines: Vec<GraphLine> = Vec::new();
    let mut color_idx = 0;

    for (container_id, container_name) in containers {
        let container_samples: Vec<&DataSample> = samples
            .iter()
            .filter(|s| s.container_id == *container_id)
            .collect();

        for metric in metrics {
            let points: Vec<GraphPoint> = container_samples
                .iter()
                .filter_map(|sample| {
                    sample.get(*metric).map(|value| {
                        let time_secs = sample.timestamp_ms as f64 / 1000.0;
                        let display_value = if metric.is_memory() {
                            value / 1024.0
                        } else {
                            value
                        };
                        GraphPoint {
                            x: time_secs,
                            y: display_value,
                            label: format!("{}: {:.2}", metric.label(), display_value),
                        }
                    })
                })
                .collect();

            if !points.is_empty() {
                let name = if containers.len() > 1 || metrics.len() > 1 {
                    format!("{} - {}", container_name, metric.label())
                } else {
                    metric.label().to_string()
                };

                lines.push(GraphLine {
                    points,
                    name,
                    color: LINE_COLORS[color_idx % LINE_COLORS.len()],
                });
                color_idx += 1;
            }
        }
    }

    let event_markers: Vec<EventMarker> = events
        .iter()
        .filter(|e| !config.significant_events_only || e.is_significant)
        .map(|e| EventMarker {
            x: e.timestamp_ms as f64 / 1000.0,
            label: format!("{}: {}", e.event_type, e.message),
            is_significant: e.is_significant,
        })
        .collect();

    render_plot(ui, &lines, &event_markers, config);
}

/// Render a live-updating graph with a separate events panel
/// Returns the updated highlighted event index (for state tracking)
pub fn render_live_graph_with_events_panel(
    ui: &mut Ui,
    samples: &[DataSample],
    events: &[TestEvent],
    containers: &[(Uuid, String)],
    metrics: &[MetricType],
    config: &GraphConfig,
    highlighted_event: &mut Option<usize>,
) {
    if samples.is_empty() {
        ui.label("Waiting for data...");
        return;
    }

    // Build lines for the graph
    let mut lines: Vec<GraphLine> = Vec::new();
    let mut color_idx = 0;

    for (container_id, container_name) in containers {
        let container_samples: Vec<&DataSample> = samples
            .iter()
            .filter(|s| s.container_id == *container_id)
            .collect();

        for metric in metrics {
            let points: Vec<GraphPoint> = container_samples
                .iter()
                .filter_map(|sample| {
                    sample.get(*metric).map(|value| {
                        let time_secs = sample.timestamp_ms as f64 / 1000.0;
                        let display_value = if metric.is_memory() {
                            value / 1024.0
                        } else {
                            value
                        };
                        GraphPoint {
                            x: time_secs,
                            y: display_value,
                            label: format!("{}: {:.2}", metric.label(), display_value),
                        }
                    })
                })
                .collect();

            if !points.is_empty() {
                let name = if containers.len() > 1 || metrics.len() > 1 {
                    format!("{} - {}", container_name, metric.label())
                } else {
                    metric.label().to_string()
                };

                lines.push(GraphLine {
                    points,
                    name,
                    color: LINE_COLORS[color_idx % LINE_COLORS.len()],
                });
                color_idx += 1;
            }
        }
    }

    // Build event markers
    let significant_events: Vec<&TestEvent> = events
        .iter()
        .filter(|e| !config.significant_events_only || e.is_significant)
        .collect();

    let event_markers: Vec<EventMarker> = significant_events
        .iter()
        .map(|e| EventMarker {
            x: e.timestamp_ms as f64 / 1000.0,
            label: format!("{}: {}", e.event_type, e.message),
            is_significant: e.is_significant,
        })
        .collect();

    // Render the graph first (takes full width)
    render_plot_with_highlight(ui, &lines, &event_markers, config, *highlighted_event);

    // Render the events panel below the graph
    ui.add_space(10.0);
    render_events_panel(ui, &significant_events, samples, highlighted_event, config.height.min(150.0));
}

/// Render the events panel as a table with detailed metadata
fn render_events_panel(
    ui: &mut Ui,
    events: &[&TestEvent],
    samples: &[DataSample],
    highlighted_event: &mut Option<usize>,
    max_height: f32,
) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("📋 Events").strong());
        ui.label(RichText::new(format!("({})", events.len())).weak());
    });

    if events.is_empty() {
        ui.label(RichText::new("No events detected").weak().italics());
        return;
    }

    // Compute metadata for all events
    let metadata = compute_event_metadata(events, samples, None);

    // Sticky header row (outside scroll area)
    egui::Grid::new("events_table_header")
        .num_columns(8)
        .spacing([16.0, 6.0])
        .show(ui, |ui| {
            ui.label(RichText::new("#").strong().underline());
            ui.label(RichText::new("Time").strong().underline());
            ui.label(RichText::new("Event").strong().underline());
            ui.label(RichText::new("Mem Before").strong().underline());
            ui.label(RichText::new("Mem After").strong().underline());
            ui.label(RichText::new("Δ Memory").strong().underline());
            ui.label(RichText::new("Duration").strong().underline());
            ui.label(RichText::new("CPU").strong().underline());
            ui.end_row();
        });

    ui.separator();

    // Scrollable data rows
    egui::ScrollArea::both()
        .id_salt("events_table_scroll")
        .max_height(max_height - 30.0) // Account for header height
        .show(ui, |ui| {
            egui::Grid::new("events_table_data")
                .num_columns(8)
                .spacing([16.0, 6.0])
                .striped(true)
                .show(ui, |ui| {
                    // Data rows
                    for (idx, (event, meta)) in events.iter().zip(metadata.iter()).enumerate() {
                        let is_highlighted = *highlighted_event == Some(idx);
                        let time_secs = event.timestamp_ms as f64 / 1000.0;

                        let highlight_color = Color32::from_rgb(255, 150, 100);
                        let event_color = Color32::from_rgb(255, 200, 100);

                        // # (Event count) column
                        let count_text = format!("#{}", meta.event_count);
                        let count_label = if is_highlighted {
                            RichText::new(count_text).strong().color(highlight_color)
                        } else {
                            RichText::new(count_text).weak()
                        };
                        let r1 = ui.label(count_label);

                        // Time column
                        let time_text = format!("{:.2}s", time_secs);
                        let time_label = if is_highlighted {
                            RichText::new(time_text).strong().color(highlight_color)
                        } else {
                            RichText::new(time_text).monospace()
                        };
                        let r2 = ui.label(time_label);

                        // Event type column (with block height if available)
                        let event_text = if let Some(height) = meta.block_height {
                            format!("{} (h:{})", event.event_type, height)
                        } else {
                            event.event_type.clone()
                        };
                        let event_label = if is_highlighted {
                            RichText::new(event_text).strong().color(event_color)
                        } else {
                            RichText::new(event_text).color(event_color)
                        };
                        let r3 = ui.label(event_label);

                        // Memory Before column
                        let mem_before_text = meta.memory_before
                            .map(|m| format_memory(m))
                            .unwrap_or_else(|| "-".to_string());
                        let mem_before_label = if is_highlighted {
                            RichText::new(mem_before_text).strong()
                        } else {
                            RichText::new(mem_before_text).weak()
                        };
                        let r4 = ui.label(mem_before_label);

                        // Memory After column
                        let mem_after_text = meta.memory_after
                            .map(|m| format_memory(m))
                            .unwrap_or_else(|| "-".to_string());
                        let mem_after_label = if is_highlighted {
                            RichText::new(mem_after_text).strong()
                        } else {
                            RichText::new(mem_after_text).weak()
                        };
                        let r5 = ui.label(mem_after_label);

                        // Memory Delta column (colored based on positive/negative)
                        let (delta_text, delta_color) = if let Some(delta) = meta.memory_delta {
                            let text = format_memory_delta(delta);
                            let color = if delta > 0.0 {
                                Color32::from_rgb(255, 100, 100) // Red for increase
                            } else if delta < 0.0 {
                                Color32::from_rgb(100, 255, 100) // Green for decrease
                            } else {
                                Color32::GRAY
                            };
                            (text, color)
                        } else {
                            ("-".to_string(), Color32::GRAY)
                        };
                        let delta_label = if is_highlighted {
                            RichText::new(delta_text).strong().color(delta_color)
                        } else {
                            RichText::new(delta_text).color(delta_color)
                        };
                        let r6 = ui.label(delta_label);

                        // Duration column
                        let duration_text = meta.duration_ms
                            .map(|d| {
                                if d >= 1000 {
                                    format!("{:.2}s", d as f64 / 1000.0)
                                } else {
                                    format!("{}ms", d)
                                }
                            })
                            .unwrap_or_else(|| "-".to_string());
                        let duration_label = if is_highlighted {
                            RichText::new(duration_text).strong()
                        } else {
                            RichText::new(duration_text).weak()
                        };
                        let r7 = ui.label(duration_label);

                        // CPU column (with spike indicator)
                        let cpu_text = if let Some(cpu) = meta.cpu_around_event {
                            if meta.cpu_spike {
                                format!("{:.1}% ⚡", cpu)
                            } else {
                                format!("{:.1}%", cpu)
                            }
                        } else {
                            "-".to_string()
                        };
                        let cpu_color = if meta.cpu_spike {
                            Color32::from_rgb(255, 200, 100) // Yellow for spike
                        } else {
                            Color32::GRAY
                        };
                        let cpu_label = if is_highlighted {
                            RichText::new(cpu_text).strong().color(cpu_color)
                        } else {
                            RichText::new(cpu_text).color(cpu_color)
                        };
                        let r8 = ui.label(cpu_label);

                        ui.end_row();

                        // Track hover state for any cell in the row
                        if r1.hovered() || r2.hovered() || r3.hovered() || r4.hovered()
                            || r5.hovered() || r6.hovered() || r7.hovered() || r8.hovered() {
                            *highlighted_event = Some(idx);
                        }
                    }
                });

            // Clear highlight when not hovering any event
            if !ui.ui_contains_pointer() {
                *highlighted_event = None;
            }
        });
}

/// Render the actual plot
fn render_plot(
    ui: &mut Ui,
    lines: &[GraphLine],
    event_markers: &[EventMarker],
    config: &GraphConfig,
) {
    render_plot_with_highlight(ui, lines, event_markers, config, None)
}

/// Render the actual plot with optional event highlighting
fn render_plot_with_highlight(
    ui: &mut Ui,
    lines: &[GraphLine],
    event_markers: &[EventMarker],
    config: &GraphConfig,
    highlighted_event_idx: Option<usize>,
) {
    if lines.is_empty() {
        ui.label("No data to display");
        return;
    }

    // Legend (only for metric lines, not events)
    ui.horizontal(|ui| {
        for line in lines {
            ui.colored_label(line.color, &format!("■ {}", line.name));
        }
    });

    // Compute bounds
    let (min_x, max_x, min_y, max_y) = compute_bounds(lines, config);

    let mut plot = Plot::new("bench_graph")
        .height(config.height)
        .allow_drag(config.interactive)
        .allow_zoom(config.interactive)
        .allow_scroll(config.interactive)
        .x_axis_label("Time (seconds)")
        .y_axis_label(if config.metrics.first().map(|m| m.is_memory()).unwrap_or(true) {
            "Memory (MiB)"
        } else {
            "Value"
        })
        .legend(egui_plot::Legend::default());

    // Set bounds if specified
    if config.x_range.is_some() || config.y_scale.is_some() {
        plot = plot.include_x(min_x).include_x(max_x).include_y(min_y).include_y(max_y);
    }

    plot.show(ui, |plot_ui| {
        // Draw lines
        for line in lines {
            let plot_points: PlotPoints = PlotPoints::from_iter(
                line.points.iter().map(|p| [p.x, p.y])
            );
            plot_ui.line(
                Line::new(plot_points)
                    .color(line.color)
                    .name(&line.name)
                    .width(2.0),
            );
        }

        // Draw event markers (without names - they don't appear in legend)
        for (idx, marker) in event_markers.iter().enumerate() {
            let is_highlighted = highlighted_event_idx == Some(idx);
            let color = if is_highlighted {
                // Bright highlight color when hovered
                Color32::from_rgb(255, 100, 50)
            } else if marker.is_significant {
                Color32::from_rgba_premultiplied(255, 200, 100, 150)
            } else {
                Color32::from_rgba_premultiplied(150, 150, 150, 80)
            };

            let width = if is_highlighted { 3.0 } else { 1.0 };

            // Don't use .name() so events don't appear in the legend
            plot_ui.vline(
                VLine::new(marker.x)
                    .color(color)
                    .width(width)
                    .style(if is_highlighted {
                        egui_plot::LineStyle::Solid
                    } else {
                        egui_plot::LineStyle::dashed_loose()
                    }),
            );
        }
    });
}

/// Compute bounds for the plot
fn compute_bounds(lines: &[GraphLine], config: &GraphConfig) -> (f64, f64, f64, f64) {
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;

    for line in lines {
        for point in &line.points {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
        }
    }

    // Apply config overrides
    if let Some((x_min, x_max)) = config.x_range {
        min_x = x_min;
        max_x = x_max;
    }
    if let Some((y_min, y_max)) = config.y_scale {
        min_y = y_min;
        max_y = y_max;
    }

    // Add some padding
    let y_padding = (max_y - min_y) * 0.1;
    min_y = (min_y - y_padding).max(0.0);
    max_y += y_padding;

    (min_x, max_x, min_y, max_y)
}

/// Render a comparison graph between two test results
pub fn render_comparison_graph(
    ui: &mut Ui,
    baseline: &TestResult,
    comparison: &TestResult,
    metric: MetricType,
    config: &GraphConfig,
) {
    let baseline_name = format!("Baseline: {}", baseline.config.name);
    let comparison_name = format!("Comparison: {}", comparison.config.name);

    let mut lines = Vec::new();

    // Baseline lines
    for container in &baseline.config.containers {
        let points: Vec<GraphPoint> = baseline
            .samples_for_container(container.id)
            .iter()
            .filter_map(|sample| {
                sample.get(metric).map(|value| {
                    let display_value = if metric.is_memory() {
                        value / 1024.0
                    } else {
                        value
                    };
                    GraphPoint {
                        x: sample.timestamp_ms as f64 / 1000.0,
                        y: display_value,
                        label: format!("{}: {:.2}", baseline_name, display_value),
                    }
                })
            })
            .collect();

        if !points.is_empty() {
            lines.push(GraphLine {
                points,
                name: baseline_name.clone(),
                color: LINE_COLORS[0],
            });
        }
    }

    // Comparison lines
    for container in &comparison.config.containers {
        let points: Vec<GraphPoint> = comparison
            .samples_for_container(container.id)
            .iter()
            .filter_map(|sample| {
                sample.get(metric).map(|value| {
                    let display_value = if metric.is_memory() {
                        value / 1024.0
                    } else {
                        value
                    };
                    GraphPoint {
                        x: sample.timestamp_ms as f64 / 1000.0,
                        y: display_value,
                        label: format!("{}: {:.2}", comparison_name, display_value),
                    }
                })
            })
            .collect();

        if !points.is_empty() {
            lines.push(GraphLine {
                points,
                name: comparison_name.clone(),
                color: LINE_COLORS[1],
            });
        }
    }

    render_plot(ui, &lines, &[], config);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph_points(n: usize) -> Vec<GraphPoint> {
        (0..n)
            .map(|i| GraphPoint {
                x: i as f64,
                y: 100.0 + i as f64 * 10.0,
                label: format!("Point {}", i),
            })
            .collect()
    }

    #[test]
    fn test_graph_config_default() {
        let config = GraphConfig::default();
        assert_eq!(config.metrics.len(), 1);
        assert!(config.show_events);
        assert!(config.interactive);
    }

    #[test]
    fn test_graph_config_single_metric() {
        let config = GraphConfig::single_metric(MetricType::PmaRss);
        assert_eq!(config.metrics, vec![MetricType::PmaRss]);
    }

    #[test]
    fn test_graph_config_multi_metric() {
        let config = GraphConfig::multi_metric(vec![MetricType::VmRss, MetricType::PmaRss]);
        assert_eq!(config.metrics.len(), 2);
    }

    #[test]
    fn test_compute_bounds_empty() {
        let lines: Vec<GraphLine> = vec![];
        let config = GraphConfig::default();
        let (min_x, max_x, _min_y, _max_y) = compute_bounds(&lines, &config);
        assert_eq!(min_x, f64::MAX);
        assert_eq!(max_x, f64::MIN);
    }

    #[test]
    fn test_compute_bounds_with_data() {
        let lines = vec![GraphLine {
            points: make_graph_points(10),
            name: "Test".to_string(),
            color: LINE_COLORS[0],
        }];
        let config = GraphConfig::default();
        let (min_x, max_x, min_y, max_y) = compute_bounds(&lines, &config);

        assert_eq!(min_x, 0.0);
        assert_eq!(max_x, 9.0);
        // min_y = 100, max_y = 190, with padding
        assert!(min_y < 100.0);
        assert!(max_y > 190.0);
    }

    #[test]
    fn test_compute_bounds_with_override() {
        let lines = vec![GraphLine {
            points: make_graph_points(10),
            name: "Test".to_string(),
            color: LINE_COLORS[0],
        }];
        let config = GraphConfig {
            x_range: Some((0.0, 20.0)),
            y_scale: Some((0.0, 500.0)),
            ..Default::default()
        };
        let (min_x, max_x, min_y, max_y) = compute_bounds(&lines, &config);

        assert_eq!(min_x, 0.0);
        assert_eq!(max_x, 20.0);
        assert_eq!(min_y, 0.0);
        assert!(max_y > 500.0); // Padding added
    }

    #[test]
    fn test_graph_line() {
        let line = GraphLine {
            points: make_graph_points(5),
            name: "Test Line".to_string(),
            color: LINE_COLORS[0],
        };

        assert_eq!(line.points.len(), 5);
        assert_eq!(line.name, "Test Line");
    }

    #[test]
    fn test_event_marker() {
        let marker = EventMarker {
            x: 5.0,
            label: "Checkpoint".to_string(),
            is_significant: true,
        };

        assert_eq!(marker.x, 5.0);
        assert!(marker.is_significant);
    }
}
