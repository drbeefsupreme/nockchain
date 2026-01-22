//! Time-series graph visualization using egui_plot
//!
//! Displays memory usage and other metrics over time with event markers.

use egui::{Color32, RichText, Ui};
use egui_plot::{Line, Plot, PlotPoints, VLine};
use uuid::Uuid;

use crate::config::MetricType;
use crate::storage::{DataSample, TestEvent, TestResult};

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
pub fn render_graph(ui: &mut Ui, result: &TestResult, config: &GraphConfig) {
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

    // Render the plot
    render_plot(ui, &lines, &event_markers, config);

    // Render events panel if there are events
    if !significant_events.is_empty() {
        ui.add_space(10.0);
        // Use a dummy highlighted_event since we don't track hover state for results view
        let mut highlighted: Option<usize> = config.highlighted_event;
        render_events_panel(ui, &significant_events, &mut highlighted, 100.0);
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

    // Render the events panel below the graph for now
    // (side-by-side layout is complex with egui_plot)
    ui.add_space(10.0);
    render_events_panel(ui, &significant_events, highlighted_event, config.height.min(150.0));
}

/// Render the events panel as a table
fn render_events_panel(
    ui: &mut Ui,
    events: &[&TestEvent],
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

    egui::ScrollArea::vertical()
        .max_height(max_height)
        .show(ui, |ui| {
            egui::Grid::new("events_table")
                .num_columns(3)
                .spacing([20.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    // Header row
                    ui.label(RichText::new("Time").strong().underline());
                    ui.label(RichText::new("Event").strong().underline());
                    ui.label(RichText::new("Container").strong().underline());
                    ui.end_row();

                    // Data rows
                    for (idx, event) in events.iter().enumerate() {
                        let is_highlighted = *highlighted_event == Some(idx);
                        let time_secs = event.timestamp_ms as f64 / 1000.0;

                        // Time column
                        let time_text = format!("{:.2}s", time_secs);
                        let time_label = if is_highlighted {
                            RichText::new(time_text).strong().color(Color32::from_rgb(255, 150, 100))
                        } else {
                            RichText::new(time_text).monospace()
                        };
                        let time_response = ui.label(time_label);

                        // Event type column
                        let event_label = if is_highlighted {
                            RichText::new(&event.event_type).strong().color(Color32::from_rgb(255, 200, 100))
                        } else {
                            RichText::new(&event.event_type).color(Color32::from_rgb(255, 200, 100))
                        };
                        let event_response = ui.label(event_label);

                        // Container column (extract from container_id - show short form)
                        let container_text = format!("{}", &event.container_id.to_string()[..8]);
                        let container_label = if is_highlighted {
                            RichText::new(container_text).strong()
                        } else {
                            RichText::new(container_text).weak()
                        };
                        let container_response = ui.label(container_label);

                        ui.end_row();

                        // Track hover state for any cell in the row
                        if time_response.hovered() || event_response.hovered() || container_response.hovered() {
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
