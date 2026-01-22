//! Main application state and UI
//!
//! Ties together all the panels and manages the overall application state.

use std::thread;

use eframe::egui;
use egui::{CentralPanel, Context, RichText, SidePanel, TopBottomPanel, Ui};
use uuid::Uuid;

use crate::config::{MetricType, TestConfig};
use crate::git_panel::GitPanel;
use crate::graph::{render_graph, render_live_graph, GraphConfig};
use crate::runner::{RunnerHandle, RunnerMessage, TestRunner};
use crate::storage::{DataSample, TestEvent, TestResult, TestStorage};
use crate::terminal::TerminalPanel;
use crate::test_panel::{TestListPanel, TestPanel};

/// Application view/tab
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AppView {
    /// Create a new test
    #[default]
    NewTest,
    /// View saved tests
    SavedTests,
    /// View results
    Results,
    /// Compare results
    Compare,
    /// Settings
    Settings,
}

impl AppView {
    fn label(&self) -> &'static str {
        match self {
            AppView::NewTest => "New Test",
            AppView::SavedTests => "Saved Tests",
            AppView::Results => "Results",
            AppView::Compare => "Compare",
            AppView::Settings => "Settings",
        }
    }
}

/// Running test state
struct RunningTestState {
    test_id: Uuid,
    config: TestConfig,
    samples: Vec<DataSample>,
    events: Vec<TestEvent>,
    terminal_id: Uuid,
    elapsed_secs: f64,
    total_secs: f64,
    sample_count: usize,
}

/// Main application
pub struct BenchApp {
    /// Current view
    view: AppView,

    /// Test creation panel
    test_panel: TestPanel,

    /// Saved tests list
    test_list: TestListPanel,

    /// Git panel
    git_panel: GitPanel,

    /// Terminal panel
    terminals: TerminalPanel,

    /// Storage for results
    storage: Option<TestStorage>,

    /// Storage error message
    storage_error: Option<String>,

    /// Runner handle
    runner: Option<RunnerHandle>,

    /// Currently running test
    running_test: Option<RunningTestState>,

    /// Results list
    results: Vec<crate::storage::TestResultSummary>,

    /// Selected result for viewing
    selected_result: Option<TestResult>,

    /// Comparison baseline
    compare_baseline: Option<TestResult>,

    /// Comparison target
    compare_target: Option<TestResult>,

    /// Graph configuration
    graph_config: GraphConfig,

    /// Status message
    status: Option<String>,

    /// Docker availability
    docker_available: Option<bool>,

    /// Data directory path
    data_dir: String,
}

impl Default for BenchApp {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchApp {
    /// Create a new application
    pub fn new() -> Self {
        let data_dir = TestStorage::default_location();
        let storage = TestStorage::new(&data_dir).ok();

        let results = storage
            .as_ref()
            .and_then(|s| s.list_results().ok())
            .unwrap_or_default();

        let configs = storage
            .as_ref()
            .and_then(|s| s.list_configs().ok())
            .unwrap_or_default();

        let mut test_list = TestListPanel::new();
        test_list.set_configs(configs);

        Self {
            view: AppView::default(),
            test_panel: TestPanel::default(),
            test_list,
            git_panel: GitPanel::new(),
            terminals: TerminalPanel::new(),
            storage,
            storage_error: None,
            runner: None,
            running_test: None,
            results,
            selected_result: None,
            compare_baseline: None,
            compare_target: None,
            graph_config: GraphConfig::default(),
            status: None,
            docker_available: None,
            data_dir: data_dir.to_string_lossy().to_string(),
        }
    }

    /// Initialize the runner (call after app creation)
    pub fn init_runner(&mut self) {
        let (runner, rx, tx) = TestRunner::new();

        // Spawn the runner thread
        thread::spawn(move || {
            runner.run();
        });

        let handle = RunnerHandle::new(tx, rx);

        // Check Docker availability
        let _ = handle.check_docker();

        self.runner = Some(handle);
    }

    /// Process runner messages
    fn process_runner_messages(&mut self) {
        if let Some(ref mut runner) = self.runner {
            for msg in runner.poll() {
                match msg {
                    RunnerMessage::Started { test_id } => {
                        self.status = Some(format!("Test {} started", test_id));
                    }
                    RunnerMessage::Sample {
                        test_id,
                        container_id: _,
                        sample,
                    } => {
                        if let Some(ref mut running) = self.running_test {
                            if running.test_id == test_id {
                                running.samples.push(sample);
                            }
                        }
                    }
                    RunnerMessage::Event { test_id, event } => {
                        if let Some(ref mut running) = self.running_test {
                            if running.test_id == test_id {
                                running.events.push(event);
                            }
                        }
                    }
                    RunnerMessage::Log {
                        test_id,
                        container_id: _,
                        line,
                        is_error,
                    } => {
                        if let Some(ref running) = self.running_test {
                            if running.test_id == test_id {
                                if is_error {
                                    self.terminals.push_error(running.terminal_id, &line);
                                } else {
                                    self.terminals.push_line(running.terminal_id, &line);
                                }
                            }
                        }
                    }
                    RunnerMessage::Progress {
                        test_id,
                        elapsed_secs,
                        total_secs,
                        sample_count,
                    } => {
                        if let Some(ref mut running) = self.running_test {
                            if running.test_id == test_id {
                                running.elapsed_secs = elapsed_secs;
                                running.total_secs = total_secs;
                                running.sample_count = sample_count;
                            }
                        }
                    }
                    RunnerMessage::Completed { test_id, result } => {
                        // Check if the result indicates failure
                        let status_msg = if result.status == crate::storage::TestStatus::Failed {
                            let error_msg = result.error.as_deref().unwrap_or("Unknown error");
                            format!("Test {} failed: {}", test_id, error_msg)
                        } else {
                            format!("Test {} completed", test_id)
                        };
                        self.status = Some(status_msg.clone());

                        // Save result
                        if let Some(ref storage) = self.storage {
                            let _ = storage.save_result(&result);
                            self.refresh_results();
                        }

                        // Mark terminal as inactive
                        if let Some(ref running) = self.running_test {
                            if running.test_id == test_id {
                                if let Some(term) = self.terminals.get_mut(running.terminal_id) {
                                    if result.status == crate::storage::TestStatus::Failed {
                                        let error_msg = result.error.as_deref().unwrap_or("Unknown error");
                                        term.push_error(&format!("Test failed: {}", error_msg));
                                    } else {
                                        term.push_system("Test completed");
                                    }
                                    term.mark_inactive();
                                }
                            }
                        }

                        // Clear running state
                        self.running_test = None;
                    }
                    RunnerMessage::Failed { test_id, error } => {
                        self.status = Some(format!("Test {} failed: {}", test_id, error));

                        if let Some(ref running) = self.running_test {
                            if running.test_id == test_id {
                                if let Some(term) = self.terminals.get_mut(running.terminal_id) {
                                    term.push_error(&format!("Test failed: {}", error));
                                    term.mark_inactive();
                                }
                            }
                        }

                        self.running_test = None;
                    }
                    RunnerMessage::DockerAvailable(available) => {
                        self.docker_available = Some(available);
                        if !available {
                            self.status = Some("Docker is not available".to_string());
                        }
                    }
                }
            }
        }
    }

    /// Refresh the results list from storage
    fn refresh_results(&mut self) {
        if let Some(ref storage) = self.storage {
            self.results = storage.list_results().unwrap_or_default();
        }
    }

    /// Start a test
    fn start_test(&mut self, config: TestConfig) {
        if self.running_test.is_some() {
            self.status = Some("A test is already running".to_string());
            return;
        }

        let test_id = Uuid::new_v4();

        // Create terminal for this test
        let terminal_id = self.terminals.add_terminal(&config.name);
        self.terminals.push_system(terminal_id, "Starting test...");

        // Create running state
        self.running_test = Some(RunningTestState {
            test_id,
            config: config.clone(),
            samples: Vec::new(),
            events: Vec::new(),
            terminal_id,
            elapsed_secs: 0.0,
            total_secs: config.duration_secs as f64,
            sample_count: 0,
        });

        // Send to runner
        if let Some(ref runner) = self.runner {
            let _ = runner.start_test(config);
        }
    }

    /// Stop the current test
    fn stop_test(&mut self) {
        if let Some(ref running) = self.running_test {
            if let Some(ref runner) = self.runner {
                let _ = runner.stop_test(running.test_id);
            }
            self.terminals
                .push_system(running.terminal_id, "Stopping test...");
        }
    }

    /// Show the navigation sidebar
    fn show_sidebar(&mut self, ui: &mut Ui) {
        ui.heading("Nockchain Bench");
        ui.separator();

        for view in [
            AppView::NewTest,
            AppView::SavedTests,
            AppView::Results,
            AppView::Compare,
            AppView::Settings,
        ] {
            if ui
                .selectable_label(self.view == view, view.label())
                .clicked()
            {
                self.view = view;
            }
        }

        ui.separator();

        // Docker status
        match self.docker_available {
            Some(true) => {
                ui.colored_label(egui::Color32::GREEN, "Docker: Available");
            }
            Some(false) => {
                ui.colored_label(egui::Color32::RED, "Docker: Unavailable");
            }
            None => {
                ui.label("Docker: Checking...");
            }
        }

        // Running test indicator
        if let Some(ref running) = self.running_test {
            ui.separator();
            ui.label(RichText::new("Running Test").strong());
            ui.label(&running.config.name);
            ui.add(egui::ProgressBar::new(
                (running.elapsed_secs / running.total_secs) as f32,
            ));
            ui.label(format!(
                "{:.0}s / {:.0}s | {} samples",
                running.elapsed_secs, running.total_secs, running.sample_count
            ));
            if ui.button("Stop").clicked() {
                self.stop_test();
            }
        }
    }

    /// Show the main content area
    fn show_content(&mut self, ui: &mut Ui) {
        match self.view {
            AppView::NewTest => self.show_new_test(ui),
            AppView::SavedTests => self.show_saved_tests(ui),
            AppView::Results => self.show_results(ui),
            AppView::Compare => self.show_compare(ui),
            AppView::Settings => self.show_settings(ui),
        }
    }

    /// Show the new test view
    fn show_new_test(&mut self, ui: &mut Ui) {
        ui.heading("Create New Test");
        ui.separator();

        let response = self.test_panel.show(ui);

        if response.run_requested {
            let config = self.test_panel.get_config();
            self.start_test(config);
        }

        if response.save_requested {
            let config = self.test_panel.get_config();
            if let Some(ref storage) = self.storage {
                match storage.save_config(&config) {
                    Ok(_) => {
                        self.status = Some("Test configuration saved".to_string());
                        // Refresh the saved tests list
                        if let Ok(configs) = storage.list_configs() {
                            self.test_list.set_configs(configs);
                        }
                    }
                    Err(e) => {
                        self.status = Some(format!("Failed to save: {}", e));
                    }
                }
            }
        }
    }

    /// Show saved tests view
    fn show_saved_tests(&mut self, ui: &mut Ui) {
        ui.heading("Saved Tests");
        ui.separator();

        let response = self.test_list.show(ui);

        if let Some(id) = response.load_requested {
            if let Some(ref storage) = self.storage {
                if let Ok(config) = storage.load_config(id) {
                    self.test_panel = TestPanel::new(config);
                    self.view = AppView::NewTest;
                }
            }
        }

        if let Some(id) = response.run_requested {
            if let Some(ref storage) = self.storage {
                if let Ok(config) = storage.load_config(id) {
                    self.start_test(config);
                }
            }
        }

        if let Some(id) = response.delete_requested {
            // Would delete the config
            self.status = Some(format!("Delete requested for {}", id));
        }
    }

    /// Show results view
    fn show_results(&mut self, ui: &mut Ui) {
        ui.heading("Test Results");
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Refresh").clicked() {
                self.refresh_results();
            }
            ui.label(format!("{} results", self.results.len()));
        });

        ui.separator();

        // Results list
        egui::ScrollArea::vertical()
            .max_height(200.0)
            .show(ui, |ui| {
                for summary in &self.results {
                    let is_selected = self
                        .selected_result
                        .as_ref()
                        .map(|r| r.id == summary.id)
                        .unwrap_or(false);

                    let label = format!(
                        "{} | {} | {} samples | {}",
                        summary.name,
                        summary.status.label(),
                        summary.sample_count,
                        summary.started_at.format("%Y-%m-%d %H:%M")
                    );

                    if ui.selectable_label(is_selected, label).clicked() {
                        // Load the full result
                        if let Some(ref storage) = self.storage {
                            if let Ok(result) = storage.load_result(summary.id) {
                                self.selected_result = Some(result);
                            }
                        }
                    }
                }
            });

        // Selected result details
        if let Some(ref result) = self.selected_result {
            ui.separator();
            ui.heading(&result.config.name);

            ui.horizontal(|ui| {
                let status_color = match result.status {
                    crate::storage::TestStatus::Success => egui::Color32::GREEN,
                    crate::storage::TestStatus::Failed => egui::Color32::RED,
                    crate::storage::TestStatus::Cancelled => egui::Color32::YELLOW,
                    crate::storage::TestStatus::Running => egui::Color32::LIGHT_BLUE,
                };
                ui.colored_label(status_color, format!("Status: {}", result.status.label()));
                ui.label(format!("Duration: {:.1}s", result.duration().as_secs_f64()));
                ui.label(format!("Samples: {}", result.samples.len()));
            });

            // Show error message if test failed
            if let Some(ref error) = result.error {
                ui.colored_label(egui::Color32::RED, format!("Error: {}", error));
            }

            ui.separator();

            // Graph
            render_graph(ui, result, &self.graph_config);

            // Statistics
            ui.separator();
            ui.heading("Statistics");

            for container in &result.config.containers {
                if let Some(stats) = result.statistics.get(&container.id) {
                    ui.collapsing(&container.name, |ui| {
                        egui::Grid::new(format!("stats_{}", container.id))
                            .num_columns(4)
                            .show(ui, |ui| {
                                ui.label("Metric");
                                ui.label("Average");
                                ui.label("Peak");
                                ui.label("P95");
                                ui.end_row();

                                for metric in &result.config.metrics {
                                    if let (Some(avg), Some(peak), Some(p95)) = (
                                        stats.average.get(metric),
                                        stats.peak.get(metric),
                                        stats.p95.get(metric),
                                    ) {
                                        ui.label(metric.label());
                                        ui.label(format!("{:.2}", avg));
                                        ui.label(format!("{:.2}", peak));
                                        ui.label(format!("{:.2}", p95));
                                        ui.end_row();
                                    }
                                }
                            });
                    });
                }
            }
        }
    }

    /// Show compare view
    fn show_compare(&mut self, ui: &mut Ui) {
        ui.heading("Compare Results");
        ui.separator();

        ui.label("Select two results to compare:");

        ui.horizontal(|ui| {
            ui.label("Baseline:");
            egui::ComboBox::from_id_salt("baseline")
                .selected_text(
                    self.compare_baseline
                        .as_ref()
                        .map(|r| r.config.name.as_str())
                        .unwrap_or("Select..."),
                )
                .show_ui(ui, |ui| {
                    for summary in &self.results {
                        if ui.selectable_label(false, &summary.name).clicked() {
                            if let Some(ref storage) = self.storage {
                                if let Ok(result) = storage.load_result(summary.id) {
                                    self.compare_baseline = Some(result);
                                }
                            }
                        }
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("Comparison:");
            egui::ComboBox::from_id_salt("comparison")
                .selected_text(
                    self.compare_target
                        .as_ref()
                        .map(|r| r.config.name.as_str())
                        .unwrap_or("Select..."),
                )
                .show_ui(ui, |ui| {
                    for summary in &self.results {
                        if ui.selectable_label(false, &summary.name).clicked() {
                            if let Some(ref storage) = self.storage {
                                if let Ok(result) = storage.load_result(summary.id) {
                                    self.compare_target = Some(result);
                                }
                            }
                        }
                    }
                });
        });

        if self.compare_baseline.is_some() && self.compare_target.is_some() {
            ui.separator();

            let baseline = self.compare_baseline.as_ref().unwrap();
            let target = self.compare_target.as_ref().unwrap();

            let comparison = crate::storage::TestComparison::compare(baseline, target);

            // Summary
            ui.horizontal(|ui| {
                if comparison.is_regression {
                    ui.colored_label(egui::Color32::RED, "REGRESSION DETECTED");
                } else {
                    ui.colored_label(egui::Color32::GREEN, "No regression");
                }
                ui.label(format!("Confidence: {:.0}%", comparison.confidence * 100.0));
            });

            ui.separator();

            // Changes table
            egui::Grid::new("comparison_grid")
                .num_columns(3)
                .show(ui, |ui| {
                    ui.label("Container / Metric");
                    ui.label("Change");
                    ui.label("");
                    ui.end_row();

                    for (container_name, changes) in &comparison.changes {
                        for (metric, change) in changes {
                            ui.label(format!("{} - {}", container_name, metric.label()));
                            let (text, color) = if *change > 5.0 {
                                (format!("+{:.1}%", change), egui::Color32::RED)
                            } else if *change < -5.0 {
                                (format!("{:.1}%", change), egui::Color32::GREEN)
                            } else {
                                (format!("{:.1}%", change), egui::Color32::GRAY)
                            };
                            ui.colored_label(color, text);
                            ui.end_row();
                        }
                    }
                });

            // Comparison graph
            ui.separator();
            ui.heading("Comparison Graph");

            egui::ComboBox::from_id_salt("compare_metric")
                .selected_text(self.graph_config.metrics.first().map(|m| m.label()).unwrap_or("VmRss"))
                .show_ui(ui, |ui| {
                    for metric in MetricType::all() {
                        if ui.selectable_label(false, metric.label()).clicked() {
                            self.graph_config.metrics = vec![*metric];
                        }
                    }
                });

            crate::graph::render_comparison_graph(
                ui,
                baseline,
                target,
                self.graph_config.metrics.first().copied().unwrap_or(MetricType::VmRss),
                &self.graph_config,
            );
        }
    }

    /// Show settings view
    fn show_settings(&mut self, ui: &mut Ui) {
        ui.heading("Settings");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Data Directory:");
            ui.text_edit_singleline(&mut self.data_dir);
            if ui.button("Change").clicked() {
                match TestStorage::new(&self.data_dir) {
                    Ok(storage) => {
                        self.storage = Some(storage);
                        self.storage_error = None;
                        self.refresh_results();
                    }
                    Err(e) => {
                        self.storage_error = Some(e.to_string());
                    }
                }
            }
        });

        if let Some(ref error) = self.storage_error {
            ui.colored_label(egui::Color32::RED, error);
        }

        ui.separator();

        // Graph settings
        ui.heading("Graph Settings");
        ui.checkbox(&mut self.graph_config.show_events, "Show events on graphs");
        ui.checkbox(
            &mut self.graph_config.significant_events_only,
            "Significant events only",
        );
        ui.horizontal(|ui| {
            ui.label("Graph height:");
            ui.add(egui::DragValue::new(&mut self.graph_config.height).range(100.0..=800.0));
        });

        ui.separator();

        // Git settings
        ui.heading("Git Repository");
        self.git_panel.show(ui);
    }

    /// Show the terminal panel at the bottom
    fn show_terminals(&mut self, ui: &mut Ui) {
        if !self.terminals.is_empty() {
            self.terminals.show(ui);
        }
    }

    /// Show live graph during test run
    fn show_live_view(&mut self, ui: &mut Ui) {
        if let Some(ref running) = self.running_test {
            ui.separator();
            ui.heading("Live View");

            let containers: Vec<(Uuid, String)> = running
                .config
                .containers
                .iter()
                .map(|c| (c.id, c.name.clone()))
                .collect();

            render_live_graph(
                ui,
                &running.samples,
                &running.events,
                &containers,
                &running.config.metrics,
                &self.graph_config,
            );
        }
    }
}

impl eframe::App for BenchApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        // Process runner messages
        self.process_runner_messages();

        // Request continuous repaint while test is running
        if self.running_test.is_some() {
            ctx.request_repaint();
        }

        // Sidebar
        SidePanel::left("sidebar")
            .resizable(true)
            .default_width(200.0)
            .show(ctx, |ui| {
                self.show_sidebar(ui);
            });

        // Bottom terminal panel
        TopBottomPanel::bottom("terminal")
            .resizable(true)
            .default_height(150.0)
            .show(ctx, |ui| {
                self.show_terminals(ui);
            });

        // Status bar
        TopBottomPanel::bottom("status")
            .exact_height(20.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(ref status) = self.status {
                        ui.label(status);
                    }
                });
            });

        // Main content
        CentralPanel::default().show(ctx, |ui| {
            self.show_content(ui);

            // Show live view if test is running
            self.show_live_view(ui);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_view_labels() {
        assert_eq!(AppView::NewTest.label(), "New Test");
        assert_eq!(AppView::Results.label(), "Results");
    }

    #[test]
    fn test_app_default() {
        let app = BenchApp::new();
        assert_eq!(app.view, AppView::NewTest);
        assert!(app.running_test.is_none());
    }
}
