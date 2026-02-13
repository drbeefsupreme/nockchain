//! Main application state and UI
//!
//! Ties together all the panels and manages the overall application state.

use std::thread;

use eframe::egui;
use egui::{CentralPanel, Context, RichText, SidePanel, TopBottomPanel, Ui};
use uuid::Uuid;

use crate::config::{BenchmarkMode, MetricType, SolExtractOptions, TestConfig};
use crate::git_panel::GitPanel;
use crate::graph::{render_graph, render_live_graph_with_events_panel, GraphConfig};
use crate::runner::{RunnerHandle, RunnerMessage, TestRunner};
use crate::storage::{
    DataSample, SolBenchResult, SolSweepResult, TestEvent, TestResult, TestStorage,
};
use crate::terminal::TerminalPanel;
use crate::test_panel::{TestListPanel, TestPanel};

/// Application view/tab
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AppView {
    /// Create a new test
    #[default]
    NewTest,
    /// Create SOL archive test fixtures
    SolTestCreator,
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
            AppView::SolTestCreator => "SOL Test Creator",
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

/// Running SOL archive creation state
struct RunningSolArchiveState {
    job_id: Uuid,
    terminal_id: Uuid,
    phase: String,
    blocks_archived: usize,
    target_blocks: u64,
    mempool_snapshots_done: usize,
    mempool_snapshots_total: usize,
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

    /// SOL archive creator form state
    sol_extract_options: SolExtractOptions,

    /// SOL archive creator validation error
    sol_extract_error: Option<String>,

    /// Currently running SOL archive creation job
    running_sol_archive: Option<RunningSolArchiveState>,

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

    /// Highlighted event index (for hover highlighting in live view)
    highlighted_event: Option<usize>,

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
            sol_extract_options: SolExtractOptions::default(),
            sol_extract_error: None,
            running_sol_archive: None,
            results,
            selected_result: None,
            compare_baseline: None,
            compare_target: None,
            graph_config: GraphConfig::default(),
            highlighted_event: None,
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
                                        let error_msg =
                                            result.error.as_deref().unwrap_or("Unknown error");
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
                    RunnerMessage::SolArchiveStarted { job_id } => {
                        self.status = Some(format!("SOL archive job {} started", job_id));
                    }
                    RunnerMessage::SolArchiveLog {
                        job_id,
                        line,
                        is_error,
                    } => {
                        if let Some(ref running) = self.running_sol_archive {
                            if running.job_id == job_id {
                                if is_error {
                                    self.terminals.push_error(running.terminal_id, &line);
                                } else {
                                    self.terminals.push_line(running.terminal_id, &line);
                                }
                            }
                        }
                    }
                    RunnerMessage::SolArchiveProgress {
                        job_id,
                        phase,
                        blocks_archived,
                        target_blocks,
                        mempool_snapshots_done,
                        mempool_snapshots_total,
                    } => {
                        if let Some(ref mut running) = self.running_sol_archive {
                            if running.job_id == job_id {
                                running.phase = format!("{phase:?}");
                                running.blocks_archived = blocks_archived;
                                running.target_blocks = target_blocks;
                                running.mempool_snapshots_done = mempool_snapshots_done;
                                running.mempool_snapshots_total = mempool_snapshots_total;
                            }
                        }
                    }
                    RunnerMessage::SolArchiveCompleted {
                        job_id,
                        output_path,
                        blocks_archived,
                        txs_archived,
                        elapsed_secs,
                    } => {
                        if let Some(ref running) = self.running_sol_archive {
                            if running.job_id == job_id {
                                if let Some(term) = self.terminals.get_mut(running.terminal_id) {
                                    term.push_system(&format!(
                                        "Archive created: {} ({} blocks, {} txs, {:.1}s)",
                                        output_path.display(),
                                        blocks_archived,
                                        txs_archived,
                                        elapsed_secs
                                    ));
                                    term.mark_inactive();
                                }
                            }
                        }
                        self.status =
                            Some(format!("SOL archive created: {}", output_path.display()));
                        self.running_sol_archive = None;
                    }
                    RunnerMessage::SolArchiveFailed { job_id, error } => {
                        self.status = Some(format!("SOL archive creation failed: {}", error));
                        if let Some(ref running) = self.running_sol_archive {
                            if running.job_id == job_id {
                                if let Some(term) = self.terminals.get_mut(running.terminal_id) {
                                    term.push_error(&format!("Archive creation failed: {}", error));
                                    term.mark_inactive();
                                }
                            }
                        }
                        self.running_sol_archive = None;
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

    fn estimated_total_secs(config: &TestConfig) -> f64 {
        match config.benchmark_mode {
            BenchmarkMode::Container => config.duration_secs as f64,
            BenchmarkMode::SpeedOfLightBench => 0.0,
            BenchmarkMode::SpeedOfLightSweep => {
                let case_count = config.sol_sweep.case_count().unwrap_or(0) as f64;
                case_count * config.sol_sweep.repeats as f64 * config.sol_sweep.duration_secs as f64
            }
        }
    }

    fn running_progress_text(running: &RunningTestState) -> String {
        match running.config.benchmark_mode {
            BenchmarkMode::Container => {
                format!(
                    "{:.0}s / {:.0}s | {} samples",
                    running.elapsed_secs, running.total_secs, running.sample_count
                )
            }
            BenchmarkMode::SpeedOfLightSweep => {
                let total_runs = running
                    .config
                    .sol_sweep
                    .case_count()
                    .unwrap_or(0)
                    .saturating_mul(running.config.sol_sweep.repeats as usize);
                format!(
                    "{:.0}s / ~{:.0}s | {} / {} runs",
                    running.elapsed_secs, running.total_secs, running.sample_count, total_runs
                )
            }
            BenchmarkMode::SpeedOfLightBench => {
                format!(
                    "{:.0}s elapsed | {} timeline samples",
                    running.elapsed_secs,
                    running.samples.len().max(running.sample_count)
                )
            }
        }
    }

    fn running_sol_archive_progress_text(running: &RunningSolArchiveState) -> String {
        if running.phase == "MempoolReplay" {
            let total = running.mempool_snapshots_total.max(1);
            return format!(
                "{} | mempool {}/{}",
                running.phase, running.mempool_snapshots_done, total
            );
        }
        format!(
            "{} | blocks {}/{}",
            running.phase, running.blocks_archived, running.target_blocks
        )
    }

    /// Start a test
    fn start_test(&mut self, config: TestConfig) {
        if self.running_test.is_some() {
            self.status = Some("A test is already running".to_string());
            return;
        }
        if self.running_sol_archive.is_some() {
            self.status = Some("SOL archive creation is already running".to_string());
            return;
        }

        if config.benchmark_mode == BenchmarkMode::SpeedOfLightSweep {
            self.status = Some(
                "SOL sweep mode has been removed from the GUI. Use SOL bench or Container."
                    .to_string(),
            );
            return;
        }

        if let Err(error) = config.validate() {
            self.status = Some(format!("Invalid test configuration: {error}"));
            return;
        }

        let test_id = Uuid::new_v4();

        // Create terminal for this test
        let terminal_id = self.terminals.add_terminal(&config.name);
        self.terminals.push_system(terminal_id, "Starting test...");
        let total_secs = Self::estimated_total_secs(&config);

        // Create running state
        self.running_test = Some(RunningTestState {
            test_id,
            config: config.clone(),
            samples: Vec::new(),
            events: Vec::new(),
            terminal_id,
            elapsed_secs: 0.0,
            total_secs,
            sample_count: 0,
        });

        // Send to runner with our test_id so we can correlate messages
        if let Some(ref runner) = self.runner {
            let _ = runner.start_test(test_id, config);
        }
    }

    /// Stop the current test
    fn stop_test(&mut self) {
        if let Some(ref running) = self.running_test {
            if let Some(ref runner) = self.runner {
                let _ = runner.stop_test(running.test_id);
            }
            let message = match running.config.benchmark_mode {
                BenchmarkMode::Container => "Stopping test...",
                BenchmarkMode::SpeedOfLightBench => {
                    "Stop requested. SOL bench stops after current replay phase."
                }
                BenchmarkMode::SpeedOfLightSweep => {
                    "Stop requested. SOL sweep stops after current run."
                }
            };
            self.terminals.push_system(running.terminal_id, message);
        }
    }

    fn start_sol_archive_creation(&mut self) {
        if self.running_test.is_some() {
            self.status = Some(
                "A benchmark test is currently running. Wait for it to finish first.".to_string(),
            );
            return;
        }
        if self.running_sol_archive.is_some() {
            self.status = Some("SOL archive creation is already running".to_string());
            return;
        }

        if let Err(error) = self.sol_extract_options.validate() {
            self.sol_extract_error = Some(error.clone());
            self.status = Some(format!("Invalid SOL creator configuration: {}", error));
            return;
        }
        self.sol_extract_error = None;

        let job_id = Uuid::new_v4();
        let terminal_id = self.terminals.add_terminal("SOL Archive Creator");
        self.terminals
            .push_system(terminal_id, "Starting SOL archive creation...");

        self.running_sol_archive = Some(RunningSolArchiveState {
            job_id,
            terminal_id,
            phase: "Initializing".to_string(),
            blocks_archived: 0,
            target_blocks: self.sol_extract_options.block_count,
            mempool_snapshots_done: 0,
            mempool_snapshots_total: 0,
        });

        if let Some(ref runner) = self.runner {
            if let Err(error) = runner.create_sol_archive(job_id, self.sol_extract_options.clone())
            {
                self.status = Some(format!("Failed to start SOL archive creation: {}", error));
                if let Some(ref running) = self.running_sol_archive {
                    if running.job_id == job_id {
                        if let Some(term) = self.terminals.get_mut(running.terminal_id) {
                            term.push_error("Failed to queue SOL archive creation");
                            term.mark_inactive();
                        }
                    }
                }
                self.running_sol_archive = None;
            }
        } else {
            self.status = Some("Runner is not initialized".to_string());
            if let Some(ref running) = self.running_sol_archive {
                if running.job_id == job_id {
                    if let Some(term) = self.terminals.get_mut(running.terminal_id) {
                        term.push_error("Runner is not initialized");
                        term.mark_inactive();
                    }
                }
            }
            self.running_sol_archive = None;
        }
    }

    /// Show the navigation sidebar
    fn show_sidebar(&mut self, ui: &mut Ui) {
        ui.heading("Nockchain Bench");
        ui.separator();

        for view in [
            AppView::NewTest,
            AppView::SolTestCreator,
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
            let progress = if running.total_secs > 0.0 {
                (running.elapsed_secs / running.total_secs).clamp(0.0, 1.0) as f32
            } else {
                0.0
            };
            let mut progress_bar = egui::ProgressBar::new(progress);
            if running.total_secs <= 0.0 {
                progress_bar = progress_bar.animate(true).text("running");
            }
            ui.add(progress_bar);
            ui.label(Self::running_progress_text(running));
            if ui.button("Stop").clicked() {
                self.stop_test();
            }
        }

        if let Some(ref running) = self.running_sol_archive {
            ui.separator();
            ui.label(RichText::new("Creating SOL Archive").strong());
            let progress = (running.blocks_archived as f64 / running.target_blocks.max(1) as f64)
                .clamp(0.0, 1.0) as f32;
            ui.add(egui::ProgressBar::new(progress));
            ui.label(Self::running_sol_archive_progress_text(running));
        }
    }

    /// Show the main content area
    fn show_content(&mut self, ui: &mut Ui) {
        match self.view {
            AppView::NewTest => self.show_new_test(ui),
            AppView::SolTestCreator => self.show_sol_test_creator(ui),
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

    fn show_sol_test_creator(&mut self, ui: &mut Ui) {
        ui.heading("SOL Test Creator");
        ui.label("Create `.solarch` archives from checkpoint + kernel inputs.");
        ui.separator();

        egui::Grid::new("sol_extract_options")
            .num_columns(2)
            .spacing([20.0, 8.0])
            .show(ui, |ui| {
                ui.label("Blocks:");
                ui.add(
                    egui::DragValue::new(&mut self.sol_extract_options.block_count)
                        .range(1..=u64::MAX),
                );
                ui.end_row();

                ui.label("Checkpoint:");
                ui.text_edit_singleline(&mut self.sol_extract_options.checkpoint_path);
                ui.end_row();

                ui.label("Kernel:");
                ui.text_edit_singleline(&mut self.sol_extract_options.kernel_path);
                ui.end_row();

                ui.label("Output Archive:");
                let mut output = self
                    .sol_extract_options
                    .output_archive
                    .clone()
                    .unwrap_or_default();
                if ui.text_edit_singleline(&mut output).changed() {
                    self.sol_extract_options.output_archive = if output.trim().is_empty() {
                        None
                    } else {
                        Some(output)
                    };
                }
                ui.end_row();

                ui.label("Blocks per fetch:");
                ui.add(
                    egui::DragValue::new(&mut self.sol_extract_options.chunk_size)
                        .range(1..=u64::MAX),
                );
                ui.end_row();

                ui.label("Include Mempool:");
                ui.checkbox(&mut self.sol_extract_options.include_mempool, "");
                ui.end_row();

                ui.label("Work Dir:");
                ui.text_edit_singleline(&mut self.sol_extract_options.work_dir);
                ui.end_row();
            });

        ui.label(format!(
            "Effective output: {}",
            self.sol_extract_options.effective_output_archive()
        ));

        if let Some(ref error) = self.sol_extract_error {
            ui.colored_label(egui::Color32::RED, error);
        }

        ui.separator();
        let creator_running = self.running_sol_archive.is_some();
        if ui
            .add_enabled(
                !creator_running,
                egui::Button::new(RichText::new("Create SOL Archive").strong()),
            )
            .clicked()
        {
            self.start_sol_archive_creation();
        }
        if ui.button("Reset Defaults").clicked() {
            self.sol_extract_options = SolExtractOptions::default();
            self.sol_extract_error = None;
        }

        if let Some(ref running) = self.running_sol_archive {
            ui.separator();
            ui.label(RichText::new("Archive Creation Progress").strong());
            let progress = (running.blocks_archived as f64 / running.target_blocks.max(1) as f64)
                .clamp(0.0, 1.0) as f32;
            ui.add(egui::ProgressBar::new(progress));
            ui.label(Self::running_sol_archive_progress_text(running));
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
            .id_salt("results_list_scroll")
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

            // Graph - use the metrics that were actually collected in this test
            let result_graph_config = GraphConfig {
                metrics: result.config.metrics.clone(),
                containers: result.config.containers.iter().map(|c| c.id).collect(),
                height: 300.0,
                ..self.graph_config.clone()
            };
            render_graph(
                ui, result, &result_graph_config, &mut self.highlighted_event,
            );

            if let Some(sol_bench) = &result.sol_bench {
                ui.separator();
                self.show_sol_bench_summary(ui, sol_bench);
            }

            if let Some(sol_sweep) = &result.sol_sweep {
                ui.separator();
                self.show_sol_sweep_summary(ui, sol_sweep);
            }

            // Statistics
            ui.separator();
            ui.heading("Statistics");

            let mut stat_ids: Vec<_> = result.statistics.keys().copied().collect();
            stat_ids.sort_by_key(|id| id.to_string());
            for container_id in stat_ids {
                if let Some(stats) = result.statistics.get(&container_id) {
                    let container_name = result
                        .config
                        .containers
                        .iter()
                        .find(|container| container.id == container_id)
                        .map(|container| container.name.clone())
                        .unwrap_or_else(|| format!("Container {}", &container_id.to_string()[..8]));

                    ui.collapsing(container_name, |ui| {
                        egui::Grid::new(format!("stats_{}", container_id))
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

    fn show_sol_bench_summary(&self, ui: &mut Ui, bench: &SolBenchResult) {
        ui.heading("SOL Bench Summary");
        egui::Grid::new("sol_bench_summary_grid")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("Blocks poked:");
                ui.label(bench.blocks_poked.to_string());
                ui.end_row();

                ui.label("Failed pokes:");
                ui.label(bench.failed_pokes.to_string());
                ui.end_row();

                ui.label("Init time:");
                ui.label(format!("{:.2}s", bench.init_time_secs));
                ui.end_row();

                ui.label("Replay time:");
                ui.label(format!("{:.2}s", bench.total_poke_time_secs));
                ui.end_row();

                ui.label("Throughput:");
                ui.label(format!("{:.2} blocks/s", bench.blocks_per_second));
                ui.end_row();

                ui.label("Checkpoints:");
                ui.label(bench.checkpoint_count.to_string());
                ui.end_row();

                ui.label("Checkpoint total:");
                ui.label(format!("{:.2}s", bench.checkpoint_total_time_secs));
                ui.end_row();

                ui.label("Checkpoint avg:");
                ui.label(
                    bench
                        .checkpoint_avg_time_secs
                        .map(|value| format!("{:.2}s", value))
                        .unwrap_or_else(|| "n/a".to_string()),
                );
                ui.end_row();
            });

        if let Some(profile) = &bench.memory_profile {
            ui.separator();
            ui.label(RichText::new("Memory Scorecard").strong());
            egui::Grid::new("sol_bench_scorecard")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("Peak RSS:");
                    ui.label(format!("{:.2} MiB", profile.scorecard.peak_rss_mib));
                    ui.end_row();

                    ui.label("P95 RSS:");
                    ui.label(format!("{:.2} MiB", profile.scorecard.p95_rss_mib));
                    ui.end_row();

                    ui.label("Checkpoint peak RSS:");
                    ui.label(
                        profile
                            .scorecard
                            .checkpoint_peak_rss_mib
                            .map(|value| format!("{:.2} MiB", value))
                            .unwrap_or_else(|| "n/a".to_string()),
                    );
                    ui.end_row();

                    ui.label("Checkpoint sec/GiB:");
                    ui.label(
                        profile
                            .scorecard
                            .checkpoint_seconds_per_gib
                            .map(|value| format!("{:.3}", value))
                            .unwrap_or_else(|| "n/a".to_string()),
                    );
                    ui.end_row();

                    ui.label("GC pause p95:");
                    ui.label(
                        profile
                            .scorecard
                            .gc_pause_p95_ms
                            .map(|value| format!("{:.1} ms", value))
                            .unwrap_or_else(|| "n/a".to_string()),
                    );
                    ui.end_row();

                    ui.label("GC / 1k blocks:");
                    ui.label(format!("{:.2}", profile.scorecard.gc_events_per_1k_blocks));
                    ui.end_row();

                    ui.label("Page-fault bursts:");
                    ui.label(profile.scorecard.page_fault_burst_count.to_string());
                    ui.end_row();
                });

            ui.collapsing("Phase Summaries", |ui| {
                egui::Grid::new("sol_phase_summaries")
                    .num_columns(5)
                    .show(ui, |ui| {
                        ui.label("Phase");
                        ui.label("Duration");
                        ui.label("Samples");
                        ui.label("Peak RSS");
                        ui.label("Minor faults Δ");
                        ui.end_row();

                        for phase in &profile.phase_summaries {
                            ui.label(format!("{:?}", phase.kind));
                            ui.label(format!("{}ms", phase.duration_ms));
                            ui.label(phase.sample_count.to_string());
                            ui.label(format!(
                                "{:.2} MiB",
                                phase.peak_rss_bytes as f64 / 1024.0 / 1024.0
                            ));
                            ui.label(phase.minor_faults_delta.to_string());
                            ui.end_row();
                        }
                    });
            });

            ui.collapsing("Checkpoint Profiles", |ui| {
                egui::Grid::new("sol_checkpoint_profiles")
                    .num_columns(5)
                    .show(ui, |ui| {
                        ui.label("#");
                        ui.label("Duration");
                        ui.label("Peak RSS");
                        ui.label("Recovery");
                        ui.label("Throughput");
                        ui.end_row();

                        for (idx, checkpoint) in profile.checkpoint_profiles.iter().enumerate() {
                            ui.label((idx + 1).to_string());
                            ui.label(format!("{}ms", checkpoint.duration_ms));
                            ui.label(format!(
                                "{:.2} MiB",
                                checkpoint.peak_rss_bytes as f64 / 1024.0 / 1024.0
                            ));
                            ui.label(
                                checkpoint
                                    .recovery_ms
                                    .map(|value| format!("{}ms", value))
                                    .unwrap_or_else(|| "n/a".to_string()),
                            );
                            ui.label(
                                checkpoint
                                    .throughput_mib_per_s()
                                    .map(|value| format!("{:.2} MiB/s", value))
                                    .unwrap_or_else(|| "n/a".to_string()),
                            );
                            ui.end_row();
                        }
                    });
            });
        }
    }

    fn show_sol_sweep_summary(&self, ui: &mut Ui, sweep: &SolSweepResult) {
        ui.heading("SOL Sweep Summary");
        ui.label(format!("Runs: {}", sweep.runs.len()));
        ui.label(format!("Cases: {}", sweep.summaries.len()));

        ui.separator();
        egui::Grid::new("sol_sweep_summary_grid")
            .num_columns(6)
            .show(ui, |ui| {
                ui.label("Candidate");
                ui.label("Chunk");
                ui.label("Memory");
                ui.label("Peak RSS mean");
                ui.label("Ckpt MiB/s mean");
                ui.label("Fault bursts mean");
                ui.end_row();

                for summary in &sweep.summaries {
                    ui.label(&summary.case.candidate);
                    ui.label(summary.case.chunk_size.to_string());
                    ui.label(&summary.case.memory_limit);
                    ui.label(format!("{:.2}", summary.peak_rss_mib_mean));
                    ui.label(
                        summary
                            .checkpoint_mib_per_s_mean
                            .map(|value| format!("{:.2}", value))
                            .unwrap_or_else(|| "n/a".to_string()),
                    );
                    ui.label(
                        summary
                            .page_fault_bursts_mean
                            .map(|value| format!("{:.2}", value))
                            .unwrap_or_else(|| "n/a".to_string()),
                    );
                    ui.end_row();
                }
            });
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
                .selected_text(
                    self.graph_config
                        .metrics
                        .first()
                        .map(|m| m.label())
                        .unwrap_or("VmRss"),
                )
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
                self.graph_config
                    .metrics
                    .first()
                    .copied()
                    .unwrap_or(MetricType::VmRss),
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
            &mut self.graph_config.significant_events_only, "Significant events only",
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
        // Clone data we need from running_test to avoid borrow conflicts
        let data = self.running_test.as_ref().map(|running| {
            (
                running
                    .config
                    .containers
                    .iter()
                    .map(|c| (c.id, c.name.clone()))
                    .collect::<Vec<_>>(),
                running.samples.clone(),
                running.events.clone(),
                running.config.metrics.clone(),
                running.config.benchmark_mode,
            )
        });

        if let Some((containers, samples, events, metrics, mode)) = data {
            ui.separator();
            ui.heading("Live View");

            if mode == BenchmarkMode::SpeedOfLightSweep && samples.is_empty() {
                ui.label(
                    "SOL sweep does not stream per-second samples. Use progress and terminal logs.",
                );
                return;
            }
            if mode == BenchmarkMode::SpeedOfLightBench && samples.is_empty() {
                ui.label("SOL bench memory timeline appears when replay completes.");
            }

            render_live_graph_with_events_panel(
                ui, &samples, &events, &containers, &metrics, &self.graph_config,
                &mut self.highlighted_event,
            );
        }
    }
}

impl eframe::App for BenchApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        // Process runner messages
        self.process_runner_messages();

        // Request continuous repaint while long-running jobs are active
        if self.running_test.is_some() || self.running_sol_archive.is_some() {
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

        // Status bar - scrollable for long error messages
        TopBottomPanel::bottom("status")
            .resizable(true)
            .default_height(60.0)
            .min_height(40.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("status_bar_scroll")
                    .max_height(ui.available_height())
                    .show(ui, |ui| {
                        if let Some(ref status) = self.status {
                            ui.add(egui::Label::new(status).wrap());
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
    use crate::config::BenchmarkMode;

    #[test]
    fn test_app_view_labels() {
        assert_eq!(AppView::NewTest.label(), "New Test");
        assert_eq!(AppView::SolTestCreator.label(), "SOL Test Creator");
        assert_eq!(AppView::Results.label(), "Results");
    }

    #[test]
    fn test_app_default() {
        let app = BenchApp::new();
        assert_eq!(app.view, AppView::NewTest);
        assert!(app.running_test.is_none());
        assert!(app.running_sol_archive.is_none());
    }

    #[test]
    fn test_estimated_total_secs_for_modes() {
        let container = TestConfig::default();
        assert_eq!(BenchApp::estimated_total_secs(&container), 300.0);

        let mut bench = TestConfig::default();
        bench.benchmark_mode = BenchmarkMode::SpeedOfLightBench;
        assert_eq!(BenchApp::estimated_total_secs(&bench), 0.0);

        let mut sweep = TestConfig::default();
        sweep.benchmark_mode = BenchmarkMode::SpeedOfLightSweep;
        sweep.sol_sweep.candidates_csv = "a,b".to_string();
        sweep.sol_sweep.chunk_sizes_csv = "8,16".to_string();
        sweep.sol_sweep.memory_limits_csv = "8g".to_string();
        sweep.sol_sweep.repeats = 2;
        sweep.sol_sweep.duration_secs = 60;
        assert_eq!(BenchApp::estimated_total_secs(&sweep), 480.0);
    }

    #[test]
    fn test_running_progress_text_modes() {
        let mut base = TestConfig::default();
        base.benchmark_mode = BenchmarkMode::Container;
        let running = RunningTestState {
            test_id: Uuid::new_v4(),
            config: base.clone(),
            samples: vec![],
            events: vec![],
            terminal_id: Uuid::new_v4(),
            elapsed_secs: 12.0,
            total_secs: 300.0,
            sample_count: 7,
        };
        assert!(BenchApp::running_progress_text(&running).contains("samples"));

        let mut sweep = base.clone();
        sweep.benchmark_mode = BenchmarkMode::SpeedOfLightSweep;
        sweep.sol_sweep.candidates_csv = "a".to_string();
        sweep.sol_sweep.chunk_sizes_csv = "8".to_string();
        sweep.sol_sweep.memory_limits_csv = "8g".to_string();
        sweep.sol_sweep.repeats = 3;
        let running = RunningTestState {
            test_id: Uuid::new_v4(),
            config: sweep,
            samples: vec![],
            events: vec![],
            terminal_id: Uuid::new_v4(),
            elapsed_secs: 20.0,
            total_secs: 180.0,
            sample_count: 1,
        };
        assert!(BenchApp::running_progress_text(&running).contains("runs"));
    }

    #[test]
    fn test_start_test_rejects_sol_sweep_mode() {
        let mut app = BenchApp::new();
        let mut config = TestConfig::default();
        config.benchmark_mode = BenchmarkMode::SpeedOfLightSweep;

        app.start_test(config);

        assert!(app.running_test.is_none());
        assert!(app
            .status
            .as_deref()
            .is_some_and(|status| status.contains("SOL sweep mode has been removed")));
    }

    #[test]
    fn test_start_sol_archive_creation_validation_error() {
        let mut app = BenchApp::new();
        app.sol_extract_options.block_count = 0;

        app.start_sol_archive_creation();

        assert!(app.running_sol_archive.is_none());
        assert!(app.sol_extract_error.is_some());
        assert!(app
            .status
            .as_deref()
            .is_some_and(|status| status.contains("Invalid SOL creator configuration")));
    }

    #[test]
    fn test_start_test_rejects_when_sol_archive_running() {
        let mut app = BenchApp::new();
        app.running_sol_archive = Some(RunningSolArchiveState {
            job_id: Uuid::new_v4(),
            terminal_id: Uuid::new_v4(),
            phase: "Blocks".to_string(),
            blocks_archived: 10,
            target_blocks: 100,
            mempool_snapshots_done: 0,
            mempool_snapshots_total: 0,
        });

        app.start_test(TestConfig::default());

        assert!(app
            .status
            .as_deref()
            .is_some_and(|status| status.contains("SOL archive creation is already running")));
    }
}
