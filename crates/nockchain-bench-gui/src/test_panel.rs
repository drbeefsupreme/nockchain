//! Test creation and configuration panel
//!
//! Provides UI for creating and configuring benchmark tests.

use egui::{CollapsingHeader, Grid, RichText, Ui};
use uuid::Uuid;

use crate::config::{MetricType, TestConfig};
use crate::docker_panel::ContainerListPanel;

/// Panel for creating and editing a test configuration
pub struct TestPanel {
    /// The test configuration being edited
    pub config: TestConfig,

    /// Container list panel
    pub containers: ContainerListPanel,

    /// Validation error message
    pub error: Option<String>,

    /// Selected metrics (as a set for easy toggling)
    selected_metrics: std::collections::HashSet<MetricType>,

    /// New tag being entered
    new_tag: String,
}

impl Default for TestPanel {
    fn default() -> Self {
        Self::new(TestConfig::default())
    }
}

impl TestPanel {
    /// Create a new panel for editing a test config
    pub fn new(config: TestConfig) -> Self {
        let selected_metrics: std::collections::HashSet<MetricType> =
            config.metrics.iter().copied().collect();

        let mut containers = ContainerListPanel::new();
        for container in &config.containers {
            containers.add(container.clone());
        }

        Self {
            config,
            containers,
            error: None,
            selected_metrics,
            new_tag: String::new(),
        }
    }

    /// Create a panel for a new test
    pub fn new_test(name: impl Into<String>) -> Self {
        Self::new(TestConfig::new(name))
    }

    /// Get the test ID
    pub fn id(&self) -> Uuid {
        self.config.id
    }

    /// Validate the current configuration
    pub fn validate(&mut self) -> bool {
        // Update config with current state
        self.sync_config();

        match self.config.validate() {
            Ok(()) => {
                self.error = None;
                true
            }
            Err(msg) => {
                self.error = Some(msg);
                false
            }
        }
    }

    /// Sync the config with the current panel state
    fn sync_config(&mut self) {
        self.config.containers = self.containers.configs();
        self.config.metrics = self.selected_metrics.iter().copied().collect();
    }

    /// Take the configuration (consumes the panel)
    pub fn take_config(mut self) -> TestConfig {
        self.sync_config();
        self.config
    }

    /// Get the current config (cloned)
    pub fn get_config(&mut self) -> TestConfig {
        self.sync_config();
        self.config.clone()
    }

    /// Render the panel
    pub fn show(&mut self, ui: &mut Ui) -> TestPanelResponse {
        let mut response = TestPanelResponse::default();

        // Basic info
        Grid::new("test_panel_basic")
            .num_columns(2)
            .spacing([20.0, 8.0])
            .show(ui, |ui| {
                ui.label("Test Name:");
                ui.text_edit_singleline(&mut self.config.name);
                ui.end_row();

                ui.label("Description:");
                let mut desc = self.config.description.clone().unwrap_or_default();
                if ui.text_edit_multiline(&mut desc).changed() {
                    self.config.description = if desc.is_empty() { None } else { Some(desc) };
                }
                ui.end_row();

                ui.label("Duration:");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.config.duration_secs)
                            .range(1..=86400)
                            .suffix(" secs"),
                    );
                    // Quick buttons
                    if ui.small_button("1m").clicked() {
                        self.config.duration_secs = 60;
                    }
                    if ui.small_button("5m").clicked() {
                        self.config.duration_secs = 300;
                    }
                    if ui.small_button("15m").clicked() {
                        self.config.duration_secs = 900;
                    }
                    if ui.small_button("1h").clicked() {
                        self.config.duration_secs = 3600;
                    }
                });
                ui.end_row();

                ui.label("Sample Interval:");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.config.sample_interval_ms)
                            .range(100..=60000)
                            .suffix(" ms"),
                    );
                    // Quick buttons
                    if ui.small_button("100ms").clicked() {
                        self.config.sample_interval_ms = 100;
                    }
                    if ui.small_button("500ms").clicked() {
                        self.config.sample_interval_ms = 500;
                    }
                    if ui.small_button("1s").clicked() {
                        self.config.sample_interval_ms = 1000;
                    }
                    if ui.small_button("5s").clicked() {
                        self.config.sample_interval_ms = 5000;
                    }
                });
                ui.end_row();
            });

        ui.separator();

        // Containers section
        CollapsingHeader::new(RichText::new("Containers").strong())
            .default_open(true)
            .show(ui, |ui| {
                self.containers.show(ui);
            });

        ui.separator();

        // Metrics section
        CollapsingHeader::new(RichText::new("Metrics to Track").strong())
            .default_open(true)
            .show(ui, |ui| {
                self.show_metrics(ui);
            });

        ui.separator();

        // Tags section
        CollapsingHeader::new("Tags")
            .default_open(false)
            .show(ui, |ui| {
                self.show_tags(ui);
            });

        // Error display
        if let Some(ref error) = self.error {
            ui.separator();
            ui.colored_label(egui::Color32::RED, error);
        }

        // Action buttons
        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .button(RichText::new("Run Test").strong())
                .clicked()
            {
                if self.validate() {
                    response.run_requested = true;
                }
            }
            if ui.button("Validate").clicked() {
                self.validate();
                response.validated = true;
            }
            if ui.button("Save Template").clicked() {
                if self.validate() {
                    response.save_requested = true;
                }
            }
            if ui.button("Reset").clicked() {
                *self = TestPanel::default();
                response.reset = true;
            }
        });

        response
    }

    /// Show metrics selection
    fn show_metrics(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            if ui.button("Select All").clicked() {
                for metric in MetricType::all() {
                    self.selected_metrics.insert(*metric);
                }
            }
            if ui.button("Select None").clicked() {
                self.selected_metrics.clear();
            }
            if ui.button("Defaults").clicked() {
                self.selected_metrics = MetricType::defaults().into_iter().collect();
            }
        });

        ui.separator();

        // Memory metrics
        ui.label(RichText::new("Memory Metrics:").strong());
        ui.horizontal_wrapped(|ui| {
            for metric in MetricType::all() {
                if metric.is_memory() {
                    let selected = self.selected_metrics.contains(metric);
                    if ui.checkbox(&mut selected.clone(), metric.label()).clicked() {
                        if selected {
                            self.selected_metrics.remove(metric);
                        } else {
                            self.selected_metrics.insert(*metric);
                        }
                    }
                    if ui.small_button("?").on_hover_text(metric.description()).clicked() {}
                }
            }
        });

        // Other metrics
        ui.label(RichText::new("Other Metrics:").strong());
        ui.horizontal_wrapped(|ui| {
            for metric in MetricType::all() {
                if !metric.is_memory() {
                    let selected = self.selected_metrics.contains(metric);
                    if ui.checkbox(&mut selected.clone(), metric.label()).clicked() {
                        if selected {
                            self.selected_metrics.remove(metric);
                        } else {
                            self.selected_metrics.insert(*metric);
                        }
                    }
                    if ui.small_button("?").on_hover_text(metric.description()).clicked() {}
                }
            }
        });

        ui.label(format!("{} metrics selected", self.selected_metrics.len()));
    }

    /// Show tags editor
    fn show_tags(&mut self, ui: &mut Ui) {
        let mut to_remove = None;

        ui.horizontal_wrapped(|ui| {
            for (i, tag) in self.config.tags.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("#{}", tag));
                    if ui.small_button("x").clicked() {
                        to_remove = Some(i);
                    }
                });
            }
        });

        if let Some(i) = to_remove {
            self.config.tags.remove(i);
        }

        ui.horizontal(|ui| {
            ui.text_edit_singleline(&mut self.new_tag);
            if ui.button("Add Tag").clicked() && !self.new_tag.is_empty() {
                self.config.tags.push(std::mem::take(&mut self.new_tag));
            }
        });
    }
}

/// Response from the test panel
#[derive(Default)]
pub struct TestPanelResponse {
    /// Whether validation was requested
    pub validated: bool,

    /// Whether run was requested
    pub run_requested: bool,

    /// Whether save was requested
    pub save_requested: bool,

    /// Whether reset was performed
    pub reset: bool,
}

/// Panel for listing saved tests
pub struct TestListPanel {
    /// Saved test configs
    configs: Vec<TestConfig>,

    /// Currently selected test index
    selected: Option<usize>,

    /// Search filter
    search: String,
}

impl Default for TestListPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl TestListPanel {
    /// Create a new test list panel
    pub fn new() -> Self {
        Self {
            configs: Vec::new(),
            selected: None,
            search: String::new(),
        }
    }

    /// Set the list of configs
    pub fn set_configs(&mut self, configs: Vec<TestConfig>) {
        self.configs = configs;
        self.selected = None;
    }

    /// Get the selected config
    pub fn selected(&self) -> Option<&TestConfig> {
        self.selected.and_then(|i| self.configs.get(i))
    }

    /// Render the panel
    pub fn show(&mut self, ui: &mut Ui) -> TestListResponse {
        let mut response = TestListResponse::default();

        // Search
        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.search);
            if ui.button("Clear").clicked() {
                self.search.clear();
            }
        });

        ui.separator();

        // Filter configs
        let filtered: Vec<(usize, &TestConfig)> = self
            .configs
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                self.search.is_empty()
                    || c.name.to_lowercase().contains(&self.search.to_lowercase())
                    || c.tags.iter().any(|t| t.to_lowercase().contains(&self.search.to_lowercase()))
            })
            .collect();

        if filtered.is_empty() {
            ui.label("No tests found");
            return response;
        }

        // List
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (idx, config) in filtered {
                let is_selected = self.selected == Some(idx);
                let text = format!(
                    "{}\n{} container(s) | {} metric(s)",
                    config.name,
                    config.containers.len(),
                    config.metrics.len()
                );

                if ui.selectable_label(is_selected, text).clicked() {
                    self.selected = Some(idx);
                    response.selected = Some(config.id);
                }
            }
        });

        // Actions for selected
        if let Some(idx) = self.selected {
            if let Some(config) = self.configs.get(idx) {
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Load").clicked() {
                        response.load_requested = Some(config.id);
                    }
                    if ui.button("Run").clicked() {
                        response.run_requested = Some(config.id);
                    }
                    if ui
                        .button(RichText::new("Delete").color(egui::Color32::RED))
                        .clicked()
                    {
                        response.delete_requested = Some(config.id);
                    }
                });
            }
        }

        response
    }
}

/// Response from the test list panel
#[derive(Default)]
pub struct TestListResponse {
    /// Test that was selected
    pub selected: Option<Uuid>,

    /// Test that was requested to load
    pub load_requested: Option<Uuid>,

    /// Test that was requested to run
    pub run_requested: Option<Uuid>,

    /// Test that was requested to delete
    pub delete_requested: Option<Uuid>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_panel_new() {
        let panel = TestPanel::default();
        assert!(!panel.selected_metrics.is_empty());
        assert!(panel.error.is_none());
    }

    #[test]
    fn test_test_panel_validate_empty() {
        let mut panel = TestPanel::default();
        // No containers - should fail
        assert!(!panel.validate());
        assert!(panel.error.is_some());
    }

    #[test]
    fn test_test_panel_validate_with_container() {
        let mut panel = TestPanel::default();
        panel.containers.add_default("Container 1");
        assert!(panel.validate());
        assert!(panel.error.is_none());
    }

    #[test]
    fn test_test_panel_get_config() {
        let mut panel = TestPanel::new_test("My Test");
        panel.containers.add_default("Container 1");
        panel.selected_metrics.insert(MetricType::PmaRss);

        let config = panel.get_config();
        assert_eq!(config.name, "My Test");
        assert_eq!(config.containers.len(), 1);
        assert!(config.metrics.contains(&MetricType::PmaRss));
    }

    #[test]
    fn test_test_list_panel() {
        let mut panel = TestListPanel::new();

        let configs = vec![
            TestConfig::new("Test 1"),
            TestConfig::new("Test 2"),
            TestConfig::new("Other Test"),
        ];

        panel.set_configs(configs);
        assert_eq!(panel.configs.len(), 3);
        assert!(panel.selected().is_none());
    }

    #[test]
    fn test_test_list_search() {
        let panel = TestListPanel {
            configs: vec![
                TestConfig::new("Test 1"),
                TestConfig::new("Other"),
            ],
            selected: None,
            search: "Test".to_string(),
        };

        // The filtering happens in show(), but we can test the logic
        let matches: Vec<&TestConfig> = panel
            .configs
            .iter()
            .filter(|c| c.name.to_lowercase().contains(&panel.search.to_lowercase()))
            .collect();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "Test 1");
    }
}
