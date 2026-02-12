//! Docker container configuration panel
//!
//! Provides UI for configuring Nockchain Docker containers with various options.

use egui::{CollapsingHeader, Grid, RichText, Ui};
use uuid::Uuid;

use crate::config::{ContainerConfig, PersistenceMode};

/// Panel for editing Docker container configuration
pub struct DockerPanel {
    /// The configuration being edited
    pub config: ContainerConfig,

    /// Validation error message (if any)
    pub error: Option<String>,

    /// Whether the panel is in edit mode
    pub editing: bool,

    /// New environment variable key being entered
    new_env_key: String,

    /// New environment variable value being entered
    new_env_value: String,

    /// New extra arg being entered
    new_extra_arg: String,
}

impl Default for DockerPanel {
    fn default() -> Self {
        Self::new(ContainerConfig::default())
    }
}

impl DockerPanel {
    /// Create a new panel for editing a container config
    pub fn new(config: ContainerConfig) -> Self {
        Self {
            config,
            error: None,
            editing: true,
            new_env_key: String::new(),
            new_env_value: String::new(),
            new_extra_arg: String::new(),
        }
    }

    /// Create a panel for a new container
    pub fn new_container(name: impl Into<String>) -> Self {
        Self::new(ContainerConfig::new(name))
    }

    /// Get the container ID
    pub fn id(&self) -> Uuid {
        self.config.id
    }

    /// Validate the current configuration
    pub fn validate(&mut self) -> bool {
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

    /// Take the configuration (consumes the panel)
    pub fn take_config(self) -> ContainerConfig {
        self.config
    }

    /// Render the panel
    pub fn show(&mut self, ui: &mut Ui) -> DockerPanelResponse {
        let mut response = DockerPanelResponse::default();

        if !self.editing {
            // Read-only summary view
            self.show_summary(ui, &mut response);
            return response;
        }

        // Editable form
        Grid::new(format!("docker_panel_{}", self.config.id))
            .num_columns(2)
            .spacing([20.0, 8.0])
            .show(ui, |ui| {
                // Name
                ui.label("Name:");
                ui.text_edit_singleline(&mut self.config.name);
                ui.end_row();

                // Docker Image
                ui.label("Docker Image:");
                ui.text_edit_singleline(&mut self.config.image);
                ui.end_row();

                // Git Reference
                ui.label("Git Ref (optional):");
                let mut git_ref = self.config.git_ref.clone().unwrap_or_default();
                if ui.text_edit_singleline(&mut git_ref).changed() {
                    self.config.git_ref = if git_ref.is_empty() {
                        None
                    } else {
                        Some(git_ref)
                    };
                }
                ui.end_row();

                // Persistence Mode
                ui.label("Persistence Mode:");
                ui.horizontal(|ui| {
                    for mode in PersistenceMode::all() {
                        if ui
                            .selectable_label(self.config.persistence_mode == *mode, mode.label())
                            .clicked()
                        {
                            self.config.persistence_mode = *mode;
                        }
                    }
                });
                ui.end_row();

                // Checkpoint Interval (only if checkpoint mode)
                if self.config.persistence_mode == PersistenceMode::Checkpoint {
                    ui.label("Checkpoint Interval:");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.config.checkpoint_interval_secs)
                                .range(10..=3600)
                                .suffix(" secs"),
                        );
                    });
                    ui.end_row();
                }

                // Memory Limit
                ui.label("Memory Limit:");
                ui.text_edit_singleline(&mut self.config.memory_limit);
                ui.end_row();

                // Number of Threads
                ui.label("Mining Threads:");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.config.num_threads)
                            .range(0..=64)
                            .suffix(" (0=auto)"),
                    );
                });
                ui.end_row();

                // Checkboxes
                ui.label("Options:");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.config.enable_mining, "Mining");
                    ui.checkbox(&mut self.config.use_fakenet, "Fakenet");
                    ui.checkbox(&mut self.config.enable_fast_sync, "Fast Sync");
                });
                ui.end_row();
            });

        // Environment Variables
        CollapsingHeader::new("Environment Variables")
            .default_open(false)
            .show(ui, |ui| {
                self.show_env_vars(ui);
            });

        // Extra Arguments
        CollapsingHeader::new("Extra CLI Arguments")
            .default_open(false)
            .show(ui, |ui| {
                self.show_extra_args(ui);
            });

        // Error display
        if let Some(ref error) = self.error {
            ui.separator();
            ui.colored_label(egui::Color32::RED, error);
        }

        // Buttons
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Validate").clicked() {
                self.validate();
                response.validated = true;
            }
            if ui.button("Reset").clicked() {
                self.config = ContainerConfig::default();
                self.error = None;
                response.reset = true;
            }
        });

        response
    }

    /// Show a read-only summary view
    fn show_summary(&self, ui: &mut Ui, response: &mut DockerPanelResponse) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(&self.config.name).strong());
            ui.label("|");
            ui.label(self.config.persistence_mode.label());
            ui.label("|");
            ui.label(&self.config.memory_limit);
            if ui.small_button("Edit").clicked() {
                response.edit_requested = true;
            }
        });
    }

    /// Show environment variables editor
    fn show_env_vars(&mut self, ui: &mut Ui) {
        let mut to_remove = None;

        for (i, (key, value)) in self.config.env_vars.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!("{}={}", key, value));
                if ui.small_button("x").clicked() {
                    to_remove = Some(i);
                }
            });
        }

        if let Some(i) = to_remove {
            self.config.env_vars.remove(i);
        }

        ui.horizontal(|ui| {
            ui.text_edit_singleline(&mut self.new_env_key);
            ui.label("=");
            ui.text_edit_singleline(&mut self.new_env_value);
            if ui.button("Add").clicked() && !self.new_env_key.is_empty() {
                self.config.env_vars.push((
                    std::mem::take(&mut self.new_env_key),
                    std::mem::take(&mut self.new_env_value),
                ));
            }
        });
    }

    /// Show extra arguments editor
    fn show_extra_args(&mut self, ui: &mut Ui) {
        let mut to_remove = None;

        for (i, arg) in self.config.extra_args.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(arg);
                if ui.small_button("x").clicked() {
                    to_remove = Some(i);
                }
            });
        }

        if let Some(i) = to_remove {
            self.config.extra_args.remove(i);
        }

        ui.horizontal(|ui| {
            ui.text_edit_singleline(&mut self.new_extra_arg);
            if ui.button("Add").clicked() && !self.new_extra_arg.is_empty() {
                self.config
                    .extra_args
                    .push(std::mem::take(&mut self.new_extra_arg));
            }
        });
    }
}

/// Response from the Docker panel
#[derive(Default)]
pub struct DockerPanelResponse {
    /// Whether the panel was validated
    pub validated: bool,

    /// Whether the panel was reset
    pub reset: bool,

    /// Whether edit mode was requested
    pub edit_requested: bool,
}

/// Panel for listing and managing multiple containers
pub struct ContainerListPanel {
    /// Container panels
    panels: Vec<DockerPanel>,

    /// Currently selected container index
    selected: Option<usize>,
}

impl Default for ContainerListPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl ContainerListPanel {
    /// Create a new empty container list
    pub fn new() -> Self {
        Self {
            panels: Vec::new(),
            selected: None,
        }
    }

    /// Add a new container
    pub fn add(&mut self, config: ContainerConfig) -> Uuid {
        let id = config.id;
        self.panels.push(DockerPanel::new(config));
        self.selected = Some(self.panels.len() - 1);
        id
    }

    /// Add a new default container
    pub fn add_default(&mut self, name: impl Into<String>) -> Uuid {
        let config = ContainerConfig::new(name);
        self.add(config)
    }

    /// Remove a container by index
    pub fn remove(&mut self, index: usize) {
        if index < self.panels.len() {
            self.panels.remove(index);
            if self.panels.is_empty() {
                self.selected = None;
            } else if let Some(sel) = self.selected {
                if sel >= self.panels.len() {
                    self.selected = Some(self.panels.len() - 1);
                }
            }
        }
    }

    /// Remove a container by ID
    pub fn remove_by_id(&mut self, id: Uuid) {
        if let Some(index) = self.panels.iter().position(|p| p.id() == id) {
            self.remove(index);
        }
    }

    /// Get all container configs
    pub fn configs(&self) -> Vec<ContainerConfig> {
        self.panels.iter().map(|p| p.config.clone()).collect()
    }

    /// Get the number of containers
    pub fn len(&self) -> usize {
        self.panels.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.panels.is_empty()
    }

    /// Validate all containers
    pub fn validate_all(&mut self) -> bool {
        self.panels.iter_mut().all(|p| p.validate())
    }

    /// Render the panel
    pub fn show(&mut self, ui: &mut Ui) -> ContainerListResponse {
        let mut response = ContainerListResponse::default();

        // Add button
        ui.horizontal(|ui| {
            if ui.button("+ Add Container").clicked() {
                let name = format!("Container {}", self.panels.len() + 1);
                response.added = Some(self.add_default(name));
            }
            if !self.is_empty() {
                ui.label(format!("{} container(s)", self.len()));
            }
        });

        ui.separator();

        if self.is_empty() {
            ui.label("No containers configured. Add one to get started.");
            return response;
        }

        // Container tabs/list
        ui.horizontal(|ui| {
            for (i, panel) in self.panels.iter().enumerate() {
                let is_selected = self.selected == Some(i);
                let label = RichText::new(&panel.config.name);
                let label = if is_selected { label.strong() } else { label };

                if ui.selectable_label(is_selected, label).clicked() {
                    self.selected = Some(i);
                }
            }
        });

        ui.separator();

        // Selected container panel
        if let Some(index) = self.selected {
            if let Some(panel) = self.panels.get_mut(index) {
                let panel_response = panel.show(ui);

                if panel_response.edit_requested {
                    panel.editing = true;
                }

                // Delete button
                ui.separator();
                if ui
                    .button(RichText::new("Delete Container").color(egui::Color32::RED))
                    .clicked()
                {
                    response.removed = Some(panel.id());
                }
            }
        }

        // Handle removal
        if let Some(id) = response.removed {
            self.remove_by_id(id);
        }

        response
    }
}

/// Response from the container list panel
#[derive(Default)]
pub struct ContainerListResponse {
    /// Container that was added (if any)
    pub added: Option<Uuid>,

    /// Container that was removed (if any)
    pub removed: Option<Uuid>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docker_panel_new() {
        let panel = DockerPanel::default();
        assert!(panel.editing);
        assert!(panel.error.is_none());
    }

    #[test]
    fn test_docker_panel_validate_valid() {
        let mut panel = DockerPanel::default();
        assert!(panel.validate());
        assert!(panel.error.is_none());
    }

    #[test]
    fn test_docker_panel_validate_invalid() {
        let mut panel = DockerPanel::new(ContainerConfig {
            name: String::new(),
            ..Default::default()
        });
        assert!(!panel.validate());
        assert!(panel.error.is_some());
    }

    #[test]
    fn test_container_list_panel() {
        let mut panel = ContainerListPanel::new();
        assert!(panel.is_empty());

        let id1 = panel.add_default("Container 1");
        let id2 = panel.add_default("Container 2");

        assert_eq!(panel.len(), 2);
        assert_eq!(panel.selected, Some(1)); // Last added is selected

        panel.remove_by_id(id1);
        assert_eq!(panel.len(), 1);
        assert_eq!(panel.panels[0].id(), id2);
    }

    #[test]
    fn test_container_list_configs() {
        let mut panel = ContainerListPanel::new();
        panel.add_default("Container 1");
        panel.add_default("Container 2");

        let configs = panel.configs();
        assert_eq!(configs.len(), 2);
        assert_eq!(configs[0].name, "Container 1");
        assert_eq!(configs[1].name, "Container 2");
    }

    #[test]
    fn test_container_list_validate_all() {
        let mut panel = ContainerListPanel::new();
        panel.add_default("Container 1");
        panel.add_default("Container 2");

        assert!(panel.validate_all());

        // Add an invalid container
        panel.add(ContainerConfig {
            name: String::new(),
            ..Default::default()
        });
        assert!(!panel.validate_all());
    }
}
