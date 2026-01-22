//! Terminal output panel with tabs
//!
//! Displays container logs and build output with a tabbed interface.

use std::collections::HashMap;

use egui::{Color32, RichText, ScrollArea, Ui};
use uuid::Uuid;

/// Maximum number of lines to keep per terminal
const MAX_LINES: usize = 10000;

/// A single terminal instance
#[derive(Debug, Clone)]
pub struct Terminal {
    /// Unique identifier
    pub id: Uuid,

    /// Display name (shown in tab)
    pub name: String,

    /// Output lines
    pub lines: Vec<TerminalLine>,

    /// Whether to auto-scroll
    pub auto_scroll: bool,

    /// Whether this terminal is active/running
    pub active: bool,
}

impl Terminal {
    /// Create a new terminal
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            lines: Vec::new(),
            auto_scroll: true,
            active: true,
        }
    }

    /// Add a line to the terminal
    pub fn push_line(&mut self, line: impl Into<String>) {
        self.lines.push(TerminalLine::stdout(line));
        self.trim_lines();
    }

    /// Add an error line to the terminal
    pub fn push_error(&mut self, line: impl Into<String>) {
        self.lines.push(TerminalLine::stderr(line));
        self.trim_lines();
    }

    /// Add a system message (e.g., "Container started")
    pub fn push_system(&mut self, line: impl Into<String>) {
        self.lines.push(TerminalLine::system(line));
        self.trim_lines();
    }

    /// Clear all output
    pub fn clear(&mut self) {
        self.lines.clear();
    }

    /// Mark the terminal as inactive
    pub fn mark_inactive(&mut self) {
        self.active = false;
    }

    /// Trim lines to stay under the limit
    fn trim_lines(&mut self) {
        if self.lines.len() > MAX_LINES {
            let remove_count = self.lines.len() - MAX_LINES;
            self.lines.drain(0..remove_count);
        }
    }

    /// Get the last N lines
    pub fn last_lines(&self, n: usize) -> &[TerminalLine] {
        let start = self.lines.len().saturating_sub(n);
        &self.lines[start..]
    }
}

/// A line of terminal output
#[derive(Debug, Clone)]
pub struct TerminalLine {
    /// The text content
    pub text: String,

    /// Line type (stdout, stderr, system)
    pub line_type: LineType,

    /// Timestamp when this line was added
    pub timestamp: std::time::Instant,
}

impl TerminalLine {
    /// Create a stdout line
    pub fn stdout(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            line_type: LineType::Stdout,
            timestamp: std::time::Instant::now(),
        }
    }

    /// Create a stderr line
    pub fn stderr(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            line_type: LineType::Stderr,
            timestamp: std::time::Instant::now(),
        }
    }

    /// Create a system message line
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            line_type: LineType::System,
            timestamp: std::time::Instant::now(),
        }
    }

    /// Get the color for this line type
    pub fn color(&self) -> Color32 {
        match self.line_type {
            LineType::Stdout => Color32::LIGHT_GRAY,
            LineType::Stderr => Color32::from_rgb(255, 100, 100),
            LineType::System => Color32::from_rgb(100, 200, 255),
        }
    }
}

/// Type of terminal line
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineType {
    Stdout,
    Stderr,
    System,
}

/// Panel containing multiple terminals with tabs
pub struct TerminalPanel {
    /// All terminals
    terminals: HashMap<Uuid, Terminal>,

    /// Order of terminal tabs
    tab_order: Vec<Uuid>,

    /// Currently selected terminal
    selected: Option<Uuid>,
}

impl Default for TerminalPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalPanel {
    /// Create a new terminal panel
    pub fn new() -> Self {
        Self {
            terminals: HashMap::new(),
            tab_order: Vec::new(),
            selected: None,
        }
    }

    /// Add a new terminal and return its ID
    pub fn add_terminal(&mut self, name: impl Into<String>) -> Uuid {
        let terminal = Terminal::new(name);
        let id = terminal.id;
        self.terminals.insert(id, terminal);
        self.tab_order.push(id);
        self.selected = Some(id);
        id
    }

    /// Get a terminal by ID
    pub fn get(&self, id: Uuid) -> Option<&Terminal> {
        self.terminals.get(&id)
    }

    /// Get a mutable terminal by ID
    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut Terminal> {
        self.terminals.get_mut(&id)
    }

    /// Remove a terminal
    pub fn remove(&mut self, id: Uuid) {
        self.terminals.remove(&id);
        self.tab_order.retain(|&i| i != id);

        if self.selected == Some(id) {
            self.selected = self.tab_order.first().copied();
        }
    }

    /// Get the currently selected terminal
    pub fn selected(&self) -> Option<&Terminal> {
        self.selected.and_then(|id| self.terminals.get(&id))
    }

    /// Get the currently selected terminal mutably
    pub fn selected_mut(&mut self) -> Option<&mut Terminal> {
        self.selected.and_then(|id| self.terminals.get_mut(&id))
    }

    /// Select a terminal by ID
    pub fn select(&mut self, id: Uuid) {
        if self.terminals.contains_key(&id) {
            self.selected = Some(id);
        }
    }

    /// Get all terminal IDs
    pub fn terminal_ids(&self) -> &[Uuid] {
        &self.tab_order
    }

    /// Check if there are any terminals
    pub fn is_empty(&self) -> bool {
        self.terminals.is_empty()
    }

    /// Get the number of terminals
    pub fn len(&self) -> usize {
        self.terminals.len()
    }

    /// Push a line to a specific terminal
    pub fn push_line(&mut self, id: Uuid, line: impl Into<String>) {
        if let Some(terminal) = self.terminals.get_mut(&id) {
            terminal.push_line(line);
        }
    }

    /// Push an error line to a specific terminal
    pub fn push_error(&mut self, id: Uuid, line: impl Into<String>) {
        if let Some(terminal) = self.terminals.get_mut(&id) {
            terminal.push_error(line);
        }
    }

    /// Push a system message to a specific terminal
    pub fn push_system(&mut self, id: Uuid, line: impl Into<String>) {
        if let Some(terminal) = self.terminals.get_mut(&id) {
            terminal.push_system(line);
        }
    }

    /// Render the terminal panel
    pub fn show(&mut self, ui: &mut Ui) {
        if self.is_empty() {
            ui.label("No terminals open");
            return;
        }

        // Tab bar
        ui.horizontal(|ui| {
            let mut to_close = None;

            for &id in &self.tab_order {
                if let Some(terminal) = self.terminals.get(&id) {
                    let is_selected = self.selected == Some(id);
                    let name = if terminal.active {
                        format!("{} [running]", terminal.name)
                    } else {
                        terminal.name.clone()
                    };

                    let tab_text = if is_selected {
                        RichText::new(&name).strong()
                    } else {
                        RichText::new(&name)
                    };

                    if ui.selectable_label(is_selected, tab_text).clicked() {
                        self.selected = Some(id);
                    }

                    // Close button
                    if ui.small_button("x").clicked() {
                        to_close = Some(id);
                    }

                    ui.separator();
                }
            }

            if let Some(id) = to_close {
                self.remove(id);
            }
        });

        ui.separator();

        // Terminal content
        if let Some(id) = self.selected {
            if let Some(terminal) = self.terminals.get_mut(&id) {
                // Controls
                ui.horizontal(|ui| {
                    ui.checkbox(&mut terminal.auto_scroll, "Auto-scroll");
                    if ui.button("Clear").clicked() {
                        terminal.clear();
                    }
                    ui.label(format!("{} lines", terminal.lines.len()));
                });

                ui.separator();

                // Scrollable output area
                let scroll_height = ui.available_height();
                ScrollArea::vertical()
                    .max_height(scroll_height)
                    .stick_to_bottom(terminal.auto_scroll)
                    .show(ui, |ui| {
                        ui.style_mut().visuals.override_text_color = Some(Color32::LIGHT_GRAY);
                        ui.style_mut().spacing.item_spacing.y = 0.0;

                        for line in &terminal.lines {
                            ui.label(RichText::new(&line.text).color(line.color()).monospace());
                        }
                    });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_new() {
        let terminal = Terminal::new("Test");
        assert_eq!(terminal.name, "Test");
        assert!(terminal.lines.is_empty());
        assert!(terminal.auto_scroll);
        assert!(terminal.active);
    }

    #[test]
    fn test_terminal_push_lines() {
        let mut terminal = Terminal::new("Test");

        terminal.push_line("stdout line");
        terminal.push_error("stderr line");
        terminal.push_system("system message");

        assert_eq!(terminal.lines.len(), 3);
        assert_eq!(terminal.lines[0].line_type, LineType::Stdout);
        assert_eq!(terminal.lines[1].line_type, LineType::Stderr);
        assert_eq!(terminal.lines[2].line_type, LineType::System);
    }

    #[test]
    fn test_terminal_clear() {
        let mut terminal = Terminal::new("Test");
        terminal.push_line("line 1");
        terminal.push_line("line 2");
        assert_eq!(terminal.lines.len(), 2);

        terminal.clear();
        assert!(terminal.lines.is_empty());
    }

    #[test]
    fn test_terminal_last_lines() {
        let mut terminal = Terminal::new("Test");
        for i in 0..10 {
            terminal.push_line(format!("line {}", i));
        }

        let last = terminal.last_lines(3);
        assert_eq!(last.len(), 3);
        assert_eq!(last[0].text, "line 7");
        assert_eq!(last[2].text, "line 9");
    }

    #[test]
    fn test_terminal_panel_add_remove() {
        let mut panel = TerminalPanel::new();
        assert!(panel.is_empty());

        let id1 = panel.add_terminal("Terminal 1");
        let id2 = panel.add_terminal("Terminal 2");

        assert_eq!(panel.len(), 2);
        assert_eq!(panel.selected, Some(id2)); // Most recent is selected

        panel.remove(id2);
        assert_eq!(panel.len(), 1);
        assert_eq!(panel.selected, Some(id1)); // Falls back to remaining
    }

    #[test]
    fn test_terminal_panel_select() {
        let mut panel = TerminalPanel::new();
        let id1 = panel.add_terminal("Terminal 1");
        let _id2 = panel.add_terminal("Terminal 2");

        panel.select(id1);
        assert_eq!(panel.selected, Some(id1));
    }

    #[test]
    fn test_terminal_panel_push_lines() {
        let mut panel = TerminalPanel::new();
        let id = panel.add_terminal("Test");

        panel.push_line(id, "stdout");
        panel.push_error(id, "stderr");
        panel.push_system(id, "system");

        let terminal = panel.get(id).unwrap();
        assert_eq!(terminal.lines.len(), 3);
    }

    #[test]
    fn test_line_colors() {
        let stdout = TerminalLine::stdout("test");
        let stderr = TerminalLine::stderr("test");
        let system = TerminalLine::system("test");

        // Just verify they return different colors
        assert_ne!(stdout.color(), stderr.color());
        assert_ne!(stdout.color(), system.color());
    }
}
