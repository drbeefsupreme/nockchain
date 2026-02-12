//! Terminal output panel with tabs
//!
//! Displays container logs and build output with a tabbed interface.
//! Supports ANSI color codes for colorized output.

use std::collections::HashMap;

use egui::{Color32, RichText, ScrollArea, Ui};
use uuid::Uuid;

/// Maximum number of lines to keep per terminal
const MAX_LINES: usize = 10000;

/// A segment of text with a specific color
#[derive(Debug, Clone)]
pub struct ColoredSegment {
    pub text: String,
    pub color: Color32,
}

/// Parse ANSI escape codes and return colored segments
pub fn parse_ansi(text: &str) -> Vec<ColoredSegment> {
    let mut segments = Vec::new();
    let mut current_color = Color32::LIGHT_GRAY;
    let mut current_text = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Start of escape sequence
            if chars.peek() == Some(&'[') {
                chars.next(); // consume '['

                // Collect the escape code parameters
                let mut code = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch.is_ascii_digit() || ch == ';' {
                        code.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }

                // Check for the terminator (usually 'm' for colors)
                if chars.peek() == Some(&'m') {
                    chars.next(); // consume 'm'

                    // Save current text segment if any
                    if !current_text.is_empty() {
                        segments.push(ColoredSegment {
                            text: std::mem::take(&mut current_text),
                            color: current_color,
                        });
                    }

                    // Parse the color code
                    current_color = parse_ansi_color(&code);
                }
                // Skip other escape codes
                continue;
            }
        }
        current_text.push(c);
    }

    // Push remaining text
    if !current_text.is_empty() {
        segments.push(ColoredSegment {
            text: current_text,
            color: current_color,
        });
    }

    segments
}

/// Parse ANSI color code string into a Color32
fn parse_ansi_color(code: &str) -> Color32 {
    let parts: Vec<&str> = code.split(';').collect();

    // Handle empty or "0" (reset)
    if code.is_empty() || code == "0" {
        return Color32::LIGHT_GRAY;
    }

    let mut i = 0;
    while i < parts.len() {
        match parts[i] {
            "0" => return Color32::LIGHT_GRAY,               // Reset
            "1" => {}                                        // Bold - we'll just ignore for now
            "2" => {}                                        // Dim
            "3" => {}                                        // Italic
            "4" => {}                                        // Underline
            "30" => return Color32::from_rgb(0, 0, 0),       // Black
            "31" => return Color32::from_rgb(205, 49, 49),   // Red
            "32" => return Color32::from_rgb(13, 188, 121),  // Green
            "33" => return Color32::from_rgb(229, 229, 16),  // Yellow
            "34" => return Color32::from_rgb(36, 114, 200),  // Blue
            "35" => return Color32::from_rgb(188, 63, 188),  // Magenta
            "36" => return Color32::from_rgb(17, 168, 205),  // Cyan
            "37" => return Color32::from_rgb(229, 229, 229), // White
            "90" => return Color32::from_rgb(102, 102, 102), // Bright black (gray)
            "91" => return Color32::from_rgb(241, 76, 76),   // Bright red
            "92" => return Color32::from_rgb(35, 209, 139),  // Bright green
            "93" => return Color32::from_rgb(245, 245, 67),  // Bright yellow
            "94" => return Color32::from_rgb(59, 142, 234),  // Bright blue
            "95" => return Color32::from_rgb(214, 112, 214), // Bright magenta
            "96" => return Color32::from_rgb(41, 184, 219),  // Bright cyan
            "97" => return Color32::from_rgb(255, 255, 255), // Bright white
            "38" => {
                // Extended color (256-color or RGB)
                if i + 1 < parts.len() {
                    match parts[i + 1] {
                        "5" => {
                            // 256-color mode: 38;5;n
                            if i + 2 < parts.len() {
                                if let Ok(n) = parts[i + 2].parse::<u8>() {
                                    return ansi_256_to_rgb(n);
                                }
                            }
                            i += 2;
                        }
                        "2" => {
                            // RGB mode: 38;2;r;g;b
                            if i + 4 < parts.len() {
                                if let (Ok(r), Ok(g), Ok(b)) = (
                                    parts[i + 2].parse::<u8>(),
                                    parts[i + 3].parse::<u8>(),
                                    parts[i + 4].parse::<u8>(),
                                ) {
                                    return Color32::from_rgb(r, g, b);
                                }
                            }
                            i += 4;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    Color32::LIGHT_GRAY
}

/// Convert ANSI 256-color palette index to RGB
fn ansi_256_to_rgb(n: u8) -> Color32 {
    match n {
        // Standard colors (0-7)
        0 => Color32::from_rgb(0, 0, 0),
        1 => Color32::from_rgb(205, 49, 49),
        2 => Color32::from_rgb(13, 188, 121),
        3 => Color32::from_rgb(229, 229, 16),
        4 => Color32::from_rgb(36, 114, 200),
        5 => Color32::from_rgb(188, 63, 188),
        6 => Color32::from_rgb(17, 168, 205),
        7 => Color32::from_rgb(229, 229, 229),
        // Bright colors (8-15)
        8 => Color32::from_rgb(102, 102, 102),
        9 => Color32::from_rgb(241, 76, 76),
        10 => Color32::from_rgb(35, 209, 139),
        11 => Color32::from_rgb(245, 245, 67),
        12 => Color32::from_rgb(59, 142, 234),
        13 => Color32::from_rgb(214, 112, 214),
        14 => Color32::from_rgb(41, 184, 219),
        15 => Color32::from_rgb(255, 255, 255),
        // 216 colors (16-231): 6x6x6 color cube
        16..=231 => {
            let n = n - 16;
            let r = (n / 36) % 6;
            let g = (n / 6) % 6;
            let b = n % 6;
            let to_val = |v: u8| if v == 0 { 0 } else { 55 + v * 40 };
            Color32::from_rgb(to_val(r), to_val(g), to_val(b))
        }
        // Grayscale (232-255): 24 shades
        232..=255 => {
            let gray = 8 + (n - 232) * 10;
            Color32::from_rgb(gray, gray, gray)
        }
    }
}

/// Strip ANSI escape codes from text (for plain text display)
pub fn strip_ansi(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip escape sequence
            if chars.peek() == Some(&'[') {
                chars.next();
                // Skip until we hit the terminator
                while let Some(&ch) = chars.peek() {
                    chars.next();
                    if ch.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

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
    /// The raw text content (may contain ANSI codes)
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

    /// Get the default color for this line type (used when no ANSI codes)
    pub fn default_color(&self) -> Color32 {
        match self.line_type {
            LineType::Stdout => Color32::LIGHT_GRAY,
            LineType::Stderr => Color32::from_rgb(255, 100, 100),
            LineType::System => Color32::from_rgb(100, 200, 255),
        }
    }

    /// Parse ANSI codes and return colored segments
    pub fn colored_segments(&self) -> Vec<ColoredSegment> {
        let segments = parse_ansi(&self.text);
        if segments.is_empty() {
            // No ANSI codes, return plain text with default color
            vec![ColoredSegment {
                text: self.text.clone(),
                color: self.default_color(),
            }]
        } else {
            segments
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
                    .id_salt(format!("terminal_scroll_{}", id))
                    .max_height(scroll_height)
                    .stick_to_bottom(terminal.auto_scroll)
                    .show(ui, |ui| {
                        ui.style_mut().spacing.item_spacing.y = 0.0;

                        for line in &terminal.lines {
                            // Render each line with ANSI colors
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 0.0;
                                for segment in line.colored_segments() {
                                    ui.label(
                                        RichText::new(&segment.text)
                                            .color(segment.color)
                                            .monospace(),
                                    );
                                }
                            });
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
        assert_ne!(stdout.default_color(), stderr.default_color());
        assert_ne!(stdout.default_color(), system.default_color());
    }

    #[test]
    fn test_parse_ansi_basic() {
        // Test green text followed by more text
        let segments = parse_ansi("\x1b[32mGreen text\x1b[0m normal");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].text, "Green text");
        assert_eq!(segments[0].color, Color32::from_rgb(13, 188, 121)); // Green
        assert_eq!(segments[1].text, " normal");
        assert_eq!(segments[1].color, Color32::LIGHT_GRAY); // Reset

        // Test just green text ending with reset (only 1 segment since trailing reset produces nothing)
        let segments2 = parse_ansi("\x1b[32mGreen only\x1b[0m");
        assert_eq!(segments2.len(), 1);
        assert_eq!(segments2[0].text, "Green only");
    }

    #[test]
    fn test_parse_ansi_256_color() {
        // Test 256-color mode (e.g., \x1b[38;5;246m for gray)
        let segments = parse_ansi("\x1b[38;5;246mGray text\x1b[0m");
        assert!(!segments.is_empty());
        assert_eq!(segments[0].text, "Gray text");
    }

    #[test]
    fn test_strip_ansi() {
        let text = "\x1b[32mGreen\x1b[0m and \x1b[31mRed\x1b[0m";
        let stripped = strip_ansi(text);
        assert_eq!(stripped, "Green and Red");
    }

    #[test]
    fn test_parse_ansi_no_codes() {
        let segments = parse_ansi("Plain text");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Plain text");
    }
}
