//! Minimal native file/folder picker helpers without external GUI dependencies.

use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogMode {
    OpenFile,
    SaveFile,
    OpenFolder,
}

enum DialogCommandResult {
    Selected(Option<PathBuf>),
    NotFound,
    Failed(String),
}

pub fn pick_path(mode: DialogMode, initial_path: Option<&str>) -> Result<Option<PathBuf>, String> {
    #[cfg(target_os = "linux")]
    {
        pick_path_linux(mode, initial_path)
    }
    #[cfg(target_os = "macos")]
    {
        pick_path_macos(mode, initial_path)
    }
    #[cfg(target_os = "windows")]
    {
        pick_path_windows(mode, initial_path)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let _ = (mode, initial_path);
        Err("Native file picker is not supported on this platform".to_string())
    }
}

#[cfg(target_os = "linux")]
fn pick_path_linux(
    mode: DialogMode,
    initial_path: Option<&str>,
) -> Result<Option<PathBuf>, String> {
    let mut zenity_args = vec!["--file-selection".to_string()];
    match mode {
        DialogMode::OpenFile => {}
        DialogMode::SaveFile => {
            zenity_args.push("--save".to_string());
            zenity_args.push("--confirm-overwrite".to_string());
        }
        DialogMode::OpenFolder => {
            zenity_args.push("--directory".to_string());
        }
    }
    if let Some(initial) = initial_path.filter(|path| !path.trim().is_empty()) {
        zenity_args.push("--filename".to_string());
        zenity_args.push(initial.to_string());
    }

    match run_dialog_command("zenity", &zenity_args) {
        DialogCommandResult::Selected(path) => return Ok(path),
        DialogCommandResult::NotFound => {}
        DialogCommandResult::Failed(err) => return Err(err),
    }

    let mut kdialog_args = vec![match mode {
        DialogMode::OpenFile => "--getopenfilename",
        DialogMode::SaveFile => "--getsavefilename",
        DialogMode::OpenFolder => "--getexistingdirectory",
    }
    .to_string()];
    if let Some(initial) = initial_path.filter(|path| !path.trim().is_empty()) {
        kdialog_args.push(initial.to_string());
    }

    match run_dialog_command("kdialog", &kdialog_args) {
        DialogCommandResult::Selected(path) => Ok(path),
        DialogCommandResult::NotFound => {
            Err("No native file picker found. Install `zenity` or `kdialog`.".to_string())
        }
        DialogCommandResult::Failed(err) => Err(err),
    }
}

#[cfg(target_os = "macos")]
fn pick_path_macos(
    mode: DialogMode,
    _initial_path: Option<&str>,
) -> Result<Option<PathBuf>, String> {
    let script = match mode {
        DialogMode::OpenFile => "POSIX path of (choose file)",
        DialogMode::SaveFile => "POSIX path of (choose file name)",
        DialogMode::OpenFolder => "POSIX path of (choose folder)",
    };
    match run_dialog_command("osascript", &["-e".to_string(), script.to_string()]) {
        DialogCommandResult::Selected(path) => Ok(path),
        DialogCommandResult::NotFound => {
            Err("`osascript` not found, cannot open file picker".to_string())
        }
        DialogCommandResult::Failed(err) => Err(err),
    }
}

#[cfg(target_os = "windows")]
fn pick_path_windows(
    mode: DialogMode,
    initial_path: Option<&str>,
) -> Result<Option<PathBuf>, String> {
    let escaped_initial = initial_path
        .filter(|path| !path.trim().is_empty())
        .unwrap_or("")
        .replace('\\', "\\\\")
        .replace('\'', "''");
    let script = match mode {
        DialogMode::OpenFile => format!(
            "$d=New-Object System.Windows.Forms.OpenFileDialog;$d.InitialDirectory='{escaped_initial}';if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){{Write-Output $d.FileName}}"
        ),
        DialogMode::SaveFile => format!(
            "$d=New-Object System.Windows.Forms.SaveFileDialog;$d.InitialDirectory='{escaped_initial}';if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){{Write-Output $d.FileName}}"
        ),
        DialogMode::OpenFolder => format!(
            "$d=New-Object System.Windows.Forms.FolderBrowserDialog;$d.SelectedPath='{escaped_initial}';if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){{Write-Output $d.SelectedPath}}"
        ),
    };
    let args = vec![
        "-NoProfile".to_string(),
        "-Command".to_string(),
        "Add-Type -AssemblyName System.Windows.Forms;".to_string() + &script,
    ];
    match run_dialog_command("powershell", &args) {
        DialogCommandResult::Selected(path) => Ok(path),
        DialogCommandResult::NotFound => {
            Err("`powershell` not found, cannot open file picker".to_string())
        }
        DialogCommandResult::Failed(err) => Err(err),
    }
}

fn run_dialog_command(cmd: &str, args: &[String]) -> DialogCommandResult {
    let output = match Command::new(cmd).args(args).output() {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return DialogCommandResult::NotFound;
        }
        Err(err) => {
            return DialogCommandResult::Failed(format!("Failed to launch {cmd}: {err}"));
        }
    };

    if output.status.success() {
        let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if value.is_empty() {
            return DialogCommandResult::Selected(None);
        }
        return DialogCommandResult::Selected(Some(PathBuf::from(value)));
    }

    if matches!(output.status.code(), Some(1) | Some(130)) {
        return DialogCommandResult::Selected(None);
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let details = if stderr.is_empty() {
        format!("status {}", output.status)
    } else {
        format!("status {}: {}", output.status, stderr)
    };
    DialogCommandResult::Failed(format!("{cmd} failed with {details}"))
}
