//! Baseline benchmark configuration module.
//!
//! Loads a TOML config with profile-based overrides and emits shell-friendly
//! KEY=VALUE pairs for Bash script consumption.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Resolved baseline configuration after profile merging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineConfig {
    pub fixtures_dir: String,
    pub output_root: String,
    pub passes: u32,
    pub enable_checkpointing: bool,
    pub envs: String,
    pub docker_memory: String,
}

/// Profile overrides — all fields optional; only non-None values override defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileOverrides {
    pub fixtures_dir: Option<String>,
    pub output_root: Option<String>,
    pub passes: Option<u32>,
    pub enable_checkpointing: Option<bool>,
    pub envs: Option<String>,
    pub docker_memory: Option<String>,
}

/// Raw TOML file structure with defaults and optional profiles.
#[derive(Debug, Clone, Deserialize)]
pub struct ConfigFile {
    pub defaults: BaselineConfig,
    pub quick: Option<ProfileOverrides>,
    pub full: Option<ProfileOverrides>,
}

/// Load configuration from a TOML file and resolve the specified profile.
///
/// Profile resolution: start with `[defaults]`, then overlay the selected
/// profile section. Unknown profile names return an error.
pub fn load_config(path: &Path, profile: &str) -> Result<BaselineConfig, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read config file {:?}: {}", path, e))?;

    let config_file: ConfigFile =
        toml::from_str(&content).map_err(|e| format!("Failed to parse TOML: {}", e))?;

    let mut resolved = config_file.defaults;

    let overrides = match profile {
        "quick" => config_file.quick,
        "full" => config_file.full,
        "defaults" => None,
        other => return Err(format!("Unknown profile: '{}'. Valid: quick, full, defaults", other)),
    };

    if let Some(ov) = overrides {
        if let Some(v) = ov.fixtures_dir {
            resolved.fixtures_dir = v;
        }
        if let Some(v) = ov.output_root {
            resolved.output_root = v;
        }
        if let Some(v) = ov.passes {
            resolved.passes = v;
        }
        if let Some(v) = ov.enable_checkpointing {
            resolved.enable_checkpointing = v;
        }
        if let Some(v) = ov.envs {
            resolved.envs = v;
        }
        if let Some(v) = ov.docker_memory {
            resolved.docker_memory = v;
        }
    }

    Ok(resolved)
}

/// Emit resolved configuration as shell-friendly KEY=VALUE lines.
///
/// Each line is formatted as `UPPER_KEY=value` suitable for `eval` in Bash.
pub fn dump_shell_vars(config: &BaselineConfig) -> String {
    let mut lines = Vec::new();
    lines.push(format!("FIXTURES_DIR={}", shell_quote(&config.fixtures_dir)));
    lines.push(format!("OUTPUT_ROOT={}", shell_quote(&config.output_root)));
    lines.push(format!("PASSES={}", config.passes));
    lines.push(format!(
        "ENABLE_CHECKPOINTING={}",
        config.enable_checkpointing
    ));
    lines.push(format!("ENVS={}", shell_quote(&config.envs)));
    lines.push(format!("DOCKER_MEMORY={}", shell_quote(&config.docker_memory)));
    lines.join("\n")
}

/// Compute SHA-256 hex digest of the raw config file content.
pub fn config_sha256(path: &Path) -> Result<String, String> {
    let content = std::fs::read(path)
        .map_err(|e| format!("Failed to read config file {:?}: {}", path, e))?;
    let mut hasher = Sha256::new();
    hasher.update(&content);
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

/// Shell-quote a string value (wrap in single quotes, escape internal quotes).
fn shell_quote(s: &str) -> String {
    if s.contains('\'') || s.contains(' ') || s.contains('"') || s.contains('$') {
        format!("'{}'", s.replace('\'', "'\\''"))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dump_shell_vars_format() {
        let config = BaselineConfig {
            fixtures_dir: "bench-artifacts/fixtures".to_string(),
            output_root: "bench-artifacts/sol-baseline".to_string(),
            passes: 3,
            enable_checkpointing: false,
            envs: "native".to_string(),
            docker_memory: "16g".to_string(),
        };
        let output = dump_shell_vars(&config);
        assert!(output.contains("PASSES=3"));
        assert!(output.contains("FIXTURES_DIR=bench-artifacts/fixtures"));
        assert!(output.contains("ENABLE_CHECKPOINTING=false"));
    }

    #[test]
    fn test_shell_quote_special_chars() {
        assert_eq!(shell_quote("simple"), "simple");
        assert_eq!(shell_quote("has space"), "'has space'");
    }
}
