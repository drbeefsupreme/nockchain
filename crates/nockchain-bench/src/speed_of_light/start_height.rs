//! Helpers for resolving start heights for speed-of-light runs.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum StartHeightError {
    #[error("checkpoint height required but unavailable")]
    MissingCheckpointHeight,
}

/// Resolve the start height based on explicit and checkpoint-derived heights.
pub fn resolve_start_height(
    explicit_start: Option<u64>,
    checkpoint_height: Option<u64>,
) -> Result<u64, StartHeightError> {
    if let Some(height) = explicit_start {
        return Ok(height);
    }
    if let Some(height) = checkpoint_height {
        return Ok(height + 1);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_start_height_explicit_overrides() {
        let resolved = resolve_start_height(Some(5), Some(10)).expect("should resolve");
        assert_eq!(resolved, 5);
    }

    #[test]
    fn test_resolve_start_height_from_checkpoint() {
        let resolved = resolve_start_height(None, Some(7)).expect("should resolve");
        assert_eq!(resolved, 8);
    }

    #[test]
    fn test_resolve_start_height_defaults_to_zero() {
        let resolved = resolve_start_height(None, None).expect("should resolve");
        assert_eq!(resolved, 0);
    }
}
