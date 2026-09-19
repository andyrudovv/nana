//! Helpers shared by the crate's tests.

use std::path::{Path, PathBuf};

/// Models live at the workspace root, but `cargo test` runs from the crate
/// directory, so a relative path silently resolves to nothing and every
/// model-backed test skips without saying so.
pub fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/wake")
}
