pub mod detector;
pub mod frontend;
pub mod spotter;
#[cfg(test)]
pub(crate) mod testing;

pub use detector::{Detection, WakeWord, STEP_MS};
pub use frontend::Frontend;
pub use spotter::{build_reference, Spotter};
