pub mod detector;
pub mod frontend;
#[cfg(test)]
pub(crate) mod testing;

pub use detector::{Detection, WakeWord, STEP_MS};
pub use frontend::Frontend;
