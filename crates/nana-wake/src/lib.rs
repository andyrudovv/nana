pub mod detector;
pub mod frontend;
pub mod spotter;
#[cfg(test)]
pub(crate) mod testing;
pub mod verify;

pub use detector::{Detection, WakeWord, STEP_MS};
pub use frontend::Frontend;
pub use spotter::{build_reference, Spotter};
pub use verify::{transcript_confirms, PreTriggerSilence, DEFAULT_SPELLINGS};
