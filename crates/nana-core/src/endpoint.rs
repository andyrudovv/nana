//! An endpoint is one microphone + speaker pair Nana can hold a conversation on.
//!
//! The daemon owns a set of these rather than "the microphone", so the planned
//! multi-room setup is an extra implementation instead of a rewrite.

use std::fmt;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EndpointId(pub String);

impl fmt::Display for EndpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Drives whether acoustic echo cancellation is needed at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputClass {
    /// Nothing reaches the microphone; barge-in is trivial.
    Headphones,
    /// Nana hears herself; AEC is mandatory.
    Speakers,
    /// Backend gave no usable metadata.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AecMode {
    Off,
    /// Cancel our own synthesised speech using the render stream as reference.
    Render,
}

impl OutputClass {
    /// Unknown is treated as speakers: a needless AEC pass costs CPU, a missing
    /// one makes Nana interrupt herself.
    pub fn aec_mode(self) -> AecMode {
        match self {
            OutputClass::Headphones => AecMode::Off,
            OutputClass::Speakers | OutputClass::Unknown => AecMode::Render,
        }
    }
}

/// Whether the user is actually at this endpoint. Feeds endpoint priority once
/// room endpoints exist; on a single desktop it is recorded but not acted on.
#[derive(Debug, Clone, Copy, Default)]
pub struct Presence {
    pub idle: Option<Duration>,
    pub session_locked: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct EndpointInfo {
    pub id: EndpointId,
    pub input_name: String,
    pub output_name: String,
    pub output_class: OutputClass,
}

impl EndpointInfo {
    pub fn aec_mode(&self) -> AecMode {
        self.output_class.aec_mode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_output_falls_back_to_running_aec() {
        assert_eq!(OutputClass::Unknown.aec_mode(), AecMode::Render);
        assert_eq!(OutputClass::Headphones.aec_mode(), AecMode::Off);
    }
}
