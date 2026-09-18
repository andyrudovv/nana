//! Boundaries between the orchestrator and the things it drives.
//!
//! Kept deliberately narrow so a provider can be swapped without the state
//! machine noticing (R4).

use crate::lang::Lang;
use crate::Result;

/// What speech recognition returns for one utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// `None` when the model reported a language Nana does not handle.
    pub lang: Option<Lang>,
    /// Mean token probability. A usable ranking signal, **not** a calibrated
    /// probability — thresholds on it have to be measured, not assumed.
    pub confidence: f32,
    /// Model's own estimate that the audio holds no speech at all.
    pub no_speech: f32,
}

impl Transcript {
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Whether this transcript may be acted on.
    ///
    /// Whisper hallucinates fluent text from room tone and reports it with high
    /// token confidence and a no-speech probability of zero, so neither number
    /// can gate on its own. An out-of-set language is the cheap tell: measured
    /// on silence it produced Korean. VAD still has to gate the call itself.
    pub fn is_usable(&self) -> bool {
        !self.is_empty() && self.lang.is_some()
    }
}

/// Synchronous on purpose: recognition is CPU-bound, so the caller decides
/// which blocking pool it runs on rather than the implementation pretending to
/// be async.
pub trait Asr: Send + Sync {
    /// `pcm` is 16 kHz mono, as produced by capture.
    fn transcribe(&self, pcm: &[i16]) -> Result<Transcript>;

    /// Human-readable model identity, for telemetry and benchmark output.
    fn model(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(text: &str, lang: Option<Lang>) -> Transcript {
        Transcript { text: text.into(), lang, confidence: 0.75, no_speech: 0.0 }
    }

    #[test]
    fn hallucination_in_another_language_is_rejected() {
        // Observed from four seconds of room tone, conf 0.75, no-speech 0.00.
        assert!(!transcript("때문에", None).is_usable());
    }

    #[test]
    fn blank_transcript_is_rejected() {
        assert!(!transcript("   ", Some(Lang::Ru)).is_usable());
    }

    #[test]
    fn ordinary_reply_is_usable() {
        assert!(transcript("какая погода", Some(Lang::Ru)).is_usable());
    }
}
