//! Turning a stream of speech probabilities into utterance boundaries.
//!
//! Separate from the model on purpose: endpointing is where the latency budget
//! is won or lost, and it needs to be testable without loading an ONNX graph.

use nana_core::audio::{SAMPLE_RATE, VAD_WINDOW_SAMPLES};

/// 32 ms — one Silero window at 16 kHz.
pub const WINDOW_MS: u32 = (VAD_WINDOW_SAMPLES as u32 * 1000) / SAMPLE_RATE;

/// How long to wait for silence before calling the utterance finished.
///
/// A fixed threshold is the biggest hidden cost in the latency budget, so the
/// orchestrator picks a mode per utterance instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SilenceMode {
    /// The words so far already match a fast-path command; nothing more is coming.
    Short,
    Base,
    /// The partial transcript breaks off mid-clause, so the speaker is still going.
    Extended,
}

impl SilenceMode {
    pub fn millis(self) -> u32 {
        match self {
            SilenceMode::Short => 250,
            SilenceMode::Base => 500,
            SilenceMode::Extended => 900,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EndpointConfig {
    pub speech_threshold: f32,
    /// Shorter bursts are coughs, clicks and door noise, not speech.
    pub min_speech_ms: u32,
    /// Stops a stuck stream from holding a turn open forever.
    pub max_utterance_ms: u32,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self { speech_threshold: 0.5, min_speech_ms: 300, max_utterance_ms: 12_000 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// Speech has been going for `min_speech_ms`. The audio before this point
    /// is still in the pre-roll ring, so nothing is lost by confirming late.
    SpeechStart,
    SpeechEnd {
        speech_ms: u32,
        reason: EndReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    Silence,
    /// Hit `max_utterance_ms`; cut rather than wait.
    TooLong,
}

pub struct Endpointer {
    config: EndpointConfig,
    silence: SilenceMode,
    in_speech: bool,
    speech_run_ms: u32,
    silence_run_ms: u32,
    utterance_ms: u32,
}

impl Endpointer {
    pub fn new(config: EndpointConfig) -> Self {
        Self {
            config,
            silence: SilenceMode::Base,
            in_speech: false,
            speech_run_ms: 0,
            silence_run_ms: 0,
            utterance_ms: 0,
        }
    }

    /// Changes how long the trailing silence has to be. Takes effect on the
    /// next window, so it can be tightened as partial transcripts arrive.
    pub fn set_silence_mode(&mut self, mode: SilenceMode) {
        self.silence = mode;
    }

    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    pub fn reset(&mut self) {
        self.in_speech = false;
        self.speech_run_ms = 0;
        self.silence_run_ms = 0;
        self.utterance_ms = 0;
        self.silence = SilenceMode::Base;
    }

    /// Feeds one window's speech probability.
    pub fn push(&mut self, probability: f32) -> Option<VadEvent> {
        let speech = probability >= self.config.speech_threshold;
        if self.in_speech {
            self.push_in_speech(speech)
        } else {
            self.push_in_silence(speech)
        }
    }

    fn push_in_silence(&mut self, speech: bool) -> Option<VadEvent> {
        if !speech {
            self.speech_run_ms = 0;
            return None;
        }
        self.speech_run_ms += WINDOW_MS;
        if self.speech_run_ms < self.config.min_speech_ms {
            return None;
        }
        self.in_speech = true;
        self.utterance_ms = self.speech_run_ms;
        self.silence_run_ms = 0;
        Some(VadEvent::SpeechStart)
    }

    fn push_in_speech(&mut self, speech: bool) -> Option<VadEvent> {
        self.utterance_ms += WINDOW_MS;
        if speech {
            self.silence_run_ms = 0;
        } else {
            self.silence_run_ms += WINDOW_MS;
        }

        if self.silence_run_ms >= self.silence.millis() {
            // The trailing silence is not part of what the user said.
            let speech_ms = self.utterance_ms.saturating_sub(self.silence_run_ms);
            self.finish();
            return Some(VadEvent::SpeechEnd { speech_ms, reason: EndReason::Silence });
        }
        if self.utterance_ms >= self.config.max_utterance_ms {
            let speech_ms = self.utterance_ms;
            self.finish();
            return Some(VadEvent::SpeechEnd { speech_ms, reason: EndReason::TooLong });
        }
        None
    }

    fn finish(&mut self) {
        self.in_speech = false;
        self.speech_run_ms = 0;
        self.silence_run_ms = 0;
        self.utterance_ms = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn windows(ms: u32) -> usize {
        (ms / WINDOW_MS) as usize
    }

    fn feed(ep: &mut Endpointer, probability: f32, ms: u32) -> Vec<VadEvent> {
        (0..windows(ms)).filter_map(|_| ep.push(probability)).collect()
    }

    #[test]
    fn window_is_32ms() {
        assert_eq!(WINDOW_MS, 32);
    }

    #[test]
    fn a_cough_never_opens_an_utterance() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        assert!(feed(&mut ep, 0.9, 150).is_empty(), "150 ms is below min_speech_ms");
        assert!(!ep.in_speech());
    }

    #[test]
    fn speech_then_silence_produces_one_boundary() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        assert_eq!(feed(&mut ep, 0.9, 1000), vec![VadEvent::SpeechStart]);
        let events = feed(&mut ep, 0.0, 600);
        assert_eq!(events.len(), 1);
        let VadEvent::SpeechEnd { speech_ms, reason } = events[0] else {
            panic!("expected an end event, got {:?}", events[0]);
        };
        assert_eq!(reason, EndReason::Silence);
        // Trailing silence is excluded, so this is roughly the 1000 ms spoken.
        assert!((960..=1024).contains(&speech_ms), "speech_ms was {speech_ms}");
    }

    #[test]
    fn a_pause_shorter_than_the_threshold_does_not_end_the_utterance() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        feed(&mut ep, 0.9, 500);
        assert!(feed(&mut ep, 0.0, 300).is_empty(), "300 ms is under the 500 ms base");
        assert!(feed(&mut ep, 0.9, 300).is_empty());
        assert!(ep.in_speech(), "the utterance is still open");
    }

    #[test]
    fn short_mode_ends_sooner_than_base() {
        let mut short = Endpointer::new(EndpointConfig::default());
        short.set_silence_mode(SilenceMode::Short);
        feed(&mut short, 0.9, 500);
        assert!(!feed(&mut short, 0.0, 288).is_empty(), "Short ends at 250 ms");

        let mut base = Endpointer::new(EndpointConfig::default());
        feed(&mut base, 0.9, 500);
        assert!(feed(&mut base, 0.0, 288).is_empty(), "Base still waits at 288 ms");
    }

    #[test]
    fn extended_mode_keeps_waiting_through_a_mid_clause_pause() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        ep.set_silence_mode(SilenceMode::Extended);
        feed(&mut ep, 0.9, 500);
        assert!(feed(&mut ep, 0.0, 700).is_empty(), "Extended waits 900 ms");
        assert!(!feed(&mut ep, 0.0, 300).is_empty());
    }

    #[test]
    fn a_stuck_stream_is_cut_rather_than_held_open() {
        let config = EndpointConfig { max_utterance_ms: 1_000, ..Default::default() };
        let mut ep = Endpointer::new(config);
        let events = feed(&mut ep, 0.9, 5_000);
        assert_eq!(events.first(), Some(&VadEvent::SpeechStart));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, VadEvent::SpeechEnd { reason: EndReason::TooLong, .. })),
            "expected a forced cut, got {events:?}"
        );
    }

    #[test]
    fn the_detector_is_reusable_across_utterances() {
        let mut ep = Endpointer::new(EndpointConfig::default());
        for _ in 0..3 {
            assert_eq!(feed(&mut ep, 0.9, 600), vec![VadEvent::SpeechStart]);
            assert_eq!(feed(&mut ep, 0.0, 600).len(), 1);
            assert!(!ep.in_speech());
        }
    }
}
