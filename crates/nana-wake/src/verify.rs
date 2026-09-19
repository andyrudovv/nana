//! Cutting false activations on a short wake word.
//!
//! "Nana" is two syllables and sits inside ordinary words — банана, ананас —
//! and falls out of running speech as «она на». A similarity threshold alone
//! cannot separate those, so detection is confirmed by two cheap tests before
//! the assistant reacts.

use nana_core::audio::{SAMPLE_RATE, VAD_WINDOW_SAMPLES};

/// Spellings the recogniser may return for the wake word.
pub const DEFAULT_SPELLINGS: [&str; 2] = ["нана", "nana"];

/// Does the transcript contain the wake word as a word of its own?
///
/// This is the test that separates «Нана, включи свет» from «купил банана»:
/// the recogniser sees the whole word, so a substring match inside a longer
/// token is rejected where an acoustic score could not tell them apart.
pub fn transcript_confirms(text: &str, spellings: &[&str]) -> bool {
    tokens(text).any(|token| spellings.iter().any(|s| token == *s))
}

fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).map(|t| t.to_lowercase())
}

/// Requires the wake word to begin a fresh burst of speech.
///
/// Two facts separate «Нана, включи свет» from «я купил банана», and neither
/// needs a recogniser:
///
/// - Addressing an assistant follows a pause. A word buried in a sentence does
///   not.
/// - The wake word is short. If speech has already been running for seconds
///   when the detector fires, the match cannot be at the start of that speech.
///
/// Checking only "is it quiet now" fails the second case: the sentence about
/// bananas was itself preceded by a pause.
pub struct PreTriggerSilence {
    required_quiet_ms: u32,
    /// Longest burst a detection may sit in and still count as its beginning.
    max_burst_ms: u32,
    threshold: f32,
    quiet_ms: u32,
    burst_ms: u32,
    quiet_before_burst_ms: u32,
}

/// One VAD window.
const WINDOW_MS: u32 = (VAD_WINDOW_SAMPLES as u32 * 1000) / SAMPLE_RATE;

impl PreTriggerSilence {
    pub fn new(required_quiet_ms: u32, max_burst_ms: u32, speech_threshold: f32) -> Self {
        Self {
            required_quiet_ms,
            max_burst_ms,
            threshold: speech_threshold,
            quiet_ms: required_quiet_ms,
            burst_ms: 0,
            quiet_before_burst_ms: required_quiet_ms,
        }
    }

    /// Feeds one VAD window's speech probability.
    pub fn push(&mut self, probability: f32) {
        if probability >= self.threshold {
            if self.burst_ms == 0 {
                self.quiet_before_burst_ms = self.quiet_ms;
            }
            self.burst_ms = self.burst_ms.saturating_add(WINDOW_MS);
            self.quiet_ms = 0;
        } else {
            self.quiet_ms = self.quiet_ms.saturating_add(WINDOW_MS);
            if self.quiet_ms >= self.required_quiet_ms {
                self.burst_ms = 0;
            }
        }
    }

    pub fn quiet_before_burst_ms(&self) -> u32 {
        self.quiet_before_burst_ms
    }

    pub fn burst_ms(&self) -> u32 {
        self.burst_ms
    }

    /// Whether a trigger arriving now begins a fresh piece of speech.
    pub fn satisfied(&self) -> bool {
        self.quiet_before_burst_ms >= self.required_quiet_ms && self.burst_ms <= self.max_burst_ms
    }

    pub fn reset(&mut self) {
        self.quiet_ms = self.required_quiet_ms;
        self.burst_ms = 0;
        self.quiet_before_burst_ms = self.required_quiet_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn confirms(text: &str) -> bool {
        transcript_confirms(text, &DEFAULT_SPELLINGS)
    }

    #[test]
    fn the_wake_word_on_its_own_is_accepted() {
        assert!(confirms("Нана, включи свет"));
        assert!(confirms("нана"));
        assert!(confirms("Окей, Нана!"));
        assert!(confirms("nana what is the weather"));
    }

    #[test]
    fn words_that_merely_contain_it_are_rejected() {
        for text in ["купил банана", "порезал ананас", "нанайцы", "нанометр", "бананами"]
        {
            assert!(!confirms(text), "{text:?} should not trigger");
        }
    }

    #[test]
    fn running_speech_that_sounds_similar_is_rejected() {
        // The recogniser resolves these into separate words; an acoustic
        // score on its own cannot.
        assert!(!confirms("она на столе"));
        assert!(!confirms("дана команда"));
        assert!(!confirms("Анна пришла"));
    }

    /// 300 ms of quiet required; a detection may sit at most 900 ms into a burst.
    fn gate() -> PreTriggerSilence {
        PreTriggerSilence::new(300, 900, 0.5)
    }

    fn feed(gate: &mut PreTriggerSilence, probability: f32, windows: usize) {
        for _ in 0..windows {
            gate.push(probability);
        }
    }

    #[test]
    fn addressing_the_assistant_after_a_pause_passes() {
        let mut gate = gate();
        feed(&mut gate, 0.0, 30); // quiet room
        feed(&mut gate, 0.9, 16); // "nana" itself, about 500 ms
        assert!(gate.satisfied(), "a pause preceded the wake word");
    }

    #[test]
    fn the_word_buried_in_a_sentence_is_rejected() {
        let mut gate = gate();
        feed(&mut gate, 0.0, 30); // the pause before the sentence started
        feed(&mut gate, 0.9, 60); // "я купил банана" - continuous speech
        assert!(!gate.satisfied(), "the pause belonged to the start of the sentence, not the word");
    }

    #[test]
    fn a_pause_shorter_than_required_does_not_clear_the_gate() {
        let mut gate = gate();
        feed(&mut gate, 0.9, 20);
        feed(&mut gate, 0.0, 8); // 256 ms, under the 300 ms required
        feed(&mut gate, 0.9, 16); // then the wake word
        assert!(!gate.satisfied(), "quiet was {} ms", gate.quiet_before_burst_ms());
    }
}
