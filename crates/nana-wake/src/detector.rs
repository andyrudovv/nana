//! Wake-word detection on top of the shared feature chain.

use std::path::Path;

use nana_core::{Error, Result};
use ort::session::Session;
use ort::value::Tensor;

use crate::frontend::{Frontend, EMBED_DIM, STEP_FRAMES, WINDOW_EMBEDDINGS};

/// Milliseconds one embedding step advances.
pub const STEP_MS: usize = STEP_FRAMES * 10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub score: f32,
    /// Milliseconds of audio seen when the detector fired.
    pub at_ms: usize,
}

pub struct WakeWord {
    frontend: Frontend,
    classifier: Session,
    input: String,
    output: String,
    threshold: f32,
    /// Steps to stay quiet after firing, so one utterance triggers once.
    cooldown_steps: usize,
    quiet_for: usize,
    steps: usize,
}

impl WakeWord {
    pub fn load(models_dir: impl AsRef<Path>, classifier: impl AsRef<Path>) -> Result<Self> {
        let path = classifier.as_ref();
        if !path.exists() {
            return Err(Error::ModelMissing(path.to_path_buf()));
        }
        let builder = Session::builder().map_err(to_err)?;
        let mut builder = builder.with_intra_threads(1).map_err(to_err)?;
        let session = builder
            .commit_from_file(path)
            .map_err(|e| Error::Vad(format!("loading {}: {e}", path.display())))?;

        // Names differ between released models ("x.1"/"53" vs
        // "onnx::Flatten_0"/"13"), so take them from the graph.
        let input = session
            .inputs()
            .first()
            .ok_or_else(|| Error::Vad("classifier has no input".into()))?
            .name()
            .to_string();
        let output = session
            .outputs()
            .first()
            .ok_or_else(|| Error::Vad("classifier has no output".into()))?
            .name()
            .to_string();

        Ok(Self {
            frontend: Frontend::load(models_dir)?,
            classifier: session,
            input,
            output,
            threshold: 0.5,
            cooldown_steps: 1_000 / STEP_MS,
            quiet_for: 0,
            steps: 0,
        })
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn reset(&mut self) {
        self.frontend.reset();
        self.quiet_for = 0;
        self.steps = 0;
    }

    /// Feeds audio and returns the score of every completed step, plus a
    /// detection when one crosses the threshold outside the cooldown.
    pub fn push(&mut self, samples: &[i16]) -> Result<(Vec<f32>, Option<Detection>)> {
        let produced = self.frontend.push(samples)?;
        let mut scores = Vec::with_capacity(produced);
        let mut detection = None;

        for _ in 0..produced {
            self.steps += 1;
            let Some(window) = self.frontend.window() else { continue };
            let score = self.score(window)?;
            scores.push(score);

            self.quiet_for = self.quiet_for.saturating_sub(1);
            if score >= self.threshold && self.quiet_for == 0 {
                self.quiet_for = self.cooldown_steps;
                detection.get_or_insert(Detection { score, at_ms: self.steps * STEP_MS });
            }
        }
        Ok((scores, detection))
    }

    fn score(&mut self, window: Vec<f32>) -> Result<f32> {
        let input = Tensor::from_array(([1_usize, WINDOW_EMBEDDINGS, EMBED_DIM], window))
            .map_err(to_err)?;
        let out =
            self.classifier.run(ort::inputs![self.input.as_str() => input]).map_err(to_err)?;
        let (_, score) = out[self.output.as_str()].try_extract_tensor::<f32>().map_err(to_err)?;
        score.first().copied().ok_or_else(|| Error::Vad("classifier returned nothing".into()))
    }
}

fn to_err<E: std::fmt::Display>(e: E) -> Error {
    Error::Vad(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::models_dir;

    #[test]
    fn cooldown_covers_a_whole_utterance() {
        assert_eq!(STEP_MS, 80);
        assert_eq!(1_000 / STEP_MS, 12, "a second of quiet after firing");
    }

    #[test]
    fn missing_classifier_reports_the_path() {
        let dir = models_dir();
        let Err(err) = WakeWord::load(&dir, dir.join("nope.onnx")) else {
            panic!("a missing classifier should fail");
        };
        assert!(matches!(err, Error::ModelMissing(_)), "got {err:?}");
    }

    #[test]
    fn silence_does_not_trigger_a_pretrained_model() {
        let dir = models_dir();
        let Ok(mut wake) = WakeWord::load(&dir, dir.join("hey_jarvis_v0.1.onnx")) else {
            eprintln!("skipping: {} is missing", dir.display());
            return;
        };
        let (scores, detection) = wake.push(&vec![0_i16; 16_000 * 3]).expect("push");
        assert!(!scores.is_empty(), "three seconds should yield steps");
        assert!(detection.is_none(), "silence fired the detector");
        let peak = scores.iter().cloned().fold(f32::MIN, f32::max);
        assert!(peak < 0.1, "silence scored {peak}");
    }
}
