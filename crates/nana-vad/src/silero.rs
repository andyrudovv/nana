//! Silero VAD v5 through ONNX Runtime.

use std::path::{Path, PathBuf};

use nana_core::audio::{SAMPLE_RATE, VAD_WINDOW_SAMPLES};
use nana_core::{Error, Result};
use ort::session::Session;
use ort::value::Tensor;

/// Recurrent state shape is [2, batch, 128]; batch is always 1 here.
const STATE_LEN: usize = 2 * 128;

/// Samples of the previous window the model expects in front of the new one.
///
/// Silero v5 is fed 64 + 512 samples at 16 kHz, not 512. The graph declares a
/// dynamic input shape, so feeding it 512 alone does not fail — it quietly
/// returns near-zero for speech, which is how this was missed the first time.
const CONTEXT_SAMPLES: usize = 64;

pub struct SileroVad {
    session: Session,
    /// Carried between calls; the model is recurrent, so dropping this would
    /// restart its notion of the conversation on every window.
    state: Vec<f32>,
    /// Tail of the previous window, prepended to the next one.
    context: Vec<f32>,
}

impl SileroVad {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(Error::ModelMissing(PathBuf::from(path)));
        }
        let vad_error =
            |e: &dyn std::fmt::Display| Error::Vad(format!("loading {}: {e}", path.display()));
        // The builder's error type carries the builder back, so these cannot be
        // chained with `?` against one error type.
        let builder = Session::builder().map_err(|e| vad_error(&e))?;
        let mut builder = builder.with_intra_threads(1).map_err(|e| vad_error(&e))?;
        let session = builder.commit_from_file(path).map_err(|e| vad_error(&e))?;
        Ok(Self { session, state: vec![0.0; STATE_LEN], context: vec![0.0; CONTEXT_SAMPLES] })
    }

    /// Clears the recurrent state. Call between utterances so one turn's tail
    /// cannot bias the start of the next.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
    }

    /// Speech probability for exactly one [`VAD_WINDOW_SAMPLES`] window.
    pub fn probe(&mut self, window: &[i16]) -> Result<f32> {
        if window.len() != VAD_WINDOW_SAMPLES {
            return Err(Error::Vad(format!(
                "expected {VAD_WINDOW_SAMPLES} samples, got {}",
                window.len()
            )));
        }
        let audio: Vec<f32> = window.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
        let framed = with_context(&self.context, &audio);

        let to_vad = |e: ort::Error| Error::Vad(e.to_string());
        let input = Tensor::from_array(([1_usize, framed.len()], framed)).map_err(to_vad)?;
        let state = Tensor::from_array(([2_usize, 1_usize, 128_usize], self.state.clone()))
            .map_err(to_vad)?;
        let rate = Tensor::from_array(([1_usize], vec![SAMPLE_RATE as i64])).map_err(to_vad)?;

        let outputs = self
            .session
            .run(ort::inputs!["input" => input, "state" => state, "sr" => rate])
            .map_err(|e| Error::Vad(e.to_string()))?;

        let (_, next_state) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Vad(format!("reading stateN: {e}")))?;
        self.state.copy_from_slice(next_state);

        let (_, probability) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Vad(format!("reading output: {e}")))?;
        let probability =
            probability.first().copied().ok_or_else(|| Error::Vad("empty output".into()))?;

        self.context.copy_from_slice(&audio[audio.len() - CONTEXT_SAMPLES..]);
        Ok(probability)
    }
}

/// Builds the model input: previous tail first, then the new window.
fn with_context(context: &[f32], window: &[f32]) -> Vec<f32> {
    let mut framed = Vec::with_capacity(context.len() + window.len());
    framed.extend_from_slice(context);
    framed.extend_from_slice(window);
    framed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_input_carries_the_previous_tail_in_front() {
        let context: Vec<f32> = vec![0.25; CONTEXT_SAMPLES];
        let window: Vec<f32> = (0..VAD_WINDOW_SAMPLES).map(|i| i as f32).collect();
        let framed = with_context(&context, &window);

        assert_eq!(framed.len(), CONTEXT_SAMPLES + VAD_WINDOW_SAMPLES, "16 kHz wants 576");
        assert_eq!(framed[CONTEXT_SAMPLES - 1], 0.25, "context sits in front");
        assert_eq!(framed[CONTEXT_SAMPLES], 0.0, "the new window follows it");
    }

    #[test]
    fn a_window_of_the_wrong_length_is_rejected_rather_than_guessed() {
        let Ok(mut vad) = SileroVad::load("models/silero_vad.onnx") else {
            return; // model is not committed; nothing to check here
        };
        let Err(err) = vad.probe(&[0; VAD_WINDOW_SAMPLES - 1]) else {
            panic!("a short window should be rejected");
        };
        assert!(matches!(err, Error::Vad(_)), "got {err:?}");
    }

    #[test]
    fn silence_scores_far_below_the_speech_threshold() {
        let Ok(mut vad) = SileroVad::load("models/silero_vad.onnx") else {
            return;
        };
        for _ in 0..10 {
            let p = vad.probe(&[0; VAD_WINDOW_SAMPLES]).expect("probe");
            assert!(p < 0.1, "silence scored {p}");
        }
    }

    #[test]
    fn missing_model_reports_the_path() {
        let Err(err) = SileroVad::load("models/no-such-vad.onnx") else {
            panic!("loading a missing model should fail");
        };
        assert!(matches!(err, Error::ModelMissing(_)), "got {err:?}");
    }
}
