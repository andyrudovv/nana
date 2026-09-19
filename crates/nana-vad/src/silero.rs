//! Silero VAD v5 through ONNX Runtime.

use std::path::{Path, PathBuf};

use nana_core::audio::{SAMPLE_RATE, VAD_WINDOW_SAMPLES};
use nana_core::{Error, Result};
use ort::session::Session;
use ort::value::Tensor;

/// Recurrent state shape is [2, batch, 128]; batch is always 1 here.
const STATE_LEN: usize = 2 * 128;

pub struct SileroVad {
    session: Session,
    /// Carried between calls; the model is recurrent, so dropping this would
    /// restart its notion of the conversation on every window.
    state: Vec<f32>,
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
        Ok(Self { session, state: vec![0.0; STATE_LEN] })
    }

    /// Clears the recurrent state. Call between utterances so one turn's tail
    /// cannot bias the start of the next.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
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

        let to_vad = |e: ort::Error| Error::Vad(e.to_string());
        let input = Tensor::from_array(([1_usize, VAD_WINDOW_SAMPLES], audio)).map_err(to_vad)?;
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
        probability.first().copied().ok_or_else(|| Error::Vad("empty output".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_reports_the_path() {
        let Err(err) = SileroVad::load("models/no-such-vad.onnx") else {
            panic!("loading a missing model should fail");
        };
        assert!(matches!(err, Error::ModelMissing(_)), "got {err:?}");
    }
}
