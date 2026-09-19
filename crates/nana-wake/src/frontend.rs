//! Shared openWakeWord feature chain: audio -> mel -> 96-dim embeddings.
//!
//! Both the pretrained classifiers and few-shot templates consume the same
//! embeddings, so this runs once regardless of how detection is decided.

use std::collections::VecDeque;
use std::path::Path;

use nana_core::{Error, Result};
use ort::session::Session;
use ort::value::Tensor;

/// Mel hop at 16 kHz: one frame per 10 ms.
pub const MEL_HOP: usize = 160;
/// Frames the graph drops to padding — measured, not assumed: feeding S
/// samples returns S/160 - 3 frames.
pub const MEL_PADDING_FRAMES: usize = 3;
/// Mel frames the embedding model consumes, i.e. a 760 ms window.
pub const EMBED_FRAMES: usize = 76;
/// How far the embedding window advances per step: 80 ms.
pub const STEP_FRAMES: usize = 8;
pub const STEP_SAMPLES: usize = STEP_FRAMES * MEL_HOP;
/// Embeddings a classifier window holds, spanning roughly 2 s of audio.
pub const WINDOW_EMBEDDINGS: usize = 16;
pub const EMBED_DIM: usize = 96;
const MEL_BINS: usize = 32;

/// Samples to feed the mel graph to get exactly `frames` new frames back.
pub const fn samples_for_frames(frames: usize) -> usize {
    (frames + MEL_PADDING_FRAMES) * MEL_HOP
}

/// Samples of history the mel graph needs in front of new audio.
pub const MEL_CONTEXT: usize = MEL_PADDING_FRAMES * MEL_HOP;

pub struct Frontend {
    mel: Session,
    embed: Session,
    /// Audio not yet turned into mel frames, preceded by the context tail.
    raw: Vec<f32>,
    /// How far into `raw` the mel stage has already consumed.
    consumed: usize,
    mel_frames: VecDeque<f32>,
    embeddings: VecDeque<f32>,
}

impl Frontend {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        Ok(Self {
            mel: open(&dir.join("melspectrogram.onnx"))?,
            embed: open(&dir.join("embedding_model.onnx"))?,
            raw: Vec::new(),
            consumed: 0,
            mel_frames: VecDeque::new(),
            embeddings: VecDeque::new(),
        })
    }

    pub fn reset(&mut self) {
        self.raw.clear();
        self.consumed = 0;
        self.mel_frames.clear();
        self.embeddings.clear();
    }

    /// Feeds audio and returns how many new embeddings became available.
    ///
    /// Chunk size is irrelevant to the result: steps are taken on sample
    /// counts, not on call boundaries.
    pub fn push(&mut self, samples: &[i16]) -> Result<usize> {
        self.raw.extend(samples.iter().map(|s| *s as f32 / i16::MAX as f32));
        // The graph cannot produce frames for the first three hops, so skip
        // past them once before any stepping can happen.
        if self.consumed < MEL_CONTEXT && self.raw.len() >= MEL_CONTEXT {
            self.consumed = MEL_CONTEXT;
        }

        let mut produced = 0;
        while self.consumed >= MEL_CONTEXT && self.raw.len() >= self.consumed + STEP_SAMPLES {
            let window = &self.raw[self.consumed - MEL_CONTEXT..self.consumed + STEP_SAMPLES];
            let frames = mel_frames(&mut self.mel, window)?;
            self.consumed += STEP_SAMPLES;

            self.mel_frames.extend(frames);
            let keep = EMBED_FRAMES * MEL_BINS;
            while self.mel_frames.len() > keep {
                self.mel_frames.pop_front();
            }
            if self.mel_frames.len() == keep {
                self.embed_latest()?;
                produced += 1;
            }
            self.discard_consumed_audio();
        }
        Ok(produced)
    }

    /// Keeps only what a future step still needs as context.
    fn discard_consumed_audio(&mut self) {
        let keep_from = self.consumed.saturating_sub(MEL_CONTEXT);
        if keep_from > 0 {
            self.raw.drain(..keep_from);
            self.consumed -= keep_from;
        }
    }

    fn embed_latest(&mut self) -> Result<()> {
        let frames: Vec<f32> = self.mel_frames.iter().copied().collect();
        let input = Tensor::from_array(([1_usize, EMBED_FRAMES, MEL_BINS, 1_usize], frames))
            .map_err(vad_err)?;
        let out = self.embed.run(ort::inputs!["input_1" => input]).map_err(vad_err)?;
        let (_, embedding) = out["conv2d_19"].try_extract_tensor::<f32>().map_err(vad_err)?;

        self.embeddings.extend(embedding.iter().copied());
        while self.embeddings.len() > WINDOW_EMBEDDINGS * EMBED_DIM {
            self.embeddings.pop_front();
        }
        Ok(())
    }

    /// The most recent classifier window, or `None` until enough audio has
    /// passed through (about two seconds).
    pub fn window(&self) -> Option<Vec<f32>> {
        (self.embeddings.len() == WINDOW_EMBEDDINGS * EMBED_DIM)
            .then(|| self.embeddings.iter().copied().collect())
    }
}

/// One mel step. openWakeWord scales the graph output before the embedding
/// model; the constants come from its reference implementation.
fn mel_frames(mel: &mut Session, window: &[f32]) -> Result<Vec<f32>> {
    let input = Tensor::from_array(([1_usize, window.len()], window.to_vec())).map_err(vad_err)?;
    let out = mel.run(ort::inputs!["input" => input]).map_err(vad_err)?;
    let (_, frames) = out["output"].try_extract_tensor::<f32>().map_err(vad_err)?;
    Ok(frames.iter().map(|v| v / 10.0 + 2.0).collect())
}

fn open(path: &Path) -> Result<Session> {
    if !path.exists() {
        return Err(Error::ModelMissing(path.to_path_buf()));
    }
    let builder = Session::builder().map_err(|e| Error::Vad(e.to_string()))?;
    let mut builder = builder.with_intra_threads(1).map_err(|e| Error::Vad(e.to_string()))?;
    builder
        .commit_from_file(path)
        .map_err(|e| Error::Vad(format!("loading {}: {e}", path.display())))
}

fn vad_err(e: ort::Error) -> Error {
    Error::Vad(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::models_dir;

    #[test]
    fn frame_arithmetic_matches_the_measured_graph() {
        // Measured: 1280 -> 5 frames, 2560 -> 13, 12800 -> 77, 16000 -> 97.
        for (samples, frames) in [(1280, 5), (2560, 13), (12800, 77), (16000, 97)] {
            assert_eq!(samples / MEL_HOP - MEL_PADDING_FRAMES, frames, "{samples} samples");
        }
    }

    #[test]
    fn a_step_asks_for_its_own_padding_context() {
        assert_eq!(samples_for_frames(STEP_FRAMES), 1760);
        assert_eq!(samples_for_frames(STEP_FRAMES) / MEL_HOP - MEL_PADDING_FRAMES, STEP_FRAMES);
    }

    #[test]
    fn chunk_size_does_not_change_the_result() {
        let Ok(mut a) = Frontend::load(models_dir()) else {
            eprintln!("skipping: {} is missing", models_dir().display());
            return;
        };
        let mut b = Frontend::load(models_dir()).expect("second frontend");
        let audio: Vec<i16> =
            (0..32_000).map(|i| ((i as f32 / 11.0).sin() * 6000.0) as i16).collect();

        let mut from_one_push = 0;
        from_one_push += a.push(&audio).expect("push all at once");

        let mut from_many = 0;
        for chunk in audio.chunks(317) {
            from_many += b.push(chunk).expect("push in odd chunks");
        }

        assert!(from_one_push > 0, "two seconds of audio produced no embeddings");
        assert_eq!(from_one_push, from_many, "embedding count depends on chunking");
        assert_eq!(a.window(), b.window(), "window contents depend on chunking");
    }

    #[test]
    fn the_classifier_window_spans_about_two_seconds() {
        let span_ms = (WINDOW_EMBEDDINGS - 1) * STEP_FRAMES * 10 + EMBED_FRAMES * 10;
        assert_eq!(span_ms, 1960);
    }
}
