//! Wake-word detection through rustpotter.
//!
//! Chosen over training an openWakeWord classifier because a wakeword
//! *reference* is built from a handful of recordings of the actual phrase.
//! No synthetic corpus, no GPU training run, and "Хей, Нана" works the same
//! day it is recorded.

use std::path::Path;

use nana_core::{Error, Result};
use rustpotter::{
    Rustpotter, RustpotterConfig, SampleFormat, ScoreMode, WakewordRef, WakewordRefBuildFromFiles,
    WakewordSave,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub name: String,
    pub score: f32,
    pub avg_score: f32,
    /// Milliseconds of audio seen when the detector fired.
    pub at_ms: usize,
}

pub struct Spotter {
    inner: Rustpotter,
    /// rustpotter only accepts exact frames, so partial ones wait here.
    pending: Vec<i16>,
    frame: usize,
    consumed: usize,
}

impl Spotter {
    /// `threshold` is the similarity a live window must reach. Start near 0.5
    /// and tune on recordings of the real room.
    pub fn load(wakeword: impl AsRef<Path>, threshold: f32) -> Result<Self> {
        let path = wakeword.as_ref();
        if !path.exists() {
            return Err(Error::ModelMissing(path.to_path_buf()));
        }
        let mut config = RustpotterConfig::default();
        config.fmt.sample_rate = nana_core::audio::SAMPLE_RATE as usize;
        config.fmt.sample_format = SampleFormat::I16;
        config.fmt.channels = 1;
        config.detector.threshold = threshold;
        // Averaged score is a cheap pre-filter; keep it below the real
        // threshold so it only rejects the clearly unrelated.
        config.detector.avg_threshold = (threshold - 0.2).max(0.0);
        config.detector.min_scores = 3;
        config.detector.score_mode = ScoreMode::Max;

        let mut inner = Rustpotter::new(&config).map_err(Error::Wake)?;
        inner
            .add_wakeword_from_file("nana", &path.to_string_lossy())
            .map_err(|e| Error::Wake(format!("loading {}: {e}", path.display())))?;
        let frame = inner.get_samples_per_frame();
        Ok(Self { inner, pending: Vec::with_capacity(frame * 2), frame, consumed: 0 })
    }

    pub fn reset(&mut self) {
        self.inner.reset();
        self.pending.clear();
        self.consumed = 0;
    }

    /// Feeds audio of any length and returns the first detection it completes.
    pub fn push(&mut self, samples: &[i16]) -> Option<Detection> {
        self.pending.extend_from_slice(samples);
        let mut found = None;
        while self.pending.len() >= self.frame {
            let frame: Vec<i16> = self.pending.drain(..self.frame).collect();
            self.consumed += self.frame;
            if let Some(d) = self.inner.process_samples(frame) {
                found.get_or_insert(Detection {
                    name: d.name,
                    score: d.score,
                    avg_score: d.avg_score,
                    at_ms: self.consumed * 1000 / nana_core::audio::SAMPLE_RATE as usize,
                });
            }
        }
        found
    }

    /// Partial detection score, for tuning the threshold against real audio.
    pub fn partial_score(&self) -> Option<f32> {
        self.inner.get_partial_detection().map(|d| d.score)
    }
}

/// Builds a wakeword reference from recordings of the phrase.
///
/// rustpotter wants 3 to 8 of them; more records cost proportionally more CPU
/// per frame, since each one is compared separately.
pub fn build_reference(
    name: &str,
    samples: &[impl AsRef<Path>],
    out: impl AsRef<Path>,
) -> Result<usize> {
    if samples.len() < 3 {
        return Err(Error::Wake(format!("need at least 3 recordings, got {}", samples.len())));
    }
    let paths: Vec<String> =
        samples.iter().map(|p| p.as_ref().to_string_lossy().into_owned()).collect();
    let wakeword = WakewordRef::new_from_sample_files(name.to_string(), None, None, paths, 16)
        .map_err(Error::Wake)?;
    wakeword.save_to_file(&out.as_ref().to_string_lossy()).map_err(Error::Wake)?;
    Ok(samples.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_wakeword_reports_the_path() {
        let Err(err) = Spotter::load("models/wake/nope.rpw", 0.5) else {
            panic!("a missing wakeword should fail");
        };
        assert!(matches!(err, Error::ModelMissing(_)), "got {err:?}");
    }

    #[test]
    fn too_few_recordings_is_refused_with_a_useful_message() {
        let Err(err) = build_reference("nana", &["a.wav", "b.wav"], "out.rpw") else {
            panic!("two recordings should be refused");
        };
        assert!(err.to_string().contains("at least 3"), "got {err}");
    }
}
