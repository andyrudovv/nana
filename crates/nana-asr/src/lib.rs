//! Speech recognition through whisper.cpp.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use nana_core::lang::Lang;
use nana_core::ports::{Asr, Transcript};
use nana_core::{Error, Result};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

/// Routes whisper.cpp's own stdout chatter into tracing. Safe to call twice.
pub fn install_logging() {
    whisper_rs::install_logging_hooks();
}

pub struct WhisperAsr {
    /// `WhisperState` is Send but not Sync, and it holds its own reference to
    /// the model, so one mutex-guarded state satisfies the port without
    /// reloading weights per utterance.
    state: Mutex<WhisperState>,
    model: String,
    threads: i32,
}

impl WhisperAsr {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let path = model_path.as_ref();
        if !path.exists() {
            return Err(Error::ModelMissing(PathBuf::from(path)));
        }
        let context = WhisperContext::new_with_params(path, WhisperContextParameters::default())
            .map_err(|e| Error::Asr(format!("loading {}: {e}", path.display())))?;
        let state = context.create_state().map_err(|e| Error::Asr(e.to_string()))?;
        Ok(Self {
            state: Mutex::new(state),
            model: path.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
            threads: default_threads(),
        })
    }

    pub fn with_threads(mut self, threads: i32) -> Self {
        self.threads = threads.max(1);
        self
    }
}

impl Asr for WhisperAsr {
    fn transcribe(&self, pcm: &[i16]) -> Result<Transcript> {
        let audio = to_f32(pcm);
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.threads);
        // "auto" makes whisper detect the language while decoding; the separate
        // detect_language flag stops after detection and returns no text.
        params.set_language(Some("auto"));
        params.set_translate(false);
        params.set_no_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        let mut state = self.state.lock().map_err(|_| Error::Asr("state poisoned".into()))?;
        state.full(params, &audio).map_err(|e| Error::Asr(e.to_string()))?;

        let mut text = String::new();
        let mut prob_sum = 0.0f32;
        let mut prob_count = 0usize;
        let mut no_speech = 0.0f32;
        for i in 0..state.full_n_segments() {
            let Some(segment) = state.get_segment(i) else { continue };
            if let Ok(chunk) = segment.to_str_lossy() {
                text.push_str(&chunk);
            }
            no_speech = no_speech.max(segment.no_speech_probability());
            for t in 0..segment.n_tokens() {
                if let Some(token) = segment.get_token(t) {
                    prob_sum += token.token_probability();
                    prob_count += 1;
                }
            }
        }

        let lang = whisper_rs::get_lang_str(state.full_lang_id_from_state()).and_then(Lang::parse);
        Ok(Transcript {
            text: text.trim().to_string(),
            lang,
            confidence: if prob_count == 0 { 0.0 } else { prob_sum / prob_count as f32 },
            no_speech,
        })
    }

    fn model(&self) -> &str {
        &self.model
    }
}

/// whisper.cpp wants normalised floats; capture hands us i16.
fn to_f32(pcm: &[i16]) -> Vec<f32> {
    pcm.iter().map(|s| *s as f32 / i16::MAX as f32).collect()
}

fn default_threads() -> i32 {
    std::thread::available_parallelism().map(|n| n.get().min(8) as i32).unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_scale_ints_map_into_unit_range() {
        let out = to_f32(&[i16::MAX, 0, -i16::MAX]);
        assert_eq!(out, vec![1.0, 0.0, -1.0]);
    }

    #[test]
    fn missing_model_reports_the_path_instead_of_panicking() {
        let Err(err) = WhisperAsr::load("models/does-not-exist.bin") else {
            panic!("loading a missing model should fail");
        };
        assert!(matches!(err, Error::ModelMissing(_)), "got {err:?}");
    }

    #[test]
    fn threads_never_drop_below_one() {
        assert!(default_threads() >= 1);
    }
}
