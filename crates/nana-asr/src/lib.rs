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
    audio_ctx: AudioCtx,
    bounded_latency: bool,
    max_tokens: i32,
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
            audio_ctx: AudioCtx::Auto,
            bounded_latency: true,
            // A spoken command is far shorter than this; the cap only ever
            // truncates a decoder that has stopped making progress.
            max_tokens: 96,
        })
    }

    pub fn with_threads(mut self, threads: i32) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Overrides how much encoder context to use. [`AudioCtx::Auto`] is the
    /// default and is what makes recognition fit the latency budget.
    pub fn with_audio_ctx(mut self, audio_ctx: AudioCtx) -> Self {
        self.audio_ctx = audio_ctx;
        self
    }

    /// Primes the model so the first real utterance does not pay cold start.
    ///
    /// Deliberately warms at the full context. Growing `audio_ctx` beyond what
    /// a state has already seen corrupts that call — warming at 186 then asking
    /// for 403 returned empty text tagged as English — while shrinking it is
    /// safe. Priming at the maximum makes every later size a shrink.
    pub fn warm_up(&self) -> Result<()> {
        let silence = vec![0_i16; nana_core::audio::SAMPLE_RATE as usize];
        self.decode(&silence, AudioCtx::Full).map(|_| ())
    }

    /// Allows whisper's temperature fallback back on, trading the latency
    /// ceiling for accuracy on hard audio. Off by default.
    pub fn with_temperature_fallback(mut self) -> Self {
        self.bounded_latency = false;
        self
    }
}

impl Asr for WhisperAsr {
    fn transcribe(&self, pcm: &[i16]) -> Result<Transcript> {
        self.decode(pcm, self.audio_ctx)
    }

    fn model(&self) -> &str {
        &self.model
    }
}

impl WhisperAsr {
    fn decode(&self, pcm: &[i16], audio_ctx: AudioCtx) -> Result<Transcript> {
        let audio = to_f32(pcm);
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.threads);
        let seconds = pcm.len() as f32 / nana_core::audio::SAMPLE_RATE as f32;
        if let Some(ctx) = audio_ctx.frames(seconds) {
            params.set_audio_ctx(ctx);
        }
        // "auto" makes whisper detect the language while decoding; the separate
        // detect_language flag stops after detection and returns no text.
        params.set_language(Some("auto"));
        params.set_translate(false);
        // Each turn stands alone. Left on, whisper conditions the decode on the
        // previous utterance's tokens: a warm-up on silence was enough to make
        // the next real utterance come back empty and tagged as English.
        params.set_no_context(true);
        params.set_no_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        if self.bounded_latency {
            // Two unbounded paths: whisper retries at rising temperatures when
            // decoding looks bad, and a repetition loop can run until the
            // segment is full. Both are capped rather than trusted.
            params.set_temperature_inc(0.0);
            params.set_max_tokens(self.max_tokens);
        }

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
}

/// How much of the encoder's 30 s window to actually run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioCtx {
    /// Sized to the utterance. Measured 9x faster than [`Full`](AudioCtx::Full)
    /// on a 1.1 s clip and 4.5x on a 5.9 s one, with the same transcript.
    #[default]
    Auto,
    /// The whole padded window, as whisper.cpp does by default.
    Full,
    Fixed(i32),
}

impl AudioCtx {
    pub fn frames(self, seconds: f32) -> Option<i32> {
        match self {
            AudioCtx::Auto => Some(audio_ctx_for(seconds)),
            AudioCtx::Full => None,
            AudioCtx::Fixed(frames) => Some(frames),
        }
    }
}

/// Encoder frames covering `seconds` of audio, with headroom.
///
/// The encoder runs at 50 frames per second after downsampling; the default of
/// 1500 covers the full 30 s window.
/// Below the real frame count the decoder collapses into repetition and gets
/// *slower*, not just worse: 64 frames on a 1.1 s clip took 3123 ms against
/// 614 ms at 119. Above it, measured time is flat, so headroom is free.
pub fn audio_ctx_for(seconds: f32) -> i32 {
    const FRAMES_PER_SECOND: f32 = 50.0;
    const HEADROOM: f32 = 1.15;
    const MARGIN: f32 = 64.0;
    let frames = seconds * FRAMES_PER_SECOND;
    ((frames * HEADROOM + MARGIN).ceil() as i32).clamp(128, 1500)
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
    fn audio_ctx_scales_with_the_utterance() {
        // Each value must clear the clip's real frame count (seconds * 50).
        for seconds in [1.0_f32, 2.5, 5.9, 12.0] {
            let frames = (seconds * 50.0) as i32;
            assert!(audio_ctx_for(seconds) > frames, "{seconds}s got too small a context");
        }
        assert!(audio_ctx_for(4.0) < audio_ctx_for(10.0));
    }

    #[test]
    fn audio_ctx_never_exceeds_the_full_window() {
        assert_eq!(audio_ctx_for(60.0), 1500, "30 s is all the encoder has");
        assert_eq!(audio_ctx_for(0.1), 128, "measured: below ~96 short clips collapse");
    }

    #[test]
    fn threads_never_drop_below_one() {
        assert!(default_threads() >= 1);
    }
}
