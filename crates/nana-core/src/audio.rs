//! Pipeline-wide audio constants and format description.

/// Silero VAD, whisper.cpp and the wake-word models are all trained at 16 kHz.
/// Everything downstream of capture assumes this rate; resampling happens once.
pub const SAMPLE_RATE: u32 = 16_000;

/// 20 ms at [`SAMPLE_RATE`] — the frame size WebRTC's AEC expects.
pub const FRAME_SAMPLES: usize = 320;

/// Silero v5 accepts exactly this window at 16 kHz and nothing else.
pub const VAD_WINDOW_SAMPLES: usize = 512;

/// How much audio the pre-roll ring keeps. Covers the wake word plus the
/// command spoken straight after it.
pub const PREROLL_SECONDS: f32 = 3.0;

/// Handed to ASR ahead of the detection instant: the detector fires 200-500 ms
/// late, and users run the command into the wake word without pausing.
pub const PREROLL_LOOKBACK_SECONDS: f32 = 1.5;

pub const fn seconds_to_samples(seconds: f32) -> usize {
    (SAMPLE_RATE as f32 * seconds) as usize
}

/// What a capture device actually gave us, before conversion to the pipeline format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

impl SourceFormat {
    /// True when the device speaks the pipeline format natively and no
    /// resampling or downmixing stage is needed.
    pub fn is_pipeline_native(&self) -> bool {
        self.sample_rate == SAMPLE_RATE && self.channels == 1
    }
}
