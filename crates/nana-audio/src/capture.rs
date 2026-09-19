//! Microphone capture into the pre-roll ring.
//!
//! Opens whatever the OS currently calls the default input, so the stream
//! follows the user's device choice instead of pinning one (R10).

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig, SupportedStreamConfig};
use nana_core::audio::{seconds_to_samples, SourceFormat, PREROLL_SECONDS, SAMPLE_RATE};
use nana_core::{Error, Result};

use crate::ring::PreRoll;

/// Shared with the audio callback. A mutex is enough here: the callback holds
/// it only for a memcpy of one buffer, fifty times a second.
pub type SharedRing = Arc<Mutex<PreRoll>>;

pub struct Capture {
    /// Dropping the stream stops capture; kept alive for that reason alone.
    _stream: cpal::Stream,
    ring: SharedRing,
    format: SourceFormat,
}

impl Capture {
    /// Capture into a ring sized for wake-word pre-roll.
    pub fn start_default() -> Result<Self> {
        Self::start_default_with_capacity(seconds_to_samples(PREROLL_SECONDS))
    }

    /// Capture into a ring of an explicit size. Recording tools hold more than
    /// the pre-roll window, which would otherwise silently truncate them.
    pub fn start_default_with_capacity(capacity_samples: usize) -> Result<Self> {
        let device =
            cpal::default_host().default_input_device().ok_or(Error::NoDevice("default input"))?;
        Self::start(&device, Arc::new(Mutex::new(PreRoll::new(capacity_samples))))
    }

    pub fn start(device: &Device, ring: SharedRing) -> Result<Self> {
        let supported = pipeline_config(device)?;
        let format =
            SourceFormat { sample_rate: supported.sample_rate(), channels: supported.channels() };
        let config: StreamConfig = supported.config();
        let channels = format.channels as usize;

        let sink = Arc::clone(&ring);
        let on_error = |err| tracing::error!(%err, "capture stream error");

        let stream = match supported.sample_format() {
            SampleFormat::I16 => device.build_input_stream::<i16, _, _>(
                config,
                move |data, _| push_mono(&sink, data, channels, |s| s),
                on_error,
                None,
            ),
            SampleFormat::F32 => device.build_input_stream::<f32, _, _>(
                config,
                move |data, _| push_mono(&sink, data, channels, f32_to_i16),
                on_error,
                None,
            ),
            other => {
                return Err(Error::UnsupportedFormat(format!("sample format {other:?}")));
            }
        }
        .map_err(|e| Error::Stream(e.to_string()))?;

        stream.play().map_err(|e| Error::Stream(e.to_string()))?;
        Ok(Self { _stream: stream, ring, format })
    }

    pub fn format(&self) -> SourceFormat {
        self.format
    }

    pub fn ring(&self) -> &SharedRing {
        &self.ring
    }

    /// Most recent `count` samples. Used by pre-roll extraction and by tooling.
    pub fn tail(&self, count: usize) -> Vec<i16> {
        self.ring.lock().expect("ring mutex poisoned").tail(count)
    }

    /// Samples written since `cursor`, with the cursor to use next.
    ///
    /// Returns `None` when the window has already been overwritten: a
    /// real-time consumer has to know it fell behind rather than silently
    /// resume from wherever it lands.
    pub fn read_from(&self, cursor: u64) -> Option<(Vec<i16>, u64)> {
        let ring = self.ring.lock().expect("ring mutex poisoned");
        let available = ring.written().saturating_sub(cursor) as usize;
        let samples = ring.range(cursor, available)?;
        Some((samples, cursor + available as u64))
    }

    pub fn captured(&self) -> u64 {
        self.ring.lock().expect("ring mutex poisoned").written()
    }
}

/// Asks the device for the pipeline format directly.
///
/// PipeWire resamples well, so accepting 16 kHz mono from the server is both
/// simpler and better than carrying our own resampler.
fn pipeline_config(device: &Device) -> Result<SupportedStreamConfig> {
    let configs = device.supported_input_configs().map_err(|e| Error::Device(e.to_string()))?;
    let mut fallback = None;
    for range in configs {
        if range.min_sample_rate() > SAMPLE_RATE || range.max_sample_rate() < SAMPLE_RATE {
            continue;
        }
        if !matches!(range.sample_format(), SampleFormat::I16 | SampleFormat::F32) {
            continue;
        }
        match range.channels() {
            1 => return range.try_with_sample_rate(SAMPLE_RATE).ok_or_else(unsupported),
            _ if fallback.is_none() => fallback = range.try_with_sample_rate(SAMPLE_RATE),
            _ => {}
        }
    }
    fallback.ok_or_else(unsupported)
}

fn unsupported() -> Error {
    Error::UnsupportedFormat(format!("device cannot deliver {SAMPLE_RATE} Hz i16 or f32"))
}

fn push_mono<T: Copy>(ring: &SharedRing, data: &[T], channels: usize, convert: impl Fn(T) -> i16) {
    let mono: Vec<i16> = if channels <= 1 {
        data.iter().map(|s| convert(*s)).collect()
    } else {
        downmix(data, channels, convert)
    };
    ring.lock().expect("ring mutex poisoned").push(&mono);
}

fn downmix<T: Copy>(data: &[T], channels: usize, convert: impl Fn(T) -> i16) -> Vec<i16> {
    data.chunks_exact(channels)
        .map(|frame| {
            let sum: i32 = frame.iter().map(|s| convert(*s) as i32).sum();
            (sum / channels as i32) as i16
        })
        .collect()
}

fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

/// Peak and RMS of a block, for verifying a capture without listening to it.
pub fn levels(samples: &[i16]) -> (i16, f64) {
    let peak = samples.iter().map(|s| s.saturating_abs()).max().unwrap_or(0);
    let rms = if samples.is_empty() {
        0.0
    } else {
        let sum: f64 = samples.iter().map(|s| (*s as f64).powi(2)).sum();
        (sum / samples.len() as f64).sqrt()
    };
    (peak, rms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_scale_float_maps_to_full_scale_int() {
        assert_eq!(f32_to_i16(1.0), i16::MAX);
        assert_eq!(f32_to_i16(-1.0), -i16::MAX);
        assert_eq!(f32_to_i16(0.0), 0);
    }

    #[test]
    fn out_of_range_float_clamps_instead_of_wrapping() {
        assert_eq!(f32_to_i16(4.2), i16::MAX);
        assert_eq!(f32_to_i16(-4.2), -i16::MAX);
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        let interleaved: [i16; 6] = [100, 300, -200, 0, 1000, 1000];
        assert_eq!(downmix(&interleaved, 2, |s| s), vec![200, -100, 1000]);
    }

    #[test]
    fn partial_trailing_frame_is_dropped_rather_than_skewed() {
        let interleaved: [i16; 5] = [100, 300, -200, 0, 999];
        assert_eq!(downmix(&interleaved, 2, |s| s), vec![200, -100]);
    }

    #[test]
    fn levels_of_silence_are_zero() {
        assert_eq!(levels(&[0; 64]), (0, 0.0));
        assert_eq!(levels(&[]), (0, 0.0));
    }

    #[test]
    fn levels_survive_the_most_negative_sample() {
        // i16::MIN.abs() overflows; saturating_abs is why this passes.
        let (peak, _) = levels(&[i16::MIN, 5]);
        assert_eq!(peak, i16::MAX);
    }
}
