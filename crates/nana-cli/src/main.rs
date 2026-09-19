use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use nana_asr::WhisperAsr;
use nana_audio::capture::{levels, Capture};
use nana_audio::pipewire::PwDump;
use nana_audio::registry::{AudioTopology, NodeDirection};
use nana_audio::{classify_output, DesktopEndpoints};
use nana_core::endpoint::{AecMode, OutputClass};
use nana_core::ports::Asr;
use nana_vad::{EndReason, EndpointConfig, Endpointer, SileroVad, VadEvent, WINDOW_MS};

#[derive(Parser)]
#[command(name = "nana", about = "Nana development tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show audio devices and the endpoint Nana would use right now
    Devices,

    /// Run voice activity detection over a 16 kHz mono WAV
    Vad {
        wav: PathBuf,
        #[arg(short, long, default_value = "models/silero_vad.onnx")]
        model: PathBuf,
    },

    /// Transcribe a 16 kHz mono WAV and report the time it took
    Transcribe {
        wav: PathBuf,
        #[arg(short, long, default_value = "models/ggml-base.bin")]
        model: PathBuf,
        /// Repeat to see the spread rather than a single sample
        #[arg(short, long, default_value_t = 1)]
        runs: usize,
    },

    /// Capture from the default microphone to check the input path
    Record {
        #[arg(short, long, default_value_t = 5.0)]
        seconds: f32,
        /// Write the captured audio as 16 kHz mono WAV
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("NANA_LOG")
                .unwrap_or_else(|_| "info,ort=warn,whisper_rs=warn".into()),
        )
        .init();

    match Cli::parse().command {
        Command::Devices => devices(),
        Command::Record { seconds, out } => record(seconds, out),
        Command::Transcribe { wav, model, runs } => transcribe(wav, model, runs),
        Command::Vad { wav, model } => vad(wav, model),
    }
}

fn vad(wav: PathBuf, model: PathBuf) -> Result<()> {
    let samples = read_wav(&wav)?;
    let mut detector =
        SileroVad::load(&model).with_context(|| format!("loading {}", model.display()))?;
    let mut endpointer = Endpointer::new(EndpointConfig::default());

    let mut probabilities: Vec<f32> = Vec::new();
    let mut inference = Duration::ZERO;
    let mut windows = 0usize;
    let mut speech_windows = 0usize;
    let mut start_ms = 0u32;

    println!("{} — {:.2}s", wav.display(), samples.len() as f32 / 16_000.0);
    let (windows_of_audio, _remainder) =
        samples.as_chunks::<{ nana_core::audio::VAD_WINDOW_SAMPLES }>();
    for chunk in windows_of_audio {
        let at = Instant::now();
        let probability = detector.probe(chunk)?;
        inference += at.elapsed();
        probabilities.push(probability);
        let now_ms = windows as u32 * WINDOW_MS;
        windows += 1;
        if probability >= 0.5 {
            speech_windows += 1;
        }
        match endpointer.push(probability) {
            Some(VadEvent::SpeechStart) => start_ms = now_ms,
            Some(VadEvent::SpeechEnd { speech_ms, reason }) => {
                let tag = match reason {
                    EndReason::Silence => "silence",
                    EndReason::TooLong => "cut, too long",
                };
                println!(
                    "  speech {:.2}s..{:.2}s  ({speech_ms} ms, {tag})",
                    start_ms as f32 / 1000.0,
                    (start_ms + speech_ms) as f32 / 1000.0
                );
            }
            None => {}
        }
    }

    let per_window = inference.as_secs_f64() * 1000.0 / windows.max(1) as f64;
    println!(
        "{windows} windows, {speech_windows} above threshold ({:.0}%), {per_window:.3} ms per window",
        100.0 * speech_windows as f64 / windows.max(1) as f64
    );
    if let Some(max) =
        probabilities.iter().cloned().fold(None::<f32>, |m, p| Some(m.map_or(p, |m: f32| m.max(p))))
    {
        let mean = probabilities.iter().sum::<f32>() / probabilities.len() as f32;
        let min = probabilities.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("probability  min {min:.4}  mean {mean:.4}  max {max:.4}");
    }
    if speech_windows == 0 {
        println!("  no speech detected");
    }
    Ok(())
}

fn transcribe(wav: PathBuf, model: PathBuf, runs: usize) -> Result<()> {
    nana_asr::install_logging();
    let samples = read_wav(&wav)?;
    let audio_seconds = samples.len() as f32 / nana_core::audio::SAMPLE_RATE as f32;

    let loading = Instant::now();
    let asr = WhisperAsr::load(&model).with_context(|| format!("loading {}", model.display()))?;
    println!("model {} loaded in {:.2}s", asr.model(), loading.elapsed().as_secs_f32());
    println!("audio {audio_seconds:.2}s from {}", wav.display());

    for run in 1..=runs.max(1) {
        let started = Instant::now();
        let transcript = asr.transcribe(&samples)?;
        let elapsed = started.elapsed();
        // Real time factor: below 1.0 means faster than the audio is long.
        let rtf = elapsed.as_secs_f32() / audio_seconds.max(f32::EPSILON);
        println!(
            "\nrun {run}: {:.0} ms  (rtf {rtf:.2})  lang {}  conf {:.2}  no-speech {:.2}",
            elapsed.as_millis(),
            transcript.lang.map(|l| l.code()).unwrap_or("?"),
            transcript.confidence,
            transcript.no_speech
        );
        match (transcript.is_empty(), transcript.is_usable()) {
            (true, _) => println!("  (no speech recognised)"),
            (false, false) => println!("  rejected, language not handled: {}", transcript.text),
            (false, true) => println!("  {}", transcript.text),
        }
    }
    Ok(())
}

fn read_wav(path: &PathBuf) -> Result<Vec<i16>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.channels == 1 && spec.sample_rate == nana_core::audio::SAMPLE_RATE,
        "expected 16 kHz mono, got {} Hz / {} ch",
        spec.sample_rate,
        spec.channels
    );
    Ok(reader.samples::<i16>().collect::<std::result::Result<_, _>>()?)
}

fn record(seconds: f32, out: Option<PathBuf>) -> Result<()> {
    // Size the ring for the whole request: the default one only holds pre-roll.
    let wanted = (nana_core::audio::SAMPLE_RATE as f32 * seconds).ceil() as usize;
    let capture = Capture::start_default_with_capacity(wanted)
        .context("could not open the default microphone")?;
    let format = capture.format();
    println!("capturing {seconds:.1}s at {} Hz / {} ch", format.sample_rate, format.channels);
    if !format.is_pipeline_native() {
        println!("note: device is not 16 kHz mono; downmixed on the fly");
    }

    let started = Instant::now();
    let mut next_tick = Duration::from_secs(1);
    let total = Duration::from_secs_f32(seconds);
    while started.elapsed() < total {
        std::thread::sleep(Duration::from_millis(50));
        if started.elapsed() >= next_tick {
            let (peak, rms) = levels(&capture.tail(nana_core::audio::SAMPLE_RATE as usize));
            println!("  {:>4.1}s  peak {peak:>6}  rms {rms:>8.1}", started.elapsed().as_secs_f32());
            next_tick += Duration::from_secs(1);
        }
    }

    let samples = capture.tail(wanted);
    let (peak, rms) = levels(&samples);
    println!(
        "captured {} samples ({:.2}s), peak {peak}, rms {rms:.1}",
        samples.len(),
        samples.len() as f32 / nana_core::audio::SAMPLE_RATE as f32
    );
    if peak == 0 {
        println!("warning: pure silence. Check that the source is not muted:");
        println!("  wpctl get-volume @DEFAULT_AUDIO_SOURCE@");
    }

    if let Some(path) = out {
        write_wav(&path, &samples)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn write_wav(path: &PathBuf, samples: &[i16]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: nana_core::audio::SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    Ok(())
}

fn devices() -> Result<()> {
    let endpoints = DesktopEndpoints::new(PwDump::default());
    let topology = endpoints.topology()?;
    let endpoint = DesktopEndpoints::<PwDump>::from_topology(&topology)?;

    println!("endpoint  {}", endpoint.id);
    println!("  input   {}", endpoint.input_name);
    println!("  output  {}", endpoint.output_name);
    println!("  class   {}", describe_class(endpoint.output_class));
    println!("  aec     {}", describe_aec(endpoint.aec_mode()));

    print_nodes(&topology, "inputs", NodeDirection::Source);
    print_nodes(&topology, "outputs", NodeDirection::Sink);
    Ok(())
}

fn print_nodes(topology: &AudioTopology, title: &str, direction: NodeDirection) {
    println!("\n{title}");
    for node in topology.by_direction(direction) {
        let marker = if topology.is_default(node) { "*" } else { " " };
        let class = match direction {
            NodeDirection::Sink => format!("  [{}]", short_class(classify_output(node))),
            NodeDirection::Source => String::new(),
        };
        println!("  {marker} {}{}", node.description, class);
    }
}

fn short_class(class: OutputClass) -> &'static str {
    match class {
        OutputClass::Headphones => "headphones",
        OutputClass::Speakers => "speakers",
        OutputClass::Unknown => "unknown",
    }
}

fn describe_class(class: OutputClass) -> &'static str {
    match class {
        OutputClass::Headphones => "headphones — no echo path",
        OutputClass::Speakers => "speakers — echo path",
        OutputClass::Unknown => "unknown — assumed speakers",
    }
}

fn describe_aec(mode: AecMode) -> &'static str {
    match mode {
        AecMode::Off => "off",
        AecMode::Render => "on, render reference",
    }
}
