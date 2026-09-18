use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use nana_audio::capture::{levels, Capture};
use nana_audio::pipewire::PwDump;
use nana_audio::registry::{AudioTopology, NodeDirection};
use nana_audio::{classify_output, DesktopEndpoints};
use nana_core::endpoint::{AecMode, OutputClass};

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
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    match Cli::parse().command {
        Command::Devices => devices(),
        Command::Record { seconds, out } => record(seconds, out),
    }
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
