use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use nana_asr::{AudioCtx, WhisperAsr};
use nana_audio::capture::{levels, Capture};
use nana_audio::pipewire::PwDump;
use nana_audio::registry::{AudioTopology, NodeDirection};
use nana_audio::{classify_output, DesktopEndpoints};
use nana_core::endpoint::{AecMode, OutputClass};
use nana_core::lang::Lang;
use nana_core::ports::Asr;
use nana_vad::{
    EndReason, EndpointConfig, Endpointer, SilenceMode, SileroVad, VadEvent, WINDOW_MS,
};
use nana_wake::{
    build_reference, transcript_confirms, PreTriggerSilence, Spotter, DEFAULT_SPELLINGS,
};

#[derive(Parser)]
#[command(name = "nana", about = "Nana development tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum WakeAction {
    /// Record samples of the phrase, trimmed to speech by VAD
    Record {
        #[arg(short, long, default_value = "nana")]
        name: String,
        /// rustpotter wants 3 to 8; more cost proportionally more CPU
        #[arg(short, long, default_value_t = 6)]
        count: usize,
        #[arg(short, long, default_value = "models/wake/samples")]
        dir: PathBuf,
    },
    /// Build a wake word reference from recorded samples
    Build {
        #[arg(short, long, default_value = "nana")]
        name: String,
        #[arg(short, long, default_value = "models/wake/samples")]
        dir: PathBuf,
    },
    /// Listen on the microphone with the full false-activation gate
    Listen {
        #[arg(short, long, default_value = "models/wake/nana.rpw")]
        wakeword: PathBuf,
        #[arg(short, long, default_value_t = 0.5)]
        threshold: f32,
        #[arg(short, long, default_value_t = 30.0)]
        seconds: f32,
        /// Report every candidate, including the ones the gate rejects
        #[arg(short, long)]
        verbose: bool,
    },
    /// Score a WAV against a wake word
    Scan {
        wav: PathBuf,
        #[arg(short, long, default_value = "models/wake/nana.rpw")]
        wakeword: PathBuf,
        #[arg(short, long, default_value_t = 0.5)]
        threshold: f32,
    },
}

#[derive(Subcommand)]
enum Command {
    /// Show audio devices and the endpoint Nana would use right now
    Devices,

    /// Record samples, build a wake word from them, or run detection
    Wake {
        #[command(subcommand)]
        action: WakeAction,
    },

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
        /// Encoder audio context. "auto" sizes it to the clip, a number pins it.
        #[arg(short, long, default_value = "auto")]
        audio_ctx: String,
        /// Re-enable whisper's temperature retries, removing the latency ceiling
        #[arg(long)]
        temperature_fallback: bool,
    },

    /// Ask Claude one question and report time to first token and cache use
    Ask {
        text: String,
        #[arg(short, long, default_value = nana_llm::DEFAULT_MODEL)]
        model: String,
        /// Replaces the built-in spoken-style prompt
        #[arg(short, long)]
        system: Option<String>,
        #[arg(long, default_value_t = 300)]
        max_tokens: u32,
        /// Repeat the same request to watch the prompt cache warm up
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
        Command::Transcribe { wav, model, runs, audio_ctx, temperature_fallback } => {
            transcribe(wav, model, runs, &audio_ctx, temperature_fallback)
        }
        Command::Vad { wav, model } => vad(wav, model),
        Command::Ask { text, model, system, max_tokens, runs } => {
            ask(&text, &model, system, max_tokens, runs)
        }
        Command::Wake { action } => wake(action),
    }
}

/// Spoken-style rules from the design, enough to hear the register.
const ASK_SYSTEM: &str = "Ты голосовой ассистент Нана. Отвечай по-человечески, \
в один-три коротких предложения, без списков, markdown и эмодзи. Числа пиши словами. \
Отвечай на языке собеседника.";

fn ask(
    text: &str,
    model: &str,
    system: Option<String>,
    max_tokens: u32,
    runs: usize,
) -> Result<()> {
    use futures_util::StreamExt;
    use nana_core::ports::{Brain, BrainEvent, BrainRequest, Message, Role};

    let key = std::env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY is not set; export it in this shell")?;
    let brain = nana_llm::ClaudeBrain::new(key, model)?;
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;

    for run in 1..=runs.max(1) {
        let request = BrainRequest {
            system: system.clone().unwrap_or_else(|| ASK_SYSTEM.to_string()),
            messages: vec![Message { role: Role::User, content: text.to_string() }],
            max_tokens,
        };
        let started = Instant::now();
        let mut first_token = None;
        runtime.block_on(async {
            let mut stream = brain.stream(request);
            while let Some(event) = stream.next().await {
                match event? {
                    BrainEvent::Text(piece) => {
                        first_token.get_or_insert_with(|| started.elapsed());
                        print!("{piece}");
                        std::io::Write::flush(&mut std::io::stdout())?;
                    }
                    BrainEvent::Done { usage, stop_reason } => {
                        println!(
                            "\n[run {run}] first token {} ms, total {} ms, stop {:?}",
                            first_token.map_or(0, |d| d.as_millis()),
                            started.elapsed().as_millis(),
                            stop_reason
                        );
                        println!(
                            "         tokens in {} out {}, cache read {} write {}",
                            usage.input_tokens,
                            usage.output_tokens,
                            usage.cache_read_tokens,
                            usage.cache_write_tokens
                        );
                    }
                }
            }
            anyhow::Ok(())
        })?;
    }
    Ok(())
}

fn wake(action: WakeAction) -> Result<()> {
    match action {
        WakeAction::Record { name, count, dir } => wake_record(&name, count, &dir),
        WakeAction::Build { name, dir } => wake_build(&name, &dir),
        WakeAction::Listen { wakeword, threshold, seconds, verbose } => {
            wake_listen(&wakeword, threshold, seconds, verbose)
        }
        WakeAction::Scan { wav, wakeword, threshold } => wake_scan(&wav, &wakeword, threshold),
    }
}

/// Captures one utterance per prompt, trimmed to the speech VAD found.
///
/// Trimming matters more here than at runtime: the reference is compared by
/// dynamic time warping, so leading silence in a sample distorts every later
/// match against it.
fn wake_record(name: &str, count: usize, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut detector = SileroVad::load("models/silero_vad.onnx")?;
    let capture = Capture::start_default_with_capacity(16_000 * 10)
        .context("could not open the default microphone")?;

    println!("recording {count} samples into {}", dir.display());
    println!("say the phrase after each prompt, then pause\n");

    for index in 1..=count {
        println!("  [{index}/{count}] speak now");
        let segment = record_one_utterance(&capture, &mut detector)?;
        let path = dir.join(format!("{name}_{index}.wav"));
        write_wav(&path, &segment)?;
        println!(
            "        {:.2}s -> {}",
            segment.len() as f32 / nana_core::audio::SAMPLE_RATE as f32,
            path.display()
        );
        std::thread::sleep(Duration::from_millis(700));
    }
    println!("\nnow run: nana-cli wake build --name {name}");
    Ok(())
}

fn record_one_utterance(capture: &Capture, detector: &mut SileroVad) -> Result<Vec<i16>> {
    const PAD_MS: usize = 120;
    let window = nana_core::audio::VAD_WINDOW_SAMPLES;
    let pad = PAD_MS * nana_core::audio::SAMPLE_RATE as usize / 1000;

    let mut endpointer = Endpointer::new(EndpointConfig::default());
    detector.reset();
    let mut cursor = capture.captured();
    let mut local: Vec<i16> = Vec::new();
    let mut consumed = 0usize;
    let mut start_at = 0usize;
    let deadline = Instant::now() + Duration::from_secs(12);

    loop {
        anyhow::ensure!(Instant::now() < deadline, "no speech heard in 12s");
        std::thread::sleep(Duration::from_millis(30));
        let Some((samples, next)) = capture.read_from(cursor) else {
            cursor = capture.captured();
            continue;
        };
        cursor = next;
        local.extend_from_slice(&samples);

        while consumed + window <= local.len() {
            let probability = detector.probe(&local[consumed..consumed + window])?;
            consumed += window;
            match endpointer.push(probability) {
                Some(VadEvent::SpeechStart) => {
                    start_at = consumed.saturating_sub(window * 12);
                }
                Some(VadEvent::SpeechEnd { .. }) => {
                    // Drop the trailing silence the endpointer waited through.
                    let silence = SilenceMode::Base.millis() as usize
                        * nana_core::audio::SAMPLE_RATE as usize
                        / 1000;
                    let end = consumed.saturating_sub(silence);
                    let from = start_at.saturating_sub(pad);
                    let to = (end + pad).min(local.len());
                    anyhow::ensure!(to > from, "captured an empty segment");
                    return Ok(local[from..to].to_vec());
                }
                None => {}
            }
        }
    }
}

fn wake_build(name: &str, dir: &Path) -> Result<()> {
    let mut samples: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e == "wav")
                && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(name))
        })
        .collect();
    samples.sort();
    anyhow::ensure!(!samples.is_empty(), "no {name}_*.wav files in {}", dir.display());

    let out = dir.parent().unwrap_or(Path::new(".")).join(format!("{name}.rpw"));
    let used = build_reference(name, &samples, &out)?;
    println!("built {} from {used} samples", out.display());
    for s in &samples {
        println!("  {}", s.display());
    }
    Ok(())
}

fn wake_scan(wav: &Path, wakeword: &Path, threshold: f32) -> Result<()> {
    let samples = read_wav(&wav.to_path_buf())?;
    let mut spotter = Spotter::load(wakeword, threshold)?;
    match spotter.push(&samples) {
        Some(d) => println!(
            "DETECTED \"{}\" at {:.2}s  score {:.3}  avg {:.3}",
            d.name,
            d.at_ms as f32 / 1000.0,
            d.score,
            d.avg_score
        ),
        None => println!(
            "no detection (best partial {:.3}, threshold {threshold:.2})",
            spotter.partial_score().unwrap_or(0.0)
        ),
    }
    Ok(())
}

/// Runs the layered gate a short wake word needs.
///
/// The spotter only proposes; a candidate becomes an activation after it also
/// begins a fresh burst of speech and the recogniser confirms the word stands
/// on its own rather than inside another one.
fn wake_listen(wakeword: &Path, threshold: f32, seconds: f32, verbose: bool) -> Result<()> {
    let mut spotter = Spotter::load(wakeword, threshold)?;
    let mut vad = SileroVad::load("models/silero_vad.onnx")?;
    let mut gate = PreTriggerSilence::new(300, 900, 0.5);
    // One word is all the transcript has to show: pin the language to ours and
    // cap the decode so a short clip cannot loop.
    let asr = WhisperAsr::load("models/ggml-base.bin")
        .context("loading the recogniser")?
        .with_languages(&[Lang::Ru, Lang::En])
        .with_max_tokens(12);
    asr.warm_up()?;

    let capture = Capture::start_default().context("could not open the default microphone")?;
    println!("listening for {} at threshold {threshold:.2}", wakeword.display());
    println!(
        "gate: 300 ms quiet before the word, at most 900 ms into a burst, then transcript check\n"
    );

    let started = Instant::now();
    let window = nana_core::audio::VAD_WINDOW_SAMPLES;
    let mut cursor = 0u64;
    let mut pending: Vec<i16> = Vec::new();
    let mut stats = GateStats::default();

    while started.elapsed() < Duration::from_secs_f32(seconds) {
        std::thread::sleep(Duration::from_millis(30));
        let Some((samples, next)) = capture.read_from(cursor) else {
            cursor = capture.captured();
            continue;
        };
        cursor = next;
        if samples.is_empty() {
            continue;
        }

        pending.extend_from_slice(&samples);
        while pending.len() >= window {
            let chunk: Vec<i16> = pending.drain(..window).collect();
            gate.push(vad.probe(&chunk)?);
        }

        let Some(detection) = spotter.push(&samples) else { continue };
        stats.candidates += 1;
        let at = started.elapsed().as_secs_f32();

        if !gate.satisfied() {
            stats.rejected_by_burst += 1;
            if verbose {
                println!(
                    "  [{at:5.1}s] rejected: {} ms into a burst (quiet before it {} ms), score {:.3}",
                    gate.burst_ms(),
                    gate.quiet_before_burst_ms(),
                    detection.score
                );
            }
            continue;
        }

        let heard = capture.tail(nana_core::audio::seconds_to_samples(2.0));
        let transcript = asr.transcribe(&heard)?;
        if !transcript_confirms(&transcript.text, &DEFAULT_SPELLINGS) {
            stats.rejected_by_transcript += 1;
            if verbose {
                println!(
                    "  [{at:5.1}s] rejected: heard {:?} ({:?}), score {:.3}",
                    transcript.text.trim(),
                    transcript.lang,
                    detection.score
                );
            }
            continue;
        }

        stats.activated += 1;
        println!(
            "  [{at:5.1}s] ACTIVATED  score {:.3}  heard {:?}",
            detection.score,
            transcript.text.trim()
        );
    }

    println!(
        "\n{} candidate(s): {} activated, {} rejected as mid-burst, {} rejected by transcript",
        stats.candidates, stats.activated, stats.rejected_by_burst, stats.rejected_by_transcript
    );
    Ok(())
}

#[derive(Default)]
struct GateStats {
    candidates: usize,
    activated: usize,
    rejected_by_burst: usize,
    rejected_by_transcript: usize,
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

fn transcribe(
    wav: PathBuf,
    model: PathBuf,
    runs: usize,
    audio_ctx: &str,
    temperature_fallback: bool,
) -> Result<()> {
    nana_asr::install_logging();
    let samples = read_wav(&wav)?;
    let audio_seconds = samples.len() as f32 / nana_core::audio::SAMPLE_RATE as f32;

    let loading = Instant::now();
    let ctx = match audio_ctx {
        "full" => AudioCtx::Full,
        "auto" => AudioCtx::Auto,
        value => {
            AudioCtx::Fixed(value.parse().context("--audio-ctx wants full, auto or a number")?)
        }
    };
    let asr = WhisperAsr::load(&model)
        .with_context(|| format!("loading {}", model.display()))?
        .with_audio_ctx(ctx);
    let asr = if temperature_fallback { asr.with_temperature_fallback() } else { asr };
    println!("model {} loaded in {:.2}s", asr.model(), loading.elapsed().as_secs_f32());
    println!(
        "audio {audio_seconds:.2}s from {}  (audio-ctx {:?} -> {:?})",
        wav.display(),
        ctx,
        ctx.frames(audio_seconds)
    );
    asr.warm_up()?;

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
