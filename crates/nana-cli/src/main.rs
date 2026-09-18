use anyhow::Result;
use clap::{Parser, Subcommand};
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
    }
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
