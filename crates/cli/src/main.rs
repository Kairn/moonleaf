#![warn(clippy::pedantic)]

mod commands;

use std::io;

use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "moonleaf",
    version,
    about = "A wind tunnel for LLM inference infrastructure"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Global RNG seed for reproducible workloads
    #[arg(long, global = true)]
    seed: Option<u64>,

    /// Output format for results
    #[arg(long, global = true, default_value = "text")]
    output: OutputFormat,
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Subcommand)]
enum Commands {
    /// Benchmark an OpenAI-compatible streaming endpoint
    Run(commands::run::Args),

    /// Run the inference simulator as a standalone server
    Sim(commands::sim::Args),

    /// Run the same workload against two targets and compare results
    Compare(commands::compare::Args),

    /// Calibrate measurement accuracy against the injector engine
    Validate(commands::validate::Args),
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    match &cli.command {
        Commands::Run(args) => commands::run::execute(args),
        Commands::Sim(args) => commands::sim::execute(args).await?,
        Commands::Compare(args) => commands::compare::execute(args),
        Commands::Validate(args) => commands::validate::execute(args),
    }

    Ok(())
}
