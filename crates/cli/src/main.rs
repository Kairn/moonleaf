#![warn(clippy::pedantic)]

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "moonleaf", version, about = "A wind tunnel for LLM inference infrastructure")]
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
    Run(Run),

    /// Run the inference simulator as a standalone server
    Sim(Sim),

    /// Run the same workload against two targets and compare results
    Compare(Compare),

    /// Calibrate measurement accuracy against the injector engine
    Validate(Validate),
}

#[derive(Parser)]
struct Run {
    /// Target endpoint URL
    #[arg(long, group = "target_source")]
    target: Option<String>,

    /// Start the simulator in-process with this profile instead of hitting a remote target
    #[arg(long, group = "target_source")]
    sim: Option<String>,

    /// Number of concurrent workers (closed-loop mode)
    #[arg(long, default_value_t = 1)]
    concurrency: u32,

    /// Total number of requests to send
    #[arg(long, default_value_t = 100)]
    requests: u32,
}

#[derive(Parser)]
struct Sim {
    // TODO: --port (u16, default 8080), --profile (String, required)
}

#[derive(Parser)]
struct Compare {
    // TODO: --target-a (String), --target-b (String), --requests (u32, default 100)
}

#[derive(Parser)]
struct Validate {
    // TODO: --requests (u32, default 1000)
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run(args) => {
            println!("run: target={:?} sim={:?} concurrency={} requests={}",
                args.target, args.sim, args.concurrency, args.requests);
        }
        Commands::Sim(_args) => {
            println!("sim: (not yet implemented)");
        }
        Commands::Compare(_args) => {
            println!("compare: (not yet implemented)");
        }
        Commands::Validate(_args) => {
            println!("validate: (not yet implemented)");
        }
    }
}
