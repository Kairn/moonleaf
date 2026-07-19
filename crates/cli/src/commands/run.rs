use clap::Parser;

#[derive(Parser)]
pub struct Args {
    /// Target endpoint URL
    #[arg(long, group = "target_source")]
    pub target: Option<String>,

    /// Start the simulator in-process with this profile
    #[arg(long, group = "target_source")]
    pub sim: Option<String>,

    /// Number of concurrent workers (closed-loop mode)
    #[arg(long, default_value_t = 1)]
    pub concurrency: u32,

    /// Total number of requests to send
    #[arg(long, default_value_t = 100)]
    pub requests: u32,
}

pub fn execute(args: &Args) {
    println!(
        "run: target={:?} sim={:?} concurrency={} requests={}",
        args.target, args.sim, args.concurrency, args.requests
    );
}
