use clap::Parser;

#[derive(Parser)]
pub struct Args {
    /// First target endpoint URL
    #[arg(long)]
    pub target_a: String,

    /// Second target endpoint URL
    #[arg(long)]
    pub target_b: String,

    /// Total number of requests to send to each target
    #[arg(long, default_value_t = 100)]
    pub requests: u32,
}

pub fn execute(args: &Args) {
    println!(
        "compare: a={} b={} requests={}",
        args.target_a, args.target_b, args.requests
    );
}
