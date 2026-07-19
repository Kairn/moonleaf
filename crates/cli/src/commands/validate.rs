use clap::Parser;

#[derive(Parser)]
pub struct Args {
    /// Number of requests for calibration sampling
    #[arg(long, default_value_t = 1000)]
    pub requests: u32,
}

pub fn execute(args: &Args) {
    println!("validate: requests={}", args.requests);
}
