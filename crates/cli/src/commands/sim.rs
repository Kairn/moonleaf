use clap::Parser;

#[derive(Parser)]
pub struct Args {
    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// GPU profile preset name or path to custom TOML
    #[arg(long)]
    pub profile: String,
}

pub fn execute(args: &Args) {
    println!("sim: port={} profile={}", args.port, args.profile);
}
