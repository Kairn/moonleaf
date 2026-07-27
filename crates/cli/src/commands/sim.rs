use std::io;
use std::net::{Ipv4Addr, SocketAddr};

use clap::Parser;
use moonleaf_sim::Server;
use tokio::signal;
use tokio::signal::unix::SignalKind;

#[derive(Parser)]
pub struct Args {
    /// Port to listen on
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// GPU profile preset name or path to custom TOML
    #[arg(long)]
    pub profile: String,
}

/// Serves the simulator until interrupted.
///
/// # Errors
///
/// Returns the underlying [`io::Error`] if the port cannot be bound or the
/// accept loop fails.
pub async fn execute(args: &Args) -> io::Result<()> {
    let server = Server::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, args.port))).await?;

    tracing::info!(
        address = %server.local_addr()?,
        profile = %args.profile,
        "simulator listening"
    );

    server.serve(shutdown_signal()).await
}

/// Resolves on the first interrupt, which is what starts the graceful drain.
///
/// Both Ctrl-C and SIGTERM count: the standalone simulator is meant to be
/// runnable under a process supervisor or in a container, and those send
/// SIGTERM. Whichever arrives first wins; the other is left unhandled.
///
/// Linux and macOS are the shipped targets. A Windows build would need the
/// SIGTERM arm replaced with [`std::future::pending`], since the signal only
/// exists on Unix.
async fn shutdown_signal() {
    let interrupt = async {
        signal::ctrl_c().await.expect("install the Ctrl-C handler");
    };

    let terminate = async {
        signal::unix::signal(SignalKind::terminate())
            .expect("install the SIGTERM handler")
            .recv()
            .await;
    };

    tokio::select! {
        () = interrupt => tracing::info!("interrupted, draining in-flight requests"),
        () = terminate => tracing::info!("termination requested, draining in-flight requests"),
    }
}
