//! Binding and serving the simulator's HTTP surface.
//!
//! Binding is separate from serving on purpose: a caller that asks for port 0
//! needs the OS-assigned port back *before* any traffic is sent, which is how
//! the in-process deployment shape finds its loopback address.

use std::future::Future;
use std::io;
use std::net::SocketAddr;

use axum::Router;
use axum::routing::{get, post};
use tokio::net::TcpListener;

use crate::routes;

/// A bound, not-yet-serving simulator.
pub struct Server {
    listener: TcpListener,
}

impl Server {
    /// Binds the listening socket.
    ///
    /// Pass port 0 to let the OS assign one and read it back with
    /// [`Server::local_addr`].
    ///
    /// # Errors
    ///
    /// Returns the underlying [`io::Error`] if the address cannot be bound —
    /// typically the port is already in use or not permitted.
    pub async fn bind(address: SocketAddr) -> io::Result<Self> {
        Ok(Self {
            listener: TcpListener::bind(address).await?,
        })
    }

    /// The address actually bound, with the port resolved.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`io::Error`] if the socket address cannot be
    /// read back.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Serves until `shutdown` resolves, then drains in-flight requests.
    ///
    /// Draining matters more than it looks: once completions stream, an abrupt
    /// stop would truncate responses mid-stream and show up as client-side
    /// errors that no server actually caused.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`io::Error`] if the accept loop fails.
    pub async fn serve<F>(self, shutdown: F) -> io::Result<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        axum::serve(self.listener, router())
            .with_graceful_shutdown(shutdown)
            .await
    }
}

/// Every route the simulator answers.
fn router() -> Router {
    Router::new()
        .route("/healthz", get(routes::health::healthz))
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
}
