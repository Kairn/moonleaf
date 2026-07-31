#![warn(clippy::pedantic)]

//! The inference simulator: an OpenAI-compatible server with no GPU behind it.
//!
//! The HTTP surface is the whole public API. Both deployment shapes — a
//! standalone process and an in-process background task on a loopback port —
//! go through [`Server`], and traffic crosses real TCP either way.

mod error;
pub mod injector;
mod routes;
pub mod server;
pub mod stream;
pub mod tokenizer;

pub use server::Server;

/// The single model id this simulator serves.
///
/// `GET /v1/models` advertises it and `POST /v1/chat/completions` requires it,
/// so an unknown model fails here the same way it fails against a real backend
/// rather than silently producing a mislabeled run.
pub const MODEL_ID: &str = "moonleaf-sim";

/// Whether this simulator serves `id`.
///
/// The request path and `/v1/models` have to agree on this, so they ask the
/// same question here rather than each comparing ids themselves. When the
/// served id starts coming from configuration, this is the only place that
/// changes.
#[must_use]
pub fn serves_model(id: &str) -> bool {
    id == MODEL_ID
}
