//! `GET /healthz` — liveness.

use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct Health {
    status: &'static str,
}

/// Answers as soon as the server is accepting connections.
pub(crate) async fn healthz() -> Json<Health> {
    Json(Health { status: "ok" })
}
