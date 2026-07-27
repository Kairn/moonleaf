//! `GET /v1/models` — discovery.
//!
//! This list is the single source of truth for which model ids the simulator
//! accepts; `/v1/chat/completions` rejects anything not on it.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use serde::Serialize;

use crate::MODEL_ID;

#[derive(Serialize)]
pub struct ModelList {
    object: &'static str,
    data: Vec<Model>,
}

#[derive(Serialize)]
pub struct Model {
    id: &'static str,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

pub(crate) async fn list_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![Model {
            id: MODEL_ID,
            object: "model",
            created: unix_timestamp(),
            owned_by: "moonleaf",
        }],
    })
}

/// Seconds since the Unix epoch, or 0 on a clock set before it.
fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
