//! Error responses in OpenAI's shape.
//!
//! Clients key off these bodies — the measurement client's error taxonomy
//! counts them, and a gateway pointed at the simulator expects the same
//! envelope a real backend sends. Axum's default rejections are plain text, so
//! every failure path is routed through [`ApiError`] instead.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::MODEL_ID;

/// A failure carrying the status it should be served with.
#[derive(Clone, Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub detail: ErrorDetail,
}

/// The inner object of an OpenAI error body.
#[derive(Clone, Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
    /// The offending request field, when one field is to blame.
    pub param: Option<&'static str>,
    /// A stable machine-readable tag, when the error has one.
    pub code: Option<&'static str>,
}

/// The `{"error": {...}}` wrapper OpenAI puts around [`ErrorDetail`].
#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorDetail,
}

impl ApiError {
    /// A malformed or unusable request body: HTTP 400.
    pub fn invalid_request(message: impl Into<String>, param: Option<&'static str>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            detail: ErrorDetail {
                message: message.into(),
                error_type: "invalid_request_error",
                param,
                code: None,
            },
        }
    }

    /// A model this simulator does not serve: HTTP 404.
    ///
    /// The message names the served id, so one failed request is enough to
    /// find the right one.
    pub fn model_not_found(requested: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            detail: ErrorDetail {
                message: format!(
                    "The model `{requested}` does not exist. This simulator serves `{MODEL_ID}`."
                ),
                error_type: "invalid_request_error",
                param: Some("model"),
                code: Some("model_not_found"),
            },
        }
    }

    /// A route that exists but has nothing behind it yet: HTTP 501.
    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            detail: ErrorDetail {
                message: message.into(),
                error_type: "server_error",
                param: None,
                code: None,
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(ErrorEnvelope { error: self.detail })).into_response()
    }
}
