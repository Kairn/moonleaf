//! `POST /v1/chat/completions` — the streaming endpoint.
//!
//! The handler is thin on purpose: validate, sample a [`StreamPlan`], then
//! replay it — sleep each planned delay, emit each planned chunk — and close
//! with the `[DONE]` sentinel.

use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use moonleaf_core::protocol::ChatCompletionRequest;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio_stream::StreamExt as _;

use crate::error::ApiError;
use crate::injector::InjectorConfig;
use crate::serves_model;
use crate::stream::StreamPlan;

pub(crate) async fn chat_completions(
    State(config): State<InjectorConfig>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    // Taking the rejection by value rather than letting axum handle it is what
    // turns a plain-text 400 into an OpenAI-shaped error body.
    let request = match payload {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return ApiError::invalid_request(rejection.body_text(), None).into_response();
        }
    };

    if let Err(error) = validate(&request) {
        return error.into_response();
    }

    // A request-supplied seed pins this response completely — timings, length,
    // id — the way a seed pins sampling on a real backend. Server-level
    // seeding arrives with the injector flags.
    let mut rng = match request.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is set after the Unix epoch")
        .as_secs();
    let plan = StreamPlan::sample(&request, config, created, &mut rng);

    let events = plan
        .chunks
        .into_iter()
        .map(|planned| {
            let event = Event::default()
                .json_data(&planned.chunk)
                .expect("chunk serializes to JSON");
            (planned.delay, event)
        })
        .chain(std::iter::once((
            Duration::ZERO,
            Event::default().data("[DONE]"),
        )));

    let stream = tokio_stream::iter(events).then(|(delay, event)| async move {
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        Ok::<_, Infallible>(event)
    });

    Sse::new(stream).into_response()
}

/// Checks a request against everything the simulator requires before it will
/// stream anything.
///
/// The rules, in the order they are applied:
///
/// 1. `model` must be one the simulator serves, else
///    [`ApiError::model_not_found`].
/// 2. `stream` must be true — non-streaming benchmarking is out of scope, and
///    a request that quietly fell back to it would produce meaningless TTFT.
/// 3. `messages` must not be empty.
/// 4. `max_tokens`, when present, must be at least 1.
///
/// Rules 2-4 are [`ApiError::invalid_request`] with the offending field name
/// as `param`.
fn validate(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    if !serves_model(&request.model) {
        return Err(ApiError::model_not_found(&request.model));
    }

    if !request.stream {
        return Err(ApiError::invalid_request(
            "streaming must be true",
            Some("stream"),
        ));
    }

    if request.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "messages must not be empty",
            Some("messages"),
        ));
    }

    if let Some(max_tokens) = request.max_tokens
        && max_tokens == 0
    {
        return Err(ApiError::invalid_request(
            "max_tokens must be at least 1",
            Some("max_tokens"),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use moonleaf_core::protocol::{Message, Role};

    use super::*;
    use crate::MODEL_ID;

    fn valid_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: MODEL_ID.to_owned(),
            messages: vec![Message {
                role: Role::User,
                content: "hi".to_owned(),
            }],
            stream: true,
            stream_options: None,
            max_tokens: None,
            temperature: None,
            seed: None,
            ignore_eos: None,
        }
    }

    #[test]
    fn valid_request_passes() {
        assert!(validate(&valid_request()).is_ok());
    }

    #[test]
    fn positive_max_tokens_passes() {
        let request = ChatCompletionRequest {
            max_tokens: Some(1),
            ..valid_request()
        };

        assert!(validate(&request).is_ok());
    }

    #[test]
    fn unknown_model_is_not_found() {
        let request = ChatCompletionRequest {
            model: "llama-3-70b".to_owned(),
            ..valid_request()
        };

        let error = validate(&request).unwrap_err();

        assert_eq!(error.status, StatusCode::NOT_FOUND);
        assert_eq!(error.detail.param, Some("model"));
        assert_eq!(error.detail.code, Some("model_not_found"));
        // The served id is what the caller needs to recover.
        assert!(error.detail.message.contains(MODEL_ID));
    }

    #[test]
    fn non_streaming_request_is_rejected() {
        let request = ChatCompletionRequest {
            stream: false,
            ..valid_request()
        };

        let error = validate(&request).unwrap_err();

        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.detail.param, Some("stream"));
    }

    #[test]
    fn empty_messages_are_rejected() {
        let request = ChatCompletionRequest {
            messages: Vec::new(),
            ..valid_request()
        };

        let error = validate(&request).unwrap_err();

        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.detail.param, Some("messages"));
    }

    #[test]
    fn zero_max_tokens_is_rejected() {
        let request = ChatCompletionRequest {
            max_tokens: Some(0),
            ..valid_request()
        };

        let error = validate(&request).unwrap_err();

        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.detail.param, Some("max_tokens"));
    }

    #[test]
    fn model_is_checked_before_the_other_rules() {
        // Everything is wrong at once; the model is what gets reported.
        let request = ChatCompletionRequest {
            model: "llama-3-70b".to_owned(),
            messages: Vec::new(),
            stream: false,
            max_tokens: Some(0),
            ..valid_request()
        };

        let error = validate(&request).unwrap_err();

        assert_eq!(error.status, StatusCode::NOT_FOUND);
        assert_eq!(error.detail.param, Some("model"));
    }
}
