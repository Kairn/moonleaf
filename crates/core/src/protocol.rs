//! OpenAI-compatible chat completions protocol types.
//!
//! The same types serve both directions: the measurement client serializes
//! requests and deserializes streaming chunks, the simulator does the reverse.
//!
//! Deserialization is deliberately permissive. The client must work against
//! backends that offer no cooperation, so unknown fields are ignored and every
//! field a real server might omit has a default — a surprising payload should
//! never take down a benchmark run. Serialization is the opposite: unset
//! request options are left out entirely rather than sent as `null`.

use serde::{Deserialize, Serialize};

/// A chat completions request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    /// Defaulted so a non-streaming request still deserializes and can be
    /// rejected with a real message instead of failing as a malformed body.
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// vLLM extension: keep decoding to `max_tokens` instead of stopping at an
    /// end-of-sequence token, so output length is controlled by the workload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
}

/// One message in a request's conversation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Opts into a final chunk carrying token counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

/// One SSE frame of a streaming response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub model: String,
    /// Empty on the final usage-only chunk sent under `include_usage`.
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    /// Present but null on content chunks when `include_usage` is set.
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkChoice {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub delta: Delta,
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
}

/// The incremental piece of a message carried by one chunk. The first chunk of
/// a choice typically carries the role, the last carries neither field.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Why a choice stopped producing tokens.
///
/// Servers invent values here, so unrecognized reasons are preserved as
/// [`FinishReason::Other`] rather than failing the parse.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    #[serde(untagged)]
    Other(String),
}

/// Server-reported token counts. Preferred over client-side estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

impl ChatCompletionChunk {
    /// Content text carried by the first choice's delta, if any.
    ///
    /// Returns `None` for role-only, empty, and usage-only chunks.
    #[must_use]
    pub fn content(&self) -> Option<&str> {
        self.choices.first()?.delta.content.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn content_chunk_parses() {
        let raw = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,
                      "model":"m","choices":[{"index":0,"delta":{"role":"assistant",
                      "content":"Hello"},"finish_reason":null}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();

        assert_eq!(chunk.content(), Some("Hello"));
        assert_eq!(chunk.choices[0].delta.role, Some(Role::Assistant));
        assert_eq!(chunk.choices[0].finish_reason, None);
        assert_eq!(chunk.usage, None);
    }

    #[test]
    fn final_usage_chunk_parses() {
        // vLLM's terminating chunk under `include_usage`: no choices at all.
        let raw = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,
                      "model":"m","choices":[],"usage":{"prompt_tokens":10,
                      "completion_tokens":5,"total_tokens":15}}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();

        assert!(chunk.choices.is_empty());
        assert_eq!(chunk.content(), None);
        assert_eq!(
            chunk.usage,
            Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            })
        );
    }

    #[test]
    fn chunk_with_unknown_fields_parses() {
        let raw = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,
                      "model":"m","system_fingerprint":"fp_x","service_tier":"default",
                      "choices":[{"index":0,"delta":{"content":"hi"},"logprobs":null,
                      "finish_reason":null}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();

        assert_eq!(chunk.content(), Some("hi"));
    }

    #[test]
    fn chunk_missing_optional_fields_parses() {
        let raw = r#"{"choices":[{"delta":{"content":"hi"}}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();

        assert_eq!(chunk.content(), Some("hi"));
        assert_eq!(chunk.id, "");
        assert_eq!(chunk.created, 0);
        assert_eq!(chunk.choices[0].index, 0);
    }

    #[test]
    fn empty_delta_chunk_yields_no_content() {
        let raw = r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;

        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();

        assert_eq!(chunk.content(), None);
        assert_eq!(chunk.choices[0].finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn known_finish_reasons_round_trip() {
        for (raw, expected) in [
            ("\"stop\"", FinishReason::Stop),
            ("\"length\"", FinishReason::Length),
            ("\"content_filter\"", FinishReason::ContentFilter),
            ("\"tool_calls\"", FinishReason::ToolCalls),
        ] {
            let parsed: FinishReason = serde_json::from_str(raw).unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(serde_json::to_string(&parsed).unwrap(), raw);
        }
    }

    #[test]
    fn unknown_finish_reason_falls_back_to_other() {
        let parsed: FinishReason = serde_json::from_str("\"abort\"").unwrap();

        assert_eq!(parsed, FinishReason::Other("abort".to_owned()));
        assert_eq!(serde_json::to_string(&parsed).unwrap(), "\"abort\"");
    }

    fn minimal_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "m".to_owned(),
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
    fn request_omits_unset_optional_fields() {
        let value = serde_json::to_value(minimal_request()).unwrap();
        let object = value.as_object().unwrap();

        for key in [
            "stream_options",
            "max_tokens",
            "temperature",
            "seed",
            "ignore_eos",
        ] {
            assert!(!object.contains_key(key), "{key} should not be serialized");
        }
        assert_eq!(object["stream"], json!(true));
    }

    #[test]
    fn request_serializes_set_options() {
        let request = ChatCompletionRequest {
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            max_tokens: Some(128),
            seed: Some(7),
            ignore_eos: Some(true),
            ..minimal_request()
        };

        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["stream_options"], json!({"include_usage": true}));
        assert_eq!(value["max_tokens"], json!(128));
        assert_eq!(value["seed"], json!(7));
        assert_eq!(value["ignore_eos"], json!(true));
    }

    #[test]
    fn request_without_stream_field_deserializes() {
        let raw = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;

        let request: ChatCompletionRequest = serde_json::from_str(raw).unwrap();

        assert!(!request.stream);
        assert_eq!(request.messages[0].role, Role::User);
    }
}
