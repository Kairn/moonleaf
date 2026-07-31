//! End-to-end tests over real loopback HTTP.
//!
//! The simulator is only ever reached across a socket — never by in-process
//! calls — so the tests exercise it the same way the measurement client will.

use std::io;
use std::net::{Ipv4Addr, SocketAddr};

use moonleaf_sim::injector::InjectorConfig;
use moonleaf_sim::{MODEL_ID, Server};
use reqwest::StatusCode;
use serde_json::{Value, json};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

/// A running simulator on an OS-assigned loopback port.
struct TestServer {
    address: SocketAddr,
    shutdown: oneshot::Sender<()>,
    serving: JoinHandle<io::Result<()>>,
}

impl TestServer {
    async fn start() -> Self {
        let server = Server::bind(
            SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            InjectorConfig::default(),
        )
        .await
        .expect("bind loopback");
        let address = server.local_addr().expect("read back bound address");

        let (shutdown, signal) = oneshot::channel();
        let serving = tokio::spawn(async move {
            server
                .serve(async move {
                    signal.await.ok();
                })
                .await
        });

        Self {
            address,
            shutdown,
            serving,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.address)
    }

    /// Signals shutdown and waits for the accept loop to finish draining.
    async fn stop(self) -> io::Result<()> {
        self.shutdown.send(()).expect("server still listening");
        self.serving.await.expect("serve task panicked")
    }
}

/// Posts a chat completion body and returns the status and parsed JSON.
async fn post_completion(server: &TestServer, body: Value) -> (StatusCode, Value) {
    let response = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send request");
    let status = response.status();

    (status, response.json().await.expect("response is JSON"))
}

fn valid_body() -> Value {
    json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    })
}

#[tokio::test]
async fn binding_port_zero_reports_the_assigned_port() {
    let server = TestServer::start().await;

    // The in-process deployment shape depends on this: the client has to learn
    // the port before it can send anything.
    assert_ne!(server.address.port(), 0);

    server.stop().await.expect("clean shutdown");
}

#[tokio::test]
async fn healthz_reports_ok() {
    let server = TestServer::start().await;

    let response = reqwest::get(server.url("/healthz")).await.expect("GET");
    assert_eq!(response.status(), StatusCode::OK);

    let body: Value = response.json().await.expect("response is JSON");
    assert_eq!(body["status"], "ok");

    server.stop().await.expect("clean shutdown");
}

#[tokio::test]
async fn models_advertises_the_served_id() {
    let server = TestServer::start().await;

    let response = reqwest::get(server.url("/v1/models")).await.expect("GET");
    assert_eq!(response.status(), StatusCode::OK);

    let body: Value = response.json().await.expect("response is JSON");
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"].as_array().expect("data is an array").len(), 1);
    assert_eq!(body["data"][0]["id"], MODEL_ID);
    assert_eq!(body["data"][0]["object"], "model");

    server.stop().await.expect("clean shutdown");
}

#[tokio::test]
async fn valid_request_streams_a_completion() {
    let server = TestServer::start().await;

    // Two tokens keep the default 200 ms TTFT + 25 ms gap run short; the
    // full streaming behavior gets its own suite against fast configs.
    let mut body = valid_body();
    body["max_tokens"] = json!(2);
    let response = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response.headers()["content-type"]
        .to_str()
        .expect("readable content-type");
    assert!(
        content_type.starts_with("text/event-stream"),
        "not SSE: {content_type}"
    );

    let text = response.text().await.expect("read the whole stream");
    assert!(
        text.contains("chat.completion.chunk"),
        "no chunks in: {text}"
    );
    assert!(text.ends_with("data: [DONE]\n\n"), "unterminated: {text}");

    server.stop().await.expect("clean shutdown");
}

#[tokio::test]
async fn unknown_model_is_not_found() {
    let server = TestServer::start().await;

    let body = json!({
        "model": "llama-3-70b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    });
    let (status, body) = post_completion(&server, body).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "model_not_found");
    assert_eq!(body["error"]["param"], "model");

    server.stop().await.expect("clean shutdown");
}

#[tokio::test]
async fn non_streaming_request_is_rejected() {
    let server = TestServer::start().await;

    let body = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": false,
    });
    let (status, body) = post_completion(&server, body).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["param"], "stream");

    server.stop().await.expect("clean shutdown");
}

#[tokio::test]
async fn malformed_json_is_rejected_in_openai_shape() {
    let server = TestServer::start().await;

    let response = reqwest::Client::new()
        .post(server.url("/v1/chat/completions"))
        .header("content-type", "application/json")
        .body("{\"model\": ")
        .send()
        .await
        .expect("send request");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body: Value = response.json().await.expect("response is JSON");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        !body["error"]["message"]
            .as_str()
            .expect("message is a string")
            .is_empty()
    );

    server.stop().await.expect("clean shutdown");
}

#[tokio::test]
async fn missing_required_field_is_rejected() {
    let server = TestServer::start().await;

    // No `messages` at all — this fails in deserialization, before validation.
    let (status, body) = post_completion(&server, json!({"model": MODEL_ID, "stream": true})).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");

    server.stop().await.expect("clean shutdown");
}

#[tokio::test]
async fn graceful_shutdown_releases_the_port() {
    let server = TestServer::start().await;
    let address = server.address;

    reqwest::get(server.url("/healthz"))
        .await
        .expect("server is up");

    server.stop().await.expect("clean shutdown");

    assert!(
        reqwest::get(format!("http://{address}/healthz"))
            .await
            .is_err(),
        "server should not answer after shutdown"
    );
}
