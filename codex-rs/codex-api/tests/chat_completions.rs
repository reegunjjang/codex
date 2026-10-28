#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use codex_api::AuthError;
use codex_api::AuthProvider;
use codex_api::ChatCompletionsClient;
use codex_api::ChatCompletionsRequest;
use codex_api::Provider;
use codex_api::ResponseEvent;
use codex_api::chat_messages_from_response_items;
use codex_client::HttpTransport;
use codex_client::Request;
use codex_client::RequestBody;
use codex_client::Response;
use codex_client::StreamResponse;
use codex_client::TransportError;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use http::HeaderMap;
use http::StatusCode;
use serde_json::Value;
use tokio::sync::Mutex;

#[derive(Clone)]
struct FixtureTransport {
    body: String,
    status: StatusCode,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl HttpTransport for FixtureTransport {
    async fn execute(&self, req: Request) -> Result<Response, TransportError> {
        self.requests.lock().await.push(req);
        Ok(Response {
            status: self.status,
            headers: HeaderMap::new(),
            body: Bytes::from(self.body.clone()),
        })
    }

    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError> {
        self.requests.lock().await.push(req);
        Ok(StreamResponse {
            status: self.status,
            headers: HeaderMap::new(),
            bytes: Box::pin(futures::stream::iter(vec![Ok::<_, TransportError>(
                Bytes::from(self.body.clone()),
            )])),
        })
    }
}

#[tokio::test]
async fn chat_client_adapts_non_streaming_text_and_multiple_tool_calls() -> Result<()> {
    let body = r#"{
        "id":"chatcmpl-nonstream",
        "choices":[{"index":0,"message":{"role":"assistant","content":"working","tool_calls":[
            {"id":"call-a","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a\"}"}},
            {"id":"call-b","type":"function","function":{"name":"list_dir","arguments":"{}"}}
        ]},"finish_reason":"tool_calls"}],
        "usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}
    }"#;
    let requests = Arc::new(Mutex::new(Vec::new()));
    let client = ChatCompletionsClient::new(
        FixtureTransport {
            body: body.into(),
            status: StatusCode::OK,
            requests: requests.clone(),
        },
        provider(),
        Arc::new(NoAuth),
    );
    let mut stream = client
        .complete_request(
            ChatCompletionsRequest {
                model: "deepseek-chat".into(),
                messages: vec![serde_json::json!({"role":"user","content":"inspect"})],
                tools: None,
                tool_choice: None,
                parallel_tool_calls: Some(true),
                stream: false,
                stream_options: None,
                tool_name_mappings: Vec::new(),
            },
            HeaderMap::new(),
        )
        .await?;

    let mut text = String::new();
    let mut calls = Vec::new();
    let mut completed = None;
    while let Some(event) = stream.next().await {
        match event? {
            ResponseEvent::OutputTextDelta(delta) => text.push_str(&delta),
            ResponseEvent::OutputItemDone(item @ ResponseItem::FunctionCall { .. }) => {
                calls.push(item)
            }
            ResponseEvent::Completed {
                response_id,
                token_usage,
                end_turn,
                ..
            } => completed = Some((response_id, token_usage, end_turn)),
            _ => {}
        }
    }
    assert_eq!(text, "working");
    assert_eq!(calls.len(), 2);
    assert!(matches!(&calls[0], ResponseItem::FunctionCall { call_id, .. } if call_id == "call-a"));
    assert!(matches!(&calls[1], ResponseItem::FunctionCall { call_id, .. } if call_id == "call-b"));
    let (response_id, usage, end_turn) = completed.expect("response should complete");
    assert_eq!(response_id, "chatcmpl-nonstream");
    assert_eq!(usage.expect("usage should be preserved").total_tokens, 8);
    assert_eq!(end_turn, Some(false));

    let requests = requests.lock().await;
    assert!(requests[0].url.ends_with("/chat/completions"));
    let Some(RequestBody::Json(body)) = requests[0].body.as_ref() else {
        panic!("expected JSON request body");
    };
    assert_eq!(body["stream"], false);
    assert!(body.get("stream_options").is_none());
    Ok(())
}

#[derive(Clone, Default)]
struct NoAuth;

impl AuthProvider for NoAuth {
    fn add_auth_headers(&self, _headers: &mut HeaderMap) -> Result<(), AuthError> {
        Ok(())
    }
}

fn provider() -> Provider {
    Provider {
        name: "DeepSeek".into(),
        base_url: "https://example.test/v1".into(),
        query_params: None,
        headers: HeaderMap::new(),
        retry: codex_api::RetryConfig {
            max_attempts: 1,
            base_delay: Duration::from_millis(1),
            retry_429: false,
            retry_5xx: false,
            retry_transport: false,
        },
        stream_idle_timeout: Duration::from_secs(2),
    }
}

#[test]
fn chat_history_maps_roles_and_tool_results_for_continuation() -> Result<()> {
    let items = vec![
        ResponseItem::Message {
            encrypted_content: None,
            status: None,
            id: None,
            role: "user".into(),
            content: vec![ContentItem::InputText {
                text: "inspect".into(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCall {
            encrypted_content: None,
            status: None,
            id: None,
            name: "read_file".into(),
            namespace: None,
            arguments: r#"{"path":"src/lib.rs"}"#.into(),
            encrypted_function_args: None,
            call_id: "call-a".into(),
            internal_chat_message_metadata_passthrough: None,
        },
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: Some("call-a".into()),
            name: Some("read_file".into()),
            namespace: None,
            output: FunctionCallOutputPayload {
                body: FunctionCallOutputBody::Text("file contents".into()),
                success: Some(true),
            },
            internal_chat_message_metadata_passthrough: None,
        },
    ];
    let messages = chat_messages_from_response_items(Some("follow safety rules"), &items)?;
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[2]["role"], "assistant");
    assert_eq!(messages[2]["tool_calls"][0]["id"], "call-a");
    assert_eq!(messages[3]["role"], "tool");
    assert_eq!(messages[3]["tool_call_id"], "call-a");
    assert_eq!(messages[3]["content"], "file contents");
    Ok(())
}

#[tokio::test]
async fn chat_client_streams_text_and_reassembles_fragmented_parallel_tool_calls() -> Result<()> {
    let body = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"checking\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call-a\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"src/lib.rs\\\"}\"}},{\"index\":1,\"id\":\"call-b\",\"type\":\"function\",\"function\":{\"name\":\"list_dir\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":4,\"total_tokens\":16}}\n\n",
        "data: [DONE]\n\n",
    );
    let requests = Arc::new(Mutex::new(Vec::new()));
    let client = ChatCompletionsClient::new(
        FixtureTransport {
            body: body.into(),
            status: StatusCode::OK,
            requests: requests.clone(),
        },
        provider(),
        Arc::new(NoAuth),
    );
    let mut stream = client
        .stream_request(
            ChatCompletionsRequest {
                model: "deepseek-chat".into(),
                messages: vec![serde_json::json!({"role":"user","content":"inspect"})],
                tools: None,
                tool_choice: None,
                parallel_tool_calls: Some(true),
                stream: true,
                stream_options: None,
                tool_name_mappings: Vec::new(),
            },
            HeaderMap::new(),
        )
        .await?;

    let mut text = String::new();
    let mut calls = Vec::new();
    let mut completed = None;
    while let Some(event) = stream.next().await {
        match event? {
            ResponseEvent::OutputTextDelta(delta) => text.push_str(&delta),
            ResponseEvent::OutputItemDone(item @ ResponseItem::FunctionCall { .. }) => {
                calls.push(item)
            }
            ResponseEvent::Completed {
                token_usage,
                end_turn,
                ..
            } => completed = Some((token_usage, end_turn)),
            _ => {}
        }
    }
    assert_eq!(text, "checking");
    assert_eq!(calls.len(), 2);
    assert!(
        matches!(&calls[0], ResponseItem::FunctionCall { name, call_id, arguments, .. }
        if name == "read_file" && call_id == "call-a" && arguments == r#"{"path":"src/lib.rs"}"#)
    );
    assert!(
        matches!(&calls[1], ResponseItem::FunctionCall { name, call_id, .. }
        if name == "list_dir" && call_id == "call-b")
    );
    let (usage, end_turn) = completed.expect("stream should complete");
    assert_eq!(usage.expect("usage should be preserved").total_tokens, 16);
    assert_eq!(end_turn, Some(false));

    let requests = requests.lock().await;
    assert!(requests[0].url.ends_with("/chat/completions"));
    let Some(RequestBody::EncodedJson(encoded_body)) = requests[0].body.as_ref() else {
        panic!("expected encoded JSON request body");
    };
    let request: Value = serde_json::from_slice(encoded_body.as_bytes())?;
    assert_eq!(request["model"], "deepseek-chat");
    assert_eq!(request["messages"][0]["role"], "user");
    Ok(())
}

#[tokio::test]
async fn chat_client_surfaces_http_rate_limit_and_retry_after() -> Result<()> {
    let client = ChatCompletionsClient::new(
        FixtureTransport {
            body: r#"{"error":{"message":"slow down"}}"#.into(),
            status: StatusCode::TOO_MANY_REQUESTS,
            requests: Arc::new(Mutex::new(Vec::new())),
        },
        provider(),
        Arc::new(NoAuth),
    );
    let result = client
        .stream_request(
            ChatCompletionsRequest {
                model: "deepseek-flash".into(),
                messages: Vec::new(),
                tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                stream: true,
                stream_options: None,
                tool_name_mappings: Vec::new(),
            },
            HeaderMap::new(),
        )
        .await;
    assert!(matches!(
        result,
        Err(codex_api::ApiError::RateLimitExceeded { message, .. }) if message == "slow down"
    ));
    Ok(())
}
