use crate::auth::SharedAuthProvider;
use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::telemetry::SseTelemetry;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use codex_client::RequestCompression;
use codex_client::RequestTelemetry;
use codex_http_client::ByteStream;
use codex_http_client::RetryAfter;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::ImageReference;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::time::timeout;

/// A native OpenAI-compatible Chat Completions request.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<Value>,
    /// Maps flattened Chat function names back to Codex namespaced tools.
    #[serde(skip)]
    pub tool_name_mappings: Vec<ChatToolNameMapping>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatToolNameMapping {
    pub chat_name: String,
    pub namespace: Option<String>,
    pub name: String,
}

/// Converts Codex's Responses-shaped conversation history to Chat messages.
/// Provider-private reasoning and bookkeeping records are intentionally omitted;
/// unsupported executable or multimodal items return an error instead of vanishing.
pub fn chat_messages_from_response_items(
    instructions: Option<&str>,
    items: &[ResponseItem],
) -> Result<Vec<Value>, ApiError> {
    let mut messages = Vec::new();
    if let Some(instructions) = instructions.filter(|text| !text.is_empty()) {
        messages.push(serde_json::json!({ "role": "system", "content": instructions }));
    }
    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                if !matches!(role.as_str(), "system" | "developer" | "user" | "assistant") {
                    return Err(ApiError::InvalidRequest {
                        message: format!("unsupported Chat Completions message role `{role}`"),
                    });
                }
                let mut parts = Vec::new();
                for part in content {
                    match part {
                        ContentItem::InputText { text } | ContentItem::OutputText { text, .. } => {
                            parts.push(serde_json::json!({ "type": "text", "text": text }));
                        }
                        ContentItem::InputImage { image, detail } => match image {
                            ImageReference::Inline { image_url } => {
                                let mut image = serde_json::json!({ "url": image_url });
                                if let Some(detail) = detail {
                                    image["detail"] = serde_json::to_value(detail).map_err(|err| ApiError::InvalidRequest { message: format!("could not encode image detail: {err}") })?;
                                }
                                parts.push(serde_json::json!({ "type": "image_url", "image_url": image }));
                            }
                            ImageReference::File { .. } => return Err(ApiError::InvalidRequest {
                                message: "Chat Completions does not support Codex file-backed image references".into(),
                            }),
                        },
                        ContentItem::InputAudio { .. } => return Err(ApiError::InvalidRequest {
                            message: "Chat Completions audio input is not supported by this adapter".into(),
                        }),
                    }
                }
                let content = if parts.len() == 1 && parts[0]["type"] == "text" {
                    parts[0]["text"].clone()
                } else {
                    Value::Array(parts)
                };
                messages.push(serde_json::json!({ "role": role, "content": content }));
            }
            ResponseItem::FunctionCall {
                name,
                namespace,
                arguments,
                call_id,
                ..
            } => {
                append_chat_tool_call(
                    &mut messages,
                    call_id,
                    &chat_tool_name(namespace.as_deref(), name),
                    arguments,
                )?;
            }
            ResponseItem::CustomToolCall {
                name,
                namespace,
                input,
                call_id,
                ..
            } => {
                append_chat_tool_call(
                    &mut messages,
                    call_id,
                    &chat_tool_name(namespace.as_deref(), name),
                    input,
                )?;
            }
            ResponseItem::FunctionCallOutput {
                call_id, output, ..
            } => {
                let call_id = call_id.as_deref().ok_or_else(|| ApiError::InvalidRequest {
                    message: "tool result is missing its call_id in Chat conversation history"
                        .into(),
                })?;
                append_chat_tool_result(&mut messages, call_id, output)?;
            }
            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                append_chat_tool_result(&mut messages, call_id, output)?;
            }
            ResponseItem::Reasoning { .. } => {
                // Reasoning is private to the upstream Responses provider.
            }
            ResponseItem::AdditionalTools { .. }
            | ResponseItem::AgentMessage { .. }
            | ResponseItem::LocalShellCall { .. }
            | ResponseItem::ToolSearchCall { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ImageGenerationCall { .. }
            | ResponseItem::Compaction { .. }
            | ResponseItem::ConfigurationUpdate { .. }
            | ResponseItem::CompactionTrigger { .. }
            | ResponseItem::ContextCompaction { .. }
            | ResponseItem::Other => {
                return Err(ApiError::InvalidRequest {
                    message: format!(
                        "Chat Completions cannot represent Codex history item `{}`",
                        item.item_type()
                    ),
                });
            }
        }
    }
    Ok(messages)
}

fn append_chat_tool_result(
    messages: &mut Vec<Value>,
    call_id: &str,
    output: &codex_protocol::models::FunctionCallOutputPayload,
) -> Result<(), ApiError> {
    let content = match &output.body {
        FunctionCallOutputBody::Text(text) => text.clone(),
        FunctionCallOutputBody::ContentItems(items) => {
            let mut text = Vec::new();
            for item in items {
                if let codex_protocol::models::FunctionCallOutputContentItem::InputText {
                    text: item_text,
                } = item
                {
                    text.push(item_text.as_str());
                } else {
                    return Err(ApiError::InvalidRequest {
                        message: "Chat Completions cannot represent non-text tool output items"
                            .into(),
                    });
                }
            }
            text.join("\n")
        }
    };
    messages
        .push(serde_json::json!({ "role": "tool", "tool_call_id": call_id, "content": content }));
    Ok(())
}

fn append_chat_tool_call(
    messages: &mut Vec<Value>,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> Result<(), ApiError> {
    let is_assistant_message = messages
        .last()
        .is_some_and(|message| message["role"] == "assistant");
    if !is_assistant_message {
        messages.push(
            serde_json::json!({ "role": "assistant", "content": Value::Null, "tool_calls": [] }),
        );
    }
    let message = messages
        .last_mut()
        .ok_or_else(|| ApiError::InvalidRequest {
            message: "could not construct assistant tool-call message".into(),
        })?;
    if message.get("tool_calls").is_none() {
        message["tool_calls"] = Value::Array(Vec::new());
    }
    let calls = message
        .get_mut("tool_calls")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| ApiError::InvalidRequest {
            message: "assistant tool_calls field is not an array".into(),
        })?;
    calls.push(serde_json::json!({
        "id": call_id,
        "type": "function",
        "function": { "name": name, "arguments": arguments },
    }));
    Ok(())
}

/// Direct `/chat/completions` transport for providers configured with `wire_api = "chat"`.
pub struct ChatCompletionsClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<std::sync::Arc<dyn SseTelemetry>>,
}

impl<T: HttpTransport> ChatCompletionsClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        mut self,
        request: Option<std::sync::Arc<dyn RequestTelemetry>>,
        sse: Option<std::sync::Arc<dyn SseTelemetry>>,
    ) -> Self {
        self.session = self.session.with_request_telemetry(request);
        self.sse_telemetry = sse;
        self
    }

    pub async fn stream_request(
        &self,
        mut request: ChatCompletionsRequest,
        extra_headers: HeaderMap,
    ) -> Result<ResponseStream, ApiError> {
        request.stream = true;
        request
            .stream_options
            .get_or_insert_with(|| serde_json::json!({ "include_usage": true }));
        let mut headers = extra_headers;
        headers.insert(
            http::header::ACCEPT,
            HeaderValue::from_static("text/event-stream"),
        );
        let body = EncodedJsonBody::encode(&request)
            .map_err(|err| ApiError::Stream(format!("failed to encode chat request: {err}")))?;
        let tool_name_mappings = request.tool_name_mappings.clone();
        let response = self
            .session
            .stream_encoded_json_with(
                Method::POST,
                "/chat/completions",
                headers,
                Some(body),
                |request| {
                    request.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                    request.compression = RequestCompression::None;
                },
            )
            .await?;
        if !response.status.is_success() {
            let status = response.status;
            let retry_after = RetryAfter::from_headers(&response.headers);
            let mut body = Vec::new();
            let mut bytes = response.bytes;
            while let Some(chunk) = bytes.next().await {
                match chunk {
                    Ok(chunk) => body.extend_from_slice(&chunk),
                    Err(error) => {
                        return Err(ApiError::Stream(format!(
                            "failed reading Chat Completions HTTP error: {error}"
                        )));
                    }
                }
            }
            let message = serde_json::from_slice::<Value>(&body)
                .ok()
                .and_then(|value| {
                    value
                        .pointer("/error/message")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned());
            return Err(match status {
                http::StatusCode::TOO_MANY_REQUESTS => ApiError::RateLimitExceeded {
                    message,
                    retry_after,
                },
                http::StatusCode::SERVICE_UNAVAILABLE => ApiError::ServerOverloaded { retry_after },
                _ => ApiError::Api { status, message },
            });
        }
        let upstream_request_id = response
            .headers
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let idle_timeout = self.session.provider().stream_idle_timeout;
        let sse_telemetry = self.sse_telemetry.clone();
        Ok(spawn_chat_stream(
            response.bytes,
            upstream_request_id,
            idle_timeout,
            tool_name_mappings,
            sse_telemetry,
        ))
    }
}

fn spawn_chat_stream(
    bytes: ByteStream,
    upstream_request_id: Option<String>,
    idle_timeout: Duration,
    tool_name_mappings: Vec<ChatToolNameMapping>,
    telemetry: Option<std::sync::Arc<dyn SseTelemetry>>,
) -> ResponseStream {
    let (tx_event, rx_event) = mpsc::channel(128);
    tokio::spawn(async move {
        let mut events = bytes.eventsource();
        let mut response_id = String::new();
        let mut text = String::new();
        let mut tools = BTreeMap::<u64, PartialToolCall>::new();
        let mut token_usage = None;
        let mut end_turn = None;
        loop {
            if tx_event.is_closed() {
                return;
            }
            let poll_start = Instant::now();
            let next_event = timeout(idle_timeout, events.next()).await;
            if let Some(telemetry) = telemetry.as_ref() {
                telemetry.on_sse_poll(&next_event, poll_start.elapsed());
            }
            let event = match next_event {
                Ok(Some(Ok(event))) => event,
                Ok(Some(Err(error))) => {
                    let _ = tx_event
                        .send(Err(ApiError::Stream(error.to_string())))
                        .await;
                    return;
                }
                Ok(None) => break,
                Err(_) => {
                    let _ = tx_event
                        .send(Err(ApiError::Stream(
                            "idle timeout waiting for Chat Completions SSE".into(),
                        )))
                        .await;
                    return;
                }
            };
            if event.data.trim() == "[DONE]" {
                break;
            }
            let chunk: Value = match serde_json::from_str(&event.data) {
                Ok(value) => value,
                Err(error) => {
                    let _ = tx_event
                        .send(Err(ApiError::Stream(format!(
                            "malformed Chat Completions event: {error}"
                        ))))
                        .await;
                    return;
                }
            };
            if let Some(error) = chunk.get("error") {
                let _ = tx_event
                    .send(Err(ApiError::Stream(format!(
                        "Chat Completions provider error: {error}"
                    ))))
                    .await;
                return;
            }
            if let Some(id) = chunk.get("id").and_then(Value::as_str) {
                response_id = id.to_owned();
            }
            if let Some(usage) = chunk.get("usage")
                && let (Some(input), Some(output), Some(total)) = (
                    usage.get("prompt_tokens").and_then(Value::as_i64),
                    usage.get("completion_tokens").and_then(Value::as_i64),
                    usage.get("total_tokens").and_then(Value::as_i64),
                )
            {
                token_usage = Some(TokenUsage {
                    input_tokens: input,
                    cached_input_tokens: 0,
                    cache_write_input_tokens: 0,
                    output_tokens: output,
                    reasoning_output_tokens: 0,
                    total_tokens: total,
                    codex_rollout_budget_units: None,
                });
            }
            if let Some(choice) = chunk
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|choices| choices.first())
            {
                if let Some(content) = choice.pointer("/delta/content") {
                    match content {
                        Value::String(delta) => {
                            text.push_str(delta);
                            let _ = tx_event
                                .send(Ok(ResponseEvent::OutputTextDelta(delta.clone())))
                                .await;
                        }
                        Value::Null => {}
                        _ => {
                            let _ = tx_event.send(Err(ApiError::Stream("Chat Completions provider returned non-string content; multimodal deltas are unsupported".into()))).await;
                            return;
                        }
                    }
                }
                if let Some(calls) = choice
                    .pointer("/delta/tool_calls")
                    .and_then(Value::as_array)
                {
                    for call in calls {
                        let Some(index) = call.get("index").and_then(Value::as_u64) else {
                            let _ = tx_event
                                .send(Err(ApiError::Stream(
                                    "Chat Completions tool call omitted index".into(),
                                )))
                                .await;
                            return;
                        };
                        let partial = tools.entry(index).or_default();
                        if let Some(id) = call.get("id").and_then(Value::as_str) {
                            partial.id.push_str(id);
                        }
                        if let Some(name) = call.pointer("/function/name").and_then(Value::as_str) {
                            partial.name.push_str(name);
                        }
                        if let Some(arguments) =
                            call.pointer("/function/arguments").and_then(Value::as_str)
                        {
                            partial.arguments.push_str(arguments);
                        }
                    }
                }
                if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                    match reason {
                        "stop" => end_turn = Some(true),
                        "tool_calls" | "function_call" | "length" => {
                            end_turn = Some(false);
                        }
                        "content_filter" => {
                            let _ = tx_event.send(Err(ApiError::ContentFilter)).await;
                            return;
                        }
                        other => {
                            let _ = tx_event
                                .send(Err(ApiError::Stream(format!(
                                    "unsupported Chat Completions finish_reason `{other}`"
                                ))))
                                .await;
                            return;
                        }
                    }
                }
            } else if chunk.get("usage").is_none()
                && chunk.get("choices").is_none()
                && chunk.get("id").is_none()
            {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "malformed Chat Completions chunk: expected id, choices, or usage".into(),
                    )))
                    .await;
                return;
            }
        }

        if !text.is_empty() {
            let item = ResponseItem::Message {
                encrypted_content: None,
                status: None,
                id: None,
                role: "assistant".into(),
                content: vec![ContentItem::OutputText {
                    text,
                    annotations: None,
                    logprobs: None,
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            };
            if tx_event
                .send(Ok(ResponseEvent::OutputItemDone(item)))
                .await
                .is_err()
            {
                return;
            }
        }
        let had_tool_calls = !tools.is_empty();
        for (_, tool) in tools {
            if tool.id.is_empty() || tool.name.is_empty() {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "Chat Completions returned an incomplete tool call (missing id or name)"
                            .into(),
                    )))
                    .await;
                return;
            }
            if serde_json::from_str::<Value>(&tool.arguments).is_err() {
                let _ = tx_event
                    .send(Err(ApiError::Stream(format!(
                        "Chat Completions returned malformed arguments for tool {}",
                        tool.name
                    ))))
                    .await;
                return;
            }
            let mapping = tool_name_mappings
                .iter()
                .find(|mapping| mapping.chat_name == tool.name);
            let item = if let Some(mapping) = mapping {
                ResponseItem::FunctionCall {
                    encrypted_content: None,
                    status: None,
                    id: None,
                    name: mapping.name.clone(),
                    namespace: mapping.namespace.clone(),
                    arguments: tool.arguments,
                    encrypted_function_args: None,
                    call_id: tool.id,
                    internal_chat_message_metadata_passthrough: None,
                }
            } else {
                ResponseItem::FunctionCall {
                    encrypted_content: None,
                    status: None,
                    id: None,
                    name: tool.name,
                    namespace: None,
                    arguments: tool.arguments,
                    encrypted_function_args: None,
                    call_id: tool.id,
                    internal_chat_message_metadata_passthrough: None,
                }
            };
            if tx_event
                .send(Ok(ResponseEvent::OutputItemDone(item)))
                .await
                .is_err()
            {
                return;
            }
        }
        let _ = tx_event
            .send(Ok(ResponseEvent::Completed {
                response_id,
                token_usage,
                usage_metadata: None,
                end_turn: end_turn.or(Some(!had_tool_calls)),
            }))
            .await;
    });
    ResponseStream {
        rx_event,
        upstream_request_id,
        interrupt: None,
    }
}

fn chat_tool_name(namespace: Option<&str>, name: &str) -> String {
    match namespace {
        Some(namespace) => format!("{namespace}__{name}"),
        None => name.to_owned(),
    }
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}
