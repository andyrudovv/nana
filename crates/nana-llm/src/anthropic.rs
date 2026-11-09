//! Anthropic Messages API: request body and stream-event mapping.
//!
//! Pure functions over JSON, so the wire format is testable without a
//! network or a key.

use nana_core::ports::{BrainEvent, BrainRequest, Role, Usage};
use nana_core::{Error, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::sse::SseEvent;

/// `thinking` is omitted on purpose: Haiku 4.5 leaves it off by default, and
/// reasoning tokens would land straight in time-to-first-sound.
pub fn build_body(model: &str, request: &BrainRequest) -> Value {
    let messages: Vec<Value> = request
        .messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            json!({ "role": role, "content": m.content })
        })
        .collect();

    json!({
        "model": model,
        "max_tokens": request.max_tokens,
        "stream": true,
        // The breakpoint closes the stable prefix; everything after varies.
        "system": [{
            "type": "text",
            "text": request.system,
            "cache_control": { "type": "ephemeral" },
        }],
        "messages": messages,
    })
}

#[derive(Debug, Deserialize, Default)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
}

/// Folds the stream's events into [`BrainEvent`]s, remembering the pieces
/// of usage that arrive at different times.
#[derive(Default)]
pub struct EventMapper {
    usage: Usage,
    stop_reason: Option<String>,
}

impl EventMapper {
    pub fn map(&mut self, event: &SseEvent) -> Result<Option<BrainEvent>> {
        let name = event.event.as_deref().unwrap_or_default();
        match name {
            "message_start" => {
                let v = parse(&event.data)?;
                let usage: WireUsage = field(&v, &["message", "usage"]);
                self.usage.input_tokens = usage.input_tokens;
                self.usage.cache_read_tokens = usage.cache_read_input_tokens;
                self.usage.cache_write_tokens = usage.cache_creation_input_tokens;
                self.usage.output_tokens = usage.output_tokens;
                Ok(None)
            }
            "content_block_delta" => {
                let v = parse(&event.data)?;
                // Only text is spoken; other delta kinds are not ours to use.
                if v["delta"]["type"] == "text_delta" {
                    if let Some(text) = v["delta"]["text"].as_str().filter(|t| !t.is_empty()) {
                        return Ok(Some(BrainEvent::Text(text.to_string())));
                    }
                }
                Ok(None)
            }
            "message_delta" => {
                let v = parse(&event.data)?;
                let usage: WireUsage = field(&v, &["usage"]);
                // Cumulative, so it replaces rather than adds.
                self.usage.output_tokens = usage.output_tokens;
                self.stop_reason = v["delta"]["stop_reason"].as_str().map(str::to_string);
                Ok(None)
            }
            "message_stop" => Ok(Some(BrainEvent::Done {
                usage: self.usage,
                stop_reason: self.stop_reason.take(),
            })),
            "error" => {
                let v = parse(&event.data)?;
                let kind = v["error"]["type"].as_str().unwrap_or("error");
                let message = v["error"]["message"].as_str().unwrap_or("no message");
                Err(Error::Llm(format!("{kind}: {message}")))
            }
            // ping, content_block_start/stop and anything added later.
            _ => Ok(None),
        }
    }
}

fn parse(data: &str) -> Result<Value> {
    serde_json::from_str(data).map_err(|e| Error::Llm(format!("malformed stream event: {e}")))
}

fn field<T: for<'de> Deserialize<'de> + Default>(value: &Value, path: &[&str]) -> T {
    let mut cursor = value;
    for key in path {
        cursor = &cursor[*key];
    }
    serde_json::from_value(cursor.clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nana_core::ports::Message;

    fn ev(name: &str, data: &str) -> SseEvent {
        SseEvent { event: Some(name.into()), data: data.into() }
    }

    #[test]
    fn the_body_marks_the_system_prefix_for_caching_and_streams() {
        let request = BrainRequest {
            system: "Ты Нана.".into(),
            messages: vec![Message { role: Role::User, content: "привет".into() }],
            max_tokens: 300,
        };
        let body = build_body("claude-haiku-4-5", &request);
        assert_eq!(body["model"], "claude-haiku-4-5");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 300);
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["messages"][0]["role"], "user");
        assert!(body.get("thinking").is_none(), "reasoning must stay off");
    }

    #[test]
    fn a_whole_reply_maps_to_text_then_done_with_usage() {
        let mut m = EventMapper::default();
        let start = r#"{"message":{"usage":{"input_tokens":25,"output_tokens":1,"cache_read_input_tokens":20,"cache_creation_input_tokens":3}}}"#;
        assert_eq!(m.map(&ev("message_start", start)).unwrap(), None);
        assert_eq!(m.map(&ev("ping", "{}")).unwrap(), None);
        let delta = r#"{"delta":{"type":"text_delta","text":"Привет"}}"#;
        assert_eq!(
            m.map(&ev("content_block_delta", delta)).unwrap(),
            Some(BrainEvent::Text("Привет".into()))
        );
        let end = r#"{"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}"#;
        assert_eq!(m.map(&ev("message_delta", end)).unwrap(), None);
        let done = m.map(&ev("message_stop", "{}")).unwrap();
        assert_eq!(
            done,
            Some(BrainEvent::Done {
                usage: Usage {
                    input_tokens: 25,
                    output_tokens: 12,
                    cache_read_tokens: 20,
                    cache_write_tokens: 3
                },
                stop_reason: Some("end_turn".into()),
            })
        );
    }

    #[test]
    fn non_text_deltas_are_ignored() {
        let mut m = EventMapper::default();
        let thinking = r#"{"delta":{"type":"thinking_delta","thinking":"hm"}}"#;
        assert_eq!(m.map(&ev("content_block_delta", thinking)).unwrap(), None);
    }

    #[test]
    fn an_error_event_becomes_an_error_with_its_message() {
        let mut m = EventMapper::default();
        let data = r#"{"error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let err = m.map(&ev("error", data)).unwrap_err();
        assert!(err.to_string().contains("overloaded_error: Overloaded"), "got {err}");
    }

    #[test]
    fn malformed_json_is_reported_not_swallowed() {
        let mut m = EventMapper::default();
        assert!(m.map(&ev("content_block_delta", "{nope")).is_err());
    }
}
