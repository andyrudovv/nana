//! Language-model clients behind the `Brain` port.

mod anthropic;
mod sse;

use std::collections::VecDeque;
use std::pin::Pin;
use std::time::Duration;

use futures_util::stream::{self, Stream, StreamExt};
use nana_core::ports::{Brain, BrainEvent, BrainRequest, BrainStream};
use nana_core::{Error, Result};

use anthropic::EventMapper;
use sse::SseParser;

pub const DEFAULT_MODEL: &str = "claude-haiku-4-5";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";

pub struct ClaudeBrain {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl ClaudeBrain {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        // No total timeout: a reply streams for as long as it takes. A silent
        // socket is what must not hang forever.
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .read_timeout(Duration::from_secs(20))
            .build()
            .map_err(|e| Error::Llm(format!("building the http client: {e}")))?;
        Ok(Self {
            client,
            api_key: api_key.into(),
            model: model.into(),
            base_url: DEFAULT_BASE_URL.into(),
        })
    }

    /// Points the client elsewhere, for tests against a local server.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

type Chunks = Pin<Box<dyn Stream<Item = reqwest::Result<Vec<u8>>> + Send>>;

enum State {
    Start(reqwest::RequestBuilder),
    Streaming {
        chunks: Chunks,
        parser: SseParser,
        mapper: EventMapper,
        ready: VecDeque<BrainEvent>,
        finished: bool,
    },
    End,
}

impl Brain for ClaudeBrain {
    fn stream(&self, request: BrainRequest) -> BrainStream {
        let body = anthropic::build_body(&self.model, &request);
        let builder = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&body);

        Box::pin(stream::unfold(State::Start(builder), |state| async move { step(state).await }))
    }

    fn model(&self) -> &str {
        &self.model
    }
}

/// Produces the next event, or ends the stream. Errors end it too: a broken
/// reply is not resumed halfway.
async fn step(mut state: State) -> Option<(Result<BrainEvent>, State)> {
    loop {
        match state {
            State::End => return None,
            State::Start(builder) => match open(builder).await {
                Ok(chunks) => {
                    state = State::Streaming {
                        chunks,
                        parser: SseParser::default(),
                        mapper: EventMapper::default(),
                        ready: VecDeque::new(),
                        finished: false,
                    }
                }
                Err(e) => return Some((Err(e), State::End)),
            },
            State::Streaming { mut chunks, mut parser, mut mapper, mut ready, mut finished } => {
                if let Some(event) = ready.pop_front() {
                    let next = if finished && ready.is_empty() {
                        State::End
                    } else {
                        State::Streaming { chunks, parser, mapper, ready, finished }
                    };
                    return Some((Ok(event), next));
                }
                if finished {
                    return None;
                }
                match chunks.next().await {
                    Some(Ok(bytes)) => {
                        for sse in parser.push(&bytes) {
                            match mapper.map(&sse) {
                                Ok(Some(event)) => {
                                    finished |= matches!(event, BrainEvent::Done { .. });
                                    ready.push_back(event);
                                }
                                Ok(None) => {}
                                Err(e) => return Some((Err(e), State::End)),
                            }
                        }
                        state = State::Streaming { chunks, parser, mapper, ready, finished };
                    }
                    Some(Err(e)) => {
                        return Some((Err(Error::Llm(format!("stream read: {e}"))), State::End))
                    }
                    None => {
                        let err = Error::Llm("stream ended before the reply was complete".into());
                        return Some((Err(err), State::End));
                    }
                }
            }
        }
    }
}

async fn open(builder: reqwest::RequestBuilder) -> Result<Chunks> {
    let response = builder.send().await.map_err(|e| Error::Llm(format!("request failed: {e}")))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        let body: String = body.chars().take(500).collect();
        return Err(Error::Llm(format!("http {status}: {body}")));
    }
    Ok(Box::pin(response.bytes_stream().map(|r| r.map(|b| b.to_vec()))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nana_core::ports::{Message, Role};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn request() -> BrainRequest {
        BrainRequest {
            system: "Ты Нана.".into(),
            messages: vec![Message { role: Role::User, content: "привет".into() }],
            max_tokens: 300,
        }
    }

    /// Serves one canned HTTP response and returns what the client sent.
    async fn serve(status: &str, body: String) -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let status = status.to_string();
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut received = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = socket.read(&mut buf).await.unwrap();
                received.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&received).to_string();
                if let Some(head_end) = text.find("\r\n\r\n") {
                    let length = text
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if received.len() >= head_end + 4 + length {
                        break;
                    }
                }
            }
            let response = format!(
                "HTTP/1.1 {status}\\r\\ncontent-type: text/event-stream\\r\\nconnection: close\\r\\n\\r\\n{body}"
            )
            .replace("\\r\\n", "\r\n");
            socket.write_all(response.as_bytes()).await.unwrap();
            String::from_utf8_lossy(&received).to_string()
        });
        (url, task)
    }

    const STREAM: &str = "event: message_start\ndata: {\"message\":{\"usage\":{\"input_tokens\":9}}}\n\n\
        event: content_block_delta\ndata: {\"delta\":{\"type\":\"text_delta\",\"text\":\"При\"}}\n\n\
        event: content_block_delta\ndata: {\"delta\":{\"type\":\"text_delta\",\"text\":\"вет\"}}\n\n\
        event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n\
        event: message_stop\ndata: {}\n\n";

    #[tokio::test]
    async fn a_reply_streams_in_order_and_sends_the_expected_request() {
        let (url, sent) = serve("200 OK", STREAM.to_string()).await;
        let brain = ClaudeBrain::new("secret-key", DEFAULT_MODEL).unwrap().with_base_url(url);

        let events: Vec<_> = brain.stream(request()).collect().await;
        let events: Vec<BrainEvent> = events.into_iter().map(|e| e.unwrap()).collect();
        assert_eq!(events[0], BrainEvent::Text("При".into()));
        assert_eq!(events[1], BrainEvent::Text("вет".into()));
        let BrainEvent::Done { usage, stop_reason } = &events[2] else { panic!("no Done") };
        assert_eq!((usage.input_tokens, usage.output_tokens), (9, 2));
        assert_eq!(stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(events.len(), 3);

        let sent = sent.await.unwrap().to_lowercase();
        assert!(sent.starts_with("post /v1/messages"), "got {sent}");
        assert!(sent.contains("x-api-key: secret-key"));
        assert!(sent.contains("anthropic-version: 2023-06-01"));
        assert!(sent.contains("\"cache_control\""));
    }

    #[tokio::test]
    async fn an_http_error_surfaces_its_status_and_body_without_the_key() {
        let body = r#"{"error":{"type":"authentication_error","message":"invalid x-api-key"}}"#;
        let (url, _) = serve("401 Unauthorized", body.to_string()).await;
        let brain = ClaudeBrain::new("secret-key", DEFAULT_MODEL).unwrap().with_base_url(url);

        let results: Vec<_> = brain.stream(request()).collect().await;
        assert_eq!(results.len(), 1, "an error ends the stream");
        let message = results[0].as_ref().unwrap_err().to_string();
        assert!(message.contains("401") && message.contains("authentication_error"), "{message}");
        assert!(!message.contains("secret-key"));
    }

    #[tokio::test]
    async fn a_stream_cut_before_message_stop_is_an_error_not_a_silent_end() {
        let cut = STREAM.split("event: message_stop").next().unwrap().to_string();
        let (url, _) = serve("200 OK", cut).await;
        let brain = ClaudeBrain::new("k", DEFAULT_MODEL).unwrap().with_base_url(url);

        let results: Vec<_> = brain.stream(request()).collect().await;
        assert!(matches!(results.last(), Some(Err(_))), "truncation must be visible");
    }
}
