//! Incremental Server-Sent Events framing.
//!
//! Network reads split anywhere, including inside a multi-byte character, so
//! bytes are buffered until a whole event has arrived and only then decoded.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

#[derive(Default)]
pub struct SseParser {
    buffer: Vec<u8>,
}

impl SseParser {
    /// Feeds raw bytes and returns every event they complete.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        // CRLF and LF framing are both legal; fold them into one.
        self.buffer.extend(bytes.iter().copied().filter(|b| *b != b'\r'));
        let mut events = Vec::new();
        while let Some(end) = self.buffer.windows(2).position(|w| w == b"\n\n") {
            let block: Vec<u8> = self.buffer.drain(..end + 2).collect();
            if let Some(event) = parse_block(&String::from_utf8_lossy(&block)) {
                events.push(event);
            }
        }
        events
    }
}

fn parse_block(block: &str) -> Option<SseEvent> {
    let mut event = None;
    let mut data: Vec<&str> = Vec::new();
    for line in block.lines() {
        // A leading colon marks a comment, used as a keep-alive.
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => event = Some(value.to_string()),
            "data" => data.push(value),
            _ => {}
        }
    }
    (!data.is_empty() || event.is_some()).then(|| SseEvent { event, data: data.join("\n") })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_complete_event_is_returned() {
        let mut p = SseParser::default();
        let out = p.push(b"event: ping\ndata: {\"a\":1}\n\n");
        assert_eq!(out, vec![SseEvent { event: Some("ping".into()), data: "{\"a\":1}".into() }]);
    }

    #[test]
    fn an_event_split_across_reads_waits_for_its_end() {
        let mut p = SseParser::default();
        assert!(p.push(b"event: x\nda").is_empty());
        assert!(p.push(b"ta: hel").is_empty());
        let out = p.push(b"lo\n\n");
        assert_eq!(out[0].data, "hello");
    }

    #[test]
    fn a_character_split_between_reads_survives() {
        let bytes = "data: Нана\n\n".as_bytes();
        let cut = "data: Н".len() - 1; // inside the two-byte «Н»
        let mut p = SseParser::default();
        assert!(p.push(&bytes[..cut]).is_empty());
        assert_eq!(p.push(&bytes[cut..])[0].data, "Нана");
    }

    #[test]
    fn crlf_framing_and_comments_are_handled() {
        let mut p = SseParser::default();
        let out = p.push(b": keep-alive\r\n\r\nevent: a\r\ndata: 1\r\n\r\n");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].event.as_deref(), Some("a"));
    }

    #[test]
    fn several_events_in_one_read_come_out_in_order() {
        let mut p = SseParser::default();
        let out = p.push(b"data: 1\n\ndata: 2\n\n");
        assert_eq!(out.iter().map(|e| e.data.as_str()).collect::<Vec<_>>(), ["1", "2"]);
    }

    #[test]
    fn multiple_data_lines_join_with_a_newline() {
        let mut p = SseParser::default();
        assert_eq!(p.push(b"data: a\ndata: b\n\n")[0].data, "a\nb");
    }
}
