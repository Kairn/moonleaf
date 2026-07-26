//! Incremental parser for Server-Sent Events streams.
//!
//! Bytes arrive from the socket in arbitrary pieces: one read may carry half a
//! line, several whole events, or a line terminator split down the middle. The
//! parser buffers the unconsumed tail and yields events as they complete.
//!
//! It is a pure bytes-to-events transform with no clock inside it. When a
//! single socket read yields several events they genuinely arrived together,
//! and timestamping them individually would invent arrival times the socket
//! never observed.
//!
//! There is no failure mode here. Incomplete input is [`None`], not an error:
//! push more bytes and ask again. Well-framed garbage passes straight through
//! as [`Event::Data`] and fails later in the caller's deserializer.

use std::mem;

/// One dispatched SSE event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// An event's data payload, uninterpreted. For OpenAI-compatible servers
    /// this is the JSON of one streaming chunk.
    Data(String),
    /// The `[DONE]` sentinel that terminates an `OpenAI` stream.
    Done,
}

/// Sentinel payload marking the end of an OpenAI-compatible stream.
const DONE_PAYLOAD: &str = "[DONE]";

/// Incremental SSE parser.
///
/// Feed it bytes with [`Parser::push`], then drain completed events with
/// [`Parser::next_event`] until it returns [`None`].
///
/// ```
/// use moonleaf_core::sse::{Event, Parser};
///
/// let mut parser = Parser::new();
/// parser.push(b"data: hello\n\nda");
/// assert_eq!(parser.next_event(), Some(Event::Data("hello".to_owned())));
/// assert_eq!(parser.next_event(), None); // "da" is an incomplete line
///
/// parser.push(b"ta: [DONE]\n\n");
/// assert_eq!(parser.next_event(), Some(Event::Done));
/// ```
#[derive(Debug, Default)]
pub struct Parser {
    /// Bytes received but not yet formed into a complete line.
    buf: Vec<u8>,
    /// Data lines accumulated for the event currently being assembled. Each
    /// line contributes its value plus a trailing `\n`.
    data: String,
}

impl Parser {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a socket read to the buffer.
    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Pull the next completed event, or [`None`] if more bytes are needed.
    pub fn next_event(&mut self) -> Option<Event> {
        loop {
            let (line_len, terminator_len) = find_terminator(&self.buf)?;

            // Decoding per complete line rather than per read is what keeps a
            // multi-byte character split across two reads intact.
            let line = String::from_utf8_lossy(&self.buf[..line_len]).into_owned();
            self.buf.drain(..line_len + terminator_len);

            if line.is_empty() {
                if let Some(event) = self.dispatch() {
                    return Some(event);
                }
                continue;
            }

            if line.starts_with(':') {
                continue; // comment, typically a keep-alive
            }

            let (field, value) = parse_field(&line);
            if field == "data" {
                self.data.push_str(value);
                self.data.push('\n');
            }
            // Every other field (`event`, `id`, `retry`, anything unknown) is
            // irrelevant to us and ignored.
        }
    }

    /// Finish the event under construction, triggered by a blank line.
    ///
    /// Yields [`None`] on an empty data buffer, so consecutive blank lines or a
    /// blank line after a lone comment produce no phantom event. A bare `data:`
    /// line does still dispatch — the buffer holds `"\n"`, which is not empty —
    /// yielding `Event::Data("")`.
    fn dispatch(&mut self) -> Option<Event> {
        if self.data.is_empty() {
            return None;
        }

        let mut payload = mem::take(&mut self.data);
        // Remove the trailing `\n`.
        payload.pop();
        if payload == DONE_PAYLOAD {
            Some(Event::Done)
        } else {
            Some(Event::Data(payload))
        }
    }
}

/// Locate the first line terminator in `buf`.
///
/// SSE accepts `\n`, `\r\n`, and a lone `\r`. Returns `Some((line_len,
/// terminator_len))`, so the caller reads `buf[..line_len]` and drains
/// `line_len + terminator_len` bytes, or [`None`] when no complete line is
/// available yet.
///
/// A `\r` as the final byte of the buffer is ambiguous — a lone-`\r`
/// terminator, or the first half of a `\r\n` whose `\n` is still in flight —
/// and counts as incomplete. Consuming it eagerly would make the `\n` arriving
/// on the next read look like a second, empty line, dispatching a phantom
/// event and truncating a real one.
fn find_terminator(buf: &[u8]) -> Option<(usize, usize)> {
    let i = buf.iter().position(|&b| b == b'\n' || b == b'\r')?;
    if buf[i] == b'\n' {
        return Some((i, 1));
    }
    match buf.get(i + 1) {
        None => None, // trailing \r: ambiguous, wait for more
        Some(b'\n') => Some((i, 2)),
        Some(_) => Some((i, 1)),
    }
}

/// Split an SSE field line into `(field, value)`.
///
/// Exactly one leading space is stripped from the value — a second space is
/// data. Only the first colon splits, since JSON payloads are full of colons
/// and every one after the first belongs to the value. A line with no colon at
/// all is a field name with an empty value.
///
/// Comment lines starting with `:` never reach here; the caller filters them.
fn parse_field(line: &str) -> (&str, &str) {
    let mut parts = line.splitn(2, ':');
    let field = parts.next().unwrap_or("");
    let value = parts.next().unwrap_or("");
    (field, value.strip_prefix(' ').unwrap_or(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ChatCompletionChunk;

    /// A realistic `OpenAI` stream: role-only opener, two content chunks, a
    /// keep-alive comment, the `finish_reason` chunk, the usage chunk, `[DONE]`.
    /// Flush left because raw-string indentation would become payload bytes.
    const OPENAI_STREAM: &str = r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

: keep-alive

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}

data: [DONE]

"#;

    /// Feed each slice as a separate "socket read", draining after every one.
    fn parse_all(reads: &[&[u8]]) -> Vec<Event> {
        let mut parser = Parser::new();
        let mut events = Vec::new();
        for read in reads {
            parser.push(read);
            while let Some(event) = parser.next_event() {
                events.push(event);
            }
        }
        events
    }

    fn data(payload: &str) -> Event {
        Event::Data(payload.to_owned())
    }

    // --- find_terminator ---------------------------------------------------

    #[test]
    fn terminator_lf() {
        assert_eq!(find_terminator(b"abc\ndef"), Some((3, 1)));
    }

    #[test]
    fn terminator_crlf() {
        assert_eq!(find_terminator(b"abc\r\ndef"), Some((3, 2)));
    }

    #[test]
    fn terminator_bare_cr() {
        assert_eq!(find_terminator(b"abc\rdef"), Some((3, 1)));
    }

    #[test]
    fn terminator_trailing_cr_is_incomplete() {
        assert_eq!(find_terminator(b"abc\r"), None);
    }

    #[test]
    fn terminator_absent() {
        assert_eq!(find_terminator(b"abc"), None);
        assert_eq!(find_terminator(b""), None);
    }

    #[test]
    fn terminator_at_start_marks_empty_line() {
        assert_eq!(find_terminator(b"\nabc"), Some((0, 1)));
        assert_eq!(find_terminator(b"\r\nabc"), Some((0, 2)));
    }

    // --- parse_field -------------------------------------------------------

    #[test]
    fn field_strips_exactly_one_leading_space() {
        assert_eq!(parse_field("data: x"), ("data", "x"));
        assert_eq!(parse_field("data:x"), ("data", "x"));
        assert_eq!(parse_field("data:  x"), ("data", " x"));
    }

    #[test]
    fn field_splits_on_first_colon_only() {
        assert_eq!(parse_field(r#"data: {"a":1}"#), ("data", r#"{"a":1}"#));
    }

    #[test]
    fn field_without_value() {
        assert_eq!(parse_field("data:"), ("data", ""));
        assert_eq!(parse_field("data"), ("data", ""));
    }

    // --- Parser ------------------------------------------------------------

    #[test]
    fn single_event() {
        assert_eq!(parse_all(&[b"data: hello\n\n"]), vec![data("hello")]);
    }

    #[test]
    fn multiple_events_in_one_push() {
        assert_eq!(
            parse_all(&[b"data: a\n\ndata: b\n\n"]),
            vec![data("a"), data("b")]
        );
    }

    #[test]
    fn done_sentinel() {
        assert_eq!(
            parse_all(&[b"data: a\n\ndata: [DONE]\n\n"]),
            vec![data("a"), Event::Done]
        );
    }

    #[test]
    fn incomplete_event_yields_nothing() {
        // No blank line, so the event has not dispatched yet.
        assert_eq!(parse_all(&[b"data: hello\n"]), vec![]);
    }

    #[test]
    fn multi_line_data_joined_with_newline() {
        assert_eq!(
            parse_all(&[b"data: line1\ndata: line2\n\n"]),
            vec![data("line1\nline2")]
        );
    }

    #[test]
    fn comment_is_ignored() {
        assert_eq!(
            parse_all(&[b": keep-alive\ndata: x\n: another\n\n"]),
            vec![data("x")]
        );
    }

    #[test]
    fn blank_lines_without_data_do_not_dispatch() {
        assert_eq!(parse_all(&[b": ping\n\n\n\ndata: x\n\n"]), vec![data("x")]);
    }

    #[test]
    fn empty_data_value_still_dispatches() {
        assert_eq!(parse_all(&[b"data:\n\n"]), vec![data("")]);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        assert_eq!(
            parse_all(&[b"event: message\nid: 42\nretry: 100\ndata: x\n\n"]),
            vec![data("x")]
        );
    }

    #[test]
    fn no_space_after_colon() {
        assert_eq!(parse_all(&[b"data:x\n\n"]), vec![data("x")]);
    }

    #[test]
    fn crlf_line_endings() {
        assert_eq!(parse_all(&[b"data: x\r\n\r\n"]), vec![data("x")]);
    }

    #[test]
    fn crlf_split_across_pushes() {
        // The \r ends one read and the \n opens the next: one terminator, not
        // two. Treating it as two dispatches a phantom event here.
        assert_eq!(parse_all(&[b"data: x\r", b"\n\r\n"]), vec![data("x")]);
    }

    #[test]
    fn bare_cr_line_endings() {
        // The final \r is ambiguous and stays buffered until the next read
        // resolves it — here as the \n half of a CRLF.
        assert_eq!(
            parse_all(&[b"data: x\r\rdata: y\r\r", b"\n"]),
            vec![data("x"), data("y")]
        );
    }

    #[test]
    fn multibyte_utf8_split_across_pushes() {
        let stream = "data: café\n\n".as_bytes();
        let split = stream.iter().position(|&b| b == 0xC3).unwrap() + 1;

        assert_eq!(
            parse_all(&[&stream[..split], &stream[split..]]),
            vec![data("café")]
        );
    }

    #[test]
    fn every_split_point_yields_same_events() {
        let bytes = OPENAI_STREAM.as_bytes();
        let expected = parse_all(&[bytes]);
        assert!(!expected.is_empty());

        for i in 0..=bytes.len() {
            let (head, tail) = bytes.split_at(i);
            assert_eq!(parse_all(&[head, tail]), expected, "split at byte {i}");
        }
    }

    #[test]
    fn byte_at_a_time_yields_same_events() {
        let bytes = OPENAI_STREAM.as_bytes();
        let expected = parse_all(&[bytes]);

        let singles: Vec<&[u8]> = bytes.chunks(1).collect();
        assert_eq!(parse_all(&singles), expected);
    }

    #[test]
    fn openai_stream_reassembles_into_chunks() {
        let events = parse_all(&[OPENAI_STREAM.as_bytes()]);
        assert_eq!(events.last(), Some(&Event::Done));

        let mut text = String::new();
        let mut usage = None;
        for event in &events {
            let Event::Data(payload) = event else {
                continue;
            };
            let chunk: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
            if let Some(content) = chunk.content() {
                text.push_str(content);
            }
            if chunk.usage.is_some() {
                usage = chunk.usage;
            }
        }

        assert_eq!(text, "Hello there");
        assert_eq!(usage.unwrap().completion_tokens, 2);
    }
}
