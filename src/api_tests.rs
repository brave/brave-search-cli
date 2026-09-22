//! Tests for `api.rs`; a child module, so private items stay reachable.

use super::*;
use std::io::Cursor;

// ── agent configuration ──────────────────────────────────────────

/// `recv_response`, `global` and `per_call` would each cap a stream however much data flows.
#[test]
fn streaming_agent_has_no_total_deadline() {
    let timeouts = streaming_agent(30, 300).config().timeouts();

    assert_eq!(timeouts.recv_response, None, "would cap the whole stream");
    assert_eq!(timeouts.global, None, "would cap the whole stream");
    assert_eq!(timeouts.per_call, None, "would cap the whole stream");
    assert_eq!(
        timeouts.recv_body,
        Some(Duration::from_secs(300)),
        "bounds silence between reads"
    );
}

/// Only mid-stream silence gets the research budget; on the dial phases a blackholed host would
/// hang for minutes.
#[test]
fn streaming_agent_keeps_the_research_budget_off_the_dial_phases() {
    let timeouts = streaming_agent(30, 300).config().timeouts();
    let dial = Some(Duration::from_secs(30));

    assert_eq!(timeouts.resolve, dial);
    assert_eq!(timeouts.connect, dial);
    // The header wait inherits these two, and the rustls handshake runs under `send_request`.
    assert_eq!(timeouts.send_request, dial);
    assert_eq!(timeouts.send_body, dial);
}

/// An explicit `--timeout` reaches every phase.
#[test]
fn streaming_agent_passes_an_explicit_timeout_to_every_phase() {
    let timeouts = streaming_agent(7, 7).config().timeouts();
    let t = Some(Duration::from_secs(7));

    for got in [
        timeouts.resolve,
        timeouts.connect,
        timeouts.send_request,
        timeouts.send_body,
        timeouts.recv_body,
    ] {
        assert_eq!(got, t);
    }
}

/// The blocking agent has one total deadline and no phase deadline that could fire earlier.
#[test]
fn blocking_agent_has_a_total_deadline_and_nothing_else() {
    let timeouts = agent(30).config().timeouts();

    assert_eq!(timeouts.global, Some(Duration::from_secs(30)));
    for unset in [
        timeouts.per_call,
        timeouts.resolve,
        timeouts.connect,
        timeouts.send_request,
        timeouts.send_body,
        timeouts.recv_response,
        timeouts.recv_body,
    ] {
        assert_eq!(unset, None);
    }
}

// ── is_json ──────────────────────────────────────────────────────

/// stdout promises one JSON document per line, so `{oops` or a proxy's plaintext error inside a
/// 200 must be rejected.
#[test]
fn is_json_accepts_one_document_and_nothing_else() {
    for ok in [
        &br#"{"n":0}"#[..],
        b"[]",
        b"{}",
        b"1",
        b"\"s\"",
        b"null",
        b"true",
        b"  {}  ",       // insignificant whitespace around a document
        b"{\n \"a\":1}", // interior newlines are legal JSON, if not one line
    ] {
        assert!(is_json(ok), "rejected {:?}", String::from_utf8_lossy(ok));
    }
    for bad in [
        &b"{oops"[..],
        b"",
        b"   ",
        b"{} {}", // trailing content is a second document
        b"{}x",
        b"plain text error",
        b"<html>oops</html>",
        b"\xff\xfe",       // invalid UTF-8 is rejected for free
        b"\xEF\xBB\xBF{}", // a BOM is not JSON whitespace
    ] {
        assert!(!is_json(bad), "accepted {:?}", String::from_utf8_lossy(bad));
    }
}

/// The two paths trim differently: `str::trim` removes U+00A0, `sse_data`'s `trim_ascii` does
/// not. Pinned so unifying the trimming is a visible change.
#[test]
fn is_json_sees_what_each_path_left_behind() {
    assert!(
        is_json("\u{a0}{}".trim().as_bytes()),
        "str::trim takes NBSP"
    );
    assert!(!is_json("\u{a0}{}".as_bytes()), "trim_ascii leaves it");
    // U+FEFF is not White_Space: a BOM-prefixed blocking body is non-JSON, unlike on the SSE path.
    assert_eq!("\u{feff}{}".trim(), "\u{feff}{}");
}

// ── sse_data ─────────────────────────────────────────────────────

/// Every field shape a server can send. A misfiled line either puts non-JSON on stdout or drops
/// an answer chunk.
#[test]
fn sse_data_classifies_every_field_shape() {
    let json: &[u8] = b"{\"n\":0}";
    let done: &[u8] = b"[DONE]";
    let cases: &[(&[u8], Option<&[u8]>)] = &[
        // Nothing, one space, two spaces or a tab after the colon: all the same payload.
        (b"data:{\"n\":0}", Some(json)),
        (b"data: {\"n\":0}", Some(json)),
        (b"data:  {\"n\":0}", Some(json)),
        (b"data:\t{\"n\":0}", Some(json)),
        (b"data: {\"n\":0} ", Some(json)),
        // `read_line_bounded` strips only one trailing \r.
        (b"data: {\"n\":0}\r", Some(json)),
        // The sentinel, however it is padded…
        (b"data: [DONE]", Some(done)),
        (b"data:[DONE]", Some(done)),
        (b"data:   [DONE]", Some(done)),
        (b"data: [DONE] ", Some(done)),
        (b"data: [DONE]\t", Some(done)),
        // …but only as the whole payload, and only uppercase.
        (b"data: [DONE]x", Some(b"[DONE]x")),
        (b"data: [done]", Some(b"[done]")),
        (
            b"data: {\"text\":\"[DONE]\"}",
            Some(b"{\"text\":\"[DONE]\"}"),
        ),
        // Interior whitespace is content.
        (b"data: a\tb", Some(b"a\tb")),
        // Empty and whitespace-only payloads dispatch no event (WHATWG 9.2.6).
        (b"data:", None),
        (b"data: ", None),
        (b"data:  ", None),
        (b"data: \t ", None),
        // Other fields, comments, the blank line between events.
        (b"data", None),
        (b"datax: y", None),
        (b"DATA: x", None), // field names are case-sensitive
        (b": keep-alive", None),
        (b": data: not a record", None),
        (b"event: message", None),
        (b"id: 42", None),
        (b"retry: 3000", None),
        (b"", None),
    ];

    for (line, want) in cases {
        assert_eq!(
            sse_data(line),
            *want,
            "misclassified {:?}",
            String::from_utf8_lossy(line)
        );
    }
}

/// A leading BOM would hide the first record, so it is stripped; one inside the payload is
/// content.
#[test]
fn sse_data_strips_a_leading_bom_but_not_one_inside_the_payload() {
    assert_eq!(sse_data("\u{feff}data: {}".as_bytes()), Some(&b"{}"[..]));
    assert_eq!(
        sse_data("data: \u{feff}{}".as_bytes()),
        Some("\u{feff}{}".as_bytes())
    );
}

/// The spec strips the BOM once per stream; bx strips it per line, which is over-permissive but
/// never lossy.
#[test]
fn sse_data_strips_a_bom_from_any_line_not_just_the_first() {
    assert_eq!(sse_data("\u{feff}data: {}".as_bytes()), Some(&b"{}"[..]));
    // …but only one. A second BOM sits where the field name should be.
    assert_eq!(sse_data("\u{feff}\u{feff}data: {}".as_bytes()), None);
}

/// With no space after the colon the BOM is payload and must reach stdout byte-for-byte.
#[test]
fn sse_data_keeps_a_bom_that_begins_the_payload() {
    assert_eq!(
        sse_data("data:\u{feff}{}".as_bytes()),
        Some("\u{feff}{}".as_bytes())
    );
}

/// `trim_ascii` ignores non-ASCII whitespace, so an NBSP-padded `[DONE]` is emitted as a record.
#[test]
fn sse_data_does_not_recognise_a_sentinel_padded_with_non_ascii_space() {
    assert_eq!(
        sse_data("data: [DONE]\u{a0}".as_bytes()),
        Some("[DONE]\u{a0}".as_bytes())
    );
}

/// Pins the trim set: form feed goes; vertical tab and U+00A0 stay. Each only appears in an
/// already-invalid payload.
#[test]
fn sse_data_trims_ascii_whitespace_and_nothing_else() {
    assert_eq!(sse_data(b"data: \x0c"), None); // form feed: trimmed away
    assert_eq!(sse_data(b"data: \x0b"), Some(&b"\x0b"[..])); // vertical tab: kept
    assert_eq!(
        sse_data("data: \u{a0}{}".as_bytes()),
        Some("\u{a0}{}".as_bytes()) // NBSP: kept
    );
    // Bytes are passed through unvalidated — no UTF-8 or JSON check happens here.
    assert_eq!(sse_data(b"data: \xff\xfe"), Some(&[0xff, 0xfe][..]));
}

// ── format_error / exit_code_for_status ──────────────────────────

/// Agents branch on the exit code: 3 fix your key, 4 back off, 5 retry later, 1 unusable result.
#[test]
fn exit_code_for_status_maps_every_class() {
    for (status, want) in [
        (200, 1), // never consulted for a success, but pinned so the fallthrough is visible
        (204, 1),
        (302, 1),
        (400, 1),
        (401, 3),
        (403, 3),
        (429, 4),
        (499, 1),
        (500, 5),
        (599, 5),
        (600, 1),
    ] {
        assert_eq!(exit_code_for_status(status), want, "status {status}");
    }
}

/// Only a real error envelope is formatted; other JSON would render as `unknown (302)`, which
/// says less than the status.
#[test]
fn format_error_uses_the_envelope_only_when_there_is_one() {
    let (msg, code) = format_error(302, r#"{"redirect":1}"#);
    assert_eq!(msg, "HTTP 302");
    assert_eq!(code, 1);

    for body in ["not json at all", "", "[]", r#"{"error":"a string"}"#] {
        assert_eq!(format_error(500, body).0, "HTTP 500", "{body}");
    }
}

/// Code plus detail, code alone, and the placeholder for neither.
#[test]
fn format_error_reads_the_envelope() {
    let with_detail = r#"{"error":{"code":"RATE_LIMITED","detail":"slow down"}}"#;
    assert_eq!(
        format_error(429, with_detail).0,
        "rate limited (429) — slow down\nhint: retry after a short delay, or upgrade plan for higher rate limits"
    );

    let code_only = r#"{"error":{"code":"SUBSCRIPTION_TOKEN_INVALID"}}"#;
    assert!(
        format_error(401, code_only)
            .0
            .starts_with("subscription token invalid (401)")
    );

    assert!(
        format_error(400, r#"{"error":{}}"#)
            .0
            .starts_with("unknown (400)")
    );
}

/// A hint is an instruction; it belongs only where there is something to do.
#[test]
fn format_error_hints_only_where_there_is_an_action() {
    assert!(format_error(401, "{}").0.contains("bx config show-key"));
    assert!(format_error(403, "{}").0.contains("different API plan"));
    assert!(
        format_error(429, "{}")
            .0
            .contains("retry after a short delay")
    );
    for quiet in [400, 404, 302, 500] {
        assert!(
            !format_error(quiet, "{}").0.contains("hint:"),
            "status {quiet}"
        );
    }
}

// ── is_timeout ───────────────────────────────────────────────────

/// ureq timeouts and the SSE size-limit error are both `ErrorKind::Other`; confusing them would
/// tell an agent to raise --timeout for a failure no timeout fixes.
#[test]
fn is_timeout_recognises_only_a_wrapped_ureq_timeout() {
    assert!(is_timeout(&io::Error::other(ureq::Error::Timeout(
        ureq::Timeout::RecvBody
    ))));
    // A different ureq error, wrapped identically.
    assert!(!is_timeout(&io::Error::other(ureq::Error::HostNotFound)));
    // Our own oversized-line error: ErrorKind::Other, non-ureq payload.
    assert!(!is_timeout(&io::Error::other(
        "SSE line exceeds maximum size"
    )));
    // A kind-only error has no inner value to downcast.
    assert!(!is_timeout(&io::Error::from(io::ErrorKind::TimedOut)));
}

/// ureq wraps exactly once (`Error::into_io`), so a doubly wrapped timeout is not one of ureq's.
#[test]
fn is_timeout_does_not_peel_a_second_layer_of_wrapping() {
    let once = io::Error::other(ureq::Error::Timeout(ureq::Timeout::RecvBody));
    assert!(!is_timeout(&io::Error::other(once)));
}

/// Every phase's timeout counts; otherwise the user sees a raw ureq string instead of the stall
/// message.
#[test]
fn is_timeout_accepts_every_phase() {
    for phase in [
        ureq::Timeout::Global,
        ureq::Timeout::PerCall,
        ureq::Timeout::Resolve,
        ureq::Timeout::Connect,
        ureq::Timeout::SendRequest,
        ureq::Timeout::SendBody,
        ureq::Timeout::RecvResponse,
        ureq::Timeout::RecvBody,
    ] {
        assert!(
            is_timeout(&io::Error::other(ureq::Error::Timeout(phase))),
            "{phase:?}"
        );
    }
}

// ── write_record ─────────────────────────────────────────────────

/// A writer that fails with `BrokenPipe` at one step: record, newline, or flush.
struct FailAt {
    step: usize,
    fail_at: usize,
    written: Vec<u8>,
    flushes: usize,
}

impl FailAt {
    fn new(fail_at: usize) -> Self {
        Self {
            step: 0,
            fail_at,
            written: Vec::new(),
            flushes: 0,
        }
    }
    fn tick(&mut self) -> io::Result<()> {
        self.step += 1;
        if self.step == self.fail_at {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        Ok(())
    }
}

impl Write for FailAt {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.tick()?;
        self.written.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.tick()?;
        self.flushes += 1;
        Ok(())
    }
}

/// One record, one newline, flushed, so a piped consumer sees each line as it lands.
#[test]
fn write_record_appends_exactly_one_newline_and_flushes() {
    let mut out = FailAt::new(usize::MAX);
    assert!(write_record(&mut out, b"{\"n\":0}"));
    assert_eq!(out.written, b"{\"n\":0}\n");
    assert_eq!(out.flushes, 1, "a record left sitting in a buffer");
}

/// A pipe closed at any of the three steps is a clean stop, not exit 1.
#[test]
fn write_record_reports_a_closed_reader_at_every_step() {
    for fail_at in 1..=3 {
        let mut out = FailAt::new(fail_at);
        assert!(
            !write_record(&mut out, b"{\"n\":0}"),
            "step {fail_at}: a closed reader was not reported as a stop"
        );
    }
}

/// A failed record is not followed by a newline, which would read as a blank document.
#[test]
fn write_record_writes_no_newline_when_the_record_itself_fails() {
    let mut out = FailAt::new(1);
    assert!(!write_record(&mut out, b"{\"n\":0}"));
    assert!(out.written.is_empty(), "wrote {:?}", out.written);
    assert_eq!(out.flushes, 0);
}

/// Partial writes are retried: a one-byte-at-a-time writer still gets the record and a newline.
#[test]
fn write_record_survives_a_writer_that_accepts_one_byte_at_a_time() {
    struct Dribble(Vec<u8>);
    impl Write for Dribble {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.extend_from_slice(&buf[..1]);
            Ok(1)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let mut out = Dribble(Vec::new());
    assert!(write_record(&mut out, b"{\"n\":0}"));
    assert_eq!(out.0, b"{\"n\":0}\n");
}

// ── read_line_bounded ────────────────────────────────────────────

#[test]
fn read_line_bounded_normal_lines() {
    let input = Cursor::new(b"hello\nworld\n");
    let mut reader = BufReader::new(input);
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, b"hello");

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, b"world");

    assert!(!read_line_bounded(&mut reader, &mut buf).unwrap());
}

#[test]
fn read_line_bounded_crlf() {
    let input = Cursor::new(b"line\r\n");
    let mut reader = BufReader::new(input);
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, b"line"); // \r stripped
}

/// SSE never dispatches an unterminated line; emitting it would put half a JSON document on
/// stdout under exit 0.
#[test]
fn read_line_bounded_discards_an_unterminated_final_line() {
    let input = Cursor::new(b"whole\npartial");
    let mut reader = BufReader::new(input);
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, b"whole");

    assert!(!read_line_bounded(&mut reader, &mut buf).unwrap());
    assert!(buf.is_empty(), "the partial line leaked: {buf:?}");
}

/// A mid-line read error must not look like EOF, which ends the stream at exit 0.
#[test]
fn read_line_bounded_propagates_a_read_error() {
    struct Reset;
    impl io::Read for Reset {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::ErrorKind::ConnectionReset.into())
        }
    }

    let mut reader = BufReader::new(Reset);
    let mut buf = Vec::new();

    let err = read_line_bounded(&mut reader, &mut buf).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::ConnectionReset);
    assert!(!is_timeout(&err), "a reset would be reported as a stall");
}

/// The `\r` strip runs on the assembled line, so a `\r\n` split across fills is still handled.
#[test]
fn read_line_bounded_strips_a_cr_split_across_a_buffer_boundary() {
    let input = Cursor::new(b"ab\r\n");
    let mut reader = BufReader::with_capacity(3, input); // fills as "ab\r", then "\n"
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, b"ab");
}

/// Only the terminator's `\r` is stripped; an earlier one is content.
#[test]
fn read_line_bounded_strips_only_the_last_cr() {
    let input = Cursor::new(b"a\r\r\n");
    let mut reader = BufReader::new(input);
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, b"a\r");
}

/// The buffer is reused across records, so leftovers must not prefix the next line.
#[test]
fn read_line_bounded_clears_a_dirty_buffer_on_entry() {
    let input = Cursor::new(b"fresh\n");
    let mut reader = BufReader::new(input);
    let mut buf = b"stale".to_vec();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, b"fresh");
}

/// A bare `\r\n` event separator reads as an empty line, not a lone `\r`.
#[test]
fn read_line_bounded_reads_a_bare_crlf_as_an_empty_line() {
    let input = Cursor::new(b"\r\ndata: x\n");
    let mut reader = BufReader::new(input);
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert!(buf.is_empty(), "got {buf:?}");
}

#[test]
fn read_line_bounded_empty_lines() {
    let input = Cursor::new(b"\n\ndata\n");
    let mut reader = BufReader::new(input);
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, b"");

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, b"");

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, b"data");
}

#[test]
fn read_line_bounded_rejects_oversized_line() {
    let oversized = vec![b'x'; MAX_SSE_LINE_SIZE + 1];
    let input = Cursor::new(oversized);
    let mut reader = BufReader::new(input);
    let mut buf = Vec::new();

    let err = read_line_bounded(&mut reader, &mut buf).unwrap_err();
    assert!(
        err.to_string().contains("exceeds maximum size"),
        "expected size limit error, got: {err}"
    );
    // Both errors are `ErrorKind::Other`; a kind check would print the --timeout hint here.
    assert!(!is_timeout(&err));
}

/// The cap is checked only between fills, so one fill holding the overrun and its newline is
/// taken whole. The memory bound relies on the production `BufReader`'s fill size.
#[test]
fn read_line_bounded_checks_the_cap_only_between_chunks() {
    let mut data = vec![b'x'; MAX_SSE_LINE_SIZE + 1];
    data.push(b'\n');
    // A bare Cursor is its own BufRead: fill_buf hands back everything at once.
    let mut reader = Cursor::new(data);
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf.len(), MAX_SSE_LINE_SIZE + 1);
}

#[test]
fn read_line_bounded_accepts_line_at_limit() {
    let mut data = vec![b'x'; MAX_SSE_LINE_SIZE];
    data.push(b'\n');
    let input = Cursor::new(data);
    let mut reader = BufReader::new(input);
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf.len(), MAX_SSE_LINE_SIZE);
}

#[test]
fn read_line_bounded_utf8_multibyte() {
    // Multi-byte UTF-8 (é = 0xC3 0xA9, 🦀 = 0xF0 0x9F 0xA6 0x80)
    let input = Cursor::new("café 🦀\n".as_bytes().to_vec());
    let mut reader = BufReader::new(input);
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, "café 🦀".as_bytes());
}

#[test]
fn read_line_bounded_utf8_across_buffer_boundary() {
    // Capacity 2 splits é (C3 A9) across fills; per-chunk UTF-8 validation would reject it.
    let data = "hé\n".as_bytes().to_vec();
    let input = Cursor::new(data);
    let mut reader = BufReader::with_capacity(2, input);
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, "hé".as_bytes());
}

#[test]
fn read_line_bounded_binary_passthrough() {
    // Non-UTF-8 bytes pass through.
    let input = Cursor::new(vec![0xFF, 0xFE, b'\n']);
    let mut reader = BufReader::new(input);
    let mut buf = Vec::new();

    assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
    assert_eq!(buf, vec![0xFF, 0xFE]);
}

/// A body over the limit is flagged `oversized` through the real read loop, since that is what
/// the operator can act on. A body exactly at the limit is read whole: `LimitReader` errors when
/// nothing is left on entry and `read_to_end` always reads once more for EOF, hence the `+1`.
#[test]
fn an_oversized_body_is_detected_by_the_read_loop_and_flagged() {
    use std::io::Write as _;
    use std::net::TcpListener;

    for (len, want_body, want_oversized) in [(4096usize, 4096usize, false), (4097, 0, true)] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Some(sock) = listener.incoming().flatten().next() {
                let mut sock = sock;
                let mut junk = String::new();
                let mut r = BufReader::new(sock.try_clone().unwrap());
                loop {
                    junk.clear();
                    if r.read_line(&mut junk).unwrap_or(0) == 0 || junk.trim().is_empty() {
                        break;
                    }
                }
                let _ = sock.write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n"
                    )
                    .as_bytes(),
                );
                let _ = sock.write_all(&vec![b'x'; len]);
            }
        });
        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .http_status_as_error(false)
                .timeout_global(Some(Duration::from_secs(10)))
                .build(),
        );
        let r = get_raw(&agent, &format!("http://127.0.0.1:{port}/"), &[], 4096).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.body.len(), want_body, "body of {len} against a 4096 cap");
        assert_eq!(r.oversized(), want_oversized, "body of {len}");
    }
}
