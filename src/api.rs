use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use ureq::tls::{PemItem, RootCerts, TlsConfig};

const USER_AGENT: &str = concat!("bx/", env!("CARGO_PKG_VERSION"));

/// The largest CA bundle bx reads. Reading stops one byte past this bound, so the
/// cap holds for a file of any size — including special files whose metadata
/// reports a length of zero.
const MAX_CA_BUNDLE_SIZE: usize = 5 * 1024 * 1024;

/// Process-wide TLS configuration. The CLI initializes this at most once,
/// before any request is made.
static TLS_CONFIG: OnceLock<TlsConfig> = OnceLock::new();

/// Cap on one SSE line: a peer can withhold the newline forever, so the line buffer needs a
/// bound of its own. Checked between fills, so the real ceiling is this plus one `BufReader`
/// fill — enough for the memory bound, which is the point.
const MAX_SSE_LINE_SIZE: usize = 1024 * 1024; // 1 MB

/// Reads a single line with a size cap. Returns `false` at EOF, discarding (and warning about)
/// a final line that never reached its terminator.
///
/// Uses raw bytes (`Vec<u8>`) because `fill_buf()` can split multi-byte
/// UTF-8 sequences at its 8 KB buffer boundary, which would cause
/// spurious `from_utf8` errors if we validated per chunk.
fn read_line_bounded<R: BufRead>(reader: &mut R, buf: &mut Vec<u8>) -> io::Result<bool> {
    buf.clear();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            // EOF mid-line: SSE never dispatches an unterminated line, and emitting one puts
            // half a JSON document on a stream that promises one per line. Say so rather than
            // lose it quietly — under Content-Length or chunked framing a clean EOF means the
            // body was complete, so this really is a record the server failed to terminate.
            if !buf.is_empty() {
                eprintln!(
                    "warning: discarded {} bytes of an unterminated final line",
                    buf.len()
                );
                buf.clear();
            }
            return Ok(false);
        }
        if let Some(pos) = available.iter().position(|&b| b == b'\n') {
            buf.extend_from_slice(&available[..pos]);
            reader.consume(pos + 1);
            // Strip trailing \r for \r\n sequences
            if buf.last() == Some(&b'\r') {
                buf.pop();
            }
            return Ok(true);
        }
        // No newline yet — append whole buffer
        let len = available.len();
        buf.extend_from_slice(available);
        reader.consume(len);
        if buf.len() > MAX_SSE_LINE_SIZE {
            return Err(io::Error::other(format!(
                "SSE line exceeds maximum size ({MAX_SSE_LINE_SIZE} bytes)"
            )));
        }
    }
}

/// Configures an explicit PEM CA bundle for all HTTP agents.
///
/// This is opt-in. When unset, ureq retains its default WebPKI roots. When set,
/// the supplied certificates become the complete trust-root set; certificate
/// and hostname verification remain enabled.
pub fn configure_ca_bundle(path: &Path) -> Result<(), String> {
    let tls_config = load_ca_bundle(path)?;
    TLS_CONFIG
        .set(tls_config)
        .map_err(|_| "TLS configuration was already initialized".to_string())
}

fn load_ca_bundle(path: &Path) -> Result<TlsConfig, String> {
    let read_err = |e| format!("cannot read CA bundle {}: {e}", path.display());
    let mut pem = Vec::new();
    fs::File::open(path)
        .map_err(read_err)?
        .take(MAX_CA_BUNDLE_SIZE as u64 + 1)
        .read_to_end(&mut pem)
        .map_err(read_err)?;
    if pem.len() > MAX_CA_BUNDLE_SIZE {
        return Err(format!(
            "CA bundle {} exceeds the {} MiB limit",
            path.display(),
            MAX_CA_BUNDLE_SIZE / (1024 * 1024)
        ));
    }

    let mut certs = Vec::new();
    for item in ureq::tls::parse_pem(&pem) {
        match item.map_err(|e| format!("invalid CA bundle {}: {e}", path.display()))? {
            PemItem::Certificate(cert) => certs.push(cert),
            PemItem::PrivateKey(_) => {
                return Err(format!(
                    "CA bundle {} contains a private key",
                    path.display()
                ));
            }
            // PemItem is #[non_exhaustive]; it yields certificates and keys today.
            _ => {}
        }
    }

    if certs.is_empty() {
        return Err(format!(
            "CA bundle {} contains no PEM certificates",
            path.display()
        ));
    }

    Ok(TlsConfig::builder()
        .root_certs(RootCerts::new_with_certs(&certs))
        .build())
}

fn tls_config() -> TlsConfig {
    TLS_CONFIG.get().cloned().unwrap_or_default()
}

fn agent(timeout_secs: u64) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .tls_config(tls_config())
            .http_status_as_error(false)
            .max_redirects(0)
            .timeout_global(Some(Duration::from_secs(timeout_secs)))
            .user_agent(USER_AGENT)
            .build(),
    )
}

/// Agent for SSE streaming — research responses run for minutes, so no total deadline.
///
/// `timeout_recv_response` must stay unset: ureq 3.3.0 keeps checking it while reading the
/// body, measured from the headers, so it would cap the whole stream. `timeout_recv_body` is
/// re-armed on every read, so it bounds silence instead. ureq#1194 (merged, unreleased as of
/// 3.4.0) fixes that leak *and* makes `recv_body` a total budget, which would break this
/// design — `stream_outlives_the_timeout_…` is the tripwire.
///
/// `dial_secs` covers everything before the first body byte, including the wait for headers,
/// which inherits the send-phase deadlines. Only silence mid-stream earns `read_secs`.
fn streaming_agent(dial_secs: u64, read_secs: u64) -> ureq::Agent {
    let dial = Some(Duration::from_secs(dial_secs));
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .tls_config(tls_config())
            .http_status_as_error(false)
            .max_redirects(0)
            .timeout_resolve(dial)
            .timeout_connect(dial)
            .timeout_send_request(dial)
            .timeout_send_body(dial)
            .timeout_recv_body(Some(Duration::from_secs(read_secs)))
            .user_agent(USER_AGENT)
            .build(),
    )
}

/// Maps an HTTP status code to a process exit code.
///   0 = success
///   1 = the request was made; the result is unusable
///   2 = (reserved — clap argument parsing)
///   3 = auth/permission error (401, 403)
///   4 = rate limited (429)
///   5 = server/network error (5xx, timeouts)
fn exit_code_for_status(status: u16) -> i32 {
    match status {
        401 | 403 => 3,
        429 => 4,
        500..=599 => 5,
        _ => 1,
    }
}

/// Formats an API error response for stderr output.
/// Returns the formatted message and the appropriate exit code.
fn format_error(status: u16, body: &str) -> (String, i32) {
    // Only an actual error envelope goes through the envelope formatter: any other JSON — a
    // redirect body, a 4xx from a proxy that does not speak Brave — would come out as
    // "unknown ({status})", which says less than the status alone.
    let mut msg = match serde_json::from_str::<serde_json::Value>(body) {
        Ok(v) if v["error"].is_object() => {
            let code = v["error"]["code"].as_str().unwrap_or("UNKNOWN");
            let detail = v["error"]["detail"].as_str().unwrap_or("");
            if detail.is_empty() {
                format!("{} ({status})", code.to_lowercase().replace('_', " "))
            } else {
                format!(
                    "{} ({status}) — {detail}",
                    code.to_lowercase().replace('_', " ")
                )
            }
        }
        _ => format!("HTTP {status}"),
    };

    match status {
        401 => msg.push_str("\nhint: check your API key with `bx config show-key`"),
        403 => msg.push_str("\nhint: this endpoint may require a different API plan"),
        429 => msg
            .push_str("\nhint: retry after a short delay, or upgrade plan for higher rate limits"),
        _ => {}
    }

    (msg, exit_code_for_status(status))
}

/// Maximum bytes of raw response body to write to stderr on errors.
const MAX_ERROR_BODY_DISPLAY: usize = 1024;
/// Maximum bytes to read for non-streaming API responses.
const MAX_RESPONSE_BODY_SIZE: u64 = 3 * 1024 * 1024; // 3 MB

/// Writes a response body to stderr, truncated for safety.
fn write_body_stderr(body: &str) {
    let stderr = io::stderr();
    let mut err = stderr.lock();
    if body.len() > MAX_ERROR_BODY_DISPLAY {
        err.write_all(&body.as_bytes()[..MAX_ERROR_BODY_DISPLAY])
            .ok();
        let _ = write!(err, "\n... [truncated, {} bytes total]\n", body.len());
    } else {
        err.write_all(body.as_bytes()).ok();
        if !body.ends_with('\n') {
            err.write_all(b"\n").ok();
        }
    }
}

/// True if an I/O error is a ureq timeout. ureq wraps those as `ErrorKind::Other`,
/// so the kind alone cannot distinguish them.
fn is_timeout(e: &io::Error) -> bool {
    e.get_ref()
        .and_then(|inner| inner.downcast_ref::<ureq::Error>())
        .is_some_and(|e| matches!(e, ureq::Error::Timeout(_)))
}

/// Prints an error message + raw body to stderr and exits.
fn write_error_and_exit(status: u16, body: &str) -> ! {
    let (msg, code) = format_error(status, body);
    eprintln!("error: {msg}");
    write_body_stderr(body);
    std::process::exit(code);
}

fn read_body_or_exit(resp: ureq::http::Response<ureq::Body>) -> (u16, String) {
    let status = resp.status().as_u16();
    // +1 works around ureq LimitReader rejecting bodies exactly at the limit
    let body = match resp
        .into_body()
        .into_with_config()
        .limit(MAX_RESPONSE_BODY_SIZE + 1)
        .read_to_string()
    {
        Ok(body) => body,
        Err(e) => {
            // An unreadable body (oversized, or not UTF-8 — ureq decodes lossily only for
            // `text/*`) must not cost the status its exit code: a 401 reported as a network
            // error sends an agent into backoff instead of fixing its key.
            eprintln!("error: failed to read response body: {e}");
            std::process::exit(match status {
                200..=299 => 5,
                _ => exit_code_for_status(status),
            });
        }
    };
    (status, body)
}

/// True if `bytes` is exactly one JSON document. stdout promises one per line, and a first-byte
/// sniff still lets `{oops` — or a proxy's HTML error page inside a 200 — reach an agent's
/// parser. `IgnoredAny` validates without building a `Value`; `from_slice` rejects trailing
/// content, and rejects invalid UTF-8 for free.
fn is_json(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde::de::IgnoredAny>(bytes).is_ok()
}

fn handle_response(resp: ureq::http::Response<ureq::Body>) {
    let (status, body) = read_body_or_exit(resp);

    // Redirects are not followed, so a 3xx body is never an answer either.
    if !matches!(status, 200..=299) {
        write_error_and_exit(status, &body);
    }

    let trimmed = body.trim();
    if !is_json(trimmed.as_bytes()) {
        eprintln!("error: unexpected non-JSON response");
        write_body_stderr(&body);
        std::process::exit(1);
    }

    // A pretty-printed body still spans several lines — compacting it would mean re-serialising,
    // which this pass-through does not do. A closed reader is a clean stop, hence `let _`.
    let _ = write_record(&mut io::stdout().lock(), trimmed.as_bytes());
}

/// Writes one record and a newline, flushed. Returns `false` if the reader is gone — a clean
/// stop; any other failure is fatal, because output silently lost under exit 0 is the worst
/// thing this CLI can hand an agent.
///
/// The `flush` is a no-op for today's `StdoutLock` (unconditionally a `LineWriter`), but std
/// documents that only for a terminal (rust-lang/rust#60673).
fn write_record(out: &mut impl Write, record: &[u8]) -> bool {
    let written = out
        .write_all(record)
        .and_then(|()| out.write_all(b"\n"))
        .and_then(|()| out.flush());
    match written {
        Ok(()) => true,
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => false,
        Err(e) => {
            eprintln!("error: writing to stdout: {e}");
            std::process::exit(1);
        }
    }
}

/// Sends a GET request and prints the response body to stdout.
pub fn get(base_url: &str, path: &str, api_key: &str, timeout: u64) {
    get_with_headers(base_url, path, api_key, &[], timeout);
}

/// Sends a GET request with additional headers.
pub fn get_with_headers(
    base_url: &str,
    path: &str,
    api_key: &str,
    headers: &[(&str, &str)],
    timeout: u64,
) {
    let url = format!("{base_url}{path}");
    let mut req = agent(timeout)
        .get(&url)
        .header("X-Subscription-Token", api_key);
    for &(k, v) in headers {
        req = req.header(k, v);
    }

    match req.call() {
        Ok(resp) => handle_response(resp),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(5);
        }
    }
}

/// Sends a POST request with a JSON body and prints the response to stdout.
pub fn post_json(
    base_url: &str,
    path: &str,
    api_key: &str,
    body: &serde_json::Value,
    headers: &[(&str, &str)],
    timeout: u64,
) {
    let url = format!("{base_url}{path}");
    let mut req = agent(timeout)
        .post(&url)
        .header("X-Subscription-Token", api_key)
        .header("Content-Type", "application/json");
    for &(k, v) in headers {
        req = req.header(k, v);
    }

    let payload = serde_json::to_string(body).expect("failed to serialize JSON body");

    match req.send(payload.as_bytes()) {
        Ok(resp) => handle_response(resp),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(5);
        }
    }
}

/// Payload of an SSE `data:` line; `None` for anything else and for the empty payloads that
/// dispatch no event (WHATWG 9.2.6).
///
/// Two deliberate deviations from the spec, both safe only because payloads are JSON: it trims
/// ASCII whitespace where the spec removes exactly one U+0020, which is what catches a padded
/// `data: [DONE] ` and whitespace-only heartbeats; and it strips a BOM from any line, not once
/// per stream, because a stream-leading BOM would otherwise hide the first record entirely.
fn sse_data(line: &[u8]) -> Option<&[u8]> {
    let line = line.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(line);
    let data = line.strip_prefix(b"data:")?.trim_ascii();
    (!data.is_empty()).then_some(data)
}

/// Sends a POST request with a JSON body and streams the SSE response line-by-line. Each
/// `data:` record is printed to stdout, one JSON document per line. Stops at `data: [DONE]`,
/// and exits 1 if the response carried no records.
///
/// `dial_timeout` bounds getting to the first byte; `read_timeout` bounds silence after that.
pub fn post_json_stream(
    base_url: &str,
    path: &str,
    api_key: &str,
    body: &serde_json::Value,
    dial_timeout: u64,
    read_timeout: u64,
) {
    let url = format!("{base_url}{path}");
    let req = streaming_agent(dial_timeout, read_timeout)
        .post(&url)
        .header("X-Subscription-Token", api_key)
        .header("Content-Type", "application/json");

    let payload = serde_json::to_string(body).expect("failed to serialize JSON body");

    match req.send(payload.as_bytes()) {
        Ok(resp) => {
            let status = resp.status().as_u16();
            // Redirects are not followed, so a 3xx body is not an answer however much it looks
            // like one — and a 3xx carrying `data:` lines used to stream them out under exit 0.
            if !matches!(status, 200..=299) {
                let (_, body) = read_body_or_exit(resp);
                write_error_and_exit(status, &body);
            }

            let (_, body) = resp.into_parts();
            let mut reader = BufReader::new(body.into_reader());
            let mut out = io::stdout().lock();
            let mut line: Vec<u8> = Vec::new();
            let mut emitted = false;

            loop {
                let has_data = match read_line_bounded(&mut reader, &mut line) {
                    Ok(has) => has,
                    Err(e) => {
                        if is_timeout(&e) {
                            eprintln!(
                                "error: no data for {read_timeout}s\n\
                                 hint: research answers pause for minutes — raise --timeout"
                            );
                        } else {
                            eprintln!("error: reading stream: {e}");
                        }
                        std::process::exit(5);
                    }
                };
                if !has_data {
                    break;
                }

                let Some(data) = sse_data(&line) else {
                    continue;
                };
                if data == b"[DONE]" {
                    break;
                }
                if !is_json(data) {
                    eprintln!("error: unexpected non-JSON record in stream");
                    write_body_stderr(&String::from_utf8_lossy(data));
                    std::process::exit(1);
                }
                // A closed reader (`bx answers … | head`): draining the rest burns minutes
                // and quota. Returning also skips the check below, which would be a lie.
                if !write_record(&mut out, data) {
                    return;
                }
                emitted = true;
            }

            // Exit 0 with empty stdout is indistinguishable from an answer that had nothing
            // to say; the status separates a redirect or a 204 from a 200 that ignored
            // `stream`.
            if !emitted {
                eprintln!(
                    "error: HTTP {status} carried no SSE records\n\
                     hint: check --base-url / --endpoint, or retry with --no-stream"
                );
                std::process::exit(1);
            }
        }
        // With `timeout_recv_response` unset the header wait inherits the send-phase
        // deadline, so ureq blames `send request` for a request that was sent and never
        // answered.
        Err(ureq::Error::Timeout(ureq::Timeout::SendRequest)) => {
            eprintln!("error: no response headers within {dial_timeout}s");
            std::process::exit(5);
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(5);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use std::io::Cursor;

    // ── agent configuration ──────────────────────────────────────────

    /// Pins the timeouts a stream depends on. `recv_response`, `global` and `per_call` are the
    /// three that silently cap a stream no matter how much data flows.
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

    /// Only silence mid-stream earns the research budget. Spending it on DNS, connect or the
    /// wait for headers turns a blackholed host into a five-minute hang.
    #[test]
    fn streaming_agent_keeps_the_research_budget_off_the_dial_phases() {
        let timeouts = streaming_agent(30, 300).config().timeouts();
        let dial = Some(Duration::from_secs(30));

        assert_eq!(timeouts.resolve, dial);
        assert_eq!(timeouts.connect, dial);
        // The wait for response headers inherits these two, and the rustls handshake runs on
        // `send_request` (ureq timings.rs: RecvResponse => [SendRequest, SendBody]).
        assert_eq!(timeouts.send_request, dial);
        assert_eq!(timeouts.send_body, dial);
    }

    /// An explicit `--timeout` is the user saying how patient they are; it reaches every phase.
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

    /// The blocking agent is the opposite trade: one total deadline, and no phase deadline that
    /// could fire earlier than the user asked for.
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

    /// stdout promises one JSON document per line. The first-byte sniff this replaced let
    /// `{oops` through, and the streaming path had no guard at all — so a proxy's plaintext
    /// error inside a 200 reached an agent's parser under exit 0.
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

    /// The two paths trim differently, and `is_json` is where that shows. `str::trim` follows
    /// Unicode, so it removes U+00A0 before the guard ever runs; `sse_data`'s `trim_ascii` does
    /// not, so an NBSP-padded record reaches `is_json` intact and is rejected. Neither is wrong
    /// — but a future "unify the trimming" change would move a case across this line.
    #[test]
    fn is_json_sees_what_each_path_left_behind() {
        assert!(
            is_json("\u{a0}{}".trim().as_bytes()),
            "str::trim takes NBSP"
        );
        assert!(!is_json("\u{a0}{}".as_bytes()), "trim_ascii leaves it");
        // A BOM survives `str::trim` (U+FEFF is not White_Space), so a BOM-prefixed body is a
        // non-JSON response on the blocking path — where `sse_data` would have stripped it.
        assert_eq!("\u{feff}{}".trim(), "\u{feff}{}");
    }

    // ── sse_data ─────────────────────────────────────────────────────

    /// Every field shape a server can send, in one table. stdout promises one JSON document
    /// per line, so a misfiled line either injects a non-JSON line into an agent's parser or
    /// silently drops an answer chunk.
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
            // read_line_bounded strips one trailing \r; a second one used to ride along.
            (b"data: {\"n\":0}\r", Some(json)),
            // The sentinel, however it is padded. The last three used to be emitted as
            // records, and the stream then ran on to EOF.
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
            // Empty and whitespace-only payloads dispatch no event (WHATWG 9.2.6). The last
            // two used to reach stdout as a near-blank line.
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

    /// The spec strips one BOM at the start of the stream. Without this the first record
    /// hides behind it and a whole answer reads as "no SSE records"; a BOM *inside* a
    /// payload is content and must survive.
    #[test]
    fn sse_data_strips_a_leading_bom_but_not_one_inside_the_payload() {
        assert_eq!(sse_data("\u{feff}data: {}".as_bytes()), Some(&b"{}"[..]));
        assert_eq!(
            sse_data("data: \u{feff}{}".as_bytes()),
            Some("\u{feff}{}".as_bytes())
        );
    }

    /// The spec strips the BOM once, when decoding the stream; bx strips it per line, which is
    /// laxer — a conforming client would read `\u{feff}data` as an unknown field and ignore the
    /// line. Delivering the record instead is over-permissive, never lossy.
    #[test]
    fn sse_data_strips_a_bom_from_any_line_not_just_the_first() {
        assert_eq!(sse_data("\u{feff}data: {}".as_bytes()), Some(&b"{}"[..]));
        // …but only one. A second BOM sits where the field name should be.
        assert_eq!(sse_data("\u{feff}\u{feff}data: {}".as_bytes()), None);
    }

    /// Without the space the BOM is unambiguously payload — the form a broken encoder produces,
    /// and the one that must survive so the record reaches stdout byte-for-byte.
    #[test]
    fn sse_data_keeps_a_bom_that_begins_the_payload() {
        assert_eq!(
            sse_data("data:\u{feff}{}".as_bytes()),
            Some("\u{feff}{}".as_bytes())
        );
    }

    /// `trim_ascii` cannot see non-ASCII whitespace, so an NBSP-padded sentinel is not the
    /// sentinel: it is emitted as a record and the stream runs on to EOF.
    #[test]
    fn sse_data_does_not_recognise_a_sentinel_padded_with_non_ascii_space() {
        assert_eq!(
            sse_data("data: [DONE]\u{a0}".as_bytes()),
            Some("[DONE]\u{a0}".as_bytes())
        );
    }

    /// Pins the exact trim set. `trim_ascii` is ASCII-only, so it takes form feed (which is
    /// not JSON whitespace) and leaves vertical tab and U+00A0 (which are not ASCII
    /// whitespace) — all of which only ever appear in an already-invalid payload.
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

    /// The exit code is a contract agents branch on: 3 means fix your key, 4 means back off,
    /// 5 means retry later, 1 means the request happened and the result is unusable.
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

    /// Only a real error envelope goes through the envelope formatter. Any other JSON — a
    /// redirect body, a proxy's own error shape — used to come out as `unknown (302)`, which
    /// says strictly less than the status it replaced.
    #[test]
    fn format_error_uses_the_envelope_only_when_there_is_one() {
        let (msg, code) = format_error(302, r#"{"redirect":1}"#);
        assert_eq!(msg, "HTTP 302");
        assert_eq!(code, 1);

        for body in ["not json at all", "", "[]", r#"{"error":"a string"}"#] {
            assert_eq!(format_error(500, body).0, "HTTP 500", "{body}");
        }
    }

    /// The envelope's own shapes: code plus detail, code alone, and the placeholder for an
    /// envelope that carries neither.
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

    /// ureq hides timeouts inside `ErrorKind::Other`, which is where our own size-limit
    /// error lands too. Confusing them tells an agent to raise --timeout for a failure no
    /// timeout can fix — and since both read failures now share one arm, this predicate is
    /// the only thing keeping them apart.
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

    /// ureq wraps exactly once (`Error::into_io`), and `From<io::Error> for Error` unwraps
    /// rather than nesting — so a second layer never comes from ureq, and peeling one would
    /// mean inventing a recursion the error type does not have.
    #[test]
    fn is_timeout_does_not_peel_a_second_layer_of_wrapping() {
        let once = io::Error::other(ureq::Error::Timeout(ureq::Timeout::RecvBody));
        assert!(!is_timeout(&io::Error::other(once)));
    }

    /// Any phase's timeout is still a timeout. The stall message is right for `RecvBody`, but
    /// classifying the others as "not a timeout" would print a raw ureq string instead.
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

    /// A writer that fails at one chosen step. `write_record` performs three in order — the
    /// record, the newline, the flush — and a `| head` can break the pipe at any of them.
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

    /// One record, one newline, flushed — a piped consumer must see each line as it lands,
    /// not when a buffer somewhere happens to fill.
    #[test]
    fn write_record_appends_exactly_one_newline_and_flushes() {
        let mut out = FailAt::new(usize::MAX);
        assert!(write_record(&mut out, b"{\"n\":0}"));
        assert_eq!(out.written, b"{\"n\":0}\n");
        assert_eq!(out.flushes, 1, "a record left sitting in a buffer");
    }

    /// `bx answers … | head -1` can break the pipe at any of the three steps. A missing
    /// `false` at any one of them turns a clean stop into exit 1 "writing to stdout".
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

    /// `and_then` short-circuits: a record that could not be written must not be followed by a
    /// newline, or the reader sees a blank line where a document should have been.
    #[test]
    fn write_record_writes_no_newline_when_the_record_itself_fails() {
        let mut out = FailAt::new(1);
        assert!(!write_record(&mut out, b"{\"n\":0}"));
        assert!(out.written.is_empty(), "wrote {:?}", out.written);
        assert_eq!(out.flushes, 0);
    }

    /// A partial write is not an error — `write_all` loops. A writer that dribbles one byte at a
    /// time must still produce exactly the record and one newline.
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

    const TEST_CA_PEM: &str = r#"-----BEGIN CERTIFICATE-----
MIIDCzCCAfOgAwIBAgIUBfAmNZdoIAZsb4Q478qFjzIQY+4wDQYJKoZIhvcNAQEL
BQAwFTETMBEGA1UEAwwKYngtdGVzdC1jYTAeFw0yNjA4MjYxNjAwMzZaFw0zNjA4
MjMxNjAwMzZaMBUxEzARBgNVBAMMCmJ4LXRlc3QtY2EwggEiMA0GCSqGSIb3DQEB
AQUAA4IBDwAwggEKAoIBAQCokCDFgvDNkWRgOIir7dOh63rGq8ECuoXXa53iniAA
uipw4LBP5D4xSgMbGK7fJlgJcYlIom4GkNytxes8oqwlC1zl3Aal5X3gv4Zof481
ow7xWRjzoXPHXLyJtT/sBkMo44x1lXG4usS+PiMEt2SDrAdrChPXGDlF/Fs+LKfG
cNJDeNe7nMHS5LZZR4Cn9f1tMPm3hHwEnLnx+P4cADcySd0pQAwWXxEj/Qr0pXXh
Az+9m+Em8Kx6ajrV1J169Vi48BFhlrDynt4BFngaJiBGiAbKJEZ/yxGsw4q8WCjM
CzQR9vLaNvAQzrEa0i0NP6gaiZer4RqdyFtrVvDBdhKzAgMBAAGjUzBRMB0GA1Ud
DgQWBBQh8EWmPu1uuigO+J/6VHyqCQ6qATAfBgNVHSMEGDAWgBQh8EWmPu1uuigO
+J/6VHyqCQ6qATAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQA0
cv47BeHkoI/mWEi42SkyZUSwjpElDqrXVW3EWuB+QOD44GTi7mPz0evBls/H8oCu
XakAfe6IPZwCZuc9UDnJ/Y7B+PhER6+hxlak/JRx/Zv19FbNRVPUoy7BbG9VyGLk
GtP57plpPHmhfO4BOpR7AfrtUPVxcgPvL9klPML71pXNrTxhXn15VqSL5gcjQhmj
S5sByNCzf4QBSL0J9qKKih7d/aex9V5LbfR8WMUqSmmFhgbXPaKRGzvTt42TlwUh
N+91wkWoXD9AwvA7U5gI4x7IsiD24vVyveZ1uT02tsGjl3QP6n/ziKyauJm3r0DT
kL94O//ACgZIe4oyVogm
-----END CERTIFICATE-----
"#;

    /// Wraps DER bytes in a PEM section carrying `label`.
    fn pem_block(label: &str, der: &[u8]) -> String {
        let body = base64::engine::general_purpose::STANDARD.encode(der);
        format!("-----BEGIN {label}-----\n{body}\n-----END {label}-----\n")
    }

    /// A PKCS#8 Ed25519 structure whose 32-byte seed is all zeros. `parse_pem`
    /// classifies a section by its label and base64-decodes the body without
    /// inspecting it, so these bytes reach the private-key branch of the guard.
    fn zeroed_pkcs8_der() -> Vec<u8> {
        let mut der = vec![
            0x30, 0x2e, // SEQUENCE, 46 bytes
            0x02, 0x01, 0x00, // INTEGER 0 — PKCS#8 version
            0x30, 0x05, // SEQUENCE, 5 bytes — AlgorithmIdentifier
            0x06, 0x03, 0x2b, 0x65, 0x70, // OID 1.3.101.112 — Ed25519
            0x04, 0x22, // OCTET STRING, 34 bytes — privateKey
            0x04, 0x20, // OCTET STRING, 32 bytes — Ed25519 seed
        ];
        der.extend_from_slice(&[0u8; 32]);
        der
    }

    #[test]
    fn load_ca_bundle_accepts_certificate_pem() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ca.pem");
        fs::write(&path, TEST_CA_PEM).unwrap();

        let config = load_ca_bundle(&path).unwrap();
        assert!(matches!(config.root_certs(), RootCerts::Specific(_)));
    }

    #[test]
    fn load_ca_bundle_rejects_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ca.pem");
        fs::write(&path, "").unwrap();

        let error = load_ca_bundle(&path).unwrap_err();
        assert!(error.contains("contains no PEM certificates"));
    }

    #[test]
    fn load_ca_bundle_rejects_malformed_pem() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ca.pem");
        fs::write(
            &path,
            "-----BEGIN CERTIFICATE-----\nnot-base64\n-----END CERTIFICATE-----\n",
        )
        .unwrap();

        let error = load_ca_bundle(&path).unwrap_err();
        assert!(error.contains("invalid CA bundle"));
    }

    #[test]
    fn load_ca_bundle_rejects_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.pem");

        let error = load_ca_bundle(&path).unwrap_err();
        assert!(error.contains("cannot read CA bundle"));
    }

    #[test]
    fn load_ca_bundle_rejects_oversize_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ca.pem");
        fs::write(&path, vec![b'x'; MAX_CA_BUNDLE_SIZE + 1]).unwrap();

        let error = load_ca_bundle(&path).unwrap_err();
        assert!(error.contains("exceeds the 5 MiB limit"), "{error}");
    }

    #[test]
    fn load_ca_bundle_rejects_private_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ca.pem");
        fs::write(&path, pem_block("PRIVATE KEY", &zeroed_pkcs8_der())).unwrap();

        let error = load_ca_bundle(&path).unwrap_err();
        assert!(error.contains("contains a private key"));
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

    /// A line that never reached its terminator is a truncated event, and SSE never
    /// dispatches one. Emitting it put half a JSON document on stdout under exit 0 — the
    /// one remaining path where this CLI produced corrupt output and called it success.
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

    /// A reader that fails mid-line must not look like EOF: EOF ends the stream at exit 0,
    /// an error has to reach the exit-5 arm.
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

    /// `fill_buf` can hand back a `\r` and its `\n` in two separate chunks. The strip runs after
    /// the line is assembled, so it still finds it — doing it per chunk would not.
    #[test]
    fn read_line_bounded_strips_a_cr_split_across_a_buffer_boundary() {
        let input = Cursor::new(b"ab\r\n");
        let mut reader = BufReader::with_capacity(3, input); // fills as "ab\r", then "\n"
        let mut buf = Vec::new();

        assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
        assert_eq!(buf, b"ab");
    }

    /// Only the terminator's own `\r` goes. An earlier one is content — the SSE grammar puts no
    /// meaning on it, and dropping bytes a server sent is not this function's decision.
    #[test]
    fn read_line_bounded_strips_only_the_last_cr() {
        let input = Cursor::new(b"a\r\r\n");
        let mut reader = BufReader::new(input);
        let mut buf = Vec::new();

        assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
        assert_eq!(buf, b"a\r");
    }

    /// The buffer is reused across every record of a stream, so a caller's leftovers must not
    /// prefix the next line.
    #[test]
    fn read_line_bounded_clears_a_dirty_buffer_on_entry() {
        let input = Cursor::new(b"fresh\n");
        let mut reader = BufReader::new(input);
        let mut buf = b"stale".to_vec();

        assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
        assert_eq!(buf, b"fresh");
    }

    /// `\r\n` separates SSE events; it must read as an empty line, not as a one-byte `\r`
    /// that the classifier then has to special-case.
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
        // Guards against "simplifying" is_timeout into an ErrorKind check: both errors are
        // ErrorKind::Other, and this one would then print the --timeout hint.
        assert!(!is_timeout(&err));
    }

    /// The cap is only checked on the branch that found no newline, so a single `fill_buf`
    /// carrying both the overrun and its terminator is taken whole. In production the reader
    /// is a `BufReader`, which bounds the overrun to one 8 KB fill; this pins that the
    /// bound depends on that wrapping.
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
        // Force a 2-byte UTF-8 char (é = C3 A9) to be split across buffer fills.
        // "hé\n" = [68, C3, A9, 0A]. With buffer capacity 2, first fill_buf()
        // returns [68, C3] and second returns [A9, 0A]. The old String-based code
        // would reject [68, C3] as invalid UTF-8; Vec<u8> handles this correctly.
        let data = "hé\n".as_bytes().to_vec();
        let input = Cursor::new(data);
        let mut reader = BufReader::with_capacity(2, input);
        let mut buf = Vec::new();

        assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
        assert_eq!(buf, "hé".as_bytes());
    }

    #[test]
    fn read_line_bounded_binary_passthrough() {
        // Non-UTF-8 bytes should pass through without error (unlike from_utf8).
        let input = Cursor::new(vec![0xFF, 0xFE, b'\n']);
        let mut reader = BufReader::new(input);
        let mut buf = Vec::new();

        assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
        assert_eq!(buf, vec![0xFF, 0xFE]);
    }
}
