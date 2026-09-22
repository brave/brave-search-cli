use std::io::{self, BufRead, BufReader, Read, Write};
use std::time::Duration;
use std::time::Instant;

const USER_AGENT: &str = concat!("bx/", env!("CARGO_PKG_VERSION"));

/// Cap on one SSE line: a peer can withhold the newline forever. Checked between fills, so the
/// real ceiling is this plus one `BufReader` fill.
const MAX_SSE_LINE_SIZE: usize = 1024 * 1024; // 1 MB

/// Reads one line, minus its `\n` or `\r\n`, capped at [`MAX_SSE_LINE_SIZE`]. Returns `false` at
/// EOF, discarding an unterminated final line with a warning.
///
/// Works on bytes because `fill_buf()` can split a multi-byte UTF-8 sequence.
fn read_line_bounded<R: BufRead>(reader: &mut R, buf: &mut Vec<u8>) -> io::Result<bool> {
    buf.clear();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            // SSE never dispatches an unterminated line; emitting it would put half a JSON
            // document on stdout.
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
            if buf.last() == Some(&b'\r') {
                buf.pop();
            }
            return Ok(true);
        }
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

pub(crate) fn agent(timeout_secs: u64) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .timeout_global(Some(Duration::from_secs(timeout_secs)))
            .user_agent(USER_AGENT)
            .build(),
    )
}

/// Agent for SSE streaming: no total deadline, since research responses run for minutes.
///
/// `timeout_recv_response` must stay unset: ureq 3.4.0 also applies it to the body, capping the
/// stream. `timeout_recv_body` is re-armed per read, so it bounds silence; ureq 3.4.1 (#1194)
/// makes it a total budget, hence the exact pin in Cargo.toml, guarded by
/// `stream_outlives_the_timeout_while_chunks_keep_arriving`.
///
/// `dial_secs` bounds everything before the first body byte; `read_secs` bounds silence after.
fn streaming_agent(dial_secs: u64, read_secs: u64) -> ureq::Agent {
    let dial = Some(Duration::from_secs(dial_secs));
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
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
pub(crate) fn exit_code_for_status(status: u16) -> i32 {
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
    // Only a real error envelope is formatted: other JSON (a proxy's 4xx) would render as
    // "unknown ({status})", which says less than the status alone.
    let mut msg = match crate::query::parse_json(body) {
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
    // +1: ureq's LimitReader rejects a body exactly at the limit.
    let body = match resp
        .into_body()
        .into_with_config()
        .limit(MAX_RESPONSE_BODY_SIZE + 1)
        .read_to_string()
    {
        Ok(body) => body,
        Err(e) => {
            // An unreadable body (oversized, or not UTF-8) must not cost the status its exit
            // code: a 401 reported as a network error sends an agent into backoff.
            eprintln!("error: failed to read response body: {e}");
            std::process::exit(match status {
                200..=299 => 5,
                _ => exit_code_for_status(status),
            });
        }
    };
    (status, body)
}

/// True if `bytes` is exactly one JSON document: stdout promises one per line, so `{oops` or a
/// proxy's HTML page inside a 200 must not pass. `from_slice` also rejects trailing content and
/// invalid UTF-8.
fn is_json(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde::de::IgnoredAny>(bytes).is_ok()
}

fn handle_response(resp: ureq::http::Response<ureq::Body>) {
    let (status, body) = read_body_or_exit(resp);

    // Redirects are not followed, so a 3xx body is never an answer.
    if !matches!(status, 200..=299) {
        write_error_and_exit(status, &body);
    }

    let trimmed = body.trim();
    if !is_json(trimmed.as_bytes()) {
        eprintln!("error: unexpected non-JSON response");
        write_body_stderr(&body);
        std::process::exit(1);
    }

    // Passed through as-is, so a pretty-printed body stays multi-line. A closed reader is a
    // clean stop, hence `let _`.
    let _ = write_record(&mut io::stdout().lock(), trimmed.as_bytes());
}

/// Writes one record and a newline, flushed. Returns `false` if the reader is gone (a clean
/// stop); any other failure exits 1, since output lost under exit 0 misleads an agent.
///
/// The `flush` is a no-op for today's `StdoutLock`, but std documents line buffering only for a
/// terminal (rust-lang/rust#60673).
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

/// Payload of an SSE `data:` line; `None` for other lines and for empty payloads, which dispatch
/// no event (WHATWG 9.2.6).
///
/// Two deviations from the spec, safe because payloads are JSON: it trims all ASCII whitespace,
/// catching `data: [DONE] ` and blank heartbeats; and it strips a BOM from any line, since a
/// stream-leading one would hide the first record.
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
            // Redirects are not followed, so a 3xx body is never an answer, even one carrying
            // `data:` lines.
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
                // Reader closed (`bx answers … | head`): stop rather than drain minutes of
                // stream and quota. Returning also skips the no-records check below.
                if !write_record(&mut out, data) {
                    return;
                }
                emitted = true;
            }

            // Exit 0 with empty stdout would look like an empty answer. The status separates a
            // 204 from a 200 that ignored `stream`.
            if !emitted {
                eprintln!(
                    "error: HTTP {status} carried no SSE records\n\
                     hint: check --base-url / --endpoint, or retry with --no-stream"
                );
                std::process::exit(1);
            }
        }
        // With `timeout_recv_response` unset the header wait inherits the send deadline, so ureq
        // reports a sent-but-unanswered request as `SendRequest`.
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

// ── Non-exiting core (used by `council`) ─────────────────────────────
//
// The same transport returning errors as values: `process::exit` in a fan-out would let one
// member's 401 discard the others' replies.

/// A fully read response. Only `Retry-After` is kept: it is the one header any caller acts on.
pub struct Resp {
    pub status: u16,
    pub body: Vec<u8>,
    retry_after: Option<String>,
    /// The body exceeded the caller's limit, so `body` is empty for a reason the operator can
    /// act on.
    oversized: bool,
    /// The read faulted; `body` keeps what arrived, since the status line means it is paid for.
    truncated: bool,
}

impl Resp {
    /// Builds one directly, for tests that exercise `parse` without a socket.
    #[cfg(test)]
    pub fn new(status: u16, body: &[u8], headers: &[(&str, &str)]) -> Self {
        Resp {
            status,
            body: body.to_vec(),
            retry_after: headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("retry-after"))
                .map(|(_, v)| (*v).to_string()),
            oversized: false,
            truncated: false,
        }
    }

    /// A response whose body was refused for its size, for tests.
    #[cfg(test)]
    pub fn too_large(status: u16) -> Self {
        Resp {
            status,
            body: Vec::new(),
            retry_after: None,
            oversized: true,
            truncated: false,
        }
    }

    /// Whether the body was refused for its size rather than lost to a fault.
    pub fn oversized(&self) -> bool {
        self.oversized
    }

    /// Whether the read faulted, so `body` may be a prefix — or complete, if the deadline fired
    /// on the final read.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// `Retry-After` as delta-seconds or IMF-fixdate. Any other form, zero, or a past date is
    /// `None`, which the caller reads as "use the computed backoff": clock skew must not turn a
    /// rate limit into a busy retry.
    pub fn retry_after(&self) -> Option<Duration> {
        let v = self.retry_after.as_deref()?.trim();
        if let Some(secs) = crate::council::util::unsigned::<u64>(v) {
            return (secs > 0).then(|| Duration::from_secs(secs));
        }
        let at = crate::council::util::http_date(v)?;
        at.checked_sub(crate::council::util::secs_now())
            .filter(|&d| d > 0)
            .map(Duration::from_secs)
    }
}

/// A reader that fails once a deadline passes.
///
/// No ureq timeout, `timeout_global` included, bounds a slowly dripped body: an expired deadline
/// is re-armed as a one-second timeout per read. Wrapping ureq's reader keeps its framing and
/// size limit, and `read_to_end`'s `Interrupted` retries.
struct Until<R> {
    inner: R,
    at: Instant,
}

impl<R: Read> Read for Until<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if Instant::now() >= self.at {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "budget elapsed"));
        }
        self.inner.read(buf)
    }
}

/// Reads a response whole. **A failed body read is not an error here**: the status line arrived,
/// so the generation is paid for, and an `Err` would let `is_retryable` re-send it. An empty body
/// inherits the status's classification instead (`Fatal` on a 200, retryable on a 5xx).
fn read_resp(resp: ureq::http::Response<ureq::Body>, limit: u64, deadline: Instant) -> Resp {
    let status = resp.status().as_u16();
    let retry_after = resp
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    // +1: see `read_body_or_exit`.
    let mut body = resp.into_body();
    let mut reader = Until {
        inner: body.with_config().limit(limit.saturating_add(1)).reader(),
        at: deadline,
    };
    let mut buf = Vec::new();
    let failed = reader.read_to_end(&mut buf).err();
    // Refused for size is actionable (lower `--max-tokens`); "unreadable" is not. ureq wraps its
    // own error in the `io::Error`.
    let oversized = failed.as_ref().is_some_and(|e| {
        e.get_ref()
            .and_then(|i| i.downcast_ref::<ureq::Error>())
            .is_some_and(|e| matches!(e, ureq::Error::BodyExceedsLimit(_)))
    });
    // Only an oversized body is cleared. A faulted read keeps what arrived: the deadline can
    // fire on `read_to_end`'s final call after a complete body.
    if oversized {
        buf.clear();
    }
    Resp {
        status,
        body: buf,
        retry_after,
        oversized,
        truncated: failed.is_some() && !oversized,
    }
}

/// True when a transport failure proves the request never reached the provider, so a retry
/// cannot pay twice for one generation.
///
/// ureq reports both a failed dial and a connection dropped after generation as `Io`, so only
/// kinds that arise solely while dialling are retried, not EOF, reset or broken pipe. Neither
/// TLS variant is retried: `Tls(&str)` is a config fault that repeats, and `Rustls(_)` can be
/// raised reading the response head, after the answer exists.
pub fn is_retryable(e: &ureq::Error) -> bool {
    match e {
        ureq::Error::ConnectionFailed | ureq::Error::HostNotFound => true,
        ureq::Error::Io(e) => matches!(
            e.kind(),
            io::ErrorKind::ConnectionRefused
                | io::ErrorKind::HostUnreachable
                | io::ErrorKind::NetworkUnreachable
                | io::ErrorKind::NetworkDown
                | io::ErrorKind::AddrNotAvailable
                // A missing connector target (unix socket, proxy); raised only while dialling.
                | io::ErrorKind::NotFound
        ),
        _ => false,
    }
}

/// Bound on a `get_raw` body read, which no agent timeout reaches; matches the callers'
/// 15-second global timeout.
const GET_DEADLINE: Duration = Duration::from_secs(15);

pub fn get_raw(
    agent: &ureq::Agent,
    url: &str,
    headers: &[(&str, String)],
    limit: u64,
) -> Result<Resp, ureq::Error> {
    let mut req = agent.get(url);
    for (k, v) in headers {
        req = req.header(*k, v);
    }
    // Taken before `call()`: inline in `read_resp`'s arguments it would start after the head
    // arrived, giving the body a fresh window.
    let deadline = Instant::now() + GET_DEADLINE;
    Ok(read_resp(req.call()?, limit, deadline))
}

/// POSTs `body` with `recv` as the deadline for the **whole call**, typically a member's
/// remaining budget.
///
/// Only `timeout_global` bounds the call as a whole: per-phase timeouts each get the full budget,
/// and `timeout_recv_body` bounds silence, not a dripping body. The agent leaves it `None` so
/// each call sets its own.
pub fn post_raw(
    agent: &ureq::Agent,
    url: &str,
    headers: &[(&str, String)],
    body: &[u8],
    limit: u64,
    recv: Duration,
) -> Result<Resp, ureq::Error> {
    // Taken before `send`: otherwise a peer could stall the head until `recv` nearly elapses,
    // then drip the body through a second full window.
    let deadline = Instant::now() + recv;
    let mut req = agent
        .post(url)
        .config()
        .timeout_global(Some(recv))
        .build()
        .header("Content-Type", "application/json");
    for (k, v) in headers {
        req = req.header(*k, v);
    }
    Ok(read_resp(req.send(body)?, limit, deadline))
}

#[cfg(test)]
#[path = "api_tests.rs"]
mod tests;
