use std::io::{self, BufRead, BufReader, Write};
use std::time::Duration;

const USER_AGENT: &str = concat!("bx/", env!("CARGO_PKG_VERSION"));

/// Maximum bytes per SSE line. BufReader::lines() buffers unboundedly until
/// a newline — an attacker can continuously send bytes without ever sending
/// a newline or EOF, exhausting memory. Defense-in-depth cap.
/// Ref: https://doc.rust-lang.org/std/io/trait.BufRead.html#method.read_line
const MAX_SSE_LINE_SIZE: usize = 1024 * 1024; // 1 MB

/// Reads a single line with a size cap. Returns `false` at EOF.
///
/// Uses raw bytes (`Vec<u8>`) because `fill_buf()` can split multi-byte
/// UTF-8 sequences at its 8 KB buffer boundary, which would cause
/// spurious `from_utf8` errors if we validated per chunk.
fn read_line_bounded<R: BufRead>(reader: &mut R, buf: &mut Vec<u8>) -> io::Result<bool> {
    buf.clear();
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(!buf.is_empty());
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

/// Builds a query string from key-value pairs, skipping None values.
/// Values are URL-encoded.
pub fn build_query(params: &[(&str, Option<&str>)]) -> String {
    let mut parts = Vec::new();
    for &(key, val) in params {
        if let Some(v) = val {
            parts.push(format!("{}={}", key, urlencoding::encode(v)));
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("?{}", parts.join("&"))
    }
}

/// Builds a query string that supports repeated keys (e.g. ids=a&ids=b).
pub fn build_query_repeated(
    params: &[(&str, Option<&str>)],
    repeated: &[(&str, &[String])],
) -> String {
    let mut parts = Vec::new();
    for &(key, val) in params {
        if let Some(v) = val {
            parts.push(format!("{}={}", key, urlencoding::encode(v)));
        }
    }
    for &(key, vals) in repeated {
        for v in vals {
            parts.push(format!("{}={}", key, urlencoding::encode(v)));
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("?{}", parts.join("&"))
    }
}

fn agent(timeout_secs: u64) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .timeout_global(Some(Duration::from_secs(timeout_secs)))
            .user_agent(USER_AGENT)
            .build(),
    )
}

/// Agent for SSE streaming — per-phase timeouts only, no global deadline.
/// Research mode responses can take up to ~300s; a global timeout would kill the stream.
fn streaming_agent(timeout_secs: u64) -> ureq::Agent {
    let t = Some(Duration::from_secs(timeout_secs));
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .timeout_connect(t)
            .timeout_send_request(t)
            .timeout_send_body(t)
            .timeout_recv_response(t)
            .user_agent(USER_AGENT)
            .build(),
    )
}

/// Maps an HTTP status code to a process exit code.
///   0 = success
///   1 = client error (4xx general)
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
    let mut msg = if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        let code = v["error"]["code"].as_str().unwrap_or("UNKNOWN");
        let detail = v["error"]["detail"].as_str().unwrap_or("");
        if !detail.is_empty() {
            format!(
                "{} ({status}) — {detail}",
                code.to_lowercase().replace('_', " ")
            )
        } else {
            format!("{} ({status})", code.to_lowercase().replace('_', " "))
        }
    } else {
        format!("HTTP {status}")
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
            eprintln!("error: failed to read response body: {e}");
            std::process::exit(5);
        }
    };
    (status, body)
}

fn handle_response(resp: ureq::http::Response<ureq::Body>) {
    let (status, body) = read_body_or_exit(resp);

    if status >= 400 {
        write_error_and_exit(status, &body);
    }

    // Guard against non-JSON 2xx responses (e.g. proxy HTML pages)
    let trimmed = body.trim_start();
    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        eprintln!("error: unexpected non-JSON response");
        write_body_stderr(&body);
        std::process::exit(1);
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    out.write_all(body.as_bytes()).ok();
    if !body.ends_with('\n') {
        out.write_all(b"\n").ok();
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
    timeout: u64,
) {
    let url = format!("{base_url}{path}");
    let req = agent(timeout)
        .post(&url)
        .header("X-Subscription-Token", api_key)
        .header("Content-Type", "application/json");

    let payload = serde_json::to_string(body).expect("failed to serialize JSON body");

    match req.send(payload.as_bytes()) {
        Ok(resp) => handle_response(resp),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(5);
        }
    }
}

/// Sends a POST request with a JSON body and streams SSE response line-by-line.
/// Each `data:` line is printed to stdout. Stops at `data: [DONE]`.
pub fn post_json_stream(
    base_url: &str,
    path: &str,
    api_key: &str,
    body: &serde_json::Value,
    timeout: u64,
) {
    let url = format!("{base_url}{path}");
    let req = streaming_agent(timeout)
        .post(&url)
        .header("X-Subscription-Token", api_key)
        .header("Content-Type", "application/json");

    let payload = serde_json::to_string(body).expect("failed to serialize JSON body");

    match req.send(payload.as_bytes()) {
        Ok(resp) => {
            let status = resp.status().as_u16();
            if status >= 400 {
                let (_, body) = read_body_or_exit(resp);
                write_error_and_exit(status, &body);
            }

            let (_, body) = resp.into_parts();
            let mut reader = BufReader::new(body.into_reader());
            let stdout = io::stdout();
            let mut out = stdout.lock();
            let mut line: Vec<u8> = Vec::new();

            loop {
                let has_data = match read_line_bounded(&mut reader, &mut line) {
                    Ok(has) => has,
                    Err(e) => {
                        eprintln!("error: reading stream: {e}");
                        std::process::exit(5);
                    }
                };
                if !has_data {
                    break;
                }

                if line.is_empty() {
                    continue;
                }

                if let Some(data) = line.strip_prefix(b"data:") {
                    let data = data.strip_prefix(b" ").unwrap_or(data);
                    if data == b"[DONE]" {
                        break;
                    }
                    out.write_all(data).ok();
                    out.write_all(b"\n").ok();
                    out.flush().ok();
                }
            }
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
    use std::io::Cursor;

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

    #[test]
    fn read_line_bounded_no_trailing_newline() {
        let input = Cursor::new(b"partial");
        let mut reader = BufReader::new(input);
        let mut buf = Vec::new();

        assert!(read_line_bounded(&mut reader, &mut buf).unwrap());
        assert_eq!(buf, b"partial");

        assert!(!read_line_bounded(&mut reader, &mut buf).unwrap());
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
