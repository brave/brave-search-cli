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
/// Values are URL-encoded. Extras override params with the same key.
pub fn build_query(params: &[(&str, Option<&str>)], extras: &[(&str, &str)]) -> String {
    let mut parts = Vec::new();
    for &(key, val) in params {
        if let Some(v) = val {
            if let Some(&(_, ev)) = extras.iter().find(|(k, _)| *k == key) {
                eprintln!("warning: --extra '{key}={ev}' overrides existing parameter");
                continue;
            }
            parts.push(format!("{}={}", key, urlencoding::encode(v)));
        }
    }
    for &(key, val) in extras {
        parts.push(format!(
            "{}={}",
            urlencoding::encode(key),
            urlencoding::encode(val)
        ));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("?{}", parts.join("&"))
    }
}

/// Infers a JSON type from a string value:
/// i64 → integer, finite f64 → float, "true"/"false" → bool, else string.
fn auto_detect_json_type(v: &str) -> serde_json::Value {
    if let Ok(n) = v.parse::<i64>() {
        n.into()
    } else if let Ok(f) = v.parse::<f64>() {
        if f.is_finite() {
            serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .unwrap_or_else(|| v.into())
        } else {
            v.into()
        }
    } else if v == "true" {
        true.into()
    } else if v == "false" {
        false.into()
    } else {
        v.into()
    }
}

/// Builds a JSON object from pre-typed key-value pairs, skipping None values.
pub fn build_json_body(params: &[(&str, Option<serde_json::Value>)]) -> serde_json::Value {
    let mut map = serde_json::Map::with_capacity(params.len());
    for (key, val) in params {
        if let Some(v) = val {
            map.insert((*key).into(), v.clone());
        }
    }
    serde_json::Value::Object(map)
}

/// Merges extra KEY=VALUE pairs into a JSON body. Warns on collision.
pub fn merge_extra_into_json(body: &mut serde_json::Value, extras: &[(&str, &str)]) {
    if extras.is_empty() {
        return;
    }
    let Some(obj) = body.as_object_mut() else {
        eprintln!("error: --extra requires a JSON object body");
        std::process::exit(1);
    };
    for &(key, val) in extras {
        if obj.contains_key(key) {
            eprintln!("warning: --extra '{key}={val}' overrides existing parameter");
        }
        obj.insert(key.into(), auto_detect_json_type(val));
    }
}

/// Builds a query string that supports repeated keys (e.g. ids=a&ids=b).
/// Extras override params with the same key.
pub fn build_query_repeated(
    params: &[(&str, Option<&str>)],
    repeated: &[(&str, &[String])],
    extras: &[(&str, &str)],
) -> String {
    let mut parts = Vec::new();
    for &(key, val) in params {
        if let Some(v) = val {
            if let Some(&(_, ev)) = extras.iter().find(|(k, _)| *k == key) {
                eprintln!("warning: --extra '{key}={ev}' overrides existing parameter");
                continue;
            }
            parts.push(format!("{}={}", key, urlencoding::encode(v)));
        }
    }
    for &(key, vals) in repeated {
        if extras.iter().any(|(k, _)| *k == key) {
            continue;
        }
        for v in vals {
            parts.push(format!("{}={}", key, urlencoding::encode(v)));
        }
    }
    for &(key, val) in extras {
        parts.push(format!(
            "{}={}",
            urlencoding::encode(key),
            urlencoding::encode(val)
        ));
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

/// Sends a POST request with a JSON body and streams SSE response line-by-line.
/// Each `data:` line is printed to stdout. Stops at `data: [DONE]`.
pub fn post_json_stream(
    base_url: &str,
    path: &str,
    api_key: &str,
    body: &serde_json::Value,
    headers: &[(&str, &str)],
    timeout: u64,
) {
    let url = format!("{base_url}{path}");
    let mut req = streaming_agent(timeout)
        .post(&url)
        .header("X-Subscription-Token", api_key)
        .header("Content-Type", "application/json");
    for &(k, v) in headers {
        req = req.header(k, v);
    }

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

    // ── auto_detect_json_type ────────────────────────────────────────

    #[test]
    fn auto_detect_integers() {
        assert_eq!(auto_detect_json_type("20"), serde_json::json!(20));
        assert_eq!(auto_detect_json_type("-5"), serde_json::json!(-5));
        assert_eq!(auto_detect_json_type("0"), serde_json::json!(0));
        assert_eq!(auto_detect_json_type("00020"), serde_json::json!(20)); // leading zeros
    }

    #[test]
    fn auto_detect_booleans() {
        assert_eq!(auto_detect_json_type("true"), serde_json::json!(true));
        assert_eq!(auto_detect_json_type("false"), serde_json::json!(false));
    }

    #[test]
    fn auto_detect_strings() {
        assert_eq!(auto_detect_json_type("US"), serde_json::json!("US"));
        assert_eq!(auto_detect_json_type("en-US"), serde_json::json!("en-US"));
        assert_eq!(
            auto_detect_json_type("moderate"),
            serde_json::json!("moderate")
        );
        assert_eq!(auto_detect_json_type("pd"), serde_json::json!("pd"));
        assert_eq!(auto_detect_json_type(""), serde_json::json!(""));
    }

    #[test]
    fn auto_detect_floats() {
        assert_eq!(auto_detect_json_type("1.5"), serde_json::json!(1.5));
        assert_eq!(auto_detect_json_type("-3.14"), serde_json::json!(-3.14));
        assert_eq!(auto_detect_json_type("0.0"), serde_json::json!(0.0));
        assert_eq!(auto_detect_json_type("1e5"), serde_json::json!(1e5));
        // Version-like strings naturally fail f64::parse
        assert_eq!(auto_detect_json_type("1.2.3"), serde_json::json!("1.2.3"));
        // Non-finite values stay as strings
        assert_eq!(auto_detect_json_type("inf"), serde_json::json!("inf"));
        assert_eq!(auto_detect_json_type("NaN"), serde_json::json!("NaN"));
        assert_eq!(auto_detect_json_type("-inf"), serde_json::json!("-inf"));
    }

    #[test]
    fn auto_detect_case_sensitive_bool() {
        assert_eq!(auto_detect_json_type("TRUE"), serde_json::json!("TRUE"));
        assert_eq!(auto_detect_json_type("True"), serde_json::json!("True"));
        assert_eq!(auto_detect_json_type("FALSE"), serde_json::json!("FALSE"));
    }

    #[test]
    fn auto_detect_i64_overflow_becomes_float() {
        // Values exceeding i64 range that parse as finite f64 become floats
        assert_eq!(
            auto_detect_json_type("99999999999999999999"),
            serde_json::json!(1e20)
        );
    }

    #[test]
    fn auto_detect_not_number_strings() {
        assert_eq!(auto_detect_json_type("20x"), serde_json::json!("20x"));
        assert_eq!(auto_detect_json_type("abc"), serde_json::json!("abc"));
    }

    #[test]
    fn auto_detect_whitespace_not_trimmed() {
        assert_eq!(auto_detect_json_type(" 42 "), serde_json::json!(" 42 "));
        assert_eq!(auto_detect_json_type(" true "), serde_json::json!(" true "));
    }

    #[test]
    fn auto_detect_i64_boundaries() {
        assert_eq!(
            auto_detect_json_type("9223372036854775807"), // i64::MAX
            serde_json::json!(9223372036854775807_i64)
        );
        assert_eq!(
            auto_detect_json_type("9223372036854775808"), // i64::MAX + 1 → f64
            serde_json::json!(9.223372036854776e+18)
        );
        assert_eq!(
            auto_detect_json_type("-9223372036854775808"), // i64::MIN
            serde_json::json!(-9223372036854775808_i64)
        );
    }

    #[test]
    fn auto_detect_null_is_string() {
        assert_eq!(auto_detect_json_type("null"), serde_json::json!("null"));
    }

    // ── build_json_body ──────────────────────────────────────────────

    #[test]
    fn build_json_body_mixed_types() {
        let body = build_json_body(&[
            ("q", Some("test query".into())),
            ("count", Some(20.into())),
            ("spellcheck", Some(true.into())),
            ("freshness", None),
        ]);
        assert_eq!(body["q"], "test query");
        assert_eq!(body["count"], 20);
        assert_eq!(body["spellcheck"], true);
        assert!(body.get("freshness").is_none());
    }

    #[test]
    fn build_json_body_preserves_types() {
        // Verify that values are not auto-detected — strings stay strings
        let body = build_json_body(&[("q", Some("true".into())), ("tag", Some("42".into()))]);
        assert_eq!(body["q"], "true"); // not a boolean
        assert_eq!(body["tag"], "42"); // not a number
    }

    #[test]
    fn build_json_body_empty() {
        let body = build_json_body(&[]);
        assert_eq!(body, serde_json::json!({}));
    }

    #[test]
    fn build_json_body_all_none() {
        let body: serde_json::Value = build_json_body(&[("a", None), ("b", None)]);
        assert_eq!(body, serde_json::json!({}));
    }

    #[test]
    fn build_json_body_single() {
        let body = build_json_body(&[("q", Some("hello".into()))]);
        assert_eq!(body, serde_json::json!({"q": "hello"}));
    }

    #[test]
    fn build_json_body_json_special_chars() {
        let body = build_json_body(&[("q", Some("hello \"world\"\nnewline".into()))]);
        // serde_json handles escaping — just verify it round-trips
        let s = serde_json::to_string(&body).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed["q"], "hello \"world\"\nnewline");
    }

    // ── merge_extra_into_json ────────────────────────────────────────

    #[test]
    fn merge_extra_adds_new_keys() {
        let mut body = serde_json::json!({"q": "test"});
        merge_extra_into_json(&mut body, &[("count", "5"), ("custom", "val")]);
        assert_eq!(body["count"], 5);
        assert_eq!(body["custom"], "val");
        assert_eq!(body["q"], "test"); // unchanged
    }

    #[test]
    fn merge_extra_overrides_existing() {
        let mut body = serde_json::json!({"count": 20});
        merge_extra_into_json(&mut body, &[("count", "5")]);
        assert_eq!(body["count"], 5);
    }

    #[test]
    fn merge_extra_empty_noop() {
        let mut body = serde_json::json!({"q": "test"});
        let original = body.clone();
        merge_extra_into_json(&mut body, &[]);
        assert_eq!(body, original);
    }

    #[test]
    fn merge_extra_empty_on_non_object() {
        // Empty extras returns early before the object check, so non-object
        // bodies are fine. Non-empty extras on a non-object body exits with
        // an error (stdin mode in cmd_answers can receive arbitrary JSON).
        let mut body = serde_json::json!([1, 2, 3]);
        merge_extra_into_json(&mut body, &[]);
        assert_eq!(body, serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn merge_extra_auto_detects_types() {
        let mut body = serde_json::json!({});
        merge_extra_into_json(&mut body, &[("n", "42"), ("b", "true"), ("s", "hello")]);
        assert_eq!(body["n"], 42);
        assert_eq!(body["b"], true);
        assert_eq!(body["s"], "hello");
    }

    #[test]
    fn merge_extra_duplicate_last_wins() {
        let mut body = serde_json::json!({"q": "test"});
        merge_extra_into_json(&mut body, &[("k", "1"), ("k", "2")]);
        assert_eq!(body["k"], 2);
    }

    // ── build_query ──────────────────────────────────────────────────

    #[test]
    fn build_query_all_none() {
        assert_eq!(build_query(&[("a", None), ("b", None)], &[]), "");
    }

    #[test]
    fn build_query_single() {
        assert_eq!(build_query(&[("q", Some("test"))], &[]), "?q=test");
    }

    #[test]
    fn build_query_multiple() {
        assert_eq!(
            build_query(&[("q", Some("test")), ("count", Some("20"))], &[]),
            "?q=test&count=20"
        );
    }

    #[test]
    fn build_query_skips_none() {
        assert_eq!(
            build_query(&[("q", Some("test")), ("x", None), ("c", Some("5"))], &[]),
            "?q=test&c=5"
        );
    }

    #[test]
    fn build_query_url_encodes_values() {
        assert_eq!(
            build_query(&[("q", Some("hello world")), ("x", Some("a&b"))], &[]),
            "?q=hello%20world&x=a%26b"
        );
    }

    #[test]
    fn build_query_unicode() {
        assert_eq!(build_query(&[("q", Some("café"))], &[]), "?q=caf%C3%A9");
    }

    #[test]
    fn build_query_empty() {
        assert_eq!(build_query(&[], &[]), "");
    }

    #[test]
    fn build_query_extras_appended() {
        assert_eq!(
            build_query(&[("q", Some("test"))], &[("extra", "val")]),
            "?q=test&extra=val"
        );
    }

    #[test]
    fn build_query_extras_override() {
        assert_eq!(
            build_query(
                &[("q", Some("test")), ("count", Some("20"))],
                &[("count", "5")]
            ),
            "?q=test&count=5"
        );
    }

    #[test]
    fn build_query_extras_no_collision_when_param_is_none() {
        assert_eq!(
            build_query(&[("freshness", None)], &[("freshness", "pw")]),
            "?freshness=pw"
        );
    }

    #[test]
    fn build_query_extras_url_encodes() {
        assert_eq!(
            build_query(&[], &[("q", "hello world"), ("a&b", "c=d")]),
            "?q=hello%20world&a%26b=c%3Dd"
        );
    }

    #[test]
    fn build_query_empty_extras_noop() {
        assert_eq!(build_query(&[("q", Some("test"))], &[]), "?q=test");
    }

    // ── build_query_repeated ─────────────────────────────────────────

    #[test]
    fn build_query_repeated_basic() {
        let ids = vec!["a".into(), "b".into()];
        assert_eq!(
            build_query_repeated(&[("lang", Some("en"))], &[("ids", &ids)], &[]),
            "?lang=en&ids=a&ids=b"
        );
    }

    #[test]
    fn build_query_repeated_empty_repeated() {
        let ids: Vec<String> = vec![];
        assert_eq!(
            build_query_repeated(&[("lang", Some("en"))], &[("ids", &ids)], &[]),
            "?lang=en"
        );
    }

    #[test]
    fn build_query_repeated_only_repeated() {
        let ids = vec!["x".into(), "y".into(), "z".into()];
        assert_eq!(
            build_query_repeated(&[], &[("ids", &ids)], &[]),
            "?ids=x&ids=y&ids=z"
        );
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
