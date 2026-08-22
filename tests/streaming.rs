//! `bx answers` against a local socket: what happens when a stream is long, stalled,
//! truncated, redirected, not a stream at all, or nobody is reading it any more.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// How long a stalling server holds the connection open. Must comfortably exceed the
/// `--timeout` under test so that bx's timeout fires first, not the server hanging up.
const STALL: Duration = Duration::from_secs(15);

/// Upper bound for "bx gave up promptly" assertions.
const PATIENCE: Duration = Duration::from_secs(10);

/// Binds a throwaway listener and hands the accepted socket to `respond`, request already
/// drained. Returns its port. Every server below is one of these plus a few writes.
fn serve(respond: impl FnOnce(&mut TcpStream) + Send + 'static) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            read_request(&mut sock);
            respond(&mut sock);
        }
    });

    port
}

const SSE_HEAD: &str = "HTTP/1.1 200 OK\r\n\
                        Content-Type: text/event-stream\r\n\
                        Transfer-Encoding: chunked\r\n\r\n";

/// Serves one SSE response: `chunks` events spaced `gap` apart. When `finish` is false
/// the server goes silent afterwards instead of sending `[DONE]`.
fn serve_sse(chunks: usize, gap: Duration, finish: bool) -> u16 {
    serve(move |sock| {
        if sock.write_all(SSE_HEAD.as_bytes()).is_err() {
            return;
        }
        for i in 0..chunks {
            thread::sleep(gap);
            if write_chunk(sock, &format!("data: {{\"n\":{i}}}\n\n")).is_err() {
                return;
            }
        }
        if finish {
            let _ = write_chunk(sock, "data: [DONE]\n\n");
            let _ = sock.write_all(b"0\r\n\r\n");
        } else {
            thread::sleep(STALL);
        }
    })
}

/// Serves an SSE 200 whose body is exactly `events`, so a test can choose its own
/// framing — CRLF, say, which is what real servers send.
fn serve_sse_raw(events: &'static str) -> u16 {
    serve(move |sock| {
        if sock.write_all(SSE_HEAD.as_bytes()).is_ok() {
            let _ = write_chunk(sock, events);
            let _ = sock.write_all(b"0\r\n\r\n");
        }
    })
}

/// Serves a `Content-Length` error response whose body drips densely enough to defeat
/// ureq's own deadline, so only bx's own bounds end it. Chunked will not do: ureq's decoder
/// is bounded by `timeout_global` and would hide the case under test.
fn serve_dripped_error_body(status: &'static str, headers_after: Duration) -> u16 {
    serve(move |sock| {
        thread::sleep(headers_after);
        let body = format!(
            r#"{{"error":{{"code":"RATE_LIMITED","detail":"{}"}}}}"#,
            "x".repeat(20_000)
        );
        let head = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        if sock.write_all(head.as_bytes()).is_err() {
            return;
        }
        for b in body.as_bytes() {
            thread::sleep(Duration::from_millis(2));
            if sock.write_all(&[*b]).and_then(|()| sock.flush()).is_err() {
                return;
            }
        }
    })
}

/// Serves an error response whose body arrives one byte at a time, `gap` apart. Every read
/// lands inside the per-read timeout, so only a total deadline stops it.
fn serve_dripped_error(status: &'static str, gap: Duration) -> u16 {
    serve(move |sock| {
        if sock.write_all(chunked_head(status).as_bytes()).is_err() {
            return;
        }
        // Far more drips than any bounded read should accept, but few enough that a
        // regression fails the PATIENCE assertion instead of stalling the suite.
        for _ in 0..60 {
            thread::sleep(gap);
            if write_chunk(sock, "x").is_err() {
                return;
            }
        }
    })
}

/// Serves an SSE 200 that only ever sends comments — real servers use these as keep-alives.
/// Every read succeeds, so the per-read timeout never fires and no record ever appears.
fn serve_heartbeats_only(gap: Duration) -> u16 {
    serve(move |sock| {
        if sock.write_all(SSE_HEAD.as_bytes()).is_err() {
            return;
        }
        for _ in 0..60 {
            thread::sleep(gap);
            if write_chunk(sock, ":ping\n\n").is_err() {
                return;
            }
        }
    })
}

/// Serves a 200 whose body dribbles in under a `Content-Length` that is never satisfied.
/// The drip must be dense: once `timeout_global` expires ureq clamps the remaining budget
/// to a fresh 1s read rather than giving up, so a byte arriving every few milliseconds
/// keeps resetting it and the request outlives its own deadline indefinitely. A sparse
/// drip is cut by `timeout_global` normally and proves nothing.
fn serve_dripped_ok(gap: Duration) -> u16 {
    serve(move |sock| {
        let head = "HTTP/1.1 200 OK\r\n\
                    Content-Type: application/json\r\n\
                    Content-Length: 4096\r\n\r\n";
        if sock.write_all(head.as_bytes()).is_err() {
            return;
        }
        for _ in 0..4096 {
            thread::sleep(gap);
            if sock.write_all(b"x").and_then(|()| sock.flush()).is_err() {
                return;
            }
        }
    })
}

/// Serves response headers and then nothing at all, holding the connection open.
fn serve_headers_then_silence(status: &'static str) -> u16 {
    serve(move |sock| {
        let _ = sock.write_all(chunked_head(status).as_bytes());
        let _ = sock.flush();
        thread::sleep(STALL);
    })
}

/// Serves one non-chunked response, then closes. `Content-Length` is derived from `body`
/// so that editing a fixture cannot silently turn a test into a premature-EOF test.
fn serve_response(status: &'static str, content_type: &'static str, body: &'static str) -> u16 {
    serve(move |sock| {
        let head = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let _ = sock.write_all(head.as_bytes());
        let _ = sock.write_all(body.as_bytes());
        let _ = sock.flush();
    })
}

/// Accepts a connection and never replies at all — not even response headers.
fn serve_silence() -> u16 {
    serve(|_| thread::sleep(STALL))
}

fn chunked_head(status: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: application/json\r\n\
         Transfer-Encoding: chunked\r\n\r\n"
    )
}

/// Drains the request head *and* body. Bytes left unread in the receive queue make
/// close() emit RST instead of FIN, discarding the response we already wrote — which
/// surfaces client-side as a spurious "Peer disconnected".
fn read_request(sock: &mut TcpStream) {
    let mut buf = [0u8; 1024];
    let mut seen = Vec::new();
    let head_end = loop {
        if let Some(i) = seen.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        match sock.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => seen.extend_from_slice(&buf[..n]),
        }
    };

    let mut remaining = content_length(&seen[..head_end]).saturating_sub(seen.len() - head_end);
    while remaining > 0 {
        match sock.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => remaining = remaining.saturating_sub(n),
        }
    }
}

fn content_length(head: &[u8]) -> usize {
    String::from_utf8_lossy(head)
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())?
        })
        .unwrap_or(0)
}

fn write_chunk(sock: &mut TcpStream, body: &str) -> std::io::Result<()> {
    write!(sock, "{:X}\r\n{body}\r\n", body.len())?;
    sock.flush()
}

/// Runs `bx answers` against the local server, isolated from the user's real config.
fn run_bx(port: u16, timeout_secs: u32) -> (Output, Duration) {
    let cfg = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cfg.path(), "{}").unwrap();

    let started = Instant::now();
    let output = Command::new(env!("CARGO_BIN_EXE_bx"))
        .args(["answers", "ping"])
        .args(["--api-key", "TEST"])
        .args(["--base-url", &format!("http://127.0.0.1:{port}")])
        .args(["--timeout", &timeout_secs.to_string()])
        .arg("--config")
        .arg(cfg.path())
        .env_remove("BRAVE_SEARCH_API_KEY")
        .env_remove("BRAVE_SEARCH_BASE_URL")
        .env("NO_PROXY", "*")
        .output()
        .unwrap();

    (output, started.elapsed())
}

#[test]
fn stream_outlives_the_timeout_while_chunks_keep_arriving() {
    // 3s of streaming under a 2s timeout: fine, because no single gap comes close to 2s.
    // Anchoring the deadline to the response headers instead would abort this at 2s.
    const CHUNKS: usize = 5;
    let port = serve_sse(CHUNKS, Duration::from_millis(600), true);
    let (out, elapsed) = run_bx(port, 2);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "expected exit 0, got {stderr}");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert!(
        elapsed > Duration::from_secs(2),
        "stream finished in {elapsed:?}, too fast to exercise the timeout"
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    let expected: Vec<String> = (0..CHUNKS).map(|i| format!("{{\"n\":{i}}}")).collect();
    assert_eq!(stdout.lines().collect::<Vec<_>>(), expected);
}

#[test]
fn stalled_stream_times_out_with_an_actionable_message() {
    let port = serve_sse(1, Duration::from_millis(100), false);
    let (out, elapsed) = run_bx(port, 1);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "stderr: {stderr}");
    assert!(elapsed < PATIENCE, "took {elapsed:?} to give up");
    // The agent must be able to tell a stall from a dead API, and know the knob.
    assert!(stderr.contains("no data for 1s"), "stderr: {stderr}");
    assert!(stderr.contains("--timeout"), "stderr: {stderr}");
    // Whatever arrived before the stall is still delivered.
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "{\"n\":0}");
}

#[test]
fn a_stream_of_pure_heartbeats_does_not_hang_forever() {
    // Nothing else bounds this: bytes keep arriving, so `timeout_recv_body` is re-armed on
    // every read, and the streaming agent deliberately has no total deadline. Only the
    // watchdog armed until the *first record* stops it.
    let port = serve_heartbeats_only(Duration::from_millis(200));
    let (out, elapsed) = run_bx(port, 1);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "stderr: {stderr}");
    assert!(elapsed < PATIENCE, "took {elapsed:?} to give up");
    assert!(
        stderr.contains("exceeded the 1s timeout"),
        "stderr: {stderr}"
    );
    assert!(out.stdout.is_empty(), "no record was ever sent");
}

#[test]
fn a_dripped_body_cannot_outlive_the_timeout_by_much() {
    // The blocking path is the one with `timeout_global`, and it is not the hard bound its
    // name suggests — see `abort_after`. Must be `bx web`, not `bx answers`: the streaming
    // agent has no total deadline to escape in the first place.
    let port = serve_dripped_ok(Duration::from_millis(2));
    let started = Instant::now();
    let out = run_bx_web_with(port, Stdio::piped(), 1);
    let elapsed = started.elapsed();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "stderr: {stderr}");
    assert!(elapsed < PATIENCE, "took {elapsed:?} to give up");
}

#[test]
fn a_reader_that_leaves_mid_record_is_not_a_clean_stop() {
    // `bx answers … | head -c 100` on a large record: the OS takes a pipe buffer's worth
    // and then the reader is gone, leaving half a JSON document on stdout. Reporting that
    // as success would have an agent parse a truncated document under exit 0. A reader
    // that leaves *between* records still exits 0 — that is `closed_reader_stops_the_stream`.
    let record: &'static str = Box::leak(format!("{{\"a\":\"{}\"}}", "x".repeat(500_000)).into());
    let port = serve_sse_raw(Box::leak(
        format!("data: {record}\n\ndata: [DONE]\n\n").into(),
    ));

    let cfg = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cfg.path(), "{}").unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_bx"))
        .args(["answers", "ping", "--api-key", "TEST", "--timeout", "10"])
        .args(["--base-url", &format!("http://127.0.0.1:{port}")])
        .arg("--config")
        .arg(cfg.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_remove("BRAVE_SEARCH_API_KEY")
        .env_remove("BRAVE_SEARCH_BASE_URL")
        .env("NO_PROXY", "*")
        .spawn()
        .unwrap();

    // Take a pipe buffer's worth, then close — the writer is left mid-record.
    let mut head = [0u8; 100];
    let mut stdout = child.stdout.take().unwrap();
    stdout.read_exact(&mut head).unwrap();
    drop(stdout);

    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {stderr}");
    assert!(stderr.contains("writing to stdout"), "stderr: {stderr}");
}

#[test]
fn closed_reader_stops_the_stream() {
    // `bx answers … | head -1`: once the reader is gone, draining the rest would
    // burn minutes of wall clock and metered quota for output nobody receives.
    let port = serve_sse(100, Duration::from_millis(100), true);
    let cfg = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cfg.path(), "{}").unwrap();

    let started = Instant::now();
    let mut child = Command::new(env!("CARGO_BIN_EXE_bx"))
        .args(["answers", "ping"])
        .args(["--api-key", "TEST"])
        .args(["--base-url", &format!("http://127.0.0.1:{port}")])
        .args(["--timeout", "30"])
        .arg("--config")
        .arg(cfg.path())
        .env_remove("BRAVE_SEARCH_API_KEY")
        .env_remove("BRAVE_SEARCH_BASE_URL")
        .env("NO_PROXY", "*")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    // Read one line, then drop the pipe — the rest of the 10s stream goes nowhere.
    let mut first = String::new();
    let mut reader = BufReader::new(child.stdout.take().unwrap());
    reader.read_line(&mut first).unwrap();
    drop(reader); // closes the read end

    let status = child.wait().unwrap();
    let elapsed = started.elapsed();

    assert_eq!(first.trim(), "{\"n\":0}");
    assert!(
        elapsed < Duration::from_secs(5),
        "kept draining for {elapsed:?} after the reader closed"
    );
    assert!(
        status.success(),
        "a closed reader is not an error: {status}"
    );
}

#[test]
fn dripped_error_body_does_not_hold_the_cli_open() {
    // The success path is deliberately unbounded, but an error body must not inherit that:
    // a server trickling bytes just under the per-read timeout would otherwise stall the
    // CLI for as long as it kept trickling, with the status code already in hand.
    // --timeout 30 (not 1) so that the 5s cap, and not --timeout, is what ends this.
    let port = serve_dripped_error("429 Too Many Requests", Duration::from_millis(300));
    let (out, elapsed) = run_bx(port, 30);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(4), "stderr: {stderr}");
    assert!(
        elapsed < PATIENCE,
        "kept reading the error body for {elapsed:?}"
    );
    assert!(stderr.contains("429"), "stderr: {stderr}");
}

#[test]
fn error_body_that_never_arrives_does_not_hold_the_cli_open() {
    // The status is known from the headers, so a body that never comes must not extend the
    // run. A deadline checked between reads could not do this: one ureq read is
    // uninterruptible and would block for the whole --timeout (measured: 20s at
    // --timeout 20). --timeout 30 here so only the 5s cap can keep it under PATIENCE.
    let port = serve_headers_then_silence("503 Service Unavailable");
    let (out, elapsed) = run_bx(port, 30);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "stderr: {stderr}");
    assert!(stderr.contains("503"), "stderr: {stderr}");
    assert!(
        elapsed < PATIENCE,
        "waited {elapsed:?} for a body it did not need"
    );
}

#[test]
fn non_200_is_an_error_even_when_below_400() {
    // Redirects are not followed (max_redirects(0)), and only 200 carries a stream.
    // Parsing a redirect as SSE would exit 0 having printed nothing at all.
    let port = serve_response("302 Found", "text/html", "");
    let (out, elapsed) = run_bx(port, 5);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {stderr}");
    assert!(stderr.contains("302"), "stderr: {stderr}");
    assert!(out.stdout.is_empty(), "stdout: {:?}", out.stdout);
    assert!(elapsed < PATIENCE, "hung for {elapsed:?}");
    // A bodiless error writes no raw-body section, so there is no stray blank line.
    assert!(!stderr.contains("\n\n"), "stderr: {stderr:?}");
}

#[test]
fn blocking_path_rejects_a_redirect_carrying_json() {
    // `bx web` and friends, not `answers`. Without this, reverting the blocking path to
    // `>= 400` puts a proxy's redirect body on stdout as if it were results, exit 0 —
    // and the whole suite still passes, because every other test drives `answers`.
    let port = serve_response(
        "302 Found",
        "application/json",
        r#"{"redirect":"/elsewhere"}"#,
    );

    let cfg = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cfg.path(), "{}").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_bx"))
        .args(["web", "ping"])
        .args(["--api-key", "TEST"])
        .args(["--base-url", &format!("http://127.0.0.1:{port}")])
        .arg("--config")
        .arg(cfg.path())
        .env_remove("BRAVE_SEARCH_API_KEY")
        .env_remove("BRAVE_SEARCH_BASE_URL")
        .env("NO_PROXY", "*")
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {stderr}");
    assert!(out.stdout.is_empty(), "redirect body reached stdout");
    assert!(stderr.contains("302"), "stderr: {stderr}");
}

#[test]
fn a_200_that_is_not_a_stream_is_an_error() {
    // An upstream that ignored `stream`, or a proxy answering 200 with its own page:
    // every line fails the `data:` test, so the old code exited 0 having printed nothing.
    let body = r#"{"choices":[{"message":{"content":"hi"}}]}"#;
    let port = serve_response("200 OK", "application/json", body);
    let (out, elapsed) = run_bx(port, 5);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {stderr}");
    assert!(out.stdout.is_empty(), "stdout: {:?}", out.stdout);
    assert!(stderr.contains("no answer data"), "stderr: {stderr}");
    assert!(elapsed < PATIENCE, "hung for {elapsed:?}");
}

#[test]
fn crlf_framing_and_a_done_only_stream() {
    // Real SSE servers use CRLF, which no other test exercises. This stream is also
    // `[DONE]`-and-nothing-else: a completion with no answer in it is still an error,
    // because stdout would otherwise be empty on a successful exit.
    let port = serve_sse_raw("data: [DONE]\r\n\r\n");
    let (out, _) = run_bx(port, 5);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {stderr}");
    assert!(out.stdout.is_empty(), "stdout: {:?}", out.stdout);
    assert!(stderr.contains("no answer data"), "stderr: {stderr}");
}

#[test]
fn a_record_that_is_not_json_is_an_error() {
    // stdout is one JSON record per line. A proxy page arriving as a `data:` payload used
    // to be forwarded verbatim under exit 0 — including any terminal escapes in it.
    let port = serve_sse_raw("data: <html>not json</html>\n\ndata: [DONE]\n\n");
    let (out, _) = run_bx(port, 5);

    // Exit 1, the same as the blocking path gives an unusable 200 — nothing was lost in
    // transit, so this is not exit 5.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {stderr}");
    assert!(out.stdout.is_empty(), "stdout: {:?}", out.stdout);
    assert!(stderr.contains("not JSON"), "stderr: {stderr}");
}

#[test]
fn a_stream_cut_mid_record_is_reported() {
    // A clean EOF part-way through an event proves data was lost — unlike a missing
    // `[DONE]`, which loses nothing. Records already delivered stay on stdout.
    let port = serve_sse_raw("data: {\"n\":0}\n\ndata: {\"n\":");
    let (out, _) = run_bx(port, 5);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "stderr: {stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
    assert!(stderr.contains("mid-record"), "stderr: {stderr}");
}

#[test]
fn a_record_split_across_data_fields_arrives_as_one_line() {
    // Legal SSE: one event, several `data:` fields. Treating each line as its own record
    // handed the agent two invalid fragments and exited 5 on a perfectly good stream.
    let port = serve_sse_raw("data: {\"n\":\ndata: 0}\n\ndata: [DONE]\n\n");
    let (out, _) = run_bx(port, 5);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    // One line, and still valid JSON parsing to the value the server meant.
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout.lines().count(), 1, "stdout: {stdout:?}");
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(v["n"], 0);
}

#[test]
fn a_final_event_without_a_trailing_blank_line_is_not_dropped() {
    // Assembling events means holding a record until the event ends. A stream that stops
    // after a complete `data:` line must still deliver it.
    let port = serve_sse_raw("data: {\"n\":0}\n");
    let (out, _) = run_bx(port, 5);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
}

#[test]
fn a_slow_consumer_does_not_get_a_truncated_record() {
    // The watchdog is armed until the first record. If it is still armed while that record
    // is being written, a reader that is slow to drain leaves bx blocked inside the write
    // until the timer fires and kills the process mid-record.
    const SIZE: usize = 500_000;
    let record: &'static str = Box::leak(format!("{{\"a\":\"{}\"}}", "x".repeat(SIZE)).into());
    let port = serve(move |sock| {
        if sock.write_all(SSE_HEAD.as_bytes()).is_ok() {
            let _ = write_chunk(sock, &format!("data: {record}\n\n"));
            let _ = write_chunk(sock, "data: [DONE]\n\n");
            let _ = sock.write_all(b"0\r\n\r\n");
        }
    });

    let cfg = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cfg.path(), "{}").unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_bx"))
        .args(["answers", "ping", "--api-key", "TEST", "--timeout", "1"])
        .args(["--base-url", &format!("http://127.0.0.1:{port}")])
        .arg("--config")
        .arg(cfg.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_remove("BRAVE_SEARCH_API_KEY")
        .env_remove("BRAVE_SEARCH_BASE_URL")
        .env("NO_PROXY", "*")
        .spawn()
        .unwrap();

    // Longer than the watchdog's timeout+1s, so a still-armed timer would fire mid-write.
    let mut stdout = child.stdout.take().unwrap();
    thread::sleep(Duration::from_secs(3));
    let mut got = Vec::new();
    stdout.read_to_end(&mut got).unwrap();
    let out = child.wait_with_output().unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert_eq!(got.len(), record.len() + 1, "record truncated: {stderr}");
}

#[test]
fn a_fragment_after_a_complete_record_is_reported_not_emitted() {
    // The first field is a whole document, so line framing closes it and it is delivered.
    // The fragment after it is not, and must be reported rather than passed off as a
    // record — with or without the trailing newline that decides nothing here.
    for body in [
        "data: {\"a\":1}\ndata: {\"b\"",
        "data: {\"a\":1}\ndata: {\"b\"\n",
    ] {
        let out = run_bx(serve_sse_raw(Box::leak(body.to_string().into())), 5).0;
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(5), "{body:?}: {stderr}");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "{\"a\":1}\n",
            "{body:?}"
        );
    }
}

#[test]
fn a_partial_trailing_comment_is_not_reported_as_a_lost_record() {
    // A keep-alive cut in half loses nothing; only an unterminated `data:` line does.
    let port = serve_sse_raw("data: {\"n\":0}\n\n:pin");
    let (out, _) = run_bx(port, 5);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
}

#[test]
fn a_bom_on_the_stream_does_not_hide_the_first_record() {
    // The blocking path stripped it; the streaming path dropped the line, then reported
    // "no answer data" and blamed the user's --base-url for the proxy's byte order mark.
    let port = serve_sse_raw("\u{feff}data: {\"n\":0}\n\ndata: [DONE]\n\n");
    let (out, _) = run_bx(port, 5);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
}

#[test]
fn a_streamed_scalar_is_rejected_like_a_blocking_one() {
    // stdout promises a JSON document per line. `data: 42` parses, but a bare scalar is
    // never an API record, and the blocking path has always refused one.
    let port = serve_sse_raw("data: 42\n\ndata: [DONE]\n\n");
    let (out, _) = run_bx(port, 5);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {stderr}");
    assert!(stderr.contains("not JSON"), "stderr: {stderr}");
    assert!(out.stdout.is_empty());
}

#[test]
fn a_complete_final_record_is_not_discarded_as_truncation() {
    // A stream that ends without its last newline has still delivered everything. Treating
    // any unterminated tail as data loss threw the record away and reported exit 5.
    for body in ["data: {\"n\":0}\n\ndata: [DONE]", "data: {\"n\":0}"] {
        let out = run_bx(serve_sse_raw(Box::leak(body.to_string().into())), 5).0;
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{body:?}: {stderr}");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "{\"n\":0}\n",
            "{body:?}"
        );
    }
}

#[test]
fn line_framed_events_survive_a_missing_final_newline() {
    // The loop frames by line; EOF used to fall back to strict SSE and condemn the whole
    // buffer, so two complete records vanished with "stream ended mid-record".
    let port = serve_sse_raw("data: {\"n\":0}\ndata: {\"n\":1}");
    let (out, _) = run_bx(port, 5);

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "{\"n\":0}\n{\"n\":1}\n"
    );
}

#[test]
fn a_blocking_body_is_normalised_to_one_line() {
    // The streaming path forbids a raw CR and a blank line on stdout; the blocking path
    // used to pass a body's own `\r\n`, trailing blank line or leading newline straight
    // through, breaking the same contract.
    for body in ["{\"a\":1}\r\n", "{\"a\":1}\n\n", "\n{\"a\":1}"] {
        let port = serve_response(
            "200 OK",
            "application/json",
            Box::leak(body.to_string().into()),
        );
        let out = run_bx_web(port, Stdio::piped());
        assert!(out.status.success(), "{body:?}");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "{\"a\":1}\n",
            "{body:?}"
        );
    }
}

#[test]
fn a_padded_record_still_closes_its_event() {
    // `sse_payload` preserves padding, because a trailing space can be content in a record
    // split across fields. The line-framing gate looked at the last raw byte, so one
    // trailing space ran two complete records together and lost both.
    for body in [
        "data: {\"n\":0} \ndata: {\"n\":1} \n\ndata: [DONE]\n\n",
        "data: {\"n\":0} \ndata: {\"n\":1} ",
    ] {
        let out = run_bx(serve_sse_raw(Box::leak(body.to_string().into())), 5).0;
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{body:?}: {stderr}");
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "{\"n\":0}\n{\"n\":1}\n",
            "{body:?}"
        );
    }
}

#[test]
fn events_framed_one_per_line_are_not_run_together() {
    // Some OpenAI-compatible proxies omit the blank separator. Assembling those lines into
    // one event produced two concatenated documents, failed validation, and lost the lot.
    let port = serve_sse_raw("data: {\"n\":0}\ndata: {\"n\":1}\n\ndata: [DONE]\n\n");
    let (out, _) = run_bx(port, 5);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "{\"n\":0}\n{\"n\":1}\n"
    );
}

#[test]
fn a_record_split_mid_string_is_still_assembled() {
    // The counterpart to the test above: a genuine continuation is not yet a valid
    // document, so it must keep accumulating rather than being flushed early.
    let port = serve_sse_raw("data: {\"a\":\ndata: 1}\n\ndata: [DONE]\n\n");
    let (out, _) = run_bx(port, 5);

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"a\": 1}\n");
}

#[test]
fn a_raw_carriage_return_does_not_reach_stdout() {
    // CR is legal whitespace between JSON tokens, so it survives validation — and would
    // then make a terminal overwrite the line it was printed on.
    let port = serve_sse_raw("data: {\"a\":1\r,\"b\":2}\n\ndata: [DONE]\n\n");
    let (out, _) = run_bx(port, 5);

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.stdout.contains(&b'\r'), "{:?}", out.stdout);
    assert_eq!(out.stdout.iter().filter(|&&b| b == b'\n').count(), 1);
}

#[test]
fn an_unusable_record_and_an_unusable_body_agree_on_the_exit_code() {
    // The same content through both paths must classify the same way: nothing was lost in
    // transit, so it is exit 1 "result unusable", not exit 5 "server/network".
    let streamed = run_bx(serve_sse_raw("data: <html>\n\n"), 5).0;
    let blocking = run_bx_web(
        serve_response("200 OK", "application/json", "<html>"),
        Stdio::piped(),
    );
    assert_eq!(streamed.status.code(), Some(1));
    assert_eq!(blocking.status.code(), Some(1));
}

#[test]
fn a_byte_order_mark_does_not_make_a_body_unreadable() {
    // Some proxies prepend a BOM. It is legal UTF-8 and illegal JSON, so it used to turn
    // every response into "unexpected non-JSON response" at exit 1.
    let port = serve_response("200 OK", "application/json", "\u{feff}{\"ok\":true}");
    let out = run_bx_web(port, Stdio::piped());

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"ok\":true}\n");
}

#[test]
fn crlf_framed_events_reach_stdout_unchanged() {
    let port = serve_sse_raw("data: {\"n\":0}\r\n\r\ndata: [DONE]\r\n\r\n");
    let (out, _) = run_bx(port, 5);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
}

#[test]
fn extra_stream_false_takes_the_blocking_path() {
    // `--extra stream=false` asks the API for a plain JSON answer. Selecting the client
    // path from a stale flag instead of the finished body fed that JSON to the SSE parser.
    let body = r#"{"choices":[{"message":{"content":"hi"}}]}"#;
    let port = serve_response("200 OK", "application/json", body);

    let cfg = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cfg.path(), "{}").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_bx"))
        .args(["answers", "ping"])
        .args(["--api-key", "TEST"])
        .args(["--base-url", &format!("http://127.0.0.1:{port}")])
        .args(["--extra", "stream=false"])
        .arg("--config")
        .arg(cfg.path())
        .env_remove("BRAVE_SEARCH_API_KEY")
        .env_remove("BRAVE_SEARCH_BASE_URL")
        .env("NO_PROXY", "*")
        .output()
        .unwrap();

    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), body);
}

/// Runs `bx web` — the blocking path — against the local server, with stdout going
/// wherever the caller says.
fn run_bx_web(port: u16, stdout: Stdio) -> Output {
    run_bx_web_with(port, stdout, 30)
}

fn run_bx_web_with(port: u16, stdout: Stdio, timeout_secs: u32) -> Output {
    let cfg = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cfg.path(), "{}").unwrap();
    Command::new(env!("CARGO_BIN_EXE_bx"))
        .args(["web", "ping"])
        .args(["--api-key", "TEST"])
        .args(["--base-url", &format!("http://127.0.0.1:{port}")])
        .args(["--timeout", &timeout_secs.to_string()])
        .arg("--config")
        .arg(cfg.path())
        .stdout(stdout)
        .stderr(Stdio::piped())
        .env_remove("BRAVE_SEARCH_API_KEY")
        .env_remove("BRAVE_SEARCH_BASE_URL")
        .env("NO_PROXY", "*")
        .output()
        .unwrap()
}

#[test]
fn a_body_at_the_size_limit_arrives_whole() {
    // ureq's `LimitReader` errors on a body that reaches the limit exactly, so the limit is
    // set one byte high. Lose that `+1` and every maximum-size response fails instead.
    const LIMIT: usize = 3 * 1024 * 1024;
    let body: &'static str = Box::leak(format!(r#"{{"a":"{}"}}"#, "x".repeat(LIMIT - 8)).into());
    assert_eq!(body.len(), LIMIT);

    let out = run_bx_web(
        serve_response("200 OK", "application/json", body),
        Stdio::piped(),
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert_eq!(out.stdout.len(), LIMIT + 1, "body plus its newline");
}

#[cfg(unix)]
#[test]
fn a_failed_stdout_write_is_reported_not_swallowed() {
    // Writing to a full disk (or a `head` that closed early on the blocking path) must not
    // look like success: an agent reads exit 0 as "the JSON above is complete".
    let port = serve_response("200 OK", "application/json", r#"{"ok":true}"#);
    let full = std::fs::OpenOptions::new().write(true).open("/dev/full");
    let Ok(full) = full else { return }; // not every unix has /dev/full

    let out = run_bx_web(port, Stdio::from(full));

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {stderr}");
    assert!(stderr.contains("writing to stdout"), "stderr: {stderr}");
}

#[test]
fn missing_response_headers_time_out() {
    // The header wait has no timeout of its own; it inherits the send-phase deadline.
    let port = serve_silence();
    let (out, elapsed) = run_bx(port, 1);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "stderr: {stderr}");
    assert!(stderr.contains("timeout"), "stderr: {stderr}");
    assert!(elapsed < PATIENCE, "hung for {elapsed:?}");
    // Without the lower bound the test would also pass if the request failed instantly
    // for an unrelated reason — the property is that the wait is bounded *by --timeout*.
    assert!(
        elapsed >= Duration::from_secs(1),
        "gave up in {elapsed:?}, so the timeout is not what fired"
    );
}

#[test]
fn a_rate_limit_keeps_its_exit_code_when_the_body_drags() {
    // The watchdog guards time-to-response; once the status is known it must stand down.
    // Left armed it fired during the error-body read and replaced 429 with exit 5, losing
    // the one classification an agent branches on — and the retry hint with it.
    // Headers at 3s under `--timeout 3`: the watchdog fires at 4s, while the error-body
    // read is allowed until 6s. The drip is dense enough that ureq's own deadline does not
    // end it first, so the two bounds genuinely cross.
    let port = serve_dripped_error_body("429 Too Many Requests", Duration::from_secs(3));
    let out = run_bx_web_with(port, Stdio::piped(), 3);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(4), "stderr: {stderr}");
    assert!(stderr.contains("429"), "stderr: {stderr}");
}
