//! SSE streaming timeout behaviour, against a local socket.
//!
//! For streamed answers `--timeout` bounds each *read*, not the total length of the
//! stream: deep research emits nothing for minutes at a time while it synthesises, but
//! each individual gap is short-lived. These tests pin both halves — a long stream must
//! survive, a stalled one must still die — plus what happens when the reader leaves.

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

/// Serves one SSE response: `chunks` events spaced `gap` apart. When `finish` is false
/// the server goes silent afterwards instead of sending `[DONE]`. Returns the port.
fn serve_sse(chunks: usize, gap: Duration, finish: bool) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    thread::spawn(move || {
        let Ok((mut sock, _)) = listener.accept() else {
            return;
        };
        read_request(&mut sock);
        let ok = sock.write_all(
            b"HTTP/1.1 200 OK\r\n\
              Content-Type: text/event-stream\r\n\
              Transfer-Encoding: chunked\r\n\r\n",
        );
        if ok.is_err() {
            return;
        }

        for i in 0..chunks {
            thread::sleep(gap);
            if write_chunk(&mut sock, &format!("data: {{\"n\":{i}}}\n\n")).is_err() {
                return;
            }
        }

        if finish {
            let _ = write_chunk(&mut sock, "data: [DONE]\n\n");
            let _ = sock.write_all(b"0\r\n\r\n");
        } else {
            thread::sleep(STALL);
        }
    });

    port
}

/// Accepts a connection and never replies at all — not even response headers.
fn serve_silence() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            read_request(&mut sock);
            thread::sleep(STALL);
        }
    });

    port
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
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    // Read one line, then drop the pipe — the rest of the 10s stream goes nowhere.
    let mut first = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut first)
        .unwrap();
    drop(child.stdout.take());

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
fn missing_response_headers_time_out() {
    // The header wait has no timeout of its own; it inherits the send-phase deadline.
    let port = serve_silence();
    let (out, elapsed) = run_bx(port, 1);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "expected failure, got success");
    assert!(stderr.contains("timeout"), "stderr: {stderr}");
    assert!(elapsed < PATIENCE, "hung for {elapsed:?}");
}
