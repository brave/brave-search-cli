//! SSE streaming behaviour, against a local socket.
//!
//! For streamed answers `--timeout` bounds each *read*, not the total length of the
//! stream: deep research pauses while it synthesises, so each gap is bounded but the total
//! is not. These tests pin both halves — a long stream must survive, a stalled one must
//! still die — plus what reaches stdout, and what happens when the reader leaves.
//!
//! Line-level parsing is unit-tested in `src/api.rs::sse_data`; everything here needs a
//! real socket, a real exit code, or both.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Output, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use tempfile::{NamedTempFile, TempDir};

/// How long a stalling server holds the connection open. Must comfortably exceed the
/// `--timeout` under test so that bx's timeout fires first, not the server hanging up.
const STALL: Duration = Duration::from_secs(15);

/// Upper bound for "bx gave up promptly" assertions.
const PATIENCE: Duration = Duration::from_secs(10);

const SSE: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n";
const JSON: &str = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n";

/// Serves one SSE response: `chunks` events spaced `gap` apart. When `finish` is false
/// the server goes silent afterwards instead of sending `[DONE]`. Returns the port.
fn serve_sse(chunks: usize, gap: Duration, finish: bool) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    thread::spawn(move || {
        let Ok((mut sock, _)) = listener.accept() else {
            return;
        };
        let _ = read_request(&mut sock);
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

/// Serves one Content-Length-framed response verbatim, then closes.
fn serve_raw(head: &str, body: &str) -> u16 {
    serve_owned(head.to_owned(), body.to_owned(), true)
}

/// Serves one response with neither `Content-Length` nor chunking, so the body is
/// close-delimited and EOF is indistinguishable from a mid-stream cut.
fn serve_close_delimited(body: &str) -> u16 {
    serve_owned(SSE.to_owned(), body.to_owned(), false)
}

fn serve_owned(head: String, body: String, framed: bool) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let _ = read_request(&mut sock);
            let framing = if framed {
                format!("Content-Length: {}\r\n", body.len())
            } else {
                "Connection: close\r\n".to_owned()
            };
            let _ = write!(sock, "{head}{framing}\r\n{body}");
            let _ = sock.flush();
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
            let _ = read_request(&mut sock);
            thread::sleep(STALL);
        }
    });

    port
}

/// Drains the request head *and* body, returning the body. Bytes left unread in the receive
/// queue make close() emit RST instead of FIN, discarding the response we already wrote — which
/// surfaces client-side as a spurious "Peer disconnected". bx always sends a `Content-Length`
/// body (ureq's `send(&[u8])` is length-delimited), never chunked.
fn read_request(sock: &mut TcpStream) -> String {
    let mut buf = [0u8; 1024];
    let mut seen = Vec::new();
    let head_end = loop {
        if let Some(i) = seen.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        match sock.read(&mut buf) {
            Ok(0) | Err(_) => return String::new(),
            Ok(n) => seen.extend_from_slice(&buf[..n]),
        }
    };

    let want = content_length(&seen[..head_end]);
    while seen.len() - head_end < want {
        match sock.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => seen.extend_from_slice(&buf[..n]),
        }
    }
    String::from_utf8_lossy(&seen[head_end..]).into_owned()
}

/// Serves one response and hands the request body back to the test. The only way to assert
/// what bx actually put on the wire — `resolve_stream` rewrites a body the user authored, and
/// nothing else in this suite inspects a single outgoing byte.
fn serve_capturing(head: &str, body: &str) -> (u16, mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (tx, rx) = mpsc::channel();
    let (head, body) = (head.to_owned(), body.to_owned());
    thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let _ = tx.send(read_request(&mut sock));
            let _ = write!(sock, "{head}Content-Length: {}\r\n\r\n{body}", body.len());
            let _ = sock.flush();
        }
    });
    (port, rx)
}

fn content_length(head: &[u8]) -> usize {
    String::from_utf8_lossy(head)
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if !name.eq_ignore_ascii_case("content-length") {
                return None;
            }
            value.trim().parse().ok()
        })
        .unwrap_or(0)
}

fn write_chunk(sock: &mut TcpStream, body: &str) -> std::io::Result<()> {
    write!(sock, "{:X}\r\n{body}\r\n", body.len())?;
    sock.flush()
}

/// Everything the binary can read from its environment, redirected away from the
/// developer's own: both API-key variables, the base URL, every proxy variable ureq's
/// `Proxy::try_from_env` consults, and the home directories `dirs` derives the config path
/// from — `resolve_api_key`'s legacy fallback would otherwise *migrate and delete* a real
/// `~/.config/brave-search/api_key`.
///
/// The temporaries are returned because they must outlive the child: a `NamedTempFile`
/// dropped here would delete the config, and an explicit `--config` at a missing path is
/// exit 2. The file itself stays empty — `load_config` reads that as "no config".
fn bx(port: u16) -> (Command, NamedTempFile, TempDir) {
    bx_argv(port, &["answers", "ping"])
}

/// `bx`, with the subcommand and query chosen — `answers -` selects stdin mode, the only mode
/// where bx rewrites a body the caller authored, and a GET subcommand exercises the other
/// half of the shared stdout policy.
fn bx_argv(port: u16, argv: &[&str]) -> (Command, NamedTempFile, TempDir) {
    let cfg = NamedTempFile::new().unwrap();
    let home = TempDir::new().unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_bx"));
    cmd.args(argv)
        .args(["--api-key", "TEST"])
        .args(["--base-url", &format!("http://127.0.0.1:{port}")])
        .arg("--config")
        .arg(cfg.path())
        .env("HOME", home.path())
        .env("XDG_CONFIG_HOME", home.path().join(".config"))
        .env_remove("BRAVE_SEARCH_API_KEY")
        .env_remove("BRAVE_API_KEY")
        .env_remove("BRAVE_SEARCH_BASE_URL")
        .env("NO_PROXY", "*");
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"] {
        cmd.env_remove(var).env_remove(var.to_lowercase());
    }
    (cmd, cfg, home)
}

/// Runs `bx answers` to completion against the local server, and reports how long it took.
fn run_bx(port: u16, args: &[&str]) -> (Output, Duration) {
    let (mut cmd, _cfg, _home) = bx(port);
    let started = Instant::now();
    let output = cmd.args(args).output().unwrap();
    (output, started.elapsed())
}

// ── timeout semantics ────────────────────────────────────────────────

#[test]
fn stream_outlives_the_timeout_while_chunks_keep_arriving() {
    // 3s of streaming under a 2s timeout: fine, because no single gap comes close to 2s.
    // Anchoring the deadline to the response headers instead would abort this at 2s.
    //
    // This is also the tripwire for the ureq upgrade. ureq#1194 (merged, unreleased as of
    // 3.4.0) makes `timeout_recv_body` a *total* budget rather than one re-armed per read; the
    // day that ships, this test goes red and `streaming_agent` needs rethinking, not the test.
    const CHUNKS: usize = 10;
    let port = serve_sse(CHUNKS, Duration::from_millis(300), true);
    let (out, elapsed) = run_bx(port, &["--timeout", "2"]);

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
    let (out, elapsed) = run_bx(port, &["--timeout", "2"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "stderr: {stderr}");
    assert!(elapsed >= Duration::from_secs(2), "gave up in {elapsed:?}");
    assert!(elapsed < PATIENCE, "took {elapsed:?} to give up");
    // The agent must be able to tell a stall from a dead API, and know the knob.
    assert!(stderr.contains("no data for 2s"), "stderr: {stderr}");
    assert!(stderr.contains("--timeout"), "stderr: {stderr}");
    // Whatever arrived before the stall is still delivered — the per-record flush is what
    // survives the `process::exit`, which cannot flush while the stdout lock is held.
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
}

#[test]
fn a_timeout_configured_alongside_research_still_wins() {
    // The whole wiring in one shot: flag → body → answers_timeout → streaming_agent. The
    // 300s research default must not reinstate itself over an explicit --timeout, and this
    // is the only automated cover of that path.
    let port = serve_sse(1, Duration::from_millis(100), false);
    let (out, elapsed) = run_bx(port, &["--enable-research", "--timeout", "2"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "stderr: {stderr}");
    assert!(elapsed < PATIENCE, "used the 300s default: {elapsed:?}");
    assert!(stderr.contains("no data for 2s"), "stderr: {stderr}");
}

#[test]
fn heartbeats_re_arm_the_read_deadline() {
    // The timeout bounds silence, not the gap between *records* — a research pause punctuated
    // by keep-alives must survive. Four 400ms comments under a 1s timeout: if only records
    // re-armed the deadline, the 1.6s without one would kill this.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let _ = read_request(&mut sock);
            let _ = sock.write_all(
                b"HTTP/1.1 200 OK\r\n\
                  Content-Type: text/event-stream\r\n\
                  Transfer-Encoding: chunked\r\n\r\n",
            );
            for _ in 0..4 {
                thread::sleep(Duration::from_millis(400));
                if write_chunk(&mut sock, ": ping\n\n").is_err() {
                    return;
                }
            }
            let _ = write_chunk(&mut sock, "data: {\"n\":0}\n\ndata: [DONE]\n\n");
            let _ = sock.write_all(b"0\r\n\r\n");
        }
    });

    let (out, elapsed) = run_bx(port, &["--timeout", "1"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "exit {:?}: {stderr}",
        out.status.code()
    );
    assert!(
        elapsed > Duration::from_secs(1),
        "finished in {elapsed:?}, too fast to exercise the timeout"
    );
    // Comments are not records: only the JSON reaches stdout.
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
}

#[test]
fn a_config_file_timeout_bounds_a_research_stream() {
    // `--timeout` is not the only source: `configured_timeout` reads the config file too, and
    // a value there must beat the 300s research default just the same. Otherwise a user with
    // `{"timeout": 2}` on disk would wait five minutes for a stalled research answer.
    let port = serve_sse(1, Duration::from_millis(100), false);
    let (mut cmd, cfg, _home) = bx(port);
    std::fs::write(cfg.path(), r#"{"timeout": 2}"#).unwrap();

    let started = Instant::now();
    let out = cmd.arg("--enable-research").output().unwrap();
    let elapsed = started.elapsed();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "{stderr}");
    assert!(elapsed < PATIENCE, "used the 300s default: {elapsed:?}");
    assert!(stderr.contains("no data for 2s"), "{stderr}");
}

#[test]
fn a_comment_only_stream_carries_no_records() {
    // A server that heartbeats and then hangs up produced exit 0 with empty stdout, which an
    // agent cannot tell from an answer that had nothing to say.
    let port = serve_raw(SSE, ": keep-alive\n\n: ping\n\n");
    let (out, _) = run_bx(port, &["--timeout", "5"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("HTTP 200 carried no SSE records"),
        "{stderr}"
    );
}

#[test]
fn a_record_larger_than_one_buffer_fill_arrives_intact() {
    // `read_line_bounded` assembles a line across `fill_buf` calls, and the reader's buffer is
    // 8 KB. A 12 KB record split over three TCP writes must land as one line, not three.
    let big = "x".repeat(12 * 1024);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let expected = big.clone();
    thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let _ = read_request(&mut sock);
            let _ = sock.write_all(
                b"HTTP/1.1 200 OK\r\n\
                  Content-Type: text/event-stream\r\n\
                  Transfer-Encoding: chunked\r\n\r\n",
            );
            let record = format!("data: {{\"x\":\"{big}\"}}\n\ndata: [DONE]\n\n");
            for part in record.as_bytes().chunks(record.len() / 3 + 1) {
                let _ = write_chunk(&mut sock, &String::from_utf8_lossy(part));
                thread::sleep(Duration::from_millis(10));
            }
            let _ = sock.write_all(b"0\r\n\r\n");
        }
    });

    let (out, _) = run_bx(port, &["--timeout", "5"]);

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        format!("{{\"x\":\"{expected}\"}}\n")
    );
}

#[test]
fn a_timeout_mid_line_emits_nothing_partial() {
    // The stall arrives with half a record already buffered. That half must never reach
    // stdout — exit 5 with clean output is the only honest outcome.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let _ = read_request(&mut sock);
            let _ = sock.write_all(
                b"HTTP/1.1 200 OK\r\n\
                  Content-Type: text/event-stream\r\n\
                  Transfer-Encoding: chunked\r\n\r\n",
            );
            let _ = write_chunk(&mut sock, "data: {\"n\":0");
            thread::sleep(STALL);
        }
    });

    let (out, elapsed) = run_bx(port, &["--timeout", "2"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "{stderr}");
    assert!(stderr.contains("no data for 2s"), "{stderr}");
    assert!(out.stdout.is_empty(), "leaked {:?}", out.stdout);
    assert!(elapsed < PATIENCE, "took {elapsed:?}");
}

#[test]
fn a_connection_that_cannot_be_made_is_a_network_error() {
    // Nothing is listening: the generic transport-error arm, which no other test reaches.
    let port = TcpListener::bind("127.0.0.1:0")
        .map(|l| l.local_addr().unwrap().port())
        .unwrap(); // listener dropped, so the port is closed
    let (out, elapsed) = run_bx(port, &["--timeout", "5"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "{stderr}");
    assert!(out.stdout.is_empty());
    assert!(elapsed < PATIENCE, "took {elapsed:?}");
}

#[test]
fn missing_response_headers_time_out() {
    // The header wait has no timeout of its own; it inherits the send-phase deadline. ureq
    // therefore blames `send request`, so bx renames the phase — but the assertion stays
    // off the exact wording, which is a ureq implementation detail.
    let port = serve_silence();
    let (out, elapsed) = run_bx(port, &["--timeout", "2"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "stderr: {stderr}");
    assert!(stderr.contains("no response headers"), "stderr: {stderr}");
    assert!(elapsed >= Duration::from_secs(2), "gave up in {elapsed:?}");
    assert!(elapsed < PATIENCE, "hung for {elapsed:?}");
}

// ── what reaches stdout ──────────────────────────────────────────────

#[test]
fn a_truncated_final_line_is_never_emitted() {
    // A line that never reached its terminator is half a record. Emitting it put half a
    // JSON document on a stream that promises one per line, under exit 0 — corrupt output
    // an agent cannot detect. Both framings that can reach EOF mid-line are covered; under
    // Content-Length the body was provably complete, so the record really was dropped, and
    // dropping it silently is the other half of the same failure.
    for port in [
        serve_raw(SSE, "data: {\"n\":0}\n\ndata: {\"n\":1"),
        serve_close_delimited("data: {\"n\":0}\n\ndata: {\"n\":1"),
    ] {
        let (out, _) = run_bx(port, &["--timeout", "5"]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{stderr}");
        assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
        assert!(
            stderr.contains("unterminated final line"),
            "dropped it silently: {stderr:?}"
        );
    }
}

#[test]
fn a_sole_record_without_its_terminator_is_reported_not_just_missing() {
    // The `carried no SSE records` message is false here — one record arrived, and bx threw it
    // away. Exit 5 is not the answer (a stream ending `data: [DONE]` with no trailing newline
    // is correct and common), so the warning is what makes the exit code readable.
    let port = serve_raw(SSE, "data: {\"n\":0}");
    let (out, _) = run_bx(port, &["--timeout", "5"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("unterminated final line"), "{stderr}");
    assert!(out.stdout.is_empty());
}

#[test]
fn a_sentinel_without_its_terminator_is_still_a_clean_end() {
    // The case that rules out exit 5: everything the caller asked for arrived, and only the
    // sentinel lost its newline. Warned about, but still a success.
    let port = serve_raw(SSE, "data: {\"n\":0}\n\ndata: [DONE]");
    let (out, _) = run_bx(port, &["--timeout", "5"]);

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
}

#[test]
fn a_padded_done_terminates_the_stream() {
    // `data: [DONE] ` used to be written out as the record "[DONE] " while the stream ran
    // on to EOF, so a padded sentinel both corrupted stdout and leaked later records.
    let port = serve_raw(
        SSE,
        "data: {\"n\":0}\n\ndata: [DONE] \n\ndata: {\"n\":1}\n\n",
    );
    let (out, _) = run_bx(port, &["--timeout", "5"]);

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
}

#[test]
fn a_crlf_framed_stream_with_a_bom_and_heartbeats_yields_clean_records() {
    // Everything a real intermediary adds around the records: a stream-leading BOM (which
    // used to hide the first record entirely), CRLF framing, comment heartbeats and the
    // fields bx ignores. Only the JSON documents may reach stdout.
    let port = serve_raw(
        SSE,
        "\u{feff}data: {\"n\":0}\r\n\r\n\
         : ping\r\n\r\n\
         event: message\r\nid: 7\r\nretry: 3000\r\ndata: {\"n\":1}\r\n\r\n\
         data:  \r\n\r\n\
         data: [DONE]\r\n\r\n",
    );
    let (out, _) = run_bx(port, &["--timeout", "5"]);

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
fn records_after_done_are_dropped() {
    let port = serve_raw(
        SSE,
        "data: {\"n\":0}\n\ndata: [DONE]\n\ndata: {\"n\":1}\n\n",
    );
    let (out, _) = run_bx(port, &["--timeout", "5"]);

    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
}

#[test]
fn a_multi_line_data_event_is_rejected_rather_than_split() {
    // SSE says consecutive `data:` fields join with LF. bx does not reassemble events, and the
    // Brave endpoint sends one `data:` per event — so each half is emitted separately, and each
    // half is a JSON fragment. Before records were validated, both reached stdout under exit 0.
    // This pins the limitation, so adopting event-level parsing stays a deliberate change.
    let port = serve_raw(SSE, "data: {\"a\":\ndata: 1}\n\ndata: [DONE]\n\n");
    let (out, _) = run_bx(port, &["--timeout", "5"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {stderr}");
    assert!(stderr.contains("non-JSON record"), "stderr: {stderr}");
    assert!(out.stdout.is_empty(), "{:?}", out.stdout);
}

#[test]
fn a_record_that_is_not_json_never_reaches_stdout() {
    // stdout promises one JSON document per line. A proxy or WAF that injects a plaintext or
    // HTML error into a 200 used to have it delivered verbatim under exit 0, straight into an
    // agent's parser. `{oops` covers the half that a first-byte sniff would have let through.
    for record in ["plain text error", "<html>oops</html>", "{oops"] {
        let port = serve_raw(SSE, &format!("data: {record}\n\ndata: [DONE]\n\n"));
        let (out, _) = run_bx(port, &["--timeout", "5"]);

        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{record}: {stderr}");
        assert!(stderr.contains("non-JSON record"), "{record}: {stderr}");
        assert!(stderr.contains(record), "body not shown: {stderr}");
        assert!(out.stdout.is_empty(), "{record}: {:?}", out.stdout);
    }
}

#[test]
fn a_blocking_body_that_is_not_json_never_reaches_stdout() {
    // The same promise on the other transport. `{oops` passed the old first-byte sniff.
    for body in ["{oops not json", "plain text", "   ", ""] {
        let port = serve_raw(JSON, body);
        let (out, _) = run_bx(port, &["--no-stream", "--timeout", "5"]);

        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{body:?}: {stderr}");
        assert!(stderr.contains("non-JSON response"), "{body:?}: {stderr}");
        assert!(out.stdout.is_empty(), "{body:?}");
    }
}

#[test]
fn a_json_array_body_is_an_answer() {
    // The `[` half of the guard: some endpoints answer with a bare array.
    let port = serve_raw(JSON, "[{\"a\":1}]");
    let (out, _) = run_bx(port, &["--no-stream", "--timeout", "5"]);

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "[{\"a\":1}]\n");
}

#[test]
fn a_stream_cut_mid_frame_is_an_error_not_a_short_answer() {
    // With chunked framing ureq detects the cut, so a truncated answer is exit 5 rather
    // than a silently short one. The record that did arrive is still delivered.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let _ = read_request(&mut sock);
            let _ = sock.write_all(
                b"HTTP/1.1 200 OK\r\n\
                  Content-Type: text/event-stream\r\n\
                  Transfer-Encoding: chunked\r\n\r\n",
            );
            let _ = write_chunk(&mut sock, "data: {\"n\":0}\n\n");
        }
    });

    let (out, _) = run_bx(port, &["--timeout", "5"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "stderr: {stderr}");
    assert!(stderr.contains("reading stream"), "stderr: {stderr}");
    assert!(
        !stderr.contains("no data for"),
        "reported a stall: {stderr}"
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");
}

#[test]
fn an_oversized_line_is_a_stream_error_not_a_stall() {
    // Both read failures share one arm now, so the wrong branch would tell an agent to
    // raise --timeout for a line it can never receive, however long it waits. The line must
    // overshoot by more than one `BufReader` fill: the cap is checked between fills, so a
    // line whose newline arrives in the chunk that crosses it is accepted whole.
    let body = format!("data: {}\n", "x".repeat(2 * 1024 * 1024));
    let port = serve_raw(SSE, &body);
    let (out, _) = run_bx(port, &["--timeout", "5"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "stderr: {stderr}");
    assert!(stderr.contains("exceeds maximum size"), "stderr: {stderr}");
    assert!(!stderr.contains("no data for"), "stderr: {stderr}");
}

// ── exit codes ───────────────────────────────────────────────────────

#[test]
fn a_stream_with_no_records_is_not_a_success() {
    // Exit 0 with empty stdout is indistinguishable from a real answer that had nothing to
    // say. The two bodies take the two different loop exits: the sentinel, and EOF. The
    // status in the message is what tells a redirect or a 204 apart from a 200 that
    // ignored `stream`.
    for body in ["data:\n\ndata: [DONE]\n\n", ""] {
        let port = serve_raw(SSE, body);
        let (out, _) = run_bx(port, &["--timeout", "5"]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{body:?}: {stderr}");
        assert!(
            stderr.contains("HTTP 200 carried no SSE records"),
            "{stderr}"
        );
        assert!(out.stdout.is_empty(), "{body:?}");
    }
}

#[test]
fn a_204_carries_no_records_and_says_so() {
    // A 204 is a success with nothing in it, so it reaches the record check. The status in the
    // message is what tells it apart from a 200 that ignored `stream`.
    let port = serve_raw("HTTP/1.1 204 No Content\r\n", "");
    let (out, _) = run_bx(port, &["--timeout", "5"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("HTTP 204 carried no SSE records"),
        "{stderr}"
    );
}

#[test]
fn a_redirect_is_an_error_even_when_it_carries_records() {
    // Redirects are not followed, so a 3xx body is never an answer — but both transports only
    // tested `>= 400`, so a 302 carrying `data:` lines streamed them to stdout under exit 0,
    // and a 302 carrying JSON was printed as the answer. 304 has no body at all and must not
    // hang waiting for one.
    for (status, head, body, args) in [
        (
            302,
            "HTTP/1.1 302 Found\r\nLocation: http://example.com/\r\n",
            "data: {\"redirect\":1}\n\ndata: [DONE]\n\n",
            &["--timeout", "5"][..],
        ),
        (
            302,
            "HTTP/1.1 302 Found\r\nContent-Type: application/json\r\n",
            r#"{"redirect":1}"#,
            &["--no-stream", "--timeout", "5"][..],
        ),
        (
            304,
            "HTTP/1.1 304 Not Modified\r\n",
            "",
            &["--timeout", "5"][..],
        ),
    ] {
        let port = serve_raw(head, body);
        let (out, elapsed) = run_bx(port, args);

        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{status}: {stderr}");
        // `format_error` must not run a redirect body through the API-error envelope, which
        // reported `{"redirect":1}` as `unknown (302)` — less than the status alone says.
        assert!(stderr.contains(&format!("HTTP {status}")), "{stderr}");
        assert!(out.stdout.is_empty(), "{status}: {:?}", out.stdout);
        assert!(elapsed < PATIENCE, "{status}: took {elapsed:?}");
    }
}

#[test]
fn an_unreadable_error_body_keeps_the_status_exit_code() {
    // ureq only decodes lossily for `text/*`, so a latin-1 error page from a proxy — here one
    // raw 0xE9 — fails `read_to_string`. Reporting that as exit 5 sends an agent into backoff
    // when the status already said "fix your key"; the status is the more reliable signal.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let _ = read_request(&mut sock);
            let body = b"{\"error\":{\"code\":\"X\",\"detail\":\"Caf\xe9\"}}";
            let _ = write!(
                sock,
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\n\r\n",
                body.len()
            );
            let _ = sock.write_all(body);
            let _ = sock.flush();
        }
    });

    let (out, _) = run_bx(port, &["--no-stream", "--timeout", "5"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("failed to read response body"), "{stderr}");
}

#[test]
fn an_error_status_on_the_streaming_path_keeps_its_exit_code() {
    // post_json_stream reads and reports >= 400 itself. A 401 there must still be exit 3
    // with the body on stderr, not the exit 1 of a stream that carried no records.
    let port = serve_raw(
        "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\n",
        r#"{"error":{"code":"UNAUTHORIZED","detail":"bad key"}}"#,
    );
    let (out, _) = run_bx(port, &["--timeout", "5"]);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("bad key"), "{stderr}");
    assert!(out.stdout.is_empty());
}

// ── transport dispatch ───────────────────────────────────────────────

#[test]
fn the_body_not_the_flag_decides_the_transport() {
    // Dispatching on the flag sent a non-streaming request and fed the JSON reply to the
    // SSE parser: exit 0 and no output at all. `stream=0` is the same bug one JSON type
    // over, since `--extra` types a bare 0 as a number. The padding on the body pins the
    // other half of the contract: exactly one JSON document per line, whitespace trimmed
    // from both ends.
    let body = r#"{"choices":[{"message":{"content":"hi"}}]}"#;
    for extra in ["stream=false", "stream=0"] {
        let port = serve_raw(JSON, &format!("\n  {body}  \n"));
        let (out, _) = run_bx(port, &["--extra", extra]);

        assert!(
            out.status.success(),
            "{extra}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&out.stdout), format!("{body}\n"));
    }
}

/// Runs `bx answers -` with `body` on stdin and returns the request body bx actually sent.
fn run_stdin(head: &str, reply: &str, stdin: &str, args: &[&str]) -> (Output, String) {
    let (port, sent) = serve_capturing(head, reply);
    let (mut cmd, _cfg, _home) = bx_argv(port, &["answers", "-"]);
    let mut child = cmd
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    let sent = sent.recv_timeout(PATIENCE).unwrap_or_default();
    (out, sent)
}

#[test]
fn a_stdin_body_gains_the_transport_it_will_be_parsed_with() {
    // A stdin body that omits `stream` was sent as-is while bx parsed the reply as SSE — and
    // the OpenAI-compatible default is `false`, so the server answered with JSON and bx read
    // nothing. Nothing else the caller wrote may change.
    let (out, sent) = run_stdin(
        SSE,
        "data: {\"n\":0}\n\ndata: [DONE]\n\n",
        r#"{"messages":[{"role":"user","content":"x"}],"model":"m"}"#,
        &["--timeout", "5"],
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"n\":0}\n");

    let sent: serde_json::Value = serde_json::from_str(&sent).unwrap();
    assert_eq!(sent["stream"], serde_json::json!(true));
    assert_eq!(sent["model"], serde_json::json!("m"));
    assert_eq!(sent["messages"][0]["content"], serde_json::json!("x"));
}

#[test]
fn a_stdin_body_honours_the_no_stream_flag() {
    // Stdin mode never consulted `--no-stream`, so once bx started writing its decision back
    // it put `"stream": true` on the wire *against* an explicit flag.
    let (out, sent) = run_stdin(
        JSON,
        r#"{"choices":[]}"#,
        r#"{"messages":[]}"#,
        &["--no-stream", "--timeout", "5"],
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert_eq!(String::from_utf8_lossy(&out.stdout), "{\"choices\":[]}\n");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&sent).unwrap()["stream"],
        serde_json::json!(false)
    );
}

#[test]
fn a_stdin_body_that_sets_stream_itself_still_wins() {
    // The body is the more specific instruction, so it beats the flag in both directions — and
    // a numeric `0`, which `--extra`-style typing produces, is normalised on the wire so the
    // request and bx's own parser cannot disagree.
    let (out, sent) = run_stdin(
        JSON,
        r#"{"choices":[]}"#,
        r#"{"messages":[],"stream":0}"#,
        &["--timeout", "5"],
    );

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&sent).unwrap()["stream"],
        serde_json::json!(false)
    );
}

// ── the reader leaving ───────────────────────────────────────────────

#[test]
fn closed_reader_stops_the_stream() {
    // `bx answers … | head -1`: once the reader is gone, draining the rest would burn
    // minutes of wall clock and metered quota for output nobody receives.
    let port = serve_sse(100, Duration::from_millis(100), true);
    let (mut cmd, _cfg, _home) = bx(port);

    let started = Instant::now();
    let mut child = cmd
        .args(["--timeout", "30"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Read one line, then close the pipe — the rest of the 10s stream goes nowhere.
    let mut first = String::new();
    let mut reader = BufReader::new(child.stdout.take().unwrap());
    reader.read_line(&mut first).unwrap();
    drop(reader);

    let out = child.wait_with_output().unwrap();
    let elapsed = started.elapsed();

    assert_eq!(first.trim(), "{\"n\":0}");
    assert!(
        elapsed < Duration::from_secs(5),
        "kept draining for {elapsed:?} after the reader closed"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "a closed reader is not an error: {stderr}"
    );
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
}

#[test]
fn a_reader_that_leaves_before_the_first_record_is_still_a_clean_stop() {
    // `| head -0`. The BrokenPipe return jumps over the no-records check, and it must:
    // reporting "carried no SSE records" when the reader simply left would be a lie.
    let port = serve_sse(10, Duration::from_millis(50), true);
    let (mut cmd, _cfg, _home) = bx(port);

    let mut child = cmd
        .args(["--timeout", "30"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());

    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
}

#[test]
fn a_closed_reader_on_the_blocking_path_is_also_a_clean_stop() {
    // `bx answers … --no-stream | head -0`. The blocking path ignores `write_record`'s return
    // deliberately — a reader that left is not this command's failure — and that must not drift
    // apart from the streaming path, which the same `| head` idiom hits.
    let port = serve_raw(JSON, r#"{"choices":[]}"#);
    let (mut cmd, _cfg, _home) = bx(port);

    let mut child = cmd
        .args(["--no-stream", "--timeout", "5"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());

    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
}

/// `> /dev/full`: ENOSPC is not a closed reader. The blocking path used to swallow write
/// errors outright and exit 0 having written nothing — the one failure this CLI must never
/// hide. The policy is shared by every command, so a GET is checked too, not just `answers`.
#[cfg(target_os = "linux")]
#[test]
fn output_that_cannot_be_written_is_never_reported_as_success() {
    for (head, body, argv) in [
        (
            SSE,
            "data: {\"n\":0}\n\ndata: [DONE]\n\n",
            &["answers", "ping"][..],
        ),
        (
            JSON,
            r#"{"choices":[]}"#,
            &["answers", "ping", "--no-stream"][..],
        ),
        (JSON, r#"{"results":[]}"#, &["suggest", "ping"][..]),
    ] {
        let port = serve_raw(head, body);
        let (mut cmd, _cfg, _home) = bx_argv(port, argv);
        let full = std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/full")
            .unwrap();

        let out = cmd
            .args(["--timeout", "5"])
            .stdout(Stdio::from(full))
            .stderr(Stdio::piped())
            .output()
            .unwrap();

        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{argv:?}: {stderr}");
        assert!(stderr.contains("writing to stdout"), "{argv:?}: {stderr}");
    }
}
