//! `bx council` end to end, against local sockets.
//!
//! Unix only: on Windows `dirs` resolves config and state through Known Folders, which no
//! environment variable redirects, so every spawn would use the real profile.
#![cfg(unix)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use tempfile::TempDir;

mod common;
use common::{Bounded, bx_in, bx_root, finish};

/// `(path, body) -> (status, json)`, decided from the request: members run in parallel, so a
/// queue of canned replies would mix their rounds.
type Handler = Arc<dyn Fn(&str, &str) -> (u16, String) + Send + Sync>;

/// Serves on a detached thread for the life of the process. Returns the port and the recorded
/// request bodies.
fn serve(handler: Handler) -> (u16, Recorded) {
    let (port, bodies, _) = serve_with_heads(handler);
    (port, bodies)
}

type Recorded = Arc<Mutex<Vec<String>>>;

/// `serve`, also recording each request's request and header lines.
fn serve_with_heads(handler: Handler) -> (u16, Recorded, Recorded) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (seen, heads) = (Recorded::default(), Recorded::default());
    let (recorded, headed) = (Arc::clone(&seen), Arc::clone(&heads));

    thread::spawn(move || {
        for sock in listener.incoming().flatten() {
            let (handler, recorded, headed) = (
                Arc::clone(&handler),
                Arc::clone(&recorded),
                Arc::clone(&headed),
            );
            thread::spawn(move || {
                let mut sock = sock;
                let Some((path, head, body)) = read_request_head(&mut sock) else {
                    return;
                };
                headed.lock().unwrap().push(head);
                recorded.lock().unwrap().push(body.clone());
                let (status, payload) = handler(&path, &body);
                let _ = sock.write_all(
                    format!(
                        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                        payload.len()
                    )
                    .as_bytes(),
                );
            });
        }
    });
    (port, seen, heads)
}

fn read_request(sock: &mut std::net::TcpStream) -> Option<(String, String)> {
    read_request_head(sock).map(|(path, _, body)| (path, body))
}

/// `(path, request and header lines, body)`.
fn read_request_head(sock: &mut std::net::TcpStream) -> Option<(String, String, String)> {
    let mut r = BufReader::new(sock.try_clone().ok()?);
    let mut line = String::new();
    r.read_line(&mut line).ok()?;
    let path = line.split_whitespace().nth(1)?.to_string();
    let (mut len, mut head) = (0usize, line.clone());
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).ok()? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
        head.push_str(&h);
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).ok()?;
    Some((path, head, String::from_utf8_lossy(&body).into_owned()))
}

/// The free pre-flights (credit probe, roster read): not generations, so request counts skip them.
fn is_free_probe(path: &str) -> bool {
    path.contains("/api/v1/key") || path.contains("/v1/models")
}

/// The common shape: a credit probe, a search endpoint, and a chat handler.
fn mock(chat: impl Fn(&str) -> (u16, String) + Send + Sync + 'static) -> Handler {
    Arc::new(move |path: &str, body: &str| match path {
        p if p.contains("/api/v1/key") => {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        }
        p if p.contains("/llm/context") => search_hit(),
        // Empty: the "cannot read a list" case `check_models` lets through, so `a/one` passes.
        p if p.contains("/v1/models") => (200, r#"{"data":[]}"#.into()),
        _ => chat(body),
    })
}

/// One OpenRouter chat completion.
fn reply(text: &str) -> (u16, String) {
    (
        200,
        serde_json::json!({
            "model": "test/model", "provider": "Mock",
            "choices": [{"finish_reason": "stop", "message": {"content": text}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01,
                      "is_byok": false, "cost_details": {"upstream_inference_cost": 0.01}}
        })
        .to_string(),
    )
}

fn tool_call(query: &str) -> (u16, String) {
    (
        200,
        serde_json::json!({
            "model": "test/model",
            "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
                {"id": "c1", "function": {"name": "search",
                 "arguments": format!("{{\"query\":\"{query}\"}}")}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5}
        })
        .to_string(),
    )
}

/// True once the conversation carries a tool result, so the next reply should be the review.
fn answered(body: &str) -> bool {
    body.contains(r#""role":"tool""#)
}

fn search_hit() -> (u16, String) {
    (
        200,
        serde_json::json!({"grounding": {"generic": [
            {"url": "https://example.test/a", "title": "A", "snippets": ["evidence text"]}]}})
        .to_string(),
    )
}

struct Run {
    out: Output,
    dir: TempDir,
}

impl Run {
    fn manifest(&self) -> serde_json::Value {
        serde_json::from_slice(&self.out.stdout).unwrap_or_else(|e| {
            panic!(
                "stdout is not JSON: {e}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                String::from_utf8_lossy(&self.out.stdout),
                String::from_utf8_lossy(&self.out.stderr)
            )
        })
    }

    /// The retrieved URLs, read through the manifest's `sources_file`, the path the run published.
    fn sources(&self, member: usize) -> Vec<String> {
        let m = self.manifest();
        let path = m["sources_file"]
            .as_str()
            .unwrap_or_else(|| panic!("no sources_file in the manifest: {m}"));
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        doc["members"][member]["sources"]
            .as_array()
            .unwrap_or_else(|| panic!("member {member} has no sources array: {doc}"))
            .iter()
            .map(|u| u.as_str().unwrap_or_default().to_string())
            .collect()
    }
    fn code(&self) -> i32 {
        self.out.status.code().unwrap_or(-1)
    }
    fn stderr(&self) -> String {
        String::from_utf8_lossy(&self.out.stderr).into_owned()
    }
    /// Panics on a missing file, so a `contains` assert cannot mistake absence for wrong content.
    fn read(&self, name: &str) -> String {
        let path = self.dir.path().join("run").join(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("{}: {e}\n--- stderr ---\n{}", path.display(), self.stderr())
        })
    }
}

/// The sandbox, plus the mock as this wire's and Brave's base URL and a credential for each.
/// Every other transport keeps a dead base URL and no key.
fn bx(wire: &str, port: u16, home: &std::path::Path) -> Command {
    let (up, base) = (
        wire.to_ascii_uppercase(),
        format!("http://127.0.0.1:{port}"),
    );
    let mut c = bx_in(home);
    c.env(format!("BX_COUNCIL_{up}_BASE_URL"), &base)
        // Each wire's own prefix: `check_key` refuses another provider's key shape.
        .env(
            format!("{up}_API_KEY"),
            match wire {
                "anthropic" => "sk-ant-mocktoken",
                "openai" => "sk-proj-mocktoken",
                _ => "sk-or-v1-mocktoken",
            },
        )
        .env("BRAVE_SEARCH_BASE_URL", &base)
        .env("BRAVE_SEARCH_API_KEY", "brave-mock-token");
    c
}

/// A council whose whole panel is routed over `--via openrouter` to one local mock.
fn council(port: u16, args: &[&str]) -> Run {
    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter", "--out"])
        .arg(dir.path().join("run"))
        .args(args)
        .bounded_output();
    Run { out, dir }
}

/// The two-member panel in a caller-owned home, for tests that plant files before the run.
/// `None` runs in the default run directory.
fn council_in(
    port: u16,
    home: &std::path::Path,
    out: Option<&std::path::Path>,
    args: &[&str],
) -> Output {
    let mut c = bx("openrouter", port, home);
    c.args(["council", "SUBJECT", "--via", "openrouter"])
        .args(["-m", "a/one,b/two", "--progress", "false"])
        .args(args);
    if let Some(p) = out {
        c.arg("--out").arg(p);
    }
    c.bounded_output()
}

/// The Anthropic wire with a `council.json`: the route to `computed` pricing, since rates apply
/// only on direct transports.
fn council_rated(port: u16, config: &str, args: &[&str]) -> Run {
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(cfg_dir.join("council.json"), config).unwrap();
    let out = bx("anthropic", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "anthropic",
            "-m",
            "anthropic/claude-opus-5.5,b/two",
            "--progress",
            "false",
            "--out",
        ])
        .arg(dir.path().join("run"))
        .args(args)
        .bounded_output();
    Run { out, dir }
}

/// Answers the free probe, then sends the head after `head_delay` and a body byte every 200 ms
/// that never completes. No read goes silent, so only a whole-call deadline bounds it.
fn drip_server(head_delay: std::time::Duration) -> u16 {
    drip_server_unless(head_delay, |_, _| None)
}

/// `drip_server`, except for the requests `answer` replies to in full.
fn drip_server_unless(
    head_delay: std::time::Duration,
    answer: impl Fn(&str, &str) -> Option<(u16, String)> + Send + Sync + 'static,
) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let answer = Arc::new(answer);
    thread::spawn(move || {
        for sock in listener.incoming().flatten() {
            let answer = Arc::clone(&answer);
            thread::spawn(move || {
                let mut sock = sock;
                let Some((path, body)) = read_request(&mut sock) else {
                    return;
                };
                let full = if is_free_probe(&path) {
                    Some((200, r#"{"data":{}}"#.to_string()))
                } else {
                    answer(&path, &body)
                };
                if let Some((status, b)) = full {
                    let _ = sock.write_all(
                        format!(
                            "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
                            b.len()
                        )
                        .as_bytes(),
                    );
                    return;
                }
                thread::sleep(head_delay);
                let _ = sock.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 200000\r\n\r\n",
                );
                for _ in 0..2_000 {
                    if sock.write_all(b"x").is_err() {
                        return;
                    }
                    let _ = sock.flush();
                    thread::sleep(std::time::Duration::from_millis(200));
                }
            });
        }
    });
    port
}

/// An Anthropic-shaped 200: `ant_parse` refuses `reply`'s OpenRouter shape.
fn anthropic_reply(text: &str) -> (u16, String) {
    (
        200,
        serde_json::json!({
            "model": "claude-opus-5-5", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 100, "output_tokens": 50}
        })
        .to_string(),
    )
}

/// A long, well-sectioned reply, over the 400-character floor.
fn prose() -> String {
    "## A. ANSWER\n".to_string() + &"grounded prose with evidence. ".repeat(40)
}

#[test]
fn a_full_panel_delivers_and_writes_every_artifact() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);

    assert_eq!(r.code(), 0, "stderr: {}", r.stderr());
    let m = r.manifest();
    assert_eq!(
        m["quorum"],
        serde_json::json!({"delivered":2,"total":2,"required":1,"ok":true})
    );
    assert_eq!(m["state"], "ok");
    assert_eq!(m["members"][0]["model"], "a/one", "requested order");
    assert_eq!(m["members"][1]["model"], "b/two");
    assert_eq!(m["members"][0]["status"], "delivered");
    assert_eq!(m["cost"]["llm_usd"], 0.02, "two members at $0.01 each");
    assert_eq!(m["cost_source"], "reported");

    assert!(r.read("manifest.json").contains("\"quorum\""));
    assert!(r.read("brief.md").contains("SUBJECT"));
    let index = r.read("index.md");
    assert!(
        index.contains("a/one") && index.contains("b/two"),
        "{index}"
    );
    assert!(r.read("00-a_one.md").contains("grounded prose"));
    assert!(r.read("01-b_two.md").contains("grounded prose"));
    // Written even when nothing was retrieved: the manifest names it.
    assert!(r.read("sources.json").contains("\"schema\""));

    let bodies = seen.lock().unwrap().clone();
    let chat = bodies
        .iter()
        .find(|b| b.contains("messages"))
        .expect("a chat body");
    let v: serde_json::Value = serde_json::from_str(chat).unwrap();
    assert_eq!(v["tools"][0]["function"]["name"], "search");
    assert_eq!(
        v["messages"][1]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert!(v["session_id"].as_str().is_some_and(|s| !s.is_empty()));
}

#[test]
fn one_member_failing_does_not_take_the_others_with_it() {
    let (port, _) = serve(mock(|body| {
        if body.contains("b/two") {
            (
                400,
                r#"{"error":{"code":"bad_request","message":"nope"}}"#.into(),
            )
        } else {
            reply(&prose())
        }
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--min-quorum",
            "1",
            "--progress",
            "false",
        ],
    );

    assert_eq!(r.code(), 0, "quorum of 1 was met: {}", r.stderr());
    let m = r.manifest();
    assert_eq!(m["quorum"]["delivered"], 1);
    assert_eq!(m["state"], "ok");
    let rejected: Vec<_> = m["members"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|x| x["status"] == "rejected")
        .collect();
    assert_eq!(rejected.len(), 1);
    assert_eq!(rejected[0]["reason"], "provider_error");
    // A 400 is not retried: re-sending a malformed request fails the same way.
    assert_eq!(rejected[0]["action"], "none");
    assert!(r.read("01-b_two.md").contains("REJECTED: provider_error"));
}

#[test]
fn a_uniform_auth_failure_is_exit_3_and_a_missed_quorum_is_exit_6() {
    let (port, _) = serve(mock(|_| (401, r#"{"error":{"message":"bad key"}}"#.into())));
    assert_eq!(
        council(port, &["-m", "a/one,b/two", "--progress", "false"]).code(),
        3
    );

    let short = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--min-quorum",
            "1",
        ],
    );
    assert_eq!(short.code(), 3, "still uniform");

    // Mixed causes: nothing more specific than "quorum not met".
    let (port, _) = serve(mock(|body| {
        if body.contains("b/two") {
            (400, r#"{"error":{"message":"bad request"}}"#.into())
        } else {
            (401, r#"{"error":{"message":"bad key"}}"#.into())
        }
    }));
    assert_eq!(
        council(port, &["-m", "a/one,b/two", "--progress", "false"]).code(),
        6
    );
}

#[test]
fn a_tool_call_is_executed_and_its_result_is_fed_back() {
    let (port, seen) = serve(mock(|body| {
        if answered(body) {
            reply(&prose())
        } else {
            tool_call("rust ureq timeouts")
        }
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);

    assert_eq!(r.code(), 0, "stderr: {}", r.stderr());
    let m = r.manifest();
    assert_eq!(m["brave"]["calls"], 2, "one search per member");
    assert_eq!(m["brave"]["grounded"], 2);
    assert_eq!(m["brave"]["unique_urls"], 1);
    assert_eq!(m["members"][0]["sources_count"], 1);
    assert_eq!(r.sources(0)[0], "https://example.test/a");
    assert_eq!(m["members"][0]["sources_shared"], 1, "both members hit it");
    // Absence is the assertion: a count beside a still-inlined list passes every other check.
    assert!(
        m["members"][0].get("sources").is_none(),
        "the URLs must leave the manifest when sources.json holds them: {m}"
    );

    let bodies = seen.lock().unwrap().clone();
    let follow = bodies
        .iter()
        .find(|b| b.contains("\"role\":\"tool\""))
        .expect("a follow-up carrying the tool result");
    assert!(follow.contains("\"tool_call_id\":\"c1\""), "{follow}");
    assert!(follow.contains("evidence text"));
    assert!(follow.contains("<search_results>"));
}

#[test]
fn a_short_reply_is_reprompted_then_rejected_but_still_written() {
    let (port, seen) = serve(mock(|_| reply("Wrong file. That is chat_template.rs.")));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);

    assert_eq!(r.code(), 6);
    let m = r.manifest();
    assert_eq!(m["members"][0]["reason"], "no_answer");
    assert_eq!(m["members"][0]["terminated_by"], "short_reply");
    assert!(
        r.read("00-a_one.md").contains("Wrong file"),
        "text is published anyway"
    );

    let n = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|b| b.contains("messages"))
        .count();
    assert_eq!(n, 6, "2 members × (1 attempt + 2 re-prompts)");
}

#[test]
fn truncation_escalates_tokens_and_never_changes_effort() {
    let (port, seen) = serve(mock(|body| {
        if body.contains(r#""max_completion_tokens":2000"#) {
            (
                200,
                serde_json::json!({"choices":[{"finish_reason":"length","message":{"content":"cut"}}],
                                   "usage":{"prompt_tokens":1,"completion_tokens":1}})
                .to_string(),
            )
        } else {
            reply(&prose())
        }
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--max-tokens",
            "2000",
        ],
    );

    assert_eq!(r.code(), 0, "stderr: {}", r.stderr());
    let bodies: Vec<serde_json::Value> = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|b| b.contains("messages"))
        .map(|b| serde_json::from_str(b).unwrap())
        .collect();
    let toks: Vec<u64> = bodies
        .iter()
        .map(|b| b["max_completion_tokens"].as_u64().unwrap())
        .collect();
    assert!(
        toks.contains(&2000) && toks.contains(&3000),
        "×3/2: {toks:?}"
    );
    // Effort is part of the cached prefix: changing it invalidates the cache.
    assert!(
        bodies.iter().all(|b| b["reasoning"]["effort"] == "high"),
        "effort must be identical across rounds"
    );
    assert_eq!(
        r.manifest()["members"][0]["limits"]["max_tokens"]["escalations"],
        1
    );
}

#[test]
fn a_truncated_turn_with_a_half_written_search_call_never_reaches_brave() {
    let brave = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = Arc::clone(&brave);
    let (port, _) = serve(Arc::new(move |path: &str, body: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":{}}"#.into());
        }
        if path.contains("/llm/context") {
            counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return search_hit();
        }
        if body.contains(r#""max_completion_tokens":2000"#) {
            return (
                200,
                serde_json::json!({"choices": [{"finish_reason": "length", "message": {
                    "tool_calls": [{"id": "c1", "function":
                        {"name": "search", "arguments": "{\"query\":\"hal"}}]}}], "usage": {}})
                .to_string(),
            );
        }
        reply(&prose())
    }));
    let r = council(
        port,
        &["-m", "a/one", "--max-tokens", "2000", "--progress", "false"],
    );
    assert_eq!(r.manifest()["members"][0]["status"], "delivered");
    assert_eq!(brave.load(std::sync::atomic::Ordering::SeqCst), 0);
}

/// With no published ceiling, the 400 answering a raise is the ceiling; the draft stays on disk.
#[test]
fn a_400_on_an_escalated_round_ends_as_truncated_at_cap_not_provider_error() {
    let (port, _) = serve(mock(|body| {
        if body.contains(r#""max_completion_tokens":2000"#) {
            (
                200,
                serde_json::json!({"choices":[{"finish_reason":"length",
                    "message":{"content":"PAID DRAFT, cut off"}}],"usage":{}})
                .to_string(),
            )
        } else {
            (
                400,
                r#"{"error":{"message":"max_tokens too large"}}"#.into(),
            )
        }
    }));
    let r = council(
        port,
        &["-m", "a/one", "--max-tokens", "2000", "--progress", "false"],
    );
    let one = &r.manifest()["members"][0];
    assert_eq!(one["reason"], "truncated", "{one}");
    assert_eq!(one["terminated_by"], "truncated_at_cap", "{one}");
    assert!(r.read("00-a_one.md").contains("PAID DRAFT"));
}

/// A spend-limit 400 is the provider's, even right after a raise.
#[test]
fn a_400_after_a_raise_that_names_no_token_limit_stays_a_provider_error() {
    let (port, _) = serve(mock(|body| {
        if body.contains(r#""max_completion_tokens":2000"#) {
            (
                200,
                serde_json::json!({"choices":[{"finish_reason":"length",
                    "message":{"content":"cut off"}}],"usage":{}})
                .to_string(),
            )
        } else {
            let msg = "You have reached your specified API usage limits.";
            (400, format!(r#"{{"error":{{"message":"{msg}"}}}}"#))
        }
    }));
    let r = council(
        port,
        &["-m", "a/one", "--max-tokens", "2000", "--progress", "false"],
    );
    let one = &r.manifest()["members"][0];
    assert_eq!(one["reason"], "provider_error", "{one}");
    assert_eq!(one["terminated_by"], "error", "{one}");
}

/// Withdrawing `tools` would re-bill the whole cached prefix.
#[test]
fn an_exhausted_search_budget_yields_a_sentinel_and_keeps_the_tools_attached() {
    let (port, seen) = serve(mock(|body| {
        if body.contains("search budget exhausted") {
            reply(&prose())
        } else {
            tool_call("more evidence please")
        }
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--max-searches",
            "1",
        ],
    );

    assert_eq!(r.code(), 0, "stderr: {}", r.stderr());
    assert_eq!(
        r.manifest()["brave"]["calls"],
        2,
        "one search each, then the cap"
    );

    let bodies = seen.lock().unwrap().clone();
    let sentinel = bodies
        .iter()
        .find(|b| b.contains("search budget exhausted"))
        .expect("the over-budget call must still get a result");
    let v: serde_json::Value = serde_json::from_str(sentinel).unwrap();
    assert!(v["tools"].is_array(), "tools must still be attached");
}

/// An unanswered tool call leaves the conversation permanently invalid.
#[test]
fn an_unknown_tool_name_is_refused_with_a_result_rather_than_executed() {
    let (port, seen) = serve(mock(|body| {
        if body.contains("unknown tool") {
            reply(&prose())
        } else {
            (
                200,
                serde_json::json!({"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[
                    {"id":"c9","function":{"name":"send_email","arguments":"{\"query\":\"x\"}"}}]}}],
                    "usage":{}})
                .to_string(),
            )
        }
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);

    assert_eq!(r.code(), 0, "stderr: {}", r.stderr());
    assert_eq!(
        r.manifest()["brave"]["calls"],
        0,
        "it must not have searched"
    );
    let bodies = seen.lock().unwrap().clone();
    let refusal = bodies
        .iter()
        .find(|b| b.contains("unknown tool"))
        .expect("the refusal is fed back as a tool result");
    assert!(refusal.contains("\"tool_call_id\":\"c9\""));
}

#[test]
fn unknown_tool_calls_do_not_use_up_the_per_turn_search_allowance() {
    let (port, _) = serve(mock(|body| {
        if answered(body) {
            return reply(&prose());
        }
        let calls: Vec<serde_json::Value> = (0..9)
            .map(|i| {
                let name = if i < 8 { "web_search" } else { "search" };
                serde_json::json!({"id": format!("c{i}"),
                    "function": {"name": name, "arguments": "{\"query\":\"q\"}"}})
            })
            .collect();
        (
            200,
            serde_json::json!({"choices":[{"finish_reason":"tool_calls",
                "message":{"tool_calls": calls}}],"usage":{}})
            .to_string(),
        )
    }));
    let r = council(port, &["-m", "a/one", "--progress", "false"]);
    assert_eq!(r.manifest()["brave"]["grounded"], 1, "{}", r.manifest());
}

/// The per-call receive timeout is the member's remaining budget, so `--max-seconds` bounds a
/// round in flight.
#[test]
fn the_budget_stops_a_stalled_round_and_the_member_is_still_reported() {
    let stall = Arc::new(|path: &str, _: &str| {
        if is_free_probe(path) {
            (200, r#"{"data":{}}"#.to_string())
        } else {
            std::thread::sleep(std::time::Duration::from_secs(20));
            reply(&prose())
        }
    });
    let (port, _) = serve(stall);
    let started = std::time::Instant::now();
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-seconds",
            "1",
            "--progress",
            "false",
        ],
    );

    assert!(
        started.elapsed() < std::time::Duration::from_secs(15),
        "the budget did not fire: {:?}",
        started.elapsed()
    );
    assert_eq!(r.code(), 6, "quorum not met: {}", r.stderr());
    let m = r.manifest();
    assert_eq!(
        m["members"].as_array().unwrap().len(),
        2,
        "the panel must not shrink"
    );
    assert_eq!(m["members"][0]["status"], "rejected");
    assert!(!r.read("00-a_one.md").is_empty());
}

/// Exit 2, "nothing sent": zero is a sentinel or a degenerate panel, never a smaller cap.
#[test]
fn every_numeric_flag_refuses_a_value_that_is_not_a_bound() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    // `=` form: clap reads a leading `-` as a flag. The dry run must refuse what the run refuses.
    let cap = "--max-usd must be between 0 and 1000000";
    let bad: [(&[&str], &str); 12] = [
        (&["--max-usd", "NaN"], ""),
        (&["--max-usd", "inf"], ""),
        (&["--max-usd", "-1"], ""),
        (&["--dry-run", "--max-usd=-1"], cap),
        (&["--dry-run", "--max-usd=nan"], cap),
        (&["--dry-run", "--max-usd=-inf"], cap),
        // A ceiling too: `1e100` is finite and non-negative.
        (&["--dry-run", "--max-usd=1e100"], cap),
        (&["--min-quorum", "0"], "0 is not in 1"),
        (&["--concurrency", "0"], "0 is not in 1"),
        (&["--max-searches", "0"], "0 is not in 1"),
        (&["--max-seconds", "0"], "0 is not in 1"),
        (&["--max-seconds", "86401"], "is not in 1"),
    ];
    for (flag, want) in bad {
        let mut args = vec!["-m", "a/one,b/two", "--progress", "false"];
        args.extend_from_slice(flag);
        let r = council(port, &args);
        assert_eq!(r.code(), 2, "{flag:?} was accepted: {}", r.stderr());
        assert!(r.stderr().contains(want), "{flag:?}: {}", r.stderr());
    }
    assert!(
        seen.lock().unwrap().is_empty(),
        "a refused plan sends nothing"
    );

    // Each range's boundary still runs a whole panel.
    for good in [["--concurrency", "1"], ["--max-seconds", "86400"]] {
        let args = ["-m", "a/one,b/two", "--progress", "false", good[0], good[1]];
        let r = council(port, &args);
        assert_eq!(
            r.manifest()["members"].as_array().unwrap().len(),
            2,
            "{good:?} was refused: {}",
            r.stderr()
        );
    }
}

/// `max_usd` in council.json never passes through clap, so `plan` bounds it the same way.
#[test]
fn a_negative_or_huge_max_usd_from_council_json_is_refused_before_the_dry_run() {
    for bad in ["-1", "1e100"] {
        let dir = TempDir::new().unwrap();
        let cfg_dir = dir.path().join("brave-search");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::write(
            cfg_dir.join("council.json"),
            format!(r#"{{"max_usd": {bad}}}"#),
        )
        .unwrap();
        let out = bx_in(dir.path())
            .args([
                "council",
                "SUBJECT",
                "-m",
                "a/one,b/two",
                "--tools",
                "none",
                "--dry-run",
            ])
            .env("OPENROUTER_API_KEY", "sk-or-v1-mocktoken")
            .bounded_output();
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{bad}: {err}");
        assert!(err.contains("must be between 0 and"), "{bad}: {err}");
    }
}

#[test]
fn a_bad_cap_is_named_by_where_it_came_from_and_never_in_three_hundred_digits() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one", "--dry-run", "--max-usd=1e300"]);
    assert_eq!(r.code(), 2);
    assert!(r.stderr().contains("got 1e300"), "{}", r.stderr());

    let r = council_rated(port, r#"{"max_usd": -1}"#, &["--dry-run"]);
    assert_eq!(r.code(), 2);
    assert!(
        r.stderr().contains("council.json max_usd must be between"),
        "{}",
        r.stderr()
    );
}

#[test]
fn status_json_is_always_a_whole_document() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let v: serde_json::Value =
        serde_json::from_str(r.read("status.json").trim()).expect("status.json must always parse");
    assert_eq!(v["total"], 2);
    assert_eq!(v["done"], 2);
    // No temp file from the atomic write survives.
    let leftovers: Vec<_> = std::fs::read_dir(r.dir.path().join("run"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn status_json_carries_both_bounds_the_run_is_measured_against() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--max-usd",
            "12.5",
            "--max-seconds",
            "77",
        ],
    );
    let v: serde_json::Value = serde_json::from_str(r.read("status.json").trim()).unwrap();
    assert_eq!(v["max_seconds"], 77);
    assert_eq!(v["max_usd"], 12.5);
    assert_eq!(r.manifest()["cost"]["max_usd"], 12.5);

    // Uncapped: present and null, never absent.
    let (port2, _) = serve(mock(|_| reply(&prose())));
    let plain = council(port2, &["-m", "a/one,b/two", "--progress", "false"]);
    let v2: serde_json::Value = serde_json::from_str(plain.read("status.json").trim()).unwrap();
    assert!(v2["max_usd"].is_null(), "{v2}");
}

/// Held at the first chat request: its arrival proves the pre-flight write happened, and the
/// writer rewrites only on `member_done`.
#[test]
fn a_live_run_answers_status_and_publishes_its_cap_before_the_first_reply() {
    // `(chat requests that arrived, released)`.
    let gate = Arc::new((Mutex::new((0usize, false)), std::sync::Condvar::new()));
    let held = Arc::clone(&gate);
    let (port, _) = serve(mock(move |_| {
        let (lock, cv) = &*held;
        let mut st = lock.lock().unwrap();
        st.0 += 1;
        cv.notify_all();
        while !st.1 {
            let (next, timed_out) = cv
                .wait_timeout(st, std::time::Duration::from_secs(30))
                .unwrap();
            st = next;
            if timed_out.timed_out() {
                break;
            }
        }
        drop(st);
        reply(&prose())
    }));

    let dir = TempDir::new().unwrap();
    let run_dir = dir.path().join("run");
    let child = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter", "--out"])
        .arg(&run_dir)
        .args([
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--max-usd",
            "12.5",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let (lock, cv) = &*gate;
    let mut st = lock.lock().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while st.0 == 0 && std::time::Instant::now() < deadline {
        let (next, _) = cv
            .wait_timeout(st, std::time::Duration::from_millis(100))
            .unwrap();
        st = next;
    }
    let arrived = st.0;
    drop(st);
    let live = std::fs::read_to_string(run_dir.join("status.json"));
    let polled = bx("openrouter", port, dir.path())
        .args(["council", "status", "--out"])
        .arg(&run_dir)
        .bounded_output();

    // Release before asserting: a failure must not become a hang.
    {
        let mut open = lock.lock().unwrap();
        open.1 = true;
        cv.notify_all();
    }
    let out = finish(child);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

    assert!(arrived > 0, "no chat request in 30s — stderr: {stderr}");
    let live =
        live.unwrap_or_else(|e| panic!("no status.json before the first request: {e}\n{stderr}"));
    let v: serde_json::Value = serde_json::from_str(live.trim()).expect(&live);
    assert_eq!(v["max_usd"], 12.5, "a live run misreported its cap: {v}");
    // No member had finished: this is the pre-flight file, not the writer's.
    assert_eq!(v["done"], 0, "{v}");
    assert_eq!(v["total"], 2, "{v}");
    assert_eq!(out.status.code(), Some(0), "{stderr}");

    assert_eq!(
        polled.status.code(),
        Some(0),
        "polling a live run must not report the code for 'nothing sent': {}",
        String::from_utf8_lossy(&polled.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&polled.stdout).unwrap();
    assert_eq!(v["done"], 0, "no member had finished yet");
    assert_eq!(v["total"], 2, "and the panel size is known from the start");
}

/// In both the text and the JSON dry run.
#[test]
fn a_sub_micro_cap_is_never_printed_as_no_cap() {
    let dry = |format: &str| {
        let dir = TempDir::new().unwrap();
        let out = bx("openrouter", 1, dir.path())
            .args(["council", "SUBJECT", "--via", "openrouter"])
            .args(["-m", "a/one,b/two", "--dry-run", "--format", format])
            .args(["--max-usd", "0.0000001"])
            .bounded_output();
        assert_eq!(
            out.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    let text = dry("text");
    assert!(text.contains("--max-usd $0.0000001,"), "{text}");
    assert!(!text.contains("--max-usd $0,"), "{text}");

    let v: serde_json::Value = serde_json::from_str(&dry("json")).unwrap();
    assert_eq!(v["cost"]["max_usd"], 1e-7, "{v}");
}

fn dry_run(format: &str, extra: &[&str]) -> String {
    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", 1, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter"])
        .args(["-m", "a/one,b/two", "--dry-run", "--format", format])
        .args(extra)
        .bounded_output();
    assert_eq!(out.status.code(), Some(0));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn a_text_dry_run_names_the_clock() {
    let text = dry_run("text", &["--max-seconds", "777"]);
    assert!(text.contains("seconds=777"), "{text}");
}

#[test]
fn the_dry_run_counts_its_egress_in_members_not_providers() {
    let v: serde_json::Value = serde_json::from_str(&dry_run("json", &[])).unwrap();
    assert_eq!(v["egress"]["members"], 2, "{v}");
    assert!(v["egress"].get("providers").is_none(), "{v}");
    assert!(dry_run("text", &[]).contains("to each of 2 members"));
}

#[test]
fn the_dry_run_cost_note_states_no_private_measurement() {
    let v: serde_json::Value = serde_json::from_str(&dry_run("json", &[])).unwrap();
    let note = v["cost"]["note"].as_str().unwrap();
    assert!(!note.contains("measured"), "{note}");
}

#[test]
fn a_pre_flight_refusal_sends_nothing_and_exits_2() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    // Two spellings of one model collapse to one member; one model named once runs.
    let r = council(
        port,
        &[
            "-m",
            "direct:anthropic/claude-opus-5.5,anthropic/claude-opus-5.5",
            "--progress",
            "false",
        ],
    );
    assert_eq!(r.code(), 2);
    // Names what happened, not the rule: "needs 2" invites a third spelling of the same model.
    assert!(
        r.stderr().contains("duplicate models were merged")
            && r.stderr().contains("2 entries name one model"),
        "{}",
        r.stderr()
    );
    assert!(seen.lock().unwrap().is_empty(), "nothing may be sent");

    // Too large: only the ceiling is claimed, since one model named once runs.
    let many: Vec<String> = (0..40).map(|i| format!("v{i}/m")).collect();
    let r = council(port, &["-m", &many.join(","), "--progress", "false"]);
    assert_eq!(r.code(), 2);
    assert!(r.stderr().contains("at most 32"), "{}", r.stderr());
    assert!(
        !r.stderr().contains("between 2"),
        "the message claims a minimum the code does not enforce: {}",
        r.stderr()
    );

    // A quorum larger than the panel can never be met.
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--min-quorum",
            "3",
            "--progress",
            "false",
        ],
    );
    assert_eq!(r.code(), 2);
    assert!(r.stderr().contains("exceeds the panel size"));

    // A panel of one: the size is named, because the panel may be what shrank.
    let r = council(
        port,
        &["-m", "only/one", "--min-quorum", "2", "--progress", "false"],
    );
    assert_eq!(r.code(), 2, "{}", r.stderr());
    assert!(
        r.stderr().contains("exceeds the panel size (1)"),
        "{}",
        r.stderr()
    );
    assert!(seen.lock().unwrap().is_empty(), "a refusal spent money");
}

#[test]
fn a_dry_run_refuses_an_exhausted_balance_with_the_real_runs_code() {
    let (port, _) = serve(Arc::new(|_: &str, _: &str| {
        (200, r#"{"data":{"limit":10,"limit_remaining":0}}"#.into())
    }));
    let r = council(port, &["-m", "a/one", "--dry-run"]);
    assert_eq!(r.code(), 8, "{}", r.stderr());
}

/// Enforced against real spend between rounds, not a pre-flight estimate.
#[test]
fn max_usd_stops_members_on_real_spend() {
    // Anthropic direct has a compiled-in rate, so its cost is computed here.
    let (port, seen) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "content": [{"type": "text", "text": "short"}],
                "usage": {"input_tokens": 4_000_000, "output_tokens": 200_000}
            })
            .to_string(),
        )
    }));
    // Both rated so both reach the cap; full slugs, since `rate_for` keys on `bare`, which keeps
    // the vendor.
    let r = council_rated(
        port,
        r#"{"rates":{"anthropic/claude-opus-5.5":[5,25],"b/two":[5,25]}}"#,
        &["--max-usd", "0.01"],
    );
    let m = r.manifest();
    // Not the exit code: under load a member can lose the mock and be rejected for that instead.
    // `exit_for` is unit-tested.
    let members = m["members"].as_array().unwrap();
    assert!(
        members.iter().any(|x| x["reason"] == "max_usd"),
        "no member was stopped by the cap: {m}"
    );
    // The run starts: the cap is measured, not predicted.
    assert!(
        !seen.lock().unwrap().is_empty(),
        "the cap was enforced without ever asking what a round costs"
    );
    for x in members {
        assert!(x["rounds"].as_u64().unwrap() < 4, "ran past the cap: {x}");
    }
}

/// The empty truncated round is still charged.
#[test]
fn a_truncated_round_followed_by_the_spend_cap_is_gated_as_cost_not_truncated() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({"choices": [{"finish_reason": "length", "message": {"content": ""}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 50, "cost": 0.01}})
            .to_string(),
        )
    }));
    let r = council(
        port,
        &["-m", "a/one", "--max-usd", "0.005", "--progress", "false"],
    );
    let one = &r.manifest()["members"][0];
    assert_eq!(one["reason"], "max_usd", "{one}");
    assert_eq!(one["terminated_by"], "max_usd", "{one}");
    assert_eq!(one["cost"]["llm_usd"], 0.01, "{one}");
    assert_eq!(r.code(), 7, "{}", r.stderr());
}

#[test]
fn a_provider_402_after_the_cap_is_spent_is_still_reported_as_the_providers_error() {
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
    let (b_sent, a_paid) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let wait = |flag: &AtomicBool| {
        while !flag.load(SeqCst) {
            thread::sleep(std::time::Duration::from_millis(20));
        }
    };
    let (port, _) = serve(mock(move |body: &str| {
        if body.contains(r#""model":"a/one""#) {
            // Not before `b/two` has sent its round, so the cap cannot stop it unsent.
            wait(&b_sent);
            a_paid.store(true, SeqCst);
            return (
                200,
                serde_json::json!({
                    "choices": [{"finish_reason": "stop", "message": {"content": prose()}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 1.0}
                })
                .to_string(),
            );
        }
        // Answer only once `a/one`'s bill has had time to land on the shared total.
        b_sent.store(true, SeqCst);
        wait(&a_paid);
        thread::sleep(std::time::Duration::from_secs(1));
        (
            402,
            r#"{"error":{"message":"Insufficient credits","metadata":{"error_type":"payment_required"}}}"#
                .into(),
        )
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-usd",
            "0.5",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    let b = &m["members"][1];
    assert_eq!(b["reason"], "provider_error", "{b}");
    assert_eq!(b["terminated_by"], "error", "{b}");
    assert_eq!(b["action"], "none", "{b}");
}

#[test]
fn the_text_format_is_stable_and_the_run_directory_is_private() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--format",
            "text",
            "--progress",
            "false",
        ],
    );
    let out = String::from_utf8_lossy(&r.out.stdout);
    assert!(out.starts_with("council: 2/2 delivered"), "{out}");
    assert!(out.contains("  ok      a/one"), "{out}");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(r.dir.path().join("run"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700, "run directories hold model output");
    }
}

/// `--out` can point anywhere, so the umask must not decide who reads model output.
#[cfg(unix)]
#[test]
fn every_artifact_is_written_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(
        port,
        &["-m", "a/one,b/two", "--progress", "false", "--trace"],
    );
    assert_eq!(r.code(), 0, "{}", r.stderr());

    let run = r.dir.path().join("run");
    for name in [
        "manifest.json",
        "index.md",
        "brief.md",
        "status.json",
        "00-a_one.md",
        "00-a_one.transcript.json",
    ] {
        let mode = std::fs::metadata(run.join(name))
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "{name} is {:o}", mode & 0o777);
    }
    // And whatever else the run wrote: `trace.jsonl` is appended, not renamed into place.
    for e in std::fs::read_dir(&run).unwrap().flatten() {
        let mode = e.metadata().unwrap().permissions().mode();
        assert_eq!(
            mode & 0o777,
            0o600,
            "{:?} is {:o}",
            e.file_name(),
            mode & 0o777
        );
    }
}

/// A model echoing its tool input would put the keys in the trace.
#[test]
fn the_brave_key_is_redacted_as_well_as_the_inference_keys() {
    let (port, _) = serve(mock(|body| {
        if answered(body) {
            reply(&prose())
        } else {
            tool_call("brave-mock-token and sk-or-v1-mocktoken")
        }
    }));
    let r = council(
        port,
        &["-m", "a/one,b/two", "--trace", "--progress", "false"],
    );
    let trace = r.read("trace.jsonl");
    assert!(!trace.is_empty());
    assert!(!trace.contains("brave-mock-token"), "the Brave key leaked");
    assert!(
        !trace.contains("sk-or-v1-mocktoken"),
        "the inference key leaked"
    );
    assert!(
        trace.contains("***"),
        "and it was redacted, not merely absent"
    );
}

#[test]
fn a_failed_member_still_gets_a_transcript() {
    let (port, _) = serve(mock(|_| {
        (
            400,
            r#"{"error":{"code":"bad_request","message":"nope"}}"#.into(),
        )
    }));
    let r = council(
        port,
        &["-m", "a/one,b/two", "--trace", "--progress", "false"],
    );
    assert_eq!(r.code(), 6);
    let t = r.read("00-a_one.transcript.json");
    assert!(!t.is_empty(), "a fatal member wrote no transcript");
    let v: serde_json::Value = serde_json::from_str(&t).unwrap();
    assert_eq!(v["model"], "a/one");
}

#[test]
fn a_successful_transcript_ends_with_the_answer_it_published() {
    let (port, _) = serve(mock(|body| {
        if answered(body) {
            reply(&prose())
        } else {
            tool_call("evidence")
        }
    }));
    let r = council(
        port,
        &["-m", "a/one,b/two", "--trace", "--progress", "false"],
    );
    let v: serde_json::Value = serde_json::from_str(&r.read("00-a_one.transcript.json")).unwrap();
    let roles: Vec<&str> = v["turns"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["role"].as_str())
        .collect();
    assert_eq!(
        roles.last(),
        Some(&"assistant"),
        "the member's own answer is missing: {roles:?}"
    );
}

/// A long round-1 deliberation that also calls a tool must not outrank the final review.
#[test]
fn the_published_reply_is_the_last_one_not_the_longest_one() {
    let (port, _) = serve(mock(|body| {
        if answered(body) {
            reply(&("## A. ANSWER\n".to_string() + &"the real review. ".repeat(40)))
        } else {
            (
                200,
                serde_json::json!({"choices":[{"finish_reason":"tool_calls","message":{
                    "content": "Let me think about this. ".repeat(400),
                    "tool_calls":[{"id":"c1","function":{"name":"search",
                                   "arguments":"{\"query\":\"q\"}"}}]}}],
                    "usage":{}})
                .to_string(),
            )
        }
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    assert_eq!(r.code(), 0, "{}", r.stderr());
    let published = r.read("00-a_one.md");
    assert!(
        published.contains("the real review"),
        "published the deliberation instead"
    );
    assert!(!published.contains("Let me think about this"));
}

/// The counterweight to the last-reply rule: a short terminal turn does not erase a long draft.
#[test]
fn a_long_draft_survives_a_short_follow_up() {
    let (port, _) = serve(mock(|body| {
        if answered(body) {
            reply("Already covered above.")
        } else {
            (
                200,
                serde_json::json!({"choices":[{"finish_reason":"tool_calls","message":{
                    "content": format!("## A. FATAL\n{}", "a real finding. ".repeat(60)),
                    "tool_calls":[{"id":"c1","function":{"name":"search",
                                   "arguments":"{\"query\":\"q\"}"}}]}}],
                    "usage":{}})
                .to_string(),
            )
        }
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let published = r.read("00-a_one.md");
    assert!(
        published.contains("a real finding"),
        "the paid-for review was deleted by a shorter later turn:\n{published}"
    );
}

/// Complete beats cut off, whatever the lengths.
#[test]
fn a_finished_draft_outranks_a_longer_turn_cut_off_after_it() {
    use serde_json::json;
    let (port, _) = serve(mock(|body| {
        let (finish, content, calls) = if answered(body) {
            ("length", "cut off mid-sentence. ".repeat(200), json!(null))
        } else {
            let call = json!([{"id": "c1", "function":
                {"name": "search", "arguments": "{\"query\":\"q\"}"}}]);
            ("tool_calls", prose(), call)
        };
        (
            200,
            json!({"choices": [{"finish_reason": finish,
                "message": {"content": content, "tool_calls": calls}}], "usage": {}})
            .to_string(),
        )
    }));
    let r = council(
        port,
        &["-m", "a/one", "--max-steps", "2", "--progress", "false"],
    );
    let one = &r.manifest()["members"][0];
    assert_eq!(one["status"], "delivered", "{one}");
    let published = r.read("00-a_one.md");
    assert!(published.contains("## A. ANSWER") && !published.contains("cut off"));
}

/// A turn that truncates at the cap has still been paid for in full.
#[test]
fn text_from_a_turn_that_truncated_at_the_cap_is_still_published() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({"choices":[{"finish_reason":"length","message":{
                "content": format!("## A. FATAL\n{}", "cut off mid-sentence. ".repeat(60))}}],
                "usage":{}})
            .to_string(),
        )
    }));
    // Start at the cap so the first truncation is terminal.
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--max-tokens",
            "128000",
        ],
    );
    assert_eq!(r.code(), 6, "truncated at the cap is a rejection");
    let m = r.manifest();
    assert_eq!(m["members"][0]["reason"], "truncated");
    assert!(
        m["members"][0]["chars"].as_u64().unwrap_or(0) > 100,
        "the text was discarded: {}",
        m["members"][0]
    );
    assert!(r.read("00-a_one.md").contains("cut off mid-sentence"));
}

/// Every run appends to `history.jsonl`; a line written in two syscalls would interleave.
#[test]
fn concurrent_runs_never_interleave_a_history_line() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let shared = TempDir::new().unwrap();

    let kids: Vec<_> = (0..6)
        .map(|i| {
            let dir = shared.path().join(format!("r{i}"));
            bx("openrouter", port, shared.path())
                .args([
                    "council",
                    "SUBJECT",
                    "--via",
                    "openrouter",
                    "--progress",
                    "false",
                ])
                .args(["-m", "a/one,b/two,c/three", "--out"])
                .arg(&dir)
                .stdout(std::process::Stdio::null())
                .spawn()
                .unwrap()
        })
        .collect();
    for k in kids {
        finish(k);
    }

    let log = shared.path().join("bx/council/history.jsonl");
    let body = std::fs::read_to_string(&log).expect("--out must not suppress the history log");
    let lines: Vec<&str> = body.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 18, "6 runs × 3 members");
    for (i, l) in lines.iter().enumerate() {
        serde_json::from_str::<serde_json::Value>(l)
            .unwrap_or_else(|e| panic!("history line {i} is corrupt: {e}\n{l}"));
    }
}

#[test]
fn an_append_after_a_torn_tail_starts_on_its_own_line() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let log = dir.path().join("bx/council/history.jsonl");
    std::fs::create_dir_all(log.parent().unwrap()).unwrap();
    std::fs::write(&log, "{\"run\":\"torn\"").unwrap();
    council_in(port, dir.path(), Some(&dir.path().join("run")), &[]);
    let body = std::fs::read_to_string(&log).unwrap();
    let whole = body
        .lines()
        .filter(|l| serde_json::from_str::<serde_json::Value>(l).is_ok())
        .count();
    assert_eq!(whole, 2, "only the torn record may be lost:\n{body}");
}

#[test]
fn a_write_failure_after_inference_is_not_reported_as_nothing_sent() {
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    // Planted mid-run: `--out` refuses a directory that already holds `index.md`.
    let (index, once) = (run.join("index.md"), std::sync::Once::new());
    let (port, _) = serve(mock(move |_| {
        once.call_once(|| std::fs::create_dir(&index).unwrap());
        reply(&prose())
    }));

    let out = council_in(port, dir.path(), Some(&run), &[]);

    assert_eq!(out.status.code(), Some(0), "the panel delivered");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("warning:"),
        "the failure must still be reported: {stderr}"
    );
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).expect("manifest on stdout");
    assert_eq!(m["quorum"]["delivered"], 2);
}

#[test]
fn a_run_whose_index_write_failed_never_points_a_reader_at_it() {
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    // Planted mid-run, past the `--out` guard: only the index write then fails.
    let plant = run.join("index.md");
    let (port, _) = serve(mock(move |_| {
        std::fs::create_dir_all(&plant).ok();
        reply(&prose())
    }));
    let out = council_in(port, dir.path(), Some(&run), &[]);
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).expect("manifest on stdout");
    assert!(m.get("index").is_none(), "{m}");
    assert!(!m["hint"].as_str().unwrap().contains("index.md"), "{m}");
    assert_eq!(m["log_write_errors"], 1, "{m}");
}

#[test]
fn a_failed_manifest_write_is_counted_and_under_text_goes_to_stderr() {
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    // Planted mid-run, past the finished-run guard: only the manifest's rename then fails.
    let plant = run.join("manifest.json").join("x");
    let (port, _) = serve(mock(move |_| {
        std::fs::create_dir_all(&plant).ok();
        reply(&prose())
    }));
    let json = council_in(port, dir.path(), Some(&run), &[]);
    let m: serde_json::Value = serde_json::from_slice(&json.stdout).expect("manifest on stdout");
    assert_eq!(m["log_write_errors"], 1, "{m}");

    std::fs::remove_dir_all(&run).unwrap();
    let text = council_in(port, dir.path(), Some(&run), &["--format", "text"]);
    let err = String::from_utf8_lossy(&text.stderr);
    let m: serde_json::Value = err
        .lines()
        .find_map(|l| serde_json::from_str(l).ok())
        .unwrap_or_else(|| panic!("no manifest on stderr:\n{err}"));
    assert_eq!(m["log_write_errors"], 1, "{m}");
    assert!(!err.contains("re-run"), "a re-run buys a new panel: {err}");
}

/// The 512-char bound is for manifest strings, never the reply.
#[test]
fn a_long_review_is_published_whole() {
    let long = "## A. FATAL\n".to_string() + &"a real finding with detail. ".repeat(400);
    let want = long.chars().count();
    let (port, _) = serve(mock(move |_| reply(&long)));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);

    assert_eq!(r.code(), 0, "{}", r.stderr());
    let body = r.read("00-a_one.md");
    assert!(
        !body.contains("[truncated]"),
        "the review was truncated:\n{}",
        &body[..200.min(body.len())]
    );
    assert!(
        body.chars().count() > want,
        "published {} chars of a {want}-char review",
        body.chars().count()
    );
    assert_eq!(r.manifest()["members"][0]["chars"], want as u64);
}

/// The turn is echoed verbatim, so answering only a prefix of its calls invalidates the
/// conversation.
#[test]
fn every_tool_call_is_answered_however_many_there_are() {
    let (port, seen) = serve(mock(|body| {
        if answered(body) {
            reply(&prose())
        } else {
            let calls: Vec<serde_json::Value> = (0..40)
                .map(|i| {
                    serde_json::json!({"id": format!("c{i}"),
                        "function": {"name": "search", "arguments": "{\"query\":\"q\"}"}})
                })
                .collect();
            (
                200,
                serde_json::json!({"choices":[{"finish_reason":"tool_calls",
                    "message":{"tool_calls": calls}}],"usage":{}})
                .to_string(),
            )
        }
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    assert_eq!(r.code(), 0, "{}", r.stderr());

    let bodies = seen.lock().unwrap().clone();
    let follow = bodies
        .iter()
        .find(|b| b.contains("\"role\":\"tool\""))
        .expect("a follow-up carrying results");
    let v: serde_json::Value = serde_json::from_str(follow).unwrap();
    let msgs = v["messages"].as_array().unwrap();
    let echoed = msgs
        .iter()
        .filter_map(|m| m["tool_calls"].as_array())
        .map(Vec::len)
        .sum::<usize>();
    let answered = msgs.iter().filter(|m| m["role"] == "tool").count();
    assert_eq!(echoed, 40, "the whole turn is echoed");
    assert_eq!(answered, 40, "and every call in it is answered");
    // Only the first 8 actually search; the rest get the sentinel.
    assert_eq!(r.manifest()["brave"]["calls"], 16, "8 per member");
}

/// Written through, a link would publish the subject to a file the planter chose, and a loose
/// file would keep its mode.
#[cfg(unix)]
#[test]
fn a_planted_link_or_loose_file_at_an_artifact_is_replaced_at_0600() {
    use std::os::unix::fs::{PermissionsExt, symlink};
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    std::fs::create_dir_all(&run).unwrap();

    let victim = dir.path().join("victim.txt");
    std::fs::write(&victim, "ORIGINAL").unwrap();
    // Member files: `--out` refuses a directory already holding a fixed artifact name.
    symlink(&victim, run.join("00-a_one.md")).unwrap();
    let loose = run.join("01-b_two.md");
    std::fs::write(&loose, "stale").unwrap();
    std::fs::set_permissions(&loose, std::fs::Permissions::from_mode(0o666)).unwrap();

    let out = council_in(port, dir.path(), Some(&run), &[]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert_eq!(
        std::fs::read_to_string(&victim).unwrap(),
        "ORIGINAL",
        "the subject was written through a symlink"
    );
    for name in ["00-a_one.md", "01-b_two.md"] {
        let p = run.join(name);
        let meta = std::fs::symlink_metadata(&p).unwrap();
        assert!(meta.file_type().is_file(), "{name} is still a symlink");
        assert_eq!(
            meta.permissions().mode() & 0o777,
            0o600,
            "{name} kept its old mode"
        );
    }
}

#[test]
fn progress_lines_go_to_stderr_and_stdout_stays_one_json_document() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two"]);
    assert!(
        r.stderr().contains("[1/2]") || r.stderr().contains("[2/2]"),
        "{}",
        r.stderr()
    );
    assert_eq!(
        String::from_utf8_lossy(&r.out.stdout)
            .trim()
            .lines()
            .count(),
        1,
        "stdout is one document per run"
    );
    r.manifest();
}

/// Appends bypass `write_private`'s temp-and-rename, so `open_append` refuses anything but a file
/// it alone links. `history.jsonl`: `--out` refuses a planted trace outright.
#[test]
fn an_append_log_that_is_not_a_plain_file_is_refused_counted_and_named() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    for plant in ["symlink", "hardlink", "fifo", "directory"] {
        let dir = TempDir::new().unwrap();
        let state = dir.path().join("bx/council");
        std::fs::create_dir_all(&state).unwrap();
        let log = state.join("history.jsonl");
        let other = dir.path().join("other.txt");
        std::fs::write(&other, "ORIGINAL").unwrap();
        match plant {
            "symlink" => std::os::unix::fs::symlink(&other, &log).unwrap(),
            "hardlink" => std::fs::hard_link(&other, &log).unwrap(),
            "fifo" => assert!(Command::new("mkfifo").arg(&log).status().unwrap().success()),
            _ => std::fs::create_dir(&log).unwrap(),
        }

        // Bounded by `finish`: a blocking open on the FIFO never returns.
        let out = council_in(
            port,
            dir.path(),
            Some(&dir.path().join("run")),
            &["--trace"],
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(0), "{plant}: {stderr}");
        let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(m["quorum"]["delivered"], 2, "{plant}: {m}");
        // One per lost row, not one per unopenable log.
        assert_eq!(m["log_write_errors"], 2, "{plant}: {m}");
        assert!(
            stderr.contains("history.jsonl"),
            "{plant}: nothing named the log: {stderr}"
        );
        assert_eq!(
            std::fs::read_to_string(&other).unwrap(),
            "ORIGINAL",
            "{plant}: the run was appended through the plant"
        );
    }
}

/// Only a directory this command creates is its to harden.
#[cfg(unix)]
#[test]
fn an_existing_out_directory_keeps_its_own_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::set_permissions(&run, std::fs::Permissions::from_mode(0o755)).unwrap();

    let out = council_in(port, dir.path(), Some(&run), &[]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        std::fs::metadata(&run).unwrap().permissions().mode() & 0o777,
        0o755,
        "the operator's directory was silently re-moded"
    );
    let f = std::fs::metadata(run.join("manifest.json")).unwrap();
    assert_eq!(f.permissions().mode() & 0o777, 0o600);
}

/// The partial is then the only copy of a paid-for reply.
#[cfg(unix)]
#[test]
fn a_failed_member_write_keeps_the_partial_it_would_otherwise_delete() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    std::fs::create_dir_all(&run).unwrap();
    // A directory where the member file belongs: the write fails, the partial does not.
    std::fs::create_dir(run.join("00-a_one.md")).unwrap();

    let out = council_in(port, dir.path(), Some(&run), &[]);

    let manifest: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        manifest["members"][0]["status"], "rejected",
        "a reply that could not be published is not delivered"
    );
    let partial = std::fs::read_to_string(run.join("00-a_one.md.partial")).unwrap_or_default();
    assert!(
        partial.contains("grounded prose"),
        "the only copy of a paid-for reply was deleted"
    );
    assert_eq!(manifest["members"][1]["status"], "delivered");
}

#[cfg(unix)]
#[test]
fn a_crashed_runs_leftovers_are_never_credited_to_the_new_run() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("00-a_one.md.partial"), "stale draft").unwrap();
    std::fs::write(run.join("00-a_one.transcript.json"), "stale transcript").unwrap();
    // Read-only from the first request on, so none of this run's writes land.
    let locked = run.clone();
    let (port, _) = serve(mock(move |_| {
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o500)).ok();
        reply(&prose())
    }));
    let out = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter", "-m", "a/one"])
        .args(["--progress", "false", "--trace", "--out"])
        .arg(&run)
        .bounded_output();
    std::fs::set_permissions(&run, std::fs::Permissions::from_mode(0o700)).unwrap();
    if run.join("manifest.json").exists() {
        eprintln!("skipped: this process writes into a 0o500 directory (root?)");
        return;
    }
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).expect("manifest on stdout");
    let member = &m["members"][0];
    assert!(member.get("file").is_none(), "{member}");
    assert!(member.get("transcript").is_none(), "{member}");
    assert_eq!(
        std::fs::read_to_string(run.join("00-a_one.md.partial")).unwrap(),
        "stale draft",
        "and the leftover is left alone"
    );
}

/// Counts re-sends, not failures, which are one more.
#[test]
fn an_exhausted_member_reports_the_retries_it_actually_made() {
    let (port, seen) = serve(mock(|_| {
        (
            500,
            r#"{"error":{"message":"upstream is down","metadata":{"error_type":"server"}}}"#.into(),
        )
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    for m in r.manifest()["members"].as_array().unwrap() {
        assert_eq!(m["status"], "rejected");
        assert_eq!(m["retries"], 3, "MAX_RETRIES is 3");
    }
    let chats = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|b| b.contains("\"messages\""))
        .count();
    assert_eq!(chats, 8, "one initial request plus MAX_RETRIES, twice");
}

/// Answers the free probes, then cuts every inference response off mid-body: 50,000 bytes
/// promised, 20 sent. Returns the port and the count of inference requests.
fn cut_off_server() -> (u16, Arc<Mutex<usize>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(Mutex::new(0usize));
    let counter = Arc::clone(&requests);
    thread::spawn(move || {
        for sock in listener.incoming().flatten() {
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                let mut sock = sock;
                let Some((path, _)) = read_request(&mut sock) else {
                    return;
                };
                if is_free_probe(&path) {
                    let body = r#"{"data":{"limit":10,"limit_remaining":9}}"#;
                    let _ = sock.write_all(
                        format!(
                            "HTTP/1.1 200 X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    );
                    return;
                }
                *counter.lock().unwrap() += 1;
                let _ = sock.write_all(
                    b"HTTP/1.1 200 X\r\nContent-Type: application/json\r\n\
                      Content-Length: 50000\r\nConnection: close\r\n\r\n{\"choices\":[{\"messag",
                );
            });
        }
    });
    (port, requests)
}

/// The status line arrived, so the answer was generated and billed; the mid-body `Io` error
/// must not be retried.
#[test]
fn a_response_cut_off_mid_body_is_not_paid_for_twice() {
    let (port, requests) = cut_off_server();
    let dir = TempDir::new().unwrap();
    let out = council_in(port, dir.path(), Some(&dir.path().join("run")), &[]);

    let manifest: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    for m in manifest["members"].as_array().unwrap() {
        assert_eq!(m["status"], "rejected");
        assert_eq!(
            m["retries"], 0,
            "a delivered generation must not be re-sent"
        );
    }
    assert_eq!(
        *requests.lock().unwrap(),
        2,
        "one request per member, never a retry"
    );
}

/// With no usage reported, the worst case is charged, so the second member finds the cap spent.
#[test]
fn a_200_cut_off_mid_body_is_charged_its_worst_case_against_max_usd() {
    let (port, requests) = cut_off_server();
    let dir = TempDir::new().unwrap();
    let cfg = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg).unwrap();
    std::fs::write(
        cfg.join("council.json"),
        r#"{"rates":{"anthropic/claude-opus-5.5":[5,25],"b/two":[5,25]}}"#,
    )
    .unwrap();
    let out = bx("anthropic", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "anthropic",
            "--progress",
            "false",
        ])
        .args([
            "-m",
            "anthropic/claude-opus-5.5,b/two",
            "--concurrency",
            "1",
        ])
        .args(["--max-usd", "0.5", "--out"])
        .arg(dir.path().join("run"))
        .bounded_output();
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(m["members"][1]["terminated_by"], "max_usd", "{m}");
    assert_eq!(*requests.lock().unwrap(), 1, "{m}");
}

/// Output is free here, so only the prompt's charge can spend the cap.
#[test]
fn a_200_cut_off_mid_body_is_charged_its_input_too() {
    let (port, requests) = cut_off_server();
    let r = council_rated(
        port,
        r#"{"rates":{"anthropic/claude-opus-5.5":[1000,0],"b/two":[1000,0]}}"#,
        &["--concurrency", "1", "--max-usd", "0.5"],
    );
    let m = r.manifest();
    assert_eq!(m["members"][1]["terminated_by"], "max_usd", "{m}");
    assert_eq!(*requests.lock().unwrap(), 1, "{m}");
}

/// URLs are attacker-controlled and reach the calling agent; only `http(s)` survives.
#[test]
fn retrieved_source_urls_are_redacted_sanitised_and_scheme_checked() {
    let handler: Handler = Arc::new(|path: &str, body: &str| match path {
        p if is_free_probe(p) => (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into()),
        p if p.contains("/llm/context") => (
            200,
            serde_json::json!({"grounding": {"generic": [
                {"url": "https://x.test/?k=brave-mock-token", "title": "T",
                 "snippets": ["evidence one"]},
                {"url": "https://y.test/\u{1b}[2Jcleared", "title": "T2",
                 "snippets": ["evidence two"]},
                {"url": "data:text/html,payload", "title": "T3",
                 "snippets": ["evidence three"]}]}})
            .to_string(),
        ),
        _ if answered(body) => reply(&prose()),
        _ => tool_call("q"),
    });
    let (port, _) = serve(handler);
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);

    let stdout = String::from_utf8_lossy(&r.out.stdout).into_owned();
    assert!(
        !stdout.contains("brave-mock-token"),
        "the Brave key reached stdout inside a source URL:\n{stdout}"
    );
    // Stdout is checked too, in case the URLs are ever inlined again.
    let sources = r.sources(0);
    assert_eq!(
        sources.len(),
        2,
        "the data: URL must be dropped: {sources:?}"
    );
    assert!(
        sources
            .iter()
            .all(|u| u.starts_with("https://") && !u.contains('\u{1b}')),
        "{sources:?}"
    );
    assert!(
        sources[0].contains("***"),
        "the key must be redacted, not merely absent: {sources:?}"
    );
}

#[test]
fn an_upper_case_https_source_is_kept_and_a_data_url_is_not() {
    let handler: Handler = Arc::new(|path: &str, body: &str| match path {
        p if is_free_probe(p) => (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into()),
        p if p.contains("/llm/context") => (
            200,
            serde_json::json!({"grounding": {"generic": [
                {"url": "HTTPS://x.test/a", "title": "T", "snippets": ["evidence one"]},
                {"url": "DATA:text/html,payload", "title": "T2", "snippets": ["evidence two"]}]}})
            .to_string(),
        ),
        _ if answered(body) => reply(&prose()),
        _ => tool_call("q"),
    });
    let (port, _) = serve(handler);
    let r = council(port, &["-m", "a/one", "--progress", "false"]);
    assert_eq!(r.sources(0), ["HTTPS://x.test/a"]);
}

/// 100 in and 50 out at $5/$25 per MTok is $0.00175.
#[test]
fn a_direct_member_is_priced_by_arithmetic_and_the_run_total_is_the_sum() {
    let (port, _) = serve(mock(|_| anthropic_reply(&prose())));
    let r = council_rated(
        port,
        r#"{"rates":{"anthropic/claude-opus-5.5":[5,25],"b/two":[1,2]}}"#,
        &[],
    );
    assert_eq!(r.code(), 0, "{}", r.stderr());
    let m = r.manifest();
    assert_eq!(
        m["members"][0]["cost"],
        serde_json::json!({"llm_usd":0.00175,"source":"computed"})
    );
    assert_eq!(
        m["members"][1]["cost"],
        serde_json::json!({"llm_usd":0.0002,"source":"computed"})
    );
    assert_eq!(m["cost_source"], "computed");
    // No search owes Brave a known zero; uncapped, `max_usd` is present and null.
    assert_eq!(
        m["cost"],
        serde_json::json!({"llm_usd":0.00195,"brave_usd":0.0,"cost_unknown":false,
                           "max_usd":serde_json::Value::Null})
    );
    // The normalised total: Anthropic's `input_tokens` is a remainder.
    assert_eq!(
        m["members"][0]["tokens"],
        serde_json::json!({"prompt":100,"completion":50})
    );
}

#[test]
fn the_dry_run_priced_flag_and_the_manifest_cost_source_agree_for_a_direct_member() {
    let (port, _) = serve(mock(|_| anthropic_reply(&prose())));
    let plan: serde_json::Value =
        serde_json::from_slice(&council_rated(port, "{}", &["--dry-run"]).out.stdout).unwrap();
    let m = council_rated(port, "{}", &[]).manifest();
    for (i, priced, source) in [(0, true, "computed"), (1, false, "unknown")] {
        assert_eq!(plan["members"][i]["priced"], priced, "{plan}");
        assert_eq!(m["members"][i]["cost"]["source"], source, "{m}");
    }
}

/// The biller reports usage, so completion is bounded at `PARALLEL_LEGS` times the request's cap.
#[test]
fn a_provider_cannot_bill_for_more_output_than_its_legs_could_have_produced() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "model": "test/model", "provider": "Mock",
                "choices": [{"finish_reason": "stop", "message": {"content": prose()}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 9_000_000}
            })
            .to_string(),
        )
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-tokens",
            "1000",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(
        // 1024: `min_out` raises the request to the wire's floor.
        m["members"][0]["tokens"]["completion"],
        1024 * 8,
        "billed for output no number of legs could have produced: {}",
        m["members"][0]["tokens"]
    );
    assert_eq!(
        m["tokens"]["completion"],
        2 * 1024 * 8,
        "the run total inherits the bound from both members"
    );
    // `peak_round` must read the bounded copy.
    assert_eq!(
        m["members"][0]["limits"]["max_tokens"]["peak_round"],
        1024 * 8,
        "peak_round escaped the bound: {}",
        m["members"][0]["limits"]
    );
}

/// Round 1 is the only request at `max_completion_tokens: 64000`, since escalation re-sends the
/// same conversation, and it truncates. Whatever replaces that draft publishes without its label.
#[test]
fn a_truncated_draft_neither_labels_nor_rejects_the_text_that_replaced_it() {
    // (round 2, the nudge's reply, what publishes, whether that is a delivered review)
    let rows = [
        // Shorter, so the draft wins round 2; the nudge's real answer replaces it.
        ("y".repeat(100), prose(), "grounded prose", true),
        // Equal length: round 2 wins on `this >= draft`, short enough for step 6 to re-prompt,
        // and the empty nudge adds nothing.
        ("B".repeat(300), String::new(), "BBB", false),
    ];
    for (second, nudge, published, delivered) in rows {
        let (port, _) = serve(mock(move |body: &str| {
            if body.contains("Provide your final answer now") {
                reply(&nudge)
            } else if body.contains(r#""max_completion_tokens":64000"#) {
                (
                    200,
                    serde_json::json!({
                        "choices": [{"finish_reason": "length",
                                     "message": {"content": "x".repeat(300)}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                    })
                    .to_string(),
                )
            } else {
                reply(&second)
            }
        }));
        let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
        let m = r.manifest();
        let x = &m["members"][0];
        if delivered {
            assert_eq!(
                x["status"], "delivered",
                "a finished reply was rejected as truncated: {m}"
            );
        }
        assert_eq!(x["finish_reason"], "stop", "{published}: {m}");
        assert_ne!(x["reason"], "truncated", "{published}: {m}");
        let body = r.read("00-a_one.md");
        assert!(
            body.contains(published),
            "{published}: the replacement was not published"
        );
        assert!(
            !body.contains("xxx"),
            "{published}: the truncated round was published"
        );
    }
}

#[test]
fn a_retried_turn_is_clamped_exactly_like_a_delivered_one() {
    let (port, _) = serve(mock(|body: &str| {
        if body.contains("tool_calls") || body.contains(r#""role":"tool""#) {
            reply(&prose())
        } else {
            // No choice at all: retryable, and carrying an impossible completion count.
            (
                200,
                serde_json::json!({
                    "choices": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 9_000_000}
                })
                .to_string(),
            )
        }
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-tokens",
            "2000",
            "--progress",
            "false",
        ],
    );
    let m = &r.manifest()["members"][0];
    let completion = m["tokens"]["completion"].as_u64().unwrap();
    let limit = m["limits"]["max_tokens"]["limit"].as_u64().unwrap();
    // Retries are summed, so the bound is a multiple of the per-turn one.
    assert!(
        completion <= limit * 8 * 4,
        "an unbounded retry reached the total: {m}"
    );
    assert!(
        completion < 9_000_000,
        "the provider's raw claim was billed: {m}"
    );
    // The peak is per round, never the running total.
    assert!(
        m["limits"]["max_tokens"]["peak_round"].as_u64().unwrap() <= limit * 8,
        "peak_round published the provider's raw claim: {m}"
    );
}

/// A rate near `f64::MAX` overflows to infinity, which `serde_json` renders as `null`.
#[test]
fn a_price_that_overflows_is_unknown_not_a_null_claiming_to_be_known() {
    let (port, _) = serve(mock(|_| anthropic_reply(&prose())));
    let r = council_rated(
        port,
        r#"{"rates":{"anthropic/claude-opus-5.5":[1e308,1e308],"b/two":[1,2]}}"#,
        &[],
    );
    assert_eq!(r.code(), 0, "{}", r.stderr());
    let m = r.manifest();
    assert_eq!(
        m["members"][0]["cost"],
        serde_json::json!({"llm_usd":null,"source":"unknown"})
    );
    assert_eq!(m["cost"]["cost_unknown"], true);
    assert_eq!(m["cost_source"], "mixed", "one priced, one not");
}

#[test]
fn a_configured_spending_cap_is_honoured_without_the_flag() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "content": [{"type": "text", "text": "short"}],
                "usage": {"input_tokens": 4_000_000, "output_tokens": 200_000}
            })
            .to_string(),
        )
    }));
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    // Both rated: an unrated second member makes the exit code a race.
    std::fs::write(
        cfg_dir.join("council.json"),
        r#"{"max_usd": 0.01, "rates":{"anthropic/claude-opus-5.5":[5,25],"b/two":[5,25]}}"#,
    )
    .unwrap();

    let out = bx("anthropic", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "anthropic",
            "-m",
            "anthropic/claude-opus-5.5,b/two",
            "--progress",
            "false",
            "--out",
        ])
        .arg(dir.path().join("run"))
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(7),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(m["members"][0]["reason"], "max_usd", "{}", m["members"][0]);
    assert_eq!(m["cost"]["max_usd"], 0.01, "{}", m["cost"]);
}

/// A later short turn must not promote a draft cut off at the cap to `delivered`.
#[test]
fn a_truncated_draft_is_never_published_as_a_finished_answer() {
    // Counted per member: the escalated re-send is byte-identical, and members run in parallel.
    let rounds = Arc::new(Mutex::new(std::collections::HashMap::<String, usize>::new()));
    let (port, _) = serve(mock(move |body: &str| {
        let member = if body.contains(r#""model":"a/one""#) {
            "a"
        } else {
            "b"
        };
        let mut seen = rounds.lock().unwrap();
        let n = seen.entry(member.to_string()).or_default();
        *n += 1;
        if *n == 1 {
            (
                200,
                serde_json::json!({
                    "choices": [{"finish_reason": "length",
                                 "message": {"content": "T".repeat(1000)}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1000}
                })
                .to_string(),
            )
        } else {
            reply("Already covered above.")
        }
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--max-tokens",
            "2000",
        ],
    );
    for m in r.manifest()["members"].as_array().unwrap() {
        assert_eq!(
            m["status"], "rejected",
            "a truncated draft was delivered as an answer: {m}"
        );
        assert_eq!(m["reason"], "truncated", "{m}");
        // Not "retry": escalation already exhausted the ceiling, so a rerun truncates again.
        assert_eq!(m["action"], "none", "{m}");
        let why = m["error"].as_str().unwrap();
        assert!(
            why.contains("lower --effort or raise --max-tokens"),
            "an action of \"none\" has to say what to do instead: {m}"
        );
        assert!(
            !why.contains("--max-searches"),
            "the output ceiling was blamed on the search budget: {m}"
        );
    }
    // Rejected, but the billed text stays on disk.
    assert!(r.read("00-a_one.md").contains("TTT"));
}

/// 4, not the mixed-failure 6: the status must survive retry exhaustion.
#[test]
fn exhausting_retries_against_a_uniform_rate_limit_exits_4() {
    let (port, _) = serve(mock(|_| {
        (
            429,
            r#"{"error":{"message":"slow down","metadata":{"error_type":"rate_limit_exceeded"}}}"#
                .into(),
        )
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    assert_eq!(r.code(), 4, "stderr:\n{}", r.stderr());
    assert_eq!(r.manifest()["exit"], 4);
}

#[test]
fn exhausted_retries_on_a_200_error_body_still_advise_retry() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            r#"{"choices":[{"finish_reason":"error","message":{"content":"x"}}],"usage":{}}"#
                .into(),
        )
    }));
    let r = council(port, &["-m", "a/one", "--progress", "false"]);
    let one = &r.manifest()["members"][0];
    assert_eq!(one["reason"], "provider_error", "{one}");
    assert_eq!(one["action"], "retry", "{one}");
}

/// Counted: a retry re-sends a byte-identical body.
#[test]
fn usage_from_a_retried_turn_is_still_reported() {
    let seen = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&seen);
    let handler: Handler = Arc::new(move |path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into());
        }
        if path.contains("/llm/context") {
            return search_hit();
        }
        // Each member's first request errors retryably, with real usage.
        if counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
            return (
                200,
                serde_json::json!({
                    "choices": [{"error": {"message": "upstream hiccup",
                                           "metadata": {"error_type": "server"}}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 2000, "cost": 0.5,
                              "is_byok": false, "cost_details": {"upstream_inference_cost": 0.5}}
                })
                .to_string(),
            );
        }
        reply(&prose())
    });
    let (port, _) = serve(handler);
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = &r.manifest()["members"][0];
    assert_eq!(m["status"], "delivered", "{m}");
    assert_eq!(m["retries"], 1, "one retry happened: {m}");
    assert!(
        m["tokens"]["completion"].as_u64().unwrap() >= 2000,
        "the retried generation's tokens were dropped: {m}"
    );
    assert!(
        m["cost"]["llm_usd"].as_f64().unwrap() >= 0.5,
        "the retried generation's cost was dropped: {m}"
    );
    assert_eq!(
        m["limits"]["max_tokens"]["peak_round"], 2000,
        "the retried generation is missing from peak_round: {m}"
    );
}

/// The run id is the OpenRouter `session_id`, so runs under one `--out` name must not share it.
#[test]
fn the_run_id_is_minted_even_when_the_directory_is_named() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let id = r.manifest()["run"].as_str().unwrap().to_string();
    assert_ne!(id, "run", "the id was taken from the --out directory name");
    // `<YYYYmmdd-HHMMSS>-<6 hex>`.
    assert_eq!(id.len(), 22, "{id}");
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .any(|b| b.contains(&format!("\"session_id\":\"{id}\""))),
        "the session id sent to the provider is not the run id"
    );
    assert!(r.manifest()["dir"].as_str().unwrap().ends_with("run"));
}

/// Counted as a failed search, but never sent or billed; the complete reply still publishes,
/// with `partial_grounding`.
#[test]
fn a_malformed_search_call_is_a_failed_search_that_is_never_billed() {
    let (port, seen) = serve(mock(|body| {
        if answered(body) {
            reply(&prose())
        } else {
            (
                200,
                serde_json::json!({
                    "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
                        {"id": "c1", "function": {"name": "search",
                         "arguments": "{\"prompt\":\"wrong key\"}"}}]}}],
                    "usage": {}
                })
                .to_string(),
            )
        }
    }));
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(
        cfg_dir.join("council.json"),
        r#"{"brave_price_per_call_usd": 0.005}"#,
    )
    .unwrap();
    let out = council_in(port, dir.path(), Some(&dir.path().join("run")), &[]);
    let m: serde_json::Value = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stderr)));
    let first = &m["members"][0];
    assert_eq!(
        first["brave"]["calls"], 1,
        "the attempt was invisible: {first}"
    );
    assert_eq!(first["brave"]["errors"], 1, "{first}");
    assert_eq!(
        first["status"], "delivered",
        "a complete, billed review was discarded over a mis-shaped tool call: {first}"
    );
    assert_eq!(
        first["partial_grounding"], true,
        "the thin evidence must still be reported: {first}"
    );
    assert_eq!(m["brave"]["calls"], 2, "the attempt is still visible");
    assert_eq!(m["brave"]["sent"], 0, "nothing reached Brave");
    assert_eq!(m["brave"]["errors"], 2);
    assert_eq!(
        m["cost"]["brave_usd"], 0.0,
        "billed for requests Brave never saw"
    );
    assert!(
        seen.lock().unwrap().iter().all(|b| !b.contains("\"q\":")),
        "no search body should have been sent"
    );
}

#[test]
fn sources_are_deduplicated_and_capped_across_the_member() {
    let handler: Handler = Arc::new(|path: &str, body: &str| match path {
        p if is_free_probe(p) => (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into()),
        // 100 distinct URLs per search, and every search returns the same ones.
        p if p.contains("/llm/context") => {
            let hits: Vec<_> = (0..100)
                .map(|i| {
                    serde_json::json!({"url": format!("https://x.test/{i}"),
                                       "title": "T", "snippets": ["evidence"]})
                })
                .collect();
            (
                200,
                serde_json::json!({"grounding": {"generic": hits}}).to_string(),
            )
        }
        _ if body.matches("\"role\":\"tool\"").count() >= 3 => reply(&prose()),
        _ => tool_call("q"),
    });
    let (port, _) = serve(handler);
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let sources = r.sources(0);
    let m = &r.manifest()["members"][0];
    assert_eq!(m["brave"]["calls"], 3, "{m}");
    assert!(sources.len() <= 64, "{} sources", sources.len());
    // The published count is the list's length, not a second tally.
    assert_eq!(m["sources_count"], sources.len(), "{m}");
    let unique: std::collections::HashSet<_> = sources.iter().collect();
    assert_eq!(unique.len(), sources.len(), "duplicates: {sources:?}");
}

// ── Every cap still writes the work it paid for ──────────────────────

#[test]
fn every_cap_still_writes_the_reply_it_already_paid_for() {
    let marker = "PAID-FOR TEXT ";
    let draft = marker.to_string() + &"x".repeat(600);

    // 1. The token ceiling: `length` forever escalates to the cap.
    let d = draft.clone();
    let (port, _) = serve(mock(move |_| {
        (
            200,
            serde_json::json!({
                "choices": [{"finish_reason": "length", "message": {"content": d}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })
            .to_string(),
        )
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-tokens",
            "1000",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(m["members"][0]["reason"], "truncated");
    assert_eq!(m["members"][0]["terminated_by"], "truncated_at_cap");
    assert!(
        r.read("00-a_one.md").contains(marker),
        "the token cap discarded a paid-for draft"
    );

    // 2. The step cap: every turn asks for a tool with a draft too short to publish.
    let d = marker.to_string() + &"x".repeat(100);
    let (port, _) = serve(mock(move |body: &str| {
        if body.contains("tool_call_id") {
            (
                200,
                serde_json::json!({
                    "choices": [{"finish_reason": "tool_calls", "message": {
                        "content": "short", "tool_calls": [{"id": "c2", "function":
                            {"name": "search", "arguments": "{\"query\":\"q\"}"}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                })
                .to_string(),
            )
        } else {
            (
                200,
                serde_json::json!({
                    "choices": [{"finish_reason": "tool_calls", "message": {
                        "content": d, "tool_calls": [{"id": "c1", "function":
                            {"name": "search", "arguments": "{\"query\":\"q\"}"}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                })
                .to_string(),
            )
        }
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-steps",
            "3",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(m["members"][0]["terminated_by"], "max_steps");
    assert!(
        r.read("00-a_one.md").contains(marker),
        "the step cap discarded a paid-for draft"
    );

    // 3. The retry cap.
    let d = draft.clone();
    let n = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (port, _) = serve(mock(move |_| {
        let i = n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if i < 2 {
            (
                200,
                serde_json::json!({
                    "choices": [{"finish_reason": "tool_calls", "message": {
                        "content": d, "tool_calls": [{"id": "c1", "function":
                            {"name": "search", "arguments": "{\"query\":\"q\"}"}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                })
                .to_string(),
            )
        } else {
            (503, r#"{"error":{"message":"gone"}}"#.into())
        }
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    assert_eq!(m["members"][0]["reason"], "provider_error");
    assert!(
        r.read("00-a_one.md").contains(marker),
        "the retry cap discarded a paid-for draft"
    );

    // 4. The clock: the round after the draft sleeps past `--max-seconds`.
    let d = draft.clone();
    let (port, _) = serve(mock(move |body: &str| {
        if body.contains(r#""role":"tool""#) {
            std::thread::sleep(std::time::Duration::from_secs(30));
            reply("too late")
        } else {
            (
                200,
                serde_json::json!({
                    "choices": [{"finish_reason": "tool_calls", "message": {
                        "content": d, "tool_calls": [{"id": "c1", "function":
                            {"name": "search", "arguments": "{\"query\":\"q\"}"}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}
                })
                .to_string(),
            )
        }
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-seconds",
            "4",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(
        m["members"][0]["terminated_by"], "budget",
        "expected a clock stop: {}",
        m["members"][0]
    );
    assert!(
        r.read("00-a_one.md").contains(marker),
        "the budget discarded a paid-for draft"
    );
}

#[test]
fn the_last_step_publishes_a_finished_review_rather_than_running_its_trailing_tool_call() {
    let searches = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = Arc::clone(&searches);
    let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":{}}"#.to_string());
        }
        if path.contains("/llm/context") {
            counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return search_hit();
        }
        (
            200,
            serde_json::json!({
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "content": prose(), "tool_calls": [{"id": "c1", "function":
                        {"name": "search", "arguments": "{\"query\":\"q\"}"}}]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })
            .to_string(),
        )
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-steps",
            "1",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(
        m["members"][0]["status"], "delivered",
        "{}",
        m["members"][0]
    );
    assert_eq!(m["members"][0]["terminated_by"], "answer");
    assert_eq!(
        searches.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "a search was bought on the last step"
    );
}

#[test]
fn a_member_the_clock_stopped_keeps_its_draft_and_is_reported_as_retryable() {
    let marker = "PAID-FOR TEXT ";
    // Under the floor: a publishable draft would be delivered instead.
    let draft = marker.to_string() + &"x".repeat(100);
    let (port, _) = serve(Arc::new(move |path: &str, body: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":{}}"#.to_string());
        }
        if path.contains("/llm/context") {
            // Outlive the budget, so the member is cut off after its first paid round.
            std::thread::sleep(std::time::Duration::from_secs(6));
            return search_hit();
        }
        let _ = body;
        (
            200,
            serde_json::json!({
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "content": draft, "tool_calls": [{"id": "c1", "function":
                        {"name": "search", "arguments": "{\"query\":\"q\"}"}}]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })
            .to_string(),
        )
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-seconds",
            "2",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(m["members"][0]["reason"], "budget", "{}", r.stderr());
    assert_eq!(
        m["members"][0]["action"], "retry",
        "a member that ran out of clock is not a member to give up on"
    );
    assert_eq!(m["members"][0]["terminated_by"], "budget");
    assert!(
        m["hint"]
            .as_str()
            .unwrap()
            .contains("Retry the rest with: -m"),
        "the hint must name the retry: {}",
        m["hint"]
    );
    assert!(
        r.read("00-a_one.md").contains(marker),
        "the budget discarded a paid-for draft"
    );
}

/// `terminated_by` still names the bound.
#[test]
fn a_finished_review_in_hand_is_delivered_when_a_bound_of_ours_stops_the_member() {
    // Every round writes the review and asks for one more search; searches sleep when asked to.
    let run = |slow_search: bool, args: &[&str]| {
        let (port, _) = serve(Arc::new(move |path: &str, body: &str| {
            if is_free_probe(path) {
                return (200, r#"{"data":{}}"#.to_string());
            }
            if path.contains("/llm/context") {
                if slow_search {
                    std::thread::sleep(std::time::Duration::from_secs(6));
                }
                return search_hit();
            }
            let content = if answered(body) { "" } else { &prose() };
            (
                200,
                serde_json::json!({
                    "choices": [{"finish_reason": "tool_calls", "message": {
                        "content": content, "tool_calls": [{"id": "c1", "function":
                            {"name": "search", "arguments": "{\"query\":\"q\"}"}}]}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01}
                })
                .to_string(),
            )
        }));
        let mut all = vec!["-m", "a/one", "--progress", "false"];
        all.extend_from_slice(args);
        council(port, &all)
    };
    for (slow, args, bound) in [
        (true, &["--max-seconds", "2"][..], "budget"),
        (false, &["--max-usd", "0.005"][..], "max_usd"),
        (false, &["--max-steps", "2"][..], "max_steps"),
    ] {
        let r = run(slow, args);
        let m = &r.manifest()["members"][0];
        assert_eq!(m["status"], "delivered", "{bound}: {m}");
        assert_eq!(m["terminated_by"], bound, "{m}");
        assert_eq!(r.code(), 0, "{bound}: {}", r.stderr());
        assert!(r.read("00-a_one.md").contains("## A. ANSWER"), "{bound}");
    }
}

#[test]
fn a_hanging_search_cannot_outrun_the_members_max_seconds() {
    let (port, _) = serve(Arc::new(|path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":{}}"#.to_string());
        }
        if path.contains("/llm/context") {
            std::thread::sleep(std::time::Duration::from_secs(30));
            return search_hit();
        }
        tool_call("q")
    }));
    let started = std::time::Instant::now();
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-seconds",
            "2",
            "--progress",
            "false",
        ],
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(15),
        "a search outlived the budget by {:?}",
        started.elapsed()
    );
    assert_eq!(r.manifest()["members"][0]["reason"], "budget");
}

#[test]
fn a_member_that_died_on_a_transport_error_never_reports_finish_reason_stop() {
    let (port, _) = serve(mock(|_| (401, r#"{"error":{"message":"no auth"}}"#.into())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    assert_eq!(r.code(), 3, "{}", r.stderr());
    assert_eq!(m["members"][0]["terminated_by"], "error");
    assert_eq!(
        m["members"][0]["finish_reason"], "error",
        "a member that died on a 401 did not stop"
    );
}

/// `unbilled`, not the `unknown` a counted round with no usage publishes.
#[test]
fn a_member_that_never_reached_its_provider_counts_no_round_and_is_unbilled() {
    // Port 1, not a freed ephemeral one: a parallel test's mock can take that.
    let r = council(
        1,
        &[
            "-m",
            "a/one,b/two",
            "--max-seconds",
            "3",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    for x in m["members"].as_array().unwrap() {
        assert_eq!(x["status"], "rejected", "{x}");
        assert_eq!(x["rounds"], 0, "{x}");
        assert_eq!(x["cost"]["source"], "unbilled", "{x}");
    }
}

/// `sanitize` and `redact` shrink the reply, so NUL padding must not clear the floor.
#[test]
fn the_gate_judges_the_text_that_will_be_published_not_the_raw_reply() {
    let padded = "\u{0}".repeat(500) + "x";
    let (port, _) = serve(mock(move |_| reply(&padded)));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    assert_eq!(m["members"][0]["status"], "rejected", "{}", r.stderr());
    assert_eq!(m["members"][0]["reason"], "no_answer");
    assert_eq!(m["members"][0]["chars"], 1, "one character is the reply");
}

#[cfg(unix)]
#[test]
fn the_manifest_names_a_transcript_only_when_one_exists() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::create_dir(run.join("00-a_one.transcript.json")).unwrap();

    let out = council_in(port, dir.path(), Some(&run), &["--trace"]);
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        m["members"][0].get("transcript").is_none(),
        "a transcript that was never written must not be named"
    );
    assert_eq!(m["members"][1]["transcript"], "01-b_two.transcript.json");
    assert_eq!(
        m["log_write_errors"], 1,
        "the failed write went uncounted: {m}"
    );
}

/// By the bare model id, as the bound and the bill do.
#[test]
fn doctor_prices_a_prefixed_member_exactly_as_the_bound_does() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(
        cfg_dir.join("council.json"),
        r#"{"panels":{"default":["direct:anthropic/claude-opus-5.5","b/two"]}}"#,
    )
    .unwrap();
    // `doctor` reaches no network. Not port 0: every command validates the Brave base URL.
    let out = bx("anthropic", 1, dir.path())
        .args(["council", "doctor", "--format", "text"])
        .bounded_output();
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .find(|l| {
            l.trim_start()
                .starts_with("direct:anthropic/claude-opus-5.5")
        })
        .unwrap_or_else(|| panic!("no line for the prefixed member:\n{text}"));
    assert!(
        line.contains("per MTok"),
        "a rated member reported as unpriceable for its spelling: {line}"
    );
}

/// A mistyped multiplier bills every cache read wrong, so it must be visible.
#[test]
fn doctor_reports_the_cache_read_multiplier_a_rate_carries() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(
        cfg_dir.join("council.json"),
        r#"{"panels":{"default":["anthropic/claude-a","anthropic/claude-b"]},
            "rates":{"anthropic/claude-a":[4,20,0.05],"anthropic/claude-b":[4,20]}}"#,
    )
    .unwrap();
    let doctor = |format: &str| {
        bx_in(dir.path())
            .args(["council", "doctor", "--format", format])
            .bounded_output()
            .stdout
    };
    let v: serde_json::Value = serde_json::from_slice(&doctor("json")).unwrap();
    let multipliers: Vec<_> = v["default_panel"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["cache_read_multiplier"].as_f64())
        .collect();
    assert_eq!(multipliers, [Some(0.05), Some(0.1)], "{v}");
    let text = String::from_utf8(doctor("text")).unwrap();
    assert!(text.contains("cache reads x0.05"), "{text}");
}

#[test]
fn a_default_run_is_named_for_its_id_and_history_points_at_it() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let out = council_in(port, dir.path(), None, &[]);
    let m: serde_json::Value = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stderr)));
    let run = m["run"].as_str().unwrap();
    let path = std::path::Path::new(m["dir"].as_str().unwrap());
    assert_eq!(
        path.file_name().unwrap().to_str().unwrap(),
        run,
        "the directory is not named for the run"
    );
    assert!(path.is_dir(), "the reported directory does not exist");
    assert!(
        path.starts_with(bx_root(dir.path()).join("bx/council")),
        "{path:?}"
    );

    let log = std::fs::read_to_string(dir.path().join("bx/council/history.jsonl")).unwrap();
    let lines: Vec<serde_json::Value> = log
        .lines()
        .map(|l| serde_json::from_str(l).expect("history is one document per line"))
        .collect();
    assert_eq!(lines.len(), 2, "one line per member");
    for l in &lines {
        assert_eq!(l["run"], run);
        assert_eq!(l["dir"], m["dir"], "a record must point at its artifacts");
        assert!(std::path::Path::new(l["dir"].as_str().unwrap()).is_dir());
    }
}

/// One deadline for the whole call: a per-read timeout never fires on a drip.
#[test]
fn a_body_dripped_a_byte_at_a_time_cannot_outlive_its_max_seconds() {
    let port = drip_server(std::time::Duration::ZERO);

    let started = std::time::Instant::now();
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-seconds",
            "3",
            "--progress",
            "false",
        ],
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(20),
        "a dripping body outlived the budget by {:?}",
        started.elapsed()
    );
    assert_eq!(r.manifest()["members"].as_array().unwrap().len(), 2);
}

/// The call's deadline is armed before `send`, not when the head arrives.
#[test]
fn a_slow_head_does_not_buy_the_body_a_second_window() {
    // Stall the head until most of the 4s budget is gone, then drip forever.
    let port = drip_server(std::time::Duration::from_millis(3_500));

    let started = std::time::Instant::now();
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-seconds",
            "4",
            "--progress",
            "false",
        ],
    );
    let elapsed = started.elapsed();
    let m = r.manifest();
    // The member's clock, not the wall's: startup and pre-flight are not what this measures.
    // A second window costs ~8s against the 4s budget.
    for one in m["members"].as_array().unwrap() {
        assert!(
            one["elapsed_ms"].as_u64().unwrap() < 6_000,
            "a stalled head bought the body a second budget: {one}"
        );
    }
    assert!(
        elapsed < std::time::Duration::from_secs(20),
        "the run outlived every budget in it: {elapsed:?}"
    );
    assert_eq!(m["members"].as_array().unwrap().len(), 2);
}

/// Scrub, then redact: redacting first misses the key, and scrubbing then reassembles it.
#[test]
fn a_credential_broken_up_by_a_control_character_is_still_redacted() {
    let (port, _) = serve(mock(|_| {
        let key = "sk-or-v1-moc\u{202e}ktoken";
        reply(&format!(
            "## A. ANSWER\nyour key is {key} and it is fine. {}",
            "grounded prose with evidence. ".repeat(40)
        ))
    }));
    let r = council(port, &["-m", "a/one", "--trace", "--progress", "false"]);
    // The transcript carries the assistant's words byte for byte.
    let t = r.read("00-a_one.transcript.json");
    assert!(!t.is_empty(), "--trace writes a transcript");
    assert!(
        !t.contains("sk-or-v1-mocktoken"),
        "a key the scrubber reassembled reached the transcript: {t}"
    );
    assert!(
        !r.read("00-a_one.md").contains("sk-or-v1-mocktoken"),
        "and the same in the published reply"
    );
}

/// `--tools none` validates no Brave key, so a short leftover must not redact ordinary words.
#[test]
fn a_brave_key_too_short_to_send_is_not_a_redaction_needle() {
    let (port, _) = serve(mock(|_| {
        reply(&format!(
            "## A. ANSWER\nOne note on the design. {}",
            "grounded prose with evidence. ".repeat(40)
        ))
    }));
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    let out = bx("openrouter", port, dir.path())
        .env("BRAVE_SEARCH_API_KEY", "note")
        .args(["council", "SUBJECT", "--via", "openrouter"])
        .args(["-m", "a/one", "--tools", "none", "--progress", "false"])
        .arg("--out")
        .arg(&run)
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let reply = std::fs::read_to_string(run.join("00-a_one.md")).unwrap();
    assert!(
        reply.contains("One note on"),
        "an unsendable key redacted a word out of a paid review: {reply}"
    );
}

/// Both the reply and its partial failed to write.
#[test]
fn a_reply_that_reached_no_file_at_all_is_named_in_the_manifest_as_none() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    std::fs::create_dir_all(&run).unwrap();
    // Both names taken by directories: `write_private` renames onto them and cannot.
    std::fs::create_dir(run.join("00-a_one.md")).unwrap();
    std::fs::create_dir(run.join("00-a_one.md.partial")).unwrap();

    let out = council_in(port, dir.path(), Some(&run), &["-m", "a/one"]);
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let one = &m["members"][0];
    assert_eq!(one["reason"], "write_failed", "{m}");
    assert!(
        one["file"].is_null(),
        "the manifest named a file nobody wrote: {m}"
    );
}

/// Refused before sending, naming both numbers: clamping up would earn the wire's 400.
#[test]
fn a_model_whose_ceiling_is_under_the_wires_floor_costs_a_refusal_not_a_seat() {
    let (port, seen) = serve(Arc::new(|path: &str, _: &str| {
        if path.contains("/api/v1/models") {
            // The roster shape this wire reads: `top_provider.max_completion_tokens`.
            return (
                200,
                serde_json::json!({"data": [
                    {"id": "a/one", "top_provider": {"max_completion_tokens": 512}},
                    {"id": "b/two", "top_provider": {"max_completion_tokens": 64000}}
                ]})
                .to_string(),
            );
        }
        if path.contains("/api/v1/key") {
            return (200, r#"{"data":{}}"#.into());
        }
        (200, reply(&prose()).1)
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    assert_eq!(
        r.code(),
        2,
        "a request the wire will reject was sent anyway"
    );
    let err = r.stderr();
    assert!(err.contains("512") && err.contains("1024"), "{err}");
    assert!(
        !seen.lock().unwrap().iter().any(|b| b.contains("messages")),
        "nothing is sent when the plan is refused"
    );
}

/// Every trace line has the `ts`/`kind` envelope.
#[test]
fn the_trace_carries_one_record_shape_and_history_is_not_in_it() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(
        port,
        &["-m", "a/one,b/two", "--trace", "--progress", "false"],
    );
    let trace = r.read("trace.jsonl");
    assert!(!trace.is_empty(), "--trace must write a trace");
    for line in trace.lines() {
        let v: serde_json::Value = serde_json::from_str(line).expect("one document per line");
        assert!(
            v.get("kind").is_some(),
            "un-enveloped record in the trace: {line}"
        );
        assert_ne!(
            v["kind"], "history",
            "a history record reached the trace: {line}"
        );
        assert!(v.get("ts").is_some(), "{line}");
    }
    let hist = std::fs::read_to_string(r.dir.path().join("bx/council/history.jsonl"))
        .expect("history.jsonl");
    for line in hist.lines() {
        let v: serde_json::Value = serde_json::from_str(line).expect("one document per line");
        assert!(v.get("run").is_some() && v.get("model").is_some(), "{line}");
    }
}

#[test]
fn a_trace_query_carrying_an_escape_decodes_without_it() {
    let (port, _) = serve(mock(|body| {
        if answered(body) {
            reply(&prose())
        } else {
            tool_call("\\u001b[2J evil")
        }
    }));
    let r = council(port, &["-m", "a/one", "--trace", "--progress", "false"]);
    let queries: Vec<String> = r
        .read("trace.jsonl")
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v["kind"] == "search")
        .map(|v| v["body"]["query"].as_str().unwrap_or_default().to_string())
        .collect();
    assert!(queries.iter().any(|q| q.contains("evil")), "{queries:?}");
    assert!(queries.iter().all(|q| !q.contains('\x1b')), "{queries:?}");
}

/// `brave.sent` is the billable count, so a search that died dialling is not in it.
#[test]
fn a_search_that_never_reached_brave_is_not_counted_as_billable() {
    let (port, _) = serve(mock(|body: &str| {
        if body.contains(r#""role":"tool""#) {
            reply(&prose())
        } else {
            tool_call("anything")
        }
    }));

    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter", "--out"])
        .arg(dir.path().join("run"))
        .args(["-m", "a/one,b/two", "--progress", "false"])
        // Overrides the mock `bx` wired in.
        .env("BRAVE_SEARCH_BASE_URL", common::DEAD)
        .bounded_output();
    let m: serde_json::Value = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stderr)));
    let b = &m["members"][0]["brave"];
    assert!(
        b["calls"].as_u64().unwrap() > 0,
        "the attempt is invisible: {b}"
    );
    assert_eq!(
        b["errors"], b["calls"],
        "every search should have errored: {b}"
    );
    assert_eq!(
        b["sent"], 0,
        "billed for a search Brave never answered: {b}"
    );
    // Zero, not unknown.
    assert_eq!(m["cost"]["brave_usd"], 0.0, "{}", m["cost"]);
}

/// Every stored turn is re-sent each round, so unbounded the upload is the triangular sum.
#[test]
fn a_conversation_that_outgrows_its_budget_stops_instead_of_uploading_the_triangle() {
    let uploaded = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = uploaded.clone();
    // Just under `MAX_ECHO_BYTES`, with a tool call so the loop continues. Two fill the budget.
    let (port, _) = serve(mock(move |body: &str| {
        seen.fetch_add(body.len(), std::sync::atomic::Ordering::SeqCst);
        (
            200,
            serde_json::json!({
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "content": "x".repeat(4_000_000),
                    "tool_calls": [{"id": "c1", "function":
                        {"name": "search", "arguments": "{\"query\":\"q\"}"}}]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })
            .to_string(),
        )
    }));
    // The budget is steps x cap x 4 bytes, so a small cap reaches it quickly.
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-tokens",
            "1024",
            "--max-steps",
            "12",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(
        m["members"][0]["terminated_by"], "context_budget",
        "expected the context bound to stop it: {}",
        m["members"][0]
    );
    // Unbounded, this mock uploads about 544 MB per member.
    let total = uploaded.load(std::sync::atomic::Ordering::SeqCst);
    assert!(
        total < 80_000_000,
        "uploaded {total} bytes — the conversation was not bounded"
    );
}

/// Results are bounded in characters, so the byte budget must hold full-size CJK results.
#[test]
fn conversation_budget_holds_every_priced_search_of_four_byte_text_at_a_lowered_cap() {
    let (port, _) = serve(Arc::new(|path: &str, body: &str| {
        if path.contains("/api/v1/models") {
            return (
                200,
                r#"{"data":[{"id":"a/one","top_provider":{"max_completion_tokens":4096}}]}"#.into(),
            );
        }
        if is_free_probe(path) {
            return (200, r#"{"data":{}}"#.into());
        }
        if path.contains("/llm/context") {
            let page = serde_json::json!({"grounding": {"generic": [{"url": "https://example.test/a",
                "title": "A", "snippets": ["日".repeat(70_000)]}]}});
            return (200, page.to_string());
        }
        if body.contains("search budget exhausted") {
            reply(&prose())
        } else {
            tool_call("q")
        }
    }));
    let r = council(
        port,
        &["-m", "a/one", "--max-tokens", "4096", "--progress", "false"],
    );
    let one = &r.manifest()["members"][0];
    assert_eq!(one["terminated_by"], "answer", "{one}");
    assert_eq!(one["brave"]["grounded"], 15, "{one}");
}

#[test]
fn a_partial_panel_succeeds_and_still_names_what_it_lost() {
    let (port, _) = serve(mock(|body: &str| {
        if body.contains(r#""model":"a/one""#) {
            (500, r#"{"error":{"message":"upstream is down"}}"#.into())
        } else {
            reply(&prose())
        }
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    assert_eq!(
        r.code(),
        0,
        "a usable panel was written off: {}",
        r.stderr()
    );
    let m = r.manifest();
    assert_eq!(
        m["quorum"],
        serde_json::json!({"delivered":1,"total":2,"required":1,"ok":true})
    );
    assert_eq!(m["members"][0]["model"], "a/one");
    assert_eq!(m["members"][0]["status"], "rejected");
    assert_eq!(m["members"][0]["reason"], "provider_error");
    let hint = m["hint"].as_str().unwrap();
    assert!(hint.contains("1 of 2 delivered"), "{hint}");
    assert!(
        hint.contains("a/one"),
        "the lost judge is not named: {hint}"
    );
    assert!(r.read("index.md").contains("a/one"));
    assert!(r.read("01-b_two.md").contains("grounded prose"));
}

/// A stateless replay is a billed re-send, charged to the same attempt budget.
#[test]
fn a_lost_openai_chain_is_replayed_once_and_counted_as_a_retry() {
    // Zero data retention sends every OpenAI request stateless, leaving no chain to lose.
    let zdr_off = |home: &std::path::Path| {
        let cfg_dir = home.join("brave-search");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::write(cfg_dir.join("council.json"), r#"{"zdr": false}"#).unwrap();
    };
    let lost = || {
        (
            404,
            serde_json::json!({"error": {"code": "previous_response_not_found",
                                         "message": "not found"}})
            .to_string(),
        )
    };
    let replayed = |bodies: &[String]| bodies.iter().any(|b| b.contains("\"store\":false"));

    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = calls.clone();
    let (port, bodies) = serve(mock(move |_: &str| {
        if seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            lost()
        } else {
            (
                200,
                serde_json::json!({
                    "id": "resp_1", "status": "completed",
                    "output": [{"type": "message",
                                "content": [{"type": "output_text", "text": prose()}]}],
                    "usage": {"input_tokens": 10, "output_tokens": 5}
                })
                .to_string(),
            )
        }
    }));
    let dir = TempDir::new().unwrap();
    zdr_off(dir.path());
    let out = bx("openai", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "openai",
            "-m",
            "openai/a,openai/b",
        ])
        .args(["--progress", "false", "--out"])
        .arg(dir.path().join("run"))
        .bounded_output();
    let m: serde_json::Value = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&out.stderr)));
    assert!(
        replayed(&bodies.lock().unwrap()),
        "no stateless replay: {m}"
    );
    let member = m["members"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["retries"].as_u64() == Some(1))
        .unwrap_or_else(|| panic!("no member counted the replay: {m}"));
    assert_eq!(member["status"], "delivered", "{member}");

    // Against a chain that is always lost, each member sends exactly `MAX_RETRIES + 1` times.
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = calls.clone();
    let (port, bodies) = serve(mock(move |_: &str| {
        seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        lost()
    }));
    let dir = TempDir::new().unwrap();
    zdr_off(dir.path());
    bx("openai", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "openai",
            "-m",
            "openai/a,openai/b",
        ])
        .args(["--progress", "false", "--out"])
        .arg(dir.path().join("run"))
        .bounded_output();
    assert!(replayed(&bodies.lock().unwrap()), "no stateless replay");
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        2 * 4,
        "two members x (MAX_RETRIES + 1) sends, the replay included"
    );
}

/// The cap reads its own accumulator, which `drive` alone would charge once per round.
#[test]
fn a_retried_generation_is_charged_to_the_cap_not_only_to_the_manifest() {
    // A retryable 200 carrying usage, forever.
    let (port, _) = serve(mock(|_: &str| {
        (
            200,
            serde_json::json!({
                "content": [], "type": "error",
                "error": {"type": "overloaded_error", "message": "busy"},
                "usage": {"input_tokens": 2_000_000, "output_tokens": 100_000}
            })
            .to_string(),
        )
    }));
    let r = council_rated(
        port,
        r#"{"rates":{"anthropic/claude-opus-5.5":[5,25],"b/two":[5,25]}}"#,
        &["--max-usd", "0.01"],
    );
    let m = r.manifest();
    let members = m["members"].as_array().unwrap();
    assert!(
        members.iter().any(|x| x["reason"] == "max_usd"),
        "retried generations never reached the cap: {m}"
    );
    // Checked inside the retry loop: at 2M prompt tokens a request, three billed, not eight.
    let prompt = m["tokens"]["prompt"].as_u64().unwrap();
    assert!(
        prompt <= 8_000_000,
        "kept retrying past the cap: {prompt} prompt tokens billed"
    );
}

#[test]
fn brave_calls_are_charged_to_the_cap_and_an_unpriced_one_is_not_a_known_cost() {
    // Anthropic-shaped: `council_rated` routes `--via anthropic`.
    let ant_tool_then_reply = |body: &str| {
        if body.contains("tool_result") {
            anthropic_reply(&prose())
        } else {
            (
                200,
                serde_json::json!({
                    "model": "claude-opus-5-5", "stop_reason": "tool_use",
                    "content": [{"type": "tool_use", "id": "c1", "name": "search",
                                 "input": {"query": "q"}}],
                    "usage": {"input_tokens": 10, "output_tokens": 5}
                })
                .to_string(),
            )
        }
    };
    // The first search exhausts the cap.
    let (port, _) = serve(mock(ant_tool_then_reply));
    let r = council_rated(
        port,
        r#"{"brave_price_per_call_usd": 1000.0, "rates":{"anthropic/claude-opus-5.5":[5,25],"b/two":[5,25]}}"#,
        &["--max-usd", "0.01"],
    );
    let m = r.manifest();
    assert!(
        m["members"]
            .as_array()
            .unwrap()
            .iter()
            .any(|x| x["reason"] == "max_usd"),
        "Brave spend never reached the cap: {m}"
    );

    // Unconfigured: the default price per call.
    let (port, _) = serve(mock(ant_tool_then_reply));
    let r = council_rated(
        port,
        r#"{"rates":{"anthropic/claude-opus-5.5":[5,25],"b/two":[5,25]}}"#,
        &[],
    );
    let m = r.manifest();
    let sent: u64 = m["brave"]["sent"].as_u64().unwrap();
    assert!(sent > 0, "{m}");
    let billed = m["cost"]["brave_usd"].as_f64().unwrap();
    assert!(
        (billed - sent as f64 * 0.005).abs() < 1e-9,
        "{sent} calls billed as ${billed}: {m}"
    );
    assert_eq!(
        m["cost"]["cost_unknown"], false,
        "every member and every search was priced: {}",
        m["cost"]
    );

    let (port, seen) = serve(mock(ant_tool_then_reply));
    let r = council_rated(
        port,
        r#"{"brave_price_per_call_usd": -1.0, "rates":{"anthropic/claude-opus-5.5":[5,25],"b/two":[5,25]}}"#,
        &[],
    );
    assert_eq!(r.code(), 2, "{}", r.stderr());
    assert!(
        r.stderr().contains("brave_price_per_call_usd"),
        "{}",
        r.stderr()
    );
    assert!(seen.lock().unwrap().is_empty());
}

/// `@<(git diff)` is a pipe, so neither refusing pipes nor a non-blocking read fixes the hang.
#[cfg(unix)]
#[test]
fn a_subject_pipe_with_no_writer_ends_instead_of_hanging_and_a_slow_one_is_read_whole() {
    let dir = TempDir::new().unwrap();
    let (port, _) = serve(mock(|_| reply(&prose())));

    // 1. No writer, ever.
    let dead = dir.path().join("dead.fifo");
    assert!(
        std::process::Command::new("mkfifo")
            .arg(&dead)
            .status()
            .unwrap()
            .success()
    );
    let run_with = |subject: &str| {
        let mut child = bx("openrouter", port, dir.path())
            .args([
                "council",
                subject,
                "--via",
                "openrouter",
                "-m",
                "a/one,b/two",
            ])
            .args(["--progress", "false", "--out"])
            .arg(dir.path().join(format!("run-{}", subject.len())))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while child.try_wait().unwrap().is_none() {
            assert!(std::time::Instant::now() < deadline, "{subject} hung");
            thread::sleep(std::time::Duration::from_millis(20));
        }
        finish(child)
    };

    let started = std::time::Instant::now();
    let out = run_with(&format!("@{}", dead.display()));
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code(), Some(2), "stderr: {err}");
    assert!(err.contains("subject is empty"), "{err}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(20),
        "hung for {:?}",
        started.elapsed()
    );

    // 2. Process substitution: a writer attached before spawn but slow to produce. Through a
    //    shell, because `<(...)` is shell syntax.
    let run = dir.path().join("run-subst");
    let out = common::sandbox(&mut Command::new("bash"), dir.path())
        .arg("-c")
        .arg(format!(
            "exec {bx} council @<(sleep 1.2; printf 'SLOW-SUBJECT-CONTENT %.0s' $(seq 20)) \
             --via openrouter -m a/one,b/two --progress false --out {run}",
            bx = common::BX,
            run = run.display()
        ))
        .env(
            "BX_COUNCIL_OPENROUTER_BASE_URL",
            format!("http://127.0.0.1:{port}"),
        )
        .env("OPENROUTER_API_KEY", "sk-or-v1-mocktoken")
        .env("BRAVE_SEARCH_BASE_URL", format!("http://127.0.0.1:{port}"))
        .env("BRAVE_SEARCH_API_KEY", "brave-mock-token")
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        std::fs::read_to_string(run.join("brief.md"))
            .unwrap()
            .contains("SLOW-SUBJECT-CONTENT"),
        "a slow process substitution was cut short"
    );
}

/// `direct:x/y` and `x/y`: size and quorum are judged on the resolved panel.
#[test]
fn two_spellings_of_one_model_are_one_member() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    // Three spellings, two models. Two spellings alone is `a_pre_flight_refusal_sends_nothing…`.
    let r = council(
        port,
        &[
            "-m",
            "direct:anthropic/claude-opus-5.5,anthropic/claude-opus-5.5,b/two",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(m["members"].as_array().unwrap().len(), 2, "{m}");
    assert_eq!(m["quorum"]["total"], 2);
}

/// A debug build runs clap's `debug_assert!`s over the declaration.
#[test]
fn the_progress_flag_takes_a_value_or_stands_alone_and_never_panics_the_command() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    for args in [
        vec!["-m", "a/one,b/two", "--progress", "false"],
        vec!["-m", "a/one,b/two", "--progress", "true"],
        vec!["-m", "a/one,b/two", "--progress"],
        vec!["-m", "a/one,b/two"],
    ] {
        let r = council(port, &args);
        assert_eq!(r.code(), 0, "{args:?}: {}", r.stderr());
        assert!(
            !r.stderr().contains("panic"),
            "{args:?} panicked: {}",
            r.stderr()
        );
    }
}

#[test]
fn a_deadline_that_expires_mid_request_is_reported_as_budget_not_provider_error() {
    let (port, _) = serve(Arc::new(|path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":{}}"#.to_string());
        }
        thread::sleep(std::time::Duration::from_secs(20));
        reply("never arrives")
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-seconds",
            "2",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(m["members"][0]["reason"], "budget", "{}", r.stderr());
    assert_eq!(m["members"][0]["action"], "retry");
    assert_eq!(m["members"][0]["terminated_by"], "budget");
    assert_eq!(m["members"][0]["finish_reason"], "budget");
}

/// OpenRouter sends its 200 head before generating, so the clock usually fires mid-body.
#[test]
fn a_200_body_cut_off_by_our_own_deadline_ends_as_budget_not_provider_error() {
    // Round one either drips or writes a review beside a search; every later round drips.
    for held in [false, true] {
        let port = drip_server_unless(std::time::Duration::ZERO, move |path, body| {
            if path.contains("/llm/context") {
                return Some(search_hit());
            }
            (held && !answered(body)).then(|| {
                let content = prose();
                (
                    200,
                    serde_json::json!({
                        "choices": [{"finish_reason": "tool_calls", "message": {
                            "content": content, "tool_calls": [{"id": "c1", "function":
                                {"name": "search", "arguments": "{\"query\":\"q\"}"}}]}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01}
                    })
                    .to_string(),
                )
            })
        });
        let r = council(
            port,
            &["-m", "a/one", "--max-seconds", "3", "--progress", "false"],
        );
        let m = &r.manifest()["members"][0];
        assert_eq!(m["terminated_by"], "budget", "held={held}: {m}");
        if held {
            assert_eq!(m["status"], "delivered", "{m}");
            assert!(r.read("00-a_one.md").contains("## A. ANSWER"));
        } else {
            assert_eq!(m["reason"], "budget", "{m}");
            assert_eq!(m["action"], "retry", "{m}");
        }
    }
}

#[test]
fn set_key_honours_a_dry_run_and_never_asks_for_a_key_it_would_discard() {
    use std::io::Write as _;

    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    let path = cfg_dir.join("council.json");
    std::fs::write(&path, r#"{"openrouter_api_key":"sk-or-v1-keepme1"}"#).unwrap();

    let mut child = bx_in(dir.path())
        .args(["council", "set-key", "--via", "openrouter", "--dry-run"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Result ignored: a dry run exits before reading, so a broken pipe is the pass.
    let _ = child
        .stdin
        .take()
        .unwrap()
        .write_all(b"sk-or-v1-shouldnotbewritten\n");
    let out = finish(child);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{err}");
    assert!(
        !err.contains("paste key"),
        "a dry run asked for a key: {err}"
    );
    assert!(err.contains("would ask for openrouter"), "{err}");
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        r#"{"openrouter_api_key":"sk-or-v1-keepme1"}"#,
        "a dry run rewrote the config"
    );

    // No `--via` and no terminal: the walk cannot run, the dry run still answers.
    let out = bx_in(dir.path())
        .args(["council", "set-key", "--dry-run"])
        .stdin(Stdio::null())
        .bounded_output();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{err}");
    assert!(
        err.contains("openrouter, anthropic, openai"),
        "a dry run with no --via named no providers: {err}"
    );
}

/// The file's key is in the wrong slot, so reading it would refuse the run.
#[test]
fn an_environment_key_outranks_the_file_key() {
    let (port, _) = serve(mock(|_| anthropic_reply(&prose())));
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(
        cfg_dir.join("council.json"),
        r#"{"anthropic_api_key": "sk-or-v1-wrongslot"}"#,
    )
    .unwrap();
    let out = bx("anthropic", port, dir.path())
        .args(["council", "SUBJECT", "--via", "anthropic", "-m", "a/one"])
        .args(["--tools", "none", "--dry-run"])
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn set_key_dry_run_answers_json_on_stdout_unless_text_is_asked_for() {
    let dir = TempDir::new().unwrap();
    let run = |format: &str| {
        let out = bx_in(dir.path())
            .args(["council", "set-key", "--via", "openrouter", "--dry-run"])
            .args(["--format", format])
            .bounded_output();
        assert_eq!(out.status.code(), Some(0));
        out.stdout
    };
    let v: serde_json::Value = serde_json::from_slice(&run("json")).unwrap();
    assert_eq!(v["dry_run"], true, "{v}");
    assert_eq!(v["would_ask"], serde_json::json!(["openrouter"]), "{v}");
    assert!(
        v["config"].as_str().unwrap().ends_with("council.json"),
        "{v}"
    );
    assert!(run("text").is_empty());
}

/// `0600` in a `0700` directory, with no temp left behind.
#[cfg(unix)]
#[test]
fn set_key_stores_a_credential_privately_and_keeps_the_other_slots() {
    use std::io::Write as _;
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(
        cfg_dir.join("council.json"),
        r#"{"openai_api_key":"sk-proj-keepme1"}"#,
    )
    .unwrap();
    std::fs::set_permissions(&cfg_dir, std::fs::Permissions::from_mode(0o755)).unwrap();

    let mut child = bx_in(dir.path())
        .args(["council", "set-key", "--via", "anthropic"])
        .env("BRAVE_SEARCH_API_KEY", "brave-mock-token")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"sk-ant-storedsecret\n")
        .unwrap();
    let out = finish(child);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let path = cfg_dir.join("council.json");
    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(saved["anthropic_api_key"], "sk-ant-storedsecret");
    assert_eq!(
        saved["openai_api_key"], "sk-proj-keepme1",
        "another slot was lost"
    );

    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600,
        "a credential file must be owner-only"
    );
    assert_eq!(
        std::fs::metadata(&cfg_dir).unwrap().permissions().mode() & 0o777,
        0o700,
        "the default config dir is tightened on every save, as config.rs does"
    );
    let leftovers: Vec<_> = std::fs::read_dir(&cfg_dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("tmp"))
        .collect();
    assert!(leftovers.is_empty(), "temp file survived: {leftovers:?}");

    assert!(!String::from_utf8_lossy(&out.stdout).contains("storedsecret"));
    assert!(!String::from_utf8_lossy(&out.stderr).contains("storedsecret"));
}

#[test]
fn a_retry_after_longer_than_the_time_left_ends_the_member_now_rather_than_sleeping() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits = Arc::new(Mutex::new(0usize));
    let seen = Arc::clone(&hits);
    thread::spawn(move || {
        for sock in listener.incoming().flatten() {
            let seen = Arc::clone(&seen);
            thread::spawn(move || {
                let mut sock = sock;
                let Some((path, _)) = read_request(&mut sock) else {
                    return;
                };
                let (status, extra, body) = if is_free_probe(&path) {
                    (200, "", r#"{"data":{}}"#)
                } else {
                    *seen.lock().unwrap() += 1;
                    (
                        429,
                        "Retry-After: 18446744073709551615\r\n",
                        r#"{"error":{"message":"slow down"}}"#,
                    )
                };
                let _ = sock.write_all(
                    format!(
                        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n{extra}\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                );
            });
        }
    });

    let started = std::time::Instant::now();
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-seconds",
            "20",
            "--progress",
            "false",
        ],
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(15),
        "slept through the budget: {:?}",
        started.elapsed()
    );
    let m = r.manifest();
    assert_eq!(
        m["members"][0]["reason"],
        "provider_error",
        "{}",
        r.stderr()
    );
    let err = m["members"][0]["error"].as_str().unwrap();
    assert!(
        err.contains("budget left"),
        "the reason must name the trade: {err}"
    );
    assert_eq!(*hits.lock().unwrap(), 2, "one attempt per member");
    assert_eq!(m["members"][0]["retries"], 0);
}

/// Chmodding a found root would chmod whatever a planted symlink points at.
#[cfg(unix)]
#[test]
fn a_state_root_that_already_exists_keeps_its_own_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx/council");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o755)).unwrap();

    let out = council_in(port, dir.path(), None, &[]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        std::fs::metadata(&root).unwrap().permissions().mode() & 0o777,
        0o755,
        "chmodded a state root we did not create"
    );
    // The run directory inside it is ours.
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let run = std::path::Path::new(m["dir"].as_str().unwrap());
    assert_eq!(
        std::fs::metadata(run).unwrap().permissions().mode() & 0o777,
        0o700,
        "a run directory holds model output and is ours"
    );
}

#[cfg(unix)]
#[test]
fn a_dry_run_refuses_a_state_root_the_real_run_cannot_create() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("bx"), "not a directory").unwrap();
    for extra in [&["--dry-run"][..], &[]] {
        let out = council_in(port, dir.path(), None, extra);
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{extra:?}: {err}");
    }
    assert!(seen.lock().unwrap().iter().all(|b| b.is_empty()));
}

/// `println!` panics on a closed pipe, and `panic = "abort"` then loses the exit code.
#[cfg(unix)]
#[test]
fn every_line_this_command_prints_survives_a_reader_that_hangs_up() {
    use std::process::Stdio;
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    // A task directory big enough that `tasks` cannot fit inside the pipe buffer.
    let tasks = dir.path().join("brave-search/council");
    std::fs::create_dir_all(&tasks).unwrap();
    for i in 0..3_000 {
        std::fs::write(tasks.join(format!("t{i}.md")), "body {doc}").unwrap();
    }

    let cases: [(&str, Vec<&str>); 4] = [
        ("doctor", vec!["council", "doctor"]),
        ("tasks", vec!["council", "tasks"]),
        (
            "dry-run",
            vec![
                "council",
                "SUBJECT",
                "--via",
                "openrouter",
                "-m",
                "a/one,b/two",
                "--dry-run",
            ],
        ),
        (
            "run",
            vec![
                "council",
                "SUBJECT",
                "--via",
                "openrouter",
                "-m",
                "a/one,b/two",
                "--progress",
                "false",
            ],
        ),
    ];
    for (name, args) in cases {
        for close in ["stdout", "stderr"] {
            let mut c = bx("openrouter", port, dir.path());
            c.args(&args).stdout(Stdio::piped()).stderr(Stdio::piped());
            let mut child = c.spawn().unwrap();
            if close == "stdout" {
                drop(child.stdout.take());
            } else {
                drop(child.stderr.take());
            }
            let out = finish(child);
            // Any code, not a signal: `doctor` reports config health in its exit status.
            assert!(
                out.status.code().is_some(),
                "{name} with {close} closed died without an exit code: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(
                !String::from_utf8_lossy(&out.stderr).contains("panic"),
                "{name} with {close} closed panicked"
            );
        }
    }
}

/// Every other mock writes its body in one `write_all`, so only this drives a read loop past
/// one iteration.
#[test]
fn a_body_that_arrives_in_many_chunks_is_read_whole() {
    let text = "grounded prose with evidence. ".repeat(14_000); // ~420 KB, ~800 reads
    let want = text.chars().count();
    let payload = serde_json::json!({
        "model": "test/model", "provider": "Mock",
        "choices": [{"finish_reason": "stop", "message": {"content": text}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50}
    })
    .to_string();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for sock in listener.incoming().flatten() {
            let payload = payload.clone();
            thread::spawn(move || {
                let mut sock = sock;
                let Some((path, _)) = read_request(&mut sock) else {
                    return;
                };
                let body = if is_free_probe(&path) {
                    r#"{"data":{}}"#.to_string()
                } else {
                    payload
                };
                if sock
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                             Content-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .is_err()
                {
                    return;
                }
                for piece in body.as_bytes().chunks(512) {
                    if sock.write_all(piece).is_err() {
                        return;
                    }
                    let _ = sock.flush();
                }
            });
        }
    });

    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    assert_eq!(r.code(), 0, "{}", r.stderr());
    let m = r.manifest();
    for member in m["members"].as_array().unwrap() {
        assert_eq!(
            member["chars"].as_u64().unwrap() as usize,
            want,
            "a chunked body was not read whole: {}",
            member["model"]
        );
    }
    assert!(
        r.read("00-a_one.md")
            .contains("grounded prose with evidence.")
    );
    assert_eq!(m["quorum"]["delivered"], 2);
}

#[test]
fn a_provider_error_is_redacted_sanitised_and_bounded_before_it_reaches_the_manifest() {
    // The key `bx()` spawns with.
    let key = "sk-or-v1-mocktoken";
    let hostile = format!(
        "upstream rejected Authorization: Bearer {key} \u{1b}[2J\u{1b}]0;pwned\u{7}{}",
        "x".repeat(2000)
    );
    let (port, _) = serve(mock(move |_| {
        // 400 is terminal: the member is rejected carrying only this string.
        (
            400,
            serde_json::json!({"error": {"message": hostile}}).to_string(),
        )
    }));

    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    let members = m["members"].as_array().unwrap();
    assert_eq!(members.len(), 2);
    for member in members {
        let err = member["error"].as_str().unwrap_or_else(|| {
            panic!("a rejected member must carry an error: {member}");
        });
        assert!(!err.contains(key), "the credential survived: {err}");
        assert!(err.contains("***"), "redaction left no marker: {err}");
        assert!(
            !err.contains('\u{1b}') && !err.contains('\u{7}'),
            "an escape sequence reached the manifest: {err:?}"
        );
        // 512 plus `truncate`'s ellipsis note.
        assert!(
            err.chars().count() <= 560,
            "unbounded provider text: {} chars",
            err.chars().count()
        );
    }
    assert!(!r.stderr().contains(key), "{}", r.stderr());
    assert!(!r.read("index.md").contains(key));
}

#[test]
fn a_provider_redirect_is_not_followed_so_the_key_never_reaches_its_location() {
    let thief = TcpListener::bind("127.0.0.1:0").unwrap();
    let thief_port = thief.local_addr().unwrap().port();
    thief.set_nonblocking(true).unwrap();
    let provider = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = provider.local_addr().unwrap().port();
    thread::spawn(move || {
        for mut sock in provider.incoming().flatten() {
            let Some((path, _)) = read_request(&mut sock) else {
                continue;
            };
            let resp = if is_free_probe(&path) {
                let body = if path.contains("/api/v1/key") {
                    r#"{"data":{"limit":10,"limit_remaining":9}}"#
                } else {
                    r#"{"data":[]}"#
                };
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
            } else {
                format!(
                    "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{thief_port}/x\r\n\
                     Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
            };
            let _ = sock.write_all(resp.as_bytes());
        }
    });
    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "openrouter",
            "-m",
            "a/one,b/two",
        ])
        .args(["--tools", "none", "--progress", "false"])
        .bounded_output();
    // The 302 itself, not a pre-flight refusal.
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    for member in m["members"].as_array().unwrap() {
        assert_eq!(member["status"], "rejected", "{member}");
        assert!(
            member["error"].as_str().unwrap().contains("302"),
            "{member}"
        );
    }
    assert!(
        thief.accept().is_err(),
        "the member followed the redirect to its Location"
    );
}

/// The unit test pins `clamp_prompt`; this pins that both charge points apply it.
#[test]
fn an_absurd_prompt_count_cannot_become_the_published_cost() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "model": "claude-opus-5-5", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": prose()}],
                // Just under 2^53: parses and stays finite through `computed_usd`.
                "usage": {"input_tokens": 9_007_199_254_740_990u64, "output_tokens": 50}
            })
            .to_string(),
        )
    }));

    let r = council_rated(
        port,
        r#"{"rates":{"anthropic/claude-opus-5.5":[5,25],"b/two":[5,25]}}"#,
        &[],
    );
    let m = r.manifest();
    let llm = m["cost"]["llm_usd"].as_f64().unwrap_or_else(|| {
        panic!("a rated panel must publish a computed cost: {}", m["cost"]);
    });
    // No byte-budgeted round prices near this; unclamped it is ~9e10.
    assert!(
        llm < 10_000.0,
        "a fabricated token count reached the published cost: ${llm}"
    );
    assert_eq!(m["cost"]["cost_unknown"], false, "{}", m["cost"]);
    // A clamp, not a refusal.
    assert_eq!(m["quorum"]["delivered"], 2, "{m}");
    assert_eq!(r.code(), 0, "{}", r.stderr());

    // The retry charge point: a 200 `type: "error"` bills on every attempt.
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "type": "error",
                "error": {"type": "overloaded_error", "message": "try again"},
                "usage": {"input_tokens": 9_007_199_254_740_990u64, "output_tokens": 50}
            })
            .to_string(),
        )
    }));
    let r = council_rated(
        port,
        r#"{"rates":{"anthropic/claude-opus-5.5":[5,25],"b/two":[5,25]}}"#,
        &[],
    );
    let m = r.manifest();
    let llm = m["cost"]["llm_usd"].as_f64().unwrap_or_else(|| {
        panic!(
            "a retried round is still billed and still reported: {}",
            m["cost"]
        );
    });
    assert!(
        llm < 10_000.0,
        "a fabricated count reached the cost through the retry arm: ${llm}"
    );
    // Every attempt failed; only the bill is under test.
    assert_eq!(m["quorum"]["delivered"], 0, "{m}");
}

/// A connected but silent writer reads EAGAIN forever, never EOF.
#[cfg(unix)]
#[test]
fn a_subject_pipe_whose_writer_stays_silent_ends_instead_of_hanging() {
    let dir = TempDir::new().unwrap();
    let fifo = dir.path().join("silent.fifo");
    assert!(
        std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    // Both ends before the spawn: with no writer the read is EOF at once, and the write end
    // alone blocks until a reader exists.
    let _reader = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&fifo)
            .unwrap()
    };
    let _writer = std::fs::OpenOptions::new().write(true).open(&fifo).unwrap();

    let (port, _) = serve(mock(|_| reply(&prose())));
    let mut child = bx("openrouter", port, dir.path())
        .args([
            "council",
            &format!("@{}", fifo.display()),
            "--via",
            "openrouter",
        ])
        .args(["-m", "a/one,b/two", "--progress", "false", "--dry-run"])
        // Bounds the subject read, so the test costs 3 seconds.
        .args(["--max-seconds", "3"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    // Polled to a deadline: the regression is a hang.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while child.try_wait().unwrap().is_none() {
        if std::time::Instant::now() >= deadline {
            child.kill().ok();
            panic!("the subject read hung on a writer that never writes");
        }
        thread::sleep(std::time::Duration::from_millis(50));
    }
    let out = finish(child);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("writing to it"), "{err}");
}

#[cfg(unix)]
#[test]
fn a_task_max_seconds_bounds_the_subject_read() {
    let dir = TempDir::new().unwrap();
    let tasks = dir.path().join("brave-search/council");
    std::fs::create_dir_all(&tasks).unwrap();
    std::fs::write(
        tasks.join("brisk.md"),
        "---\nmax_seconds: 1\n---\nReview.\n\n{doc}",
    )
    .unwrap();
    let fifo = dir.path().join("silent.fifo");
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    let _reader = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&fifo)
            .unwrap()
    };
    let _writer = std::fs::OpenOptions::new().write(true).open(&fifo).unwrap();
    let (port, _) = serve(mock(|_| reply(&prose())));
    let started = std::time::Instant::now();
    let out = bx("openrouter", port, dir.path())
        .args([
            "council",
            &format!("@{}", fifo.display()),
            "--via",
            "openrouter",
        ])
        .args([
            "-m",
            "a/one",
            "--task",
            "brisk",
            "--progress",
            "false",
            "--dry-run",
        ])
        .bounded_output();
    assert_eq!(out.status.code(), Some(2));
    // The default wait is 30s; the task's own clock is one.
    assert!(started.elapsed().as_secs() < 15, "{:?}", started.elapsed());
}

#[test]
fn ls_prints_the_directory_beside_the_id_so_status_can_be_composed_with_it() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    // No `--out`: the default state directory is the one `ls` reads.
    let run = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter"])
        .args(["-m", "a/one,b/two", "--progress", "false"])
        .bounded_output();
    assert_eq!(
        run.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );

    // The TSV form; the JSON form below must name the same run.
    let ls = bx("openrouter", port, dir.path())
        .args(["council", "ls", "--format", "text"])
        .bounded_output();
    let line = String::from_utf8_lossy(&ls.stdout).trim().to_string();
    let cols: Vec<&str> = line.splitn(4, '\t').collect();
    let [id, path, sha, state] = cols[..] else {
        panic!("ls printed {line:?}")
    };
    assert!(!id.is_empty() && !path.is_empty());
    assert_eq!(sha.len(), 16, "{sha:?}");
    // State is appended after id and path, so the chain keeps working.
    assert!(state.starts_with('\u{2713}'), "{state:?}");
    assert!(
        state.contains("2/2") && state.contains("latest"),
        "{state:?}"
    );

    let st = bx("openrouter", port, dir.path())
        .args(["council", "status", "--out", path])
        .bounded_output();
    assert_eq!(
        st.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&st.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&st.stdout).unwrap();
    assert_eq!(
        (v["done"].as_u64(), v["total"].as_u64()),
        (Some(2), Some(2))
    );

    // JSON, the default: the same run, with fields named rather than positional.
    let json = bx("openrouter", port, dir.path())
        .args(["council", "ls"])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&json.stdout).expect("ls emits JSON");
    let first = &v["runs"][0];
    assert_eq!(first["run"], id);
    assert_eq!(first["dir"], path);
    assert_eq!(first["prompt_sha"], sha);
    assert_eq!(first["latest"], true);
}

#[test]
fn a_run_names_its_directory_before_it_spends_anything_not_only_when_it_finishes() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter"])
        .args(["-m", "a/one,b/two"])
        .bounded_output();
    let err = String::from_utf8_lossy(&run.stderr);
    assert_eq!(run.status.code(), Some(0), "{err}");

    let announced = err
        .lines()
        .position(|l| l.contains("council: ") && l.contains("/council/"))
        .unwrap_or_else(|| panic!("no run directory on stderr\n{err}"));
    let first_member = err
        .lines()
        .position(|l| l.contains("[1/2]"))
        .unwrap_or_else(|| panic!("no member line\n{err}"));
    assert!(
        announced < first_member,
        "the directory was announced after work began\n{err}"
    );
}

/// The fingerprint tells apart two runs started in the same second.
#[test]
fn the_subject_fingerprint_is_the_same_in_the_announcement_the_listing_and_the_manifest() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let mut shas = vec![];
    for subject in ["FIRST SUBJECT", "A DIFFERENT SUBJECT"] {
        let run = bx("openrouter", port, dir.path())
            .args(["council", subject, "--via", "openrouter"])
            .args(["-m", "a/one,b/two"])
            .bounded_output();
        let err = String::from_utf8_lossy(&run.stderr);
        assert_eq!(run.status.code(), Some(0), "{err}");
        let line = err
            .lines()
            .find(|l| l.contains("council: ") && l.contains("subject "))
            .unwrap_or_else(|| panic!("{err}"))
            .to_string();
        let announced = line
            .split("subject ")
            .nth(1)
            .and_then(|s| s.split(' ').next())
            .unwrap()
            .to_string();
        assert_eq!(announced.len(), 16, "{line:?}");
        // The path the run just named is the one the manifest lands in.
        let path = line.rsplit(" \u{2192} ").next().unwrap().trim();
        let m: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{path}/manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(m["prompt_sha"].as_str(), Some(announced.as_str()));
        shas.push(announced);
    }
    // Two subjects, two fingerprints.
    assert_ne!(shas[0], shas[1]);

    let ls = bx("openrouter", port, dir.path())
        .args(["council", "ls", "--format", "text"])
        .bounded_output();
    let mut listed: Vec<String> = vec![];
    for row in String::from_utf8_lossy(&ls.stdout).lines() {
        let f: Vec<&str> = row.split('\t').collect();
        let (path, sha) = (f[1], f[2]);
        // Matched by path, not position: same-second runs sort on the id's random suffix.
        let m: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{path}/manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(m["prompt_sha"].as_str(), Some(sha), "{row}");
        listed.push(sha.to_string());
    }
    listed.sort();
    shas.sort();
    assert_eq!(listed, shas);
}

/// Within the id's second, runs are ordered by start time, not by the random suffix.
#[test]
fn two_runs_in_the_same_second_are_still_listed_in_the_order_they_started() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let mut shas = vec![];
    for subject in ["OLDER SUBJECT", "NEWER SUBJECT"] {
        let run = bx("openrouter", port, dir.path())
            .args(["council", subject, "--via", "openrouter"])
            .args(["-m", "a/one,b/two"])
            .bounded_output();
        let err = String::from_utf8_lossy(&run.stderr);
        assert_eq!(run.status.code(), Some(0), "{err}");
        shas.push(
            err.lines()
                .find_map(|l| l.split("subject ").nth(1))
                .and_then(|s| s.split(' ').next())
                .unwrap_or_else(|| panic!("{err}"))
                .to_string(),
        );
    }
    let ls = bx("openrouter", port, dir.path())
        .args(["council", "ls", "--format", "text"])
        .bounded_output();
    let out = String::from_utf8_lossy(&ls.stdout);
    let top = out.lines().next().unwrap_or_else(|| panic!("{out}"));
    assert!(top.contains(" · latest"), "{top}");
    assert_eq!(
        top.split('\t').nth(2),
        Some(shas[1].as_str()),
        "the run started second is not the one marked latest\n{out}"
    );
}

#[test]
fn status_without_out_reads_the_newest_run_and_says_so_plainly_when_there_are_none() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();

    // Before any run: exit 2 naming the state directory.
    let empty = bx("openrouter", port, dir.path())
        .args(["council", "status"])
        .bounded_output();
    assert_eq!(empty.status.code(), Some(2));
    let err = String::from_utf8_lossy(&empty.stderr);
    assert!(err.contains("no runs under"), "{err}");

    // Two runs of different sizes, so the newest is identifiable from `status` alone.
    for members in ["a/one,b/two", "a/one,b/two,c/three"] {
        let run = bx("openrouter", port, dir.path())
            .args(["council", "SUBJECT", "--via", "openrouter"])
            .args(["-m", members, "--progress", "false"])
            .bounded_output();
        assert_eq!(
            run.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
    let st = bx("openrouter", port, dir.path())
        .args(["council", "status"])
        .bounded_output();
    assert_eq!(
        st.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&st.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&st.stdout).unwrap();
    assert_eq!(v["total"].as_u64(), Some(3), "not the newest run: {v}");
}

#[test]
fn status_refuses_a_directory_with_neither_status_nor_manifest() {
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("not-a-run");
    std::fs::create_dir_all(&run).unwrap();
    let out = bx_in(dir.path())
        .args(["council", "status", "--out"])
        .arg(&run)
        .bounded_output();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("status.json"), "{err}");
}

/// An older layout: a manifest and replies, no `status.json`.
#[test]
fn the_status_less_layout_reads_in_ls_and_findings_and_is_refused_by_status_and_useful_yes() {
    let dir = TempDir::new().unwrap();
    let root = bx_root(dir.path()).join("bx").join("council");
    let run = root.join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("00-a_one.md"), "## A\nx\n").unwrap();
    std::fs::write(
        run.join("manifest.json"),
        r#"{"state":"ok","members":[{"model":"a/one","status":"ok","file":"00-a_one.md"}]}"#,
    )
    .unwrap();
    let bx = |args: &[&str]| bx_in(dir.path()).arg("council").args(args).bounded_output();
    let ls: serde_json::Value = serde_json::from_slice(&bx(&["ls"]).stdout).unwrap();
    assert_eq!(ls["runs"][0]["phase"], "ok", "{ls}");
    let f: serde_json::Value = serde_json::from_slice(&bx(&["findings"]).stdout).unwrap();
    assert_eq!(f["members"][0]["model"], "a/one", "{f}");
    for sub in ["status", "useful-yes"] {
        let out = bx(&[sub]);
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{sub}: {err}");
        assert!(err.contains("status.json"), "{sub}: {err}");
    }
    assert!(!root.join("scores.jsonl").exists());
}

#[test]
fn council_refuses_the_global_flags_it_cannot_honour() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    for flag in [
        ["--extra", "count=99"],
        ["--endpoint", "/res/v1/web/search"],
    ] {
        let r = council(
            port,
            &[flag[0], flag[1], "-m", "a/one,b/two", "--progress", "false"],
        );
        assert_eq!(
            r.code(),
            2,
            "{} {} was accepted and ignored\n{}",
            flag[0],
            flag[1],
            r.stderr()
        );
        assert!(r.stderr().contains(flag[0]), "{}", r.stderr());
    }
}

/// An unreserved word is a subject, so a dropped reserved word would run a paid panel.
#[test]
fn every_sub_command_word_is_free_and_never_becomes_a_paid_panel() {
    let dir = TempDir::new().unwrap();
    // No mock and no Brave key: a sub-command needs neither. `doctor` and `models` report
    // config health in their exit code, so 2 is allowed for them.
    for word in ["ls", "tasks", "models", "doctor"] {
        let judges_config = matches!(word, "models" | "doctor");
        let out = bx_in(dir.path())
            .args(["council", word])
            .env("OPENROUTER_API_KEY", "sk-or-v1-mocktoken")
            .bounded_output();
        assert!(
            out.status.code() == Some(0) || (judges_config && out.status.code() == Some(2)),
            "`bx council {word}` is not a free sub-command: {:?} {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        // Only a run's manifest has `quorum`; `doctor` legitimately lists members.
        assert!(
            !String::from_utf8_lossy(&out.stdout).contains("\"quorum\""),
            "`bx council {word}` ran a panel"
        );
    }
}

#[test]
fn set_key_walks_every_provider_and_a_bad_one_does_not_lose_the_good_ones() {
    let dir = TempDir::new().unwrap();
    // The walk needs a TTY, so a pipe is refused rather than hanging.
    let piped = bx_in(dir.path())
        .args(["council", "set-key"])
        .stdin(std::process::Stdio::piped())
        .bounded_output();
    assert_eq!(piped.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&piped.stderr).contains("needs a terminal"),
        "{}",
        String::from_utf8_lossy(&piped.stderr)
    );

    // The scriptable single-provider form stays pipeable, and writes only its own slot.
    let one = |wire: &str, key: &str| {
        let mut c = bx_in(dir.path())
            .args(["council", "set-key", "--via", wire])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        use std::io::Write as _;
        c.stdin
            .take()
            .unwrap()
            .write_all(format!("{key}\n").as_bytes())
            .unwrap();
        finish(c)
    };
    assert_eq!(
        one("openrouter", "sk-or-v1-realkeyrealkeyrealkey")
            .status
            .code(),
        Some(0)
    );
    // Another provider's prefix is refused, and the key already stored survives it.
    let wrong = one("openai", "sk-ant-wrongslot-aaaaaaaaaaaa");
    assert_eq!(wrong.status.code(), Some(2));
    let saved = std::fs::read_to_string(dir.path().join("brave-search/council.json")).unwrap();
    assert!(saved.contains("sk-or-v1-realkeyrealkeyrealkey"), "{saved}");
    assert!(!saved.contains("openai_api_key"), "{saved}");

    // `doctor` names the repair while any provider is unset.
    let doc = bx_in(dir.path())
        .args(["council", "doctor"])
        .bounded_output();
    assert!(
        String::from_utf8_lossy(&doc.stdout).contains("bx council set-key"),
        "doctor must name the onboarding command: {}",
        String::from_utf8_lossy(&doc.stdout)
    );
}

/// `council.json` is re-read just before the save.
#[test]
fn set_key_keeps_a_config_edit_made_while_it_was_prompting() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = bx_root(dir.path()).join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    let path = cfg_dir.join("council.json");
    std::fs::write(&path, "{}").unwrap();
    let mut child = bx_in(dir.path())
        .args(["council", "set-key", "--via", "openrouter"])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    // Wait for the prompt, so the edit lands after set-key has read the file.
    let mut stderr = child.stderr.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = thread::spawn(move || {
        let mut seen = Vec::new();
        let mut byte = [0u8; 1];
        while stderr.read(&mut byte).is_ok_and(|n| n == 1) {
            seen.push(byte[0]);
            if seen.ends_with(b"paste key (blank to skip): ") {
                tx.send(()).ok();
            }
        }
        seen
    });
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .expect("set-key never prompted");
    std::fs::write(&path, r#"{"aliases":{"mine":"a/one"}}"#).unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"sk-or-v1-realkeyrealkeyrealkey\n")
        .unwrap();
    let out = finish(child);
    let err = String::from_utf8_lossy(&reader.join().unwrap()).into_owned();
    assert_eq!(out.status.code(), Some(0), "{err}");
    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(
        saved["openrouter_api_key"],
        "sk-or-v1-realkeyrealkeyrealkey"
    );
    assert_eq!(
        saved["aliases"]["mine"], "a/one",
        "the edit was lost: {saved}"
    );
}

#[test]
fn set_key_says_when_an_exported_env_var_overrides_the_saved_key() {
    let dir = TempDir::new().unwrap();
    let mut child = bx_in(dir.path())
        .args(["council", "set-key", "--via", "anthropic"])
        .env("ANTHROPIC_API_KEY", "sk-ant-exported")
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"sk-ant-saved\n")
        .unwrap();
    let out = finish(child);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{err}");
    assert!(
        err.contains("ANTHROPIC_API_KEY is set and overrides"),
        "{err}"
    );
    let saved =
        std::fs::read_to_string(bx_root(dir.path()).join("brave-search/council.json")).unwrap();
    assert!(saved.contains("sk-ant-saved"), "{saved}");
}

/// A pseudo-terminal: the master to type into, and the slave as a child's stdin.
#[cfg(unix)]
fn pty() -> (std::fs::File, Stdio) {
    use std::os::fd::FromRawFd;
    let (mut master, mut slave) = (0, 0);
    // SAFETY: openpty writes two fds on success; null name/termios/winsize are allowed.
    let rc = unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, 0, "openpty: {}", std::io::Error::last_os_error());
    // SAFETY: both fds were just opened and are owned by nothing else.
    unsafe {
        (
            std::fs::File::from_raw_fd(master),
            Stdio::from(std::os::fd::OwnedFd::from_raw_fd(slave)),
        )
    }
}

/// Here the read error is a non-UTF-8 paste.
#[cfg(unix)]
#[test]
fn a_read_error_after_one_accepted_key_still_saves_that_key() {
    let dir = TempDir::new().unwrap();
    let (mut master, slave) = pty();
    let child = bx_in(dir.path())
        .args(["council", "set-key"])
        .stdin(slave)
        .spawn()
        .unwrap();
    master
        .write_all(b"sk-or-v1-realkeyrealkeyrealkey\n\xff\xfe\n")
        .unwrap();
    let out = finish(child);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("not read"), "{err}");
    let saved =
        std::fs::read_to_string(bx_root(dir.path()).join("brave-search/council.json")).unwrap();
    assert!(saved.contains("sk-or-v1-realkeyrealkeyrealkey"), "{saved}");
}

#[test]
fn a_panel_out_of_credit_says_so_in_the_exit_code() {
    let (port, _) = serve(mock(|_| {
        (
            402,
            r#"{"error":{"message":"Insufficient credits","metadata":{"error_type":"payment_required"}}}"#
                .into(),
        )
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    assert_eq!(r.code(), 8, "{}", r.stderr());
    let m = r.manifest();
    assert!(
        !m["hint"].as_str().unwrap().contains("Retry the rest"),
        "{}",
        m["hint"]
    );
    for x in m["members"].as_array().unwrap() {
        assert_eq!(x["action"], "none", "{x}");
    }
}

#[test]
fn max_tokens_cannot_be_set_above_the_ceiling_escalation_climbs_to() {
    let dir = TempDir::new().unwrap();
    let run = |v: &str| {
        bx("openrouter", 1, dir.path())
            .args([
                "council",
                "SUBJECT",
                "--via",
                "openrouter",
                "-m",
                "a/one,b/two",
            ])
            .args(["--dry-run", "--max-tokens", v])
            .bounded_output()
    };
    let over = run("128001");
    assert_eq!(over.status.code(), Some(2));
    let err = String::from_utf8_lossy(&over.stderr);
    assert!(err.contains("1..=128000"), "{err}");
    // The ceiling itself is accepted.
    assert_eq!(run("128000").status.code(), Some(0));
}

/// Otherwise every search 401s and an ungrounded review is delivered as grounded.
#[test]
fn a_brave_key_that_cannot_work_is_refused_before_a_grounded_panel_is_paid_for() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = |name: &str, extra: &[&str]| {
        bx("openrouter", port, dir.path())
            .args([
                "council",
                "SUBJECT",
                "--via",
                "openrouter",
                "-m",
                "a/one,b/two",
            ])
            .args(["--progress", "false"])
            .args(extra)
            .arg("--out")
            .arg(dir.path().join(name))
            // Shorter than `bx config set-key`'s floor, so never a real key.
            .env("BRAVE_SEARCH_API_KEY", "x")
            .bounded_output()
    };

    let out = run("grounded", &[]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(
        err.to_lowercase().contains("brave"),
        "the message must name the credential: {err}"
    );
    assert!(
        seen.lock().unwrap().is_empty(),
        "a refusal in `plan` sends nothing and bills nothing"
    );

    // `--tools none` makes no Brave call, so it needs no key.
    assert_eq!(
        run("ungrounded", &["--tools", "none"]).status.code(),
        Some(0)
    );
}

#[test]
fn a_brave_key_refused_mid_run_is_asked_once_and_named_as_the_fix() {
    let brave = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = Arc::clone(&brave);
    let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":{}}"#.to_string());
        }
        if path.contains("/llm/context") {
            counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return (401, r#"{"error":"bad key"}"#.into());
        }
        // A model that keeps searching whatever it is told.
        tool_call("q")
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--concurrency",
            "1",
            "--max-steps",
            "4",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(
        brave.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "Brave was asked again after refusing the key: {m}"
    );
    for x in m["members"].as_array().unwrap() {
        assert_eq!(x["reason"], "search_failed", "{x}");
        assert_eq!(x["action"], "none", "{x}");
        assert!(
            x["error"]
                .as_str()
                .unwrap()
                .contains("BRAVE_SEARCH_API_KEY"),
            "{x}"
        );
    }
}

/// Brave answers a bad token with a 422; an over-long query 422s too, and must not latch.
#[test]
fn a_brave_422_latches_search_off_only_when_it_names_an_invalid_token() {
    for (code, latched) in [("SUBSCRIPTION_TOKEN_INVALID", true), ("VALIDATION", false)] {
        let brave = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&brave);
        let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
            if is_free_probe(path) {
                return (200, r#"{"data":{}}"#.to_string());
            }
            if path.contains("/llm/context") {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let b = format!(r#"{{"error":{{"code":"{code}","status":422}}}}"#);
                return (422, b);
            }
            tool_call("q")
        }));
        let r = council(
            port,
            &["-m", "a/one", "--max-steps", "4", "--progress", "false"],
        );
        let m = &r.manifest()["members"][0];
        let asked = brave.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            asked == 1,
            latched,
            "{code}: Brave asked {asked} times: {m}"
        );
        let error = m["error"].as_str().unwrap();
        assert_eq!(error.contains("BRAVE_SEARCH_API_KEY"), latched, "{m}");
    }
}

/// The rejection names the quota, not the key.
#[test]
fn an_exhausted_brave_quota_stops_every_later_search_in_the_run() {
    let brave = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = Arc::clone(&brave);
    let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":{}}"#.to_string());
        }
        if path.contains("/llm/context") {
            counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return (
                429,
                r#"{"type":"ErrorResponse","error":{"code":"QUOTA_LIMITED","status":429}}"#.into(),
            );
        }
        tool_call("q")
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--concurrency",
            "1",
            "--max-steps",
            "4",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(brave.load(std::sync::atomic::Ordering::SeqCst), 1, "{m}");
    for x in m["members"].as_array().unwrap() {
        assert_eq!(x["reason"], "search_failed", "{x}");
        let error = x["error"].as_str().unwrap();
        assert!(error.contains("quota") && !error.contains("KEY"), "{x}");
    }
}

#[test]
fn an_ungrounded_run_publishes_the_search_limit_it_actually_enforced() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--tools",
            "none",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(m["members"][0]["limits"]["max_searches"]["limit"], 0, "{m}");

    let dry = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--tools",
            "none",
            "--dry-run",
            "--progress",
            "false",
        ],
    );
    let p = dry.manifest();
    assert_eq!(p["limits"]["max_searches"], 0, "{p}");
    assert_eq!(p["egress"]["max_search_queries"], 0, "{p}");
}

/// Scrubbed before serialization: after it, an ESC is six inert bytes a consumer's parser
/// turns back into a live escape.
#[test]
fn doctor_scrubs_the_names_it_publishes_before_it_serialises_them() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    // `\u001b` as JSON spells it: a raw control byte is invalid JSON.
    std::fs::write(
        cfg_dir.join("council.json"),
        r#"{"panels":{"p\u001b[2JZ":["a/one","b/two"]},"aliases":{"k\u001bY":"a/one"}}"#,
    )
    .unwrap();
    let out = bx_in(dir.path())
        .args(["council", "doctor", "--format", "json"])
        .env("OPENROUTER_API_KEY", "sk-or-v1-mocktoken")
        .bounded_output();
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The names are published, not dropped.
    assert!(
        stdout.contains("\"pZ\"") && stdout.contains("\"k\""),
        "{stdout}"
    );
    assert!(
        !stdout.contains("\\u001b"),
        "an escape reached a JSON consumer, re-armed by its own parser: {stdout}"
    );
}

/// `ls`, `status` and `prune` print the state root; only Linux reads `XDG_STATE_HOME`.
#[cfg(target_os = "linux")]
#[test]
fn the_run_commands_scrub_the_paths_they_publish_before_they_serialise_them() {
    let dir = TempDir::new().unwrap();
    let state = dir.path().join("s\u{1b}[2J");
    let run = state
        .join("bx")
        .join("council")
        .join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("status.json"), "{}").unwrap();
    for sub in [&["ls"][..], &["status"], &["prune", "--dry-run"]] {
        let out = bx_in(dir.path())
            .arg("council")
            .args(sub)
            .args(["--format", "json"])
            .env("XDG_STATE_HOME", &state)
            .bounded_output();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.contains("/s/bx/council"), "{sub:?}: {stdout}");
        assert!(!stdout.contains("\\u001b"), "{sub:?}: {stdout}");
    }
}

/// Through the state root: `--out` refuses such a path.
#[cfg(target_os = "linux")]
#[test]
fn the_run_scrubs_the_member_and_sources_paths_it_publishes() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", port, dir.path())
        .env("XDG_STATE_HOME", dir.path().join("s\u{1b}[2J"))
        .args(["council", "SUBJECT", "--via", "openrouter"])
        .args(["-m", "a/one,b/two", "--progress", "false"])
        .bounded_output();
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).expect("manifest on stdout");
    assert!(m["members"][0]["file"].is_string(), "{m}");
    assert!(m["sources_file"].is_string(), "{m}");
    assert!(!m.to_string().contains("\\u001b"), "{m}");
}

/// Published scrubbed, `a\u{200b}b` would name `ab`, another directory.
#[test]
fn an_out_dir_that_sanitizing_would_alter_is_refused_before_anything_is_sent() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let real = dir.path().join("a\u{200b}b");
    let out = council_in(port, dir.path(), Some(&real), &[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(!real.exists());
    assert!(
        seen.lock().unwrap().iter().all(|b| b.is_empty()),
        "a request was sent"
    );
}

/// `useful-*`, `tasks` and a failed `prune` scrub the paths and names they print too, while the
/// verdict record keeps the raw directory `ls` joins on.
#[cfg(target_os = "linux")]
#[test]
fn useful_tasks_and_prune_scrub_what_they_publish_before_they_serialise_it() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().unwrap();
    let esc = dir.path().join("s\u{1b}[2J");
    let cfg = esc.join("brave-search");
    std::fs::create_dir_all(&cfg).unwrap();
    std::fs::write(
        cfg.join("council.json"),
        "{\"default_task\":\"x\\u001b[2J\"}",
    )
    .unwrap();
    let root = esc.join("bx").join("council");
    let run = root.join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("status.json"), "{}").unwrap();
    std::fs::write(run.join("manifest.json"), r#"{"state":"ok"}"#).unwrap();
    // A finished old run prune cannot fully remove.
    let stuck = root.join("20260101-120000-bbbbbb");
    std::fs::create_dir_all(stuck.join("locked")).unwrap();
    std::fs::write(stuck.join("locked").join("x"), "x").unwrap();
    std::fs::write(stuck.join("manifest.json"), "{}").unwrap();
    std::fs::write(stuck.join("brief.md"), "b").unwrap();
    age_by_days(&stuck.join("brief.md"), 60);
    std::fs::set_permissions(stuck.join("locked"), std::fs::Permissions::from_mode(0o500)).unwrap();
    let bx = |args: &[&str]| {
        let out = bx_in(dir.path())
            .arg("council")
            .args(args)
            .env("XDG_STATE_HOME", &esc)
            .env("XDG_CONFIG_HOME", &esc)
            .bounded_output();
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let out_dir = run.to_str().unwrap();
    for args in [
        &["tasks"][..],
        &["useful-yes", "--out", out_dir, "--dry-run"],
        &["useful-yes", "--out", out_dir],
        &["prune"],
    ] {
        let stdout = bx(args);
        assert!(stdout.contains("/s/"), "{args:?}: {stdout}");
        assert!(!stdout.contains("\\u001b"), "{args:?}: {stdout}");
    }
    std::fs::set_permissions(stuck.join("locked"), std::fs::Permissions::from_mode(0o700)).unwrap();
    let ls: serde_json::Value = serde_json::from_str(&bx(&["ls"])).unwrap();
    let listed = ls["runs"].as_array().unwrap();
    assert!(listed.iter().any(|r| r["useful"] == true), "{ls}");
}

/// The parse error of a broken task file echoes its raw text into the JSON description.
#[test]
fn tasks_json_description_of_an_invalid_task_file_is_free_of_controls() {
    let dir = TempDir::new().unwrap();
    let tasks = dir.path().join("brave-search/council");
    std::fs::create_dir_all(&tasks).unwrap();
    std::fs::write(
        tasks.join("bad.md"),
        "---\n\u{1b}]0;pwned\u{7}: x\n---\nbody",
    )
    .unwrap();
    let out = bx_in(dir.path())
        .args(["council", "tasks"])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let bad = v["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "bad")
        .unwrap_or_else(|| panic!("{v}"));
    let desc = bad["description"].as_str().unwrap();
    assert!(desc.starts_with("(invalid:"), "{v}");
    assert!(!desc.chars().any(char::is_control), "{desc:?}");
}

/// `@default` routes opus direct, so an OpenRouter key alone cannot run it.
#[test]
fn doctor_is_judged_against_the_panel_that_would_run_not_a_count_of_keys() {
    let dir = TempDir::new().unwrap();
    let cmd = |format: &str| {
        let mut c = bx_in(dir.path());
        c.args(["council", "doctor", "--format", format]);
        c
    };
    let none = cmd("text").bounded_output();
    assert!(
        String::from_utf8_lossy(&none.stdout).contains("not set"),
        "{}",
        String::from_utf8_lossy(&none.stdout)
    );
    assert_eq!(none.status.code(), Some(2), "an unrunnable config passed");

    let one = cmd("text")
        .env("OPENROUTER_API_KEY", "sk-or-v1-mocktoken")
        .bounded_output();
    let text = String::from_utf8_lossy(&one.stdout);
    // "set" means present and well-shaped; `doctor` never calls the network.
    assert!(text.contains("\u{2713} openrouter  set"), "{text}");
    assert!(text.contains("\u{2717} anthropic   not set"), "{text}");
    assert_eq!(
        one.status.code(),
        Some(2),
        "a key set that cannot run the default panel is not a clean bill of health"
    );
    assert!(text.contains("missing     anthropic"), "{text}");

    // The JSON form agrees.
    let json = cmd("json")
        .env("OPENROUTER_API_KEY", "sk-or-v1-mocktoken")
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&json.stdout).expect("doctor emits JSON");
    assert_eq!(v["ok"], false);
    assert_eq!(v["missing_keys"][0], "anthropic");
    assert_eq!(
        v["keys"][0],
        serde_json::json!({
            "provider": "openrouter", "set": true, "detail": "set"
        })
    );
    assert_eq!(
        json.status.code(),
        Some(2),
        "the two forms agree on the code"
    );
}

/// A warning row, not a failure: the run accepts the same configuration.
#[test]
fn doctor_passes_when_only_an_unused_provider_key_is_malformed() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = bx_root(dir.path()).join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(cfg_dir.join("council.json"), r#"{"direct":[]}"#).unwrap();
    let out = bx_in(dir.path())
        .args(["council", "doctor"])
        .env("OPENROUTER_API_KEY", "sk-or-v1-mocktoken")
        .env("OPENAI_API_KEY", "sk-ant-wrongslot-aaaaaaaa")
        .env("BRAVE_SEARCH_API_KEY", "BSAtestkey000")
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(out.status.code(), Some(0), "{v}");
    assert_eq!(v["ok"], true, "{v}");
    let openai = v["keys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["provider"] == "openai")
        .unwrap();
    assert_eq!(openai["set"], false, "the row still warns: {v}");
}

#[test]
fn doctor_reports_keys_even_when_the_default_panel_cannot_resolve() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = bx_root(dir.path()).join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(cfg_dir.join("council.json"), r#"{"panels":{"default":[]}}"#).unwrap();
    let out = bx_in(dir.path())
        .args(["council", "doctor"])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!("{e}: {}", String::from_utf8_lossy(&out.stderr));
    });
    assert_eq!(out.status.code(), Some(2), "{v}");
    assert_eq!(v["ok"], false, "{v}");
    assert!(
        v["panel_error"].as_str().is_some_and(|e| !e.is_empty()),
        "{v}"
    );
    assert_eq!(v["keys"].as_array().map(Vec::len), Some(4), "{v}");
}

#[test]
fn doctor_judges_the_panel_a_default_task_pins() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = bx_root(dir.path()).join("brave-search");
    std::fs::create_dir_all(cfg_dir.join("council")).unwrap();
    std::fs::write(
        cfg_dir.join("council").join("mine.md"),
        "---\npanel: a/one,b/two\n---\nReview {doc}\n",
    )
    .unwrap();
    std::fs::write(cfg_dir.join("council.json"), r#"{"default_task":"mine"}"#).unwrap();
    let out = bx_in(dir.path())
        .args(["council", "doctor"])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!("{e}: {}", String::from_utf8_lossy(&out.stderr));
    });
    let models: Vec<&str> = v["default_panel"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["model"].as_str().unwrap())
        .collect();
    assert_eq!(models, ["a/one", "b/two"], "{v}");
}

/// Normalised, not refused: `-0.0` and `0` are one cap.
#[test]
fn a_cap_of_negative_zero_prints_as_a_cap_of_zero() {
    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", 1, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter"])
        .args(["-m", "a/one,b/two", "--dry-run", "--format", "text"])
        .arg("--max-usd=-0.0")
        .bounded_output();
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // The cap as typed, not rounded to `$0.00`.
    assert!(text.contains("--max-usd $0,"), "{text}");
    assert!(!text.contains("$-0"), "{text}");
}

#[test]
fn a_second_run_into_one_out_directory_is_refused_rather_than_overwriting_the_first() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");

    let first = council_in(port, dir.path(), Some(&run), &["-m", "a/one,b/two,c/three"]);
    assert_eq!(
        first.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let brief = std::fs::read_to_string(run.join("brief.md")).unwrap();
    let third = std::fs::read_to_string(run.join("02-c_three.md")).unwrap();
    let manifest = std::fs::read_to_string(run.join("manifest.json")).unwrap();

    let second = council_in(port, dir.path(), Some(&run), &[]);
    let err = String::from_utf8_lossy(&second.stderr);
    // Refused before `brief.md`, so nothing was sent.
    assert_eq!(second.status.code(), Some(2), "{err}");
    assert!(err.contains("already contains manifest.json"), "{err}");
    assert!(
        err.contains("--out"),
        "the message must name the repair: {err}"
    );

    // Run 1 is untouched.
    assert_eq!(
        std::fs::read_to_string(run.join("brief.md")).unwrap(),
        brief
    );
    assert_eq!(
        std::fs::read_to_string(run.join("02-c_three.md")).unwrap(),
        third
    );
    assert_eq!(
        std::fs::read_to_string(run.join("manifest.json")).unwrap(),
        manifest
    );

    // An existing empty directory is still accepted.
    let fresh = dir.path().join("prepared");
    std::fs::create_dir_all(&fresh).unwrap();
    assert_eq!(
        council_in(port, dir.path(), Some(&fresh), &[])
            .status
            .code(),
        Some(0)
    );
}

/// A 401 and a content-filter refusal: no retry hint, and the refusal keeps its own reason.
#[test]
fn a_failure_no_retry_can_change_is_neither_advertised_as_retryable_nor_misfiled() {
    let (port, _) = serve(mock(|body: &str| {
        if body.contains(r#""model":"a/one""#) {
            return (
                401,
                r#"{"error":{"message":"no credentials","code":"invalid_api_key"}}"#.to_string(),
            );
        }
        if body.contains(r#""model":"c/three""#) {
            return (
                200,
                serde_json::json!({
                    "choices": [{"finish_reason": "stop",
                                 "message": {"content": null, "refusal": "I cannot help."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1}
                })
                .to_string(),
            );
        }
        reply(&prose())
    }));
    let r = council(port, &["-m", "a/one,b/two,c/three", "--progress", "false"]);
    let m = r.manifest();
    let by = |slug: &str| {
        m["members"]
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["model"] == slug)
            .unwrap_or_else(|| panic!("{slug} is in the panel: {m}"))
            .clone()
    };

    let hint = m["hint"].as_str().unwrap();
    assert!(hint.contains("1 of 3 delivered"), "{hint}");
    assert!(
        !hint.contains("Retry the rest"),
        "neither failure is retryable: {hint}"
    );

    assert_eq!(by("a/one")["reason"], "provider_error", "{m}");
    assert_eq!(by("a/one")["action"], "none", "a 401 is not a transient");

    let refused = by("c/three");
    assert_eq!(refused["reason"], "refused", "{m}");
    assert_eq!(refused["finish_reason"], "content_filter", "{m}");
    assert_eq!(refused["action"], "none", "{m}");
}

#[test]
fn a_lost_chain_on_the_last_attempt_does_not_buy_an_extra_billed_send() {
    let sends: Arc<Mutex<std::collections::HashMap<String, usize>>> = Arc::default();
    let counted = Arc::clone(&sends);
    let (port, _) = serve(mock(move |body: &str| {
        let model = serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(|v| v["model"].as_str().map(str::to_string))
            .unwrap_or_default();
        let n = {
            let mut m = counted.lock().unwrap();
            let c = m.entry(model).or_insert(0);
            *c += 1;
            *c
        };
        // Three retryable failures spend the budget; the chain is lost on the last attempt.
        if n <= 3 {
            (
                500,
                r#"{"error":{"code":"server_error","message":"boom"}}"#.to_string(),
            )
        } else {
            (
                404,
                r#"{"error":{"code":"previous_response_not_found","message":"gone"}}"#.to_string(),
            )
        }
    }));

    let dir = TempDir::new().unwrap();
    // Zero data retention sends every OpenAI request stateless, leaving no chain to lose.
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(cfg_dir.join("council.json"), r#"{"zdr": false}"#).unwrap();
    bx("openai", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openai", "-m", "a/one,b/two"])
        .args(["--tools", "none", "--progress", "false", "--out"])
        .arg(dir.path().join("run"))
        .bounded_output();
    let sends = sends.lock().unwrap();
    assert_eq!(sends.len(), 2, "{sends:?}");
    for (model, n) in sends.iter() {
        assert_eq!(
            *n, 4,
            "{model} billed {n} generations for one round; the budget is MAX_RETRIES + 1 = 4 \
             wherever the replay lands"
        );
    }
}

#[test]
fn a_member_sleeping_out_a_retry_rechecks_the_cap_before_it_re_sends() {
    let (port, _) = serve(mock(|body: &str| {
        let model = serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(|v| v["model"].as_str().map(str::to_string))
            .unwrap_or_default();
        // The sleeper: a retryable failure that bills nothing, then a backoff.
        if model == "claude-opus-5-5" {
            return (
                500,
                r#"{"type":"error","error":{"type":"api_error","message":"boom"}}"#.to_string(),
            );
        }
        // The spender, after the sleeper's cap check: $10 against a $0.01 cap.
        thread::sleep(std::time::Duration::from_millis(150));
        (
            200,
            serde_json::json!({
                "type": "error", "content": [],
                "error": {"type": "overloaded_error", "message": "busy"},
                "usage": {"input_tokens": 2_000_000, "output_tokens": 0}
            })
            .to_string(),
        )
    }));
    let r = council_rated(
        port,
        r#"{"rates":{"anthropic/claude-opus-5.5":[5,25],"b/two":[5,25]}}"#,
        &["--max-usd", "0.01"],
    );
    let m = r.manifest();
    let sleeper = m["members"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["model"] == "anthropic/claude-opus-5.5")
        .expect("the sleeper is in the panel")
        .clone();
    assert_eq!(
        sleeper["tokens"]["prompt"], 0,
        "the sleeper woke and billed a generation after the cap was crossed: {m}"
    );
    // `retries` counts re-sends that happened; one the cap turned back is not one.
    assert_eq!(sleeper["retries"], 0, "{m}");
    assert_eq!(sleeper["reason"], "max_usd", "{m}");
}

/// `note!` (a task error on stderr) and `say!` (`doctor` on stdout) both sanitize.
#[test]
fn neither_stream_lets_a_config_file_drive_the_terminal() {
    let home = TempDir::new().unwrap();
    let esc = "\u{1b}[2J\u{1b}]0;pwned\u{7}";
    let raw = |s: String| s.contains('\u{1b}') || s.contains('\u{7}');

    // A frontmatter error quotes the bad line, through `note!`.
    let tasks = home.path().join("brave-search").join("council");
    std::fs::create_dir_all(&tasks).unwrap();
    std::fs::write(
        tasks.join("hostile.md"),
        format!("---\n{esc}not a pair\n---\nbody {{doc}}\n"),
    )
    .unwrap();
    let out = bx("openrouter", 1, home.path())
        .args(["council", "THE SUBJECT", "--task", "hostile", "--dry-run"])
        .bounded_output();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(err.contains("frontmatter"), "wrong error: {err:?}");
    assert!(!raw(err.clone()), "escape reached stderr: {err:?}");

    // `direct` entries are unvalidated, and `doctor` prints them through `say!`.
    std::fs::write(
        home.path().join("brave-search").join("council.json"),
        serde_json::json!({ "direct": [esc] }).to_string(),
    )
    .unwrap();
    let out = bx("openrouter", 1, home.path())
        .args(["council", "doctor"])
        .bounded_output();
    let doc = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(doc.contains("direct"), "doctor printed nothing: {doc:?}");
    assert!(!raw(doc.clone()), "escape reached stdout: {doc:?}");
}

#[test]
fn a_model_a_wire_does_not_serve_is_refused_before_any_member_runs() {
    let roster = |ids: &[&str]| {
        serde_json::json!({"data": ids.iter().map(|i| serde_json::json!({"id": i}))
            .collect::<Vec<_>>()})
        .to_string()
    };
    let served = roster(&["a/one", "b/two"]);
    let (port, seen) = serve(Arc::new(move |path: &str, body: &str| {
        if path.contains("/api/v1/key") {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        } else if path.contains("/v1/models") {
            (200, served.clone())
        } else if path.contains("/llm/context") {
            search_hit()
        } else {
            let _ = body;
            reply(&prose())
        }
    }));

    let bad = council(port, &["-m", "a/one,c/three", "--progress", "false"]);
    assert_eq!(bad.code(), 2, "{}", bad.stderr());
    assert!(bad.stderr().contains("c/three"), "{}", bad.stderr());
    // `seen` holds the pre-flight's own reads too; what must be absent is a generation.
    assert!(
        !seen
            .lock()
            .unwrap()
            .iter()
            .any(|b| b.contains("\"messages\"")),
        "a member was paid for before the roster was checked"
    );

    // A served panel still runs.
    assert_eq!(
        council(port, &["-m", "a/one,b/two", "--progress", "false"]).code(),
        0
    );
}

/// Near means equal ignoring separators, the usual slug-to-direct-id difference.
#[test]
fn the_refusal_names_the_near_miss_only_when_it_is_unambiguous() {
    let handler = |ids: Vec<String>| {
        Arc::new(move |path: &str, _: &str| {
            if path.contains("/api/v1/key") {
                (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
            } else if path.contains("/v1/models") {
                (
                    200,
                    serde_json::json!({"data": ids.iter().map(|i| serde_json::json!({"id": i}))
                        .collect::<Vec<_>>()})
                    .to_string(),
                )
            } else {
                reply(&prose())
            }
        }) as Handler
    };

    let one = serve(handler(vec!["a/one".into(), "a/model-5-1".into()]));
    let r = council(one.0, &["-m", "a/model-5.1,a/one", "--progress", "false"]);
    assert_eq!(r.code(), 2);
    assert!(
        r.stderr().contains("did you mean `a/model-5-1`"),
        "the one separator apart spelling was not offered: {}",
        r.stderr()
    );

    // Two entries collapse to the same loose form, so neither is *the* repair.
    let two = serve(handler(vec![
        "a/one".into(),
        "a/model-5-1".into(),
        "a/model_5_1".into(),
    ]));
    let r = council(two.0, &["-m", "a/model-5.1,a/one", "--progress", "false"]);
    assert_eq!(r.code(), 2);
    assert!(
        !r.stderr().contains("did you mean"),
        "guessed between two equally close spellings: {}",
        r.stderr()
    );
}

/// The check guards a wrong id, not provider health.
#[test]
fn an_unreadable_roster_lets_the_run_through() {
    for body in [r#"{"data":[]}"#, r#"{"error":"nope"}"#, "not json at all"] {
        let payload = body.to_string();
        let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
            if path.contains("/api/v1/key") {
                (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
            } else if path.contains("/v1/models") {
                (200, payload.clone())
            } else {
                reply(&prose())
            }
        }));
        let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
        assert_eq!(r.code(), 0, "roster {body:?} blocked a run: {}", r.stderr());
    }
}

#[test]
fn a_direct_wire_is_asked_about_the_bare_id_it_will_actually_be_sent() {
    let (port, _) = serve(Arc::new(|path: &str, _: &str| {
        if path.contains("/api/v1/key") {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        } else if path.contains("/v1/models") {
            // Bare ids, the way every direct API lists them.
            (200, r#"{"data":[{"id":"claude-good"}]}"#.into())
        } else {
            anthropic_reply(&prose())
        }
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("anthropic", port, dir.path())
        .args(["council", "SUBJECT", "--via", "anthropic"])
        .args(["-m", "anthropic/claude-good,anthropic/claude-bad"])
        .args(["--progress", "false", "--dry-run"])
        .bounded_output();
    let err = String::from_utf8_lossy(&out.stderr);

    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(
        err.contains("claude-bad"),
        "the member the wire cannot serve was not named: {err}"
    );
    assert!(
        !err.contains("claude-good"),
        "a member the wire does serve was refused, so the slug was checked and not the wire form: {err}"
    );
}

#[test]
fn the_anthropic_roster_is_requested_at_the_maximum_page_size_and_never_filtered_by_zdr() {
    let paths = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&paths);
    let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
        seen.lock().unwrap().push(path.to_string());
        (200, r#"{"data":[{"id":"one"}]}"#.into())
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("anthropic", port, dir.path())
        .args(["council", "SUBJECT", "--via", "anthropic", "-m", "a/one"])
        .args(["--tools", "none", "--dry-run"])
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(*paths.lock().unwrap(), ["/v1/models?limit=1000"]);
}

/// Anthropic lists dated ids only, and resolves the alias the request carries.
#[test]
fn an_anthropic_alias_absent_from_the_list_is_accepted_when_the_model_lookup_resolves_it() {
    let (port, _) = serve(Arc::new(|path: &str, _: &str| match path {
        "/v1/models/claude-sonnet-4-5" => (200, r#"{"id":"claude-sonnet-4-5-20250929"}"#.into()),
        p if p.starts_with("/v1/models/") => (404, r#"{"error":"x"}"#.into()),
        p if p.starts_with("/v1/models") => (
            200,
            r#"{"data":[{"id":"claude-sonnet-4-5-20250929"}]}"#.into(),
        ),
        _ => anthropic_reply(&prose()),
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("anthropic", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "anthropic",
            "--tools",
            "none",
        ])
        .args([
            "-m",
            "anthropic/claude-sonnet-4.5,anthropic/claude-nope",
            "--dry-run",
        ])
        .bounded_output();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("claude-nope"), "{err}");
    assert!(
        !err.contains("claude-sonnet-4-5"),
        "the alias was refused: {err}"
    );
}

/// Only a 404 says a model is not served; any other failed lookup fails open, as the list does.
#[test]
fn a_failed_alias_lookup_fails_open_unless_it_answers_404_or_401() {
    let (port, _) = serve(Arc::new(|path: &str, _: &str| match path {
        "/v1/models/claude-down" => (503, "{}".into()),
        "/v1/models/claude-locked" => (401, "{}".into()),
        p if p.starts_with("/v1/models/") => (404, "{}".into()),
        p if p.starts_with("/v1/models") => (200, r#"{"data":[{"id":"claude-other"}]}"#.into()),
        _ => anthropic_reply(&prose()),
    }));
    for (model, code) in [("claude-down", 0), ("claude-nope", 2), ("claude-locked", 3)] {
        let dir = TempDir::new().unwrap();
        let out = bx("anthropic", port, dir.path())
            .args([
                "council",
                "SUBJECT",
                "--via",
                "anthropic",
                "--tools",
                "none",
            ])
            .args(["-m", &format!("anthropic/{model}"), "--dry-run"])
            .bounded_output();
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(code), "{model}: {err}");
    }
}

/// Routing variants are not listed in `/models`; the base model is what is served.
#[test]
fn an_openrouter_routing_variant_is_judged_as_its_base_model() {
    let (port, _) = serve(Arc::new(|path: &str, _: &str| match path {
        p if p.contains("/v1/models") => {
            (200, r#"{"data":[{"id":"a/one"},{"id":"b/two"}]}"#.into())
        }
        _ => (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into()),
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "openrouter",
            "--tools",
            "none",
        ])
        .args(["-m", "a/one:nitro,b/two:exacto", "--dry-run"])
        .bounded_output();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{err}");
}

/// A model outside zero retention fails at the transport with no fallback.
#[test]
fn a_member_outside_the_zero_retention_roster_is_refused_and_the_opt_out_lets_it_through() {
    let (port, seen) = serve(Arc::new(|path: &str, _: &str| {
        if path.contains("/api/v1/key") {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        } else if path.contains("/v1/models") {
            // `b/two` is served, but only outside zero retention.
            let ids: &[&str] = if path.contains("zdr=true") {
                &["a/one"]
            } else {
                &["a/one", "b/two"]
            };
            (
                200,
                serde_json::json!({"data": ids.iter().map(|i| serde_json::json!({"id": i}))
                    .collect::<Vec<_>>()})
                .to_string(),
            )
        } else {
            reply(&prose())
        }
    }));

    let refused = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    assert_eq!(refused.code(), 2, "{}", refused.stderr());
    let err = refused.stderr();
    assert!(err.contains("b/two"), "{err}");
    assert!(
        err.contains("zero data retention") && err.contains(r#""zdr": false"#),
        "the refusal must name the policy and the opt-out: {err}"
    );
    assert!(
        !seen
            .lock()
            .unwrap()
            .iter()
            .any(|b| b.contains("\"messages\"")),
        "a member was paid for before the policy was checked"
    );

    // Opting out consults the unfiltered roster, where the same member is servable.
    let home = TempDir::new().unwrap();
    let cfg_dir = home.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(cfg_dir.join("council.json"), r#"{"zdr": false}"#).unwrap();
    let allowed = council_in(port, home.path(), Some(&home.path().join("run")), &[]);
    assert_eq!(
        allowed.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&allowed.stderr)
    );
}

/// A name the cache cannot place forces a live read.
#[test]
fn a_cached_roster_never_refuses_a_model_the_provider_has_since_added() {
    let served = Arc::new(Mutex::new(vec!["a/one".to_string(), "b/two".to_string()]));
    let roster = Arc::clone(&served);
    let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
        if path.contains("/api/v1/key") {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        } else if path.contains("/v1/models") {
            (
                200,
                serde_json::json!({"data": roster.lock().unwrap().iter()
                    .map(|i| serde_json::json!({"id": i})).collect::<Vec<_>>()})
                .to_string(),
            )
        } else {
            reply(&prose())
        }
    }));

    // One home for both runs, so the second sees the first's cache.
    let home = TempDir::new().unwrap();
    let first = council_in(port, home.path(), Some(&home.path().join("a")), &[]);
    assert_eq!(first.status.code(), Some(0));

    // The provider gains a model the cached list has never heard of.
    served.lock().unwrap().push("c/three".into());
    let second = council_in(
        port,
        home.path(),
        Some(&home.path().join("b")),
        &["-m", "a/one,c/three"],
    );
    assert_eq!(
        second.status.code(),
        Some(0),
        "refused on a stale roster instead of re-reading: {}",
        String::from_utf8_lossy(&second.stderr)
    );
}

/// `mandatory` refuses `none`; `supported_efforts` lists what the model accepts.
#[test]
fn a_model_that_would_reject_the_requested_effort_is_refused_before_anything_is_sent() {
    let (port, seen) = serve(Arc::new(|path: &str, _: &str| match path {
        p if p.contains("/api/v1/key") => {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        }
        p if p.contains("/v1/models") => (
            200,
            serde_json::json!({"data": [
                {"id": "a/one", "reasoning": {"mandatory": true,
                                              "supported_efforts": ["high", "low"]}},
                {"id": "b/two", "reasoning": {"mandatory": false,
                                              "supported_efforts": ["high", "low"]}}]})
            .to_string(),
        ),
        _ => panic!("nothing may be sent for a request the roster says is illegal"),
    }));
    let none = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--effort",
            "none",
            "--progress",
            "false",
        ],
    );
    assert_eq!(none.code(), 2, "{}", none.stderr());
    assert!(
        none.stderr().contains("reasoning is mandatory"),
        "{}",
        none.stderr()
    );

    let unsupported = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--effort",
            "max",
            "--progress",
            "false",
        ],
    );
    assert_eq!(unsupported.code(), 2, "{}", unsupported.stderr());
    assert!(
        unsupported
            .stderr()
            .contains("does not accept `--effort max`"),
        "{}",
        unsupported.stderr()
    );
    // It names the efforts the model takes.
    assert!(
        unsupported.stderr().contains("it takes: high, low"),
        "{}",
        unsupported.stderr()
    );
    assert!(
        seen.lock().unwrap().iter().all(|b| !b.contains("messages")),
        "a refusal must cost no generation"
    );
    // A supported effort passes: the gate refuses the request, not the model.
    let ok = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--effort",
            "high",
            "--dry-run",
            "--progress",
            "false",
        ],
    );
    assert_eq!(ok.code(), 0, "{}", ok.stderr());
}

#[test]
fn a_provider_ceiling_below_ours_lowers_the_escalation_target_rather_than_refusing() {
    let (port, _) = serve(Arc::new(|path: &str, _: &str| match path {
        p if p.contains("/api/v1/key") => {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        }
        p if p.contains("/v1/models") => (
            200,
            serde_json::json!({"data": [
                {"id": "a/one", "top_provider": {"max_completion_tokens": 32000}},
                {"id": "b/two"}]})
            .to_string(),
        ),
        _ => reply(&prose()),
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "openrouter",
            "-m",
            "a/one,b/two",
            "--dry-run",
        ])
        .bounded_output();
    let plan: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(plan["members"][0]["max_tokens_ceiling"], 32000, "{plan}");
    // The model that publishes nothing keeps the universal ceiling.
    assert_eq!(plan["members"][1]["max_tokens_ceiling"], 128_000, "{plan}");

    // Below the wire's floor no request is sendable, so this ceiling refuses.
    let (tiny, _) = serve(Arc::new(|path: &str, _: &str| match path {
        p if p.contains("/api/v1/key") => {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        }
        p if p.contains("/v1/models") => (
            200,
            serde_json::json!({"data": [
                {"id": "a/one", "top_provider": {"max_completion_tokens": 100}},
                {"id": "b/two"}]})
            .to_string(),
        ),
        _ => panic!("nothing may be sent for a request the roster says is illegal"),
    }));
    let refused = council_in(tiny, dir.path(), None, &["--dry-run"]);
    assert_eq!(refused.status.code(), Some(2));
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(err.contains("caps `a/one` at 100 output tokens"), "{err}");
}

/// OpenAI's own list is ids only.
#[test]
fn direct_openai_is_judged_against_the_public_openrouter_entry_for_its_model() {
    let (port, _) = serve(Arc::new(|path: &str, _: &str| match path {
        p if p.starts_with("/api/v1/models") => (
            200,
            serde_json::json!({"data": [
                {"id": "openai/gpt-6-astra", "top_provider": {"max_completion_tokens": 100_000},
                 "reasoning": {"mandatory": true, "supported_efforts": ["max", "high", "low"]}},
                {"id": "openai/gpt-5.5-pro",
                 "reasoning": {"supported_efforts": ["xhigh", "high", "medium"]}}]})
            .to_string(),
        ),
        p if p.starts_with("/v1/models") => (
            200,
            r#"{"data":[{"id":"gpt-6-astra"},{"id":"gpt-5.5-pro"}]}"#.into(),
        ),
        _ => reply(&prose()),
    }));
    let run = |model: &str, effort: &str| {
        let dir = TempDir::new().unwrap();
        let out = bx("openai", port, dir.path())
            .env(
                "BX_COUNCIL_OPENROUTER_BASE_URL",
                format!("http://127.0.0.1:{port}"),
            )
            .args(["council", "SUBJECT", "-m", model, "--effort", effort])
            .args(["--dry-run", "--progress", "false"])
            .bounded_output();
        (
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
            out.stdout,
        )
    };
    let (code, err, _) = run("direct:openai/gpt-6-astra", "none");
    assert_eq!(code, Some(2), "{err}");
    assert!(err.contains("reasoning is mandatory"), "{err}");
    let (code, err, _) = run("direct:openai/gpt-5.5-pro", "low");
    assert_eq!(code, Some(2), "{err}");
    assert!(err.contains("(it takes: xhigh, high, medium)"), "{err}");

    // `max` is advertised, so it is sent as `max`, and the public ceiling replaces ours.
    let (code, err, out) = run("direct:openai/gpt-6-astra", "max");
    assert_eq!(code, Some(0), "{err}");
    let plan: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(plan["members"][0]["max_tokens_ceiling"], 100_000, "{plan}");
}

/// Cached bare, OpenAI's ids would pass every effort until the cache expired.
#[test]
fn an_openai_roster_read_while_openrouter_is_down_is_not_cached() {
    let (port, _) = serve(Arc::new(|path: &str, _: &str| match path {
        p if p.starts_with("/api/v1/models") => (
            200,
            serde_json::json!({"data": [{"id": "openai/gpt-6-astra",
                "reasoning": {"mandatory": true, "supported_efforts": ["high"]}}]})
            .to_string(),
        ),
        p if p.starts_with("/v1/models") => (200, r#"{"data":[{"id":"gpt-6-astra"}]}"#.into()),
        _ => reply(&prose()),
    }));
    let (down, _) = serve(Arc::new(|_: &str, _: &str| (503, "{}".into())));
    let home = TempDir::new().unwrap();
    let run = |openrouter: u16| {
        bx("openai", port, home.path())
            .env(
                "BX_COUNCIL_OPENROUTER_BASE_URL",
                format!("http://127.0.0.1:{openrouter}"),
            )
            .args(["council", "SUBJECT", "-m", "direct:openai/gpt-6-astra"])
            .args(["--effort", "none", "--dry-run", "--progress", "false"])
            .bounded_output()
    };
    let open = run(down);
    let err = String::from_utf8_lossy(&open.stderr);
    assert_eq!(open.status.code(), Some(0), "fails open: {err}");
    let refused = run(port);
    let err = String::from_utf8_lossy(&refused.stderr);
    assert_eq!(refused.status.code(), Some(2), "{err}");
    assert!(err.contains("reasoning is mandatory"), "{err}");
}

#[test]
fn an_openai_model_found_only_by_lookup_is_judged_on_its_public_entry() {
    let (port, _) = serve(Arc::new(|path: &str, _: &str| match path {
        p if p.starts_with("/api/v1/models") => (
            200,
            serde_json::json!({"data": [{"id": "openai/gpt-6-astra",
                "reasoning": {"mandatory": true, "supported_efforts": ["high"]}}]})
            .to_string(),
        ),
        "/v1/models/gpt-6-astra" => (200, r#"{"id":"gpt-6-astra"}"#.into()),
        p if p.starts_with("/v1/models") => (200, r#"{"data":[{"id":"other"}]}"#.into()),
        _ => reply(&prose()),
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("openai", port, dir.path())
        .env(
            "BX_COUNCIL_OPENROUTER_BASE_URL",
            format!("http://127.0.0.1:{port}"),
        )
        .args(["council", "SUBJECT", "-m", "direct:openai/gpt-6-astra"])
        .args(["--effort", "none", "--dry-run", "--progress", "false"])
        .bounded_output();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("reasoning is mandatory"), "{err}");
}

#[test]
fn the_dry_run_publishes_what_the_request_will_carry_not_what_was_typed() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let out = bx("openai", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openai", "-m", "a/one,b/two"])
        .args(["--max-tokens", "4000", "--effort", "minimal", "--dry-run"])
        .bounded_output();
    let plan: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    // OpenAI's floor is 25,000.
    assert_eq!(plan["members"][0]["max_tokens"], 25_000, "{plan}");
    assert_eq!(
        plan["members"][0]["effort"], "low",
        "minimal is sent as low"
    );
    // The brief is task plus subject.
    assert!(plan["egress"]["brief_bytes"].as_u64().unwrap() > "SUBJECT".len() as u64);
}

/// Anthropic sends `none` as `low`; OpenRouter sends it as it is.
#[test]
fn the_dry_run_shows_each_member_the_effort_its_wire_sends() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let out = bx("anthropic", port, dir.path())
        .env(
            "BX_COUNCIL_OPENROUTER_BASE_URL",
            format!("http://127.0.0.1:{port}"),
        )
        .env("OPENROUTER_API_KEY", "sk-or-v1-mocktoken")
        .args([
            "council",
            "SUBJECT",
            "-m",
            "anthropic/claude-opus-5.5,a/one",
        ])
        .args(["--effort", "none", "--dry-run"])
        .bounded_output();
    let plan: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!("{e}: {}", String::from_utf8_lossy(&out.stderr));
    });
    let sent = |i: usize| {
        let m = &plan["members"][i];
        (m["transport"].clone(), m["effort"].clone())
    };
    assert_eq!(sent(0), ("anthropic".into(), "low".into()), "{plan}");
    assert_eq!(sent(1), ("openrouter".into(), "none".into()), "{plan}");
}

#[test]
fn two_runs_cannot_share_one_out_directory() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("shared");
    let run = |dir: &std::path::Path| {
        bx("openrouter", port, dir)
            .args([
                "council",
                "SUBJECT",
                "--via",
                "openrouter",
                "-m",
                "a/one,b/two",
            ])
            .args(["--progress", "false", "--out"])
            .arg(&out)
            .bounded_output()
    };
    let first = run(dir.path());
    assert_eq!(first.status.code(), Some(0));
    // Plant a live run's lock.
    std::fs::remove_file(out.join("manifest.json")).unwrap();
    std::fs::write(out.join(".council-lock"), "pid 1\n").unwrap();
    let concurrent = run(dir.path());
    assert_eq!(concurrent.status.code(), Some(2));
    let err = String::from_utf8_lossy(&concurrent.stderr);
    assert!(err.contains("another run holds"), "{err}");
    assert!(
        err.contains(".council-lock"),
        "the refusal names the file: {err}"
    );
}

/// A killed run's leftover `status.json` is not liveness.
#[test]
fn a_run_whose_status_has_gone_cold_is_not_listed_as_running() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    let run = root.join("20260905-120000-abc123");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("brief.md"), "b").unwrap();
    // No manifest, and a status a day older than twice its one-second clock.
    std::fs::write(
        run.join("status.json"),
        r#"{"done":1,"total":2,"max_seconds":1}"#,
    )
    .unwrap();
    age_by_days(&run.join("status.json"), 1);
    let out = bx_in(dir.path())
        .args(["council", "ls", "--format", "text"])
        .bounded_output();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("stalled"), "{text}");
}

#[test]
fn a_dry_run_refuses_every_out_path_the_real_run_would() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let taken = dir.path().join("finished");
    std::fs::create_dir_all(&taken).unwrap();
    std::fs::write(taken.join("manifest.json"), "{}").unwrap();
    for (what, path) in [
        ("the root", std::path::PathBuf::from("/")),
        ("a .git directory", dir.path().join(".git/x")),
    ] {
        let out = bx("openrouter", port, dir.path())
            .args([
                "council",
                "SUBJECT",
                "--via",
                "openrouter",
                "-m",
                "a/one,b/two",
                "--dry-run",
                "--out",
            ])
            .arg(&path)
            .bounded_output();
        assert_eq!(out.status.code(), Some(2), "a dry run blessed {what}");
    }
    // The lock is read, not taken: taking it would refuse the real run that follows.
    let held = dir.path().join("held");
    std::fs::create_dir_all(&held).unwrap();
    std::fs::write(held.join(".council-lock"), "pid 1\n").unwrap();
    // Nothing paid for yet, so the advice names only the live-run risk.
    let empty = council_in(port, dir.path(), Some(&held), &["--dry-run"]);
    let says = String::from_utf8_lossy(&empty.stderr).into_owned();
    assert!(says.contains("only if no run is live"), "{says}");
    std::fs::write(held.join("00-a_one.md"), "paid for").unwrap();
    let dry = council_in(port, dir.path(), Some(&held), &["--dry-run"]);
    assert_eq!(dry.status.code(), Some(2));
    let err = String::from_utf8_lossy(&dry.stderr);
    assert!(err.contains("another run holds"), "{err}");
    assert!(err.contains("would overwrite them"), "{err}");
    assert_eq!(
        std::fs::read_to_string(held.join(".council-lock")).unwrap(),
        "pid 1\n",
        "the dry run took the lock instead of reading it"
    );
}

#[cfg(unix)]
#[test]
fn a_dangling_lock_link_is_refused_by_the_dry_run_as_by_the_real_run() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("run");
    std::fs::create_dir_all(&out).unwrap();
    std::os::unix::fs::symlink(dir.path().join("gone"), out.join(".council-lock")).unwrap();
    for extra in [&["--dry-run"][..], &[]] {
        let r = council_in(port, dir.path(), Some(&out), extra);
        let err = String::from_utf8_lossy(&r.stderr);
        assert_eq!(r.status.code(), Some(2), "{extra:?}: {err}");
        assert!(err.contains("only if no run is live"), "{extra:?}: {err}");
    }
    assert!(seen.lock().unwrap().iter().all(|b| b.is_empty()));
}

#[test]
fn a_task_that_cannot_search_advertises_no_search_budget_and_auto_turns_it_back_on() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let tasks = dir.path().join("brave-search/council");
    std::fs::create_dir_all(&tasks).unwrap();
    std::fs::write(
        tasks.join("recall.md"),
        "---\ngrounded: false\n---\nAnswer.\n\n{doc}",
    )
    .unwrap();
    let plan = |extra: &[&str]| {
        let out = bx("openrouter", port, dir.path())
            .args([
                "council",
                "SUBJECT",
                "--via",
                "openrouter",
                "-m",
                "a/one,b/two",
            ])
            .args(["--task", "recall", "--dry-run", "--progress", "false"])
            .args(extra)
            .bounded_output();
        serde_json::from_slice::<serde_json::Value>(&out.stdout).unwrap()
    };
    assert_eq!(
        plan(&[])["limits"]["max_searches"],
        0,
        "a budget it cannot spend"
    );
    assert_eq!(plan(&[])["egress"]["max_search_queries"], 0);
    assert_eq!(plan(&["--tools", "auto"])["limits"]["max_searches"], 15);
    let _ = seen;
}

/// `audit` tells the model not to stop early, so it carries a longer clock; a flag still wins.
#[test]
fn a_task_carries_the_wall_clock_its_own_prompt_needs() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let seconds = |args: &[&str]| {
        let out = bx("openrouter", port, dir.path())
            .args([
                "council",
                "SUBJECT",
                "--via",
                "openrouter",
                "-m",
                "a/one,b/two",
                "--dry-run",
            ])
            .args(args)
            .bounded_output();
        serde_json::from_slice::<serde_json::Value>(&out.stdout).unwrap()["limits"]["max_seconds"]
            .as_u64()
            .unwrap()
    };
    let default = seconds(&[]);
    assert!(
        seconds(&["--task", "audit"]) > default,
        "audit needs longer"
    );
    assert_eq!(seconds(&["--task", "audit", "--max-seconds", "60"]), 60);
}

#[test]
fn every_task_owned_knob_yields_to_its_flag() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let tasks = dir.path().join("brave-search").join("council");
    std::fs::create_dir_all(&tasks).unwrap();
    std::fs::write(
        tasks.join("knobs.md"),
        "---\neffort: low\nmax_tokens: 2000\nmax_searches: 3\nmax_steps: 4\nmax_seconds: 70\n\
         grounded: false\n---\nbody {doc}",
    )
    .unwrap();
    let plan = |args: &[&str]| {
        let out = bx("openrouter", port, dir.path())
            .args([
                "council",
                "SUBJECT",
                "--task",
                "knobs",
                "--via",
                "openrouter",
            ])
            .args(["-m", "a/one,b/two", "--dry-run"])
            .args(args)
            .bounded_output();
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let l = &v["limits"];
        serde_json::json!([
            l["effort"],
            v["members"][0]["max_tokens"],
            l["max_searches"],
            l["max_steps"],
            l["max_seconds"],
        ])
    };
    assert_eq!(plan(&[]), serde_json::json!(["low", 2000, 0, 4, 70]));
    assert_eq!(
        plan(&[
            "--effort",
            "high",
            "--max-tokens",
            "3000",
            "--tools",
            "auto",
            "--max-searches",
            "5",
            "--max-steps",
            "9",
            "--max-seconds",
            "80",
        ]),
        serde_json::json!(["high", 3000, 5, 9, 80])
    );
}

/// A finite but absurd reported cost is clamped before it reaches the shared spend counter.
#[test]
fn one_absurd_reported_cost_cannot_pin_the_whole_panel() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": "stop", "message": {"content": prose()}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "cost": 1e305,
                          "is_byok": false}
            })
            .to_string(),
        )
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            // Between the clamped and the raw cost; serial, so the second sees the first's bill.
            "--max-usd",
            "20000",
            "--concurrency",
            "1",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    assert_eq!(r.code(), 0, "{}\n{}", r.stderr(), m);
    assert_eq!(m["quorum"]["delivered"], 2, "both members must run: {m}");
    for member in m["members"].as_array().unwrap() {
        let cost = &member["cost"];
        assert!(
            !cost["llm_usd"].is_null() || cost["source"] == "unknown",
            "a null price beside a claim it is known: {cost}"
        );
    }
    assert_eq!(m["cost"]["cost_unknown"], true, "{m}");
}

/// Brave bills only non-error responses.
#[test]
fn a_brave_call_that_was_refused_is_attempted_but_not_billed() {
    let (port, _) = serve(Arc::new(|path: &str, body: &str| match path {
        p if p.contains("/api/v1/key") => {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        }
        p if p.contains("/v1/models") => (200, r#"{"data":[]}"#.into()),
        p if p.contains("/llm/context") => (429, r#"{"error":"slow down"}"#.into()),
        _ if answered(body) => reply(&prose()),
        _ => tool_call("q"),
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    assert_eq!(r.code(), 0, "{}", r.stderr());
    assert!(
        m["brave"]["calls"].as_u64().unwrap() > 0,
        "a search was attempted: {m}"
    );
    assert_eq!(m["brave"]["sent"], 0, "a 429 is not a billable call: {m}");
}

#[test]
fn a_retried_turn_that_carried_text_still_leaves_it_on_disk() {
    let seen = Arc::new(Mutex::new(0usize));
    let counter = Arc::clone(&seen);
    let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":[]}"#.into());
        }
        let mut n = counter.lock().unwrap();
        *n += 1;
        // A 200 carrying a retryable provider error and a long reply.
        (
            200,
            serde_json::json!({
                "choices": [{"finish_reason": "error",
                             "message": {"content": format!("PAID DRAFT {}", "x".repeat(600))}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 900}
            })
            .to_string(),
        )
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--max-seconds",
            "20",
        ],
    );
    assert_ne!(r.code(), 0);
    let dir = r.dir.path().join("run");
    let written = std::fs::read_to_string(dir.join("00-a_one.md")).unwrap();
    assert!(
        written.contains("PAID DRAFT"),
        "the text the run was billed for reached no file:\n{written}"
    );
    // Rejected for the provider error, not as truncated.
    let m = r.manifest();
    let member = &m["members"][0];
    assert_eq!(member["status"], "rejected", "{m}");
    assert_eq!(member["reason"], "provider_error");
    assert_eq!(member["finish_reason"], "error");
    assert!(member["retries"].as_u64().unwrap() > 0, "{m}");
    assert!(member["chars"].as_u64().unwrap() > 600, "counted too: {m}");
}

#[test]
fn the_manifest_separates_the_cap_that_was_set_from_the_one_escalation_reached() {
    let (port, _) = serve(mock(|body| {
        if body.contains("\"max_completion_tokens\":1024") {
            (
                200,
                serde_json::json!({
                    "choices": [{"finish_reason": "length", "message": {"content": "cut"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1024}
                })
                .to_string(),
            )
        } else {
            reply(&prose())
        }
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-tokens",
            "1024",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    let limits = &m["members"][0]["limits"]["max_tokens"];
    assert_eq!(
        limits["limit"], 1024,
        "the configured cap must survive: {m}"
    );
    assert!(
        limits["reached"].as_u64().unwrap() > 1024,
        "and the escalated one must be named separately: {limits}"
    );
    assert_eq!(m["members"][0]["limits"]["max_tokens"]["escalations"], 1);
}

#[test]
fn the_notes_about_the_task_and_the_tools_respect_progress_false() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let tasks = dir.path().join("brave-search/council");
    std::fs::create_dir_all(&tasks).unwrap();
    std::fs::write(tasks.join("mine.md"), "Review.\n\n{doc}").unwrap();
    let run = |progress: &str| {
        let out = bx("openrouter", port, dir.path())
            .args([
                "council",
                "SUBJECT",
                "--via",
                "openrouter",
                "-m",
                "a/one,b/two",
            ])
            .args([
                "--task",
                "mine",
                "--tools",
                "none",
                "--dry-run",
                "--progress",
                progress,
            ])
            .bounded_output();
        String::from_utf8_lossy(&out.stderr).into_owned()
    };
    assert!(
        run("true").contains("loaded from"),
        "the note exists at all"
    );
    assert_eq!(run("false"), "", "--progress false must silence every note");
}

#[test]
fn two_spellings_that_reach_one_wire_id_are_one_member() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let out = bx("anthropic", port, dir.path())
        .args(["council", "SUBJECT", "--via", "anthropic", "--dry-run"])
        .args(["-m", "anthropic/claude-opus-5.5,claude-opus-5-5"])
        .bounded_output();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("duplicate models were merged"), "{err}");
}

/// Vendor SDK variables point other tools at routers; the council reads only its own.
#[test]
fn a_vendor_sdk_base_url_in_the_environment_does_not_redirect_the_council() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let out = bx_in(dir.path())
        .args([
            "council",
            "SUBJECT",
            "-m",
            "opus",
            "--tools",
            "none",
            "--dry-run",
        ])
        .env("ANTHROPIC_API_KEY", "sk-ant-mocktoken")
        .env("ANTHROPIC_BASE_URL", format!("http://127.0.0.1:{port}"))
        .env("BX_COUNCIL_ANTHROPIC_BASE_URL", "https://evil.example")
        .bounded_output();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(
        err.contains("BX_COUNCIL_ANTHROPIC_BASE_URL: anthropic base URL not allowed"),
        "{err}"
    );
    assert!(
        seen.lock().unwrap().is_empty(),
        "the vendor variable was read"
    );
}

/// An unfinished run holding a reply is kept; only the run-id format marks a run.
#[test]
fn prune_removes_old_runs_that_are_finished_or_hold_no_reply() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    // Age is faked by backdating the mtime.
    let age = |p: &std::path::Path, days: u64| {
        let f = std::fs::File::options().write(true).open(p).unwrap();
        let when = std::time::SystemTime::now() - std::time::Duration::from_secs(days * 86_400);
        f.set_times(std::fs::FileTimes::new().set_modified(when))
            .unwrap();
    };
    let make = |id: &str, finished: bool, age_days: u64| {
        let run = root.join(id);
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(run.join("brief.md"), "b").unwrap();
        if finished {
            std::fs::write(run.join("manifest.json"), "{}").unwrap();
        }
        age(&run.join("brief.md"), age_days);
    };
    make("20260101-120000-aaaaaa", true, 60); // finished and old
    make("20260101-120000-bbbbbb", true, 1); // finished and recent
    // Unfinished and old, with no reply: with and without a status.
    make("20260101-120000-cccccc", false, 60);
    make("20260101-120000-dddddd", false, 60);
    std::fs::write(
        root.join("20260101-120000-dddddd").join("status.json"),
        r#"{"done":0,"total":4,"max_seconds":900}"#,
    )
    .unwrap();
    make("20260101-120000-eeeeee", false, 60);
    {
        let d = root.join("20260101-120000-eeeeee");
        std::fs::write(d.join("00-a_one.md"), "a paid review").unwrap();
        std::fs::write(
            d.join("status.json"),
            r#"{"done":1,"total":4,"max_seconds":1}"#,
        )
        .unwrap();
        age(&d.join("status.json"), 60);
    }

    // Old non-runs. The decoy is a run id's exact width with a finished run's artifacts.
    std::fs::write(root.join("history.jsonl"), "{\"member\":\"a\"}\n").unwrap();
    age(&root.join("history.jsonl"), 60);
    let decoy = root.join("important-notes-backup");
    std::fs::create_dir_all(&decoy).unwrap();
    std::fs::write(decoy.join("brief.md"), "not a run").unwrap();
    std::fs::write(decoy.join("manifest.json"), "{}").unwrap();
    age(&decoy.join("brief.md"), 60);
    std::fs::create_dir_all(root.join("cache")).unwrap();

    let out = bx_in(dir.path())
        .args(["council", "prune"])
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let removed: Vec<&str> = v["removed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["run"].as_str().unwrap())
        .collect();
    assert_eq!(
        removed,
        [
            "20260101-120000-aaaaaa",
            "20260101-120000-cccccc",
            "20260101-120000-dddddd"
        ],
        "{v}"
    );
    assert!(!root.join("20260101-120000-aaaaaa").exists());
    assert!(
        root.join("20260101-120000-bbbbbb").exists(),
        "a recent run was removed"
    );
    assert!(
        !root.join("20260101-120000-cccccc").exists(),
        "an abandoned run past the window is exactly what this command is for"
    );
    assert!(
        !root.join("20260101-120000-dddddd").exists(),
        "a status with no reply is nothing owed"
    );
    assert!(
        root.join("20260101-120000-eeeeee").exists(),
        "an unfinished run holding a paid reply was removed"
    );
    assert!(
        root.join("20260101-120000-eeeeee")
            .join("00-a_one.md")
            .exists(),
        "the paid reply was destroyed"
    );
    assert!(
        root.join("history.jsonl").exists(),
        "the only record that a pruned run existed was deleted"
    );
    assert!(
        decoy.join("manifest.json").exists(),
        "a 22-character directory that is not a run id was reclaimed as one"
    );
    assert!(
        root.join("cache").exists(),
        "the roster cache directory itself is not prune's to remove"
    );
}

#[test]
fn status_takes_a_bare_run_id_as_well_as_a_path() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = bx("openrouter", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "openrouter",
            "-m",
            "a/one,b/two",
        ])
        .args(["--progress", "false"])
        .bounded_output();
    assert_eq!(
        run.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let id = serde_json::from_slice::<serde_json::Value>(&run.stdout).unwrap()["run"]
        .as_str()
        .unwrap()
        .to_string();
    let by_id = bx("openrouter", port, dir.path())
        .args(["council", "status", "--out", &id])
        .bounded_output();
    assert_eq!(
        by_id.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&by_id.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&by_id.stdout).unwrap();
    assert_eq!(
        (v["done"].as_u64(), v["total"].as_u64()),
        (Some(2), Some(2))
    );
}

/// Escalating the output cap cannot fix a full context window, so it is terminal.
#[test]
fn a_filled_context_window_is_terminal_and_names_its_own_lever() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({"stop_reason": "model_context_window_exceeded",
                "content": [{"type": "text", "text": "as far as it got"}],
                "usage": {"input_tokens": 10, "output_tokens": 10}})
            .to_string(),
        )
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("anthropic", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "anthropic",
            "-m",
            "a/one,b/two",
        ])
        .args(["--progress", "false", "--out"])
        .arg(dir.path().join("run"))
        .bounded_output();
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let first = &m["members"][0];
    assert_eq!(first["reason"], "truncated", "{m}");
    assert_eq!(first["terminated_by"], "context_window");
    assert!(
        first["error"].as_str().unwrap().contains("context window"),
        "{first}"
    );
    assert!(first["chars"].as_u64().unwrap() > 0, "{first}");

    // A complete answer held from an earlier turn is not stamped by this turn's stop.
    let (held_port, _) = serve(mock(|body: &str| {
        if body.contains("tool_result") {
            (
                200,
                serde_json::json!({"stop_reason": "model_context_window_exceeded",
                    "content": [{"type": "text", "text": "as far as it got"}],
                    "usage": {"input_tokens": 10, "output_tokens": 10}})
                .to_string(),
            )
        } else {
            // A trailing tool call forces a second turn.
            (
                200,
                serde_json::json!({"model": "claude-opus-5-5", "stop_reason": "tool_use",
                    "content": [{"type": "text", "text": prose()},
                                {"type": "tool_use", "id": "c1", "name": "search",
                                 "input": {"query": "q"}}],
                    "usage": {"input_tokens": 10, "output_tokens": 10}})
                .to_string(),
            )
        }
    }));
    let held_dir = TempDir::new().unwrap();
    let held_out = bx("anthropic", held_port, held_dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "anthropic",
            "-m",
            "a/one,b/two",
        ])
        .args(["--progress", "false", "--out"])
        .arg(held_dir.path().join("run"))
        .bounded_output();
    let held: serde_json::Value = serde_json::from_slice(&held_out.stdout).unwrap();
    let member = &held["members"][0];
    assert_eq!(member["status"], "delivered", "{held}");
    assert_eq!(member["terminated_by"], "answer", "{held}");
    assert_ne!(member["reason"], "truncated", "{held}");
    assert!(
        member["chars"].as_u64().unwrap_or(0) > 1_000,
        "the 16-character overflow published in place of the review: {member}"
    );
}

/// Every call needs one result, so a turn with hundreds of calls is refused whole.
#[test]
fn a_turn_with_absurdly_many_calls_is_refused_rather_than_answered() {
    let (port, _) = serve(mock(|_| {
        let calls: Vec<serde_json::Value> = (0..500)
            .map(|i| {
                serde_json::json!({"id": format!("c{i}"),
                    "function": {"name": "search", "arguments": "{\"query\":\"q\"}"}})
            })
            .collect();
        (
            200,
            serde_json::json!({"choices":[{"finish_reason":"tool_calls",
                "message":{"content": prose(), "tool_calls": calls}}],"usage":{}})
            .to_string(),
        )
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    assert_eq!(m["members"][0]["terminated_by"], "too_many_calls", "{m}");
    assert_eq!(
        m["brave"]["calls"], 0,
        "not one of them may be answered: {m}"
    );
    assert!(
        r.read("00-a_one.md").contains("## A. ANSWER"),
        "the paid review text the turn carried was lost"
    );
}

#[test]
fn a_search_that_finds_nothing_is_neither_grounding_nor_an_error() {
    let (port, _) = serve(Arc::new(|path: &str, body: &str| match path {
        p if p.contains("/api/v1/key") => {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        }
        p if p.contains("/v1/models") => (200, r#"{"data":[]}"#.into()),
        p if p.contains("/llm/context") => (200, r#"{"grounding":{"generic":[]}}"#.into()),
        _ if answered(body) => reply(&prose()),
        _ => tool_call("nothing on the web about this"),
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    assert_eq!(r.code(), 0, "{}", r.stderr());
    assert!(
        m["brave"]["sent"].as_u64().unwrap() > 0,
        "the call was made: {m}"
    );
    assert_eq!(m["brave"]["grounded"], 0, "nothing was found: {m}");
    assert_eq!(m["brave"]["errors"], 0, "and nothing failed: {m}");
    assert_eq!(m["members"][0]["status"], "delivered");
}

/// A direct wire reports no price, so a cap over an unrated direct member could never bind.
#[test]
fn a_cap_over_an_unrated_direct_member_is_refused_before_any_request() {
    let (port, seen) = serve(mock(|_| anthropic_reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = |extra: &[&str], config: Option<&str>| {
        let cfg_dir = dir.path().join("brave-search");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        let _ = std::fs::remove_file(cfg_dir.join("council.json"));
        if let Some(c) = config {
            std::fs::write(cfg_dir.join("council.json"), c).unwrap();
        }
        bx("anthropic", port, dir.path())
            .args(["council", "SUBJECT", "--via", "anthropic", "-m", "x/alpha"])
            .args(["--progress", "false", "--out"])
            .arg(dir.path().join("run"))
            .args(extra)
            .bounded_output()
    };
    for (extra, config) in [
        (&["--max-usd", "0.01"][..], None),
        (&["--max-usd", "0.01", "--dry-run"][..], None),
        (&[][..], Some(r#"{"max_usd": 0.01}"#)),
    ] {
        let out = run(extra, config);
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{extra:?} {config:?}: {err}");
        assert!(
            err.contains("x/alpha") && err.contains("\"rates\" in council.json"),
            "{err}"
        );
    }
    assert!(seen.lock().unwrap().is_empty(), "nothing may be sent");
    let rated = run(
        &["--max-usd", "0.01", "--dry-run"],
        Some(r#"{"rates": {"x/alpha": [1, 2]}}"#),
    );
    assert_eq!(rated.status.code(), Some(0), "a rated member is capped");
}

#[test]
fn every_free_command_has_a_human_form_and_it_says_the_same_thing() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let free = |args: &[&str]| {
        bx("openrouter", port, dir.path())
            .args(["council"])
            .args(args)
            .bounded_output()
    };
    let text = |args: &[&str]| {
        let out = bx("openrouter", port, dir.path())
            .args(["council"])
            .args(args)
            .args(["--format", "text"])
            .bounded_output();
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let empty = text(&["ls"]);
    assert!(empty.contains("no runs yet"), "{empty}");
    assert!(
        empty.contains("council"),
        "and where they will land: {empty}"
    );

    // `tasks` marks the default.
    let tasks = text(&["tasks"]);
    assert!(tasks.contains('\u{2605}'), "{tasks}");
    assert!(tasks.contains("design-review"), "{tasks}");
    assert!(tasks.contains("audit"), "{tasks}");

    let pruned = text(&["prune"]);
    assert!(pruned.contains("0 run(s) removed"), "{pruned}");
    assert!(
        pruned.contains("30 days"),
        "the window is part of the answer: {pruned}"
    );

    let run = bx("openrouter", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "openrouter",
            "-m",
            "a/one,b/two",
        ])
        .args(["--progress", "false"])
        .bounded_output();
    assert_eq!(
        run.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(text(&["ls"]).contains("latest"), "the newest run is marked");
    let status = text(&["status"]);
    assert!(status.contains("2/2"), "{status}");
    assert!(
        status.contains('\u{2713}'),
        "and finished, not running: {status}"
    );
    // Names the run it read: without `--out`, "the newest" can change mid-poll.
    assert!(status.contains("/bx/council/2026"), "{status}");
    let json: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&free(&["status"]).stdout)).unwrap();
    assert_eq!(json["phase"], "ok", "{json}");
    assert_eq!(json["done"], 2);
    assert!(
        json["run"].as_str().is_some_and(|r| r.starts_with("2026")),
        "{json}"
    );
    // `status` and `ls` agree about one run.
    let listed: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&free(&["ls"]).stdout)).unwrap();
    assert_eq!(listed["runs"][0]["phase"], json["phase"]);
    assert_eq!(listed["runs"][0]["run"], json["run"]);
    // No verdict yet: the key is absent, not null.
    assert!(listed["runs"][0].get("useful").is_none(), "{listed}");

    // `set-key` is exempt: it is interactive and writes nothing to stdout.
    for word in [
        "ls", "status", "tasks", "prune", "doctor", "models", "findings",
    ] {
        let out = text(&[word]);
        assert!(!out.is_empty(), "`{word} --format text` printed nothing");
        assert!(
            serde_json::from_str::<serde_json::Value>(&out).is_err(),
            "`{word} --format text` printed JSON: {out}"
        );
    }
    let scored = text(&["useful-yes"]);
    assert!(scored.starts_with("useful "), "{scored}");
    assert!(scored.contains("/bx/council/2026"), "{scored}");
    let after: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&free(&["ls"]).stdout)).unwrap();
    assert_eq!(after["runs"][0]["useful"], true, "{after}");
    assert!(
        after["runs"][0]["dir"].as_str().is_some(),
        "a sibling key still renders: {after}"
    );
}

#[test]
fn doctor_renders_every_shape_a_configuration_can_take() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(
        cfg_dir.join("council.json"),
        // `openai/unrated` routes direct and has no rate.
        r#"{"zdr": false, "direct": ["openai"],
            "panels": {"default": ["openai/unrated", "y/beta"]},
            "openrouter_api_key": "sk-ant-wrong-provider"}"#,
    )
    .unwrap();
    let out = bx_in(dir.path())
        .args(["council", "doctor", "--format", "text"])
        .bounded_output();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("not requested"), "zdr off: {text}");
    // With zdr off, OpenAI is sent store:true, which opts into storage.
    assert!(text.contains("sent store:true"), "zdr off: {text}");
    assert!(
        text.contains("direct      openai"),
        "the configured list: {text}"
    );
    assert!(
        text.contains("another provider's prefix"),
        "a key in the wrong slot is a fault, not a tick: {text}"
    );
    assert!(
        text.contains("no rate — --max-usd cannot bound"),
        "an unpriceable direct member says so: {text}"
    );
    assert_eq!(out.status.code(), Some(2), "{text}");
}

/// The text describes the current setting, so no key may claim it is the zdr one.
#[test]
fn doctor_json_retention_detail_matches_the_zdr_setting_it_reports() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(cfg_dir.join("council.json"), r#"{"zdr": false}"#).unwrap();
    let out = bx_in(dir.path())
        .args(["council", "doctor"])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let openai = v["retention"]["by_transport"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["transport"] == "openai")
        .unwrap_or_else(|| panic!("{v}"));
    assert!(
        openai["detail"].as_str().unwrap().contains("store:true"),
        "{openai}"
    );
    assert!(openai.get("under_zdr").is_none(), "{openai}");
}

/// A refusal sends no generation.
#[test]
fn the_credit_probe_maps_each_answer_to_the_code_a_member_would_have_produced() {
    let err = r#"{"error":"x"}"#;
    for (status, body, want, needle) in [
        (401, err, 3, "rejected the API key"),
        (403, err, 3, "rejected the API key"),
        // A negative account balance: a 402, with `limit_remaining` untouched.
        (
            402,
            r#"{"error":{"message":"Insufficient credits","code":402}}"#,
            8,
            "credit exhausted",
        ),
        (
            200,
            r#"{"data":{"limit":10,"limit_remaining":0}}"#,
            8,
            "credit exhausted",
        ),
        // An uncapped key, not an exhausted one.
        (
            200,
            r#"{"data":{"limit":null,"limit_remaining":null}}"#,
            0,
            "",
        ),
        // Any other answer is not a refusal.
        (500, err, 0, ""),
    ] {
        let (port, seen) = serve(Arc::new(move |path: &str, _: &str| match path {
            p if p.contains("/api/v1/key") => (status, body.into()),
            p if p.contains("/v1/models") => (200, r#"{"data":[]}"#.into()),
            p if p.contains("/llm/context") => search_hit(),
            _ => reply(&prose()),
        }));
        let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
        assert_eq!(r.code(), want, "a {status} {body} probe: {}", r.stderr());
        if want != 0 {
            assert!(r.stderr().contains(needle), "{}", r.stderr());
            assert!(
                seen.lock().unwrap().iter().all(|b| !b.contains("messages")),
                "a {status} {body} probe was followed by a generation"
            );
        }
    }
}

#[test]
fn an_all_direct_panel_sends_no_credit_probe() {
    let paths = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&paths);
    let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
        seen.lock().unwrap().push(path.to_string());
        match path {
            p if p.contains("/v1/models") => (200, r#"{"data":[]}"#.into()),
            _ => anthropic_reply(&prose()),
        }
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("anthropic", port, dir.path())
        .env(
            "BX_COUNCIL_OPENROUTER_BASE_URL",
            format!("http://127.0.0.1:{port}"),
        )
        .args([
            "council",
            "SUBJECT",
            "--via",
            "anthropic",
            "-m",
            "a/one,b/two",
        ])
        .args(["--tools", "none", "--progress", "false"])
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let paths = paths.lock().unwrap();
    assert!(
        !paths.iter().any(|p| p.contains("/api/v1/key")),
        "{paths:?}"
    );
}

/// A pre-flight carries the key, so a redirect is answered, never followed.
#[test]
fn preflight_requests_do_not_follow_redirects() {
    let (target, seen) = serve(mock(|_| reply(&prose())));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for mut sock in listener.incoming().flatten() {
            let _ = read_request(&mut sock);
            let _ = sock.write_all(
                format!(
                    "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{target}/api/v1/models\r\n\
                     Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
                .as_bytes(),
            );
        }
    });
    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "openrouter",
            "-m",
            "a/one,b/two",
        ])
        .args(["--tools", "none", "--dry-run"])
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(seen.lock().unwrap().is_empty(), "the redirect was followed");
}

/// A 403 fails open: an OpenAI key without model-read scope still infers. So does a 5xx.
#[test]
fn a_direct_roster_401_refuses_with_exit_3_and_a_403_fails_open() {
    for (status, want) in [(401, 3), (403, 0), (500, 0)] {
        let posts = Arc::new(Mutex::new(0));
        let counted = Arc::clone(&posts);
        let (port, _) = serve(Arc::new(move |path: &str, _: &str| match path {
            p if p.contains("/v1/models") => (status, r#"{"error":"x"}"#.into()),
            _ => {
                *counted.lock().unwrap() += 1;
                anthropic_reply(&prose())
            }
        }));
        let dir = TempDir::new().unwrap();
        let out = bx("anthropic", port, dir.path())
            .args([
                "council",
                "SUBJECT",
                "--via",
                "anthropic",
                "-m",
                "a/one,b/two",
            ])
            .args(["--tools", "none", "--progress", "false"])
            .bounded_output();
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(want), "{status}: {err}");
        if want == 3 {
            assert!(
                err.contains("anthropic rejected the API key (401)"),
                "{err}"
            );
            assert_eq!(*posts.lock().unwrap(), 0, "an inference request was sent");
        }
    }
}

/// The member deadline starts after the subject is read, so stdin needs its own bound.
#[test]
fn stdin_is_bounded_by_the_same_clock_a_file_subject_is() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let mut child = bx("openrouter", port, dir.path())
        .args(["council", "-", "--via", "openrouter", "-m", "a/one,b/two"])
        .args(["--dry-run", "--max-seconds", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Held open and silent: a plain read never returns.
    let held = child.stdin.take().expect("a piped stdin");
    let started = std::time::Instant::now();
    let out = finish(child);
    drop(held);
    assert!(
        started.elapsed() < std::time::Duration::from_secs(20),
        "a silent stdin was never given up on"
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("stdin was not readable"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // A writer that finishes is read whole under the same clock.
    let mut child = bx("openrouter", port, dir.path())
        .args(["council", "-", "--via", "openrouter", "-m", "a/one,b/two"])
        .args(["--dry-run", "--max-seconds", "2", "--progress", "false"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"REVIEW THIS SUBJECT\n")
        .unwrap();
    let out = finish(child);
    assert_eq!(out.status.code(), Some(0));
    let plan: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(
        plan["egress"]["brief_bytes"].as_u64().unwrap() > "REVIEW THIS SUBJECT".len() as u64,
        "{plan}"
    );
}

#[test]
fn a_truncation_with_no_step_left_to_escalate_into_is_reported_as_truncated() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "choices": [{"finish_reason": "length", "message": {"content": "cut off here"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10}
            })
            .to_string(),
        )
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--max-steps",
            "1",
            "--progress",
            "false",
        ],
    );
    let first = &r.manifest()["members"][0];
    assert_eq!(first["reason"], "truncated", "{first}");
    assert_eq!(first["finish_reason"], "length");
    // The raised cap was never sent, so it is not published.
    let limits = &first["limits"]["max_tokens"];
    assert_eq!(limits["limit"], limits["reached"], "{limits}");
}

#[test]
fn a_lost_chain_is_replayed_statelessly_rather_than_retried() {
    let seen = Arc::new(Mutex::new(0usize));
    let counter = Arc::clone(&seen);
    let (port, bodies) = serve(Arc::new(move |path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":[]}"#.into());
        }
        let mut n = counter.lock().unwrap();
        *n += 1;
        if *n == 1 {
            (
                200,
                serde_json::json!({"status": "completed", "id": "resp-1", "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "short"}]}]})
                .to_string(),
            )
        } else if *n == 2 {
            (
                200,
                serde_json::json!({"error": {"code": "previous_response_not_found",
                                             "message": "gone"}})
                .to_string(),
            )
        } else {
            (
                200,
                serde_json::json!({"status": "completed", "id": "resp-3", "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": prose()}]}]})
                .to_string(),
            )
        }
    }));
    let dir = TempDir::new().unwrap();
    // ZDR (on by default) sends every OpenAI request stateless, leaving no chain to lose.
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(cfg_dir.join("council.json"), r#"{"zdr": false}"#).unwrap();
    let out = bx("openai", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openai", "-m", "a/one,b/two"])
        .args(["--progress", "false", "--out"])
        .arg(dir.path().join("run"))
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let sent = bodies.lock().unwrap().clone();
    assert!(
        sent.iter().any(|b| b.contains("\"previous_response_id\"")),
        "no chained request was ever sent: {sent:?}"
    );
    assert!(
        sent.iter()
            .any(|b| b.contains("\"store\":false") && !b.contains("\"previous_response_id\"")),
        "no stateless replay was sent: {sent:?}"
    );
}

#[test]
fn a_default_zdr_openai_member_sends_store_false_and_never_chains() {
    let n = Arc::new(Mutex::new(0usize));
    let (port, bodies) = serve(Arc::new(move |path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":[]}"#.into());
        }
        let mut n = n.lock().unwrap();
        *n += 1;
        // A short first answer forces the second round a chain would reference.
        let text = if *n == 1 {
            "short".to_string()
        } else {
            prose()
        };
        (
            200,
            serde_json::json!({"status": "completed", "id": format!("resp-{n}"), "output": [
                {"type": "message", "content": [{"type": "output_text", "text": text}]}]})
            .to_string(),
        )
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("openai", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openai", "-m", "a/one"])
        .args(["--progress", "false", "--out"])
        .arg(dir.path().join("run"))
        .bounded_output();
    assert_eq!(out.status.code(), Some(0));
    let sent: Vec<String> = bodies
        .lock()
        .unwrap()
        .iter()
        .filter(|b| !b.is_empty())
        .cloned()
        .collect();
    assert!(sent.len() >= 2, "{sent:?}");
    assert!(
        sent.iter()
            .all(|b| b.contains("\"store\":false") && !b.contains("\"previous_response_id\"")),
        "{sent:?}"
    );
}

#[test]
fn a_missing_key_names_the_member_the_variable_and_the_next_command() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "-m",
            "anthropic/claude-opus-5.5,b/two",
            "--dry-run",
        ])
        .bounded_output();
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    for needle in [
        "no API key for anthropic",
        "anthropic/claude-opus-5.5",
        "ANTHROPIC_API_KEY",
        "bx council doctor",
    ] {
        assert!(err.contains(needle), "{needle:?} missing from: {err}");
    }
}

/// A control character in either header is refused before anything is sent.
#[test]
fn the_configured_referer_and_title_reach_the_request() {
    let (port, bodies, heads) = serve_with_heads(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(
        cfg_dir.join("council.json"),
        r#"{"referer": "https://example.test/", "title": "my council"}"#,
    )
    .unwrap();
    let out = bx("openrouter", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "openrouter",
            "-m",
            "a/one,b/two",
        ])
        .args(["--progress", "false"])
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!bodies.lock().unwrap().is_empty(), "no request was sent");
    // Every chat request carries both; the lowercase match leaves header-name case to ureq.
    let heads = heads.lock().unwrap().clone();
    let chats: Vec<&String> = heads
        .iter()
        .filter(|h| h.contains("/chat/completions"))
        .collect();
    assert!(!chats.is_empty(), "no chat request was sent");
    for h in chats {
        let h = h.to_ascii_lowercase();
        assert!(h.contains("http-referer: https://example.test/"), "{h}");
        assert!(h.contains("x-title: my council"), "{h}");
    }

    std::fs::write(cfg_dir.join("council.json"), "{\"title\": \"two\\nlines\"}").unwrap();
    let refused = bx("openrouter", port, dir.path())
        .args([
            "council",
            "SUBJECT",
            "--via",
            "openrouter",
            "-m",
            "a/one,b/two",
            "--dry-run",
        ])
        .bounded_output();
    assert_eq!(refused.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("control character"),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );
}

#[test]
fn a_dry_run_reports_what_past_runs_of_this_panel_were_charged() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    std::fs::create_dir_all(&root).unwrap();
    // The mock's empty roster lets the invented models through.
    let (port, _guard) = serve(mock(|_| reply(&prose())));
    let plan = |extra: &[&str]| {
        let out = bx("openrouter", port, dir.path())
            .args(["council", "SUBJECT", "--dry-run", "--tools", "none"])
            .args(["-m", "a/one,b/two", "--via", "openrouter"])
            .args(extra)
            .bounded_output();
        serde_json::from_slice::<serde_json::Value>(&out.stdout).unwrap_or_else(|e| {
            panic!("{e}: {}", String::from_utf8_lossy(&out.stdout));
        })
    };

    // No history: `observed` is absent, not null; "no cap" is a known null.
    let cold = plan(&[]);
    assert!(cold["cost"].get("observed").is_none(), "{}", cold["cost"]);
    assert!(cold["cost"]["max_usd"].is_null(), "{}", cold["cost"]);

    // Three priced runs: the floor for a p90.
    let rows: Vec<String> = (0..3)
        .flat_map(|i| {
            [("a/one", 1.0), ("b/two", 2.0)].map(move |(m, c)| {
                format!(
                    r#"{{"run":"r{i}","dir":"/x","model":"{m}","cost_usd":{c},"cost_source":"reported"}}"#
                )
            })
        })
        .collect();
    std::fs::write(root.join("history.jsonl"), rows.join("\n") + "\n").unwrap();

    let warm = plan(&[]);
    let seen = &warm["cost"]["observed"];
    assert_eq!(seen["run_total"]["n"], 3, "{seen}");
    assert_eq!(
        seen["run_total"]["p50"], 3.0,
        "both members, per run: {seen}"
    );
    assert_eq!(seen["members"][0]["model"], "a/one");
    assert_eq!(seen["members"][0]["p90"], 1.0, "{seen}");
    // History carries no token counts, so no rate is derivable from it.
    assert!(warm["cost"].get("priced").is_none(), "{}", warm["cost"]);
    assert!(
        seen["note"].as_str().unwrap().contains("not a bound"),
        "{seen}"
    );
    // Every row was read, so no window is mentioned.
    assert!(seen.get("window").is_none(), "{seen}");

    // The text form leads with the same figure.
    let out = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--dry-run", "--tools", "none"])
        .args([
            "-m",
            "a/one,b/two",
            "--via",
            "openrouter",
            "--format",
            "text",
        ])
        .bounded_output();
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let (obs, worst) = (text.find("observed "), text.find("worst  "));
    assert!(obs.is_some() && obs < worst, "{text}");
    assert!(text.contains("$3.00 median"), "{text}");

    // Other panels' history is ignored.
    std::fs::write(
        root.join("history.jsonl"),
        rows.join("\n")
            .replace("a/one", "y/other")
            .replace("b/two", "z/other")
            + "\n",
    )
    .unwrap();
    assert!(plan(&[])["cost"].get("observed").is_none());
}

#[test]
fn observed_ignores_history_rows_from_a_different_task() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    std::fs::create_dir_all(&root).unwrap();
    let rows: String = (0..3)
        .map(|i| {
            format!(
                r#"{{"run":"r{i}","task":"quick","model":"a/one","cost_usd":1,"cost_source":"reported"}}"#
            ) + "\n"
        })
        .collect();
    std::fs::write(root.join("history.jsonl"), rows).unwrap();
    let (port, _) = serve(mock(|_| reply(&prose())));
    let out = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--dry-run", "--tools", "none"])
        .args([
            "-m",
            "a/one",
            "--via",
            "openrouter",
            "--task",
            "design-review",
        ])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["cost"].get("observed").is_none(), "{}", v["cost"]);
}

/// Member file names are re-joined under the run directory, never trusted as written.
#[test]
fn findings_splits_each_reply_and_reads_the_member_list_from_the_manifest() {
    let (port, _) = serve(mock(|_| {
        reply(&format!(
            "## A. FATAL\n{}\n\n## E. VERDICT\ndo not ship\n",
            "the breaking condition. ".repeat(20)
        ))
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let dir = r.manifest()["dir"].as_str().unwrap().to_string();
    let out = bx_in(r.dir.path())
        .args(["council", "findings", "--out", &dir])
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let f: serde_json::Value = serde_json::from_slice(&out.stdout).expect("findings is JSON");
    assert_eq!(f["task"], "design-review");
    assert_eq!(f["members"][0]["model"], "a/one");
    assert_eq!(f["members"][0]["status"], "delivered");
    let heads: Vec<&str> = f["members"][0]["sections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["heading"].as_str().unwrap())
        .collect();
    assert_eq!(heads, ["A. FATAL", "E. VERDICT"], "{f}");
    let text = f["members"][0]["sections"][0]["text"].as_str().unwrap();
    assert!(text.starts_with("the breaking condition"), "{text}");

    let outside = r.dir.path().join("secret.md");
    std::fs::write(
        &outside,
        "## A. FATAL\nCONTENTS-OF-A-FILE-OUTSIDE-THE-RUN\n",
    )
    .unwrap();
    let path = std::path::Path::new(&dir).join("manifest.json");
    let mut planted: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    planted["members"][0]["file"] = outside.display().to_string().into();
    std::fs::write(&path, planted.to_string()).unwrap();
    let out = bx_in(r.dir.path())
        .args(["council", "findings", "--out", &dir])
        .bounded_output();
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("CONTENTS-OF-A-FILE-OUTSIDE-THE-RUN"),
        "a planted manifest read a file outside the run directory"
    );

    // Manifest strings are scrubbed too: a JSON-escaped ESC is live again once parsed.
    let mut hostile: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    hostile["task"] = "\u{1b}[2Jcleared".into();
    hostile["members"][0]["model"] = "\u{1b}[2Jm".into();
    std::fs::write(&path, hostile.to_string()).unwrap();
    let out = bx_in(r.dir.path())
        .args(["council", "findings", "--out", &dir])
        .bounded_output();
    assert!(
        !out.stdout.contains(&0x1b),
        "a planted manifest put a live escape on stdout"
    );
    let f: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(f["task"], "cleared", "{f}");
    assert_eq!(f["members"][0]["model"], "m", "{f}");
    // A field the manifest omits stays omitted, not null.
    assert!(f["members"][0].get("reason").is_none(), "{f}");

    let empty = TempDir::new().unwrap();
    let out = bx_in(empty.path())
        .args(["council", "findings", "--out"])
        .arg(empty.path())
        .bounded_output();
    assert_eq!(out.status.code(), Some(2));
    // The whole sentence: a bare "manifest.json" would also match an ENOENT message.
    assert!(
        String::from_utf8_lossy(&out.stderr)
            .contains("no manifest.json and no member replies — not a readable run"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Read from its reply files; what only the manifest names stays absent.
#[test]
fn findings_reads_a_run_that_died_before_its_manifest() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let dir = r.dir.path().join("run");
    // A fresh `status.json` would read as a live run, so it goes too.
    std::fs::remove_file(dir.join("manifest.json")).unwrap();
    std::fs::remove_file(dir.join("status.json")).unwrap();

    let out = bx_in(r.dir.path())
        .args(["council", "findings", "--out"])
        .arg(&dir)
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let f: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(f["members"].as_array().unwrap().len(), 2, "{f}");
    for i in 0..2 {
        assert!(
            f["members"][i]["file"].as_str().unwrap().ends_with(".md"),
            "{f}"
        );
        assert_eq!(
            f["members"][i]["sections"][0]["heading"], "A. ANSWER",
            "{f}"
        );
    }
    assert!(
        f["members"][0]["file"]
            .as_str()
            .unwrap()
            .contains("00-a_one"),
        "panel order is the filename's own: {f}"
    );
    // A killed run leaves a headerless `.md.partial`: flushed per turn, renamed on completion.
    let done = std::fs::read_to_string(dir.join("01-b_two.md")).unwrap();
    let (_, draft) = done.split_once("-->\n\n").unwrap();
    std::fs::write(dir.join("01-b_two.md.partial"), draft).unwrap();
    std::fs::remove_file(dir.join("01-b_two.md")).unwrap();
    let out2 = bx_in(r.dir.path())
        .args(["council", "findings", "--out"])
        .arg(&dir)
        .bounded_output();
    let f2: serde_json::Value = serde_json::from_slice(&out2.stdout).unwrap();
    assert_eq!(f2["members"].as_array().unwrap().len(), 2, "{f2}");
    assert!(
        f2["members"][1]["file"]
            .as_str()
            .unwrap()
            .ends_with(".md.partial"),
        "{f2}"
    );
    assert_eq!(
        f2["members"][1]["sections"][0]["heading"], "A. ANSWER",
        "{f2}"
    );
    // Unrecoverable from the file name (`slugify` maps `/` to `_`), so absent.
    for field in ["model", "status", "reason"] {
        assert!(
            f["members"][0].get(field).is_none(),
            "{field} was invented for a run that never wrote a manifest: {f}"
        );
    }
    assert!(f["task"].is_null(), "{f}");
    assert_eq!(f["phase"], "incomplete", "{f}");
}

#[test]
fn findings_publishes_the_path_of_each_reply() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let dir = r.dir.path().join("run");
    let out = bx_in(r.dir.path())
        .args(["council", "findings", "--out"])
        .arg(&dir)
        .bounded_output();
    let f: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let file = f["members"][0]["file"].as_str().unwrap();
    assert!(file.ends_with("00-a_one.md"), "{f}");
    assert!(std::path::Path::new(file).exists(), "{file}");
    assert_eq!(f["phase"], "ok", "{f}");
}

/// A tab or newline in a manifest field cannot add a row or a column to the text form.
#[test]
fn findings_text_form_cannot_be_given_extra_rows_by_a_manifest_field() {
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("00-x.md"), "## A\nx\n").unwrap();
    std::fs::write(
        run.join("manifest.json"),
        r#"{"members":[{"model":"a\tok\nforged","status":"ok","file":"00-x.md"}]}"#,
    )
    .unwrap();
    let out = bx_in(dir.path())
        .args(["council", "findings", "--format", "text", "--out"])
        .arg(&run)
        .bounded_output();
    let text = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{text}");
    let rows: Vec<&str> = text.lines().filter(|l| !l.starts_with(' ')).collect();
    assert_eq!(rows, ["a ok forged\tok"], "{text}");
}

/// A verdict is append-only, so an unresolvable target is refused before anything is written.
#[test]
fn scoring_a_run_that_does_not_exist_refuses_rather_than_appending_an_orphan() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    std::fs::create_dir_all(&root).unwrap();
    let bare = |args: &[&str]| {
        bx_in(dir.path())
            .args(["council"])
            .args(args)
            .bounded_output()
    };
    let refused = |args: &[&str], why: &str| {
        let out = bare(args);
        assert_eq!(out.status.code(), Some(2), "{args:?} was not refused");
        let e = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(e.contains(why), "{args:?}: wanted {why:?}, got {e}");
        assert!(
            !root.join("scores.jsonl").exists(),
            "{args:?} appended an orphan"
        );
    };

    // The bare word is reserved to refuse: as a subject it is a paid four-member panel.
    refused(&["useful"], "useful-yes");
    refused(&["useful-yes"], "no runs under");
    // A name that is not a run id stays a path.
    refused(
        &["useful-yes", "--out", "important-notes-backup"],
        "is not a run directory",
    );
    refused(
        &["useful-yes", "--out", "../escape"],
        "is not a run directory",
    );

    // No status.json is not a run; a live one cannot be judged yet.
    let run = root.join("20260907-120000-abcdef");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("brief.md"), "x").unwrap();
    refused(&["useful-yes"], "status.json");
    std::fs::write(
        run.join("status.json"),
        r#"{"ts":"now","done":1,"total":2}"#,
    )
    .unwrap();
    refused(&["useful-yes"], "or stopped too recently to tell");

    std::fs::write(
        run.join("manifest.json"),
        r#"{"state":"ok","quorum":{"delivered":2,"total":2}}"#,
    )
    .unwrap();
    let dry = bare(&["useful-no", "--dry-run"]);
    assert_eq!(dry.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&dry.stdout).contains("\"useful\":false"));
    assert!(
        !root.join("scores.jsonl").exists(),
        "--dry-run wrote a verdict"
    );
    // Both verdicts: a dry-run guard keyed on `useful` would let one through.
    let dry = bare(&["useful-yes", "20260907-120000-abcdef", "--dry-run"]);
    assert_eq!(dry.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&dry.stdout).contains("\"useful\":true"));
    assert!(
        !root.join("scores.jsonl").exists(),
        "useful-yes --dry-run wrote a verdict"
    );

    assert_eq!(bare(&["useful-no"]).status.code(), Some(0));
    let listed: serde_json::Value =
        serde_json::from_slice(&bare(&["ls"]).stdout).expect("ls is JSON");
    assert_eq!(listed["runs"][0]["useful"], false, "{listed}");
    let said =
        String::from_utf8_lossy(&bare(&["useful-yes", "--format", "text"]).stdout).into_owned();
    assert!(said.starts_with("useful "), "{said}");
    let ls = String::from_utf8_lossy(&bare(&["ls", "--format", "text"]).stdout).into_owned();
    assert!(ls.contains(" · useful"), "{ls}");
}

/// A failed manifest write names stdout as the only copy, so the two must match.
#[test]
fn stdout_and_the_manifest_file_are_one_document() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let stdout = String::from_utf8_lossy(&r.out.stdout).into_owned();
    let file = r.read("manifest.json");
    assert_eq!(
        stdout.trim_end_matches('\n'),
        file,
        "stdout and manifest.json have diverged"
    );
    assert!(
        stdout.ends_with('\n') && !file.ends_with('\n'),
        "the newline is stdout's alone"
    );
}

#[test]
fn a_failed_sources_write_keeps_the_urls_in_the_manifest_and_is_counted() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("out");
    // A directory at the artifact defeats the rename; planted mid-run because `--out` refuses
    // one that already holds it.
    let (sources, once) = (out.join("sources.json"), std::sync::Once::new());
    let handler: Handler = Arc::new(move |path: &str, body: &str| match path {
        p if is_free_probe(p) => (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into()),
        _ if !once.is_completed() => {
            once.call_once(|| std::fs::create_dir(&sources).unwrap());
            tool_call("q")
        }
        p if p.contains("/llm/context") => (
            200,
            serde_json::json!({"grounding": {"generic": [
                {"url": "https://kept.test/a", "title": "T", "snippets": ["evidence"]}]}})
            .to_string(),
        ),
        _ if answered(body) => reply(&prose()),
        _ => tool_call("q"),
    });
    let (port, _) = serve(handler);

    let run = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter"])
        .args(["-m", "a/one,b/two", "--out"])
        .arg(&out)
        .args(["--progress", "false"])
        .bounded_output();
    assert_eq!(
        run.status.code(),
        Some(0),
        "a warning, never a failure: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let m: serde_json::Value = serde_json::from_slice(&run.stdout).unwrap();
    assert!(
        m.get("sources_file").is_none(),
        "a file that was never written must not be named: {m}"
    );
    assert_eq!(
        m["members"][0]["sources"][0], "https://kept.test/a",
        "the URLs must fall back into the manifest: {m}"
    );
    assert!(std::fs::read_to_string(out.join("index.md")).is_ok_and(|i| i.contains("a/one")));
    // Counted in the manifest, so the write must precede it.
    assert!(
        m["log_write_errors"].as_u64().unwrap_or(0) > 0,
        "the lost artifact must be counted: {m}"
    );
}

#[test]
fn the_work_queue_never_runs_more_members_at_once_than_concurrency_allows() {
    let live = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let peak = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (l, p) = (Arc::clone(&live), Arc::clone(&peak));
    let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":[]}"#.into());
        }
        let now = l.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        p.fetch_max(now, std::sync::atomic::Ordering::SeqCst);
        // Long enough that any two members allowed to overlap will.
        std::thread::sleep(std::time::Duration::from_millis(300));
        l.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        reply(&prose())
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two,c/three,d/four",
            "--concurrency",
            "2",
            "--progress",
            "false",
        ],
    );
    assert_eq!(r.code(), 0, "{}", r.stderr());
    assert_eq!(
        r.manifest()["quorum"]["delivered"],
        4,
        "every member still runs"
    );
    let seen = peak.load(std::sync::atomic::Ordering::SeqCst);
    assert!(
        seen <= 2,
        "{seen} members were in flight against --concurrency 2"
    );
    assert_eq!(
        seen, 2,
        "and the queue must actually use the room it is given"
    );
}

#[test]
fn results_come_back_in_requested_order_whatever_order_they_finish_in() {
    let (port, _) = serve(mock(|body| {
        // The first member finishes last.
        if body.contains("\"model\":\"a/one\"") {
            std::thread::sleep(std::time::Duration::from_millis(400));
        }
        reply(&prose())
    }));
    let r = council(port, &["-m", "a/one,b/two,c/three", "--progress", "false"]);
    let m = r.manifest();
    let order: Vec<&str> = m["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x["model"].as_str().unwrap())
        .collect();
    assert_eq!(order, ["a/one", "b/two", "c/three"], "{m}");
    assert!(r.read("00-a_one.md").contains("a/one"));
    assert!(r.read("02-c_three.md").contains("c/three"));
}

#[test]
fn a_run_that_fails_before_publishing_still_releases_its_out_lock() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("run");
    // `brief.md` is the first write after the lock; a directory there makes it fail.
    std::fs::create_dir_all(out.join("brief.md")).unwrap();
    let failed = council_in(port, dir.path(), Some(&out), &[]);
    assert_eq!(
        failed.status.code(),
        Some(2),
        "the write was expected to fail"
    );
    assert!(
        !out.join(".council-lock").exists(),
        "the lock outlived the run that took it"
    );
}

/// With no manifest, the lock is what refuses a second `--out` over the paid replies.
#[test]
fn a_run_whose_manifest_write_failed_keeps_its_out_lock() {
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("run");
    // Planted mid-run, past the finished-run guard: only the manifest's rename then fails.
    let plant = out.join("manifest.json").join("x");
    let (port, seen) = serve(mock(move |_| {
        std::fs::create_dir_all(&plant).ok();
        reply(&prose())
    }));
    let first = council_in(port, dir.path(), Some(&out), &[]);
    assert_eq!(first.status.code(), Some(0));
    assert!(
        out.join(".council-lock").is_file(),
        "a run with no manifest released the lock guarding its replies"
    );
    let paid = std::fs::read(out.join("00-a_one.md")).unwrap();
    let asked = || {
        let s = seen.lock().unwrap();
        s.iter().filter(|b| b.contains("\"messages\"")).count()
    };
    let before = asked();
    let again = council_in(port, dir.path(), Some(&out), &[]);
    assert_eq!(again.status.code(), Some(2));
    assert_eq!(asked(), before, "the second run paid");
    assert_eq!(std::fs::read(out.join("00-a_one.md")).unwrap(), paid);
}

/// Global flags are validated in `main.rs`; other commands keep prose-only refusals.
#[test]
fn a_global_flag_refusal_answers_council_in_the_format_it_asked_for() {
    let dir = TempDir::new().unwrap();
    let refuse = |args: &[&str]| bx_in(dir.path()).args(args).bounded_output();
    let out = refuse(&["--timeout", "0", "council", "doctor"]);
    assert_eq!(out.status.code(), Some(2));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["error"]["exit"], 2, "{v}");
    assert_eq!(v["error"]["kind"], "invalid_request");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("timeout"),
        "{v}"
    );
    let text = refuse(&["--timeout", "0", "council", "doctor", "--format", "text"]);
    assert!(text.stdout.is_empty());
    assert!(String::from_utf8_lossy(&text.stderr).contains("timeout"));
    let other = refuse(&["--timeout", "0", "web", "hi"]);
    assert_eq!(other.status.code(), Some(2));
    assert!(other.stdout.is_empty());
}

/// Honours `--config` and `--api-key`, and neither prompts nor migrates a legacy key file.
#[test]
fn doctor_reports_the_brave_key_the_run_would_use_and_migrates_nothing() {
    let dir = TempDir::new().unwrap();
    let brave = |args: &[&str]| {
        let out = bx_in(dir.path())
            .args(args)
            .args(["council", "doctor"])
            .bounded_output();
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        v["keys"]
            .as_array()
            .unwrap()
            .iter()
            .find(|k| k["provider"] == "brave")
            .unwrap()
            .clone()
    };
    assert_eq!(brave(&[])["set"], false, "nothing is configured yet");

    let cfg = dir.path().join("elsewhere.json");
    std::fs::write(&cfg, r#"{"api_key":"BSAconfiguredconfiguredconf"}"#).unwrap();
    assert_eq!(brave(&["--config", cfg.to_str().unwrap()])["set"], true);
    assert_eq!(
        brave(&["--api-key", "BSAflagflagflagflagflagfla"])["set"],
        true
    );

    let legacy = dir.path().join("brave-search");
    std::fs::create_dir_all(&legacy).unwrap();
    std::fs::write(legacy.join("api_key"), "BSAlegacylegacylegacylegac").unwrap();
    assert_eq!(brave(&[])["set"], true, "the legacy key is still a key");
    // A malformed key is a fault, not "not set": the repairs differ.
    let bad = brave(&["--api-key", "not a brave key"]);
    assert_eq!(bad["set"], false);
    assert_ne!(bad["detail"], "not set — bx config set-key <KEY>", "{bad}");
    assert!(
        !legacy.join("config.json").exists(),
        "doctor migrated the legacy key it was only asked about"
    );
}

/// A discarded draft is not an answer, even when the retry after it is too short to publish.
#[test]
fn a_member_left_holding_only_a_discarded_draft_does_not_report_it_as_an_answer() {
    // Counted: a retry re-sends the same body. `--concurrency 1` makes the order exact.
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let (port, _) = serve(mock(move |_| {
        let n = seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (finish, text) = if n.is_multiple_of(2) {
            ("error", prose())
        } else {
            ("stop", "too short to publish".to_string())
        };
        (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": finish, "message": {"content": text}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        )
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--concurrency",
            "1",
        ],
    );
    let m = r.manifest();
    let member = &m["members"][0];
    assert_eq!(member["terminated_by"], "discarded_draft", "{m}");
    assert_eq!(member["finish_reason"], "error");
    assert_ne!(member["status"], "delivered");
    // Not `truncated`: nothing here reached the output ceiling.
    assert_eq!(member["reason"], "no_answer", "{m}");
    assert_eq!(member["action"], "retry");
    // Paid for, so still on disk.
    assert!(r.read("00-a_one.md").contains("grounded prose"));
}

#[test]
fn a_short_discarded_draft_is_named_discarded_not_short() {
    let n = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (port, _) = serve(mock(move |_| {
        let finish = if n.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            "error"
        } else {
            "stop"
        };
        let text = if finish == "error" {
            "short and discarded"
        } else {
            ""
        };
        (
            200,
            serde_json::json!({"choices": [{"finish_reason": finish,
                "message": {"content": text}}], "usage": {}})
            .to_string(),
        )
    }));
    let r = council(port, &["-m", "a/one", "--progress", "false"]);
    let one = &r.manifest()["members"][0];
    assert_eq!(one["terminated_by"], "discarded_draft", "{one}");
}

/// An answer outranks a scratchpad in either order and at any length, on disk and in the verdict.
#[test]
fn an_answer_outranks_a_scratchpad_in_either_order_and_at_any_length() {
    fn turn(finish: &str, msg: serde_json::Value) -> (u16, String) {
        (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": finish, "message": msg}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        )
    }
    let text = |t: String| serde_json::json!({ "content": t });
    let thought = |t: String| serde_json::json!({ "reasoning": t });
    // (turn one, which also calls a tool so the loop goes on; turn two; the answer's marker; the
    // scratchpad's marker; whether the losing scratchpad is billed text the `.partial` keeps)
    let rows = [
        // A held answer, then thinking eight times longer.
        (
            text(prose()),
            thought("thinking. ".repeat(900)),
            "grounded prose",
            "thinking. thinking.",
            true,
        ),
        // Equal length: only the kind can decide.
        (
            text("ANSWER. ".repeat(250)),
            thought("THOUGHT ".repeat(250)),
            "ANSWER.",
            "THOUGHT",
            false,
        ),
        // A held scratchpad, then a much shorter answer.
        (
            thought("thinking. ".repeat(2_000)),
            text(prose()),
            "grounded prose",
            "thinking. thinking.",
            true,
        ),
    ];
    for (mut first, second, answer, scratch, kept) in rows {
        first["tool_calls"] = serde_json::json!([{"id": "c1", "function": {
            "name": "search", "arguments": "{\"query\":\"q\"}"}}]);
        let (port, _) = serve(mock(move |body| {
            if answered(body) {
                turn("stop", second.clone())
            } else {
                turn("tool_calls", first.clone())
            }
        }));
        let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
        let m = r.manifest();
        assert_eq!(m["members"][0]["status"], "delivered", "{answer}: {m}");
        assert_eq!(m["members"][0]["terminated_by"], "answer", "{answer}: {m}");
        let body = r.read("00-a_one.md");
        assert!(body.contains(answer), "{answer}: the answer was lost");
        assert!(
            !body.contains(scratch),
            "{answer}: the scratchpad was published"
        );
        if kept {
            assert!(
                r.read("00-a_one.md.partial").contains(scratch),
                "{answer}: paid characters that lost the selection never reached disk"
            );
        }
    }
}

/// The never-shrink mark is measured on the scrubbed text on disk, not the raw reply.
#[test]
fn a_draft_longer_than_the_file_is_flushed_however_long_the_raw_text_was() {
    let (port, _) = serve(Arc::new(move |path: &str, body: &str| match path {
        p if is_free_probe(p) => (200, r#"{"data":[]}"#.into()),
        p if p.contains("/llm/context") => search_hit(),
        // 9,000 carriage returns the scrubber drops, and a tool call so the loop goes on.
        _ if !answered(body) => (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "content": prose() + &"\r".repeat(9000),
                    "tool_calls": [{"id": "c1", "function": {"name": "search",
                                    "arguments": "{\"query\":\"q\"}"}}]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        ),
        // Loses the selection but is four times the file, so it belongs on disk.
        _ => (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": "stop",
                             "message": {"reasoning": "thinking. ".repeat(500)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        ),
    }));
    let r = council(port, &["-m", "a/one", "--progress", "false"]);
    let published = r.read("00-a_one.md");
    assert!(published.contains("grounded prose"), "{published}");
    assert!(
        !published.contains('\r'),
        "the padding was meant to be scrubbed"
    );
    assert!(
        r.read("00-a_one.md.partial")
            .contains("thinking. thinking."),
        "5,000 paid characters lost to a mark measured on text that is not what the file holds"
    );
}

/// A shorter losing scratchpad writes nothing, so no `.partial` duplicates the answer.
#[test]
fn a_losing_scratchpad_that_wrote_nothing_leaves_the_partial_alone() {
    let (port, _) = serve(Arc::new(move |path: &str, body: &str| match path {
        p if is_free_probe(p) => (200, r#"{"data":[]}"#.into()),
        p if p.contains("/llm/context") => search_hit(),
        // Round two is a sixth of the answer's length, so `flush` has nothing longer to write.
        _ if !answered(body) => (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "content": prose(),
                    "tool_calls": [{"id": "c1", "function": {"name": "search",
                                    "arguments": "{\"query\":\"q\"}"}}]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        ),
        _ => (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": "stop",
                             "message": {"reasoning": "thinking. ".repeat(20)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        ),
    }));
    let r = council(port, &["-m", "a/one", "--progress", "false"]);
    assert!(r.read("00-a_one.md").contains("grounded prose"));
    assert!(
        !r.dir
            .path()
            .join("run")
            .join("00-a_one.md.partial")
            .exists(),
        "the file held the published answer and was kept anyway"
    );
    assert!(
        !r.stderr().contains("holds paid text"),
        "a copy of the reply was announced as text the reply does not have: {}",
        r.stderr()
    );
}

/// A scratchpad passes every length floor, so reasoning-only is judged by channel.
#[test]
fn a_member_that_only_thought_is_not_delivered_and_its_scratchpad_is_still_written() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "model": "test/model",
                // No `content`, yet `finish_reason: "stop"`: the provider calls it complete.
                "choices": [{"finish_reason": "stop", "message": {"reasoning": prose()}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        )
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    let member = &m["members"][0];
    assert_eq!(member["terminated_by"], "reasoning_only", "{m}");
    assert_ne!(member["status"], "delivered", "{m}");
    assert_eq!(member["reason"], "no_answer", "{m}");
    assert_eq!(member["action"], "retry", "{m}");
    // The provider's word is kept; running out of output would be `truncated`.
    assert_eq!(member["finish_reason"], "stop", "{m}");
    assert!(
        member["error"].as_str().unwrap().contains("--effort"),
        "{m}"
    );
    // Paid for, so it reaches disk.
    assert!(r.read("00-a_one.md").contains("grounded prose"));
    assert_eq!(m["quorum"]["delivered"], 0, "{m}");
    assert_ne!(m["state"], "ok", "{m}");
}

/// Brave's entry plan allows one request per second, so a panel searching at once meets 429s.
/// Only the served 200 is billable.
#[test]
fn a_rate_limited_search_is_retried_and_the_member_still_answers_grounded() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let (port, _) = serve(Arc::new(move |path: &str, body: &str| match path {
        p if p.contains("/api/v1/key") => {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        }
        p if p.contains("/v1/models") => (200, r#"{"data":[]}"#.into()),
        p if p.contains("/llm/context") => {
            if seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                (429, r#"{"error":"slow down"}"#.into())
            } else {
                search_hit()
            }
        }
        _ if answered(body) => reply(&prose()),
        _ => tool_call("q"),
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    assert_eq!(r.code(), 0, "{}", r.stderr());
    assert_eq!(m["brave"]["grounded"], 2, "the retry was never served: {m}");
    assert_eq!(
        m["brave"]["errors"], 0,
        "a retried 429 is not an error: {m}"
    );
    // The 429 attempt is not a call against quota.
    assert_eq!(m["brave"]["calls"], 2, "{m}");
    assert_eq!(m["brave"]["sent"], 2, "{m}");
}

#[test]
fn a_brave_base_url_with_a_trailing_slash_still_hits_llm_context() {
    let paths = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&paths);
    let (port, _) = serve(Arc::new(move |path: &str, body: &str| match path {
        p if is_free_probe(p) => (200, r#"{"data":{}}"#.into()),
        p if p.contains("llm/context") => {
            seen.lock().unwrap().push(p.to_string());
            search_hit()
        }
        _ if answered(body) => reply(&prose()),
        _ => tool_call("q"),
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter", "--out"])
        .arg(dir.path().join("run"))
        .args(["-m", "a/one", "--progress", "false"])
        .env("BRAVE_SEARCH_BASE_URL", format!("http://127.0.0.1:{port}/"))
        .bounded_output();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(*paths.lock().unwrap(), ["/res/v1/llm/context"]);
}

/// A Brave server fault is unbilled and often transient, so it is retried like a rate limit.
#[test]
fn a_brave_503_is_retried_once_and_the_search_still_grounds() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (port, _) = serve(Arc::new(move |path: &str, body: &str| match path {
        p if is_free_probe(p) => (200, r#"{"data":{}}"#.into()),
        p if p.contains("/llm/context") => {
            if calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                (503, r#"{"error":"down"}"#.into())
            } else {
                search_hit()
            }
        }
        _ if answered(body) => reply(&prose()),
        _ => tool_call("q"),
    }));
    let r = council(port, &["-m", "a/one", "--progress", "false"]);
    let b = &r.manifest()["members"][0]["brave"];
    assert_eq!(
        (b["grounded"].as_u64(), b["errors"].as_u64()),
        (Some(1), Some(0)),
        "{b}"
    );
}

/// A Brave 200 too large to read was served and billed; its marker names the size, not an
/// unreadable response.
#[test]
fn an_oversized_search_response_is_billed_and_named_as_too_large() {
    let (port, seen) = serve(Arc::new(|path: &str, body: &str| match path {
        p if is_free_probe(p) => (200, r#"{"data":{}}"#.into()),
        p if p.contains("/llm/context") => (200, "x".repeat(9 * 1024 * 1024)),
        _ if answered(body) => reply(&prose()),
        _ => tool_call("q"),
    }));
    let r = council(port, &["-m", "a/one", "--progress", "false"]);
    assert_eq!(r.manifest()["members"][0]["brave"]["sent"], 1);
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .any(|b| b.contains("(search error: response too large)")),
        "the size was never named to the model"
    );
}

/// A dial that never reached Brave cost nothing, so it is retried: here the listener comes up
/// while the first attempt's wait runs.
#[test]
fn a_search_that_never_reached_brave_is_retried_once_the_dial_recovers() {
    let (port, _) = serve(mock(|body| {
        if answered(body) {
            reply(&prose())
        } else {
            tool_call("q")
        }
    }));
    // Reserve a port, then free it: connections are refused until the thread binds it again.
    let brave = TcpListener::bind("127.0.0.1:0").unwrap();
    let brave_port = brave.local_addr().unwrap().port();
    drop(brave);
    thread::spawn(move || {
        thread::sleep(std::time::Duration::from_millis(300));
        let listener = TcpListener::bind(("127.0.0.1", brave_port)).unwrap();
        for mut sock in listener.incoming().flatten() {
            if read_request(&mut sock).is_some() {
                let (_, hit) = search_hit();
                let _ = sock.write_all(
                    format!(
                        "HTTP/1.1 200 X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{hit}",
                        hit.len()
                    )
                    .as_bytes(),
                );
            }
        }
    });
    let dir = TempDir::new().unwrap();
    let out = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter", "--out"])
        .arg(dir.path().join("run"))
        .args(["-m", "a/one", "--progress", "false"])
        .env(
            "BRAVE_SEARCH_BASE_URL",
            format!("http://127.0.0.1:{brave_port}"),
        )
        .bounded_output();
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(m["members"][0]["brave"]["grounded"], 1, "{m}");
}

/// A rate limit that never lifts costs three requests and one marker, never more.
#[test]
fn a_search_gives_up_after_two_rate_limit_waits_with_the_429_marker() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let (port, seen) = serve(Arc::new(move |path: &str, body: &str| match path {
        p if is_free_probe(p) => (200, r#"{"data":{}}"#.into()),
        p if p.contains("/llm/context") => {
            counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            (429, r#"{"error":"slow down"}"#.into())
        }
        _ if answered(body) => reply(&prose()),
        _ => tool_call("q"),
    }));
    let r = council(port, &["-m", "a/one", "--progress", "false"]);
    assert_eq!(r.code(), 0, "{}", r.stderr());
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 3);
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .any(|b| b.contains("(search error HTTP 429)")),
        "the marker never reached the model"
    );
}

/// A wait is taken only when twice its length fits the clock, as `Member::send` rules for the
/// same header; otherwise the marker returns at once.
#[test]
fn a_rate_limited_search_never_sleeps_away_the_budget_it_was_waiting_for() {
    let (port, _) = serve(Arc::new(move |path: &str, body: &str| match path {
        p if is_free_probe(p) => (200, r#"{"data":[]}"#.into()),
        p if p.contains("/llm/context") => (429, r#"{"error":"slow down"}"#.into()),
        _ if answered(body) => reply(&prose()),
        _ => tool_call("q"),
    }));
    let r = council(
        port,
        &["-m", "a/one", "--max-seconds", "2", "--progress", "false"],
    );
    let m = r.manifest();
    assert_eq!(r.code(), 0, "{}", r.stderr());
    let one = &m["members"][0];
    assert_eq!(one["status"], "delivered", "the clock took the member: {m}");
    assert!(
        one["brave"]["errors"].as_u64().unwrap() >= 1,
        "the refusal was meant to reach the conversation as an error: {m}"
    );
    // Brave bills only successful requests: a 429 costs the member a search, the operator nothing.
    assert_eq!(
        one["brave"]["sent"], 0,
        "a refused call was billed to the operator: {m}"
    );
    assert!(
        one["brave"]["calls"].as_u64().unwrap() >= 1,
        "the attempt still costs the member a search: {m}"
    );
    // Under the one-second wait a `.min(left)` bound would sleep.
    assert!(
        one["elapsed_ms"].as_u64().unwrap() < 700,
        "the member slept out a wait it could not outlast: {m}"
    );
}

#[test]
fn prune_dry_run_names_what_it_would_remove_and_removes_nothing() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    let run = root.join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("manifest.json"), "{}").unwrap();
    std::fs::write(run.join("brief.md"), "b").unwrap();
    let f = std::fs::File::options()
        .write(true)
        .open(run.join("brief.md"))
        .unwrap();
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(60 * 86_400);
    f.set_times(std::fs::FileTimes::new().set_modified(old))
        .unwrap();

    let prune = |extra: &[&str]| {
        bx_in(dir.path())
            .args(["council", "prune", "--format", "text"])
            .args(extra)
            .bounded_output()
    };
    // An expired roster is not listed by name, but its bytes count toward what is freed.
    let cache = root.join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    let stale = cache.join("roster-openrouter-dead.json");
    std::fs::write(&stale, "[]").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&stale)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(old))
        .unwrap();

    let dry = prune(&["--dry-run"]);
    let said = String::from_utf8_lossy(&dry.stdout);
    assert!(
        said.contains("would remove 20260101-120000-aaaaaa"),
        "{said}"
    );
    assert!(said.contains("nothing was deleted"), "{said}");
    assert!(run.exists(), "a dry run deleted the run it was describing");
    assert!(stale.exists(), "a dry run deleted a cached roster");
    let real = prune(&[]);
    let done = String::from_utf8_lossy(&real.stdout);
    assert!(done.contains("removed 20260101-120000-aaaaaa"), "{done}");
    assert!(done.contains("2 bytes of expired roster cache"), "{done}");
    assert!(!run.exists());
    assert!(!stale.exists());
}

#[test]
fn set_key_from_an_empty_pipe_is_a_failure_and_says_so_in_its_exit_code() {
    let dir = TempDir::new().unwrap();
    let out = bx_in(dir.path())
        .args(["council", "set-key", "--via", "openrouter"])
        .stdin(Stdio::null())
        .bounded_output();
    assert_eq!(out.status.code(), Some(2), "an empty pipe reported success");
    let said = String::from_utf8_lossy(&out.stderr) + String::from_utf8_lossy(&out.stdout);
    assert!(said.contains("no key on stdin"), "{said}");
    assert!(
        !dir.path()
            .join("brave-search")
            .join("council.json")
            .exists(),
        "a failed set-key wrote a config"
    );
}

/// Refused before any prompt, never rewritten without the field.
#[test]
fn set_key_refuses_rather_than_drops_an_unknown_config_field() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = bx_root(dir.path()).join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    let path = cfg_dir.join("council.json");
    let before = r#"{"future_knob":true}"#;
    std::fs::write(&path, before).unwrap();
    let mut child = bx_in(dir.path())
        .args(["council", "set-key", "--via", "openrouter"])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    let _ = child
        .stdin
        .take()
        .unwrap()
        .write_all(b"sk-or-v1-realkeyrealkeyrealkey\n");
    let out = finish(child);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(!err.contains("paste key"), "{err}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
}

/// The link's target is left alone.
#[cfg(unix)]
#[test]
fn council_set_key_replaces_a_symlinked_config_like_bx_config_does() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = bx_root(dir.path()).join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    let target = dir.path().join("elsewhere.json");
    std::fs::write(&target, "{}").unwrap();
    let path = cfg_dir.join("council.json");
    std::os::unix::fs::symlink(&target, &path).unwrap();
    let mut child = bx_in(dir.path())
        .args(["council", "set-key", "--via", "openrouter"])
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"sk-or-v1-realkeyrealkeyrealkey\n")
        .unwrap();
    let out = finish(child);
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(std::fs::symlink_metadata(&path).unwrap().is_file());
    assert!(
        std::fs::read_to_string(&path)
            .unwrap()
            .contains("sk-or-v1-realkeyrealkeyrealkey")
    );
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "{}");
}

#[test]
fn a_verdict_recorded_through_a_trailing_slash_is_still_found_by_ls() {
    let dir = TempDir::new().unwrap();
    let run = bx_root(dir.path())
        .join("bx")
        .join("council")
        .join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("brief.md"), "subject").unwrap();
    std::fs::write(run.join("status.json"), r#"{"done":1,"total":1}"#).unwrap();
    std::fs::write(run.join("manifest.json"), r#"{"state":"ok"}"#).unwrap();

    let bx = |args: &[&str]| {
        bx_in(dir.path())
            .args(["council"])
            .args(args)
            .bounded_output()
    };
    let slashed = format!("{}/", run.display());
    let scored = bx(&["useful-yes", "--out", &slashed]);
    assert_eq!(
        scored.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&scored.stderr)
    );

    let listed = bx(&["ls"]);
    let v: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert_eq!(
        v["runs"][0]["useful"], true,
        "a verdict keyed on a path nothing matches: {v}"
    );
}

/// Through `..`, or through a link to the root.
#[test]
fn a_verdict_recorded_through_another_spelling_of_the_state_root_is_listed_by_ls() {
    let dir = TempDir::new().unwrap();
    let root = bx_root(dir.path()).join("bx").join("council");
    let run = root.join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::create_dir_all(root.join("cache")).unwrap();
    std::fs::write(run.join("brief.md"), "subject").unwrap();
    std::fs::write(run.join("status.json"), r#"{"done":1,"total":1}"#).unwrap();
    std::fs::write(run.join("manifest.json"), r#"{"state":"ok"}"#).unwrap();
    let useful = |verdict: &str, out: &std::path::Path| {
        let scored = bx_in(dir.path())
            .args(["council", verdict, "--out"])
            .arg(out)
            .bounded_output();
        assert_eq!(
            scored.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&scored.stderr)
        );
        let listed = bx_in(dir.path()).args(["council", "ls"]).bounded_output();
        serde_json::from_slice::<serde_json::Value>(&listed.stdout).unwrap()["runs"][0]["useful"]
            .clone()
    };
    let dotted = root.join("cache").join("..").join("20260101-120000-aaaaaa");
    assert_eq!(useful("useful-yes", &dotted), true, "through ..");
    #[cfg(unix)]
    {
        let alias = dir.path().join("alias");
        std::os::unix::fs::symlink(&root, &alias).unwrap();
        let linked = alias.join("20260101-120000-aaaaaa");
        assert_eq!(useful("useful-no", &linked), false, "through a linked root");
    }
}

/// Named on stderr and not counted as freed.
#[cfg(unix)]
#[test]
fn prune_warns_about_a_roster_it_cannot_remove() {
    use std::os::unix::fs::PermissionsExt;
    // SAFETY: geteuid has no preconditions.
    if unsafe { libc::geteuid() } == 0 {
        return; // root unlinks through any mode
    }
    let dir = TempDir::new().unwrap();
    let cache = dir.path().join("bx").join("council").join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    let roster = cache.join("roster-openrouter-aaaa.json");
    std::fs::write(&roster, "[]").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&roster)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(
            std::time::SystemTime::now() - std::time::Duration::from_secs(30 * 86_400),
        ))
        .unwrap();
    std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o500)).unwrap();
    let out = bx_in(dir.path())
        .args(["council", "prune"])
        .bounded_output();
    std::fs::set_permissions(&cache, std::fs::Permissions::from_mode(0o700)).unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("roster-openrouter-aaaa.json"), "{err}");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["stale_cache_bytes"], 0, "{v}");
}

/// A `cache` that is a link or a file is not swept, and stderr says so.
#[cfg(unix)]
#[test]
fn prune_says_so_when_it_will_not_sweep_a_cache_it_cannot_trust() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    std::fs::create_dir_all(&root).unwrap();
    let prune = || {
        bx_in(dir.path())
            .args(["council", "prune", "--format", "json"])
            .bounded_output()
    };
    // A fresh install has no cache; that is not a warning.
    let fresh = prune();
    assert!(
        !String::from_utf8_lossy(&fresh.stderr).contains("warning:"),
        "a fresh state root warned: {}",
        String::from_utf8_lossy(&fresh.stderr)
    );

    // A directory the sweep would empty if it followed the link to it.
    let elsewhere = dir.path().join("elsewhere");
    std::fs::create_dir(&elsewhere).unwrap();
    let bait = elsewhere.join("roster-openrouter-aaaa.json");
    std::fs::write(&bait, "[]").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&bait)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(
            std::time::SystemTime::now() - std::time::Duration::from_secs(30 * 86_400),
        ))
        .unwrap();
    std::os::unix::fs::symlink(&elsewhere, root.join("cache")).unwrap();
    let out = prune();
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(said.contains("no roster was swept"), "{said}");
    assert!(bait.exists(), "the sweep followed a link out of the root");
    let j: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(j["stale_cache_bytes"], 0, "{j}");

    std::fs::remove_file(root.join("cache")).unwrap();
    std::fs::write(root.join("cache"), "not a directory").unwrap();
    let said = String::from_utf8_lossy(&prune().stderr).into_owned();
    assert!(said.contains("no roster was swept"), "{said}");
}

/// Counted, without failing a member whose reply was published.
#[test]
fn a_partial_that_could_not_be_written_is_counted_not_swallowed() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    std::fs::create_dir_all(&run).unwrap();
    // A directory where the draft flush belongs: `write_private` renames onto it and cannot.
    std::fs::create_dir(run.join("00-a_one.md.partial")).unwrap();

    let out = council_in(port, dir.path(), Some(&run), &[]);
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(m["members"][0]["status"], "delivered", "{m}");
    assert!(
        m["log_write_errors"].as_u64().unwrap() >= 1,
        "a lost draft was reported as a clean sweep: {m}"
    );
    // A write failure, not a dropped trace record.
    assert_eq!(m["trace_dropped"], 0, "{m}");
}

/// A failed write may not raise the mark. The partial is unwritable and the member long then
/// short, so one counted error means round two was never tried.
#[test]
fn a_failed_flush_does_not_lock_out_the_drafts_after_it() {
    let (port, _) = serve(mock(|body| {
        if answered(body) {
            // Shorter than round one.
            reply(&("## A. ANSWER\n".to_string() + &"the real review. ".repeat(40)))
        } else {
            (
                200,
                serde_json::json!({"choices":[{"finish_reason":"tool_calls","message":{
                    "content": "Let me think about this. ".repeat(400),
                    "tool_calls":[{"id":"c1","function":{"name":"search",
                                   "arguments":"{\"query\":\"q\"}"}}]}}],
                    "usage":{}})
                .to_string(),
            )
        }
    }));
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("run");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::create_dir(run.join("00-a_one.md.partial")).unwrap();

    let out = council_in(port, dir.path(), Some(&run), &[]);
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(m["members"][0]["status"], "delivered", "{m}");
    assert!(
        m["log_write_errors"].as_u64().unwrap() >= 2,
        "round one's failure was recorded as progress, so round two never tried: {m}"
    );
    // Warned once per member; the count says how many drafts were lost.
    let said = String::from_utf8_lossy(&out.stderr);
    assert_eq!(said.matches("warning:").count(), 1, "{said}");
}

/// `truncated`, not `empty`: the ceiling is a lever.
#[test]
fn a_last_step_cut_off_with_nothing_in_hand_is_not_an_empty_reply() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({"model":"test/model","choices":[{"finish_reason":"length",
                "message":{"content":""}}],"usage":{}})
            .to_string(),
        )
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one",
            "--max-steps",
            "1",
            "--tools",
            "none",
            "--progress",
            "false",
        ],
    );
    let m = r.manifest();
    let one = &m["members"][0];
    assert_eq!(one["reason"], "truncated", "{m}");
    assert_eq!(one["finish_reason"], "length", "{m}");
    assert!(
        one["error"].as_str().unwrap().contains("--max-tokens"),
        "an action of \"none\" has to name a lever: {m}"
    );
}

/// `delivered` certifies the completion of the published text, never of a later response.
#[test]
fn an_unfinished_turn_never_inherits_a_later_turns_completion() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":[]}"#.into());
        }
        let n = seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // In the order that makes the unfinished draft the longest text in hand.
        let (status, text) = if n.is_multiple_of(2) {
            ("in_progress", prose())
        } else {
            ("completed", "OK".to_string())
        };
        (
            200,
            serde_json::json!({"status": status, "id": format!("resp-{n}"), "output": [
                {"type": "message", "content": [{"type": "output_text", "text": text}]}]})
            .to_string(),
        )
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("openai", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openai", "-m", "a/one,b/two"])
        .args(["--progress", "false", "--concurrency", "1", "--out"])
        .arg(dir.path().join("run"))
        .bounded_output();
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let member = &m["members"][0];
    assert_ne!(member["status"], "delivered", "{m}");
    assert_eq!(member["terminated_by"], "discarded_draft");
    assert_ne!(member["finish_reason"], "stop", "{m}");
    // Paid for, so on disk, but not the answer.
    assert!(
        std::fs::read_to_string(dir.path().join("run/00-a_one.md"))
            .unwrap()
            .contains("grounded prose")
    );
}

#[cfg(unix)]
#[test]
fn a_task_file_that_is_not_a_regular_file_is_refused_rather_than_waited_on() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let tasks = dir.path().join("brave-search/council");
    std::fs::create_dir_all(&tasks).unwrap();
    assert!(
        Command::new("mkfifo")
            .arg(tasks.join("wedge.md"))
            .status()
            .is_ok_and(|s| s.success()),
        "mkfifo is required for this test"
    );
    // No reader is ever attached to the FIFO, so a blocking open never returns.
    let mut child = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--via", "openrouter"])
        .args(["-m", "a/one,b/two", "--progress", "false"])
        .args(["--task", "wedge", "--dry-run"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match child.try_wait().unwrap() {
            Some(_) => break,
            None if std::time::Instant::now() > deadline => {
                child.kill().ok();
                panic!("the run blocked on a FIFO instead of refusing it");
            }
            None => thread::sleep(std::time::Duration::from_millis(50)),
        }
    }
    let out = finish(child);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("not a regular file"), "{err}");
}

/// The cap is charged what is known to be spent, even when too little to publish as a price.
#[test]
fn a_half_reported_byok_charge_still_counts_against_the_cap() {
    let (port, seen) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "model": "test/model", "provider": "Mock",
                "choices": [{"finish_reason": "stop", "message": {"content": prose()}}],
                // BYOK: the fee is absent, the operator's own key was charged $9.
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "is_byok": true,
                          "cost_details": {"upstream_inference_cost": 9.0}}
            })
            .to_string(),
        )
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--concurrency",
            "1",
            "--max-usd",
            "1",
        ],
    );
    let m = r.manifest();
    // The first member spends past the cap, so the second is never sent.
    let chats = seen
        .lock()
        .unwrap()
        .iter()
        .filter(|b| b.contains("messages"))
        .count();
    assert_eq!(chats, 1, "the cap did not bind: {m}");
    assert_eq!(m["members"][1]["terminated_by"], "max_usd", "{m}");
    assert_eq!(m["members"][0]["cost"]["source"], "unknown", "{m}");
    assert_eq!(m["cost"]["cost_unknown"], true);
}

/// A config panel replaces the built-in of its name; a new name is appended.
#[test]
fn doctor_lists_the_panels_that_would_actually_run() {
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    std::fs::write(
        cfg_dir.join("council.json"),
        r#"{"panels": {"default": ["mine/one", "mine/two"], "mine": ["mine/three"]}}"#,
    )
    .unwrap();
    let out = bx_in(dir.path())
        .args(["council", "doctor"])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let panels = v["panels"].as_array().unwrap();
    let named = |name: &str| {
        panels
            .iter()
            .find(|p| p["name"] == name)
            .unwrap_or_else(|| panic!("{name} is missing from {v}"))
            .clone()
    };
    assert_eq!(
        named("default")["members"],
        serde_json::json!(["mine/one", "mine/two"])
    );
    // `default_panel` agrees with `panels`.
    let ran: Vec<&str> = v["default_panel"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["model"].as_str().unwrap())
        .collect();
    assert_eq!(ran, ["mine/one", "mine/two"], "{v}");
    assert!(named("fast")["members"].as_array().unwrap().len() > 1);
    assert_eq!(named("mine")["members"], serde_json::json!(["mine/three"]));
    // The alias of a shadowed built-in still names it: `@sota` is `@default`, whatever it holds.
    assert_eq!(named("default")["also"], serde_json::json!(["sota"]));
    // A shadowed built-in keeps its position; a config-only panel is appended.
    assert_eq!(panels[0]["name"], "default", "{v}");
    assert_eq!(panels.last().unwrap()["name"], "mine", "{v}");
}

#[cfg(unix)]
#[test]
fn ls_on_an_unreadable_state_root_says_so_rather_than_no_runs() {
    use std::os::unix::fs::PermissionsExt;
    // SAFETY: geteuid has no preconditions.
    if unsafe { libc::geteuid() } == 0 {
        return; // root reads any directory
    }
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o000)).unwrap();
    let out = bx_in(dir.path()).args(["council", "ls"]).bounded_output();
    std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o700)).unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("warning:") && err.contains("council"), "{err}");
}

#[test]
fn ls_says_how_many_runs_it_kept_back() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    for n in 0..23 {
        std::fs::create_dir_all(root.join(format!("2026010{}-1200{:02}-aaaaaa", n % 9, n)))
            .unwrap();
    }
    let ls = |args: &[&str]| {
        bx_in(dir.path())
            .args(["council", "ls"])
            .args(args)
            .bounded_output()
    };
    let v: serde_json::Value = serde_json::from_slice(&ls(&[]).stdout).unwrap();
    assert_eq!(v["total"], 23, "{v}");
    assert_eq!(v["shown"], 20);
    assert_eq!(v["runs"].as_array().unwrap().len(), 20);
    let text = String::from_utf8_lossy(&ls(&["--format", "text"]).stdout).into_owned();
    assert!(text.contains("3 more"), "{text}");
    assert!(text.contains("20 of 23 shown"), "{text}");
}

/// The tail read keeps the newest rows, and `scores` is later-wins, so a listed run's verdict
/// survives the clip. `read_local` would refuse the whole file instead.
#[test]
fn a_verdict_for_a_listed_run_survives_a_clipped_scores_file() {
    let dir = TempDir::new().unwrap();
    let root = bx_root(dir.path()).join("bx").join("council");
    let run = "20260101-120000-aaaaaa";
    std::fs::create_dir_all(root.join(run)).unwrap();
    std::fs::write(root.join(run).join("brief.md"), "subject").unwrap();
    let judged = root.join(run).display().to_string();

    // Past `MAX_HISTORY`, with the answer last as an append-only record puts it.
    let mut body = String::new();
    for i in 0..12_000 {
        body.push_str(&format!(
            r#"{{"schema":1,"ts":"2026-01-01T00:00:00Z","run":"old{i}","dir":"/gone/{i:0>140}","useful":false}}"#,
        ));
        body.push('\n');
    }
    body.push_str(&format!(
        r#"{{"schema":1,"ts":"2026-01-02T00:00:00Z","run":"{run}","dir":"{judged}","useful":true}}"#,
    ));
    body.push('\n');
    assert!(
        body.len() as u64 > 2 * 1024 * 1024,
        "the fixture must exceed MAX_HISTORY to exercise the clip, got {} bytes",
        body.len()
    );
    std::fs::write(root.join("scores.jsonl"), &body).unwrap();

    let out = bx_in(dir.path()).args(["council", "ls"]).bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["runs"][0]["run"], run, "{v}");
    assert_eq!(
        v["runs"][0]["useful"], true,
        "a clip lost the verdict for a run the listing shows: {v}"
    );
}

/// Only when the task searches, and missing and malformed are named apart: different repairs.
#[test]
fn a_missing_brave_key_refuses_a_grounded_run_in_the_format_the_caller_asked_for() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let run = |args: &[&str], key: Option<&str>| {
        let mut c = bx("openrouter", port, dir.path());
        c.args(["council", "SUBJECT", "--via", "openrouter"])
            .args(["-m", "a/one,b/two", "--progress", "false", "--dry-run"])
            .args(args);
        match key {
            Some(k) => c.env("BRAVE_SEARCH_API_KEY", k),
            None => c.env_remove("BRAVE_SEARCH_API_KEY"),
        };
        c.bounded_output()
    };
    let absent = run(&[], None);
    assert_eq!(absent.status.code(), Some(2));
    let v: serde_json::Value = serde_json::from_slice(&absent.stdout).unwrap();
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("not set"), "{msg}");
    assert!(
        msg.contains("bx config set-key"),
        "the repair is named: {msg}"
    );

    let bad: serde_json::Value =
        serde_json::from_slice(&run(&[], Some("has a space")).stdout).unwrap();
    let msg = bad["error"]["message"].as_str().unwrap();
    assert!(msg.contains("printable ASCII"), "{msg}");

    assert_eq!(run(&["--tools", "none"], None).status.code(), Some(0));
}

#[test]
fn a_computed_price_missing_a_rounds_tokens_is_published_as_a_floor() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let (port, _) = serve(Arc::new(move |path: &str, _: &str| {
        if is_free_probe(path) {
            return (200, r#"{"data":[]}"#.into());
        }
        // Round two reports no `usage`, which every wire parses as zeros.
        if seen
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            .is_multiple_of(2)
        {
            (
                200,
                serde_json::json!({"model": "claude-opus-5-5", "stop_reason": "tool_use",
                    "content": [{"type": "tool_use", "id": "t1", "name": "brave_search",
                                 "input": {"query": "q"}}],
                    "usage": {"input_tokens": 1000, "output_tokens": 40}})
                .to_string(),
            )
        } else {
            (
                200,
                serde_json::json!({"model": "claude-opus-5-5", "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": prose()}]})
                .to_string(),
            )
        }
    }));
    let r = council_rated(
        port,
        r#"{"rates": {"claude-opus-5-5": [5.0, 25.0]}}"#,
        &["--concurrency", "1"],
    );
    let m = r.manifest();
    assert_eq!(
        m["members"][0]["cost"]["source"], "computed_partial",
        "an incomplete count was priced as if it were whole: {m}"
    );
    // Still published as the best floor, and the total is flagged as not a bill.
    assert!(
        m["members"][0]["cost"]["llm_usd"].as_f64().unwrap() > 0.0,
        "{m}"
    );
    assert_eq!(m["cost"]["cost_unknown"], true, "{m}");
}

/// Every option's doc is a summary line, a blank line, then the detail; one paragraph would put
/// the detail in `-h` too.
#[test]
fn the_short_help_is_a_summary_and_the_long_one_carries_the_detail() {
    let dir = TempDir::new().unwrap();
    let help = |flag: &str| {
        let out = bx_in(dir.path()).args(["council", flag]).bounded_output();
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let short = help("-h");
    let long = help("--help");
    for flag in ["--max-usd", "--dry-run", "--tools", "--out", "-m, --model"] {
        assert!(short.contains(flag), "{flag} is missing from -h");
    }
    for detail in [
        "council.json is not redirected",
        "overshoot is up to one in-flight round",
        "no Brave key is needed",
        "history.jsonl stays in the state directory",
    ] {
        assert!(long.contains(detail), "--help dropped: {detail}");
        assert!(
            !short.contains(detail),
            "-h carries the long text: {detail}"
        );
    }
    // Clap's trailing annotations are stripped off the end only: splitting on the first `" ["`
    // would measure `[TARGET]`'s row as one column.
    const ANNOTATIONS: &[&str] = &["default:", "task default:", "possible values:", "env:"];
    let measure = |line: &str| {
        let mut l = line.trim_end();
        while let Some(open) = l.strip_suffix(']').and_then(|s| s.rfind(" [")) {
            let inner = &l[open + 2..l.len() - 1];
            if !ANNOTATIONS.iter().any(|a| inner.starts_with(a)) {
                break;
            }
            l = l[..open].trim_end();
        }
        l.chars().count()
    };
    if let Some(over) = short.lines().find(|l| measure(l) > 100) {
        panic!("-h has a {}-column summary: {over}", measure(over));
    }
    // Named, so a regression says which row.
    for row in ["[SUBJECT]", "[TARGET]"] {
        let line = short
            .lines()
            .find(|l| l.trim_start().starts_with(row))
            .unwrap_or_else(|| panic!("-h has no {row} row: {short}"));
        assert!(
            measure(line) <= 100,
            "the {row} row is {} columns: {line}",
            measure(line)
        );
    }
    // `[TARGET]` is the only place the run positional is named.
    assert!(short.contains("[TARGET]") && long.contains("[TARGET]"));
    assert!(
        short.lines().count() < long.lines().count() / 3,
        "-h is {} lines against --help's {}",
        short.lines().count(),
        long.lines().count()
    );
}

/// `prune` never trims `history.jsonl`, so its readers clip at `MAX_HISTORY` and must say so.
#[test]
fn a_clipped_history_window_says_it_dropped_rows() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    std::fs::create_dir_all(&root).unwrap();
    let (port, _guard) = serve(mock(|_| reply(&prose())));

    // Both members on every run, so the panel-superset filter keeps them.
    let rows: Vec<String> = (0..30_000)
        .flat_map(|i| {
            [("a/one", 1.0), ("b/two", 2.0)].map(move |(m, c)| {
                format!(
                    r#"{{"run":"r{i}","dir":"/x","model":"{m}","cost_usd":{c},"cost_source":"reported"}}"#
                )
            })
        })
        .collect();
    let body = rows.join("\n") + "\n";
    assert!(
        body.len() as u64 > 2 * 1024 * 1024,
        "the fixture must exceed MAX_HISTORY to exercise the clip"
    );
    std::fs::write(root.join("history.jsonl"), &body).unwrap();

    let out = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--dry-run", "--tools", "none"])
        .args(["-m", "a/one,b/two", "--via", "openrouter"])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let seen = &v["cost"]["observed"];
    // A partial window is still usable.
    assert!(seen["run_total"]["n"].as_u64().unwrap() > 0, "{seen}");
    assert!(
        seen["window"]
            .as_str()
            .unwrap_or_default()
            .contains("partial"),
        "a window that dropped rows must say so: {seen}"
    );

    // The text form too: an unseen older run may have cost more than the max it prints.
    let text = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "--dry-run", "--tools", "none"])
        .args(["-m", "a/one,b/two", "--via", "openrouter"])
        .args(["--format", "text"])
        .bounded_output();
    let text = String::from_utf8_lossy(&text.stdout);
    let line = text
        .lines()
        .find(|l| l.starts_with("observed"))
        .expect("the text form must carry the observed table too");
    assert!(line.contains("past the read cap"), "{line}");
}

#[test]
fn a_sub_command_takes_its_run_as_a_positional_or_as_out() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let dir = r.dir.path().join("run");
    let root = r.dir.path().join("bx").join("council");
    let call = |args: &[&std::ffi::OsStr]| {
        bx_in(r.dir.path())
            .args(["council"])
            .args(args)
            .bounded_output()
    };
    let o = std::ffi::OsStr::new;

    // Each word in TARGETED, so dropping one from the list fails here.
    for w in ["status", "findings"] {
        let out = call(&[o(w), dir.as_os_str()]);
        assert_eq!(
            out.status.code(),
            Some(0),
            "{w} <run>: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let no = call(&[o("useful-no"), dir.as_os_str()]);
    assert_eq!(
        no.status.code(),
        Some(0),
        "useful-no <run>: {}",
        String::from_utf8_lossy(&no.stderr)
    );

    let pos = call(&[o("findings"), dir.as_os_str()]);
    let via_out = call(&[o("findings"), o("--out"), dir.as_os_str()]);
    assert_eq!(
        pos.status.code(),
        Some(0),
        "{:?}",
        String::from_utf8_lossy(&pos.stderr)
    );
    assert_eq!(pos.stdout, via_out.stdout, "the two spellings must agree");

    let scored = call(&[o("useful-yes"), dir.as_os_str()]);
    assert_eq!(
        scored.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&scored.stderr)
    );
    let rows = std::fs::read_to_string(root.join("scores.jsonl")).expect("a verdict was recorded");
    // Append-only: the last row wins, and the `useful-no` above is kept too.
    let v: serde_json::Value = serde_json::from_str(rows.lines().next_back().unwrap()).unwrap();
    assert_eq!(v["useful"], true, "{v}");
    assert_eq!(v["dir"], dir.display().to_string(), "{v}");
    assert_eq!(rows.lines().count(), 2, "both verdicts are kept: {rows}");

    // Both spellings at once is refused, not resolved by precedence.
    let both = call(&[o("findings"), dir.as_os_str(), o("--out"), dir.as_os_str()]);
    assert_eq!(both.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&both.stderr).contains("pass one of them"),
        "{}",
        String::from_utf8_lossy(&both.stderr)
    );

    // When they disagree the message names both runs.
    let other = r.dir.path().join("elsewhere");
    std::fs::create_dir_all(&other).unwrap();
    let differ = call(&[
        o("findings"),
        dir.as_os_str(),
        o("--out"),
        other.as_os_str(),
    ]);
    assert_eq!(differ.status.code(), Some(2));
    let e = String::from_utf8_lossy(&differ.stderr);
    assert!(
        e.contains(&dir.display().to_string()) && e.contains(&other.display().to_string()),
        "the refusal names neither run it is refusing to choose between: {e}"
    );
}

/// Only a sub-command takes a second positional; the refusal names the repair, in JSON when asked.
#[test]
fn a_second_positional_after_a_subject_is_refused_before_anything_is_sent() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let call = |args: &[&str]| {
        let out = bx_in(dir.path())
            .arg("council")
            .args(args)
            .env("OPENROUTER_API_KEY", "sk-or-v1-test")
            .env("BRAVE_SEARCH_API_KEY", "BSAtestkey000")
            // Routed to the mock, so a refusal that failed to refuse cannot reach a real API.
            .env("BRAVE_SEARCH_BASE_URL", format!("http://127.0.0.1:{port}"))
            .env(
                "BX_COUNCIL_OPENROUTER_BASE_URL",
                format!("http://127.0.0.1:{port}"),
            )
            .bounded_output();
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        // Ours, not clap's: JSON, carrying the message stderr carries.
        let doc: serde_json::Value = serde_json::from_slice(&out.stdout)
            .unwrap_or_else(|e| panic!("{args:?} answered in prose: {e}"));
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        assert_eq!(doc["error"]["exit"], 2, "{doc}");
        assert!(
            !stderr.contains("Usage:"),
            "a usage block, not a repair: {stderr}"
        );
        (stderr, doc)
    };

    let (e, doc) = call(&["my subject", "oops", "--progress", "false"]);
    assert!(e.contains("unexpected second argument `oops`"), "{e}");
    assert!(e.contains("Quote it") && e.contains("sub-command"), "{e}");
    assert_eq!(
        doc["error"]["message"],
        e.trim().trim_start_matches("error: ")
    );

    // An unquoted subject: the first stray token is named, not the last.
    let (four, _) = call(&["fix", "the", "login", "bug"]);
    assert!(four.contains("unexpected second argument `the`"), "{four}");
    assert!(four.contains("Quote it"), "{four}");

    // A sub-command takes exactly one run, so the third is the stray.
    let (third, _) = call(&["findings", "20260101-120000-aaaaaa", "extra"]);
    assert!(
        third.contains("unexpected third argument `extra`"),
        "{third}"
    );
    assert!(!third.contains("second argument"), "{third}");

    // A flag absorbed after `--` gets its own advice: dropping `--dry-run` is the fix that spends.
    let (e2, _) = call(&["--", "review this", "--dry-run"]);
    assert!(e2.contains("Everything after `--`"), "{e2}");
    assert!(!e2.contains("Quote it"), "the wrong repair: {e2}");

    assert_eq!(seen.lock().unwrap().len(), 0, "a refusal spent money");
}

/// `useful` is in `TARGETED` only so a run after it reaches this refusal, not "takes no run".
#[test]
fn useful_names_the_two_real_spellings_whether_or_not_a_run_follows_it() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    std::fs::create_dir_all(&root).unwrap();
    let call = |args: &[&str]| bx_in(dir.path()).arg("council").args(args).bounded_output();
    for args in [
        vec!["useful"],
        vec!["useful", "20260101-120000-aaaaaa"],
        vec!["useful", "--out", "20260101-120000-aaaaaa"],
    ] {
        let out = call(&args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        let e = String::from_utf8_lossy(&out.stderr);
        assert!(
            e.contains("useful-yes") && e.contains("useful-no"),
            "{args:?}: the repair names neither spelling: {e}"
        );
        assert!(
            !e.contains("takes no run"),
            "{args:?}: the two refusals contradict each other: {e}"
        );
    }
    assert!(!root.join("scores.jsonl").exists());
}

/// No row passes a subject, so no panel can start whatever the parse does.
#[test]
fn a_refusal_clap_raises_is_answered_in_the_format_the_caller_asked_for() {
    let dir = TempDir::new().unwrap();
    let call = |args: &[&str]| bx_in(dir.path()).arg("council").args(args).bounded_output();
    for shape in [
        vec!["--min-quorum", "0"],
        vec!["--tools", "sideways"],
        vec!["--effort", "sideways"],
        vec!["--max-usd", "cheap"],
        vec!["--nosuchflag"],
        vec!["-leading-dash"],
    ] {
        let out = call(&shape);
        assert_eq!(out.status.code(), Some(2), "{shape:?}");
        let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
            panic!(
                "{shape:?} answered in prose: {e}\nstdout: {}\nstderr: {}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            )
        });
        assert_eq!(doc["error"]["exit"], 2, "{shape:?}: {doc}");
        assert_eq!(doc["error"]["kind"], "invalid_request", "{shape:?}: {doc}");
        let e = String::from_utf8_lossy(&out.stderr).into_owned();
        // One message in both places.
        assert_eq!(
            doc["error"]["message"],
            e.trim().trim_start_matches("error: "),
            "{shape:?}"
        );
        assert!(!e.contains("Usage:"), "{shape:?}: {e}");

        // Asked for text: prose on stderr, stdout empty.
        let mut prose = shape.clone();
        prose.extend(["--format", "text"]);
        let out = call(&prose);
        assert_eq!(out.status.code(), Some(2), "{prose:?}");
        assert!(
            out.stdout.is_empty(),
            "{prose:?} wrote to stdout: {}",
            String::from_utf8_lossy(&out.stdout)
        );
        assert!(
            String::from_utf8_lossy(&out.stderr).starts_with("error: "),
            "{prose:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// `--` cannot mark it a subject because clap does not report whether `--` was used.
#[test]
fn a_reserved_word_is_matched_whole_and_dash_dash_does_not_make_it_a_subject() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("bx").join("council")).unwrap();
    let bare = |args: &[&str]| bx_in(dir.path()).arg("council").args(args).bounded_output();

    let listed = bare(&["--", "ls"]);
    assert_eq!(
        listed.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&listed.stderr)
    );
    let doc: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert!(doc["root"].is_string(), "not the listing: {doc}");
    assert_eq!(doc["total"], 0, "{doc}");

    // After `--` a second word lands in the run slot, and `ls` reads no run.
    let extra = bare(&["--", "ls", "extra"]);
    assert_eq!(extra.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&extra.stderr).contains("takes no run"),
        "{}",
        String::from_utf8_lossy(&extra.stderr)
    );

    // Padded, it is neither the sub-command nor a subject: refused, naming it.
    for padded in [" ls", "ls "] {
        let out = bare(&[padded, "--format", "json"]);
        assert_eq!(out.status.code(), Some(2), "{padded:?}");
        let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert!(
            doc["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("`bx council ls`")),
            "{padded:?}: {doc}"
        );
    }

    // No `-m`: `subcommand` returns `None` once `-m` is set, so a fixture panel would skip the
    // check. `--via openrouter` keeps the default panel on the mock.
    let (port, _seen) = serve(mock(|_| reply(&prose())));
    let subject = "status report for Q3";
    let out = bx("openrouter", port, dir.path())
        .args(["council", subject, "--dry-run", "--format", "json"])
        .args(["--via", "openrouter", "--tools", "none"])
        .args(["--progress", "false"])
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{subject:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(plan["dry_run"], true, "{subject:?}: {plan}");
    // A dispatch prints `root` and no members; the plan the reverse.
    assert!(plan["root"].is_null(), "{subject:?} dispatched: {plan}");
    assert!(
        plan["members"].as_array().is_some_and(|m| !m.is_empty()),
        "{subject:?}: {plan}"
    );
}

#[test]
fn a_one_word_subject_sends_nothing_unless_it_came_from_a_file_stdin_or_m() {
    let dir = TempDir::new().unwrap();
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let council = |args: &[&str], stdin: &str| {
        let mut c = bx("openrouter", port, dir.path())
            .arg("council")
            .args(args)
            .args([
                "--via",
                "openrouter",
                "--tools",
                "none",
                "--progress",
                "false",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        c.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
        finish(c)
    };

    for (word, says) in [
        ("help", "`bx council --help`"),
        ("Doctor", "`bx council doctor`"),
        ("tax", "quote a sentence"),
    ] {
        let out = council(&[word], "");
        assert_eq!(out.status.code(), Some(2), "{word}");
        let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(doc["error"]["kind"], "invalid_request", "{word}: {doc}");
        assert!(
            doc["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains(says)),
            "{word}: {doc}"
        );
    }
    assert!(
        seen.lock().unwrap().is_empty(),
        "a one-word subject sent a request"
    );

    let file = dir.path().join("one-word.md");
    std::fs::write(&file, "tax").unwrap();
    let at_file = format!("@{}", file.display());
    for (args, stdin) in [
        (vec!["tax law", "--dry-run"], ""),
        (vec!["tax", "--dry-run", "-m", "a/one,b/two"], ""),
        (vec![at_file.as_str(), "--dry-run"], ""),
        (vec!["-", "--dry-run"], "tax"),
    ] {
        let out = council(&args, stdin);
        assert_eq!(
            out.status.code(),
            Some(0),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// One model named once is the retry `hint` prescribes; a panel that collapsed to one is a mistake.
#[test]
fn one_model_named_once_is_allowed_but_a_collapsed_panel_is_not() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "only/one", "--progress", "false"]);
    assert_eq!(r.code(), 0, "{}", r.stderr());
    let m = r.manifest();
    assert_eq!(m["quorum"]["total"], 1, "{m}");
    assert_eq!(m["quorum"]["delivered"], 1, "{m}");
    assert_eq!(m["members"][0]["status"], "delivered", "{m}");
    assert!(!seen.lock().unwrap().is_empty(), "the member must have run");

    // A literal repeat: `panel_slugs` dedups before the transport check, so refused there.
    let (port_dup, seen_dup) = serve(mock(|_| reply(&prose())));
    let literal = council(port_dup, &["-m", "a/one,a/one", "--progress", "false"]);
    assert_eq!(literal.code(), 2, "{}", literal.stderr());
    assert!(
        literal.stderr().contains("name one model"),
        "{}",
        literal.stderr()
    );
    assert!(seen_dup.lock().unwrap().is_empty(), "a refusal spent money");
}

/// A textless tool-call turn (`content: null`) is not an answer and cannot displace a scratchpad.
#[test]
fn an_empty_turn_never_erases_a_held_scratchpad() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let (port, _) = serve(Arc::new(move |path: &str, _body: &str| match path {
        p if p.contains("/api/v1/key") => {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        }
        p if p.contains("/llm/context") => search_hit(),
        p if p.contains("/v1/models") => (200, r#"{"data":[]}"#.into()),
        _ => {
            let n = seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Thinking plus a tool call, then textless tool calls with no reasoning.
            let msg = if n < 2 {
                serde_json::json!({"reasoning": "paid thinking. ".repeat(400),
                                   "tool_calls": [{"id": "c1", "function": {
                                       "name": "search", "arguments": "{\"query\":\"q\"}"}}]})
            } else {
                serde_json::json!({"content": serde_json::Value::Null,
                                   "tool_calls": [{"id": "c1", "function": {
                                       "name": "search", "arguments": "{\"query\":\"q\"}"}}]})
            };
            (
                200,
                serde_json::json!({
                    "model": "test/model",
                    "choices": [{"finish_reason": "tool_calls", "message": msg}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
                })
                .to_string(),
            )
        }
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--max-searches",
            "3",
        ],
    );
    let m = r.manifest();
    // Not an answer, but billed, so still on disk.
    assert_ne!(m["members"][0]["status"], "delivered", "{m}");
    let body = r.read("00-a_one.md");
    assert!(
        body.contains("paid thinking."),
        "the only copy of a paid scratchpad was erased by a textless turn: {} bytes",
        body.len()
    );
}

/// The truncated flag describes the text that publishes, not the turn that ended the loop.
#[test]
fn a_complete_answer_is_not_stamped_truncated_by_a_later_capped_turn() {
    // A trailing tool call makes the loop run a second turn.
    fn answer_then_call() -> (u16, String) {
        (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "content": prose(),
                    "tool_calls": [{"id": "c1", "function": {"name": "search",
                                    "arguments": "{\"query\":\"q\"}"}}]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        )
    }
    fn cut(msg: serde_json::Value) -> (u16, String) {
        (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": "length", "message": msg}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        )
    }
    let published_answer = |r: &Run, arm: &str| {
        let m = r.manifest();
        let x = &m["members"][0];
        assert_eq!(x["status"], "delivered", "{arm}: {m}");
        assert_eq!(x["terminated_by"], "answer", "{arm}: {m}");
        assert_ne!(x["reason"], "truncated", "{arm}: {m}");
        assert!(
            r.read("00-a_one.md").contains("grounded prose"),
            "{arm}: the paid review is not the text on disk"
        );
    };
    // At the ceiling, so the first truncation is terminal, not an escalation.
    let at_cap = [
        "-m",
        "a/one,b/two",
        "--progress",
        "false",
        "--max-tokens",
        "128000",
    ];

    // 1. The later turn is a long scratchpad that stops for `length`.
    let (port, _) = serve(mock(|body| {
        if answered(body) {
            cut(serde_json::json!({"reasoning": "thinking. ".repeat(2_000)}))
        } else {
            answer_then_call()
        }
    }));
    published_answer(&council(port, &at_cap), "a capped scratchpad");

    // 2. Shorter, with no reasoning channel: isolates step 2 from the kind ranking.
    let (port, _) = serve(mock(|body| {
        if answered(body) {
            cut(serde_json::json!({"content": "x".repeat(100)}))
        } else {
            answer_then_call()
        }
    }));
    published_answer(&council(port, &at_cap), "a capped shorter turn");

    // 3. Below the cap it escalates; the third turn adds nothing, so round one publishes.
    let (port, _) = serve(mock(|body| {
        if !answered(body) {
            answer_then_call()
        } else if body.contains(r#""max_completion_tokens":64000"#) {
            cut(serde_json::json!({"content": "x".repeat(100)}))
        } else {
            reply("")
        }
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    published_answer(&r, "an escalation");
    let m = r.manifest();
    assert!(
        m["members"][0]["limits"]["max_tokens"]["escalations"]
            .as_u64()
            .unwrap_or(0)
            >= 1,
        "the turn below the cap never escalated: {m}"
    );

    // 4. On the last step, below the cap: no escalation is counted with no round left.
    let (port, _) = serve(mock(|body| {
        if answered(body) {
            cut(serde_json::json!({"content": "x".repeat(100)}))
        } else {
            answer_then_call()
        }
    }));
    let r = council(
        port,
        &[
            "-m",
            "a/one,b/two",
            "--progress",
            "false",
            "--max-steps",
            "2",
        ],
    );
    published_answer(&r, "a last-step truncation");
    let m = r.manifest();
    assert_eq!(
        m["members"][0]["limits"]["max_tokens"]["escalations"], 0,
        "{m}"
    );
}

/// An empty turn is never kept, so the flag cannot come from `draft_truncated` here.
#[test]
fn a_ceiling_reached_with_nothing_in_hand_still_reports_the_ceiling() {
    let empty_at_length = |reason: &str| {
        let reason = reason.to_string();
        move |_: &str| {
            (
                200,
                serde_json::json!({
                    "model": "test/model",
                    "choices": [{"finish_reason": reason, "message": {"content": ""}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
                })
                .to_string(),
            )
        }
    };
    // At the ceiling, so the first truncation is terminal.
    let at_cap = [
        "-m",
        "a/one,b/two",
        "--progress",
        "false",
        "--max-tokens",
        "128000",
    ];
    let (port, _) = serve(mock(empty_at_length("length")));
    let r = council(port, &at_cap);
    let m = r.manifest();
    let x = &m["members"][0];
    assert_eq!(x["terminated_by"], "truncated_at_cap", "{m}");
    assert_eq!(x["reason"], "truncated", "{m}");
    assert_eq!(x["finish_reason"], "length", "{m}");
    assert_ne!(x["status"], "delivered", "{m}");
    let e = x["error"].as_str().unwrap_or_default();
    assert!(e.contains("output ceiling"), "names no lever: {e}");

    // The input side: Anthropic's own `stop_reason`.
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({"stop_reason": "model_context_window_exceeded",
                "content": [{"type": "text", "text": ""}],
                "usage": {"input_tokens": 10, "output_tokens": 0}})
            .to_string(),
        )
    }));
    let dir = TempDir::new().unwrap();
    let out = bx("anthropic", port, dir.path())
        .args(["council", "SUBJECT", "--via", "anthropic"])
        .args(["-m", "a/one,b/two", "--progress", "false", "--out"])
        .arg(dir.path().join("run"))
        .bounded_output();
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let x = &m["members"][0];
    assert_eq!(x["terminated_by"], "context_window", "{m}");
    assert_eq!(x["reason"], "truncated", "{m}");
    assert_eq!(x["finish_reason"], "length", "{m}");
    assert!(
        x["error"]
            .as_str()
            .unwrap_or_default()
            .contains("context window"),
        "{x}"
    );
}

/// A discarded turn of thinking stays a scratchpad, so the shorter real reply after it publishes.
#[test]
fn a_discarded_scratchpad_then_a_real_answer_is_clean() {
    // Per member: a retry re-sends an identical body, and a shared counter would cross members.
    let rounds = Arc::new(Mutex::new(std::collections::HashMap::<String, usize>::new()));
    let (port, _) = serve(mock(move |body: &str| {
        let member = if body.contains(r#""model":"a/one""#) {
            "a"
        } else {
            "b"
        };
        let mut seen = rounds.lock().unwrap();
        let n = seen.entry(member.to_string()).or_default();
        *n += 1;
        match *n {
            // Retryable, billed, thinking only.
            1 => (
                200,
                serde_json::json!({
                    "model": "test/model",
                    "choices": [{"finish_reason": "error",
                                 "message": {"reasoning": "paid thinking. ".repeat(200)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
                })
                .to_string(),
            ),
            // Shorter than the scratchpad and under the floor: only its kind can rank it first.
            2 => reply(&"a real finding. ".repeat(19)),
            _ => reply(&prose()),
        }
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    let x = &m["members"][0];
    assert_eq!(x["status"], "delivered", "{m}");
    assert_eq!(x["terminated_by"], "answer", "{m}");
    assert_eq!(x["retries"], 1, "{m}");
    assert_eq!(x["finish_reason"], "stop", "{m}");
    assert!(r.read("00-a_one.md").contains("grounded prose"), "{m}");
}

/// The longer discarded text stays in the `.partial`.
#[test]
fn a_discarded_answer_does_not_label_the_shorter_answer_that_replaced_it() {
    let rounds = Arc::new(Mutex::new(std::collections::HashMap::<String, usize>::new()));
    let (port, _) = serve(mock(move |body: &str| {
        let member = if body.contains(r#""model":"a/one""#) {
            "a"
        } else {
            "b"
        };
        let mut seen = rounds.lock().unwrap();
        let n = seen.entry(member.to_string()).or_default();
        *n += 1;
        match *n {
            1 => (
                200,
                serde_json::json!({
                    "model": "test/model",
                    "choices": [{"finish_reason": "error",
                                 "message": {"content": "a long discarded finding. ".repeat(120)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
                })
                .to_string(),
            ),
            // Shorter than the draft, over the floor.
            _ => reply(&"a shorter real finding. ".repeat(34)),
        }
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    let x = &m["members"][0];
    assert_eq!(x["retries"], 1, "the retry path was not exercised: {m}");
    assert_eq!(x["status"], "delivered", "{m}");
    assert_eq!(x["terminated_by"], "answer", "{m}");
    assert_eq!(x["finish_reason"], "stop", "{m}");
    let body = r.read("00-a_one.md");
    assert!(body.contains("a shorter real finding."), "{m}");
    assert!(
        r.read("00-a_one.md.partial")
            .contains("a long discarded finding."),
        "the paid draft was destroyed"
    );
}

/// Even when the regeneration carries a call.
#[test]
fn a_successful_regeneration_displaces_a_longer_discarded_draft() {
    let n = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (port, _) = serve(mock(move |body: &str| {
        if answered(body) {
            return reply("See my review above.");
        }
        if n.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            return (
                200,
                serde_json::json!({"choices": [{"finish_reason": "error",
                    "message": {"content": "a long discarded finding. ".repeat(120)}}],
                    "usage": {}})
                .to_string(),
            );
        }
        (
            200,
            serde_json::json!({"choices": [{"finish_reason": "tool_calls", "message": {
                "content": prose(), "tool_calls": [{"id": "c1", "function":
                    {"name": "search", "arguments": "{\"query\":\"q\"}"}}]}}],
                "usage": {}})
            .to_string(),
        )
    }));
    let r = council(port, &["-m", "a/one", "--progress", "false"]);
    let x = &r.manifest()["members"][0];
    assert_eq!(x["status"], "delivered", "{x}");
    assert!(r.read("00-a_one.md").contains("grounded prose"), "{x}");
}

/// Pins `keep_draft`'s floor at emptiness, not `min_reply_chars`.
#[test]
fn a_short_answer_is_re_prompted_into_a_full_review_over_a_held_scratchpad() {
    let (port, _) = serve(mock(|body| {
        if body.contains("Provide your final answer now") {
            reply(&prose())
        } else if answered(body) {
            reply(&"x".repeat(30))
        } else {
            (
                200,
                serde_json::json!({
                    "model": "test/model",
                    "choices": [{"finish_reason": "tool_calls", "message": {
                        "reasoning": "thinking. ".repeat(200),
                        "tool_calls": [{"id": "c1", "function": {"name": "search",
                                        "arguments": "{\"query\":\"q\"}"}}]}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
                })
                .to_string(),
            )
        }
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    assert_eq!(m["members"][0]["status"], "delivered", "{m}");
    assert_eq!(m["members"][0]["terminated_by"], "answer", "{m}");
    let body = r.read("00-a_one.md");
    assert!(
        body.contains("grounded prose"),
        "the short answer was never re-prompted: {} bytes",
        body.len()
    );
    assert!(
        !body.contains("thinking. thinking."),
        "the scratchpad published"
    );
}

/// Labelled `short_reply`, yet the message names `--effort`: the subject was never the constraint.
#[test]
fn a_short_scratchpad_still_names_the_effort_lever_through_the_loop() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": "stop",
                             "message": {"reasoning": "thought and never answered"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        )
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    let x = &m["members"][0];
    assert_eq!(x["terminated_by"], "short_reply", "{m}");
    assert_eq!(x["reason"], "no_answer", "{m}");
    assert!(
        x["error"].as_str().unwrap_or_default().contains("--effort"),
        "a short scratchpad named no lever: {m}"
    );
    // Paid for, so they still reach disk.
    assert!(r.read("00-a_one.md").contains("never answered"));
}

#[test]
fn a_panel_that_only_thought_exits_six_and_says_failed() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": "stop", "message": {"reasoning": prose()}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        )
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    // Uniform `no_answer` has no class of its own, so `exit_for` gives the mixed-failure code.
    assert_eq!(r.code(), 6, "{}", r.stderr());
    assert_eq!(m["state"], "failed", "{m}");
    assert_eq!(m["members"][0]["terminated_by"], "reasoning_only", "{m}");
}

#[test]
fn a_reasoning_only_member_reaches_the_durable_log_as_rejected() {
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": "stop", "message": {"reasoning": prose()}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        )
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let log = std::fs::read_to_string(r.dir.path().join("bx/council/history.jsonl"))
        .expect("--out must not suppress the history log");
    let rows: Vec<serde_json::Value> = log
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("corrupt row: {e}\n{l}")))
        .collect();
    assert_eq!(rows.len(), 2, "one line per member: {log}");
    for row in &rows {
        assert_eq!(row["status"], "rejected", "{row}");
        assert_eq!(row["reason"], "no_answer", "{row}");
        assert_eq!(row["terminated_by"], "reasoning_only", "{row}");
    }
}

/// Also in the dry-run plan, JSON and text.
#[test]
fn a_pinned_provider_reaches_the_request_from_the_config() {
    let bodies = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let seen = Arc::clone(&bodies);
    let (port, _) = serve(Arc::new(move |path: &str, body: &str| match path {
        p if p.contains("/api/v1/key") => {
            (200, r#"{"data":{"limit":10,"limit_remaining":9}}"#.into())
        }
        p if p.contains("/llm/context") => search_hit(),
        p if p.contains("/v1/models") => (200, r#"{"data":[]}"#.into()),
        _ => {
            seen.lock().unwrap().push(body.to_string());
            reply(&prose())
        }
    }));
    let dir = TempDir::new().unwrap();
    let cfg_dir = dir.path().join("brave-search");
    std::fs::create_dir_all(&cfg_dir).unwrap();
    // Only `a/one` is pinned, so `b/two` proves the pin follows the model, not the run.
    std::fs::write(
        cfg_dir.join("council.json"),
        r#"{"providers":{"a/one":["together","novita"]},"zdr":false}"#,
    )
    .unwrap();
    let out = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "-m", "a/one,b/two"])
        .args(["--via", "openrouter", "--progress", "false"])
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let sent = bodies.lock().unwrap();
    let pinned: Vec<&String> = sent.iter().filter(|b| b.contains("\"a/one\"")).collect();
    let other: Vec<&String> = sent.iter().filter(|b| b.contains("\"b/two\"")).collect();
    assert!(
        !pinned.is_empty() && !other.is_empty(),
        "{} bodies",
        sent.len()
    );
    for b in pinned {
        let v: serde_json::Value = serde_json::from_str(b).unwrap();
        assert_eq!(
            v["provider"],
            serde_json::json!({"only": ["together", "novita"]}),
            "the pin did not reach the request"
        );
    }
    for b in other {
        let v: serde_json::Value = serde_json::from_str(b).unwrap();
        assert!(
            v["provider"].is_null(),
            "an unpinned member was constrained: {v}"
        );
    }
    drop(sent);

    // Absent, not `[]`, when unpinned: an empty list reads as "pinned to nothing".
    let plan = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "-m", "a/one,b/two"])
        .args(["--via", "openrouter", "--dry-run"])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&plan.stdout).unwrap();
    let ms = v["members"].as_array().expect("the plan lists its members");
    assert_eq!(
        ms[0]["providers"],
        serde_json::json!(["together", "novita"]),
        "{v}"
    );
    assert!(
        ms[1].get("providers").is_none(),
        "an unpinned member published a pin: {v}"
    );
    let text = bx("openrouter", port, dir.path())
        .args(["council", "SUBJECT", "-m", "a/one,b/two"])
        .args(["--via", "openrouter", "--dry-run", "--format", "text"])
        .bounded_output();
    let text = String::from_utf8_lossy(&text.stdout);
    let line = text
        .lines()
        .find(|l| l.contains("a/one"))
        .expect("the plan names each member");
    assert!(line.contains("providers=together,novita"), "{text}");
    assert!(
        !text
            .lines()
            .any(|l| l.contains("b/two") && l.contains("providers=")),
        "{text}"
    );
}

/// `--out <bare-id>` means `$PWD` to a run and the state root to `findings`, so the plan says which.
#[test]
fn a_dry_run_names_the_directory_it_blessed() {
    let (port, seen) = serve(mock(|_| reply(&prose())));
    let dir = TempDir::new().unwrap();
    let out = dir.path().join("somewhere").join("myrun");
    let plan = |args: &[&str]| {
        bx("openrouter", port, dir.path())
            .args(["council", "SUBJECT", "-m", "a/one,b/two"])
            .args(["--via", "openrouter", "--dry-run"])
            .args(args)
            .bounded_output()
    };

    let named = plan(&["--out", out.to_str().unwrap()]);
    let v: serde_json::Value = serde_json::from_slice(&named.stdout).unwrap();
    // Resolved: macOS temp dirs sit behind the `/var` symlink.
    let blessed = dir.path().canonicalize().unwrap().join("somewhere/myrun");
    assert_eq!(v["out"], blessed.display().to_string(), "{v}");

    // Null, not omitted: a minted directory is a known fact, unlike an absent key.
    let anon: serde_json::Value = serde_json::from_slice(&plan(&[]).stdout).unwrap();
    assert!(
        anon["out"].is_null(),
        "a minted directory cannot be named in advance: {anon}"
    );

    let text = plan(&["--format", "text", "--out", out.to_str().unwrap()]);
    let text = String::from_utf8_lossy(&text.stdout);
    assert!(
        text.contains(&format!("writes   {}", blessed.display())),
        "{text}"
    );
    let bare = plan(&["--format", "text"]);
    assert!(
        !String::from_utf8_lossy(&bare.stdout).contains("writes"),
        "nothing to disambiguate, so nothing to say"
    );

    // The free pre-flights ran, but nothing that generates.
    let bodies = seen.lock().unwrap();
    assert!(
        !bodies.iter().any(|b| b.contains("\"messages\"")),
        "a dry run bought a completion: {bodies:?}"
    );
}

/// By either spelling: `prune <run>` must not reap the default scope instead.
#[test]
fn a_sub_command_that_takes_no_run_refuses_one_rather_than_ignoring_it() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir_all(dir.path().join("bx").join("council")).unwrap();
    let call = |args: &[&str]| {
        bx_in(dir.path())
            .args(["council"])
            .args(args)
            .bounded_output()
    };

    for spelling in [
        vec!["prune", "20260101-120000-aaaaaa"],
        vec!["prune", "--out", "20260101-120000-aaaaaa"],
    ] {
        let p = call(&spelling);
        assert_eq!(p.status.code(), Some(2), "{spelling:?}");
        let e = String::from_utf8_lossy(&p.stderr);
        assert!(
            e.contains("takes no run, by either spelling"),
            "{spelling:?}: {e}"
        );
        assert!(
            !e.contains("state root"),
            "set-key acts on the config, not the state root: {e}"
        );
        assert!(e.contains("--dry-run"), "{spelling:?}: {e}");
    }
    // Not suggested to a caller who already passed it.
    let already = call(&["prune", "20260101-120000-aaaaaa", "--dry-run"]);
    assert_eq!(already.status.code(), Some(2));
    assert!(
        !String::from_utf8_lossy(&already.stderr).contains("--dry-run"),
        "{}",
        String::from_utf8_lossy(&already.stderr)
    );

    // `set-key` is safe here: the refusal precedes dispatch, which its first outputs would prove.
    for w in ["ls", "tasks", "doctor", "models", "set-key"] {
        for spelling in [
            vec![w, "20260101-120000-aaaaaa"],
            vec![w, "--out", "20260101-120000-aaaaaa"],
        ] {
            let out = call(&spelling);
            assert_eq!(out.status.code(), Some(2), "{spelling:?} accepted a run");
            let e = String::from_utf8_lossy(&out.stderr);
            assert!(e.contains("takes no run, by either spelling"), "{e}");
            assert!(
                !e.contains("Keys are saved to") && !e.contains("needs a terminal"),
                "the refusal ran after dispatch: {e}"
            );
        }
    }
    assert!(
        !dir.path()
            .join("brave-search")
            .join("council.json")
            .exists(),
        "set-key was entered"
    );
    assert_eq!(call(&["ls"]).status.code(), Some(0));
}

/// Turn 1: a long paid scratchpad and a call. Turn 2: a 30-character content preamble.
fn scratchpad_then_preamble(body: &str) -> (u16, String) {
    let (msg, finish) = if answered(body) {
        (
            serde_json::json!({"content": "Let me check the other module."}),
            "stop",
        )
    } else {
        (
            serde_json::json!({"reasoning": "paid thinking. ".repeat(4_000),
                               "tool_calls": [{"id": "c1", "function": {
                                   "name": "search", "arguments": "{\"query\":\"q\"}"}}]}),
            "tool_calls",
        )
    };
    (
        200,
        serde_json::json!({
            "model": "test/model",
            "choices": [{"finish_reason": finish, "message": msg}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
        })
        .to_string(),
    )
}

#[test]
fn a_short_answer_does_not_evict_a_long_paid_scratchpad_from_disk() {
    let (port, _) = serve(mock(scratchpad_then_preamble));
    let r = council(port, &["-m", "a/one,b/two"]);
    // The preamble wins selection, or step 6's re-prompt never fires.
    let published = r.read("00-a_one.md");
    assert!(published.contains("Let me check"), "{published}");

    // Retention is not selection: the partial never shrinks, and outlives a shorter reply.
    let partial = r.dir.path().join("run").join("00-a_one.md.partial");
    let kept = std::fs::read_to_string(&partial).unwrap_or_default();
    assert!(
        kept.contains("paid thinking."),
        "60,000 paid characters were evicted by a 30-character preamble; kept {} bytes",
        kept.len()
    );

    assert!(
        r.stderr().contains("00-a_one.md.partial"),
        "the surviving copy is announced nowhere: {}",
        r.stderr()
    );
}

#[test]
fn a_quiet_run_that_keeps_a_partial_says_nothing_about_it() {
    let (port, _) = serve(mock(scratchpad_then_preamble));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    assert!(r.read("00-a_one.md.partial").contains("paid thinking."));
    assert!(!r.stderr().contains(".md.partial"), "{}", r.stderr());
}

#[test]
fn an_ordinary_run_still_removes_its_partial() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    assert!(r.read("00-a_one.md").contains("grounded prose"));
    assert!(
        !r.dir
            .path()
            .join("run")
            .join("00-a_one.md.partial")
            .exists(),
        "a finished reply left its partial behind"
    );
    assert!(!r.stderr().contains(".md.partial"), "{}", r.stderr());
}

/// Retention compares the scrubbed body, not the raw character count.
#[test]
fn a_reply_the_scrubber_shortens_still_removes_its_partial() {
    let crlf = "## A. ANSWER\r\n".to_string() + &"grounded prose with evidence.\r\n".repeat(40);
    let (port, _) = serve(mock(move |_| reply(&crlf)));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    assert!(r.read("00-a_one.md").contains("grounded prose"));
    assert!(
        !r.read("00-a_one.md").contains('\r'),
        "the carriage returns were meant to be scrubbed out"
    );
    assert!(
        !r.dir
            .path()
            .join("run")
            .join("00-a_one.md.partial")
            .exists(),
        "a reply the scrubber shortened left a duplicate partial behind"
    );
    assert!(
        !r.stderr().contains("holds paid text"),
        "a duplicate was announced as unpublished text: {}",
        r.stderr()
    );
}

/// Nor through a directory or FIFO: a planted `/proc/self/environ` link would publish every key.
#[test]
fn findings_never_reads_a_member_file_through_a_link() {
    let dir = TempDir::new().unwrap();
    let run = dir
        .path()
        .join("bx")
        .join("council")
        .join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("brief.md"), "subject").unwrap();
    // One real reply, so the command does not refuse outright.
    std::fs::write(run.join("00-a_one.md"), "## A. ANSWER\nreal\n").unwrap();
    // The FIFOs hang a check done by `stat` or at `open`; `findings` has no deadline of its own.
    std::os::unix::fs::symlink("/proc/self/environ", run.join("09-environ.md")).unwrap();
    std::os::unix::fs::symlink("/etc/passwd", run.join("08-passwd.md")).unwrap();
    std::fs::create_dir(run.join("07-dir.md")).unwrap();
    let fifo = run.join("06-fifo.md");
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("mkfifo")
            .success()
    );
    std::os::unix::fs::symlink(&fifo, run.join("05-tofifo.md")).unwrap();

    // Polled, so a blocking open fails here rather than hanging the suite.
    let mut child = bx_in(dir.path())
        .args(["council", "findings", "20260101-120000-aaaaaa"])
        .env("OPENROUTER_API_KEY", "sk-or-v1-CANARYVALUE")
        .env("BRAVE_SEARCH_API_KEY", "BSAtestkey000")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while child.try_wait().unwrap().is_none() {
        if std::time::Instant::now() >= deadline {
            child.kill().ok();
            panic!("findings blocked on a FIFO named like a member reply");
        }
        thread::sleep(std::time::Duration::from_millis(50));
    }
    let out = finish(child);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("CANARYVALUE"),
        "a symlinked member file put a credential on stdout"
    );
    assert!(
        !stdout.contains("root:"),
        "a symlink escaped the run directory"
    );
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["members"].as_array().unwrap().len(), 1, "{v}");
    assert!(
        v["members"][0]["file"]
            .as_str()
            .unwrap()
            .ends_with("00-a_one.md"),
        "{v}"
    );
}

/// `member_row` reads manifest-named files, not `member_files`' list, so it needs its own guard.
#[test]
fn findings_never_reads_a_manifest_named_file_through_a_link() {
    let dir = TempDir::new().unwrap();
    let run = dir
        .path()
        .join("bx")
        .join("council")
        .join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("brief.md"), "subject").unwrap();
    std::fs::write(run.join("00-a_one.md"), "## A. ANSWER\nreal\n").unwrap();
    std::os::unix::fs::symlink("/proc/self/environ", run.join("01-environ.md")).unwrap();
    let fifo = run.join("02-fifo.md");
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("mkfifo")
            .success()
    );
    // Undecodable is not refused: it is paid work on disk, so the row still names it.
    std::fs::write(run.join("03-raw.md"), [0xff, 0xfe, b'x']).unwrap();
    std::fs::write(
        run.join("manifest.json"),
        r#"{"task":"t","members":[{"model":"a/one","status":"delivered","file":"00-a_one.md"},
            {"model":"b/two","status":"delivered","file":"01-environ.md"},
            {"model":"c/three","status":"delivered","file":"02-fifo.md"},
            {"model":"d/four","status":"delivered","file":"03-raw.md"}]}"#,
    )
    .unwrap();

    // Polled, so a blocking open fails here rather than hanging the suite.
    let mut child = bx_in(dir.path())
        .args(["council", "findings", "20260101-120000-aaaaaa"])
        .env("OPENROUTER_API_KEY", "sk-or-v1-CANARYVALUE")
        .env("BRAVE_SEARCH_API_KEY", "BSAtestkey000")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while child.try_wait().unwrap().is_none() {
        if std::time::Instant::now() >= deadline {
            child.kill().ok();
            panic!("findings blocked on a FIFO a manifest row named");
        }
        thread::sleep(std::time::Duration::from_millis(50));
    }
    let out = finish(child);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("CANARYVALUE"),
        "a manifest named a link and the row published a credential: {stdout}"
    );
    // Rows stay, but a refused one names no path, or the caller would open it instead.
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let m = v["members"].as_array().unwrap();
    assert_eq!(m.len(), 4, "{v}");
    assert!(
        m[0]["sections"].is_array() && m[0]["file"].is_string(),
        "the real reply lost its text: {v}"
    );
    for (i, what) in [(1, "a link"), (2, "a FIFO")] {
        assert!(m[i]["sections"].is_null(), "{what} was read: {v}");
        assert!(m[i]["file"].is_null(), "{what} was named as a reply: {v}");
    }
    assert!(m[3]["sections"].is_null(), "invalid UTF-8 was read: {v}");
    assert!(
        m[3]["file"]
            .as_str()
            .is_some_and(|f| f.ends_with("03-raw.md")),
        "a reply that could not be decoded was reported as absent: {v}"
    );
}

#[cfg(unix)]
#[test]
fn ls_is_not_hung_by_a_run_whose_brief_is_not_a_file() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    let good = root.join("20260101-120000-aaaaaa");
    let bad = root.join("20260101-120001-bbbbbb");
    std::fs::create_dir_all(&good).unwrap();
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(good.join("brief.md"), "subject").unwrap();
    assert!(
        Command::new("mkfifo")
            .arg(bad.join("brief.md"))
            .status()
            .expect("mkfifo")
            .success()
    );

    // Polled, so a blocking open fails here rather than hanging the suite.
    let mut child = bx_in(dir.path())
        .args(["council", "ls"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while child.try_wait().unwrap().is_none() {
        if std::time::Instant::now() >= deadline {
            child.kill().ok();
            panic!("ls blocked on a FIFO brief.md and lost every other run with it");
        }
        thread::sleep(std::time::Duration::from_millis(50));
    }
    let out = finish(child);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    // The unreadable one has no fingerprint.
    assert_eq!(v["runs"].as_array().unwrap().len(), 2, "{v}");
    assert!(
        v["runs"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["prompt_sha"] == "-".repeat(16)),
        "{v}"
    );
}

/// Rows may repeat one file and each is read, so unbounded rows exhaust memory.
#[test]
fn findings_reads_no_more_manifest_rows_than_a_panel_can_hold() {
    let dir = TempDir::new().unwrap();
    let run = dir
        .path()
        .join("bx")
        .join("council")
        .join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("brief.md"), "subject").unwrap();
    std::fs::write(run.join("00-a_one.md"), "## A. X\nx\n").unwrap();
    let rows = (0..500)
        .map(|i| format!(r#"{{"model":"m{i}","status":"delivered","file":"00-a_one.md"}}"#))
        .collect::<Vec<_>>()
        .join(",");
    std::fs::write(
        run.join("manifest.json"),
        format!(r#"{{"task":"t","members":[{rows}]}}"#),
    )
    .unwrap();

    let out = bx_in(dir.path())
        .args(["council", "findings", "20260101-120000-aaaaaa"])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["members"].as_array().unwrap().len(), 32, "{v}");
}

#[test]
fn findings_reads_no_more_members_than_a_panel_can_hold() {
    let dir = TempDir::new().unwrap();
    let run = dir
        .path()
        .join("bx")
        .join("council")
        .join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("brief.md"), "subject").unwrap();
    for i in 0..200 {
        std::fs::write(run.join(format!("{:02}-m{i}.md", i % 100)), "## A. X\nx\n").unwrap();
    }
    let out = bx_in(dir.path())
        .args(["council", "findings", "20260101-120000-aaaaaa"])
        .env("BRAVE_SEARCH_API_KEY", "BSAtestkey000")
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        v["members"].as_array().unwrap().len(),
        32,
        "{}",
        v["members"]
    );
}

/// A copy (fresh `brief.md`, old id) is aged by its brief.
#[test]
fn prune_age_after_a_copy_only_delays_removal() {
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("bx/council/20200101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("manifest.json"), "{}").unwrap();
    std::fs::write(run.join("brief.md"), "b").unwrap();
    let out = bx_in(dir.path())
        .args(["council", "prune"])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["removed"], serde_json::json!([]), "{v}");
    assert!(run.join("brief.md").exists());
}

/// One that holds nothing is reclaimed.
#[test]
fn prune_keeps_a_manifest_less_run_that_holds_a_reply_even_with_no_status() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    let age = |p: &std::path::Path| {
        let f = std::fs::File::options().write(true).open(p).unwrap();
        let when = std::time::SystemTime::now() - std::time::Duration::from_secs(60 * 86_400);
        f.set_times(std::fs::FileTimes::new().set_modified(when))
            .unwrap();
    };
    let paid = root.join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&paid).unwrap();
    std::fs::write(paid.join("brief.md"), "b").unwrap();
    std::fs::write(paid.join("00-z_glm.md.partial"), "paid output").unwrap();
    age(&paid.join("brief.md"));
    let empty = root.join("20260101-120000-bbbbbb");
    std::fs::create_dir_all(&empty).unwrap();
    std::fs::write(empty.join("brief.md"), "b").unwrap();
    age(&empty.join("brief.md"));

    let out = bx_in(dir.path())
        .args(["council", "prune"])
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(paid.exists(), "a run holding paid output was deleted");
    assert!(
        paid.join("00-z_glm.md.partial").exists(),
        "the paid reply was destroyed"
    );
    assert!(
        !empty.exists(),
        "an abandoned run that spent nothing was not reclaimed"
    );
}

/// A failed rename leaves the paid text in `write_private`'s temp, for a reply or a partial.
#[test]
fn prune_keeps_an_old_unfinished_run_whose_only_paid_text_is_a_publish_temp() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(40 * 86_400);
    let runs: Vec<_> = ["01-m.tmp1-2", "01-m.md.tmp1-2"]
        .iter()
        .enumerate()
        .map(|(i, temp)| {
            let run = root.join(format!("20260101-120000-aaaaa{i}"));
            std::fs::create_dir_all(&run).unwrap();
            std::fs::write(run.join("status.json"), "{}").unwrap();
            std::fs::write(run.join(temp), "paid review text").unwrap();
            std::fs::write(run.join("brief.md"), "b").unwrap();
            std::fs::File::options()
                .write(true)
                .open(run.join("brief.md"))
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(old))
                .unwrap();
            run
        })
        .collect();

    let out = bx_in(dir.path())
        .args(["council", "prune"])
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["removed"], serde_json::json!([]), "{v}");
    assert_eq!(
        v["kept_unfinished"].as_array().map(Vec::len),
        Some(2),
        "{v}"
    );
    assert!(runs.iter().all(|r| r.exists()), "{v}");
}

/// In both forms.
#[test]
fn prune_names_old_unfinished_runs_it_keeps() {
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("bx/council/20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("00-a_one.md.partial"), "paid output").unwrap();
    std::fs::write(run.join("brief.md"), "b").unwrap();
    std::fs::File::options()
        .write(true)
        .open(run.join("brief.md"))
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(
            std::time::SystemTime::now() - std::time::Duration::from_secs(60 * 86_400),
        ))
        .unwrap();
    let prune = |format: &str| {
        bx_in(dir.path())
            .args(["council", "prune", "--format", format])
            .bounded_output()
            .stdout
    };
    let v: serde_json::Value = serde_json::from_slice(&prune("json")).unwrap();
    assert_eq!(
        v["kept_unfinished"][0]["run"], "20260101-120000-aaaaaa",
        "{v}"
    );
    let text = String::from_utf8(prune("text")).unwrap();
    assert!(
        text.contains("kept 20260101-120000-aaaaaa: unfinished, holds paid output"),
        "{text}"
    );
    assert!(run.exists());
}

#[cfg(unix)]
#[test]
fn a_run_whose_reply_scan_fails_is_kept_by_prune() {
    use std::os::unix::fs::PermissionsExt;
    // SAFETY: geteuid has no preconditions.
    if unsafe { libc::geteuid() } == 0 {
        return; // root reads any directory
    }
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("bx/council/20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("00-a_one.md"), "paid output").unwrap();
    std::fs::write(run.join("brief.md"), "b").unwrap();
    age_by_days(&run.join("brief.md"), 60);
    // Enter and write, never list: the lock can be taken, the reply cannot be seen.
    std::fs::set_permissions(&run, std::fs::Permissions::from_mode(0o300)).unwrap();
    let (code, v) = prune_json(dir.path(), &[]);
    std::fs::set_permissions(&run, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert_eq!(code, Some(0), "{v}");
    assert!(run.join("00-a_one.md").exists(), "{v}");
    assert_eq!(v["failed"], serde_json::json!([]), "{v}");
}

/// `findings` still displays only names this binary mints.
#[test]
fn prune_judges_paid_output_by_shape_not_by_the_alphabet_findings_displays() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    let aged = |name: &str, reply: Option<&std::ffi::OsStr>| {
        let run = root.join(name);
        std::fs::create_dir_all(&run).unwrap();
        if let Some(r) = reply {
            std::fs::write(run.join(r), "paid review text").unwrap();
        }
        std::fs::write(run.join("brief.md"), "subject").unwrap();
        let f = std::fs::File::options()
            .write(true)
            .open(run.join("brief.md"))
            .unwrap();
        let when = std::time::SystemTime::now() - std::time::Duration::from_secs(60 * 86_400);
        f.set_times(std::fs::FileTimes::new().set_modified(when))
            .unwrap();
        run
    };
    let o = std::ffi::OsStr::new;
    // The non-UTF-8 name is invisible to `member_files`, which goes through `to_str`.
    let spaced = aged("20260101-120000-aaaaaa", Some(o("00-my file.md")));
    let accented = aged("20260101-120000-bbbbbb", Some(o("00-café.md")));
    #[cfg(unix)]
    let raw = {
        use std::os::unix::ffi::OsStrExt;
        aged(
            "20260101-120000-cccccc",
            Some(std::ffi::OsStr::from_bytes(b"00-\xff\xfe.md")),
        )
    };
    let ordinary = aged("20260101-120000-dddddd", Some(o("00-a_one.md")));
    let empty = aged("20260101-120000-eeeeee", None);

    let out = bx_in(dir.path())
        .args(["council", "prune"])
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    for (run, what) in [
        (&spaced, "a space"),
        (&accented, "a non-ASCII character"),
        (&ordinary, "an ordinary name"),
    ] {
        assert!(run.exists(), "a run whose reply has {what} was deleted");
    }
    #[cfg(unix)]
    assert!(raw.exists(), "a run whose reply is not UTF-8 was deleted");
    assert!(
        !empty.exists(),
        "the split must not keep a run that spent nothing"
    );

    // Reading stays strict, so nothing can forge a row through an odd name.
    let read = bx_in(dir.path())
        .args(["council", "findings", "20260101-120000-aaaaaa"])
        .env("BRAVE_SEARCH_API_KEY", "BSAtestkey000")
        .bounded_output();
    assert_eq!(
        read.status.code(),
        Some(2),
        "a name outside the alphabet is not a member to display: {}",
        String::from_utf8_lossy(&read.stdout)
    );
}

/// Backdates `p`'s mtime by `days`.
#[cfg(unix)]
fn age_by_days(p: &std::path::Path, days: i64) {
    let f = std::fs::File::options().write(true).open(p).unwrap();
    f.set_times(std::fs::FileTimes::new().set_modified(when_days(days)))
        .unwrap();
}

/// `days` before now, or after it when negative.
#[cfg(unix)]
fn when_days(days: i64) -> std::time::SystemTime {
    let d = std::time::Duration::from_secs(days.unsigned_abs() * 86_400);
    if days < 0 {
        std::time::SystemTime::now() + d
    } else {
        std::time::SystemTime::now() - d
    }
}

#[cfg(unix)]
fn prune_json(state: &std::path::Path, extra: &[&str]) -> (Option<i32>, serde_json::Value) {
    let out = bx_in(state)
        .args(["council", "prune"])
        .args(extra)
        .bounded_output();
    let v = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "{e}: {} / {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    (out.status.code(), v)
}

/// The canaries sit outside the state root, where only a walk that follows links reaches them.
#[cfg(unix)]
#[test]
fn prune_unlinks_only_inside_the_run_it_removes() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    let canary_dir = dir.path().join("canary-dir");
    std::fs::create_dir_all(&canary_dir).unwrap();
    std::fs::write(canary_dir.join("keep.md"), "paid output").unwrap();
    let canary_file = dir.path().join("canary-file.md");
    std::fs::write(&canary_file, "paid output").unwrap();
    // A copy, so a mutant run as root cannot destroy the real `/etc/passwd`.
    let outside = dir.path().join("system-file");
    std::fs::copy("/etc/passwd", &outside)
        .or_else(|_| std::fs::write(&outside, "root:x:0:0:root:/root:/bin/sh\n").map(|()| 0))
        .expect("a file outside the run");

    let run = root.join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("manifest.json"), "{}").unwrap();
    std::fs::write(run.join("brief.md"), "b").unwrap();
    age_by_days(&run.join("brief.md"), 60);
    std::os::unix::fs::symlink(&canary_dir, run.join("link-to-dir")).unwrap();
    std::os::unix::fs::symlink(&canary_file, run.join("link-to-file")).unwrap();
    std::os::unix::fs::symlink(&outside, run.join("link-to-system-file")).unwrap();

    let (code, v) = prune_json(dir.path(), &[]);
    assert_eq!(code, Some(0), "{v}");
    assert!(
        !run.exists(),
        "the run was not removed, so the canaries prove nothing: {v}"
    );
    assert!(
        canary_dir.join("keep.md").exists(),
        "a link out of the run was followed and its target emptied"
    );
    assert!(canary_file.exists(), "a link out of the run was unlinked");
    assert!(
        outside.exists(),
        "a link to a file outside the run was followed and unlinked"
    );
}

/// A lock older than any run keeps nothing. `--dry-run` takes no lock: one would move a
/// brief-less run's age, its directory mtime.
#[cfg(unix)]
#[test]
fn prune_never_removes_a_run_that_holds_its_lock() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    let run = |id: &str, lock_days: Option<i64>, reply: bool| {
        let run = root.join(id);
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(run.join("brief.md"), "b").unwrap();
        age_by_days(&run.join("brief.md"), 60);
        if reply {
            std::fs::write(run.join("00-a_one.md"), "a paid review").unwrap();
        }
        if let Some(days) = lock_days {
            std::fs::write(run.join(".council-lock"), DEAD_PID_LOCK).unwrap();
            age_by_days(&run.join(".council-lock"), days);
        }
        run
    };
    let held = run("20260101-120000-aaaaaa", Some(0), false);
    let dead = run("20260101-120000-bbbbbb", Some(60), false);
    let paid = run("20260101-120000-cccccc", None, true);
    let briefless = root.join("20260101-120000-dddddd");
    std::fs::create_dir(&briefless).unwrap();
    std::fs::File::open(&briefless)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(when_days(60)))
        .unwrap();

    let (code, v) = prune_json(dir.path(), &["--dry-run"]);
    assert_eq!(code, Some(0), "{v}");
    assert_eq!(
        v["locked"],
        serde_json::json!([held.file_name().unwrap().to_str()]),
        "{v}"
    );
    assert_eq!(v["removed"].as_array().map(Vec::len), Some(2), "{v}");

    let (code, v) = prune_json(dir.path(), &[]);
    assert_eq!(code, Some(0), "{v}");
    assert!(
        held.join("brief.md").exists(),
        "a run holding its lock was removed: {v}"
    );
    assert_eq!(
        v["locked"],
        serde_json::json!([held.file_name().unwrap().to_str()]),
        "{v}"
    );
    assert!(
        !dead.exists(),
        "a lock older than any run kept a dead one: {v}"
    );
    assert!(
        !briefless.exists(),
        "the dry run moved a brief-less run's age: {v}"
    );
    assert!(paid.exists(), "{v}");
    assert!(
        !paid.join(".council-lock").exists(),
        "prune left its lock on a run it kept"
    );
}

/// Above any `pid_max`, so no process ever has it.
#[cfg(unix)]
const DEAD_PID_LOCK: &str = "pid 2147483647 at 2026-01-01T00:00:00Z\n";

/// A clock that jumps ages a live run's lock; the pid it names still decides.
#[cfg(unix)]
#[test]
fn prune_keeps_a_run_whose_aged_lock_names_a_live_pid() {
    let dir = TempDir::new().unwrap();
    let run = dir.path().join("bx/council/20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("brief.md"), "b").unwrap();
    age_by_days(&run.join("brief.md"), 31);
    let lock = run.join(".council-lock");
    std::fs::write(
        &lock,
        format!("pid {} at 2026-01-01T00:00:00Z\n", std::process::id()),
    )
    .unwrap();
    age_by_days(&lock, 31);
    for extra in [&["--dry-run"][..], &[]] {
        let (code, v) = prune_json(dir.path(), extra);
        assert_eq!(code, Some(0), "{v}");
        assert_eq!(
            v["locked"],
            serde_json::json!([run.file_name().unwrap().to_str()]),
            "{v}"
        );
        assert!(
            run.join("brief.md").exists(),
            "{extra:?} removed a live run: {v}"
        );
    }
}

/// By lstat, never by content: anything but a regular file keeps an old run's reply.
#[cfg(unix)]
#[test]
fn prune_judges_the_manifest_as_a_file_or_not_at_all() {
    use std::os::unix::fs::PermissionsExt;

    #[derive(Debug, Clone, Copy)]
    enum Plant {
        /// A link to a real sibling manifest: `metadata()` would call it a file, lstat does not.
        Link,
        Dangling,
        Dir,
        /// If `prune` ever opens it, this test hangs.
        Fifo,
        /// Regular files from here on, accepted unread.
        Mode000,
        Empty,
        NotJson,
        /// A second name for a file outside the run.
        HardLink,
    }

    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    let outside = dir.path().join("outside-the-run.json");
    std::fs::write(&outside, r#"{"done":1,"total":4}"#).unwrap();

    // (planted, removed)
    let cases: &[(Plant, bool)] = &[
        (Plant::Link, false),
        (Plant::Dangling, false),
        (Plant::Dir, false),
        (Plant::Fifo, false),
        (Plant::Mode000, true),
        (Plant::Empty, true),
        (Plant::NotJson, true),
        (Plant::HardLink, true),
    ];
    let id_of = |i: usize| format!("20260101-1200{i:02}-{i:06x}");

    for (i, (plant, _)) in cases.iter().enumerate() {
        let run = root.join(id_of(i));
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(run.join("brief.md"), "b").unwrap();
        age_by_days(&run.join("brief.md"), 60);
        std::fs::write(run.join("00-a_one.md"), "a paid review").unwrap();
        let p = run.join("manifest.json");
        match plant {
            Plant::Link => {
                let real = run.join("planted-target.json");
                std::fs::write(&real, r#"{"state":"ok","done":1,"total":4}"#).unwrap();
                std::os::unix::fs::symlink(&real, &p).unwrap();
            }
            Plant::Dangling => {
                std::os::unix::fs::symlink(run.join("never-written.json"), &p).unwrap();
            }
            Plant::Dir => std::fs::create_dir(&p).unwrap(),
            Plant::Fifo => assert!(
                std::process::Command::new("mkfifo")
                    .arg(&p)
                    .status()
                    .is_ok_and(|s| s.success()),
                "mkfifo is required for this test"
            ),
            Plant::Mode000 => {
                std::fs::write(&p, r#"{"done":1,"total":4}"#).unwrap();
                std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o000)).unwrap();
            }
            Plant::Empty => std::fs::write(&p, "").unwrap(),
            Plant::NotJson => std::fs::write(&p, "\u{1b}[2Jnot json, and never was").unwrap(),
            Plant::HardLink => std::fs::hard_link(&outside, &p).unwrap(),
        }
    }

    let (code, v) = prune_json(dir.path(), &[]);
    assert_eq!(code, Some(0), "{v}");
    let removed: std::collections::HashSet<&str> = v["removed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["run"].as_str().unwrap())
        .collect();
    for (i, (plant, expected)) in cases.iter().enumerate() {
        let id = id_of(i);
        let what = format!("manifest.json as {plant:?}");
        assert_eq!(
            removed.contains(id.as_str()),
            *expected,
            "{what} — reported wrongly: {v}"
        );
        assert_eq!(
            !root.join(&id).exists(),
            *expected,
            "{what} — on disk the opposite of what was reported"
        );
    }
    assert!(
        outside.exists(),
        "a hard link inside a run reached a file outside it"
    );
}

/// A link, directory or FIFO there is ignored, and a future mtime reads young.
#[cfg(unix)]
#[test]
fn prune_cannot_age_a_run_whose_brief_is_not_a_plain_file() {
    #[derive(Debug, Clone, Copy)]
    enum Brief {
        /// The ENOSPC shape: the first write never landed.
        Missing,
        /// Aged 60 days, so only an `lstat` `is_file` check keeps these runs.
        Link,
        Dir,
        Fifo,
        /// `SystemTime::elapsed` is `Err`, so the run is never old.
        Future,
        /// A regular file sharing an old mtime, so it does age the run.
        HardLink,
        /// The control, so a `prune` that removes nothing fails.
        Plain,
    }

    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    // `touch -r` ages a FIFO without opening it; `File::set_times` would block on the open.
    let reference = dir.path().join("aged-reference");
    std::fs::write(&reference, "r").unwrap();
    age_by_days(&reference, 60);
    // Outside the run: removal must unlink the name, not the inode.
    let outside = dir.path().join("outside-brief.md");
    std::fs::write(&outside, "paid output").unwrap();
    age_by_days(&outside, 60);

    let cases = [
        (Brief::Missing, false),
        (Brief::Link, false),
        (Brief::Dir, false),
        (Brief::Fifo, false),
        (Brief::Future, false),
        (Brief::HardLink, true),
        (Brief::Plain, true),
    ];
    let id_of = |i: usize| format!("20260101-1200{i:02}-{i:06x}");

    for (i, (brief, _)) in cases.iter().enumerate() {
        let run = root.join(id_of(i));
        std::fs::create_dir_all(&run).unwrap();
        // Finished, so only `brief.md` can keep it.
        std::fs::write(run.join("manifest.json"), "{}").unwrap();
        let p = run.join("brief.md");
        match brief {
            Brief::Missing => {}
            Brief::Link => {
                let real = run.join("brief-target.md");
                std::fs::write(&real, "b").unwrap();
                age_by_days(&real, 60);
                std::os::unix::fs::symlink(&real, &p).unwrap();
            }
            Brief::Dir => {
                std::fs::create_dir(&p).unwrap();
                std::fs::File::open(&p)
                    .unwrap()
                    .set_times(std::fs::FileTimes::new().set_modified(when_days(60)))
                    .unwrap();
            }
            Brief::Fifo => {
                assert!(
                    std::process::Command::new("mkfifo")
                        .arg(&p)
                        .status()
                        .is_ok_and(|s| s.success()),
                    "mkfifo is required for this test"
                );
                assert!(
                    std::process::Command::new("touch")
                        .arg("-r")
                        .args([&reference, &p])
                        .status()
                        .is_ok_and(|s| s.success()),
                    "touch -r is required for this test"
                );
            }
            Brief::Future => {
                std::fs::write(&p, "b").unwrap();
                age_by_days(&p, -60);
            }
            Brief::HardLink => std::fs::hard_link(&outside, &p).unwrap(),
            Brief::Plain => {
                std::fs::write(&p, "b").unwrap();
                age_by_days(&p, 60);
            }
        }
    }

    let (code, v) = prune_json(dir.path(), &[]);
    assert_eq!(code, Some(0), "{v}");
    let removed: std::collections::HashSet<&str> = v["removed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["run"].as_str().unwrap())
        .collect();
    for (i, (brief, expected)) in cases.iter().enumerate() {
        let id = id_of(i);
        assert_eq!(
            removed.contains(id.as_str()),
            *expected,
            "brief.md as {brief:?} — reported wrongly: {v}"
        );
        assert_eq!(
            !root.join(&id).exists(),
            *expected,
            "brief.md as {brief:?} — on disk the opposite of what was reported"
        );
    }
    assert!(
        outside.exists(),
        "removing a run unlinked a name it shared, not a name it owned"
    );
}

#[cfg(unix)]
#[test]
fn prune_skips_a_run_directory_that_is_itself_a_symlink() {
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    // Shaped like a run `prune` would take: finished and old.
    let elsewhere = dir.path().join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::fs::write(elsewhere.join("manifest.json"), "{}").unwrap();
    std::fs::write(elsewhere.join("00-a_one.md"), "a paid review").unwrap();
    std::fs::write(elsewhere.join("brief.md"), "b").unwrap();
    age_by_days(&elsewhere.join("brief.md"), 60);

    std::fs::create_dir_all(&root).unwrap();
    let link = root.join("20260101-120000-aaaaaa");
    std::os::unix::fs::symlink(&elsewhere, &link).unwrap();
    // So a `prune` that removes nothing fails.
    let real = root.join("20260101-120000-bbbbbb");
    std::fs::create_dir_all(&real).unwrap();
    std::fs::write(real.join("manifest.json"), "{}").unwrap();
    std::fs::write(real.join("brief.md"), "b").unwrap();
    age_by_days(&real.join("brief.md"), 60);

    let (code, v) = prune_json(dir.path(), &[]);
    assert_eq!(code, Some(0), "{v}");
    assert_eq!(v["removed"][0]["run"], "20260101-120000-bbbbbb", "{v}");
    assert_eq!(v["removed"].as_array().unwrap().len(), 1, "{v}");
    assert!(
        elsewhere.join("00-a_one.md").exists(),
        "the link was walked and the paid reply behind it destroyed"
    );
    assert!(elsewhere.join("manifest.json").exists());
    assert!(
        std::fs::symlink_metadata(&link).is_ok_and(|m| m.is_symlink()),
        "the link itself was not prune's to remove either"
    );
}

/// `remove_dir_all` stops at the first entry it cannot unlink, so a partial failure exits 2.
#[cfg(unix)]
#[test]
fn a_prune_that_fails_part_way_names_what_it_destroyed() {
    use std::os::unix::fs::PermissionsExt;
    let dir = TempDir::new().unwrap();
    let root = dir.path().join("bx").join("council");
    let run = root.join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("manifest.json"), "{}").unwrap();
    std::fs::write(run.join("brief.md"), "b").unwrap();
    for i in 0..12 {
        std::fs::write(run.join(format!("{i:02}-paid.md")), "a paid review").unwrap();
    }
    // Readable but not unlinkable from, like another user's artifact or a mount point.
    let locked = run.join("locked");
    std::fs::create_dir(&locked).unwrap();
    std::fs::write(locked.join("held.md"), "x").unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o500)).unwrap();
    age_by_days(&run.join("brief.md"), 60);

    // readdir order is fixed by now, so the destroyed count is exact on any filesystem.
    let order: Vec<String> = std::fs::read_dir(&run)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let doomed: Vec<&String> = order
        .iter()
        .take_while(|n| n.as_str() != "locked")
        .collect();
    let doomed_paid = doomed.iter().filter(|n| n.ends_with("-paid.md")).count();

    // A dry run touches nothing, so it cannot foresee the failure.
    let (dry_code, dry) = prune_json(dir.path(), &["--dry-run"]);
    assert_eq!(dry_code, Some(0), "{dry}");
    assert_eq!(dry["removed"][0]["run"], "20260101-120000-aaaaaa", "{dry}");
    assert!(dry["bytes_freed"].as_u64().unwrap() > 0, "{dry}");

    let (code, v) = prune_json(dir.path(), &[]);
    assert_eq!(code, Some(2), "a partial failure is not a success: {v}");
    assert!(v["removed"].as_array().unwrap().is_empty(), "{v}");
    assert_eq!(v["bytes_freed"], 0, "{v}");
    let left = (0..12)
        .filter(|i| run.join(format!("{i:02}-paid.md")).exists())
        .count();
    assert_eq!(
        left,
        12 - doomed_paid,
        "everything readdir listed before the entry that failed is gone: {order:?}"
    );
    let warn = v["failed"][0].as_str().unwrap_or_default();
    if doomed.is_empty() {
        assert!(warn.contains("nothing in it was removed"), "{warn}");
    } else {
        assert!(
            warn.contains(&format!(
                "{} of its {} entries were destroyed",
                doomed.len(),
                order.len()
            )),
            "the report has to say what it destroyed: {warn}"
        );
        assert!(
            warn.contains(doomed[0].as_str()),
            "and name it: {warn} / {doomed:?}"
        );
    }
    // The damaged run still lists, which is what makes the loss silent.
    let ls = bx_in(dir.path()).args(["council", "ls"]).bounded_output();
    assert_eq!(ls.status.code(), Some(0));
    assert!(
        String::from_utf8_lossy(&ls.stdout).contains("20260101-120000-aaaaaa"),
        "the run reads perfectly well, which is what makes the loss silent"
    );
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o700)).unwrap();
}

/// The transport fault is the more specific label; only the loop orders the two.
#[test]
fn a_discarded_draft_outranks_a_reasoning_only_one_through_the_loop() {
    // Even turns: retryable thinking, marked both discarded and reasoning-only. Odd turns: empty,
    // so neither flag clears and step 7 must choose.
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let (port, _) = serve(mock(move |_| {
        let n = seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (finish, msg) = if n.is_multiple_of(2) {
            (
                "error",
                serde_json::json!({"reasoning": "paid thinking. ".repeat(200)}),
            )
        } else {
            ("stop", serde_json::json!({"content": ""}))
        };
        (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": finish, "message": msg}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        )
    }));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let m = r.manifest();
    let member = &m["members"][0];
    assert_eq!(member["terminated_by"], "discarded_draft", "{m}");
    assert_eq!(member["finish_reason"], "error", "{m}");
    assert_ne!(member["status"], "delivered", "{m}");
    assert!(r.read("00-a_one.md").contains("paid thinking."));
}

/// One counter-example per condition, since on a tidy fixture they mask each other.
#[test]
fn member_files_selects_on_every_part_of_the_name() {
    use std::os::unix::ffi::OsStrExt;
    let dir = TempDir::new().unwrap();
    let run = dir
        .path()
        .join("bx")
        .join("council")
        .join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("brief.md"), "subject").unwrap();
    for name in [
        "00-a_one.md",        // the shape that must be kept
        "01-b_two.md",        // and a second, to prove ordering
        "00-.md",             // the shortest name the rule accepts: six bytes, empty slug
        "00-x.md.partial.md", // `.md` wins the suffix test; `.md.partial` is not the ending
        "1-x.md",             // one digit, not two
        "ab-x.md",            // neither position a digit
        "a0-x.md",            // second is a digit, first is not
        "0a-x.md",            // first is a digit, second is not
        "00x.md",             // no separator
        "00-x.txt",           // wrong suffix
        "00-",                // prefix only, no suffix
        "00-\u{e9}.md",       // outside `slugify`'s alphabet, so not written here
        "brief.md",           // already written above; not a member
        "index.md",           // nor this
    ] {
        let p = run.join(name);
        if !p.exists() {
            std::fs::write(&p, "## A. X\nx\n").unwrap();
        }
    }
    // Not UTF-8: dropped by `to_str`, while the good names prove the listing itself worked.
    std::fs::write(
        run.join(std::ffi::OsStr::from_bytes(b"14-\xff\xfe.md")),
        "## A. X\nx\n",
    )
    .unwrap();
    let out = bx_in(dir.path())
        .args(["council", "findings", "20260101-120000-aaaaaa"])
        .env("BRAVE_SEARCH_API_KEY", "BSAtestkey000")
        .bounded_output();
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let files: Vec<String> = v["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            std::path::Path::new(m["file"].as_str().unwrap())
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        files,
        ["00-.md", "00-a_one.md", "00-x.md.partial.md", "01-b_two.md"],
        "{v}"
    );
}

/// A run holding `files`, read by `findings` by run id, as JSON and as text.
fn planted(files: &[(&str, &str)]) -> (TempDir, serde_json::Value, String) {
    let dir = TempDir::new().unwrap();
    let run = dir
        .path()
        .join("bx")
        .join("council")
        .join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("brief.md"), "subject").unwrap();
    for (name, body) in files {
        std::fs::write(run.join(name), body).unwrap();
    }
    let read = |args: &[&str]| {
        bx_in(dir.path())
            .args(["council", "findings", "20260101-120000-aaaaaa"])
            .args(args)
            .env("BRAVE_SEARCH_API_KEY", "BSAtestkey000")
            .bounded_output()
    };
    let json = read(&[]);
    let text = read(&["--format", "text"]);
    (
        dir,
        serde_json::from_slice(&json.stdout).unwrap_or_else(|e| {
            panic!(
                "findings is not JSON: {e}\n{}",
                String::from_utf8_lossy(&json.stderr)
            )
        }),
        String::from_utf8_lossy(&text.stdout).into_owned(),
    )
}

/// Not a panel of zero.
#[test]
fn a_manifest_whose_member_list_is_not_a_list_is_an_error() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let dir = r.dir.path().join("run");
    std::fs::write(
        dir.join("manifest.json"),
        r#"{"task":"design-review","members":3}"#,
    )
    .unwrap();
    let out = bx_in(r.dir.path())
        .args(["council", "findings", "--out"])
        .arg(&dir)
        .bounded_output();
    let e = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code(), Some(2), "{e}");
    assert!(e.contains("`members` is not a list"), "{e}");
    // No fallback to the files on disk: a present manifest means corrupt, not killed.
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("\"schema\""),
        "a corrupt manifest published a findings document anyway"
    );
}

#[test]
fn a_manifest_that_lists_zero_members_is_still_a_run() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let dir = r.dir.path().join("run");
    std::fs::write(dir.join("manifest.json"), r#"{"task":"t","members":[]}"#).unwrap();
    for m in ["00-a_one.md", "01-b_two.md"] {
        std::fs::remove_file(dir.join(m)).unwrap();
    }
    let out = bx_in(r.dir.path())
        .args(["council", "findings", "--out"])
        .arg(&dir)
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let f: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(f["members"], serde_json::json!([]), "{f}");
    assert_eq!(f["task"], "t", "{f}");
}

/// Mode `0o444`, not `0o000`: the listing must succeed while the stats fail, or the fallback
/// refuses for another reason and hides the bug.
#[test]
fn an_unreadable_manifest_is_refused_rather_than_read_as_absent() {
    let dir = TempDir::new().unwrap();
    let run = dir
        .path()
        .join("bx")
        .join("council")
        .join("20260101-120000-aaaaaa");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("brief.md"), "subject").unwrap();
    std::fs::write(run.join("00-a_one.md"), "## A. X\nx\n").unwrap();
    std::fs::write(run.join("manifest.json"), r#"{"task":"t","members":[]}"#).unwrap();
    std::fs::set_permissions(&run, std::os::unix::fs::PermissionsExt::from_mode(0o444)).unwrap();
    // Root ignores the mode.
    let denied = std::fs::symlink_metadata(run.join("manifest.json")).is_err();
    if !denied {
        std::fs::set_permissions(&run, std::os::unix::fs::PermissionsExt::from_mode(0o755))
            .unwrap();
        eprintln!("skipped: this process can stat inside a 0o444 directory (root?)");
        return;
    }
    let out = bx_in(dir.path())
        .args(["council", "findings", "20260101-120000-aaaaaa"])
        .env("BRAVE_SEARCH_API_KEY", "BSAtestkey000")
        .bounded_output();
    // Before asserting, so `TempDir` can clean up.
    std::fs::set_permissions(&run, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    let e = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code(), Some(2), "{e}");
    assert!(e.contains("manifest.json"), "{e}");
    assert!(!e.contains("not a readable run"), "{e}");
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("\"phase\""),
        "an unreadable manifest was published as a killed run"
    );
}

/// A draft is a row only where no finished reply exists, and it says so.
#[test]
fn one_seat_with_a_surviving_partial_is_one_member_not_two() {
    let (_dir, v, text) = planted(&[
        ("00-a_one.md", "## A. ANSWER\nfinished\n"),
        ("00-a_one.md.partial", "## A. ANSWER\nthe long scratchpad\n"),
        ("01-b_two.md.partial", "## A. ANSWER\nnever completed\n"),
    ]);
    let members = v["members"].as_array().unwrap();
    assert_eq!(members.len(), 2, "{v}");
    assert!(
        members[0]["file"]
            .as_str()
            .unwrap()
            .ends_with("00-a_one.md"),
        "{v}"
    );
    assert_eq!(members[0]["sections"][0]["text"], "finished", "{v}");
    // Absent, never `false`, like every other field in the row.
    assert!(members[0].get("partial").is_none(), "{v}");
    assert!(
        members[1]["file"]
            .as_str()
            .unwrap()
            .ends_with("01-b_two.md.partial"),
        "{v}"
    );
    assert_eq!(members[1]["partial"], true, "{v}");
    assert_eq!(
        text.matches("A. ANSWER").count(),
        2,
        "the text form disagrees with the JSON: {text}"
    );
}

/// Only `slugify`'s alphabet: a `\n` or `\t` forges a text row, and a scrubbed ESC names no file.
#[test]
fn a_member_file_name_this_run_could_not_have_written_is_not_a_member() {
    let (_dir, v, text) = planted(&[
        ("00-a_one.md", "## A. ANSWER\nreal\n"),
        ("01-f\noo\tdelivered.md", "## A. ANSWER\nforged\n"),
        ("02-e\u{1b}[2Jsc.md", "## A. ANSWER\nescaped\n"),
    ]);
    let members = v["members"].as_array().unwrap();
    assert_eq!(members.len(), 1, "{v}");
    for m in members {
        let file = m["file"].as_str().unwrap();
        assert!(std::path::Path::new(file).exists(), "{file}");
    }
    assert_eq!(
        text.lines().filter(|l| l.contains('\t')).count(),
        1,
        "a planted name forged a row: {text:?}"
    );
}

/// Created in reverse, so the directory's own order is the wrong answer.
#[test]
fn findings_orders_a_panel_of_more_than_ten_by_seat_and_not_by_disk() {
    let names: Vec<String> = (0..13).map(|i| format!("{i:02}-m.md")).collect();
    let planted_files: Vec<(&str, &str)> = names
        .iter()
        .rev()
        .map(|n| (n.as_str(), "## A. ANSWER\nx\n"))
        .collect();
    let (_dir, v, _) = planted(&planted_files);
    let got: Vec<String> = v["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            std::path::Path::new(m["file"].as_str().unwrap())
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(got, names, "{v}");
}

/// 33 files for 32 seats: the dedupe must run before the cap, or a real reply falls off.
#[test]
fn a_surviving_draft_never_costs_a_full_panel_its_last_reply() {
    let mut files: Vec<(String, &str)> = (0..32)
        .map(|i| (format!("{i:02}-m.md"), "## A. ANSWER\nreply\n"))
        .collect();
    files.push(("00-m.md.partial".to_string(), "## A. ANSWER\ndraft\n"));
    let planted_files: Vec<(&str, &str)> = files.iter().map(|(n, b)| (n.as_str(), *b)).collect();
    let (_dir, v, _) = planted(&planted_files);
    let members = v["members"].as_array().unwrap();
    // As a set: order is another test's question.
    let mut got: Vec<String> = members
        .iter()
        .map(|m| {
            std::path::Path::new(m["file"].as_str().unwrap())
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    got.sort();
    let mut want: Vec<String> = (0..32).map(|i| format!("{i:02}-m.md")).collect();
    want.sort();
    assert_eq!(got, want, "{v}");
    assert!(
        members.iter().all(|m| m.get("partial").is_none()),
        "a draft took a seat from a finished reply: {v}"
    );
}

/// In the text form, with the run's state.
#[test]
fn findings_says_why_a_manifest_less_run_has_no_names_and_points_at_the_files() {
    let (_dir, _, text) = planted(&[("00-a_one.md", "## A. ANSWER\nhello\n")]);
    assert!(text.contains("no manifest"), "why: {text:?}");
    assert!(text.contains("00-a_one.md"), "which file: {text:?}");
    assert!(text.contains("incomplete"), "run state: {text:?}");
}

/// Not read as "the newest", which is what omitting it means.
#[test]
fn an_empty_run_name_is_refused_by_name_on_every_sub_command_that_takes_one() {
    let dir = TempDir::new().unwrap();
    let call = |args: &[&str]| {
        let out = bx_in(dir.path())
            .args(["council"])
            .args(args)
            .env("BRAVE_SEARCH_API_KEY", "BSAtestkey000")
            .bounded_output();
        (
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    for args in [
        ["findings", ""].as_slice(),
        ["useful-yes", ""].as_slice(),
        ["status", ""].as_slice(),
    ] {
        let (code, e) = call(args);
        assert_eq!(code, Some(2), "{args:?}: {e}");
        assert!(e.contains("empty string"), "{args:?}: {e}");
    }
    // clap refuses an empty `--out` itself, naming the flag.
    let (code, e) = call(&["findings", "--out", ""]);
    assert_eq!(code, Some(2), "{e}");
    assert!(e.contains("--out"), "{e}");
}

#[test]
fn a_malformed_manifest_is_an_error_and_not_a_missing_one() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(port, &["-m", "a/one,b/two", "--progress", "false"]);
    let dir = r.dir.path().join("run");
    std::fs::write(dir.join("manifest.json"), "{not json").unwrap();
    let out = bx_in(r.dir.path())
        .args(["council", "findings"])
        .arg(&dir)
        .bounded_output();
    assert_eq!(
        out.status.code(),
        Some(2),
        "a corrupt manifest was read as absent"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("manifest.json"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `--max-usd=-0.0` publishes as `0.0`. Asserted on the text: `json!(-0.0) == json!(0.0)`.
#[test]
fn one_resolution_of_the_cap_reaches_every_artifact_that_names_it() {
    let (port, _) = serve(mock(|_| reply(&prose())));
    let r = council(
        port,
        &["-m", "a/one,b/two", "--progress", "false", "--max-usd=-0.0"],
    );
    let m = r.manifest();
    assert_eq!(
        serde_json::to_string(&m["cost"]["max_usd"]).unwrap(),
        "0.0",
        "the manifest published a cap nobody typed: {m}"
    );
    let status = r.read("status.json");
    assert!(
        status.contains("\"max_usd\":0.0"),
        "status.json disagrees with the manifest: {status}"
    );
    // Enforced too: a $0 cap stops every member before its first round.
    for x in m["members"].as_array().unwrap() {
        assert_eq!(x["reason"], "max_usd", "{m}");
        assert_eq!(x["rounds"], 0, "a round was bought under a $0 cap: {m}");
    }
}

/// Only `publish` sets `answered`; the step and spend caps return before it, labelled by the
/// bound that stopped them rather than the channel.
#[test]
fn a_scratchpad_is_never_delivered_by_a_path_that_does_not_publish() {
    // Always another call, so the member can only leave through a pre-round bound.
    let (port, _) = serve(mock(|_| {
        (
            200,
            serde_json::json!({
                "model": "test/model",
                "choices": [{"finish_reason": "tool_calls", "message": {
                    "reasoning": "thinking. ".repeat(2_000),
                    "tool_calls": [{"id": "c1", "function": {
                        "name": "search", "arguments": "{\"query\":\"q\"}"}}]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}
            })
            .to_string(),
        )
    }));
    // A round reports $0.01, so round two's pre-check trips a $0.01 cap.
    for (args, want) in [
        (&[][..], "max_steps"),
        (&["--max-usd", "0.01"][..], "max_usd"),
    ] {
        let r = council(
            port,
            &[&["-m", "a/one,b/two", "--progress", "false"][..], args].concat(),
        );
        let m = r.manifest();
        let members = m["members"].as_array().unwrap();
        // On every exit, including a transport fault a loaded runner substitutes.
        for x in members {
            assert_ne!(x["status"], "delivered", "{m}");
            assert_ne!(
                x["terminated_by"], "reasoning_only",
                "only step 7 may name the channel: {m}"
            );
        }
        assert!(
            members.iter().any(|x| x["terminated_by"] == want),
            "no member stopped by {want}: {m}"
        );
        // Billed, so still on disk.
        assert!(r.read("00-a_one.md").contains("thinking."), "{m}");
    }
}

/// Set on the command, the way an exported shell variable arrives.
#[test]
fn the_sandbox_drops_every_key_a_leaking_parent_hands_down() {
    let dir = TempDir::new().unwrap();
    let mut parent = Command::new(common::BX);
    parent
        .env("ANTHROPIC_API_KEY", "sk-ant-leakedtoken")
        .env("OPENROUTER_API_KEY", "sk-or-v1-leakedtoken")
        .env("OPENAI_API_KEY", "sk-proj-leakedtoken")
        .env("BRAVE_SEARCH_API_KEY", "BSAleakedtoken000")
        .env("BRAVE_API_KEY", "BSAleakedtoken000");
    let out = common::sandbox(&mut parent, dir.path())
        .args(["council", "doctor", "--format", "json"])
        .bounded_output();
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let keys = doc["keys"].as_array().unwrap();
    let named: Vec<&str> = keys.iter().filter_map(|k| k["provider"].as_str()).collect();
    assert_eq!(
        named,
        ["openrouter", "anthropic", "openai", "brave"],
        "{doc}"
    );
    for k in keys {
        assert_eq!(k["set"], false, "a parent's key reached the child: {k}");
    }
}
