//! The exit-code contract, pinned end to end.
//!
//! An agent branches on these numbers, so the line between them is part of the interface:
//! 2 means bx could not act on what it was given and never contacted the API, so retrying
//! the same command cannot help. Every case below must therefore fail without a network.

use std::process::{Command, Output};

/// Runs `bx` fully isolated: a throwaway config, no inherited key, and a base URL that
/// refuses connections, so a case that stopped failing early would exit 5 rather than
/// reach production. The key goes in the environment, not on the command line, so a test
/// may pass its own `--api-key` without clap rejecting a duplicate flag.
fn bx(args: &[&str]) -> Output {
    bx_raw(args, &["--base-url", "http://127.0.0.1:1"])
}

fn bx_raw(args: &[&str], extra: &[&str]) -> Output {
    let cfg = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cfg.path(), "{}").unwrap();
    Command::new(env!("CARGO_BIN_EXE_bx"))
        .args(args)
        .args(extra)
        .arg("--config")
        .arg(cfg.path())
        .env("BRAVE_SEARCH_API_KEY", "TEST")
        .env_remove("BRAVE_API_KEY")
        .env_remove("BRAVE_SEARCH_BASE_URL")
        .env("NO_PROXY", "*")
        .output()
        .unwrap()
}

#[test]
fn a_rejected_base_url_is_a_usage_error() {
    // Each of these fails validation before any socket is opened.
    for url in [
        "https://evil.example.com",     // not in the allowlist
        "http://192.168.1.1:8080",      // private, not loopback
        "http://169.254.169.254:80",    // cloud metadata
        "https://127.0.0.1:8080",       // https to loopback
        "http://127.0.0.1:0",           // port 0
        "http://127.0.0.1:65536",       // port overflow
        "http://user@127.0.0.1:8080",   // userinfo smuggling
        "http://0177.0.0.1:8080",       // octal SSRF bypass
        "http://127.0.0.1.nip.io:8080", // DNS service bypass
        "http://::1:8080",              // unbracketed IPv6
    ] {
        let out = bx_raw(&["web", "q"], &["--base-url", url]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{url}: {stderr}");
    }
}

#[test]
fn a_bad_flag_value_is_a_usage_error() {
    for args in [
        vec!["web", "q", "--extra", "noequals"],
        vec!["web", "q", "--endpoint", "evil.example.com/collect"],
        vec!["web", "q", "--endpoint", "//evil.example.com"],
        vec!["web", "q", "--endpoint", "/../admin"],
        vec!["web", "q", "--include-site", "bad domain!"],
    ] {
        let out = bx(&args);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {stderr}");
    }
}

#[test]
fn a_missing_required_id_is_a_usage_error() {
    // `bx pois` and `bx descriptions` take their IDs positionally, and clap allows zero of
    // them. Sending a request with no ids would just earn a 422, so bx stops first — and
    // that stop is the user's mistake, not the API's answer.
    for cmd in ["pois", "descriptions"] {
        let out = bx(&[cmd]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{cmd}: {stderr}");
        assert!(stderr.contains("at least one POI ID"), "{cmd}: {stderr}");
    }
}

#[test]
fn an_unusable_config_is_a_usage_error() {
    // A config value is an input like any other: `timeout: 0` is unusable, and so is a
    // file that does not parse. Both used to exit 1, which reads as "the API said no".
    // No `--base-url` here: a CLI value would override the very config field under test.
    // The rejected host is RFC 2606 reserved, so a regression cannot reach a real service.
    for contents in [
        "{not json",
        r#"{"timeout":0}"#,
        r#"{"base_url":"http://evil.example.com"}"#,
    ] {
        let cfg = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(cfg.path(), contents).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_bx"))
            .args(["web", "q"])
            .arg("--config")
            .arg(cfg.path())
            .env("BRAVE_SEARCH_API_KEY", "TEST")
            .env_remove("BRAVE_API_KEY")
            .env_remove("BRAVE_SEARCH_BASE_URL")
            .env("NO_PROXY", "*")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{contents}: {stderr}");
    }
}

#[test]
fn unparseable_stdin_is_a_usage_error() {
    use std::io::Write;
    use std::process::Stdio;

    let cfg = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cfg.path(), "{}").unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_bx"))
        .args(["answers", "-", "--base-url", "http://127.0.0.1:1"])
        .arg("--config")
        .arg(cfg.path())
        .env("BRAVE_SEARCH_API_KEY", "TEST")
        .env_remove("BRAVE_API_KEY")
        .env_remove("BRAVE_SEARCH_BASE_URL")
        .env("NO_PROXY", "*")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"{not json").unwrap();
    let out = child.wait_with_output().unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("invalid JSON on stdin"), "{stderr}");
}

#[test]
fn an_api_key_that_cannot_be_a_header_is_a_usage_error() {
    // Every X-Loc-* value was validated; the token itself never was. Sources are trimmed,
    // so this is an interior control character, which reached ureq and failed there —
    // a bad argument surfacing as exit 5 "server/network error".
    let out = bx(&["web", "q", "--api-key", "bad\nkey"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    // Not clap rejecting the flag, and not a connection failure: our own check.
    assert!(
        stderr.contains("invalid header value for X-Subscription-Token"),
        "{stderr}"
    );
}

#[test]
fn config_subcommands_reject_an_unusable_file_the_same_way() {
    // `bx web` and `bx config show` call the same loader on the same file; they used to
    // disagree about whether that was the API's answer or the user's mistake.
    let cfg = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cfg.path(), "{not json").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_bx"))
        .args(["config", "show"])
        .arg("--config")
        .arg(cfg.path())
        .env_remove("BRAVE_SEARCH_API_KEY")
        .env_remove("BRAVE_API_KEY")
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn incompatible_answers_options_are_a_usage_error() {
    // The server 422s these; rejecting them here turns a wasted round trip and a misleading
    // exit 1 into exit 2. All sixteen combinations are a unit test — this pins the exit code.
    let out = bx(&["answers", "q", "--enable-research", "--enable-citations"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("cannot be used with"), "{stderr}");

    // Citations and entities are the one pair the API accepts; it must still be sent.
    let out = bx(&["answers", "q", "--enable-citations", "--enable-entities"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(5), "{stderr}");
}

#[test]
fn a_reachable_failure_is_not_swept_into_the_usage_code() {
    // The control for every assertion above: exit 2 must mean "nothing was sent", so a
    // command that gets as far as opening a socket has to report something else.
    let out = bx(&["web", "q"]);
    assert_eq!(
        out.status.code(),
        Some(5),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn a_timeout_too_large_for_the_clock_is_rejected() {
    // ureq adds this to an `Instant`; a large enough value panics with exit 101, or
    // SIGABRT in release, neither of which is a documented outcome.
    for bad in ["0", "86401", "18446744073709551615"] {
        let out = bx(&["web", "q", "--timeout", bad]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "--timeout {bad}: {stderr}");
        assert!(stderr.contains("between 1 and 86400"), "{stderr}");
    }
    // The bound is inclusive and must still reach the network.
    let out = bx(&["web", "q", "--timeout", "86400"]);
    assert_eq!(
        out.status.code(),
        Some(5),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
