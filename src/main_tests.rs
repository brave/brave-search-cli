//! Tests for `main.rs`; a child module, so private items stay reachable.

use super::*;
use clap::{CommandFactory, error::ErrorKind};

// ── answers flag conflicts ───────────────────────────────────────

/// All sixteen combinations of the four flags, so the rule matches the API's exactly (e.g.
/// `--enable-citations --enable-entities` is allowed).
#[test]
fn answers_rejects_exactly_the_combinations_the_api_rejects() {
    for bits in 0..16u8 {
        let [no_stream, citations, entities, research] =
            [0, 1, 2, 3].map(|i: u8| bits >> i & 1 == 1);
        let args = [
            Some("bx"),
            Some("answers"),
            Some("q"),
            no_stream.then_some("--no-stream"),
            citations.then_some("--enable-citations"),
            entities.then_some("--enable-entities"),
            research.then_some("--enable-research"),
        ];
        // All three need the stream for their tags; research mode supports neither citations
        // nor entities.
        let rejected = (no_stream && (citations || entities || research))
            || (research && (citations || entities));
        let parsed = Cli::try_parse_from(args.into_iter().flatten());
        assert_eq!(
            parsed.is_err(),
            rejected,
            "{:?}",
            args.into_iter().flatten().collect::<Vec<_>>()
        );
        if let Err(e) = parsed {
            assert_eq!(e.kind(), ErrorKind::ArgumentConflict, "{bits:b}");
        }
    }
}

// ── answers_timeouts ─────────────────────────────────────────────

/// Only a value bx reads as `true` earns the long read budget: a wrong short guess fails with
/// an actionable hint, a wrong long one hangs an agent for minutes.
#[test]
fn answers_timeouts_is_long_only_for_research() {
    for on in [serde_json::json!(true), serde_json::json!(1)] {
        let body = serde_json::json!({"stream": true, "enable_research": on});
        assert_eq!(
            answers_timeouts(None, &body).1,
            DEFAULT_RESEARCH_TIMEOUT,
            "{body}"
        );
    }
    for off in [
        serde_json::json!(false),
        serde_json::json!(0),
        serde_json::json!("true"), // a string is not a request for research
        serde_json::json!("yes"),
        serde_json::json!(2),
        serde_json::json!(null),
    ] {
        let body = serde_json::json!({"stream": true, "enable_research": off});
        assert_eq!(answers_timeouts(None, &body).1, DEFAULT_TIMEOUT, "{body}");
    }
    assert_eq!(
        answers_timeouts(None, &serde_json::json!({})).1,
        DEFAULT_TIMEOUT
    );
}

/// The research budget must stay off the dial phases, or a blackholed host takes minutes to
/// report.
#[test]
fn answers_timeouts_keeps_the_research_budget_off_the_dial_phases() {
    let research = serde_json::json!({"enable_research": true});
    assert_eq!(answers_timeouts(None, &research), (DEFAULT_TIMEOUT, 300));
    // An explicit value governs both, in either direction.
    assert_eq!(answers_timeouts(Some(600), &research), (600, 600));
    assert_eq!(answers_timeouts(Some(2), &research), (2, 2));
}

/// The transport is irrelevant: `--extra enable_research=true` or a stdin body can still ask
/// for blocking research.
#[test]
fn answers_timeouts_ignores_the_transport() {
    let blocking = serde_json::json!({"stream": false, "enable_research": true});
    assert_eq!(answers_timeouts(None, &blocking), (DEFAULT_TIMEOUT, 300));
}

/// A stdin body need not be an object; indexing one must not panic.
#[test]
fn answers_timeouts_survives_a_non_object_body() {
    for body in [
        serde_json::json!([1, 2, 3]),
        serde_json::json!("string"),
        serde_json::json!(42),
        serde_json::json!(null),
    ] {
        assert_eq!(
            answers_timeouts(None, &body),
            (DEFAULT_TIMEOUT, DEFAULT_TIMEOUT),
            "{body}"
        );
        assert_eq!(answers_timeouts(Some(7), &body), (7, 7), "{body}");
    }
}

/// `--timeout 30 --enable-research` must stay 30, which is why `cmd_answers` takes
/// `Option<u64>`: once resolved, an explicit 30 looks like the default.
#[test]
fn answers_timeouts_never_override_a_configured_value() {
    let research = serde_json::json!({"enable_research": true});
    assert_eq!(answers_timeouts(Some(5), &research), (5, 5));
    assert_eq!(
        answers_timeouts(Some(DEFAULT_TIMEOUT), &research),
        (DEFAULT_TIMEOUT, DEFAULT_TIMEOUT)
    );
    assert_eq!(
        answers_timeouts(Some(900), &serde_json::json!({})),
        (900, 900)
    );
}

/// The defaults bypass main's `1..=MAX_TIMEOUT` guard, which checks only configured values;
/// out of range, ureq's `Instant` maths panics.
#[test]
fn answers_timeouts_always_return_usable_values() {
    for body in [
        serde_json::json!({}),
        serde_json::json!({"enable_research": true}),
        serde_json::json!({"enable_research": 1}),
    ] {
        let (dial, read) = answers_timeouts(None, &body);
        assert!((1..=MAX_TIMEOUT).contains(&dial), "{body} → {dial}");
        assert!((1..=MAX_TIMEOUT).contains(&read), "{body} → {read}");
    }
}

/// `-h` is what most callers read, so the research default belongs on its `--timeout` line.
#[test]
fn answers_short_help_names_the_research_timeout_default() {
    let cmd = Cli::command();
    let timeout = cmd.get_arguments().find(|a| a.get_id() == "timeout");
    let short = timeout.and_then(|a| a.get_help()).unwrap().to_string();
    assert!(
        short.contains(&DEFAULT_RESEARCH_TIMEOUT.to_string()),
        "{short}"
    );
}

/// Queries stay with Brave except under council, which sends them to model providers; the
/// top-level help must say both.
#[test]
fn the_top_level_help_discloses_what_council_sends_to_model_providers() {
    let help = Cli::command().render_help().to_string();
    let about = help.split("Usage:").next().unwrap();
    for word in ["never", "council", "providers"] {
        assert!(about.contains(word), "{word}: {about}");
    }
}

/// An exit 3 or 8 with no run directory is the pre-flight, not a missed quorum.
#[test]
fn council_help_says_exit_3_and_8_can_come_from_the_free_pre_flight() {
    let mut cmd = Cli::command();
    let help = cmd.find_subcommand_mut("council").unwrap().render_help();
    let help = help.to_string();
    let exit = help.split("EXIT").nth(1).unwrap();
    assert!(exit.contains("pre-flight"), "{exit}");
}

// ── resolve_stream ───────────────────────────────────────────────

/// The body's `stream` (including the `0`/`1` that `--extra` produces) decides, or a JSON reply
/// would be fed to the SSE parser: exit 0, no output.
#[test]
fn resolve_stream_follows_the_body_before_the_flag() {
    for (v, want) in [
        (serde_json::json!(false), false),
        (serde_json::json!(0), false),
        (serde_json::json!(true), true),
        (serde_json::json!(1), true),
    ] {
        let mut body = serde_json::json!({"stream": v});
        assert_eq!(resolve_stream(&mut body, !want), want, "{body}");
    }
}

/// An unreadable value leaves the flag in charge and is rewritten to match, so the request
/// never asks for one transport while bx parses another.
#[test]
fn resolve_stream_writes_its_decision_back_into_the_body() {
    for junk in [
        serde_json::json!("false"),
        serde_json::json!(2),
        serde_json::json!(null),
        serde_json::json!({}),
    ] {
        for default in [true, false] {
            let mut body = serde_json::json!({"stream": junk});
            assert_eq!(resolve_stream(&mut body, default), default, "{junk}");
            assert_eq!(body["stream"], serde_json::json!(default), "{junk}");
        }
    }
}

/// A missing `stream` is added, so bx and the server cannot default differently.
#[test]
fn resolve_stream_adds_a_missing_key() {
    let mut body = serde_json::json!({"messages": []});
    assert!(resolve_stream(&mut body, true));
    assert_eq!(body["stream"], serde_json::json!(true));
}

/// A container `stream` is unreadable too; rewriting it leaves other keys alone.
#[test]
fn resolve_stream_replaces_a_container_and_leaves_its_neighbours_alone() {
    for junk in [serde_json::json!([1, 2]), serde_json::json!({"a": 1})] {
        let mut body = serde_json::json!({"stream": junk, "messages": [], "model": "m"});
        assert!(!resolve_stream(&mut body, false), "{junk}");
        assert_eq!(body["stream"], serde_json::json!(false), "{junk}");
        assert_eq!(body["messages"], serde_json::json!([]), "{junk}");
        assert_eq!(body["model"], serde_json::json!("m"), "{junk}");
    }
}

/// A value that already agrees with the decision is not normalised.
#[test]
fn resolve_stream_leaves_a_correct_value_alone() {
    let mut body = serde_json::json!({"stream": false});
    assert!(!resolve_stream(&mut body, true));
    assert_eq!(body, serde_json::json!({"stream": false}));
}

/// A non-object body must not panic (`IndexMut` would); it is left for the API to reject.
#[test]
fn resolve_stream_survives_a_non_object_body() {
    let mut body = serde_json::json!([1, 2, 3]);
    assert!(resolve_stream(&mut body, true));
    assert_eq!(body, serde_json::json!([1, 2, 3]));
    // …and the flag still decides, in both directions.
    let mut body = serde_json::json!("string");
    assert!(!resolve_stream(&mut body, false));
    assert_eq!(body, serde_json::json!("string"));
}

#[test]
fn subcommands_list_matches_clap_enum() {
    let cmd = Cli::command();
    let mut clap_names: std::collections::HashSet<&str> =
        cmd.get_subcommands().map(|s| s.get_name()).collect();

    // clap adds "help" implicitly; SUBCOMMANDS lists it so `bx help` is not a search query.
    clap_names.insert("help");

    let hardcoded: std::collections::HashSet<&str> = SUBCOMMANDS.iter().copied().collect();

    let missing: Vec<&&str> = clap_names.difference(&hardcoded).collect();
    let extra: Vec<&&str> = hardcoded.difference(&clap_names).collect();

    assert!(
        missing.is_empty(),
        "Subcommands in clap but missing from SUBCOMMANDS: {missing:?}"
    );
    assert!(
        extra.is_empty(),
        "Entries in SUBCOMMANDS but not in clap: {extra:?}"
    );
}

#[test]
fn value_flags_names_every_global_option_that_takes_a_value() {
    let cmd = Cli::command();
    let mut clap_flags: Vec<String> = cmd
        .get_arguments()
        .filter(|a| a.is_global_set() && a.get_action().takes_values())
        .map(|a| format!("--{}", a.get_long().unwrap()))
        .collect();
    let mut listed: Vec<String> = VALUE_FLAGS.iter().map(|f| f.to_string()).collect();
    clap_flags.sort();
    listed.sort();
    assert_eq!(listed, clap_flags);
}

#[test]
fn check_header_value_accepts_normal_ascii() {
    assert!(check_header_value("X-Test", "New York").is_ok());
    assert!(check_header_value("X-Test", "America/Los_Angeles").is_ok());
    assert!(check_header_value("X-Test", "90210").is_ok());
    assert!(check_header_value("X-Test", "47.6062").is_ok());
}

#[test]
fn check_header_value_accepts_utf8() {
    // Non-ASCII UTF-8 (accented city names) passes; http accepts it as obs-text (RFC 7230).
    assert!(check_header_value("X-Loc-City", "Zürich").is_ok());
    assert!(check_header_value("X-Loc-City", "São Paulo").is_ok());
}

#[test]
fn check_header_value_rejects_newline() {
    assert!(check_header_value("X-Test", "evil\ninjection").is_err());
}

#[test]
fn check_header_value_rejects_carriage_return() {
    assert!(check_header_value("X-Test", "evil\rinjection").is_err());
}

#[test]
fn check_header_value_rejects_null() {
    assert!(check_header_value("X-Test", "evil\0injection").is_err());
}

// ── check_base_url tests ─────────────────────────────────────────

#[test]
fn check_base_url_allows_production_urls() {
    assert_eq!(
        check_base_url("https://api.search.brave.com").unwrap(),
        "https://api.search.brave.com"
    );
    assert_eq!(
        check_base_url("https://api.search.brave.software").unwrap(),
        "https://api.search.brave.software"
    );
    assert_eq!(
        check_base_url("https://api.search.brave.com/").unwrap(),
        "https://api.search.brave.com"
    );
}

#[test]
fn check_base_url_accepts_ipv4_loopback() {
    assert_eq!(
        check_base_url("http://127.0.0.1:8080").unwrap(),
        "http://127.0.0.1:8080"
    );
    assert_eq!(
        check_base_url("http://127.0.0.1").unwrap(),
        "http://127.0.0.1"
    );
    // Full 127.0.0.0/8 range
    assert_eq!(
        check_base_url("http://127.255.255.255:3000").unwrap(),
        "http://127.255.255.255:3000"
    );
    assert_eq!(
        check_base_url("http://127.0.0.1:8080/brave").unwrap(),
        "http://127.0.0.1:8080/brave"
    );
    // Trailing slash stripped, preventing `//` when paths are appended
    assert_eq!(
        check_base_url("http://127.0.0.1:8080/").unwrap(),
        "http://127.0.0.1:8080"
    );
    assert_eq!(
        check_base_url("http://127.0.0.1:8080///").unwrap(),
        "http://127.0.0.1:8080"
    );
}

#[test]
fn check_base_url_accepts_ipv6_loopback() {
    assert_eq!(
        check_base_url("http://[::1]:8080").unwrap(),
        "http://[::1]:8080"
    );
    assert_eq!(check_base_url("http://[::1]").unwrap(), "http://[::1]");
    assert_eq!(
        check_base_url("http://[::1]:8080/v1").unwrap(),
        "http://[::1]:8080/v1"
    );
}

#[test]
fn check_base_url_accepts_localhost_hostname() {
    // Skip if localhost doesn't resolve (unusual CI environments)
    if ("localhost", 80u16).to_socket_addrs().is_err() {
        return;
    }
    let result = check_base_url("http://localhost:8080").unwrap();
    // TOCTOU defense: must be rewritten to a literal loopback IP
    assert!(
        !result.contains("localhost"),
        "must rewrite localhost to literal IP, got: {result}"
    );
    assert!(
        result.starts_with("http://127.") || result.starts_with("http://[::1]"),
        "must resolve to loopback, got: {result}"
    );
    // Case-insensitive
    assert!(check_base_url("http://LOCALHOST:8080").is_ok());
    assert!(check_base_url("http://localhost").is_ok());
    // Path preserved after localhost → IP rewrite
    let with_path = check_base_url("http://localhost:8080/brave").unwrap();
    assert!(
        with_path.ends_with(":8080/brave"),
        "path must be preserved, got: {with_path}"
    );
}

#[test]
fn check_base_url_port_boundaries() {
    assert!(check_base_url("http://127.0.0.1:1").is_ok());
    assert!(check_base_url("http://127.0.0.1:65535").is_ok());
    assert!(check_base_url("http://127.0.0.1:0").is_err()); // port 0
    assert!(check_base_url("http://127.0.0.1:65536").is_err()); // u16 overflow
    assert!(check_base_url("http://127.0.0.1:99999").is_err()); // u16 overflow
    assert!(check_base_url("http://127.0.0.1:").is_err()); // empty port
    assert!(check_base_url("http://[::1]:").is_err()); // empty port (IPv6)
}

#[test]
fn check_base_url_rejects_non_loopback_ips() {
    assert!(check_base_url("http://192.168.1.1:8080").is_err()); // RFC 1918 private
    assert!(check_base_url("http://10.0.0.1:8080").is_err()); // RFC 1918 private
    assert!(check_base_url("http://0.0.0.0:8080").is_err()); // unspecified ≠ loopback
    assert!(check_base_url("http://[::2]:8080").is_err()); // non-loopback IPv6
    assert!(check_base_url("http://169.254.169.254:80").is_err()); // link-local / cloud metadata
    assert!(check_base_url("http://[fe80::1]:8080").is_err()); // link-local IPv6
}

#[test]
fn check_base_url_rejects_ssrf_bypass_attempts() {
    // Octal notation — Rust's strict Ipv4Addr parser per RFC 6943 rejects this
    assert!(check_base_url("http://0177.0.0.1:8080").is_err());
    // Decimal IP (2130706433 = 127.0.0.1) — not a valid Ipv4Addr, not "localhost"
    assert!(check_base_url("http://2130706433:8080").is_err());
    // Shorthand (127.1) — Rust rejects non-four-octet forms
    assert!(check_base_url("http://127.1:8080").is_err());
    // Hex IP — Rust rejects, not "localhost"
    assert!(check_base_url("http://0x7f000001:8080").is_err());
    // IPv4-mapped IPv6 — Ipv6Addr::is_loopback() returns false (rust-lang/rust#69772)
    assert!(check_base_url("http://[::ffff:127.0.0.1]:8080").is_err());
    assert!(check_base_url("http://[::ffff:7f00:1]:8080").is_err());
    // Userinfo smuggling — @ makes the part after it the real host
    assert!(check_base_url("http://user:pass@127.0.0.1:8080").is_err());
    assert!(check_base_url("http://127.0.0.1@evil.com:8080").is_err());
    // DNS service bypass — not "localhost", not a valid IP literal
    assert!(check_base_url("http://127.0.0.1.nip.io:8080").is_err());
    // Percent-encoded IP — code never decodes, raw string fails both parsers
    assert!(check_base_url("http://%31%32%37.0.0.1:8080").is_err());
    // Unicode confusable 'l' (U+217C) — eq_ignore_ascii_case is ASCII-only
    assert!(check_base_url("http://\u{217C}ocalhost:8080").is_err());
}

#[test]
fn check_base_url_rejects_bad_scheme_or_structure() {
    // https for localhost — must use http://
    assert!(check_base_url("https://localhost:8080").is_err());
    assert!(check_base_url("https://127.0.0.1:8080").is_err());
    assert!(check_base_url("ftp://127.0.0.1:8080").is_err());
    assert!(check_base_url("127.0.0.1:8080").is_err()); // no scheme
    assert!(check_base_url("http://").is_err()); // empty host
    assert!(check_base_url("not-a-url").is_err());
    assert!(check_base_url("").is_err());
    assert!(check_base_url("https://evil.com").is_err()); // not in allowlist
    assert!(check_base_url("http:///127.0.0.1:8080").is_err()); // triple slash → empty host
    assert!(check_base_url("http://127.0.0.1:8080?foo=bar").is_err()); // query contaminates port
    assert!(check_base_url("http://127.0.0.1:8080#frag").is_err()); // fragment contaminates port
}

#[test]
fn check_base_url_rejects_malformed_ipv6() {
    assert!(check_base_url("http://[::1").is_err()); // missing ]
    assert!(check_base_url("http://[::1]garbage:8080").is_err()); // junk after bracket
    assert!(check_base_url("http://[::1%25eth0]:8080").is_err()); // zone ID
    assert!(check_base_url("http://[]:8080").is_err()); // empty brackets
}

#[test]
fn check_base_url_rejects_an_unbracketed_ipv6_address() {
    // Splitting on the last ':' would misread these as host and port and hand ureq a URL it
    // refuses with exit 5, not a usage error. The last is `::1` in full.
    for url in [
        "http://::1:8080",
        "http://::1",
        "http://fe80::1:80",
        "http://0:0:0:0:0:0:0:1:8080",
    ] {
        let err = check_base_url(url).unwrap_err();
        assert!(err.contains("bracket an IPv6 address"), "{url}: {err}");
    }
    assert!(check_base_url("http://localhost:8080:9090").is_err());
    // The bracketed forms are unaffected.
    assert!(check_base_url("http://[::1]:8080").is_ok());
}

#[test]
fn check_endpoint_accepts_valid_paths() {
    assert!(check_endpoint("/res/v1/web/search").is_ok());
    assert!(check_endpoint("/res/v1/llm/context").is_ok());
    assert!(check_endpoint("/beta/v2/new-endpoint").is_ok());
    assert!(check_endpoint("/").is_ok());
    assert!(check_endpoint("/res/v1/").is_ok()); // trailing slash
    assert!(check_endpoint("/res/v1/foo..bar").is_ok()); // ".." as substring
}

#[test]
fn check_endpoint_rejects_missing_slash() {
    assert!(check_endpoint("res/v1/web/search").is_err());
}

#[test]
fn check_endpoint_rejects_disallowed_chars() {
    assert!(check_endpoint("/foo?bar=1").is_err()); // query string
    assert!(check_endpoint("/foo#frag").is_err()); // fragment
    assert!(check_endpoint("/foo@bar").is_err()); // authority
    assert!(check_endpoint("/foo\\bar").is_err()); // backslash
    assert!(check_endpoint("/foo%2e").is_err()); // percent encoding
    assert!(check_endpoint("/foo;bar").is_err()); // semicolon
    assert!(check_endpoint("/foo bar").is_err()); // space
    assert!(check_endpoint("/foo\nbar").is_err()); // newline
    assert!(check_endpoint("/foo\rbar").is_err()); // carriage return
    assert!(check_endpoint("/foo\0bar").is_err()); // null
}

#[test]
fn check_endpoint_rejects_path_traversal() {
    assert!(check_endpoint("/../admin").is_err());
    assert!(check_endpoint("/res/../admin").is_err());
    assert!(check_endpoint("/res/v1/..").is_err());
}

#[test]
fn check_endpoint_rejects_consecutive_slashes() {
    assert!(check_endpoint("//evil.com").is_err());
    assert!(check_endpoint("/res//v1").is_err());
}

#[test]
fn check_endpoint_single_dot_segment_allowed() {
    assert!(check_endpoint("/res/./v1").is_ok()); // `.` is harmless (current dir)
}

#[test]
fn check_endpoint_rejects_empty_and_non_ascii() {
    assert!(check_endpoint("").is_err()); // no leading `/`
    assert!(check_endpoint("/res/caf\u{00e9}").is_err()); // non-ASCII byte rejected
}

#[test]
fn site_goggles_allowlist() {
    let result = build_site_goggles(&["docs.rs".into(), "github.com".into()], &[]);
    assert_eq!(
        result.unwrap(),
        "$discard\n$boost,site=docs.rs\n$boost,site=github.com"
    );
}

#[test]
fn site_goggles_blocklist() {
    let result = build_site_goggles(&[], &["w3schools.com".into(), "medium.com".into()]);
    assert_eq!(
        result.unwrap(),
        "$discard,site=w3schools.com\n$discard,site=medium.com"
    );
}

#[test]
fn site_goggles_empty() {
    assert!(build_site_goggles(&[], &[]).is_none());
}

#[test]
fn site_goggles_single() {
    let result = build_site_goggles(&["docs.rs".into()], &[]);
    assert_eq!(result.unwrap(), "$discard\n$boost,site=docs.rs");
}

#[test]
fn warn_goggles_shell_expansion_triggers() {
    // Smoke test: should not panic (warns on stderr)
    warn_shell_expanded_goggles("=example.org");
    warn_shell_expanded_goggles(",site=example.org");
}

#[test]
fn warn_goggles_shell_expansion_skips() {
    // Smoke test: these must not warn, but only the absence of a panic is checked
    warn_shell_expanded_goggles("$site=example.org");
    warn_shell_expanded_goggles("$boost=3,site=docs.rs");
    warn_shell_expanded_goggles("@rules.goggle");
    warn_shell_expanded_goggles("https://example.com/goggle");
    warn_shell_expanded_goggles("");
}

#[test]
fn validate_domain_accepts_valid() {
    assert!(validate_domain("docs.rs").is_ok());
    assert!(validate_domain("my-site.example.com").is_ok());
    assert!(validate_domain("under_score.io").is_ok());
}

#[test]
fn validate_domain_rejects_invalid() {
    assert!(validate_domain("").is_err());
    assert!(validate_domain("bad domain").is_err());
    assert!(validate_domain("evil!.com").is_err());
    assert!(validate_domain("path/slash").is_err());
    // Goggles syntax characters must be rejected to prevent injection
    assert!(validate_domain("$inject").is_err());
    assert!(validate_domain("a,b=c").is_err());
    assert!(validate_domain("a\nb").is_err());
}

// ── inject_default_subcommand_impl tests ────────────────────────

fn args(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

#[test]
fn inject_normal_query() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx rust-error")),
        args("bx context rust-error")
    );
}

#[test]
fn inject_known_subcommand_no_change() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx web query")),
        args("bx web query")
    );
}

/// A subcommand name is shadowed as a bare query; the documented `--` escape must recover it,
/// exactly as for `web`.
#[test]
fn inject_council_shadows_the_bare_word_and_double_dash_recovers_it() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx council @design.md")),
        args("bx council @design.md")
    );
    assert_eq!(
        inject_default_subcommand_impl(args("bx -- council")),
        args("bx context -- council")
    );
    // Only the first positional decides, so a query starting with the word is shadowed too.
    assert_eq!(
        inject_default_subcommand_impl(args("bx council of europe")),
        args("bx council of europe")
    );
    assert_eq!(
        inject_default_subcommand_impl(args("bx web of things")),
        args("bx web of things")
    );
    // A query that merely contains the word is untouched.
    assert_eq!(
        inject_default_subcommand_impl(args("bx european council summit")),
        args("bx context european council summit")
    );
}

#[test]
fn inject_double_dash_inserts_context() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx -- web")),
        args("bx context -- web")
    );
}

#[test]
fn inject_double_dash_with_normal_query() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx -- some-query")),
        args("bx context -- some-query")
    );
}

#[test]
fn inject_skips_global_value_flags() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --api-key KEY query")),
        args("bx --api-key KEY context query")
    );
}

#[test]
fn inject_help_no_change() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --help")),
        args("bx --help")
    );
}

#[test]
fn inject_double_dash_alone() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --")),
        args("bx context --")
    );
}

#[test]
fn inject_no_args() {
    assert_eq!(inject_default_subcommand_impl(args("bx")), args("bx"));
}

#[test]
fn inject_subcommand_alone() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx web")),
        args("bx web")
    );
}

#[test]
fn inject_skips_multiple_value_flags() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --api-key KEY --timeout 30 query")),
        args("bx --api-key KEY --timeout 30 context query")
    );
}

#[test]
fn inject_value_flag_then_double_dash() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --timeout 30 -- web")),
        args("bx --timeout 30 context -- web")
    );
}

#[test]
fn inject_value_flag_at_end() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --timeout")),
        args("bx --timeout")
    );
}

#[test]
fn inject_unknown_flag_before_query() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --verbose query")),
        args("bx --verbose context query")
    );
}

#[test]
fn inject_equals_form_flag() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --api-key=KEY query")),
        args("bx --api-key=KEY context query")
    );
}

// ── inject_default_subcommand_impl with --config ───────────────

#[test]
fn inject_skips_config_flag() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --config /tmp/c.json myquery")),
        args("bx --config /tmp/c.json context myquery")
    );
}

#[test]
fn inject_config_equals_form() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --config=/tmp/c.json myquery")),
        args("bx --config=/tmp/c.json context myquery")
    );
}

#[test]
fn inject_config_with_subcommand() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --config /tmp/c.json web myquery")),
        args("bx --config /tmp/c.json web myquery")
    );
}

#[test]
fn inject_config_with_other_flags() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --config /tmp/c.json --timeout 60 myquery")),
        args("bx --config /tmp/c.json --timeout 60 context myquery")
    );
}

#[test]
fn inject_config_dangling_at_end() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --config")),
        args("bx --config")
    );
}

#[test]
fn inject_config_before_double_dash() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --config /tmp/c.json -- web")),
        args("bx --config /tmp/c.json context -- web")
    );
}

// ── unescape_inline_newlines tests ──────────────────────────────

#[test]
fn unescape_newline() {
    assert_eq!(unescape_inline_newlines("a\\nb"), "a\nb");
}

#[test]
fn unescape_double_backslash() {
    assert_eq!(unescape_inline_newlines("a\\\\nb"), "a\\nb");
}

#[test]
fn unescape_noop() {
    assert_eq!(unescape_inline_newlines("plain text"), "plain text");
}

#[test]
fn unescape_trailing_backslash() {
    assert_eq!(unescape_inline_newlines("end\\"), "end\\");
}

#[test]
fn unescape_other_escape() {
    assert_eq!(unescape_inline_newlines("a\\tb"), "a\\tb");
}

#[test]
fn unescape_empty() {
    assert_eq!(unescape_inline_newlines(""), "");
}

#[test]
fn unescape_only_newline() {
    assert_eq!(unescape_inline_newlines("\\n"), "\n");
}

#[test]
fn unescape_newline_at_start() {
    assert_eq!(unescape_inline_newlines("\\ntext"), "\ntext");
}

#[test]
fn unescape_newline_at_end() {
    assert_eq!(unescape_inline_newlines("text\\n"), "text\n");
}

#[test]
fn unescape_consecutive_newlines() {
    assert_eq!(unescape_inline_newlines("a\\n\\nb"), "a\n\nb");
}

#[test]
fn unescape_mixed() {
    assert_eq!(unescape_inline_newlines("a\\nb\\\\c\\nd"), "a\nb\\c\nd");
}

#[test]
fn unescape_only_backslash() {
    assert_eq!(unescape_inline_newlines("\\"), "\\");
}

#[test]
fn unescape_triple_backslash() {
    assert_eq!(unescape_inline_newlines("\\\\\\"), "\\\\");
}

#[test]
fn unescape_triple_backslash_n() {
    assert_eq!(unescape_inline_newlines("\\\\\\n"), "\\\n");
}

#[test]
fn unescape_quadruple_backslash() {
    assert_eq!(unescape_inline_newlines("\\\\\\\\"), "\\\\");
}

// ── parse_extra ──────────────────────────────────────────────────

#[test]
fn parse_extra_basic() {
    assert_eq!(parse_extra(&["key=value".into()]), vec![("key", "value")]);
}

#[test]
fn parse_extra_value_with_equals() {
    assert_eq!(parse_extra(&["k=a=b".into()]), vec![("k", "a=b")]);
}

#[test]
fn parse_extra_empty_value() {
    assert_eq!(parse_extra(&["key=".into()]), vec![("key", "")]);
}

#[test]
fn parse_extra_multiple() {
    assert_eq!(
        parse_extra(&["a=1".into(), "b=2".into()]),
        vec![("a", "1"), ("b", "2")]
    );
}

#[test]
fn parse_extra_empty_input() {
    let empty: Vec<String> = vec![];
    assert!(parse_extra(&empty).is_empty());
}

// ── inject_default_subcommand_impl with --extra / --endpoint ─────

#[test]
fn inject_extra_before_subcommand() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --extra foo=bar web query")),
        args("bx --extra foo=bar web query")
    );
}

#[test]
fn inject_extra_before_query() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --extra foo=bar query")),
        args("bx --extra foo=bar context query")
    );
}

#[test]
fn inject_extra_equals_form() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --extra=foo=bar query")),
        args("bx --extra=foo=bar context query")
    );
}

#[test]
fn inject_multiple_extras() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --extra a=1 --extra b=2 query")),
        args("bx --extra a=1 --extra b=2 context query")
    );
}

#[test]
fn inject_endpoint_before_subcommand() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --endpoint /custom web query")),
        args("bx --endpoint /custom web query")
    );
}

#[test]
fn inject_endpoint_before_query() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --endpoint /custom query")),
        args("bx --endpoint /custom context query")
    );
}

#[test]
fn inject_extra_at_end() {
    // --extra with no value: loop ends without panic, clap will error later
    assert_eq!(
        inject_default_subcommand_impl(args("bx --extra")),
        args("bx --extra")
    );
}

#[test]
fn inject_endpoint_equals_form() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --endpoint=/custom query")),
        args("bx --endpoint=/custom context query")
    );
}

#[test]
fn inject_extra_and_endpoint_combined() {
    assert_eq!(
        inject_default_subcommand_impl(args("bx --extra a=1 --endpoint /p query")),
        args("bx --extra a=1 --endpoint /p context query")
    );
}

#[test]
fn inject_extra_value_looks_like_subcommand() {
    // --extra consumes "web=1" as its value, not as a subcommand
    assert_eq!(
        inject_default_subcommand_impl(args("bx --extra web=1 query")),
        args("bx --extra web=1 context query")
    );
}

// ── walk: one reading of argv for both readers ──────────────────

/// `subcommand_in` and `format_in` choose JSON or prose for a refusal, so they must read argv as
/// clap does: value flags consume their value, and `--` ends the options.
#[test]
fn both_argv_readers_skip_a_value_and_stop_at_the_escape() {
    let sub = |s: &str| subcommand_in(&args(s)).map(str::to_string);
    let fmt = |s: &str| format_in(&args(s)).to_string();

    // A value flag's value is a value, never a sub-command and never a `--format`.
    assert_eq!(sub("bx --config /tmp/c council x"), Some("council".into()));
    assert_eq!(sub("bx --extra web=1 query"), Some("query".into()));
    assert_eq!(
        fmt("bx council --config --format text --min-quorum 0"),
        "json"
    );
    assert_eq!(fmt("bx council --config=--format --format text q"), "text");

    // After `--` a word is neither a subcommand nor an option.
    assert_eq!(sub("bx --config -- council"), None);
    assert_eq!(sub("bx context -- council"), Some("context".into()));
    assert_eq!(fmt("bx council -- --format text"), "json");

    // And the ordinary readings still hold.
    assert_eq!(sub("bx council q"), Some("council".into()));
    assert_eq!(sub("bx --help"), None);
    assert_eq!(fmt("bx council --format text q"), "text");
    assert_eq!(fmt("bx council --format=text q"), "text");
    assert_eq!(fmt("bx council q"), "json");
}
