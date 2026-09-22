//! Tests for `cfg.rs`, as a child module so they reach its private items.

use super::*;

use super::super::wire::{ANTHROPIC, OPENAI, OPENROUTER};

#[test]
fn the_allowlist_is_the_wire_host_and_does_not_leak_across_providers() {
    assert!(check_base_url(&ANTHROPIC, "https://openrouter.ai").is_err());
    assert!(check_base_url(&OPENROUTER, "https://openrouter.ai").is_ok());
    assert!(check_base_url(&OPENAI, "https://api.openai.com/").is_ok());
}

#[test]
fn only_loopback_http_is_carved_out() {
    assert!(check_base_url(&OPENROUTER, "http://127.0.0.1:8080").is_ok());
    assert!(check_base_url(&OPENROUTER, "http://localhost:1").is_ok());
    assert!(check_base_url(&OPENROUTER, "http://evil.com").is_err());
    // Authority confusion: the real host is evil.com.
    assert!(check_base_url(&OPENROUTER, "http://127.0.0.1@evil.com").is_err());
    assert!(check_base_url(&OPENROUTER, "https://evil.com").is_err());
    assert!(
        check_base_url(&OPENROUTER, "http://[::1]:9").is_ok(),
        "bracketed IPv6"
    );
    assert!(
        check_base_url(&OPENROUTER, "http://169.254.169.254").is_err(),
        "link-local"
    );
    // The strict parser blocks the classic bypass spellings.
    assert!(
        check_base_url(&OPENROUTER, "http://0177.0.0.1").is_err(),
        "octal"
    );
    assert!(
        check_base_url(&OPENROUTER, "http://2130706433").is_err(),
        "decimal"
    );
    assert!(
        check_base_url(&OPENROUTER, "http://127.0.0.1.nip.io").is_err(),
        "dns bypass"
    );
}

#[test]
fn a_key_in_the_wrong_slot_is_refused_before_it_is_sent() {
    assert!(check_key("openrouter", "sk-ant-abcdefgh").is_err());
    assert!(check_key("anthropic", "sk-or-v1-abcdefgh").is_err());
    assert!(check_key("openai", "sk-ant-abcdefgh").is_err());
    assert!(check_key("anthropic", "sk-ant-abcdefgh").is_ok());
    assert!(check_key("anthropic", "short").is_err());
}

#[test]
fn each_wires_real_key_shape_is_accepted_in_its_own_slot_and_refused_in_the_others() {
    for own in super::super::wire::TABLE {
        let key = format!("{}abcdefgh", own.key_prefix);
        for slot in super::super::wire::TABLE {
            assert_eq!(
                check_key(slot.name, &key).is_ok(),
                slot.name == own.name,
                "{key} in {}",
                slot.name
            );
        }
    }
}

/// serde renders the offending value, and a key pasted into the wrong field is one.
#[test]
fn a_parse_error_locates_itself_without_reprinting_the_value() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("council.json");
    std::fs::write(&p, r#"{"max_usd":"sk-ant-api03-SUPERSECRET"}"#).unwrap();
    let e = load(Some(&p)).unwrap_err();
    assert!(!e.contains("SUPERSECRET"), "{e}");
    assert!(e.contains("line 1"), "{e}");
}

#[test]
fn an_unknown_council_key_is_named_but_a_secret_shaped_one_never_is() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("council.json");
    std::fs::write(&p, r#"{"paneles": {}}"#).unwrap();
    let e = load(Some(&p)).unwrap_err();
    assert!(e.contains("unknown key \"paneles\""), "{e}");
    std::fs::write(&p, r#"{"sk-or-v1-SECRET": 1}"#).unwrap();
    let e = load(Some(&p)).unwrap_err();
    assert!(e.contains("unknown key") && !e.contains("SECRET"), "{e}");
}

/// A misspelt vendor silently sent Opus through the router at another price and retention.
#[test]
fn a_direct_entry_that_names_no_direct_wire_is_refused_at_load() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("council.json");
    for bad in [r#"["Anthropic"]"#, r#"["bedrock"]"#, r#"[" anthropic"]"#] {
        std::fs::write(&p, format!(r#"{{"direct": {bad}}}"#)).unwrap();
        let e = load(Some(&p)).unwrap_err();
        assert!(e.contains("direct: unknown vendor"), "{bad}: {e}");
    }
    for ok in [r#"{"direct": []}"#, r#"{"direct": ["openai"]}"#, "{}"] {
        std::fs::write(&p, ok).unwrap();
        assert!(load(Some(&p)).is_ok(), "{ok}");
    }
}

/// An unpriced Brave call is never charged against `--max-usd`, so a typo would uncap the run.
#[test]
fn a_negative_brave_price_in_council_json_is_refused_at_load_not_silently_uncapped() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("council.json");
    for bad in ["-1", "1e303"] {
        std::fs::write(&p, format!(r#"{{"brave_price_per_call_usd": {bad}}}"#)).unwrap();
        let e = load(Some(&p)).unwrap_err();
        assert!(e.contains("brave_price_per_call_usd"), "{bad}: {e}");
    }
    std::fs::write(&p, r#"{"brave_price_per_call_usd": 0}"#).unwrap();
    assert!(load(Some(&p)).is_ok());
}

#[test]
fn a_council_debug_rendering_never_contains_a_key() {
    let c = Council {
        anthropic_api_key: Some("sk-ant-SECRET".into()),
        ..Council::default()
    };
    assert!(!format!("{c:?}").contains("SECRET"));
}

/// Parity with `config.json`: an empty file is the defaults.
#[test]
fn an_empty_council_json_means_the_same_as_an_empty_object() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("council.json");
    std::fs::write(&p, " \n").unwrap();
    assert!(load(Some(&p)).unwrap().max_usd.is_none());
}

#[test]
fn a_council_json_that_opens_with_a_bom_parses_like_one_without() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("council.json");
    std::fs::write(&p, "\u{feff}{\"zdr\": false}").unwrap();
    assert_eq!(load(Some(&p)).unwrap().zdr, Some(false));
}

#[test]
fn a_refused_base_url_never_echoes_any_part_of_itself() {
    for url in [
        "http://user:CANARY1@evil.example",
        "http://u:p@CANARY2@evil.example",
        "https://evil.example/CANARY3",
        "https://evil.example/?key=CANARY4",
        "https://evil.example/#CANARY5",
        "http://CANARY6.example",
    ] {
        let e = check_base_url(&OPENROUTER, url).unwrap_err();
        assert!(!e.contains("CANARY") && !e.contains("evil"), "{url}: {e}");
    }
}

/// A negative or non-finite rate is not a rate: a negative one made the priced worst case
/// negative, which satisfies any `--max-usd`.
#[test]
fn a_rate_that_is_not_a_rate_is_refused() {
    let mut c = Council::default();
    for bad in [
        vec![-1.0, 25.0],
        vec![5.0, -25.0],
        vec![f64::NAN, 1.0],
        vec![f64::INFINITY, 1.0],
    ] {
        c.rates.insert("v/m".into(), bad.clone());
        assert_eq!(rate_for(&c, "v/m"), None, "accepted {bad:?}");
    }
    // Zero is a real rate: a free endpoint is not an unknown one.
    c.rates.insert("v/m".into(), vec![0.0, 0.0]);
    assert_eq!(rate_for(&c, "v/m"), Some([0.0, 0.0, 0.1]));
}

#[test]
fn rates_are_slug_keyed_and_config_overrides_the_fallback() {
    let mut c = Council::default();
    assert_eq!(
        rate_for(&c, "anthropic/claude-opus-5.5"),
        Some([4.0, 20.0, 0.05])
    );
    assert_eq!(rate_for(&c, "openai/gpt-5.6-sol"), None);
    c.rates
        .insert("anthropic/claude-opus-5.5".into(), vec![1.0, 2.0]);
    assert_eq!(
        rate_for(&c, "anthropic/claude-opus-5.5"),
        Some([1.0, 2.0, 0.1])
    );
}

/// The third element prices cache reads per model: a multiplier on input, above 0 and at most 1.
#[test]
fn a_configured_rate_may_carry_a_cache_read_multiplier() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("council.json");
    std::fs::write(
        &p,
        r#"{"rates": {"a/two": [1, 2], "a/three": [1, 2, 0.5]}}"#,
    )
    .unwrap();
    let c = load(Some(&p)).unwrap();
    assert_eq!(rate_for(&c, "a/two"), Some([1.0, 2.0, 0.1]));
    assert_eq!(rate_for(&c, "a/three"), Some([1.0, 2.0, 0.5]));
    let mut c = Council::default();
    for bad in [
        vec![1.0],
        vec![1.0, 2.0, 0.0],
        vec![1.0, 2.0, 1.5],
        vec![1.0, 2.0, 0.1, 0.1],
    ] {
        c.rates.insert("v/m".into(), bad.clone());
        assert_eq!(rate_for(&c, "v/m"), None, "accepted {bad:?}");
    }
}

/// Unsuffixed v4-pro is an older, dearer build.
#[test]
fn the_deepseek_alias_names_the_dated_pro_build() {
    let c = Council::default();
    assert_eq!(
        alias(&c, "deepseek").unwrap(),
        "deepseek/deepseek-v4-pro-0813"
    );
}

#[test]
fn an_alias_chain_longer_than_nine_hops_resolves_to_the_model() {
    let mut c = Council::default();
    for i in 0..12 {
        c.aliases.insert(format!("a{i}"), format!("a{}", i + 1));
    }
    c.aliases.insert("a12".into(), "direct:v/model".into());
    assert_eq!(alias(&c, "a0").unwrap(), "direct:v/model");
}

/// Loopback is resolved once and pinned to the literal IP, as the Brave allowlist is: ureq
/// re-resolves a name on every connect.
#[test]
fn the_loopback_carve_out_resolves_and_pins_exactly_as_the_brave_path_does() {
    let w = &super::super::wire::OPENROUTER;
    assert_eq!(
        check_base_url(w, "http://localhost:8080").unwrap(),
        "http://127.0.0.1:8080"
    );
    // A literal is already pinned, v4 and v6.
    for ok in ["http://127.0.0.1:8080", "http://[::1]:8080"] {
        assert_eq!(check_base_url(w, ok).unwrap(), ok);
    }
    for bad in [
        "http://[::1",               // unterminated bracket
        "http://127.0.0.1:0",        // port 0
        "http://127.0.0.1:99999abc", // not a port
        "http://[::1]garbage:8080",  // trailing junk after the bracket
        "http://127.0.0.1:8080?q=1", // query in the authority
        "http://127.0.0.1:8080#f",   // fragment in the authority
        "http://127.0.0.1@evil.com", // userinfo: the real host is evil.com
        "http://localhost.evil.com", // DNS-service bypass
        "http://evil.com",
    ] {
        assert!(check_base_url(w, bad).is_err(), "accepted {bad}");
    }
}

#[test]
fn a_loopback_base_url_with_a_query_or_fragment_is_refused() {
    assert!(check_base_url(&OPENROUTER, "http://127.0.0.1:8080/v1").is_ok());
    for bad in ["http://127.0.0.1:8080/v1?x=1", "http://127.0.0.1:8080/#f"] {
        assert!(check_base_url(&OPENROUTER, bad).is_err(), "accepted {bad}");
    }
}

/// Each clause rejects a finite value that would still corrupt a published figure.
#[test]
fn a_price_is_publishable_only_if_it_survives_the_arithmetic_that_prints_it() {
    assert_eq!(publishable_usd(0.0), Some(0.0), "free is a price");
    assert_eq!(publishable_usd(12.5), Some(12.5));
    // Negative: a negative bound satisfies any `--max-usd`.
    assert_eq!(publishable_usd(-0.01), None);
    assert_eq!(publishable_usd(f64::NAN), None);
    assert_eq!(publishable_usd(f64::INFINITY), None);
    // Finite once scaled: `output::micro` multiplies by 1e6, and infinity renders as `null`.
    assert_eq!(publishable_usd(f64::MAX), None);
    assert!(
        publishable_usd(1e303).is_none(),
        "1e303 * 1e6 overflows to infinity"
    );
    assert_eq!(publishable_usd(1e302), Some(1e302), "1e308 is still finite");
}

/// `council.json` holds keys, so it never lives in Windows' roaming `%APPDATA%`.
#[test]
fn the_key_file_never_lives_in_a_directory_that_roams() {
    let p = path().unwrap();
    assert!(p.ends_with("brave-search/council.json"), "{p:?}");
    assert!(p.starts_with(dirs::config_local_dir().unwrap()), "{p:?}");
    if cfg!(windows) {
        assert!(
            !p.starts_with(dirs::config_dir().unwrap()),
            "the roaming profile is the one answer this must never give: {p:?}"
        );
    }
}

/// The operator's price, else Brave's list price; never the list price in place of a bad value.
#[test]
fn the_brave_price_is_the_operators_then_braves_and_never_a_papered_over_mistake() {
    let mut c = Council::default();
    assert_eq!(brave_price(&c), Some(0.005), "Brave's published list price");
    assert!(!brave_price_is_configured(&c));

    c.brave_price_per_call_usd = Some(0.009);
    assert_eq!(brave_price(&c), Some(0.009));
    assert!(brave_price_is_configured(&c));

    // An explicit zero (free credits, an internal endpoint) is kept.
    c.brave_price_per_call_usd = Some(0.0);
    assert_eq!(brave_price(&c), Some(0.0));

    for bad in [-0.001, f64::NAN, f64::INFINITY, f64::MAX] {
        c.brave_price_per_call_usd = Some(bad);
        assert_eq!(brave_price(&c), None, "{bad} was papered over");
    }
}
