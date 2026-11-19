//! Tests for `query.rs`; a child module, so private items stay reachable.

use super::*;

// ── parse_json ───────────────────────────────────────────────────

const TOKEN: &str = "$serde_json::private::RawValue";

/// Under `raw_value`, `from_str::<Value>` fails on or rewrites an object whose first key is the
/// private token; `parse_json` must treat it as an ordinary key.
#[test]
fn parse_json_is_not_affected_by_the_raw_value_token() {
    // Silent rewrite: a bare `from_str` turns the document into `{"x":1}`.
    let rewritten = format!(r#"{{"{TOKEN}":"{{VAL}}"}}"#).replace("VAL", "\\\"x\\\":1");
    let v = parse_json(&rewritten).expect("must parse as an ordinary object");
    assert!(v.get(TOKEN).is_some(), "the key was consumed");
    assert!(v.get("x").is_none(), "the document was rewritten");

    // Hard failure: a bare `from_str` rejects this.
    let refused = format!(r#"{{"{TOKEN}":"hello","b":1}}"#);
    assert_eq!(parse_json(&refused).unwrap()["b"], serde_json::json!(1));

    // Only the first key is special-cased; pinned anyway.
    let later = format!(r#"{{"a":1,"{TOKEN}":"hello"}}"#);
    assert_eq!(parse_json(&later).unwrap()["a"], serde_json::json!(1));
}

/// The token is special-cased at every depth, so walking only the top-level object would leave
/// nested ones exposed.
#[test]
fn parse_json_ignores_the_raw_value_token_at_every_depth() {
    let payload = format!(r#"{{"{TOKEN}":"{{\"x\":1}}"}}"#);
    for wrapped in [
        format!(r#"{{"a":{payload}}}"#),                 // one object deep
        format!(r#"{{"a":{{"b":{{"c":{payload}}}}}}}"#), // three deep
        format!(r#"[{payload}]"#),                       // through an array
        format!(r#"{{"a":[[{payload}]]}}"#),             // array inside an object
    ] {
        let v = parse_json(&wrapped).unwrap_or_else(|e| panic!("{wrapped}: {e}"));
        let found = std::iter::successors(Some(&v), |v| match v {
            serde_json::Value::Object(o) => o.values().next(),
            serde_json::Value::Array(a) => a.first(),
            _ => None,
        })
        .find_map(|v| v.get(TOKEN));
        assert!(found.is_some(), "token key was consumed in {wrapped}");
    }
}

/// A string that quotes the token is data: any reply discussing `query.rs` contains one.
#[test]
fn parse_json_accepts_a_string_that_merely_quotes_the_token() {
    let v = parse_json(&format!(r#"{{"text":"see `{TOKEN}` in query.rs"}}"#)).unwrap();
    assert_eq!(v["text"], format!("see `{TOKEN}` in query.rs").as_str());
}

/// The byte path is the one every provider response takes, so it gets its own pin.
#[test]
fn parse_json_slice_ignores_the_token_too() {
    let nested = format!(r#"{{"a":{{"{TOKEN}":"{{\"x\":1}}"}}}}"#);
    let v = parse_json_slice(nested.as_bytes()).expect("must parse");
    assert!(v["a"].get(TOKEN).is_some(), "the key was consumed");
    assert!(v["a"].get("x").is_none(), "the document was rewritten");
}

/// Every other shape, and every error message, must be byte-identical to `from_str`.
#[test]
fn parse_json_matches_from_str_everywhere_else() {
    for input in [
        r#"{"messages":[{"role":"user"}],"stream":false}"#,
        "[1,2,3]",
        r#""a string""#,
        "42",
        "1e10",
        "null",
        "true",
        r#"{"a":1,}"#,
        "{oops",
        "",
        "   ",
        // A lone surrogate and a duplicate key: where going through `Map` could diverge from
        // `Value`.
        r#"{"a":"\ud800"}"#,
        r#"{"a":1,"a":2}"#,
        "[1,2",
    ] {
        match (
            serde_json::from_str::<serde_json::Value>(input),
            parse_json(input),
        ) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "value differs for {input:?}"),
            (Err(a), Err(b)) => {
                assert_eq!(a.to_string(), b.to_string(), "error differs for {input:?}");
            }
            (a, b) => panic!("outcome differs for {input:?}: {a:?} vs {b:?}"),
        }
    }
}

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
    assert_eq!(auto_detect_json_type("-3.15"), serde_json::json!(-3.15));
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

// ── json_bool ────────────────────────────────────────────────────

/// `--extra stream=0` types to a JSON number; a bool-only read would ignore what the wire
/// says.
#[test]
fn json_bool_reads_booleans_and_the_numbers_0_and_1() {
    assert_eq!(json_bool(&serde_json::json!(true)), Some(true));
    assert_eq!(json_bool(&serde_json::json!(false)), Some(false));
    assert_eq!(json_bool(&serde_json::json!(1)), Some(true));
    assert_eq!(json_bool(&serde_json::json!(0)), Some(false));
}

/// Anything else is left to the caller's default: reading `2` as `true` would misreport what
/// was sent.
#[test]
fn json_bool_refuses_to_guess_at_anything_else() {
    for v in [
        serde_json::json!(-1),
        serde_json::json!(2),
        serde_json::json!(1.0), // as_i64 is None for a float, even a whole one
        serde_json::json!(0.0),
        serde_json::json!("true"), // auto-typing already natives a bare `true`
        serde_json::json!("1"),
        serde_json::json!(""),
        serde_json::json!(null),
        serde_json::json!([]),
        serde_json::json!({}),
    ] {
        assert_eq!(json_bool(&v), None, "{v}");
    }
}

/// A number outside `i64` has no `as_i64`; it reads as unset, not a panic or a saturated 1.
#[test]
fn json_bool_treats_an_out_of_range_number_as_unset() {
    let huge: serde_json::Value = serde_json::from_str("18446744073709551615").unwrap();
    assert_eq!(json_bool(&huge), None);
    let negative: serde_json::Value = serde_json::from_str("-9223372036854775809").unwrap();
    assert_eq!(json_bool(&negative), None);
}

/// An absent key indexes to `Null`; it must read as "unset", never as `false`.
#[test]
fn json_bool_treats_a_missing_key_as_unset() {
    assert_eq!(json_bool(&serde_json::json!({})["stream"]), None);
    assert_eq!(json_bool(&serde_json::json!([1, 2])["stream"]), None);
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
    let s = serde_json::to_string(&body).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert_eq!(parsed["q"], "hello \"world\"\nnewline");
}

// ── merge_extra_into_json ────────────────────────────────────────

#[test]
fn merge_extra_adds_new_keys() {
    let mut body = serde_json::json!({"q": "test"});
    merge_extra_into_json(&mut body, &[("count", "5"), ("custom", "val")]).unwrap();
    assert_eq!(body["count"], 5);
    assert_eq!(body["custom"], "val");
    assert_eq!(body["q"], "test"); // unchanged
}

#[test]
fn merge_extra_overrides_existing() {
    let mut body = serde_json::json!({"count": 20});
    merge_extra_into_json(&mut body, &[("count", "5")]).unwrap();
    assert_eq!(body["count"], 5);
}

#[test]
fn merge_extra_empty_noop() {
    let mut body = serde_json::json!({"q": "test"});
    let original = body.clone();
    merge_extra_into_json(&mut body, &[]).unwrap();
    assert_eq!(body, original);
}

#[test]
fn merge_extra_empty_on_non_object() {
    // Empty extras returns early before the object check, so non-object bodies are fine.
    let mut body = serde_json::json!([1, 2, 3]);
    merge_extra_into_json(&mut body, &[]).unwrap();
    assert_eq!(body, serde_json::json!([1, 2, 3]));
}

#[test]
fn merge_extra_rejects_non_object() {
    let mut body = serde_json::json!([1, 2, 3]);
    assert_eq!(
        merge_extra_into_json(&mut body, &[("key", "val")]).unwrap_err(),
        "--extra requires a JSON object body"
    );
    assert_eq!(body, serde_json::json!([1, 2, 3])); // unchanged

    for val in [
        serde_json::json!(null),
        serde_json::json!("str"),
        serde_json::json!(42),
        serde_json::json!(true),
    ] {
        let mut body = val.clone();
        assert!(merge_extra_into_json(&mut body, &[("k", "v")]).is_err());
    }
}

#[test]
fn merge_extra_auto_detects_types() {
    let mut body = serde_json::json!({});
    merge_extra_into_json(&mut body, &[("n", "42"), ("b", "true"), ("s", "hello")]).unwrap();
    assert_eq!(body["n"], 42);
    assert_eq!(body["b"], true);
    assert_eq!(body["s"], "hello");
}

#[test]
fn merge_extra_duplicate_last_wins() {
    let mut body = serde_json::json!({"q": "test"});
    merge_extra_into_json(&mut body, &[("k", "1"), ("k", "2")]).unwrap();
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
fn build_query_repeated_extras_override_repeated() {
    let ids = vec!["a".into(), "b".into()];
    assert_eq!(
        build_query_repeated(&[], &[("ids", &ids)], &[("ids", "override")]),
        "?ids=override"
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
