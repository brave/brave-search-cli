//! Query strings and JSON bodies for the API. No HTTP; `--extra` collision warnings go
//! to stderr.

/// Builds a query string from key-value pairs, skipping None values.
/// Values are URL-encoded. Extras override params with the same key.
pub fn build_query(params: &[(&str, Option<&str>)], extras: &[(&str, &str)]) -> String {
    build_query_repeated(params, &[], extras)
}

/// Infers a JSON type from a string value:
/// i64 → integer, finite f64 → float, "true"/"false" → bool, else string.
fn auto_detect_json_type(v: &str) -> serde_json::Value {
    if let Ok(n) = v.parse::<i64>() {
        n.into()
    } else if let Ok(f) = v.parse::<f64>() {
        serde_json::Number::from_f64(f).map_or_else(|| v.into(), serde_json::Value::Number)
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
pub fn merge_extra_into_json(
    body: &mut serde_json::Value,
    extras: &[(&str, &str)],
) -> Result<(), String> {
    if extras.is_empty() {
        return Ok(());
    }
    let Some(obj) = body.as_object_mut() else {
        return Err("--extra requires a JSON object body".into());
    };
    for &(key, val) in extras {
        if obj.contains_key(key) {
            eprintln!("warning: --extra '{key}={val}' overrides existing parameter");
        }
        obj.insert(key.into(), auto_detect_json_type(val));
    }
    Ok(())
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
        if let Some(&(_, ev)) = extras.iter().find(|(k, _)| *k == key) {
            eprintln!("warning: --extra '{key}={ev}' overrides existing parameter");
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
