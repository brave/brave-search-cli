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

/// Reads a JSON boolean, also accepting the `0`/`1` that `auto_detect_json_type` makes of
/// `--extra stream=0`. Nothing else: bx must not invent an undocumented truthiness rule.
pub fn json_bool(v: &serde_json::Value) -> Option<bool> {
    v.as_bool().or_else(|| match v.as_i64()? {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    })
}

/// A `Value` that treats serde_json's private `RawValue` token as an ordinary key, at any depth.
///
/// With the `raw_value` feature (which `council` needs to echo a provider's turn byte for byte),
/// `Value` reads an object whose first key is that token as a raw value, failing or silently
/// rewriting the document. Everything but objects is `Value`'s own, so numbers, escapes,
/// duplicate keys and error text are unchanged.
struct Plain(serde_json::Value);

impl<'de> serde::Deserialize<'de> for Plain {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{MapAccess, SeqAccess, Visitor};
        use serde_json::Value;

        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Plain;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("any valid JSON value")
            }

            // The two recursive shapes, so a nested object is walked too.
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Plain, A::Error> {
                let mut m = serde_json::Map::new();
                while let Some((k, Plain(v))) = a.next_entry()? {
                    m.insert(k, v);
                }
                Ok(Plain(Value::Object(m)))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Plain, A::Error> {
                let mut v = Vec::new();
                while let Some(Plain(x)) = a.next_element()? {
                    v.push(x);
                }
                Ok(Plain(Value::Array(v)))
            }

            // Every scalar is whatever `Value` would have made of it.
            fn visit_bool<E>(self, v: bool) -> Result<Plain, E> {
                Ok(Plain(v.into()))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Plain, E> {
                Ok(Plain(v.into()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Plain, E> {
                Ok(Plain(v.into()))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Plain, E> {
                Ok(Plain(
                    serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number),
                ))
            }
            fn visit_str<E>(self, v: &str) -> Result<Plain, E> {
                Ok(Plain(v.into()))
            }
            fn visit_string<E>(self, v: String) -> Result<Plain, E> {
                Ok(Plain(v.into()))
            }
            fn visit_unit<E>(self) -> Result<Plain, E> {
                Ok(Plain(Value::Null))
            }
        }

        d.deserialize_any(V)
    }
}

/// Parses JSON into a `Value` exactly as `serde_json::from_str` would without `raw_value` (see
/// `Plain`). Errors are serde_json's own.
pub fn parse_json(s: &str) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::from_str::<Plain>(s).map(|p| p.0)
}

/// `parse_json` for a body that arrived as bytes.
pub fn parse_json_slice(b: &[u8]) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::from_slice::<Plain>(b).map(|p| p.0)
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
#[path = "query_tests.rs"]
mod tests;
