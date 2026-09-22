//! One `static` per provider. `build` is the only code that knows a request format, `parse` the
//! only code that knows a response shape; the rest is data, so `host` is a real allowlist.

use serde::Deserialize;
use serde_json::{Value, json, value::RawValue};
use std::borrow::Cow;

use crate::query::build_json_body;

use super::types::{Call, Conv, Effort, Item, Params, Stop, Turn, Usage};
use crate::api::Resp;

/// What one wire's model list says about a model: the constraints the pre-flight judges. Each
/// wire publishes a different document, so each has its own reader.
#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct Served {
    pub id: String,
    /// The largest output cap it accepts, where published.
    #[serde(default)]
    pub max_out: Option<u32>,
    /// The effort levels it accepts; `None` judges nothing.
    #[serde(default)]
    pub efforts: Option<Vec<String>>,
    /// Reasoning cannot be turned off, so `effort: "none"` is rejected.
    #[serde(default)]
    pub effort_mandatory: bool,
    /// Whether it takes `thinking: adaptive`, which the Anthropic wire always sends; `None` where
    /// unpublished.
    #[serde(default)]
    pub adaptive: Option<bool>,
}

pub struct Wire {
    pub name: &'static str,
    /// The only host `cfg::check_base_url` accepts, besides loopback.
    pub host: &'static str,
    pub path: &'static str,
    /// The model list the pre-flight reads, on the same host.
    pub models: &'static str,
    /// The slug as this wire spells it: judged by the pre-flight and sent by `build`.
    pub id: fn(&str) -> Cow<'_, str>,
    pub tools: &'static str,
    /// (header name, value prefix): the name differs too.
    pub auth: (&'static str, &'static str),
    /// Key variable, signup page and key prefix: fields, so a new wire cannot omit one.
    pub env: &'static str,
    pub signup: &'static str,
    pub key_prefix: &'static str,
    pub headers: &'static [(&'static str, &'static str)],
    /// The smallest output cap sent. Reasoning counts against it; OpenAI can spend all of it
    /// before any text, so it needs far more.
    pub min_out: u32,
    pub effort: fn(Effort) -> Value,
    /// The level `effort` puts on the request, as the dry run shows it.
    pub level: fn(Effort) -> &'static str,
    /// Reads one element of the model list.
    pub served: fn(&Value) -> Option<Served>,
    /// Why this wire will reject this model at this effort, judged on what `effort` sends.
    pub rejects: fn(&Served, Effort) -> Option<String>,
    /// The OpenRouter namespace whose public roster entry supplies the ceiling and efforts this
    /// wire's own list omits.
    pub public: Option<&'static str>,
    pub build: fn(&Conv, &Params) -> Vec<u8>,
    /// Infallible: an unreadable body is classified by its status.
    pub parse: fn(&Resp) -> Turn,
    /// Indexed by `zdr`; see [`retention_of`].
    pub retention: [Retention; 2],
}

/// One wire's retention posture in one `zdr` state. `doctor` prints `detail` and the manifest
/// records `mechanism`, from this one table so the two cannot disagree.
pub struct Retention {
    pub mechanism: &'static str,
    /// Whether the request itself carries the guarantee.
    pub enforced_per_request: bool,
    pub detail: &'static str,
}

/// Both states are rows: `zdr` off is a configuration too (on OpenAI, an active `store: true`).
pub fn retention_of(w: &'static Wire, zdr: bool) -> &'static Retention {
    &w.retention[usize::from(zdr)]
}

pub static OPENROUTER: Wire = Wire {
    name: "openrouter",
    host: "https://openrouter.ai",
    path: "/api/v1/chat/completions",
    models: "/api/v1/models",
    id: |s| Cow::Borrowed(s),
    tools: OR_TOOLS,
    auth: ("Authorization", "Bearer "),
    env: "OPENROUTER_API_KEY",
    signup: "https://openrouter.ai/keys",
    key_prefix: "sk-or-v1-",
    headers: &[],
    min_out: 1024,
    effort: reasoning_effort,
    level: effort_on_wire,
    served: or_served,
    rejects: reasoning_rejects,
    public: None,
    build: or_build,
    parse: or_parse,
    retention: [
        Retention {
            mechanism: "provider_default",
            enforced_per_request: false,
            detail: "sent no routing restriction, so the router may pick any endpoint, \
                     subject to your account settings. \"zdr\": true restricts it",
        },
        Retention {
            mechanism: "endpoint_routing",
            enforced_per_request: true,
            detail: "sent provider.zdr, so only endpoints OpenRouter lists as zero-retention \
                     serve it, failing rather than falling back. Inference only: search \
                     queries reach Brave either way",
        },
    ],
};

pub static ANTHROPIC: Wire = Wire {
    name: "anthropic",
    host: "https://api.anthropic.com",
    path: "/v1/messages",
    models: "/v1/models?limit=1000",
    id: ant_id,
    tools: ANT_TOOLS,
    auth: ("x-api-key", ""),
    env: "ANTHROPIC_API_KEY",
    signup: "https://console.anthropic.com/settings/keys",
    key_prefix: "sk-ant-",
    headers: &[("anthropic-version", "2023-06-01")],
    min_out: 1024,
    effort: ant_effort,
    level: ant_level,
    served: ant_served,
    rejects: ant_rejects,
    public: None,
    build: ant_build,
    parse: ant_parse,
    // `zdr` changes nothing this wire sends.
    retention: [
        Retention {
            mechanism: "account_policy",
            enforced_per_request: false,
            detail: ANT_RETENTION,
        },
        Retention {
            mechanism: "account_policy",
            enforced_per_request: false,
            detail: ANT_RETENTION,
        },
    ],
};

const ANT_RETENTION: &str =
    "sent no retention control, because the API has none: your organisation's settings decide";

pub static OPENAI: Wire = Wire {
    name: "openai",
    host: "https://api.openai.com",
    path: "/v1/responses",
    models: "/v1/models",
    id: |s| Cow::Borrowed(bare_model(s)),
    tools: OAI_TOOLS,
    auth: ("Authorization", "Bearer "),
    env: "OPENAI_API_KEY",
    signup: "https://platform.openai.com/api-keys",
    key_prefix: "sk-proj-",
    headers: &[],
    min_out: 25_000,
    effort: reasoning_effort,
    level: effort_on_wire,
    served: oai_served,
    rejects: oai_rejects,
    public: Some("openai/"),
    build: oai_build,
    parse: oai_parse,
    retention: [
        Retention {
            mechanism: "store_true",
            enforced_per_request: false,
            detail: "sent store:true, so the response may be kept as retrievable application \
                     state, subject to your organisation's settings. \"zdr\": true sends \
                     store:false",
        },
        Retention {
            mechanism: "store_false",
            enforced_per_request: true,
            detail: "sent store:false, so the response is not kept as application state; any \
                     other retention follows your organisation's settings",
        },
    ],
};

pub static TABLE: [&Wire; 3] = [&OPENROUTER, &ANTHROPIC, &OPENAI];

/// A lookup, not a `match`: rustc rejects function pointers in patterns.
pub fn wire_for(name: &str) -> Result<&'static Wire, String> {
    TABLE
        .iter()
        .copied()
        .find(|w| w.name == name)
        .ok_or_else(|| {
            format!(
                "unknown transport '{name}' (expected one of: {})",
                TABLE.map(|w| w.name).join(", ")
            )
        })
}

// ── The tool, written once ───────────────────────────────────────────
//
// Macros because `concat!` takes only literals. The description must hold no unescaped `"` or
// `\`; `tool_schemas_are_valid_json_and_share_one_description` checks.

macro_rules! desc {
    () => {
        "Search the web for current, citable evidence. Use a focused natural-language or \
         keyword query of at most 50 words. Returns a JSON block of results, each with its \
         url, title, publication date where the source reported one, and extracted page text. \
         Weigh the dates: an undated or years-old page is weak evidence about current \
         behaviour. Search for what you cannot already justify; do not search to confirm what \
         you know."
    };
}

macro_rules! params {
    () => {
        r#"{"type":"object","properties":{"query":{"type":"string","description":"Search query. Plain language or keywords; 400 characters and 50 words maximum."}},"required":["query"],"additionalProperties":false}"#
    };
}

const OR_TOOLS: &str = concat!(
    r#"[{"type":"function","function":{"name":"search","description":""#,
    desc!(),
    r#"","parameters":"#,
    params!(),
    r#"}}]"#
);

const ANT_TOOLS: &str = concat!(
    r#"[{"name":"search","description":""#,
    desc!(),
    r#"","input_schema":"#,
    params!(),
    r#"}]"#
);

const OAI_TOOLS: &str = concat!(
    r#"[{"type":"function","name":"search","description":""#,
    desc!(),
    r#"","strict":true,"parameters":"#,
    params!(),
    r#"}]"#
);

// ── What each wire publishes about a model ───────────────────────────

/// OpenRouter: `top_provider.max_completion_tokens` and a `reasoning` object.
fn or_served(m: &Value) -> Option<Served> {
    let reasoning = &m["reasoning"];
    Some(Served {
        id: m["id"].as_str()?.to_string(),
        max_out: m["top_provider"]["max_completion_tokens"]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok()),
        efforts: reasoning["supported_efforts"].as_array().map(|a| {
            a.iter()
                .filter_map(|e| e.as_str().map(str::to_string))
                .collect()
        }),
        effort_mandatory: reasoning["mandatory"].as_bool().unwrap_or(false),
        // This wire never sends a thinking object.
        adaptive: None,
    })
}

/// Anthropic: `max_tokens` and `capabilities` (thinking types and effort levels).
fn ant_served(m: &Value) -> Option<Served> {
    let caps = &m["capabilities"];
    let effort = &caps["effort"];
    Some(Served {
        id: m["id"].as_str()?.to_string(),
        max_out: m["max_tokens"].as_u64().and_then(|n| u32::try_from(n).ok()),
        // Levels are keys beside `supported`, each with its own flag; `supported: false` alone
        // publishes none.
        efforts: effort.as_object().map(|o| {
            o.iter()
                .filter(|(k, v)| *k != "supported" && v["supported"].as_bool() == Some(true))
                .map(|(k, _)| k.clone())
                .collect()
        }),
        effort_mandatory: false,
        adaptive: caps["thinking"]["types"]["adaptive"]["supported"].as_bool(),
    })
}

/// OpenAI's list is ids only; [`lend`] fills in the rest from OpenRouter's public entry.
fn oai_served(m: &Value) -> Option<Served> {
    Some(Served {
        id: m["id"].as_str()?.to_string(),
        ..Served::default()
    })
}

/// Copies each model's ceiling and efforts from its public entry `{ns}{id}`, where one exists.
pub fn lend(ns: &str, own: &mut [Served], public: &[Served]) {
    for s in own {
        if let Some(p) = public.iter().find(|p| p.id.strip_prefix(ns) == Some(&s.id)) {
            (s.max_out, s.efforts, s.effort_mandatory) =
                (p.max_out, p.efforts.clone(), p.effort_mandatory);
        }
    }
}

/// Judged against `asked`, the level the request carries, never the typed one.
fn effort_unsupported(s: &Served, effort: Effort, asked: &str) -> Option<String> {
    let known = s.efforts.as_ref()?;
    if known.is_empty() {
        return Some("takes no effort level, and this transport always sends one".into());
    }
    (!known.iter().any(|e| e == asked)).then(|| {
        let typed = effort.as_str();
        let via = if typed == asked {
            String::new()
        } else {
            format!(", which `--effort {typed}` sends")
        };
        format!(
            "does not accept `--effort {asked}`{via} (it takes: {})",
            super::util::sanitize(&known.join(", "))
        )
    })
}

/// Judges the `reasoning.effort` that OpenRouter and OpenAI both send.
fn reasoning_rejects(s: &Served, effort: Effort) -> Option<String> {
    if effort == Effort::Off && s.effort_mandatory {
        return Some(
            "reasoning is mandatory on it, so `--effort none` is rejected — pick another effort"
                .to_string(),
        );
    }
    // OpenRouter disables reasoning for `none`, which only `mandatory` refuses: a Claude entry's
    // list omits `none` yet accepts it.
    if effort == Effort::Off {
        return None;
    }
    effort_unsupported(s, effort, effort_on_wire(effort))
}

/// OpenAI takes `none` as a level, so a lent list without it refuses it too.
fn oai_rejects(s: &Served, effort: Effort) -> Option<String> {
    reasoning_rejects(s, effort).or_else(|| {
        (effort == Effort::Off)
            .then(|| effort_unsupported(s, effort, Effort::Off.as_str()))
            .flatten()
    })
}

/// Every Anthropic request carries `thinking: adaptive`, so a model without it is refused.
fn ant_rejects(s: &Served, effort: Effort) -> Option<String> {
    if s.adaptive == Some(false) {
        return Some(
            "does not support adaptive thinking, which this transport always requests — route it \
             with `--via openrouter`, or pick a model that does"
                .to_string(),
        );
    }
    effort_unsupported(s, effort, ant_level(effort))
}

// ── Body assembly ────────────────────────────────────────────────────

/// A slug without its OpenRouter namespace, which the direct APIs 404 on.
pub(super) fn bare_model(slug: &str) -> &str {
    slug.rsplit_once('/').map_or(slug, |(_, bare)| bare)
}

/// The id this wire sends, which is also the id the pre-flight judges.
pub fn model_on_wire<'a>(w: &Wire, slug: &'a str) -> Cow<'a, str> {
    (w.id)(slug)
}

/// Anthropic spells `claude-opus-5.5` as `claude-opus-5-5`.
fn ant_id(slug: &str) -> Cow<'_, str> {
    Cow::Owned(bare_model(slug).replace('.', "-"))
}

/// The fixed part of every request: `v` plus the effort knob and the tools, left open at
/// `array` (`,"messages":[` or `,"input":[`) for the caller's items and `]}`. `v` always holds
/// `model`, so the reopened object is never empty.
fn open(w: &Wire, p: &Params, mut v: Value, array: &[u8]) -> Vec<u8> {
    if let (Some(obj), Value::Object(e)) = (v.as_object_mut(), (w.effort)(p.effort)) {
        obj.extend(e);
    }
    let mut b = v.to_string().into_bytes();
    b.pop();
    if p.tools {
        b.extend_from_slice(br#","tools":"#);
        b.extend_from_slice(w.tools.as_bytes());
    }
    b.extend_from_slice(array);
    b
}

/// OpenRouter's `provider` object, or `None` when neither retention nor routing is constrained;
/// omitted rather than sent as `zdr: false`. `only` is itself an allow-list; never add
/// `allow_fallbacks: false`, which means "primary provider only" and collapses a pin to its
/// first entry.
fn provider_route(p: &Params) -> Option<Value> {
    match (p.zdr, p.providers.as_slice()) {
        (false, []) => None,
        (true, []) => Some(json!({"zdr": true})),
        (false, only) => Some(json!({"only": only})),
        (true, only) => Some(json!({"zdr": true, "only": only})),
    }
}

/// Always sent, `none` included: an absent field means the provider's default, not no reasoning.
fn reasoning_effort(e: Effort) -> Value {
    json!({"reasoning": {"effort": effort_on_wire(e)}})
}

/// The effort OpenRouter and OpenAI send ([`ant_level`] for Anthropic). `minimal` goes out as
/// `low`, because few models take it.
fn effort_on_wire(e: Effort) -> &'static str {
    match e {
        Effort::Minimal => Effort::Low.as_str(),
        e => e.as_str(),
    }
}

/// The separator, between elements only.
fn sep(b: &mut Vec<u8>, first: &mut bool) {
    if !std::mem::take(first) {
        b.push(b',');
    }
}

/// Appends one array element.
fn elem(b: &mut Vec<u8>, first: &mut bool, json: &[u8]) {
    sep(b, first);
    b.extend_from_slice(json);
}

/// Serialises straight into the body: the seed is the whole subject, so a `String` copy per
/// message per round is a second copy of it.
fn push_val(b: &mut Vec<u8>, first: &mut bool, v: &Value) {
    sep(b, first);
    // A `Vec` writer cannot fail, and a panic would abort a paid run.
    serde_json::to_writer(&mut *b, v).ok();
}

/// A text block, optionally the cache breakpoint. OpenAI caches automatically and never uses it.
fn text_block(text: &str, cached: bool) -> Value {
    let mut v = json!({"type": "text", "text": text});
    if cached {
        v["cache_control"] = json!({"type": "ephemeral"});
    }
    v
}

// ── OpenRouter — POST /api/v1/chat/completions ───────────────────────

fn or_build(c: &Conv, p: &Params) -> Vec<u8> {
    let mark = c.mark_text(p.cache);
    let mut b = open(
        &OPENROUTER,
        p,
        build_json_body(&[
            ("model", Some(p.model.as_str().into())),
            ("max_completion_tokens", Some(p.max_tokens.into())),
            // The other two wires default to `auto`.
            ("tool_choice", p.tools.then(|| "auto".into())),
            // Sticky routing: a mid-run provider hop discards the cached prefix.
            ("session_id", Some(p.session.as_str().into())),
            // A silently compressed prompt would read as a bad review, not an error.
            (
                "plugins",
                Some(json!([{"id": "context-compression", "enabled": false}])),
            ),
            ("provider", provider_route(p)),
        ]),
        br#","messages":["#,
    );
    let mut first = true;
    push_val(
        &mut b,
        &mut first,
        &json!({"role": "system", "content": p.system}),
    );
    for (i, item) in c.items.iter().enumerate() {
        let cached = mark == Some(i);
        match item {
            Item::Seed(s) | Item::Nudge(s) => push_val(
                &mut b,
                &mut first,
                &json!({"role": "user", "content": [text_block(s, cached)]}),
            ),
            Item::Assistant(raw) => elem(&mut b, &mut first, raw.get().as_bytes()),
            Item::Result { id, body, .. } => push_val(
                &mut b,
                &mut first,
                &json!({"role": "tool", "tool_call_id": id, "content": body}),
            ),
        }
    }
    b.extend_from_slice(b"]}");
    b
}

/// Lifts `choices[0].message` out as raw bytes, so the echo goes back verbatim. `Option` per
/// choice, so a later choice without a `message` cannot fail the whole read.
#[derive(Deserialize)]
struct OrEcho {
    choices: Vec<Option<OrEchoChoice>>,
}
#[derive(Deserialize)]
struct OrEchoChoice {
    message: Option<Box<RawValue>>,
}

/// One message's text; `content` is `string | array | null`. Shared by the answer and error
/// paths.
fn text_of(msg: &Value) -> Option<String> {
    str_or(&msg["content"]).or_else(|| join_parts(&msg["content"]))
}

/// A body that could not be read. A 2xx was generated and charged, so its bill is marked
/// incomplete rather than reported as zero.
fn unreadable(r: &Resp) -> Turn {
    Turn::stopped(by_status(r, "unreadable response body"))
        .at(r.status)
        .billed(Usage {
            cost_incomplete: (200..300).contains(&r.status),
            ..Usage::default()
        })
}

fn or_parse(r: &Resp) -> Turn {
    let Some(v) = json_body(r) else {
        return unreadable(r);
    };
    let billed = or_usage(&v["usage"]);
    let fail = |s| Turn::stopped(s).at(r.status).billed(billed);
    // Three error shapes: an HTTP status, a top-level `error` on a 200, and a per-choice `error`.
    if r.status != 200 {
        return fail(or_error(r, &v["error"]));
    }
    let ch = &v["choices"][0];
    // Only the per-choice one carries partial content: kept as a draft, never an answer.
    if ch["error"].is_object() {
        return fail(or_error(r, &ch["error"])).salvaged(text_of(&ch["message"]), false);
    }
    if v["error"].is_object() {
        return fail(or_error(r, &v["error"]));
    }
    // No choice is a transport hiccup, not an empty answer the gate would write off.
    if v["choices"].as_array().is_none_or(|a| a.is_empty()) {
        return fail(Stop::Retry("no choice in the response".into()));
    }

    let msg = &ch["message"];
    // A refusal wears `finish_reason: "stop"`, so it is read apart, even beside content.
    let refusal = str_or(&msg["refusal"]);
    let refused = refusal.is_some();
    // `reasoning` only when nothing else came, and marked, so a scratchpad is never the review.
    let answer = text_of(msg).or(refusal);
    let thinking = if answer.is_some() {
        None
    } else {
        str_or(&msg["reasoning"])
    };
    let reasoning_only = thinking.is_some();
    let text = answer.or(thinking);
    let calls: Vec<Call> = msg["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|t| Call {
            id: t["id"].as_str().unwrap_or_default().to_string(),
            name: t["function"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            args: arg_string(&t["function"]["arguments"]),
        })
        .collect();
    if !ids_are_usable(&calls) {
        // Billed and unusable: the words still go to the draft.
        return fail(Stop::Retry(UNANSWERABLE.into())).salvaged(text, reasoning_only);
    }

    // May be lost while the calls survive; `drive` never answers calls it could not echo.
    let raw = shaped(
        serde_json::from_slice::<OrEcho>(&r.body)
            .ok()
            .and_then(|e| e.choices.into_iter().next().flatten())
            .and_then(|c| c.message),
        b'{',
    );
    let text = text.unwrap_or_default();
    let (quiet, callless) = (text.is_empty() && calls.is_empty(), calls.is_empty());
    Turn {
        text,
        calls,
        reasoning_only,
        stop: match ch["finish_reason"].as_str().filter(|s| !s.is_empty()) {
            _ if refused => Stop::Refused,
            Some("length") => Stop::Truncated,
            Some("content_filter") => Stop::Refused,
            Some("tool_calls") if callless => Stop::Retry(TOOL_STOP_NO_CALL.into()),
            // Partial content comes with it: not a finished review.
            Some("error") => Stop::Retry("provider reported finish_reason: error".into()),
            // An absent signal retries only an empty turn: many proxies omit it.
            None if quiet => Stop::Retry("choice carried no finish_reason".into()),
            _ => Stop::Done,
        },
        usage: billed,
        raw,
        chain: None,
        status: Some(r.status),
        served_model: v["model"].as_str().map(str::to_string),
        // Undocumented: read, never required.
        provider: v["provider"].as_str().map(str::to_string),
    }
}

/// OpenRouter's own error taxonomy, finer than the HTTP status. Spend limits are 402s; only the
/// in-flight one is transient.
fn or_error(r: &Resp, e: &Value) -> Stop {
    // `error_type` is the stable name; `error.code` here is an integer.
    let code = e["metadata"]["error_type"].as_str().unwrap_or_default();
    let msg = err_msg(e, code, &r.status.to_string());
    // A concurrency throttle wearing `payment_required`, so read before the code.
    if e["metadata"]["limit_source"].as_str() == Some("openrouter_in_flight_budget") {
        return Stop::Retry(msg);
    }
    match code {
        "rate_limit_exceeded"
        | "provider_overloaded"
        | "timeout"
        | "provider_unavailable"
        | "server"
        | "unmapped" => Stop::Retry(msg),
        // The request or the credential, which a re-send repeats.
        "payment_required"
        | "context_length_exceeded"
        | "authentication"
        | "permission_denied"
        | "invalid_request"
        | "invalid_prompt"
        | "not_found"
        | "precondition_failed"
        | "payload_too_large"
        | "unprocessable"
        | "string_too_long"
        | "token_limit_exceeded" => Stop::Fatal(msg),
        "content_policy_violation" | "refusal" => Stop::Refused,
        // Cut off at the cap: a larger one helps, the same one fails the same way. On a 400
        // it is the requested cap itself.
        "max_tokens_exceeded" if r.status == 200 => Stop::Truncated,
        // A provider that failed after the 200 went out: transient.
        _ if r.status == 200 => Stop::Retry(msg),
        _ => by_status(r, &msg),
    }
}

// ── Anthropic — POST /v1/messages ────────────────────────────────────

fn ant_effort(e: Effort) -> Value {
    json!({"thinking": {"type": "adaptive"}, "output_config": {"effort": ant_level(e)}})
}

/// The level `ant_effort` sends and `ant_rejects` judges. `none` is `low`, not
/// `thinking: disabled`, which Opus 5.5 rejects.
fn ant_level(e: Effort) -> &'static str {
    match e {
        Effort::Off => Effort::Low.as_str(),
        e => effort_on_wire(e),
    }
}

fn ant_build(c: &Conv, p: &Params) -> Vec<u8> {
    let mark = c.mark(p.cache);
    let model = model_on_wire(&ANTHROPIC, &p.model);
    let mut b = open(
        &ANTHROPIC,
        p,
        build_json_body(&[
            ("model", Some(model.into())),
            ("max_tokens", Some(p.max_tokens.into())),
            ("system", Some(json!([text_block(&p.system, false)]))),
        ]),
        br#","messages":["#,
    );
    let mut first = true;
    let mut i = 0;
    while i < c.items.len() {
        match &c.items[i] {
            Item::Seed(s) | Item::Nudge(s) => {
                let v = json!({"role": "user", "content": [text_block(s, mark == Some(i))]});
                push_val(&mut b, &mut first, &v);
                i += 1;
            }
            Item::Assistant(raw) => {
                // The content array is verbatim; the one-key wrapper is ours.
                elem(&mut b, &mut first, br#"{"role":"assistant","content":"#);
                b.extend_from_slice(raw.get().as_bytes());
                b.push(b'}');
                i += 1;
            }
            // One assistant turn's results must share the next user message.
            Item::Result { .. } => {
                let mut blocks = Vec::new();
                while let Some(Item::Result { id, body, is_error }) = c.items.get(i) {
                    let mut blk = json!({
                        "type": "tool_result", "tool_use_id": id,
                        "content": body, "is_error": is_error,
                    });
                    if mark == Some(i) {
                        blk["cache_control"] = json!({"type": "ephemeral"});
                    }
                    blocks.push(blk);
                    i += 1;
                }
                push_val(
                    &mut b,
                    &mut first,
                    &json!({"role": "user", "content": blocks}),
                );
            }
        }
    }
    b.extend_from_slice(b"]}");
    b
}

#[derive(Deserialize)]
struct AntEcho {
    content: Box<RawValue>,
}

fn ant_parse(r: &Resp) -> Turn {
    let Some(v) = json_body(r) else {
        return unreadable(r);
    };
    let billed = ant_usage(&v["usage"]);
    let fail = |s| Turn::stopped(s).at(r.status).billed(billed);
    if r.status != 200 || v["type"] == "error" {
        let e = &v["error"];
        let msg = err_msg(e, e["type"].as_str().unwrap_or_default(), "unknown");
        return fail(match e["type"].as_str() {
            Some("overloaded_error" | "api_error" | "rate_limit_error") => Stop::Retry(msg),
            // As on the other wires: an unlisted error on a 200 is retried.
            _ if r.status == 200 => Stop::Retry(msg),
            _ => by_status(r, &msg),
        });
    }

    let mut text = String::new();
    let mut calls = Vec::new();
    for blk in v["content"].as_array().into_iter().flatten() {
        match blk["type"].as_str() {
            Some("text") => text.push_str(blk["text"].as_str().unwrap_or_default()),
            Some("tool_use") => calls.push(Call {
                id: blk["id"].as_str().unwrap_or_default().to_string(),
                name: blk["name"].as_str().unwrap_or_default().to_string(),
                args: blk["input"].to_string(),
            }),
            _ => {}
        }
    }
    if !ids_are_usable(&calls) {
        // Billed and unusable: the words still go to the draft.
        return fail(Stop::Retry(UNANSWERABLE.into())).salvaged(Some(text), false);
    }

    let (quiet, callless) = (text.is_empty() && calls.is_empty(), calls.is_empty());
    Turn {
        text,
        calls,
        // Thinking blocks are never read into `text`.
        reasoning_only: false,
        stop: match v["stop_reason"].as_str().filter(|s| !s.is_empty()) {
            Some("max_tokens") => Stop::Truncated,
            Some("refusal") => Stop::Refused,
            Some("tool_use") if callless => Stop::Retry(TOOL_STOP_NO_CALL.into()),
            // The input filled the window: truncated, and no output cap can fix it.
            Some("model_context_window_exceeded") => Stop::OverContext,
            // Server tools only; the one tool here is a client tool.
            Some(s @ "pause_turn") => Stop::Fatal(format!("turn paused ({s})")),
            // An absent signal retries only an empty turn.
            None if quiet => Stop::Retry("response carried no stop_reason".into()),
            _ => Stop::Done,
        },
        usage: billed,
        raw: shaped(
            serde_json::from_slice::<AntEcho>(&r.body)
                .ok()
                .map(|e| e.content),
            b'[',
        ),
        chain: None,
        status: Some(r.status),
        served_model: v["model"].as_str().map(str::to_string),
        provider: Some("Anthropic".into()),
    }
}

// ── OpenAI — POST /v1/responses ──────────────────────────────────────

fn oai_build(c: &Conv, p: &Params) -> Vec<u8> {
    // Stateless replays everything and reads neither `sent` nor `chain`. Under `zdr` nothing is
    // stored, so there is nothing to chain from.
    let stateless = p.stateless || p.zdr;
    let (from, chain) = if stateless {
        (0, None)
    } else {
        (c.sent, c.chain.as_deref())
    };
    let mut b = open(
        &OPENAI,
        p,
        build_json_body(&[
            ("model", Some(model_on_wire(&OPENAI, &p.model).into())),
            ("instructions", Some(p.system.as_str().into())),
            ("max_output_tokens", Some(p.max_tokens.into())),
            ("store", Some((!stateless).into())),
            // Always: a reasoning item captured under `store: true` replays after a lost chain
            // only with its `encrypted_content`.
            ("include", Some(json!(["reasoning.encrypted_content"]))),
            // Absent, never null.
            ("previous_response_id", chain.map(Into::into)),
        ]),
        br#","input":["#,
    );
    let mut first = true;
    for item in &c.items[from.min(c.items.len())..] {
        match item {
            Item::Seed(s) | Item::Nudge(s) => push_val(
                &mut b,
                &mut first,
                &json!({"role": "user", "content": [{"type": "input_text", "text": s}]}),
            ),
            // The stored blob is the whole `output` array, so it splices in without a wrapper.
            Item::Assistant(raw) => {
                let inner = raw.get().trim();
                let inner = inner
                    .strip_prefix('[')
                    .and_then(|s| s.strip_suffix(']'))
                    .unwrap_or(inner);
                if !inner.trim().is_empty() {
                    elem(&mut b, &mut first, inner.as_bytes());
                }
            }
            Item::Result { id, body, .. } => push_val(
                &mut b,
                &mut first,
                &json!({"type": "function_call_output", "call_id": id, "output": body}),
            ),
        }
    }
    b.extend_from_slice(b"]}");
    b
}

#[derive(Deserialize)]
struct OaiEcho {
    output: Box<RawValue>,
}

fn oai_parse(r: &Resp) -> Turn {
    let Some(v) = json_body(r) else {
        return unreadable(r);
    };
    let billed = oai_usage(&v["usage"]);
    let fail = |s| Turn::stopped(s).at(r.status).billed(billed);
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut refused = false;
    for it in v["output"].as_array().into_iter().flatten() {
        match it["type"].as_str() {
            // A progress note, not an answer: dropped, so it cannot win the draft comparison.
            Some("message") if it["phase"].as_str() == Some("commentary") => {}
            Some("message") => {
                for blk in it["content"].as_array().into_iter().flatten() {
                    // A `refusal` part wears `status: "completed"`, so it is read first.
                    match (blk["refusal"].as_str(), blk["text"].as_str()) {
                        (Some(rf), _) => {
                            refused = true;
                            text.push_str(rf);
                        }
                        (None, Some(t)) => text.push_str(t),
                        _ => {}
                    }
                }
            }
            Some("function_call") => calls.push(Call {
                id: it["call_id"].as_str().unwrap_or_default().to_string(),
                name: it["name"].as_str().unwrap_or_default().to_string(),
                args: arg_string(&it["arguments"]),
            }),
            _ => {}
        }
    }
    // After the output is read, so text billed beside an error still reaches the draft.
    if r.status != 200 || v["error"].is_object() {
        let e = &v["error"];
        let code = e["code"].as_str().unwrap_or_default();
        let msg = err_msg(e, code, "unknown");
        return fail(match code {
            // A lost chain is recovered by a stateless replay.
            "previous_response_not_found" | "server_error" | "rate_limit_exceeded" => {
                Stop::Retry(msg)
            }
            // Billing needs a human, though it arrives as a 429.
            "insufficient_quota"
            | "credit_balance_exhausted"
            | "organization_spend_limit_exceeded"
            | "organization_usage_limit_exceeded"
            | "project_spend_limit_exceeded"
            // A failed response for the prompt or the account: a re-send repeats it.
            | "bio_policy"
            | "invalid_prompt"
            | "data_residency_mismatch" => Stop::Fatal(msg),
            // The billing type holds whatever the code says.
            _ if e["type"] == "insufficient_quota" => Stop::Fatal(msg),
            _ if r.status == 200 => Stop::Retry(msg),
            _ => by_status(r, &msg),
        })
        .salvaged(Some(text), false);
    }
    if !ids_are_usable(&calls) {
        // Billed and unusable: the words still go to the draft.
        return fail(Stop::Retry(UNANSWERABLE.into())).salvaged(Some(text), false);
    }

    // Never defaulted to `completed`, which would publish and chain from a malformed 200.
    let status = v["status"].as_str().unwrap_or_default();
    let incomplete = status == "incomplete";
    let reason = v["incomplete_details"]["reason"]
        .as_str()
        .unwrap_or_default();
    let filtered = reason == "content_filter" || refused;
    let quiet = text.is_empty() && calls.is_empty();
    Turn {
        text,
        calls,
        // Reasoning items are never read into `text`.
        reasoning_only: false,
        stop: match status {
            _ if filtered => Stop::Refused,
            // Only this reason is helped by a larger cap; any other is retried, not re-bought.
            _ if incomplete && reason == "max_output_tokens" => Stop::Truncated,
            _ if incomplete => Stop::Retry(format!("incomplete response ({reason:?})")),
            "completed" => Stop::Done,
            "failed" | "cancelled" => Stop::Fatal(format!("response {status}")),
            // An absent signal retries only an empty turn.
            "" if quiet => Stop::Retry("response carried no status".into()),
            "" => Stop::Done,
            // `queued`, `in_progress` and the rest. Quoted, so a status cannot pose as the
            // `(code)` tail `lost_chain` reads.
            _ => Stop::Retry(format!("response still {status:?}")),
        },
        usage: billed,
        raw: shaped(
            serde_json::from_slice::<OaiEcho>(&r.body)
                .ok()
                .map(|e| e.output),
            b'[',
        ),
        // Every status carries an `id`; only a completed one may be continued.
        chain: (status == "completed")
            .then(|| v["id"].as_str().map(str::to_string))
            .flatten(),
        status: Some(r.status),
        served_model: v["model"].as_str().map(str::to_string),
        provider: Some("OpenAI".into()),
    }
}

// ── Usage ────────────────────────────────────────────────────────────
//
// Read on every path with a body, errors and retries included: those attempts are billed too.

fn or_usage(u: &Value) -> Usage {
    let (prompt, completion) = (num(&u["prompt_tokens"]), num(&u["completion_tokens"]));
    let cost = money(&u["cost"]);
    let is_byok = u["is_byok"].as_bool();
    let upstream = money(&u["cost_details"]["upstream_inference_cost"]);
    Usage {
        prompt,
        completion,
        // Tokens with no price, or a BYOK turn missing its upstream half: the sum is a subtotal,
        // even once a later round reports both halves.
        cost_incomplete: (cost.is_none() || (is_byok == Some(true) && upstream.is_none()))
            && (prompt > 0 || completion > 0),
        cost_clamped: false,
        reasoning: opt_num(&u["completion_tokens_details"]["reasoning_tokens"]),
        cache_read: opt_num(&u["prompt_tokens_details"]["cached_tokens"]),
        cache_write: opt_num(&u["prompt_tokens_details"]["cache_write_tokens"]),
        cost,
        is_byok,
        upstream,
    }
}

fn ant_usage(u: &Value) -> Usage {
    Usage {
        // `input_tokens` is only what follows the cache; the other wires report the total.
        prompt: num(&u["input_tokens"])
            .saturating_add(num(&u["cache_read_input_tokens"]))
            .saturating_add(num(&u["cache_creation_input_tokens"])),
        completion: num(&u["output_tokens"]),
        reasoning: opt_num(&u["output_tokens_details"]["thinking_tokens"]),
        cache_read: opt_num(&u["cache_read_input_tokens"]),
        cache_write: opt_num(&u["cache_creation_input_tokens"]),
        cost: None,
        is_byok: None,
        upstream: None,
        // No price is ever reported here; `run::price` computes one.
        cost_incomplete: false,
        cost_clamped: false,
    }
}

fn oai_usage(u: &Value) -> Usage {
    Usage {
        prompt: num(&u["input_tokens"]),
        completion: num(&u["output_tokens"]),
        reasoning: opt_num(&u["output_tokens_details"]["reasoning_tokens"]),
        cache_read: opt_num(&u["input_tokens_details"]["cached_tokens"]),
        cache_write: opt_num(&u["input_tokens_details"]["cache_write_tokens"]),
        cost: None,
        is_byok: None,
        upstream: None,
        cost_incomplete: false,
        cost_clamped: false,
    }
}

// ── Shared readers ───────────────────────────────────────────────────

const UNANSWERABLE: &str = "a tool call arrived with no usable id";

/// A tool stop whose call was lost: the text beside it is a plan, not a review.
const TOOL_STOP_NO_CALL: &str = "stopped for a tool call that never arrived";

/// `message (code)`, with `unknown` for an absent code.
fn err_msg(e: &Value, code: &str, unknown: &str) -> String {
    format!(
        "{} ({})",
        e["message"].as_str().unwrap_or("provider error"),
        if code.is_empty() { unknown } else { code }
    )
}

/// Every call is answered once, by id, so an empty (or absent) or repeated id poisons the
/// conversation. A set, because a huge body can hold six figures of calls.
fn ids_are_usable(calls: &[Call]) -> bool {
    let mut seen = std::collections::HashSet::with_capacity(calls.len());
    calls
        .iter()
        .all(|c| !c.id.is_empty() && seen.insert(c.id.as_str()))
}

/// Keeps an echo only in the container this wire's builder splices: an object on OpenRouter,
/// an array on the other two. `Conv::advance` judges the rest.
fn shaped(raw: Option<Box<RawValue>>, want: u8) -> Option<Box<RawValue>> {
    raw.filter(|r| r.get().trim_start().as_bytes().first() == Some(&want))
}

/// Through `parse_json_slice`: with `raw_value` on, a plain `from_slice` rewrites an object keyed
/// by serde_json's private token, and the parsed view would disagree with the echoed bytes.
fn json_body(r: &Resp) -> Option<Value> {
    crate::query::parse_json_slice(&r.body).ok()
}

/// A malformed or unclassified body inherits its status: `Fatal` on a 2xx, because a retry would
/// buy the generation twice.
fn by_status(r: &Resp, msg: &str) -> Stop {
    let msg = match (r.oversized(), r.truncated()) {
        (true, _) => "response body exceeded the size limit; lower --max-tokens",
        (_, true) => "response body was cut off mid-read",
        _ => msg,
    };
    let m = format!("HTTP {}: {msg}", r.status);
    if super::types::retryable(r.status) {
        Stop::Retry(m)
    } else {
        Stop::Fatal(m)
    }
}

/// Concatenates the `text` of an array-shaped content field, which several providers send.
fn join_parts(v: &Value) -> Option<String> {
    let joined: String = v
        .as_array()?
        .iter()
        .filter_map(|p| p["text"].as_str())
        .collect();
    (!joined.is_empty()).then_some(joined)
}

fn str_or(v: &Value) -> Option<String> {
    v.as_str().filter(|s| !s.is_empty()).map(str::to_string)
}

/// Some providers send `arguments` already parsed rather than as a JSON string.
fn arg_string(v: &Value) -> String {
    v.as_str().map_or_else(|| v.to_string(), str::to_string)
}

/// A count, also from an integral float (`212.0`), which some providers send.
fn opt_num(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| {
        // Below 2^53, where `f64` stops counting in ones; `u64::MAX as f64` rounds up past any
        // `u64`, and `-1`, `1.5` or `1e100` are not counts.
        v.as_f64()
            .filter(|f| f.is_finite() && *f >= 0.0 && f.fract() == 0.0 && *f < (1u64 << 53) as f64)
            .map(|f| f as u64)
    })
}

/// A cost that is not finite and non-negative is unknown, never zero, a refund or infinity.
fn money(v: &Value) -> Option<f64> {
    v.as_f64().filter(|c| c.is_finite() && *c >= 0.0)
}

fn num(v: &Value) -> u64 {
    opt_num(v).unwrap_or(0)
}
