//! The one tool: `search`, backed by `POST /res/v1/llm/context`.
//!
//! Nothing here returns an error into the loop: every failure renders a marker, and grounding is
//! decided by the `Found` variant, never by sniffing the string (a real title can start with `(`).

use serde_json::{Value, json};
use std::hash::BuildHasher;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::query::build_json_body;

use super::util::{sanitize, truncate};
use crate::api::Resp;

/// Brave's limits on `q`: 1–400 characters, at most 50 words. Over them, every retry 422s too.
const MAX_QUERY_CHARS: usize = 400;
const MAX_QUERY_WORDS: usize = 50;
const RESULT_COUNT: u32 = 10;
/// One tool result, in characters of up to 4 bytes each. It is re-sent every remaining round, so
/// it is a bound, not a target.
pub(super) const MAX_RESULT_CHARS: usize = 64 * 1024;
/// Sources reach the manifest on stdout, the calling agent's context, so they are bounded.
pub(super) const MAX_SOURCES: usize = 64;
/// Generous for any citation; a longer URL or title is not one.
pub(super) const MAX_URL_CHARS: usize = 2_048;
const MAX_TITLE_CHARS: usize = 512;
/// Retries per query: waiting is cheap next to answering ungrounded.
const MAX_SEARCH_RETRIES: u32 = 2;
/// The least time a retry must leave for its search; the search itself takes what is left.
const SEARCH_GRACE: Duration = Duration::from_secs(5);

/// Snippets per result. `cap_snippets` bounds their content; this bounds the array around it.
const MAX_SNIPPETS: usize = 128;

pub struct Tool<'a> {
    pub agent: &'a ureq::Agent,
    pub base: &'a str,
    pub key: &'a str,
    /// `--timeout`, the per-request bound.
    pub timeout: Duration,
    /// The Brave status that made search unavailable, shared by the whole run; 0 while search
    /// works. A refused key or a spent quota answers every later search the same way, so none is
    /// sent after it.
    pub down: &'a std::sync::atomic::AtomicU16,
}

pub enum Found {
    /// Real retrieved content, with the URLs it came from.
    Text(String, Vec<String>),
    /// A correct answer that grounds nothing: nothing on the web for this term.
    Empty,
    /// A fault — transport or API. The only case that counts against the member.
    Failed(String),
    /// Brave refused the key or the quota is spent; no search can succeed for the rest of the run.
    Unavailable(u16),
}

impl Found {
    /// Consumes the result: its one consumer moves it into the conversation.
    pub fn render(self) -> String {
        match self {
            Found::Text(body, _) => body,
            Found::Empty => "(no results)".into(),
            Found::Failed(m) => m,
            Found::Unavailable(s) => format!(
                "(search unavailable: HTTP {s} — no search can run; answer now from what you have)"
            ),
        }
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Found::Failed(_) | Found::Unavailable(_))
    }
}

/// Reads the query from whichever of the three key spellings the model used. Other keys are
/// ignored, since none changes the query; with no recognised key the call is refused rather
/// than guessed at.
pub fn extract_query(args: &str) -> Result<String, String> {
    // The de-fanged parser: under `raw_value`, serde_json's private token key would substitute
    // keys the model never wrote.
    let v = crate::query::parse_json(args.trim())
        .map_err(|_| "(malformed tool arguments — send valid JSON)".to_string())?;
    // Arguments encoded twice arrive as a string holding the object.
    let v = match v {
        Value::String(s) => crate::query::parse_json(&s).unwrap_or(Value::String(s)),
        v => v,
    };
    let obj = v
        .as_object()
        .ok_or("(tool arguments must be a JSON object)")?;
    ["query", "q", "search_query"]
        .iter()
        .find_map(|k| Some(obj.get(*k)?.as_str()?.trim()).filter(|q| !q.is_empty()))
        .map(clamp)
        .ok_or_else(|| r#"(no query — send {"query": "…"})"#.into())
}

fn clamp(q: &str) -> String {
    let words: Vec<&str> = q.split_whitespace().take(MAX_QUERY_WORDS).collect();
    words.join(" ").chars().take(MAX_QUERY_CHARS).collect()
}

impl Tool<'_> {
    /// Bounded by `--timeout` and by the member's remaining budget, whichever is nearer.
    ///
    /// Returns the marker and whether the call was billable: a 200, as Brave bills only
    /// successful responses. A timeout on a query Brave did serve goes uncounted, at most once
    /// per search.
    pub fn search(&self, query: &str, budget_left: Duration) -> (Found, bool) {
        let down = self.down.load(Ordering::Relaxed);
        if down != 0 {
            return (Found::Unavailable(down), false);
        }
        let body = json!({"q": query, "count": RESULT_COUNT}).to_string();
        let headers = [("X-Subscription-Token", self.key.to_string())];
        let url = format!("{}/res/v1/llm/context", self.base);
        let started = std::time::Instant::now();
        let mut attempt = 0;
        loop {
            let left = budget_left.saturating_sub(started.elapsed());
            if left.is_zero() {
                return (
                    Found::Failed("(search error: budget exhausted)".into()),
                    false,
                );
            }
            let sent = crate::api::post_raw(
                self.agent,
                &url,
                &headers,
                body.as_bytes(),
                super::MAX_BODY,
                self.timeout.min(left),
            );
            // Retried: a rate limit, a server fault, or a dial that never reached Brave. None
            // is billed, and answering ungrounded is what the retry exists to prevent.
            let (marker, asked) = match sent {
                Ok(r) if r.status == 200 && r.oversized() => {
                    return (
                        Found::Failed("(search error: response too large)".into()),
                        true,
                    );
                }
                Ok(r) if r.status == 200 => return (render(&r), true),
                Ok(r) if latches(&r) => {
                    self.down.store(r.status, Ordering::Relaxed);
                    return (Found::Unavailable(r.status), false);
                }
                Ok(r) if r.status == 429 || r.status >= 500 => {
                    (format!("(search error HTTP {})", r.status), r.retry_after())
                }
                Ok(r) => {
                    return (
                        Found::Failed(format!("(search error HTTP {})", r.status)),
                        false,
                    );
                }
                Err(e) => {
                    let marker = format!("(search error: {})", first_line(&e.to_string()));
                    if !crate::api::is_retryable(&e) {
                        return (Found::Failed(marker), false);
                    }
                    (marker, None)
                }
            };
            let wait = retry_wait(asked);
            // Against the clock now: the request may have spent most of it.
            if attempt == MAX_SEARCH_RETRIES
                || !worth_waiting(wait, budget_left.saturating_sub(started.elapsed()))
            {
                return (Found::Failed(marker), false);
            }
            std::thread::sleep(wait);
            attempt += 1;
        }
    }
}

/// A refused key, an invalid token or a spent plan: no wait clears them, so search stays off.
/// An over-long query also 422s, so the code decides, not the status.
fn latches(r: &Resp) -> bool {
    let code = |want: &str| {
        crate::query::parse_json_slice(&r.body).is_ok_and(|v| v["error"]["code"] == want)
    };
    match r.status {
        401..=403 => true,
        422 => code("SUBSCRIPTION_TOKEN_INVALID"),
        429 => code("QUOTA_LIMITED"),
        _ => false,
    }
}

/// What was asked, or a second, plus up to a second of jitter: Brave sends no `Retry-After`,
/// so members that collided on a one-per-second key would otherwise collide again.
pub(super) fn retry_wait(asked: Option<Duration>) -> Duration {
    let jitter = std::collections::hash_map::RandomState::new().hash_one(()) % 1000;
    asked.unwrap_or(Duration::from_secs(1)) + Duration::from_millis(jitter)
}

/// Whether a wait leaves room for the search it is for: at most half the clock, and a search's
/// grace after it.
pub(super) fn worth_waiting(wait: Duration, left: Duration) -> bool {
    wait.saturating_mul(2) <= left && wait.saturating_add(SEARCH_GRACE) < left
}

/// The whole first line: cutting could leave a greppable half of a secret.
fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or_default().to_string()
}

/// The ISO day among `sources[url].age`'s spellings of one instant: absolute, and it does not go
/// stale like "76 days ago". Matched by shape, since the order is the provider's.
fn iso_day(age: &Value) -> Option<&str> {
    age.as_array()?.iter().find_map(|v| {
        let s = v.as_str()?;
        let b = s.as_bytes();
        (s.len() == 10
            && b[4] == b'-'
            && b[7] == b'-'
            && b.iter()
                .enumerate()
                .all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit()))
        .then_some(s)
    })
}

/// Neutralises every string in a value, keys included: local results pass through without a
/// schema, so any leaf can carry a closing tag.
fn clean(v: &mut Value) {
    match v {
        Value::String(s) => *s = neutralise(s),
        Value::Array(a) => a.iter_mut().for_each(clean),
        Value::Object(o) => {
            *o = std::mem::take(o)
                .into_iter()
                .map(|(k, mut v)| {
                    clean(&mut v);
                    (neutralise(&k), v)
                })
                .collect();
        }
        _ => {}
    }
}

/// What one character costs inside a JSON document. `sanitize` leaves no control but these, so
/// every escape is two characters.
fn escaped_cost(c: char) -> usize {
    usize::from(matches!(c, '"' | '\\' | '\n' | '\r' | '\t')) + 1
}

/// What a string costs inside a JSON document.
pub(super) fn encoded_len(s: &str) -> usize {
    s.chars().map(escaped_cost).sum()
}

/// Truncates to an *encoded* budget rather than a character count, so the cut piece fits the
/// document the caller is building.
fn cut_encoded(s: &str, budget: usize) -> String {
    let (mut used, mut end) = (0usize, 0usize);
    for (i, ch) in s.char_indices() {
        let cost = escaped_cost(ch);
        if used + cost > budget {
            break;
        }
        used += cost;
        end = i + ch.len_utf8();
    }
    format!("{}\u{2026} [truncated]", &s[..end])
}

/// Truncates snippets to a total encoded budget, cutting at most one. Bounding the content, not
/// the serialized document, keeps the result parseable.
fn cap_snippets(snippets: Vec<String>, budget: usize) -> Vec<String> {
    let mut left = budget;
    let mut out = Vec::new();
    for s in snippets.into_iter().take(MAX_SNIPPETS) {
        if left == 0 {
            break;
        }
        let n = encoded_len(&s);
        out.push(if n <= left { s } else { cut_encoded(&s, left) });
        left = left.saturating_sub(n);
    }
    out
}

/// The provider's own shape, as JSON, with each page's date and the local results. Wrapped in
/// `<search_results>` and neutralised: `serde_json` does not escape `/`, so a closing tag in a
/// snippet would survive encoding.
pub(super) fn render(r: &Resp) -> Found {
    let Ok(v) = crate::query::parse_json_slice(&r.body) else {
        return Found::Failed("(search error: unreadable response)".into());
    };
    // No `grounding` is not an empty result set but something else answering: a proxy, a portal.
    if !v["grounding"].is_object() {
        return Found::Failed("(search error: response carried no grounding)".into());
    }
    let sources = &v["sources"];
    let mut results = Vec::new();
    let mut urls = Vec::new();
    let mut used = 0;
    for item in v["grounding"]["generic"].as_array().into_iter().flatten() {
        // Raw, for the `sources` lookup, which is keyed by the URL as sent.
        let raw_url = item["url"].as_str().unwrap_or_default();
        // Dropped, not cut: a truncated URL is a citation nobody can follow. This also bounds
        // `Report::sources`.
        let url = if raw_url.chars().count() <= MAX_URL_CHARS {
            raw_url
        } else {
            ""
        };
        let title = item["title"].as_str().unwrap_or_default();
        let snippets: Vec<String> = item["snippets"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| s.as_str())
            .take(MAX_SNIPPETS)
            .map(neutralise)
            .filter(|s| !s.trim().is_empty())
            .collect();
        // Evidence is text: a snippet, or a titled URL. A bare URL or title gives nothing to read.
        if snippets.is_empty() && (url.is_empty() || title.trim().is_empty()) {
            continue;
        }
        // Bounded here, since the first entry is exempt from the running total. Neutralised
        // first, so a cut can only split an inert `&lt;`.
        let cite = |s: &str, max| truncate(&neutralise(s), max);
        let entry = build_json_body(&[
            (
                "url",
                (!url.is_empty()).then(|| cite(url, MAX_URL_CHARS).into()),
            ),
            (
                "title",
                (!title.is_empty()).then(|| cite(title, MAX_TITLE_CHARS).into()),
            ),
            ("date", iso_day(&sources[raw_url]["age"]).map(Into::into)),
            // Capped per entry, so one page can fill the budget but never overrun it.
            (
                "snippets",
                Some(cap_snippets(snippets, MAX_RESULT_CHARS).into()),
            ),
        ]);
        let len = entry.to_string().chars().count();
        // The first entry always goes in, so one oversized page is not `(no results)`.
        if !results.is_empty() && used + len > MAX_RESULT_CHARS {
            break;
        }
        used += len;
        results.push(entry);
        // Only pages the model received are cited.
        if !url.is_empty() && urls.len() < MAX_SOURCES {
            urls.push(url.to_string());
        }
    }
    // Empty containers, blank strings and bare scalars ground nothing.
    fn blank(v: &Value) -> bool {
        match v {
            Value::Object(o) => o.values().all(blank),
            Value::Array(a) => a.iter().all(blank),
            Value::String(s) => s.trim().is_empty(),
            Value::Null | Value::Bool(_) | Value::Number(_) => true,
        }
    }
    // Local results pass through verbatim, with no schema here, charged against what is left.
    let mut local = |key: &str, shape: fn(&Value) -> bool| {
        let mut x = v["grounding"][key].clone();
        clean(&mut x);
        let n = x.to_string().chars().count();
        (shape(&x) && !blank(&x) && used + n <= MAX_RESULT_CHARS).then(|| {
            used += n;
            x
        })
    };
    let map = local("map", Value::is_array);
    let poi = local("poi", |_| true);
    // After the local results: a place query answers with an empty `generic`.
    if results.is_empty() && map.is_none() && poi.is_none() {
        return Found::Empty;
    }

    let doc = build_json_body(&[
        // So each `date` reads as an age without the model guessing today.
        ("retrieved", Some(super::util::iso8601_now()[..10].into())),
        (
            "results",
            (!results.is_empty()).then_some(Value::Array(results)),
        ),
        ("map", map),
        ("poi", poi),
    ]);
    Found::Text(format!("<search_results>\n{doc}\n</search_results>"), urls)
}

/// Retrieved text is attacker-controlled, and a closing tag in it would forge trusted framing. Any
/// `<` whose tag name reads `search_results`, in any spelling, becomes `&lt;`. Newlines are kept:
/// inside a JSON string they cannot be structure.
fn neutralise(s: &str) -> String {
    let flat = sanitize(s);
    let mut out = String::with_capacity(flat.len());
    let mut rest = flat.as_str();
    while let Some(i) = rest.find('<') {
        out.push_str(&rest[..i]);
        let tail = &rest[i + 1..];
        // Only an ASCII name can match, so everything else is filler a model may read past —
        // invisible padding, lookalike slashes — as are whitespace and `\` (`<\/` is the JSON
        // spelling of this tag).
        let name: String = tail
            .chars()
            .filter(char::is_ascii)
            .skip_while(|c| matches!(c, '/' | '\\') || c.is_whitespace())
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        // Visible, so no normalisation a model applies can undo it.
        out.push_str(if name.eq_ignore_ascii_case("search_results") {
            "&lt;"
        } else {
            "<"
        });
        rest = tail;
    }
    out.push_str(rest);
    out
}
