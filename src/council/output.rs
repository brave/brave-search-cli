//! The manifest, `index.md`, the text form, and the history log's writer and reader.
//!
//! Built with `build_json_body`, not `derive(Serialize)`: ~70 optional fields, and dropping `None`
//! is that function's job.

use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::Path;

use super::types::{Item, Member, Outcome, Reason};
use super::wire;
use crate::query::build_json_body;

pub struct RunInfo {
    pub id: String,
    pub dir: std::path::PathBuf,
    pub task: String,
    pub task_source: String,
    pub prompt_sha: String,
    pub elapsed_ms: u128,
    pub required: usize,
    pub exit: i32,
    /// Every write this run meant to make and could not, whichever thread lost it.
    pub log_write_errors: u32,
    /// Trace events the queue had no room for: lossy by design, so not a write error.
    pub trace_dropped: u32,
    pub brave_usd: Option<f64>,
    /// The run's cap, `None` when uncapped.
    pub max_usd: Option<f64>,
    /// What the run asked for; each member records the mechanism its wire applied.
    pub zdr: bool,
    /// `sources.json`, set only once it is written; until then each member carries its URLs.
    pub sources_file: Option<std::path::PathBuf>,
    /// `index.md`, set only once it is written.
    pub index_file: Option<std::path::PathBuf>,
}

/// Rounds a computed figure to the microdollar. Never the cap: see [`cap_usd`].
pub(super) fn micro(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

/// The cap as typed: `{}` is the shortest decimal that round-trips, so `7` stays `7` and a tiny
/// cap never reads as `0`.
pub(super) fn cap_usd(v: f64) -> String {
    format!("{v}")
}

/// Seconds, to one decimal.
fn secs1(d: std::time::Duration) -> f64 {
    (d.as_millis() as f64 / 100.0).round() / 10.0
}

fn usd(v: f64) -> Value {
    json!(micro(v))
}

/// One schema for a member and the run total.
fn tokens(u: &super::types::Usage) -> Value {
    build_json_body(&[
        ("prompt", Some(u.prompt.into())),
        ("completion", Some(u.completion.into())),
        ("reasoning", u.reasoning.map(Into::into)),
        ("cache_read", u.cache_read.map(Into::into)),
        ("cache_write", u.cache_write.map(Into::into)),
    ])
}

/// The share of input that was a cache hit; absent, not a `NaN` turned `null`, with no input.
fn hit_rate(prompt: u64, cache_read: Option<u64>) -> Option<Value> {
    let read = cache_read?;
    (prompt > 0).then(|| json!(((read as f64 / prompt as f64) * 1000.0).round() / 1000.0))
}

fn member_json(m: &Member, limits: &Value, zdr: bool, split: bool) -> Value {
    let (status, reason, error, action, partial) = match &m.outcome {
        Outcome::Delivered { partial_grounding } => {
            ("delivered", None, None, None, Some(*partial_grounding))
        }
        Outcome::Rejected {
            reason,
            error,
            status,
        } => (
            "rejected",
            Some(reason.name()),
            Some(error.clone()),
            Some(reason.action(*status)),
            None,
        ),
    };
    build_json_body(&[
        ("model", Some(m.model.as_str().into())),
        ("served_model", m.served_model.clone().map(Into::into)),
        ("provider", m.provider.clone().map(Into::into)),
        ("transport", Some(m.transport.into())),
        // By the wire the request went to; `doctor` carries the prose.
        (
            "retention",
            wire::wire_for(m.transport).ok().map(|w| {
                let r = wire::retention_of(w, zdr);
                json!({"requested_zdr": zdr, "mechanism": r.mechanism,
                       "enforced_per_request": r.enforced_per_request})
            }),
        ),
        ("status", Some(status.into())),
        ("reason", reason.map(Into::into)),
        ("error", error.map(Into::into)),
        ("action", action.map(Into::into)),
        ("partial_grounding", partial.map(Into::into)),
        // Only when it is there: every write of it may have failed.
        (
            "file",
            m.path
                .is_file()
                .then(|| super::util::sanitize(&m.path.display().to_string()).into()),
        ),
        ("chars", Some(m.chars().into())),
        ("rounds", Some(m.rounds.into())),
        ("retries", Some(m.retries.into())),
        ("tokens", Some(tokens(&m.usage))),
        (
            "cache",
            hit_rate(m.usage.prompt, m.usage.cache_read).map(|h| json!({"hit_rate": h})),
        ),
        (
            "cost",
            Some(json!({"llm_usd": m.cost_usd.map(usd), "source": m.cost_source})),
        ),
        (
            "brave",
            // `calls` is attempts and `sent` is what reached Brave; only `sent` is billed.
            Some(json!({"calls": m.searches, "sent": m.brave_calls,
                        "grounded": m.grounded, "errors": m.search_errors})),
        ),
        // The URLs only when `sources.json` could not hold them: they would dominate stdout, and
        // they are the only record of what a member was shown.
        ("sources", (!split).then(|| m.sources.clone().into())),
        ("sources_count", Some(m.sources.len().into())),
        ("sources_shared", Some(m.sources_shared.into())),
        ("transcript", m.transcript.clone().map(Into::into)),
        ("limits", Some(limits.clone())),
        ("terminated_by", Some(m.terminated_by.into())),
        ("finish_reason", Some(m.finish.into())),
        ("elapsed_ms", Some((m.elapsed.as_millis() as u64).into())),
    ])
}

/// Every key is named for the flag that sets it. `min_reply_chars` is task-only, the threshold
/// behind `no_answer`; the gate measures the trimmed reply, not `chars`.
fn limits_json(
    m: &Member,
    max_searches: u32,
    max_steps: u32,
    max_seconds: u64,
    min_reply_chars: usize,
) -> Value {
    json!({
        // `limit` is where the member started, `reached` where escalation took it.
        "max_tokens": {"limit": m.max_tokens, "reached": m.max_tokens_reached,
                       "peak_round": m.peak_round, "escalations": m.escalations},
        "max_searches": {"limit": max_searches, "used": m.searches},
        "max_steps": {"limit": max_steps, "used": m.rounds},
        "max_seconds": {"limit": max_seconds, "used": secs1(m.elapsed)},
        "min_reply_chars": {"limit": min_reply_chars},
    })
}

/// Members in requested order, so two runs of one command diff cleanly.
pub fn manifest(
    r: &RunInfo,
    ms: &[Member],
    max_searches: u32,
    max_steps: u32,
    max_seconds: u64,
    min_reply_chars: usize,
) -> Value {
    let delivered = ms.iter().filter(|m| m.outcome.delivered()).count();
    let mut total = super::types::Usage::default();
    for m in ms {
        total.add(&m.usage);
    }
    let known: Vec<f64> = ms.iter().filter_map(|m| m.cost_usd).collect();
    // The sum of the published parts, so they add up; then checked itself, as it can overflow.
    let total_usd = (!known.is_empty())
        .then(|| known.iter().copied().map(micro).sum::<f64>())
        .and_then(super::cfg::publishable_usd);
    // A floor, an unpriced member or an unpriced Brave call leaves the total unknown.
    let any_unknown = ms
        .iter()
        .any(|m| m.cost_usd.is_none() || m.cost_source.ends_with("_partial"))
        || (!known.is_empty() && total_usd.is_none())
        || (r.brave_usd.is_none() && ms.iter().any(|m| m.brave_calls > 0));
    let sources: HashSet<&str> = ms
        .iter()
        .flat_map(|m| m.sources.iter().map(String::as_str))
        .collect();

    build_json_body(&[
        // Bumped when a field changes meaning or leaves; adding one is not a bump.
        ("schema", Some(2.into())),
        ("run", Some(r.id.as_str().into())),
        ("dir", Some(r.dir.display().to_string().into())),
        (
            "state",
            Some(
                match (delivered, delivered >= r.required) {
                    (_, true) => "ok",
                    (0, _) => "failed",
                    _ => "partial",
                }
                .into(),
            ),
        ),
        ("exit", Some(r.exit.into())),
        ("task", Some(r.task.as_str().into())),
        ("task_source", Some(r.task_source.as_str().into())),
        ("prompt_sha", Some(r.prompt_sha.as_str().into())),
        ("elapsed_ms", Some((r.elapsed_ms as u64).into())),
        (
            "quorum",
            Some(json!({"delivered": delivered, "total": ms.len(),
                        "required": r.required, "ok": delivered >= r.required})),
        ),
        ("hint", Some(hint(r, ms, delivered).into())),
        (
            "index",
            r.index_file
                .as_ref()
                .map(|p| p.display().to_string().into()),
        ),
        (
            "sources_file",
            r.sources_file
                .as_ref()
                .map(|p| p.display().to_string().into()),
        ),
        ("log_write_errors", Some(r.log_write_errors.into())),
        ("trace_dropped", Some(r.trace_dropped.into())),
        (
            "cost",
            Some(json!({
                "llm_usd": total_usd.map(usd),
                "brave_usd": r.brave_usd.map(usd),
                "cost_unknown": any_unknown,
                // Raw: the number the operator typed.
                "max_usd": r.max_usd,
            })),
        ),
        ("cost_source", Some(cost_source(ms).into())),
        ("tokens", Some(tokens(&total))),
        (
            "cache",
            hit_rate(total.prompt, total.cache_read).map(|h| json!({"hit_rate": h})),
        ),
        (
            "rounds",
            Some(ms.iter().map(|m| u64::from(m.rounds)).sum::<u64>().into()),
        ),
        (
            "brave",
            Some(json!({
                "calls": ms.iter().map(|m| m.searches).sum::<u32>(),
                "sent": ms.iter().map(|m| m.brave_calls).sum::<u32>(),
                "grounded": ms.iter().map(|m| m.grounded).sum::<u32>(),
                "errors": ms.iter().map(|m| m.search_errors).sum::<u32>(),
                "unique_urls": sources.len(),
            })),
        ),
        (
            "members",
            Some(Value::Array(
                ms.iter()
                    .map(|m| {
                        let limits =
                            limits_json(m, max_searches, max_steps, max_seconds, min_reply_chars);
                        member_json(m, &limits, r.zdr, r.sources_file.is_some())
                    })
                    .collect(),
            )),
        ),
    ])
}

/// One value when every member agrees, `mixed` otherwise.
fn cost_source(ms: &[Member]) -> &'static str {
    match ms.split_first() {
        None => "unknown",
        Some((first, rest)) if rest.iter().all(|m| m.cost_source == first.cost_source) => {
            first.cost_source
        }
        _ => "mixed",
    }
}

/// The field an agent acts on: what was lost, and the `-m` list to retry with, always last.
fn hint(r: &RunInfo, ms: &[Member], delivered: usize) -> String {
    let retryable: Vec<&str> = ms
        .iter()
        .filter_map(|m| match &m.outcome {
            Outcome::Rejected { reason, status, .. } if reason.action(*status) == "retry" => {
                Some(m.model.as_str())
            }
            _ => None,
        })
        .collect();
    let mut s = format!("{delivered} of {} delivered.", ms.len());
    if delivered > 0
        && let Some(index) = &r.index_file
    {
        s.push_str(&format!(" Read {} first.", index.display()));
    }
    // Named even when the quorum was met. The cap is not retryable (the same command stops in
    // the same place), so naming it is the advice.
    let capped: Vec<&str> = ms
        .iter()
        .filter_map(|m| match &m.outcome {
            Outcome::Rejected {
                reason: Reason::Cost,
                ..
            } => Some(m.model.as_str()),
            _ => None,
        })
        .collect();
    if !capped.is_empty() {
        s.push_str(" Stopped at --max-usd");
        if let Some(cap) = r.max_usd {
            s.push_str(&format!(" {}", cap_usd(cap)));
        }
        s.push_str(&format!(
            ": {}. Raise the cap and re-run {}.",
            capped.join(","),
            if capped.len() == 1 { "it" } else { "them" }
        ));
    }
    if !retryable.is_empty() {
        // An unchanged retry buys the same timeout again.
        if ms.iter().any(
            |m| matches!(&m.outcome, Outcome::Rejected { reason, .. } if *reason == Reason::Budget),
        ) {
            s.push_str(" Raise --max-seconds for any the clock stopped.");
        }
        s.push_str(&format!(" Retry the rest with: -m {}", retryable.join(",")));
    }
    // Only when the whole panel agrees: a mixed run has no single lever.
    if delivered == 0
        && retryable.is_empty()
        && capped.is_empty()
        && ms.iter().all(
            |m| matches!(&m.outcome, Outcome::Rejected { reason, .. } if *reason == Reason::Empty),
        )
    {
        s.push_str(" Every member returned no text; retrying unchanged cannot help.");
    }
    s
}

/// A per-member index, so a reader can choose which replies to open.
pub fn index_md(r: &RunInfo, ms: &[Member]) -> String {
    let mut s = format!("# council {}\n\ntask: {}\n\n", r.id, r.task);
    for m in ms {
        let file = match m.path.file_name() {
            Some(f) if m.path.is_file() => {
                let f = f.to_string_lossy();
                format!("[{f}]({f})")
            }
            _ => "no file".into(),
        };
        match &m.outcome {
            Outcome::Delivered { partial_grounding } => s.push_str(&format!(
                "- **{}** — {} · {} chars · {}/{} grounded{}\n",
                m.model,
                file,
                m.chars(),
                m.grounded,
                m.searches,
                if *partial_grounding {
                    " · ⚠ partial grounding"
                } else {
                    ""
                }
            )),
            // Linked and sized too: a rejected member's paid reply is written as well. The error
            // is provider text, so it is fenced and flattened.
            Outcome::Rejected {
                reason,
                error,
                status,
            } => s.push_str(&format!(
                "- ~~{}~~ — **{}**: `{}` (`{}`) · {} · {} chars\n",
                m.model,
                reason.name(),
                error.replace(['\n', '\r'], " ").replace('`', "'"),
                reason.action(*status),
                file,
                m.chars()
            )),
        }
    }
    s
}

pub fn text(r: &RunInfo, ms: &[Member]) -> String {
    let delivered = ms.iter().filter(|m| m.outcome.delivered()).count();
    let mut s = format!(
        "council: {delivered}/{} delivered · {}s → {}\n",
        ms.len(),
        r.elapsed_ms / 1000,
        r.dir.display()
    );
    for m in ms {
        s.push_str(&match &m.outcome {
            Outcome::Delivered { .. } => format!("  ok      {} — {} chars\n", m.model, m.chars()),
            Outcome::Rejected { reason, .. } => {
                format!("  reject  {} — {}\n", m.model, reason.name())
            }
        });
    }
    s
}

/// One `history.jsonl` line per member, appended as each completes.
pub fn history_line(r: &RunInfo, m: &Member) -> Value {
    build_json_body(&[
        // Bumped like the manifest's. A row without it is version 1.
        ("schema", Some(2.into())),
        ("ts", Some(super::util::iso8601_now().into())),
        ("run", Some(r.id.as_str().into())),
        // `--out` can put a run anywhere, so the id alone cannot find it.
        ("dir", Some(r.dir.display().to_string().into())),
        ("task", Some(r.task.as_str().into())),
        ("model", Some(m.model.as_str().into())),
        ("served_model", m.served_model.clone().map(Into::into)),
        ("provider", m.provider.clone().map(Into::into)),
        ("task_source", Some(r.task_source.as_str().into())),
        ("prompt_sha", Some(r.prompt_sha.as_str().into())),
        ("transport", Some(m.transport.into())),
        (
            "status",
            Some(
                if m.outcome.delivered() {
                    "delivered"
                } else {
                    "rejected"
                }
                .into(),
            ),
        ),
        (
            "reason",
            match &m.outcome {
                Outcome::Rejected { reason, .. } => Some(reason.name().into()),
                Outcome::Delivered { .. } => None,
            },
        ),
        ("chars", Some(m.chars().into())),
        ("rounds", Some(m.rounds.into())),
        ("searches", Some(m.searches.into())),
        ("grounded", Some(m.grounded.into())),
        ("cost_usd", m.cost_usd.map(usd)),
        ("cost_source", Some(m.cost_source.into())),
        ("elapsed_s", Some(json!(secs1(m.elapsed)))),
        ("terminated_by", Some(m.terminated_by.into())),
        ("retries", Some(m.retries.into())),
        ("peak_round", Some(m.peak_round.into())),
        ("escalations", Some(m.escalations.into())),
        (
            "cache_hit_rate",
            hit_rate(m.usage.prompt, m.usage.cache_read),
        ),
    ])
}

// ── Reading it back ──────────────────────────────────────────────────
//
// Beside `history_line`, so a renamed field breaks the reader where it is renamed.

pub const HISTORY: &str = "history.jsonl";

/// Verdicts live beside the log, never in it: a row with no `model` would read as a member.
pub const SCORES: &str = "scores.jsonl";

/// The most of either log a reader takes, from the end: `prune` never trims them.
pub const MAX_HISTORY: u64 = 2 * 1024 * 1024;

/// One line, borrowed from its buffer. A borrowed field needing a JSON escape skips its line;
/// `panel_slugs` refuses such a slug anyway.
#[derive(serde::Deserialize)]
pub struct Row<'a> {
    pub run: &'a str,
    pub model: &'a str,
    /// Owned: a frontmatter `name:` may need an escape. Empty on a row that predates the field.
    #[serde(default)]
    pub task: String,
    /// Absent when no price was published.
    #[serde(default)]
    pub cost_usd: Option<f64>,
    #[serde(default)]
    pub cost_source: &'a str,
}

/// Every line that parses; the rest are skipped. `schema` is not gated on: absent is version 1.
pub fn rows(buf: &str) -> Vec<Row<'_>> {
    buf.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// One verdict, joined on `dir`: two `--out` runs can share a basename, or a run id. Owned,
/// because a Windows path is escaped and cannot be borrowed.
#[derive(serde::Deserialize)]
struct Score {
    dir: String,
    useful: bool,
}

/// Verdicts by run directory; the later of two wins. Folded, not collected: only `insert`
/// documents which duplicate survives.
pub fn scores(buf: &str) -> std::collections::HashMap<String, bool> {
    buf.lines()
        .filter_map(|l| serde_json::from_str::<Score>(l).ok())
        .fold(std::collections::HashMap::new(), |mut m, s| {
            m.insert(s.dir, s.useful);
            m
        })
}

/// A published, complete cost. Floors and `unbilled` zeros would bias a cap's sizing down.
fn sample(r: &Row) -> Option<f64> {
    r.cost_usd
        .filter(|_| !r.cost_source.ends_with("_partial") && r.cost_source != "unbilled")
        .and_then(super::cfg::publishable_usd)
}

/// What a set of past runs cost. `n` counts priced samples, not rows read.
pub struct Stats {
    pub n: usize,
    pub p50: f64,
    pub p90: f64,
    pub max: f64,
}

/// Nearest rank, so every figure is a price some run paid: rank `ceil(p·n)` in integer
/// arithmetic, which a float cast gets one high. `total_cmp`, as a NaN must not panic.
fn stats(v: &mut [f64]) -> Option<Stats> {
    // Below three, p50 is an endpoint.
    let n = v.len();
    if n < 3 {
        return None;
    }
    v.sort_unstable_by(f64::total_cmp);
    let at = |num: usize, den: usize| v[(num * n).div_ceil(den).max(1) - 1];
    Some(Stats {
        n,
        p50: at(1, 2),
        p90: at(9, 10),
        max: v[n - 1],
    })
}

/// A slug without its wire prefix, so one model has one history key. The one prefix rule;
/// `Route::bare` is derived from it.
pub(super) fn bare_slug(slug: &str) -> &str {
    slug.strip_prefix("openrouter:")
        .or_else(|| slug.strip_prefix("direct:"))
        .unwrap_or(slug)
}

pub fn per_model(rows: &[Row], slug: &str) -> Option<Stats> {
    let want = bare_slug(slug);
    let mut v: Vec<f64> = rows
        .iter()
        .filter(|r| bare_slug(r.model) == want)
        .filter_map(sample)
        .collect();
    stats(&mut v)
}

/// What a run of this panel cost, over runs that fielded at least it: only the planned members
/// are summed, and a run missing any of their prices is dropped, not summed short.
pub fn per_run(rows: &[Row], planned: &[&str]) -> Option<Stats> {
    let mut by_run: Vec<&Row> = rows.iter().collect();
    // Sorted: two overlapping runs interleave their lines.
    by_run.sort_unstable_by_key(|r| r.run);
    let mut v: Vec<f64> = Vec::new();
    let mut taken: Vec<bool> = Vec::new();
    for g in by_run.chunk_by(|a, b| a.run == b.run) {
        // Each planned member claims a distinct row: two seats may name one model on two wires.
        taken.clear();
        taken.resize(g.len(), false);
        let total: Option<f64> = planned
            .iter()
            .map(|m| {
                let i =
                    (0..g.len()).find(|&i| !taken[i] && bare_slug(g[i].model) == bare_slug(m))?;
                taken[i] = true;
                sample(g[i])
            })
            .sum();
        v.extend(total);
    }
    stats(&mut v)
}

/// `sources.json`: every member's retrieved URLs, kept off stdout. `derive(Serialize)` here, to
/// borrow the URLs rather than copy them into a `Value`.
#[derive(serde::Serialize)]
struct SourcesDoc<'a> {
    schema: u8,
    run: &'a str,
    members: Vec<MemberSources<'a>>,
}

#[derive(serde::Serialize)]
struct MemberSources<'a> {
    model: &'a str,
    sources: &'a [String],
}

/// Written for every run, even one that retrieved nothing, since the manifest names it.
pub fn sources_json(r: &RunInfo, ms: &[Member]) -> Option<String> {
    let doc = SourcesDoc {
        schema: 1,
        run: &r.id,
        members: ms
            .iter()
            .map(|m| MemberSources {
                model: &m.model,
                sources: &m.sources,
            })
            .collect(),
    };
    // Infallible for this shape; `None` reaches the caller's fallback, not a panic or an empty
    // file.
    serde_json::to_string(&doc).ok()
}

/// Each member's URLs that another member also retrieved: duplicated retrieval, not agreement.
pub fn mark_shared(ms: &mut [Member]) {
    let sets: Vec<HashSet<&str>> = ms
        .iter()
        .map(|m| m.sources.iter().map(String::as_str).collect())
        .collect();
    let shared: Vec<usize> = (0..ms.len())
        .map(|i| {
            sets[i]
                .iter()
                .filter(|u| {
                    sets.iter()
                        .enumerate()
                        .any(|(j, s)| j != i && s.contains(*u))
                })
                .count()
        })
        .collect();
    for (m, n) in ms.iter_mut().zip(shared) {
        m.sources_shared = n;
    }
}

/// One member's conversation, in order. Assistant turns go in verbatim but for what `sanitize`
/// strips, so a transcript diffs against the bytes sent.
pub fn transcript(model: &str, items: &[Item], secrets: &[String]) -> String {
    // A string, not a `Value`: `json!` would re-parse the `RawValue` and lose its bytes.
    let mut out = format!("{{\"model\":{},\"turns\":[", json!(model));
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        match item {
            Item::Seed(t) => out.push_str(&json!({"role": "seed", "text": t}).to_string()),
            Item::Nudge(t) => out.push_str(&json!({"role": "nudge", "text": t}).to_string()),
            Item::Assistant(raw) => {
                out.push_str(r#"{"role":"assistant","content":"#);
                out.push_str(raw.get());
                out.push('}');
            }
            Item::Result { id, body, is_error } => out.push_str(
                &json!({"role": "tool_result", "id": id, "is_error": is_error, "body": body})
                    .to_string(),
            ),
        }
    }
    out.push_str("]}");
    // Sanitized first, so a control planted inside an echoed key cannot shield it. `serde_json`
    // escapes only C0, `"` and `\`; this strips raw DEL, C1 and the formats, in every turn.
    super::util::redact(&super::util::sanitize(&out), secrets)
}

/// A rejected member's file is written too: the reply was paid for.
pub fn write_member(path: &Path, m: &Member) -> Result<(), String> {
    // Provider text inside an HTML comment: `--` could close it, and a newline could break it.
    let one_line = |s: &str| s.replace(['\n', '\r'], " ").replace("--", "‑‑");
    let header = match &m.outcome {
        Outcome::Delivered { .. } => format!("<!-- {} -->\n\n", one_line(&m.model)),
        Outcome::Rejected { reason, error, .. } => format!(
            "<!-- {} — REJECTED: {} ({}) -->\n\n",
            one_line(&m.model),
            reason.name(),
            one_line(error)
        ),
    };
    write_private(path, &format!("{header}{}\n", m.text))
}

/// Writes `0600`, through a unique temp and a rename: a planted symlink or a loose mode is
/// replaced, never followed, and no reader sees half a document.
pub fn write_private(path: &Path, body: &str) -> Result<(), String> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let tmp = path.with_extension(format!("tmp{}-{stamp}", std::process::id()));

    let mut o = std::fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    // `?` first: the cleanup below may delete only a temp this call created.
    let mut f = o
        .open(&tmp)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    // Synced before the rename, which can otherwise land before the data.
    if let Err(e) = std::io::Write::write_all(&mut f, body.as_bytes()).and_then(|()| f.sync_all()) {
        std::fs::remove_file(&tmp).ok();
        return Err(format!("{}: {e}", path.display()));
    }
    // A failed rename keeps the temp: it is a whole, synced copy, maybe of a paid reply.
    std::fs::rename(&tmp, path)
        .map_err(|e| format!("{}: {e}; the text is in {}", path.display(), tmp.display()))
}
