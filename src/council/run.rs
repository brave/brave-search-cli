//! The run: `plan` decides everything that needs no spend, then `execute` fans out.
//!
//! `plan` refuses before any paid request: exit 2, or 3/8 when the pre-flight rejects a key or
//! credit.

use serde_json::json;
use std::io::Write;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::member::Member as Runner;
use super::output::RunInfo;
use super::resolve::{
    Refusal, Route, check_credit, check_models, check_out, env_of, panel_slugs, rate_of, state_dir,
    subject_of, transport_for, worst_case,
};
use super::types::{Effort, Member, Outcome, Params, Reason};
use super::{CouncilArgs, Event, QUEUE, Trace, cfg, gate, output, search, task, util, wire};

/// One thread and one TLS agent per member; far past any real panel.
pub(super) const MAX_PANEL: usize = 32;
/// The most `--max-usd` may be: far above any real run, so it only bounds a typo.
const MAX_CAP_USD: f64 = 1_000_000.0;

/// The one thing the task body cannot know: that it is one voice of several, judged alone.
const PANEL_FRAMING: &str = "You are one member of an independent review panel. You will not \
    see the other members' replies, so do not hedge toward a consensus you cannot observe.";

/// The default ceiling truncation escalates toward; `--max-tokens` is only where a member starts.
/// `check_models` lowers `Route::cap` to any smaller ceiling the provider publishes.
pub(super) const MODEL_MAX_OUT: u32 = 128_000;

/// A member's wall clock when neither `--max-seconds` nor the task names one.
pub(super) const DEFAULT_MAX_SECONDS: u64 = 900;
/// `Instant + Duration` panics on overflow. `ls` also bounds a planted `status.json` with it.
pub(super) const MAX_SECONDS: u64 = 86_400;

/// Everything `plan` decided, and nothing it did not.
struct Planned {
    cfg: cfg::Council,
    task: task::Task,
    plan: Vec<Route>,
    /// The rendered task and subject: `brief.md`, `prompt_sha` and every member's seed.
    brief: String,
    required: usize,
    /// Resolved once, so the deadline, the dry run and the manifest agree.
    max_seconds: u64,
    /// Resolved, validated and sign-normalised once, for the same reason.
    max_usd: Option<f64>,
}

pub fn run(
    a: CouncilArgs,
    brave_base: &str,
    brave_key: &str,
    timeout: u64,
) -> Result<i32, Refusal> {
    match plan(&a, brave_base, brave_key)? {
        // `--dry-run`.
        None => Ok(0),
        Some(p) => Ok(execute(&a, *p, brave_base, brave_key, timeout)?),
    }
}

/// Every refusal before spend. Its only requests are the free pre-flights.
fn plan(
    a: &CouncilArgs,
    brave_base: &str,
    brave_key: &str,
) -> Result<Option<Box<Planned>>, Refusal> {
    let c = cfg::load(None)?;
    let mut t = task::resolve(
        a.task
            .as_deref()
            .or(c.default_task.as_deref())
            .unwrap_or("design-review"),
    )?;
    // The flag outranks the task, which outranks the default.
    let max_seconds = a
        .max_seconds
        .or(t.max_seconds)
        .unwrap_or(DEFAULT_MAX_SECONDS);
    let subject = subject_of(a.subject.as_deref(), max_seconds, !a.models.is_empty())?;
    // A flag overrides a task-owned knob.
    if let Some(v) = a.max_tokens {
        t.max_tokens = v;
    }
    if let Some(v) = &a.effort {
        t.effort = Effort::parse(v).ok_or_else(|| format!("unknown effort '{v}'"))?;
    }
    if let Some(v) = a.max_searches {
        t.max_searches = v;
    }
    if let Some(v) = a.max_steps {
        t.max_steps = v;
    }
    // Both directions: `--tools auto` overrides a task's `grounded: false`.
    match a.tools.as_deref() {
        Some("none") => {
            if t.grounded && a.progress {
                note!("note: --tools none disables grounding for a task that expects it");
            }
            t.grounded = false;
        }
        Some("auto") => t.grounded = true,
        _ => {}
    }
    // Keyed on `grounded`, however it was turned off, so no search budget is priced for nothing.
    if !t.grounded {
        t.max_searches = 0;
    }
    if t.source != "builtin" && a.progress {
        note!("note: task '{}' loaded from {}", t.name, t.source);
    }

    cfg::check_headers(&c)?;
    let slugs = panel_slugs(&c, &a.models, &t.panel)?;
    let required = usize::from(a.min_quorum);
    // Before the dedup: this bounds the list as supplied, ahead of the per-slug lookups.
    if slugs.len() > MAX_PANEL {
        return Err(format!(
            "a council takes at most {MAX_PANEL} members, got {}",
            slugs.len()
        )
        .into());
    }

    let mut plan = Vec::new();
    // Dedup on the id actually sent per wire: `direct:x/y` and `x/y`, or two spellings a direct
    // wire strips to one, would be two bills for one answer.
    let mut routed = std::collections::HashSet::new();
    // Resolved here, so an unknown transport is refused rather than rerouted.
    let via = a.via.as_deref().map(wire::wire_for).transpose()?;
    for slug in &slugs {
        let (wname, bare) = transport_for(&c, slug, via);
        let w = wire::wire_for(wname)?;
        if !routed.insert((wname, wire::model_on_wire(w, bare).to_string())) {
            continue;
        }
        let key = cfg::key_for(&c, wname).ok_or_else(|| {
            format!(
                "no API key for {wname} (needed by {slug}); set {} or run `bx council doctor`",
                env_of(wname)
            )
        })?;
        cfg::check_key(wname, &key)?;
        // Resolved once, so the dry run, manifest, pricer and runner agree. The wire floor is part
        // of the start.
        let start_tokens = t.max_tokens.max(w.min_out);
        plan.push(Route {
            slug: slug.clone(),
            bare: bare.to_string(),
            wire: w,
            key,
            base: cfg::base_url(w)?,
            start_tokens,
            cap: start_tokens.max(MODEL_MAX_OUT),
            rate: rate_of(&c, w, bare),
        });
    }

    // One model named once runs (it is what `hint` asks for); a panel collapsed to one is refused.
    if plan.is_empty() {
        return Err(
            "no models to run; name one with -m, or run `bx council doctor` to see the \
                    roster and aliases"
                .to_string()
                .into(),
        );
    }
    if plan.len() == 1 && slugs.len() > 1 {
        return Err(format!(
            "{} entries name one model — duplicate models were merged. Name the members you \
             want, or run `bx council doctor` to see the roster and aliases",
            slugs.len()
        )
        .into());
    }
    if required > plan.len() {
        return Err(format!(
            "--min-quorum {required} exceeds the panel size ({})",
            plan.len()
        )
        .into());
    }

    let brief = t.render(&subject)?;
    // `+ 0.0` turns `-0.0`, which passes the sign check, into the `0` that was meant.
    let cap = a.max_usd.or(c.max_usd).map(|v| v + 0.0);
    // Here, not on the flag: `council.json`'s `max_usd` never passes through clap.
    let src = if a.max_usd.is_some() {
        "--max-usd"
    } else {
        "council.json max_usd"
    };
    if let Some(bad) = cap.filter(|c| !c.is_finite() || !(0.0..=MAX_CAP_USD).contains(c)) {
        let got = if bad.abs() > MAX_CAP_USD {
            format!("{bad:e}")
        } else {
            bad.to_string()
        };
        return Err(format!("{src} must be between 0 and {MAX_CAP_USD}, got {got}").into());
    }
    // A direct wire reports no price, so without a rate the cap could never bind.
    if cap.is_some()
        && let Some(r) = plan
            .iter()
            .find(|r| r.rate.is_none() && r.wire.name != "openrouter")
    {
        return Err(format!(
            "{src} cannot bound {}: it has no rate; add one under \"rates\" in council.json",
            r.slug
        )
        .into());
    }
    // Refusals that need no network come first, so nothing reaches a provider until the plan is
    // self-consistent. Council never prompts for the Brave key, so this is its whole onboarding.
    if t.grounded {
        let how = if brave_key.is_empty() {
            Some("not set".to_string())
        } else {
            crate::config::validate_api_key(brave_key)
                .err()
                .map(|e| e.to_string())
        };
        if let Some(how) = how {
            return Err(format!(
                "Brave Search key (for the search tool): {how} — set BRAVE_SEARCH_API_KEY or run \
                 `bx config set-key <KEY>`, or pass --tools none to answer from recall"
            )
            .into());
        }
    }
    // Every local precondition of the real run, so a dry run refuses what the run would. Read
    // only: the lock is asked about, not taken, and `execute` re-resolves the path.
    let blessed = match &a.out {
        Some(path) => {
            let real = check_out(path)?;
            refuse_finished_run(&real)?;
            if let Some(why) = lock_refusal(&real.join(LOCK), &real) {
                return Err(why.into());
            }
            Some(real)
        }
        None => {
            let root = state_dir()?;
            make_private_dir(&root).map_err(|e| format!("{}: {e}", root.display()))?;
            None
        }
    };
    check_models(&mut plan, cfg::zdr(&c), t.effort)?;
    // `GET /key` is free, and answers once what would otherwise be one 402 per member.
    if let Some(r) = plan.iter().find(|r| r.wire.name == "openrouter") {
        check_credit(&r.base, &r.key)?;
    }

    let planned = Planned {
        max_seconds,
        max_usd: cap,
        cfg: c,
        task: t,
        plan,
        brief,
        required,
    };
    if a.dry_run {
        print_plan(a, &planned, blessed.as_deref(), brave_base);
        return Ok(None);
    }
    // `worst_case` is no gate: it sits far above any real bill. `--max-usd` is enforced against
    // real spend between rounds, in `Member::drive`.
    Ok(Some(Box::new(planned)))
}

/// What this panel's past runs of this task were charged, from `history.jsonl`: the figure to
/// size `--max-usd` from. LLM spend only, and absent below three samples.
fn observed(plan: &[Route], task: &str) -> Option<serde_json::Value> {
    let path = state_dir().ok()?.join(output::HISTORY);
    let tail = util::read_tail(&path, output::MAX_HISTORY).ok()?;
    let mut rows = output::rows(&tail.text);
    // A quick task's rows would undersize a cap for an audit.
    rows.retain(|r| r.task.is_empty() || r.task == task);
    let slugs: Vec<&str> = plan.iter().map(|r| r.slug.as_str()).collect();
    let stat = |s: &output::Stats| {
        serde_json::json!({"n": s.n, "p50": output::micro(s.p50),
                           "p90": output::micro(s.p90), "max": output::micro(s.max)})
    };
    let members: Vec<serde_json::Value> = plan
        .iter()
        .filter_map(|r| {
            let s = output::per_model(&rows, &r.slug)?;
            let mut v = stat(&s);
            v["model"] = r.slug.as_str().into();
            Some(v)
        })
        .collect();
    let run = output::per_run(&rows, &slugs);
    if members.is_empty() && run.is_none() {
        return None;
    }
    Some(crate::query::build_json_body(&[
        ("source", Some(output::HISTORY.into())),
        ("run_total", run.as_ref().map(stat)),
        ("members", (!members.is_empty()).then(|| members.into())),
        // Present only when true: past `MAX_HISTORY` the figures describe a suffix.
        (
            "window",
            (!tail.whole).then(|| "partial: older rows are past the read cap".into()),
        ),
        (
            "note",
            Some(
                "what your own runs of these models were charged, not a bound. LLM spend only — \
                 searches are billed separately. Size --max-usd above max."
                    .into(),
            ),
        ),
    ]))
}

/// `--dry-run`: what the real run would send, in the format it would use.
fn print_plan(a: &CouncilArgs, p: &Planned, out: Option<&std::path::Path>, brave: &str) {
    let Planned {
        cfg: c,
        task: t,
        plan,
        brief,
        max_seconds,
        max_usd: cap,
        ..
    } = p;
    let (max_seconds, brief, cap) = (*max_seconds, brief.as_str(), *cap);
    let (worst, unpriced) = worst_case(c, plan, t, brief);
    let seen = observed(plan, &t.name);
    let p_len = brief.len();
    let queries = u64::from(t.max_searches) * plan.len() as u64;
    if a.format != "text" {
        say!(
            "{}",
            serde_json::json!({
                "dry_run": true,
                "task": {"name": t.name, "source": t.source},
                // Per member: effort sent, floor, ceiling and price vary by wire and model. No
                // pin omits `providers` rather than publishing `[]`.
                "members": plan.iter().map(|r| crate::query::build_json_body(&[
                    ("model", Some(r.slug.clone().into())),
                    ("transport", Some(r.wire.name.into())),
                    ("base", Some(r.base.clone().into())),
                    ("effort", Some((r.wire.level)(t.effort).into())),
                    ("max_tokens", Some(r.start_tokens.into())),
                    ("max_tokens_ceiling", Some(r.cap.into())),
                    ("priced", Some(r.rate.is_some().into())),
                    ("providers", Some(cfg::providers_for(c, &r.bare))
                        .filter(|v| !v.is_empty())
                        .map(Into::into)),
                ])).collect::<Vec<_>>(),
                // The resolved path; null when the run mints its own directory.
                "out": out.map(|d| util::sanitize(&d.display().to_string())),
                "limits": {
                    // As typed; each member's `effort` is what its wire sends.
                    "effort": t.effort.as_str(),
                    "max_searches": t.max_searches,
                    "max_steps": t.steps(),
                    "min_reply_chars": t.min_reply_chars,
                    "max_seconds": max_seconds,
                },
                // No history omits `observed`; an uncapped run publishes `max_usd: null`.
                "cost": crate::query::build_json_body(&[
                    ("observed", seen),
                    ("worst_usd", Some(worst.into())),
                    ("priced_members", Some((plan.len() - unpriced).into())),
                    ("unpriced_members", Some(unpriced.into())),
                    ("max_usd", Some(serde_json::json!(cap))),
                    ("note", Some(format!(
                        "upper bound, not a forecast: every round at the {MODEL_MAX_OUT}-token \
                         ceiling, {} billed attempts each, {} parallel legs, no cache, plus \
                         every member's whole search budget at Brave's per-call price. The \
                         *inference* of an unpriced member is not in it. Those terms dominate, \
                         so the subject barely moves this figure. --max-usd, not this, is \
                         the gate",
                        super::member::MAX_RETRIES + 1,
                        super::types::PARALLEL_LEGS
                    ).into())),
                ]),
                "egress": {
                    "brief_bytes": p_len,
                    "members": plan.len(),
                    "max_search_queries": queries,
                    "brave_base": brave,
                },
            })
        );
        return;
    }
    say!("task     {} ({})", t.name, t.source);
    for r in plan {
        let pinned = cfg::providers_for(c, &r.bare);
        say!(
            "  {:<32} {:<11} {} effort={} max_tokens={}→{}{}",
            r.slug,
            r.wire.name,
            r.base,
            (r.wire.level)(t.effort),
            r.start_tokens,
            r.cap,
            if pinned.is_empty() {
                String::new()
            } else {
                format!(" providers={}", pinned.join(","))
            }
        );
    }
    if let Some(d) = out {
        say!("writes   {}", util::sanitize(&d.display().to_string()));
    }
    say!(
        "limits   effort={} searches={} steps={} floor={} seconds={max_seconds}",
        t.effort.as_str(),
        t.max_searches,
        t.steps(),
        t.min_reply_chars
    );
    // Before `worst`: this is the figure to size a cap from.
    if let Some(t) = seen.as_ref().and_then(|s| s.get("run_total")) {
        say!(
            "observed ${:.2} median · ${:.2} p90 · ${:.2} max over {} past run(s) of this panel \
             — size --max-usd above max{}",
            t["p50"].as_f64().unwrap_or(0.0),
            t["p90"].as_f64().unwrap_or(0.0),
            t["max"].as_f64().unwrap_or(0.0),
            t["n"].as_u64().unwrap_or(0),
            match seen.as_ref().and_then(|s| s.get("window")) {
                Some(_) => " (older rows are past the read cap)",
                None => "",
            }
        );
    }
    say!(
        "worst    ${worst:.2} upper bound over {} priced member(s){}",
        plan.len() - unpriced,
        match unpriced {
            0 => String::new(),
            n => format!(" — {n} member(s) have no rate and are NOT in this figure"),
        }
    );
    say!(
        "         (includes up to {} billed attempts per round; --max-seconds bounds waiting, not spend)",
        super::member::MAX_RETRIES + 1
    );
    if let Some(cap) = cap {
        // The typed value: `{:.2}` would print a tiny cap as `$0.00`.
        say!(
            "cap      --max-usd ${}, enforced against real spend between rounds — this \
             figure is not a gate",
            output::cap_usd(cap)
        );
    }
    // Two egresses: the whole brief to each model, and model-written queries to Brave.
    say!(
        "egress   the full brief ({} bytes: task + subject) to each of {} members; up to {} \
         search queries (≤400 chars each) to {brave}",
        p_len,
        plan.len(),
        queries
    );
}

// ── The fan-out ──────────────────────────────────────────────────────

fn execute(
    a: &CouncilArgs,
    p: Planned,
    brave_base: &str,
    brave_key: &str,
    timeout: u64,
) -> Result<i32, String> {
    let (dir, id) = match &a.out {
        // Only a directory this run creates is made private. The id is still minted, never
        // taken from the directory's name.
        Some(path) => {
            let real = check_out(path)?;
            out_dir(&real)?;
            (real, util::run_id())
        }
        None => fresh_run_dir(&state_dir()?)?,
    };
    // Held for the whole run, on both paths: artifact names are fixed, so two live runs in one
    // directory would overwrite each other's replies.
    let held = out_lock(&dir)?;
    // Under the lock, so no run can publish between this check and our writes.
    if a.out.is_some() {
        refuse_finished_run(&dir)?;
    }
    output::write_private(&dir.join("brief.md"), &p.brief)?;
    // Before the first request, so `status` and `ls` can see a live run.
    output::write_private(
        &dir.join("status.json"),
        &status_json(0, p.plan.len(), p.max_seconds, p.max_usd),
    )?;

    // The run dir is operator-supplied: every published path is built from this, never `dir`.
    let shown = std::path::PathBuf::from(util::sanitize(&dir.display().to_string()));
    let mut info = RunInfo {
        // Minted, never the directory's name: it is OpenRouter's sticky-routing `session_id`.
        id,
        dir: shown,
        task: p.task.name.clone(),
        task_source: p.task.source.clone(),
        prompt_sha: util::sha256_short(p.brief.as_bytes()),
        elapsed_ms: 0,
        required: p.required,
        exit: 0,
        log_write_errors: 0,
        trace_dropped: 0,
        brave_usd: None,
        max_usd: p.max_usd,
        zdr: cfg::zdr(&p.cfg),
        sources_file: None,
        index_file: None,
    };
    // Before the first request, so a killed run is still identifiable from its own output.
    if a.progress {
        note!(
            "council: {} · {} · {} members · subject {} ({} B) → {}",
            info.id,
            info.task,
            p.plan.len(),
            info.prompt_sha,
            p.brief.len(),
            dir.display()
        );
    }
    // Every credential the run holds. Brave's only if valid: a short leftover would redact
    // ordinary words out of every reply.
    let mut secrets: Vec<String> = p.plan.iter().map(|r| r.key.clone()).collect();
    secrets.extend(
        crate::config::validate_api_key(brave_key)
            .is_ok()
            .then(|| brave_key.to_string()),
    );
    secrets.sort();
    secrets.dedup();
    let started = Instant::now();

    let total = p.plan.len();
    // Panel-wide: `--max-usd` caps the run, and a refused Brave key is refused for everyone.
    let spent = std::sync::atomic::AtomicU64::new(0);
    let search_down = std::sync::atomic::AtomicU16::new(0);
    let dropped = std::sync::atomic::AtomicU32::new(0);
    let ctx = Ctx {
        task: &p.task,
        cfg: &p.cfg,
        dir: &dir,
        brief: &p.brief,
        info: &info,
        progress: a.progress,
        trace: a.trace,
        total,
        brave_base,
        brave_key,
        secrets: &secrets,
        timeout,
        max_seconds: p.max_seconds,
        max_usd: p.max_usd,
        dropped: &dropped,
        spent: &spent,
        search_down: &search_down,
    };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let plan: &[Route] = &p.plan;
    let (mut members, trace_errors) = std::thread::scope(|scope| {
        // Inside the scope, so an unwinding panic drops `tx` and the writer's join cannot hang.
        let (tx, rx) = mpsc::sync_channel::<Event>(QUEUE);
        let writer = scope.spawn({
            let ctx = &ctx;
            move || writer_loop(rx, ctx)
        });

        // A queue, not fixed waves: `--concurrency` bounds members in flight, and a wave would
        // wait for its slowest member. Each worker returns its own results.
        let limit = a.concurrency.map_or(total, usize::from).min(total);
        let workers: Vec<_> = (0..limit)
            .map(|_| {
                let (tx, ctx, next) = (tx.clone(), &ctx, &next);
                scope.spawn(move || {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(entry) = plan.get(i) else { break };
                        mine.push((i, one_member(i, entry, ctx, tx.clone())));
                    }
                    mine
                })
            })
            .collect();
        // Re-raised, never dropped: a silently shrinking panel is the failure to prevent.
        let mut indexed: Vec<(usize, Member)> = workers
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect();
        // Requested order, so two manifests of one command diff cleanly.
        indexed.sort_by_key(|(i, _)| *i);
        let out: Vec<Member> = indexed.into_iter().map(|(_, m)| m).collect();
        // Before the join: `for ev in rx` ends only once every sender is dropped.
        drop(tx);
        (
            out,
            writer
                .join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e)),
        )
    });

    output::mark_shared(&mut members);
    info.elapsed_ms = started.elapsed().as_millis();
    info.exit = exit_for(&members, p.required);
    // Writes that were meant to land and did not. Dropped trace events are lossy by design and
    // counted apart.
    info.log_write_errors = members
        .iter()
        .fold(trace_errors, |n, m| n.saturating_add(m.write_errors));
    info.trace_dropped = dropped.load(std::sync::atomic::Ordering::Relaxed);
    // `brave_calls`, not `searches`: a malformed call is a search attempt that Brave never billed.
    info.brave_usd = brave_usd(
        cfg::brave_price(&p.cfg),
        members.iter().map(|m| m.brave_calls).sum(),
    );

    // Both before the manifest is built, so their failures are counted in it. A failed sources
    // write leaves `sources_file` unset, and the URLs stay inline.
    match output::write_private(&dir.join("index.md"), &output::index_md(&info, &members)) {
        Ok(()) => info.index_file = Some(info.dir.join("index.md")),
        Err(e) => {
            note!("warning: {e}");
            info.log_write_errors = info.log_write_errors.saturating_add(1);
        }
    }
    let sources = dir.join("sources.json");
    match output::sources_json(&info, &members)
        .ok_or_else(|| "sources.json could not be built".to_string())
        .and_then(|body| output::write_private(&sources, &body))
    {
        Ok(()) => info.sources_file = Some(info.dir.join("sources.json")),
        Err(e) => {
            note!("warning: {e} — the retrieved URLs stay in the manifest instead");
            info.log_write_errors = info.log_write_errors.saturating_add(1);
        }
    }
    let manifest = |info: &RunInfo| {
        output::manifest(
            info,
            &members,
            p.task.max_searches,
            p.task.steps(),
            p.max_seconds,
            p.task.min_reply_chars,
        )
    };
    let mut man = manifest(&info);
    // Post-spend, so warn, never Err. Under text, the manifest goes to stderr instead.
    if let Err(e) = output::write_private(&dir.join("manifest.json"), &man.to_string()) {
        info.log_write_errors = info.log_write_errors.saturating_add(1);
        man = manifest(&info);
        // Kept: with no manifest, only the lock stops a later `--out` from replacing the replies.
        std::mem::forget(held);
        if a.format == "text" {
            note!("warning: {e} — the manifest follows on stderr\n{man}");
        } else {
            note!("warning: {e} — the manifest below, on stdout, is now the only copy");
        }
    }
    // Best effort, once: makes every rename into the run dir survive a power loss.
    #[cfg(unix)]
    std::fs::File::open(&dir).and_then(|d| d.sync_all()).ok();

    // `write_all`, not `say!`, which would add a second newline. A reader that hung up is not a
    // failure; `println!` would panic, and lose the exit code after the spend.
    let rendered = match a.format.as_str() {
        "text" => output::text(&info, &members),
        _ => format!("{man}\n"),
    };
    std::io::stdout().write_all(rendered.as_bytes()).ok();
    if a.progress {
        note!(
            "council: {}/{} delivered · {:.0}s → {}",
            members.iter().filter(|m| m.outcome.delivered()).count(),
            members.len(),
            started.elapsed().as_secs_f64(),
            dir.display()
        );
    }
    // `held` drops here, after every artifact is on disk.
    Ok(info.exit)
}

/// The file that claims a run directory; `plan` reads it and `execute` creates it.
pub(super) const LOCK: &str = ".council-lock";

/// Artifact names are fixed, so a second run into a finished one would replace its replies.
/// `plan` asks for the dry run, and `execute` again under the lock.
fn refuse_finished_run(real: &std::path::Path) -> Result<(), String> {
    if real.join("manifest.json").exists() {
        return Err(format!(
            "{} already contains manifest.json; pass a fresh --out — council will not write over \
             a finished run",
            real.display()
        ));
    }
    Ok(())
}

/// Words `lock_refusal`'s refusal: deleting the lock would overwrite these.
fn holds_replies(dir: &std::path::Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|d| {
        d.flatten().any(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.as_bytes().first().is_some_and(u8::is_ascii_digit)
                && (name.ends_with(".md") || name.ends_with(".partial"))
        })
    })
}

/// Released however the run leaves, errors included. Only a run that died (`panic = "abort"`
/// skips `Drop`) or lost its manifest leaves its lock for a human to judge.
pub(super) struct OutLock(std::path::PathBuf);

impl Drop for OutLock {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).ok();
    }
}

/// Why a held lock refuses a run, or `None` when there is none. Names the file and what deleting
/// it costs; shared by `plan` and `out_lock`, so both say the same thing.
fn lock_refusal(path: &std::path::Path, dir: &std::path::Path) -> Option<String> {
    std::fs::symlink_metadata(path).is_ok().then(|| {
        format!(
            "another run holds {}; wait for it, or pass a fresh --out{}",
            path.display(),
            if holds_replies(dir) {
                " — this directory already holds replies, and deleting the lock to re-run here \
                 would overwrite them"
            } else {
                ". Delete that file only if no run is live"
            }
        )
    })
}

/// Takes the lock by `create_new`. `prune` takes it too, so no run can start in a directory it is
/// removing.
pub(super) fn lock(dir: &std::path::Path) -> std::io::Result<OutLock> {
    let path = dir.join(LOCK);
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    let mut f = o.open(&path)?;
    std::io::Write::write_all(
        &mut f,
        format!("pid {} at {}\n", std::process::id(), util::iso8601_now()).as_bytes(),
    )
    .ok();
    Ok(OutLock(path))
}

fn out_lock(dir: &std::path::Path) -> Result<OutLock, String> {
    let path = dir.join(LOCK);
    match lock(dir) {
        Ok(l) => Ok(l),
        // `create_new` is the authority: it catches a run that took the lock after `plan` looked.
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Err(lock_refusal(&path, dir)
            .unwrap_or_else(|| format!("another run holds {}", path.display()))),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// `mkdir` at `0700`: the mode is part of the create, so no later chmod can follow a swapped link.
fn private_dir(dir: &std::path::Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new().mode(0o700).create(dir)
    }
    #[cfg(not(unix))]
    std::fs::create_dir(dir)
}

/// Creates a directory private, leaving one it merely found (or a planted link) untouched.
pub(super) fn make_private_dir(dir: &std::path::Path) -> std::io::Result<()> {
    match private_dir(dir) {
        Ok(()) => return Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => return Ok(()),
        // Parents one level at a time: `create_dir_all` would leave them at the umask.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            if let Some(parent) = dir.parent() {
                make_private_dir(parent)?;
            }
        }
        Err(e) => return Err(e),
    }
    match private_dir(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(e),
    }
}

/// The `--out` leaf, created if absent, then re-checked: a link planted since `check_out` would
/// carry every write through it.
pub(super) fn out_dir(real: &std::path::Path) -> Result<(), String> {
    make_private_dir(real).map_err(|e| format!("{}: {e}", real.display()))?;
    if !std::fs::symlink_metadata(real).is_ok_and(|m| m.is_dir()) {
        return Err(format!(
            "--out {} is no longer a directory; pass the directory it names",
            real.display()
        ));
    }
    Ok(())
}

/// A run directory this call created, and the id it is named for (a collision draws anew).
pub fn fresh_run_dir(root: &std::path::Path) -> Result<(std::path::PathBuf, String), String> {
    make_private_dir(root).map_err(|e| format!("{}: {e}", root.display()))?;
    for _ in 0..8 {
        let id = util::run_id();
        let dir = root.join(&id);
        match private_dir(&dir) {
            Ok(()) => return Ok((dir, id)),
            // Only a collision is worth another draw.
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("{}: {e}", dir.display())),
        }
    }
    Err("could not draw an unused run id".into())
}

/// What every member needs and none of them owns.
struct Ctx<'a> {
    task: &'a task::Task,
    cfg: &'a cfg::Council,
    dir: &'a std::path::Path,
    brief: &'a str,
    info: &'a RunInfo,
    progress: bool,
    trace: bool,
    /// Members in the panel.
    total: usize,
    brave_base: &'a str,
    brave_key: &'a str,
    /// Every credential this run holds, so no sink is redacted against a subset.
    secrets: &'a [String],
    /// `--timeout`, the bound on one Brave request.
    timeout: u64,
    /// The resolved wall clock for one member.
    max_seconds: u64,
    max_usd: Option<f64>,
    /// Trace records dropped for want of queue space.
    dropped: &'a std::sync::atomic::AtomicU32,
    /// The panel's running spend in micro-dollars.
    spent: &'a std::sync::atomic::AtomicU64,
    search_down: &'a std::sync::atomic::AtomicU16,
}

fn one_member(idx: usize, entry: &Route, cx: &Ctx, tx: mpsc::SyncSender<Event>) -> Member {
    let (t, c) = (cx.task, cx.cfg);
    let Route {
        slug,
        bare,
        wire: w,
        key,
        base,
        start_tokens,
        cap,
        rate,
    } = entry;
    let started = Instant::now();
    let path = cx.dir.join(format!("{idx:02}-{}.md", util::slugify(slug)));
    let partial = path.with_extension("md.partial");
    let transcript = cx.trace.then(|| path.with_extension("transcript.json"));

    let agent = ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .http_status_as_error(false)
            .max_redirects(0)
            // `post_raw` bounds each call by the member's remaining clock instead.
            .timeout_global(None)
            .user_agent(concat!("bx/", env!("CARGO_PKG_VERSION")))
            .build(),
    );
    let mut headers: Vec<(&'static str, String)> = vec![(w.auth.0, format!("{}{key}", w.auth.1))];
    headers.extend(w.headers.iter().map(|(k, v)| (*k, (*v).to_string())));
    if w.name == "openrouter" {
        if let Some(r) = &c.referer {
            headers.push(("HTTP-Referer", r.clone()));
        }
        headers.push((
            "X-Title",
            c.title.clone().unwrap_or_else(|| "bx council".into()),
        ));
    }

    let start_tokens = *start_tokens;
    let trace = Trace {
        // `None` without `--trace`, which makes `emit` free.
        tx: cx.trace.then_some(&tx),
        model: slug.clone(),
        dropped: cx.dropped,
    };
    let mut runner = Runner {
        wire: w,
        params: Params {
            model: bare.clone(),
            // Not the task: the rendered brief is the seed, where the cache breakpoint sits.
            system: PANEL_FRAMING.to_string(),
            max_tokens: start_tokens,
            effort: t.effort,
            tools: t.grounded,
            cache: c
                .cache_min_tokens
                .is_none_or(|m| cx.brief.len() as u64 / 4 >= m),
            session: cx.info.id.clone(),
            stateless: cfg::zdr(c) && w.name == "openai",
            zdr: cfg::zdr(c),
            // Keyed on the bare id, so a pin follows the model, not its spelling.
            providers: cfg::providers_for(c, bare),
        },
        agent: &agent,
        url: format!("{base}{}", w.path),
        headers,
        tool: search::Tool {
            agent: &agent,
            base: cx.brave_base,
            key: cx.brave_key,
            timeout: Duration::from_secs(cx.timeout),
            down: cx.search_down,
        },
        max_searches: t.max_searches,
        max_steps: t.steps(),
        min_reply_chars: t.min_reply_chars,
        cap: *cap,
        // One deadline per member, from its start; `MAX_SECONDS` keeps the sum from overflowing.
        deadline: Instant::now() + Duration::from_secs(cx.max_seconds),
        trace,
        partial: partial.clone(),
        transcript: transcript.clone(),
        secrets: cx.secrets,
        max_usd: cx.max_usd,
        spent: cx.spent,
        rate: *rate,
        brave_usd: cfg::brave_price(c),
    };

    let mut r = runner.run(cx.brief.to_string());
    // `scrub` cleans the reply and never shortens it; `label` also bounds the short
    // provider-controlled strings the manifest carries.
    let scrub = |s: &str| util::redact(&util::sanitize(s), cx.secrets);
    let label = |s: &str| util::truncate(&scrub(s), 512);

    // Before the gate, which judges the text that will be published.
    r.text = scrub(&r.text);
    let outcome = match gate::evaluate(&r, t.min_reply_chars) {
        Outcome::Rejected {
            reason,
            error,
            status,
        } => Outcome::Rejected {
            reason,
            error: label(&error),
            status,
        },
        ok @ Outcome::Delivered { .. } => ok,
    };

    // Named only when this run wrote it: a leftover at that path is a crashed run's.
    let transcript_name = transcript
        .filter(|_| r.transcript_written)
        .and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()));
    // A member that never sent a round spent nothing; `price` would read its zeros as unreported.
    let (cost_usd, cost_source) = if r.rounds == 0 {
        (Some(0.0), "unbilled")
    } else {
        price(w.name, *rate, &r.usage)
    };
    let mut m = Member {
        model: slug.clone(),
        served_model: r.served_model.as_deref().map(&label),
        provider: r.provider.as_deref().map(&label),
        transport: w.name,
        outcome,
        text: r.text,
        path,
        usage: r.usage,
        cost_usd,
        cost_source,
        rounds: r.rounds,
        retries: r.retries,
        searches: r.searches,
        brave_calls: r.brave_calls,
        grounded: r.grounded,
        search_errors: r.search_errors,
        // Attacker-controlled: cleaned, and only `http(s)`, since a `data:` URL is a payload.
        sources: r
            .sources
            .iter()
            .filter(|u| {
                ["http://", "https://"]
                    .iter()
                    .any(|p| u.get(..p.len()).is_some_and(|h| h.eq_ignore_ascii_case(p)))
            })
            // At the length the member read, so a followed citation is never cut.
            .map(|u| util::truncate(&scrub(u), search::MAX_URL_CHARS))
            .collect(),
        sources_shared: 0,
        peak_round: r.peak_round,
        escalations: r.escalations,
        max_tokens: r.max_tokens,
        max_tokens_reached: r.max_tokens_reached,
        write_errors: r.write_errors,
        transcript: transcript_name,
        elapsed: started.elapsed(),
        terminated_by: r.terminated_by,
        finish: r.finish,
    };
    // Written before the member is joined — see `Member::partial` for why that matters.
    if let Err(e) = output::write_member(&m.path, &m) {
        note!("warning: {e}");
        m.write_errors = m.write_errors.saturating_add(1);
        // The partial is then the only copy, so `file` names it, if this run wrote it.
        if r.flushed > 0 {
            m.path = partial.clone();
        }
        // Only a delivered reply becomes `write_failed`; a rejection keeps its own reason.
        if m.outcome.delivered() {
            m.outcome = Outcome::Rejected {
                reason: Reason::WriteFailed,
                error: label(&format!("reply could not be written: {e}")),
                status: None,
            };
        }
    }
    // Removed only when it holds exactly what was published.
    else if r.draft_flushed {
        std::fs::remove_file(&partial).ok();
    }
    // Otherwise it holds paid text the reply does not, a short answer having beaten a long
    // scratchpad. It stays, and `findings` reads it.
    else if r.flushed > 0 && cx.progress {
        note!(
            "note: {} holds paid text the reply does not",
            partial.display()
        );
    }
    let lost = hand_off(
        &tx,
        slug,
        [
            ("history", output::history_line(cx.info, &m)),
            (
                "member_done",
                json!({
                    "status": if m.outcome.delivered() { "delivered" } else { "rejected" },
                    "chars": m.chars(), "searches": m.searches, "grounded": m.grounded,
                    "elapsed_s": started.elapsed().as_secs(),
                }),
            ),
        ],
    );
    m.write_errors = m.write_errors.saturating_add(lost);
    m
}

/// A finished member's records, sent blocking, since they follow `write_member` and no paid work
/// waits on them. Returns how many a gone writer refused.
pub(super) fn hand_off(
    tx: &mpsc::SyncSender<Event>,
    model: &str,
    records: [(&'static str, serde_json::Value); 2],
) -> u32 {
    records
        .into_iter()
        .map(|(kind, body)| {
            let event = Event {
                model: model.to_string(),
                kind,
                body,
            };
            u32::from(tx.send(event).is_err())
        })
        .sum()
}

/// A member's cost and its source: `reported` or `byok_upstream` on OpenRouter, `computed` from
/// the rate on a direct wire, `_partial` when some round went unpriced, else `unknown`.
pub(super) fn price(
    wire: &str,
    rate: Option<[f64; 3]>,
    u: &super::types::Usage,
) -> (Option<f64>, &'static str) {
    // Publishable or nothing on both paths: finite sums can still overflow.
    let finite = cfg::publishable_usd;
    if wire == "openrouter" {
        // A clamped figure sits above what was charged: not even a floor.
        if u.cost_clamped {
            return (None, "unknown");
        }
        // No input tokens is unknown, not free; a reported zero beside real tokens is a price.
        if u.prompt == 0 && u.spent().is_some_and(|v| v == 0.0) {
            return (None, "unknown");
        }
        return match u.spent().and_then(finite) {
            // A subtotal outranks the BYOK label: it is the fact a reader must act on.
            Some(v) if u.cost_incomplete => (Some(v), "reported_partial"),
            Some(v) if u.is_byok == Some(true) => (Some(v), "byok_upstream"),
            Some(v) => (Some(v), "reported"),
            None => (None, "unknown"),
        };
    }
    match rate {
        // Zero input is an unparsed or absent count; no real generation reports it.
        Some(_) if u.prompt == 0 => (None, "unknown"),
        Some(rate) => match finite(computed_usd(u, rate)) {
            Some(v) if u.cost_incomplete => (Some(v), "computed_partial"),
            Some(v) => (Some(v), "computed"),
            None => (None, "unknown"),
        },
        None => (None, "unknown"),
    }
}

/// Cache reads at the rate's multiplier; 5-minute writes at 1.25x input.
pub fn computed_usd(u: &super::types::Usage, [inp, out, cached]: super::cfg::Rate) -> f64 {
    let (read, write) = (u.cache_read.unwrap_or(0), u.cache_write.unwrap_or(0));
    // A provider reporting more cached than total must not wrap `fresh`.
    let fresh = u.prompt.saturating_sub(read.saturating_add(write));
    (fresh as f64 * inp
        + read as f64 * inp * cached
        + write as f64 * inp * 1.25
        + u.completion as f64 * out)
        / 1e6
}

/// A uniform cause gets its own code; a mixed failure is 6, "quorum not met".
pub fn exit_for(ms: &[Member], required: usize) -> i32 {
    if ms.iter().filter(|m| m.outcome.delivered()).count() >= required {
        return 0;
    }
    let mut codes = ms.iter().filter_map(|m| match &m.outcome {
        // The status, never the message, so a provider cannot choose the exit code.
        Outcome::Rejected {
            reason: Reason::ProviderError,
            status,
            ..
        } => Some(status_class(*status)),
        // Raising the cap is a remedy no other rejection has.
        Outcome::Rejected {
            reason: Reason::Cost,
            ..
        } => Some(7),
        Outcome::Rejected { .. } => Some(1),
        Outcome::Delivered { .. } => None,
    });
    // 1 is the unclassified catch-all, outside the documented set; it falls to 6.
    match codes.next() {
        Some(first) if first != 1 && codes.all(|c| c == first) => first,
        _ => 6,
    }
}

/// What the panel's Brave calls cost; price and product both must be publishable.
pub(super) fn brave_usd(price: Option<f64>, calls: u32) -> Option<f64> {
    price
        .and_then(cfg::publishable_usd)
        .map(|p| p * f64::from(calls))
        .and_then(cfg::publishable_usd)
}

/// `api::exit_code_for_status`, plus 8 for a 402: out of credit has one clear remedy.
fn status_class(status: Option<u16>) -> i32 {
    match status {
        Some(402) => 8,
        Some(s) => crate::api::exit_code_for_status(s),
        None => 1,
    }
}

// ── The writer thread ────────────────────────────────────────────────

/// The polling file, written at the start and after every member. Its `max_seconds` lets a
/// reader tell a killed run from a live one, which touches it at least once per member.
fn status_json(done: usize, total: usize, max_seconds: u64, max_usd: Option<f64>) -> String {
    json!({"done": done, "total": total, "ts": util::iso8601_now(),
           "max_seconds": max_seconds, "max_usd": max_usd})
    .to_string()
}

/// Sole owner of every shared file. One `write_all` per record, no `BufWriter`: `history.jsonl`
/// is shared between processes, and a split record corrupts it.
fn writer_loop(rx: mpsc::Receiver<Event>, cx: &Ctx) -> u32 {
    let Ctx {
        dir,
        secrets,
        total,
        progress,
        trace,
        max_seconds,
        max_usd,
        ..
    } = *cx;
    let mut errors = 0;
    let mut done = 0;
    // A log that cannot be opened (say a planted link) is named once; each record lost to it
    // is counted below.
    let mut tracefile = None;
    if trace {
        match open_append(&dir.join("trace.jsonl")) {
            Ok(f) => tracefile = Some(f),
            Err(e) => note!("warning: trace.jsonl: {e}"),
        }
    }
    // In the state root, which outlives run dirs. Created here too: `--out` skips it.
    let mut history = None;
    match state_dir().and_then(|d| {
        make_private_dir(&d).map_err(|e| format!("{}: {e}", d.display()))?;
        let p = d.join(output::HISTORY);
        open_append(&p).map_err(|e| format!("{}: {e}", p.display()))
    }) {
        Ok(f) => history = Some(f),
        Err(e) => note!("warning: history.jsonl: {e}"),
    }
    for ev in rx {
        let mut v = if ev.kind == "history" {
            ev.body
        } else {
            json!({"ts": util::iso8601_now(), "model": ev.model,
                   "kind": ev.kind, "body": ev.body})
        };
        // Before encoding: `serde_json` would turn an ESC into a `\u001b` that decodes live.
        // Sanitized first, so a control planted inside an echoed key cannot shield it.
        scrub_strings(&mut v, secrets);
        // Again after: a key holding `"` or `\` only matches in its escaped form.
        let line = util::redact(&v.to_string(), secrets);
        // Newline included: one syscall, so two appending processes cannot interleave.
        let record = format!("{line}\n");
        // History goes only to `history.jsonl`; the trace keeps one envelope shape.
        let sink = if ev.kind == "history" {
            history.as_mut()
        } else {
            tracefile.as_mut()
        };
        match sink {
            Some(f) => {
                // A torn tail would swallow this record into its own unparseable line.
                let record = if ends_mid_line(f) {
                    format!("\n{record}")
                } else {
                    record
                };
                if f.write_all(record.as_bytes()).is_err() {
                    errors += 1;
                }
            }
            None if ev.kind == "history" || trace => errors += 1,
            None => {}
        }
        if ev.kind == "history" {
            continue;
        }
        if ev.kind == "member_done" {
            done += 1;
            if progress {
                note!(
                    "  [{done}/{total}] {:<32} {} · {} chars · {}/{} grounded",
                    ev.model,
                    v["body"]["status"].as_str().unwrap_or("?"),
                    v["body"]["chars"],
                    v["body"]["grounded"],
                    v["body"]["searches"]
                );
            }
            if output::write_private(
                &dir.join("status.json"),
                &status_json(done, total, max_seconds, max_usd),
            )
            .is_err()
            {
                errors += 1;
            }
        }
    }
    errors
}

/// Whether a log ends without its newline: a crash or a short write left a torn record.
fn ends_mid_line(f: &mut std::fs::File) -> bool {
    use std::io::{Read, Seek, SeekFrom};
    let mut last = [0u8];
    f.seek(SeekFrom::End(-1)).is_ok() && f.read_exact(&mut last).is_ok() && last[0] != b'\n'
}

/// Sanitizes and redacts every string in `v`.
fn scrub_strings(v: &mut serde_json::Value, secrets: &[String]) {
    match v {
        serde_json::Value::String(s) => *s = util::redact(&util::sanitize(s), secrets),
        serde_json::Value::Array(a) => a.iter_mut().for_each(|v| scrub_strings(v, secrets)),
        serde_json::Value::Object(o) => o.values_mut().for_each(|v| scrub_strings(v, secrets)),
        _ => {}
    }
}

/// An append-only log, opened by name, so it must refuse whatever could be planted there: a
/// symlink, a FIFO, a hard link, a file someone else owns.
pub(super) fn open_append(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true).append(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // `O_NONBLOCK`: a FIFO with no reader would block the writer thread forever.
        o.mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    // Windows' "do not follow": the reparse point itself opens, and `is_file()` refuses it.
    // The link count and mode checks have no stable Windows equivalent.
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        o.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
    }
    let f = o.open(path)?;
    // A FIFO with a reader opens fine.
    if !f.metadata()?.is_file() {
        return Err(std::io::Error::other("not a regular file"));
    }
    // A hard link passes `O_NOFOLLOW`; a log we made has exactly one link.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if f.metadata()?.nlink() != 1 {
            return Err(std::io::Error::other("not an exclusively-linked file"));
        }
    }
    // `mode` applies only at creation. EPERM here reveals a log someone else owns.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(f)
}
