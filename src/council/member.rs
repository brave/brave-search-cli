//! One member's agentic loop. Strictly sequential; the fan-out is one of these per thread.
//!
//! The order inside a round is load-bearing; the numbers are `drive`'s steps:
//!
//! 1. **Keep the text before anything can discard it**: every later branch may return, and a
//!    truncated turn still cost full price.
//! 2. **Truncation, before any tool call is consumed**: a truncated turn's calls are not
//!    trustworthy.
//! 3. **Echo the turn once, before any result is appended.**
//! 4. **Every call gets exactly one result, or none do**: a dropped `tool_result` invalidates the
//!    conversation for good.
//! 5. **Decide the answer before the draft is overwritten**: a finished review with one trailing
//!    call publishes the review.
//! 6. A short reply is re-prompted, up to `MAX_REPROMPTS` times.
//! 7. Publish.

use serde_json::json;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::search::{Found, Tool, extract_query};
use super::types::{Conv, Item, Params, Stop, Turn, Usage};
use super::wire::Wire;
use super::{Event, MAX_BODY, Trace};

/// Searches past this in one turn are a model looping; they get a refusal as their result.
const MAX_CALLS_PER_TURN: usize = 8;
/// A turn with more calls than this ends the member: answering each would flood the
/// conversation before `conv.full()` next runs.
const MAX_CALLS_ANSWERABLE: usize = MAX_CALLS_PER_TURN * 32;
const MAX_REPROMPTS: u32 = 2;
pub(super) const MAX_RETRIES: u32 = 3;
const BASE_MS: u64 = 500;
pub(super) const BACKOFF_CAP_MS: u64 = 8_000;

#[derive(Default)]
pub struct Report {
    pub text: String,
    pub usage: Usage,
    pub rounds: u32,
    pub retries: u32,
    pub searches: u32,
    /// Searches Brave answered with a 200: the billable count. `searches` counts attempts.
    pub brave_calls: u32,
    pub grounded: u32,
    pub search_errors: u32,
    /// Of `search_errors`, the calls whose arguments named no query: they never reached Brave.
    pub malformed: u32,
    /// The Brave status that made search unavailable, once a search met it.
    pub search_status: Option<u16>,
    pub sources: Vec<String>,
    pub peak_round: u64,
    pub escalations: u32,
    /// The cap the first request carried, after the wire's floor and the provider's ceiling.
    pub max_tokens: u32,
    /// The largest cap any request in this member actually carried.
    pub max_tokens_reached: u32,
    /// Failed writes of this member's artifacts, folded into the run's `log_write_errors`.
    pub write_errors: u32,
    // The three `draft_*` flags describe the text in hand, and are rewritten whenever it changes.
    /// Whether the text in hand was cut off at the output cap or the context window.
    pub draft_truncated: bool,
    /// Whether the text in hand came from a turn the wire said to discard: paid for and kept on
    /// disk, but not an answer.
    pub draft_provisional: bool,
    /// Whether the text in hand arrived only on the reasoning channel.
    pub draft_reasoning_only: bool,
    /// Characters on disk in `partial`, scrubbed, once written. The file never shrinks, while
    /// the published text is ranked by kind, so the two are tracked apart.
    pub flushed: usize,
    /// Whether `partial` holds exactly the published text: the only case the caller may delete
    /// it. Identity, not length.
    pub draft_flushed: bool,
    /// Whether this run wrote `transcript`; a leftover file at that path is not ours.
    pub transcript_written: bool,
    pub fatal: Option<String>,
    /// The status that produced `fatal`, so the exit code never depends on message text.
    pub status: Option<u16>,
    /// Whether the published text was cut off: `draft_truncated` at the exit, never the stop of
    /// whichever turn ended the loop.
    pub truncated: bool,
    pub answered: bool,
    pub terminated_by: &'static str,
    pub finish: &'static str,
    pub served_model: Option<String>,
    pub provider: Option<String>,
}

pub struct Member<'a> {
    pub wire: &'static Wire,
    pub params: Params,
    pub agent: &'a ureq::Agent,
    pub url: String,
    pub headers: Vec<(&'static str, String)>,
    pub tool: Tool<'a>,
    pub max_searches: u32,
    pub max_steps: u32,
    pub min_reply_chars: usize,
    pub cap: u32,
    pub deadline: Instant,
    pub trace: Trace<'a>,
    /// Where the best text so far is flushed, so a crash loses at most the turn in flight.
    pub partial: PathBuf,
    /// Written once, at the end, when `--trace` is on.
    pub transcript: Option<PathBuf>,
    /// Every credential this run holds. One list, so no artifact is redacted against a subset.
    pub secrets: &'a [String],
    /// `--max-usd`, against the panel's running spend in micro-dollars: the cap is the run's.
    pub max_usd: Option<f64>,
    pub spent: &'a std::sync::atomic::AtomicU64,
    /// USD per million input and output tokens, and the cache-read multiplier, when the
    /// transport has a rate. `None` on OpenRouter, which reports its price.
    pub rate: Option<super::cfg::Rate>,
    /// Price per Brave call, when configured; charged against `--max-usd` too.
    pub brave_usd: Option<f64>,
}

/// Micro-dollars, for an integer atomic. Rounds up, so the cap is never reached late.
fn micro_usd(v: f64) -> u64 {
    (v * 1e6).ceil().max(0.0) as u64
}

impl Member<'_> {
    /// Every exit from `drive` leaves through here, so `--trace` writes a transcript for the
    /// failures too.
    pub fn run(&mut self, seed: String) -> Report {
        // Bytes of the whole conversation, which is re-sent every round: the seed, one
        // ceiling-filling generation per step (4 B/token), every search at 4 B/char.
        let generations =
            (self.max_steps as usize).saturating_mul((self.cap as usize).saturating_mul(4));
        let results =
            (self.max_searches as usize).saturating_mul(super::search::MAX_RESULT_CHARS * 4);
        let budget = seed
            .len()
            .saturating_add(generations)
            .saturating_add(results);
        let mut conv = Conv::new(seed, budget);
        let mut r = self.drive(&mut conv);
        if let Some(path) = &self.transcript {
            let body = super::output::transcript(&self.trace.model, &conv.items, self.secrets);
            match super::output::write_private(path, &body) {
                Ok(()) => r.transcript_written = true,
                Err(e) => {
                    note!("warning: {e}");
                    r.write_errors += 1;
                }
            }
        }
        r
    }

    fn drive(&mut self, conv: &mut Conv) -> Report {
        let mut r = Report {
            max_tokens: self.params.max_tokens,
            max_tokens_reached: self.params.max_tokens,
            terminated_by: "max_steps",
            finish: "stop",
            ..Report::default()
        };
        let mut reprompts = 0;
        // Whether the previous round raised `max_tokens`.
        let mut raised = false;

        for step in 1..=self.max_steps {
            if Instant::now() >= self.deadline {
                self.stop(&mut r, "budget", "budget");
                return r;
            }
            // Before the request, so the round that would overflow is never paid for.
            if conv.full() {
                self.stop(&mut r, "context_budget", "length");
                return r;
            }
            // Rounds already in flight elsewhere can still cross it: at most one per member.
            if self.over_cap() {
                self.stop(&mut r, "max_usd", "length");
                return r;
            }
            // Recorded where the request is built, so it only ever names a cap that was sent.
            r.max_tokens_reached = self.params.max_tokens;
            self.trace.emit("round", || {
                json!({"round": step, "max_tokens": self.params.max_tokens,
                       "searches": r.searches, "tools": self.params.tools})
            });

            let turn = self.send(conv, &mut r);
            let after_raise = std::mem::take(&mut raised);
            self.account(&mut r, &turn, conv.ceiling());
            r.served_model = turn.served_model.clone().or(r.served_model.take());
            r.provider = turn.provider.clone().or(r.provider.take());

            // ── 1. keep the text before anything can discard it ──
            // `r.text` is the best draft; step 5 may replace it with a terminal turn.
            let cut = matches!(turn.stop, Stop::Truncated | Stop::OverContext);
            let (kept, wrote) = self.keep_draft(&mut r, &turn.text, turn.reasoning_only, cut);
            if kept {
                r.draft_truncated = cut;
                r.draft_provisional = false;
                r.draft_reasoning_only = turn.reasoning_only;
            }

            // The input filled the window: no cap to raise.
            if matches!(turn.stop, Stop::OverContext) {
                self.at_ceiling(&mut r, "context_window");
                return r;
            }

            // ── 2. truncation, before any tool call is consumed ──
            if matches!(turn.stop, Stop::Truncated) {
                if self.params.max_tokens >= self.cap {
                    self.at_ceiling(&mut r, "truncated_at_cap");
                    return r;
                }
                // No round left to send a raised cap in.
                if step == self.max_steps {
                    self.at_ceiling(&mut r, "max_steps");
                    return r;
                }
                // Raising `max_tokens` keeps the cached prefix; changing effort would not.
                self.params.max_tokens = escalate(self.params.max_tokens, self.cap);
                (r.escalations, raised) = (r.escalations + 1, true);
                self.trace
                    .emit("escalate", || json!({"max_tokens": self.params.max_tokens}));
                continue;
            }
            if let Stop::Fatal(m) = &turn.stop {
                // A 400 naming tokens right after a raise is the model's real output limit,
                // which a roster without `max_out` cannot tell us. Any other 400, a spend limit
                // say, is the provider's.
                if after_raise
                    && turn.status == Some(400)
                    && m.to_ascii_lowercase().contains("token")
                {
                    self.at_ceiling(&mut r, "truncated_at_cap");
                    return r;
                }
                // A statusless stop past our own clock or cap is that bound, not a transport
                // fault: the call's deadline *is* the remaining budget. A provider's 401 or 402
                // is its own cause whatever the clock says.
                let ours = turn.status.is_none();
                if ours && Instant::now() >= self.deadline {
                    self.stop(&mut r, "budget", "budget");
                } else if ours && self.over_cap() {
                    self.stop(&mut r, "max_usd", "length");
                } else {
                    (r.fatal, r.status, r.terminated_by, r.finish) =
                        (Some(m.clone()), turn.status, "error", "error");
                }
                return r;
            }

            if matches!(turn.stop, Stop::Refused) {
                // Not `fatal`: a refusal is not a transport fault. The status is for the manifest
                // only; `exit_for` sends every refusal to 6.
                (r.status, r.terminated_by, r.finish) = (turn.status, "refused", "content_filter");
                return r;
            }

            // ── 3. echo the provider's turn — once, and before anything is appended to it ──
            // Everything above returns unechoed: an escalation re-sends the same conversation.
            let echoed = conv.advance(turn.raw, turn.chain);

            // ── 4. tool calls — but never at the cost of the text ──
            // Only when the echo was stored: a `tool_result` with no assistant turn before it
            // invalidates the conversation, so an unechoed turn is treated as text-only.
            if turn.calls.len() > MAX_CALLS_ANSWERABLE {
                (r.terminated_by, r.finish) = ("too_many_calls", "error");
                r.fatal = Some(format!("{} tool calls in one turn", turn.calls.len()));
                return r;
            }
            // The last step has no round left to read results: its answer is judged, and a turn
            // with none ends at the step cap rather than as reasoning-only.
            let calls = echoed && !turn.calls.is_empty();
            if calls
                && step == self.max_steps
                && (turn.reasoning_only || turn.text.trim().is_empty())
            {
                break;
            }
            if calls && step < self.max_steps {
                // Every call in the echoed turn gets exactly one result, or the verbatim
                // assistant blob leaves the conversation invalid.
                let mut searched = 0;
                for call in &turn.calls {
                    searched += usize::from(call.name == "search");
                    let out = if call.name != "search" {
                        Err(format!(
                            "(unknown tool '{}' — only 'search' is available; answer now)",
                            super::util::truncate(&super::util::sanitize(&call.name), 40)
                        ))
                    } else if searched > MAX_CALLS_PER_TURN {
                        Err("(too many tool calls in one turn — answer now)".into())
                    } else if r.searches >= self.max_searches {
                        // A sentinel, never withdrawing `tools`: that re-bills the cached prefix.
                        Err("(search budget exhausted — answer now)".into())
                    } else {
                        Ok(())
                    };
                    let (body, is_error) = match out {
                        Err(sentinel) => (sentinel, true),
                        Ok(()) => self.execute(call.args.as_str(), &mut r),
                    };
                    conv.push(Item::Result {
                        id: call.id.clone(),
                        body,
                        is_error,
                    });
                }
                continue;
            }

            // ── 5. which text is the answer ──
            // A terminal turn wins when it clears the floor or is no shorter than the draft, so
            // "already covered above" cannot delete a review held from an earlier round. An
            // empty turn never clears a floor of 0, and a scratchpad never displaces an answer.
            let (this, draft) = (turn.text.chars().count(), r.text.chars().count());
            let scratchpad_over_answer =
                turn.reasoning_only && !r.draft_reasoning_only && draft > 0;
            let answer = if !scratchpad_over_answer
                && ((this > 0 && this >= self.min_reply_chars) || this >= draft)
            {
                // The flags now describe this turn. `draft_reasoning_only` already agrees: the
                // guard above and `keep_draft`'s rank both see to it.
                (r.draft_truncated, r.draft_provisional) = (false, false);
                r.draft_flushed = wrote || (r.draft_flushed && r.text == turn.text);
                turn.text.clone()
            } else {
                // The draft stands, under its own flags; `draft_flushed` is unchanged with it.
                std::mem::take(&mut r.text)
            };

            // ── 6. short reply: re-prompt, bounded ──
            if answer.chars().count() < self.min_reply_chars && reprompts < MAX_REPROMPTS {
                r.text = answer;
                reprompts += 1;
                conv.push(Item::Nudge(
                    "Provide your final answer now. Do not describe what you plan to do.".into(),
                ));
                continue;
            }

            // ── 7. done ──
            r.text = answer;
            self.publish(&mut r);
            return r;
        }
        let finish = r.finish;
        self.stop(&mut r, "max_steps", finish);
        r
    }

    /// A bound of ours ended the member. A publishable review in hand is still delivered, and
    /// `terminated_by` names the bound either way.
    fn stop(&self, r: &mut Report, by: &'static str, finish: &'static str) {
        let publishable = !(r.draft_truncated || r.draft_provisional || r.draft_reasoning_only)
            && r.text.chars().count() >= self.min_reply_chars.max(1);
        if publishable {
            self.publish(r);
        } else {
            r.finish = finish;
        }
        r.terminated_by = by;
    }

    fn over_cap(&self) -> bool {
        self.max_usd.is_some_and(|cap| spent_usd(self.spent) >= cap)
    }

    /// A turn that hit a ceiling with no room left to grow; `lever` names the one that helps.
    /// Truncated when the text in hand is the cut-off one, or when there is none; otherwise
    /// the text in hand is published, since a later turn running out of room cuts nothing off it.
    fn at_ceiling(&self, r: &mut Report, lever: &'static str) {
        if r.draft_truncated || r.text.is_empty() {
            (r.truncated, r.terminated_by, r.finish) = (true, lever, "length");
        } else {
            self.publish(r);
        }
    }

    /// Turns the flags on the text in hand into the fields the gate reads. The only site that
    /// sets `answered`.
    fn publish(&self, r: &mut Report) {
        (r.truncated, r.finish) = if r.draft_truncated {
            (true, "length")
        } else {
            (false, "stop")
        };
        (r.answered, r.terminated_by) = (true, "answer");
        if r.draft_reasoning_only {
            (r.answered, r.terminated_by) = (false, "reasoning_only");
        }
        // Over `reasoning_only`: a discard names a transport fault, the more specific cause.
        if r.draft_provisional {
            (r.answered, r.terminated_by, r.finish) = (false, "discarded_draft", "error");
        } else if r.text.chars().count() < self.min_reply_chars {
            (r.answered, r.terminated_by) = (false, "short_reply");
        }
    }

    /// One round, with transport and response retries folded in, so the caller's `match` on
    /// `Stop` never sees `Retry`. An exhausted retry becomes `Fatal`, carrying a non-2xx status
    /// for `action`.
    fn send(&mut self, conv: &mut Conv, r: &mut Report) -> Turn {
        let (mut attempt, mut counted) = (0, false);
        let ceiling = conv.ceiling();
        loop {
            // Both bounds here, the point every retry and replay passes through. The request's
            // own deadline is what is left of `--max-seconds`.
            let left = self.deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Turn::stopped(Stop::Fatal("budget exhausted".into()));
            }
            if self.over_cap() {
                return Turn::stopped(Stop::Fatal("--max-usd reached".into()));
            }
            // Past both guards: `retries` counts re-sends that happened.
            if attempt > 0 {
                r.retries += 1;
            }
            let body = (self.wire.build)(conv, &self.params);
            let sent =
                crate::api::post_raw(self.agent, &self.url, &self.headers, &body, MAX_BODY, left);
            // A round counts once an attempt may have reached the provider.
            if !counted && !matches!(&sent, Err(e) if crate::api::is_retryable(e)) {
                (counted, r.rounds) = (true, r.rounds + 1);
            }
            let turn = match sent {
                Ok(resp) => {
                    let mut t = (self.wire.parse)(&resp);
                    // A 200 cut off mid-body was billed but reports no usage: charge a rated
                    // member its worst case, and call the bill partial either way. Bytes bound
                    // input tokens; the conversation covers a chain's server-held prefix.
                    if resp.status == 200 && resp.truncated() {
                        if t.usage.completion == 0 {
                            t.usage.cost_incomplete = true;
                            if let Some(rate) = self.rate {
                                let input = body.len().max(conv.bytes()) as f64;
                                let output = f64::from(self.params.max_tokens);
                                self.bill_usd((input * rate[0] + output * rate[1]) / 1e6);
                            }
                        }
                        // Our clock cut the body, not the provider: statusless, so `drive`
                        // names the bound. OpenRouter sends its 200 before generating.
                        if matches!(t.stop, Stop::Fatal(_)) && Instant::now() >= self.deadline {
                            t.status = None;
                        }
                    }
                    match &t.stop {
                        // Generated and billed like a delivered turn, and possibly a complete
                        // answer, so its text and its cost are both kept here: `drive` only sees
                        // the turn finally returned.
                        Stop::Retry(m) => {
                            if self.keep_draft(r, &t.text, t.reasoning_only, true).0 {
                                (r.draft_provisional, r.draft_truncated) = (true, false);
                                r.draft_reasoning_only = t.reasoning_only;
                            }
                            self.account(r, &t, ceiling);
                            // A 2xx carries no advice: a retryable error on a 200 is still one.
                            let status = t.status.filter(|s| !(200..300).contains(s));
                            Err((m.clone(), resp.retry_after(), status))
                        }
                        _ => Ok(t),
                    }
                }
                Err(e) if crate::api::is_retryable(&e) => Err((e.to_string(), None, None)),
                Err(e) => Ok(Turn::stopped(Stop::Fatal(e.to_string()))),
            };
            let (msg, after, status) = match turn {
                Ok(t) => return t,
                Err(x) => x,
            };

            // Before anything else, so the replay below is bounded too: `MAX_RETRIES + 1`
            // generations, as `worst_case` prices.
            attempt += 1;
            if attempt > MAX_RETRIES {
                // With its status: a uniform 429 exits 4, not 6.
                let mut out = Turn::stopped(Stop::Fatal(msg));
                out.status = status;
                return out;
            }

            // A lost OpenAI chain is not a transport fault: replay statelessly, without sleeping.
            if self.wire.name == "openai" && lost_chain(&msg) && !self.params.stateless {
                self.params.stateless = true;
                conv.go_stateless();
                continue;
            }
            self.trace
                .emit("retry", || json!({"attempt": attempt, "error": msg}));
            // `Retry-After` is honoured as sent, never clamped: sooner buys the same 429. A wait
            // that would leave less than itself to send the round in ends the member now.
            let left = self.deadline.saturating_duration_since(Instant::now());
            let delay = after.unwrap_or_else(|| backoff(attempt));
            if delay.saturating_mul(2) >= left {
                let mut out = Turn::stopped(Stop::Fatal(format!(
                    "{msg} (asked to wait {}s, {}s of budget left)",
                    delay.as_secs(),
                    left.as_secs()
                )));
                out.status = status;
                return out;
            }
            std::thread::sleep(delay);
        }
    }

    /// Runs one search. Never returns an error into the loop: the sentinel is the result.
    fn execute(&self, args: &str, r: &mut Report) -> (String, bool) {
        // A malformed call is a failed attempt, not a choice not to search; `malformed` keeps
        // it from blaming Brave.
        r.searches += 1;
        let query = match extract_query(args) {
            Ok(q) => q,
            Err(sentinel) => {
                (r.search_errors, r.malformed) = (r.search_errors + 1, r.malformed + 1);
                return (sentinel, true);
            }
        };
        let (found, reached_brave) = self.tool.search(
            &query,
            self.deadline.saturating_duration_since(Instant::now()),
        );
        r.brave_calls += u32::from(reached_brave);
        if reached_brave && let Some(p) = self.brave_usd {
            self.bill_usd(p);
        }
        match &found {
            Found::Text(_, urls) => {
                r.grounded += 1;
                // Deduplicated and capped across the member, not per search.
                for u in urls {
                    if r.sources.len() < super::search::MAX_SOURCES && !r.sources.contains(u) {
                        r.sources.push(u.clone());
                    }
                }
            }
            Found::Empty => {}
            Found::Failed(_) => r.search_errors += 1,
            Found::Unavailable(s) => {
                (r.search_errors, r.search_status) = (r.search_errors + 1, Some(*s));
            }
        }
        let failed = found.is_error();
        let rendered = found.render();
        self.trace.emit(
            "search",
            || json!({"query": query, "chars": rendered.len(), "ok": !failed}),
        );
        (rendered, failed)
    }
}

/// The next `max_tokens` after a truncation: ×1.5, at least +1 (`1 * 3 / 2` is 1), at most `cap`.
pub fn escalate(cur: u32, cap: u32) -> u32 {
    cur.max(1)
        .saturating_mul(3)
        .div_euclid(2)
        .max(cur.saturating_add(1))
        .min(cap)
}

impl Member<'_> {
    /// Offers `text` as the best draft and flushes it to `partial`, for every billed turn,
    /// discarded ones included. Returns whether it became the draft, and whether the file now
    /// holds it, which step 5 needs to say what `partial` is a copy of. `unfinished`: discarded
    /// or cut off.
    fn keep_draft(
        &self,
        r: &mut Report,
        text: &str,
        reasoning_only: bool,
        unfinished: bool,
    ) -> (bool, bool) {
        let (new, held) = (text.chars().count(), r.text.chars().count());
        // An answer outranks a scratchpad, a publishable turn an unfinished one, and then the
        // longer wins. The floor is only emptiness: a *short* answer still has to win, because
        // step 6's re-prompt is what turns it into a full review.
        let rank = |len: usize, scratchpad: bool, unfinished: bool| {
            (!scratchpad, !unfinished && len >= self.min_reply_chars, len)
        };
        let held_unfinished = r.draft_provisional || r.draft_truncated;
        let better = new > 0
            && (held == 0
                || rank(new, reasoning_only, unfinished)
                    > rank(held, r.draft_reasoning_only, held_unfinished));

        // On its own rule, the file never shrinking; the raw count only pre-filters, since
        // scrubbing can only shorten.
        let wrote = new > r.flushed && self.flush(r, text);
        if !better {
            r.draft_flushed &= !wrote;
            return (false, wrote);
        }
        (r.draft_flushed, r.text) = (wrote, text.to_string());
        (true, wrote)
    }

    /// Writes `text`, scrubbed, to `partial` when it is longer than what is there, and records
    /// the length only once written. Returns whether the file now holds `text`.
    fn flush(&self, r: &mut Report, text: &str) -> bool {
        let body = super::util::redact(&super::util::sanitize(text), self.secrets);
        let len = body.chars().count();
        if len <= r.flushed {
            return false;
        }
        // Counted every time, said once per member.
        if let Err(e) = super::output::write_private(&self.partial, &body) {
            if r.write_errors == 0 {
                note!("warning: {e}");
            }
            r.write_errors += 1;
            return false;
        }
        r.flushed = len;
        true
    }

    /// Bounds one billed generation, retried ones included, adds it to the member and charges
    /// the panel: the reported price, else one computed from our rate — `run::price`'s split, so
    /// the cap is checked against what the manifest publishes.
    fn account(&self, r: &mut Report, t: &Turn, ceiling: u64) {
        // Output against the legs the cap permits, input against the bytes we could have sent.
        let u = t
            .usage
            .clamp_completion(self.params.max_tokens)
            .clamp_prompt(ceiling)
            .clamp_cost()
            .metered(!t.text.is_empty() || !t.calls.is_empty());
        r.usage.add(&u);
        r.peak_round = r.peak_round.max(u.completion);
        // Unlike `price`, a turn reporting no prompt tokens is still charged: charging errs
        // towards stopping.
        if let Some(usd) = u
            .floor()
            .or_else(|| self.rate.map(|rate| super::run::computed_usd(&u, rate)))
        {
            self.bill_usd(usd);
        }
    }

    /// One charge against the panel's running total, in dollars.
    fn bill_usd(&self, usd: f64) {
        if usd.is_finite() {
            // Saturating: a wrapping add would reopen the cap after an absurd charge.
            self.spent
                .fetch_update(
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                    |cur| Some(cur.saturating_add(micro_usd(usd))),
                )
                .ok();
        }
    }
}

/// The panel's spend so far, in dollars.
fn spent_usd(spent: &std::sync::atomic::AtomicU64) -> f64 {
    spent.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6
}

/// Whether the provider said our `previous_response_id` is gone. Read from the code slot that
/// ends `wire::err_msg`'s `"{message} ({code})"`, so prose in the message cannot trigger a
/// replay that re-uploads the whole conversation.
pub(super) fn lost_chain(msg: &str) -> bool {
    msg.ends_with("(previous_response_not_found)")
}

/// Jitter over an exponential base, capped at `BACKOFF_CAP_MS`: only the delay we invent, never
/// one a server asked for. The clock is entropy enough to decorrelate threads.
pub(super) fn backoff(attempt: u32) -> Duration {
    let ceiling = (BASE_MS << attempt.min(5)).min(BACKOFF_CAP_MS);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos() as u64);
    Duration::from_millis((BASE_MS + nanos % ceiling).min(BACKOFF_CAP_MS))
}

/// Cheap when tracing is off: the payload closure runs only when somebody is listening.
impl Trace<'_> {
    pub fn emit(&self, kind: &'static str, body: impl FnOnce() -> serde_json::Value) {
        if let Some(tx) = self.tx {
            let event = Event {
                model: self.model.clone(),
                kind,
                body: body(),
            };
            // `try_send`, never `send`. See `Trace::dropped`.
            if tx.try_send(event).is_err() {
                self.dropped
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}
