//! The types shared by the loop, the gate and the three wires.
//!
//! Only `Item::Assistant` is opaque: every provider needs its own turn back byte for byte, and
//! `serde_json::Value` re-sorts keys and re-renders numbers. Our own items stay typed.

use serde_json::value::RawValue;
use std::borrow::Cow;
use std::time::Duration;

/// Reasoning effort. `as_str` indexes `ALL`, so the variant order *is* the array order.
#[derive(Clone, Copy, PartialEq)]
pub enum Effort {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl Effort {
    pub const ALL: &'static [&'static str] =
        &["none", "minimal", "low", "medium", "high", "xhigh", "max"];

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "none" => Self::Off,
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::XHigh,
            "max" => Self::Max,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        Self::ALL[self as usize]
    }
}

/// Token and money counters. Every counter but `prompt` and `completion` is `Option`: a missing
/// count must never read as zero.
#[derive(Clone, Copy, Default)]
pub struct Usage {
    pub prompt: u64,
    pub completion: u64,
    pub reasoning: Option<u64>,
    pub cache_read: Option<u64>,
    pub cache_write: Option<u64>,
    /// OpenRouter's `usage.cost`: what the account was charged.
    pub cost: Option<f64>,
    pub is_byok: Option<bool>,
    pub upstream: Option<f64>,
    /// Some billed turn reported no price, so the total is a subtotal.
    pub cost_incomplete: bool,
    /// Some price was clamped by [`Usage::clamp_cost`]. The total sits above the true charge, so
    /// it is unknown, not even a floor.
    pub cost_clamped: bool,
}

impl Usage {
    /// Bounds a reported output before it is summed or priced, at `cap` × [`PARALLEL_LEGS`]:
    /// parallel-reasoning models honestly bill past their cap. Reasoning is part of the output.
    pub fn clamp_completion(mut self, cap: u32) -> Usage {
        self.completion = self
            .completion
            .min(u64::from(cap).saturating_mul(PARALLEL_LEGS));
        self.reasoning = self.reasoning.map(|r| r.min(self.completion));
        self
    }

    /// Bounds a reported price at [`MAX_TURN_USD`]: a finite `1e305` would otherwise saturate the
    /// shared spend and pin the whole panel at `--max-usd`. Clamping tokens cannot reach a price
    /// OpenRouter reports rather than derives.
    pub fn clamp_cost(mut self) -> Usage {
        let over = |v: &Option<f64>| v.is_some_and(|v| v > MAX_TURN_USD);
        // `bill` still charges the clamp; `run::price` publishes nothing for it.
        self.cost_clamped |= over(&self.cost) || over(&self.upstream);
        self.cost = self.cost.map(|v| v.min(MAX_TURN_USD));
        self.upstream = self.upstream.map(|v| v.min(MAX_TURN_USD));
        self
    }

    /// Bounds reported input, which `computed_usd` prices, by the conversation's byte ceiling:
    /// bytes bound tokens. Not the body's length, which under chaining is one nudge. The cache
    /// counters are shares of `prompt`.
    pub fn clamp_prompt(mut self, ceiling: u64) -> Usage {
        self.prompt = self.prompt.min(ceiling);
        self.cache_read = self.cache_read.map(|v| v.min(self.prompt));
        self.cache_write = self.cache_write.map(|v| v.min(self.prompt));
        self
    }

    pub fn add(&mut self, o: &Usage) {
        self.prompt = self.prompt.saturating_add(o.prompt);
        self.completion = self.completion.saturating_add(o.completion);
        for (dst, src) in [
            (&mut self.reasoning, o.reasoning),
            (&mut self.cache_read, o.cache_read),
            (&mut self.cache_write, o.cache_write),
        ] {
            if let Some(v) = src {
                *dst = Some(dst.unwrap_or(0).saturating_add(v));
            }
        }
        for (dst, src) in [(&mut self.cost, o.cost), (&mut self.upstream, o.upstream)] {
            if let Some(v) = src {
                *dst = Some(dst.unwrap_or(0.0) + v);
            }
        }
        // Sticky: one such round taints the member's total.
        self.cost_incomplete |= o.cost_incomplete;
        self.cost_clamped |= o.cost_clamped;
        // A real OR, not `Option::or`: one BYOK round makes the member BYOK.
        self.is_byok = match (self.is_byok, o.is_byok) {
            (Some(a), Some(b)) => Some(a || b),
            (a, b) => a.or(b),
        };
    }

    /// The price to publish. Under BYOK it is OpenRouter's fee plus `upstream`, what your own key
    /// was charged, and needs both; otherwise `cost` alone, since `upstream` is BYOK-only. A
    /// missing figure is unknown, never zero.
    pub fn spent(&self) -> Option<f64> {
        match self.is_byok {
            Some(true) => match (self.cost, self.upstream) {
                (Some(fee), Some(inference)) => Some(fee + inference),
                _ => None,
            },
            _ => self.cost,
        }
    }

    /// What `--max-usd` is charged: every half of a BYOK bill that was reported. Unlike
    /// [`Usage::spent`], half a bill counts, because charging errs towards stopping.
    pub fn floor(&self) -> Option<f64> {
        match self.is_byok {
            Some(true) => (self.cost.is_some() || self.upstream.is_some())
                .then(|| self.cost.unwrap_or(0.0) + self.upstream.unwrap_or(0.0)),
            _ => self.cost,
        }
    }

    /// Marks the bill incomplete when a turn that `generated` text or a call reported no usage
    /// at all: an absent `usage` parses to zeros.
    pub fn metered(mut self, generated: bool) -> Self {
        if generated && self.prompt == 0 && self.completion == 0 && self.cost.is_none() {
            self.cost_incomplete = true;
        }
        self
    }
}

/// Why the turn ended. Only `Done` is usable; whether it asked for tools is `Turn.calls`.
pub enum Stop {
    Done,
    Truncated,
    /// The input filled the context window: no output cap can fix it, so it never escalates.
    OverContext,
    Refused,
    /// Transient; the loop sleeps and re-sends.
    Retry(String),
    /// Terminal; the member is over.
    Fatal(String),
}

/// `name` says which tool was asked for; the arguments alone cannot.
pub struct Call {
    pub id: String,
    pub name: String,
    pub args: String,
}

pub struct Turn {
    pub text: String,
    pub calls: Vec<Call>,
    pub stop: Stop,
    pub usage: Usage,
    /// The provider's own turn, verbatim. `None` on any stop the loop cannot echo.
    pub raw: Option<Box<RawValue>>,
    /// OpenAI's `response.id`, set only when the response says `completed`.
    pub chain: Option<String>,
    pub served_model: Option<String>,
    pub provider: Option<String>,
    /// The HTTP status, so nothing re-derives it from provider text.
    pub status: Option<u16>,
    /// `text` came from the reasoning channel: a billed draft, never an answer.
    pub reasoning_only: bool,
}

impl Turn {
    /// A turn with no usable answer.
    pub fn stopped(stop: Stop) -> Self {
        Turn {
            text: String::new(),
            calls: Vec::new(),
            stop,
            usage: Usage::default(),
            raw: None,
            chain: None,
            served_model: None,
            provider: None,
            status: None,
            reasoning_only: false,
        }
    }

    /// The status the turn came from; a stop that never reached a server has none.
    pub fn at(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    /// The usage a stopped turn still consumed: an error on a 200 is billed too.
    pub fn billed(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }

    /// The paid text a stopped turn still carried, kept as a draft, never an answer. Its kind
    /// travels with it, so a scratchpad cannot outrank a real answer.
    pub fn salvaged(mut self, text: Option<String>, reasoning_only: bool) -> Self {
        self.text = text.unwrap_or_default();
        self.reasoning_only = reasoning_only;
        self
    }
}

pub enum Item {
    /// The task and subject: index 0, byte-stable, the cache prefix.
    Seed(Cow<'static, str>),
    Assistant(Box<RawValue>),
    Result {
        id: String,
        body: String,
        is_error: bool,
    },
    Nudge(Cow<'static, str>),
}

/// One echoed turn: half of `MAX_BODY`, far more than the largest generation can spell.
const MAX_ECHO_BYTES: usize = 4 * 1024 * 1024;

/// How many generations one request may bill for: parallel-reasoning models report the sum of
/// their threads, with headroom over the four they run.
pub const PARALLEL_LEGS: u64 = 8;

/// The most one generation may be reported to have cost. See [`Usage::clamp_cost`].
const MAX_TURN_USD: f64 = 10_000.0;

/// The most one member's conversation may occupy, whatever its flags say. `Member::run` sizes
/// the budget from caller-supplied steps, cap and searches; this keeps their product a bound.
const MAX_CONV_BYTES: usize = 32 * 1024 * 1024;

/// Whether a provider's turn is a container every wire can splice back: a non-empty object, or
/// a non-empty array whose every element is one. Any empty or non-object element 400s every
/// remaining round.
fn usable(t: &str) -> bool {
    let t = t.trim();
    if t.len() > MAX_ECHO_BYTES {
        return false;
    }
    match t.as_bytes().first() {
        Some(b'[') => serde_json::from_str::<Vec<&RawValue>>(t)
            .is_ok_and(|v| !v.is_empty() && v.iter().all(|e| filled_object(e.get()))),
        Some(b'{') => filled_object(t),
        _ => false,
    }
}

/// A JSON object with at least one key, by one shallow parse that never re-renders the bytes.
fn filled_object(t: &str) -> bool {
    // Owned keys: a `&str` cannot borrow one spelled with an escape.
    serde_json::from_str::<std::collections::BTreeMap<String, serde::de::IgnoredAny>>(t.trim())
        .is_ok_and(|m| !m.is_empty())
}

#[derive(Default)]
pub struct Conv {
    pub items: Vec<Item>,
    /// OpenAI's `previous_response_id`.
    pub chain: Option<String>,
    /// How many items the server already holds. Read only while `!params.stateless`.
    pub sent: usize,
    /// Bytes of every item stored so far, against the budget below.
    bytes: usize,
    /// What every item together may occupy, in bytes: every stored turn is re-sent every round,
    /// so the upload is otherwise the triangular sum. Set by `Member::run`.
    budget: usize,
}

impl Conv {
    /// Clamps `budget` to [`MAX_CONV_BYTES`] past the seed, never below it, so round one can
    /// always be sent. The seed counts against the budget.
    pub fn new(seed: String, budget: usize) -> Self {
        let ceiling = seed.len().saturating_add(MAX_CONV_BYTES);
        let mut c = Conv {
            budget: budget.min(ceiling),
            ..Conv::default()
        };
        c.push(Item::Seed(Cow::Owned(seed)));
        c
    }

    /// Echoes the provider's turn if [`usable`]; if it chained, marks everything so far
    /// server-held. Returns whether it was stored: results must not follow a turn that was not.
    pub fn advance(&mut self, raw: Option<Box<RawValue>>, chain: Option<String>) -> bool {
        let offered = raw.is_some();
        let stored = if let Some(r) = raw.filter(|r| usable(r.get())) {
            self.push(Item::Assistant(r));
            true
        } else {
            false
        };
        // Dropping a turn also drops the chain: the server would still hold its unanswered calls.
        if offered && !stored {
            self.go_stateless();
            return false;
        }
        if chain.is_some() {
            (self.chain, self.sent) = (chain, self.items.len());
        }
        stored
    }

    /// The only way an item enters, so the running total cannot be skipped.
    pub fn push(&mut self, item: Item) {
        self.bytes = self.bytes.saturating_add(match &item {
            Item::Seed(s) | Item::Nudge(s) => s.len(),
            Item::Assistant(r) => r.get().len(),
            Item::Result { body, .. } => body.len(),
        });
        self.items.push(item);
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Checked before a round is sent, so the round that would overflow is never paid for.
    pub fn full(&self) -> bool {
        self.bytes >= self.budget
    }

    /// `Usage::clamp_prompt`'s input ceiling: the conversation sent is under `budget`, and the
    /// echo bound is slack.
    pub fn ceiling(&self) -> u64 {
        self.budget.saturating_add(MAX_ECHO_BYTES) as u64
    }

    /// The server holds nothing any more, so `chain` and `sent` reset together: a suffix with no
    /// chain has no beginning.
    pub fn go_stateless(&mut self) {
        self.chain = None;
        self.sent = 0;
    }

    /// The cache breakpoint: our newest item, so the whole previous prefix is a cache read.
    pub fn mark(&self, on: bool) -> Option<usize> {
        on.then(|| {
            self.items
                .iter()
                .rposition(|i| !matches!(i, Item::Assistant(_)))
        })
        .flatten()
    }

    /// The same among text items only: OpenRouter's providers do not all take array-shaped
    /// `tool` content, which a marked tool result needs.
    pub fn mark_text(&self, on: bool) -> Option<usize> {
        on.then(|| {
            self.items
                .iter()
                .rposition(|i| matches!(i, Item::Seed(_) | Item::Nudge(_)))
        })
        .flatten()
    }
}

/// Everything a wire needs that is not conversation.
pub struct Params {
    pub model: String,
    /// The panel framing only; the task is `Item::Seed`, inside the cached prefix.
    pub system: String,
    pub max_tokens: u32,
    pub effort: Effort,
    pub tools: bool,
    pub cache: bool,
    pub session: String,
    /// Set after a lost chain, and never cleared: one recovery is not evidence the chain is back.
    pub stateless: bool,
    /// `council.zdr`: OpenRouter's `provider.zdr`, and on OpenAI `store: false` with no chain.
    pub zdr: bool,
    /// `council.providers`, sent as OpenRouter's `only`; empty leaves the choice to the router.
    pub providers: Vec<String>,
}

/// Why a reply was not published, in the order the gate checks: transport signals first.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reason {
    ProviderError,
    Refused,
    Truncated,
    /// Stopped by `--max-seconds`: billed and out of clock, so retryable.
    Budget,
    /// The panel reached `--max-usd`.
    Cost,
    /// The conversation outgrew its byte budget: our limit, not the provider's.
    ContextBudget,
    Empty,
    SearchFailed,
    NoAnswer,
    /// The reply arrived but could not be written: a local fault a retry cannot repair.
    WriteFailed,
}

/// Whether re-sending the same request could answer differently: the one rule behind both
/// `wire::by_status` and [`Reason::action`].
pub fn retryable(status: u16) -> bool {
    matches!(status, 408 | 409 | 429 | 500..=599)
}

impl Reason {
    pub fn name(self) -> &'static str {
        match self {
            Self::ProviderError => "provider_error",
            Self::Refused => "refused",
            Self::Truncated => "truncated",
            Self::Budget => "budget",
            Self::ContextBudget => "context_budget",
            Self::Cost => "max_usd",
            Self::Empty => "empty",
            Self::SearchFailed => "search_failed",
            Self::NoAnswer => "no_answer",
            Self::WriteFailed => "write_failed",
        }
    }

    /// What the caller should do next, which `hint` turns into a command. The status decides a
    /// provider error: a 503 may clear, a 401 never will.
    pub fn action(self, status: Option<u16>) -> &'static str {
        match self {
            // None of these changes on a re-send; a refusal re-sends the declined prompt.
            Self::Empty | Self::Cost | Self::Truncated | Self::WriteFailed | Self::Refused => {
                "none"
            }
            Self::ProviderError => match status {
                Some(s) if !retryable(s) => "none",
                _ => "retry",
            },
            // A status here is one Brave latched, a spent quota's 429 included: none clears.
            Self::SearchFailed if status.is_some() => "none",
            Self::SearchFailed => "retry",
            Self::Budget | Self::ContextBudget | Self::NoAnswer => "retry",
        }
    }
}

pub enum Outcome {
    Delivered {
        partial_grounding: bool,
    },
    Rejected {
        reason: Reason,
        error: String,
        /// The HTTP status, carried: `error` is provider text, and this decides the exit code.
        status: Option<u16>,
    },
}

impl Outcome {
    pub fn delivered(&self) -> bool {
        matches!(self, Outcome::Delivered { .. })
    }
}

/// One row of the manifest, and the only thing the renderers see.
pub struct Member {
    pub model: String,
    pub served_model: Option<String>,
    pub provider: Option<String>,
    pub transport: &'static str,
    pub outcome: Outcome,
    pub text: String,
    pub path: std::path::PathBuf,
    pub usage: Usage,
    pub cost_usd: Option<f64>,
    /// How `cost_usd` was arrived at: `reported` or `computed` (either `_partial` when a billed
    /// round went unpriced), `byok_upstream`, `unbilled` when nothing was sent, or `unknown`.
    pub cost_source: &'static str,
    pub rounds: u32,
    pub retries: u32,
    pub searches: u32,
    /// The `searches` that reached Brave: the billable count.
    pub brave_calls: u32,
    pub grounded: u32,
    pub search_errors: u32,
    pub sources: Vec<String>,
    pub sources_shared: usize,
    /// Present only when `--trace` wrote one.
    pub transcript: Option<String>,
    pub peak_round: u64,
    pub escalations: u32,
    /// The cap the member started under.
    pub max_tokens: u32,
    /// The largest cap a request carried, after escalation.
    pub max_tokens_reached: u32,
    pub write_errors: u32,
    pub elapsed: Duration,
    pub terminated_by: &'static str,
    pub finish: &'static str,
}

impl Member {
    pub fn chars(&self) -> usize {
        self.text.chars().count()
    }
}
