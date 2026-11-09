//! `bx council`: ask N models the same grounded question in parallel, gate every reply, publish
//! what passes.
//!
//! The rule that shapes everything below: a panel that silently shrinks is the bug. Every failure
//! is named, every paid-for reply is written to disk even when rejected, and the manifest says
//! what to do next.

/// Writes one sanitized line, never panicking: a closed pipe must not abort a paid run. Sanitized
/// here, at the sink, because most lines carry untrusted text and error paths are easy to miss.
macro_rules! emit {
    ($stream:expr, $($t:tt)*) => {{
        use std::io::Write as _;
        let mut line = crate::council::util::sanitize(&format!($($t)*));
        line.push('\n');
        $stream.write_all(line.as_bytes()).ok();
    }};
}

/// One line to stdout.
macro_rules! say {
    ($($t:tt)*) => { emit!(std::io::stdout(), $($t)*) };
}

/// One line to stderr: progress, warnings and errors.
macro_rules! note {
    ($($t:tt)*) => { emit!(std::io::stderr(), $($t)*) };
}

// After the macros: `macro_rules!` is textually scoped.
mod cfg;
mod gate;
mod member;
mod output;
mod resolve;
mod run;
mod search;
mod sub;
mod task;
mod types;
/// Crate-wide: `api::Resp::retry_after` reads an HTTP-date with it.
pub(crate) mod util;
mod wire;

use clap::Args;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::mpsc;

use types::Effort;

/// 8 MiB on every transport: `Content-Length` is the server's word. Not a memory budget, since a
/// parsed `Value` costs many times the bytes; it cannot go below a `MAX_ECHO_BYTES` echo.
pub const MAX_BODY: u64 = 8 * 1024 * 1024;
/// The queue the writer drains. When it is full, trace events are dropped and counted, and a
/// finished member's records wait for room.
const QUEUE: usize = 64;

pub struct Event {
    pub model: String,
    pub kind: &'static str,
    pub body: Value,
}

pub struct Trace<'a> {
    pub tx: Option<&'a mpsc::SyncSender<Event>>,
    pub model: String,
    /// Trace records the queue had no room for. Dropped and counted, never waited on: a writer
    /// stuck on an undrained stderr must not block a worker outside its deadline checks.
    pub dropped: &'a std::sync::atomic::AtomicU32,
}

// ── CLI ──────────────────────────────────────────────────────────────

/// Shared by -h and --help: `--task` has no possible values and a sub-command is not an option,
/// so the options table cannot show either. A macro because `concat!` takes only literals.
macro_rules! lists {
    () => {
        "\
TASKS (--task)  review, headed A-E ending in VERDICT: design-review (default) · code-review ·
                audit (every line; 2400 s clock, dearer) · second-pass (checks round one's fixes)
                answer, from sources: ask · decide · fact-check
                yours: any .md in the tasks dir; `bx council tasks` describes all, names the dir
SUB-COMMANDS    doctor (= models) · tasks · ls (20 newest) · status|findings [RUN] ·
                useful-yes|useful-no [RUN] records a verdict (`useful` refuses) ·
                set-key stores model keys · prune deletes runs over 30 days old that finished
                or hold no paid output, never a live one (prune --dry-run lists them)
                These take the subject's place; to review one of these words, pass @FILE.
EXIT            0 quorum met (whatever else failed) · 2 refused, nothing paid for ·
                quorum missed: 3 auth · 4 rate limit · 5 provider · 7 --max-usd ·
                8 out of credit · 6 mixed causes · the free pre-flight also exits
                3 (key rejected) or 8 (no balance), nothing paid for"
    };
}

/// -h only; --help opens with its own START.
const AFTER_HELP: &str = concat!(
    "\
START   bx council doctor                keys, panels, prices; exit 2 names what is missing
        bx council @design.md --dry-run  the plan and your past cost; spends nothing
        bx council @design.md            a few dollars, ~15 min, blocks; --help: how to poll
        bx council \"Shard by tenant or by region?\" --task decide
        git diff main | bx council - --task code-review --max-usd 10
",
    lists!()
);

/// --help: the lists alone.
const AFTER_LONG_HELP: &str = lists!();

#[derive(Args)]
#[command(after_help = AFTER_HELP, after_long_help = AFTER_LONG_HELP)]
pub struct CouncilArgs {
    /// The question: @FILE, - for stdin, or quoted text
    ///
    /// @FILE and - take up to 1 MiB; pass anything long that way, as the OS limits one argument.
    /// A sub-command's word runs it; with -m it is refused. Without -m, any other single ASCII
    /// word is refused too. To review a word, pass it as @FILE.
    subject: Option<String>,

    /// Run for status, findings, useful-yes, useful-no [default: newest]
    ///
    /// A directory, or a run id under the state directory. Same as --out; not both.
    // After a subject, a second positional is a typo or a split subject: refused, not swallowed.
    target: Option<String>,

    // Never legal. Collected so the refusal is this module's: it names the quoting fix, in the
    // caller's format.
    #[arg(hide = true)]
    rest: Vec<String>,

    /// What the panel does, by exact name; see TASKS below [default: design-review]
    ///
    /// design-review is a red-team, not an answer; council.json's default_task replaces it. A
    /// task file may pin panel, effort, clock and search budget; flags win. `bx council tasks`
    /// describes each and prints the file format.
    #[arg(long)]
    task: Option<String>,

    /// Panel: aliases, slugs or @panel, comma-separated [default: @default]
    ///
    /// Repeatable, up to 32. Aliases: opus, astra, sol, kimi, glm, qwen, minimax, deepseek, luna,
    /// glm-flash, deepseek-flash; `bx council doctor` prints those and the panels in effect.
    /// Prefix direct: or openrouter: to pick a transport. Before anything is paid, each model is
    /// checked against its provider's roster: the name, and effort and output limit where the
    /// roster publishes them. An unreadable roster skips the check.
    #[arg(short = 'm', long = "model", value_delimiter = ',')]
    models: Vec<String>,

    /// Run directory to write, or for a sub-command to read [default: new, under the state dir]
    ///
    /// Name one to poll a backgrounded run by. A bare run id resolves under the state directory.
    /// Refused if it already holds a run's files or a live run; an empty or new directory is
    /// fine. history.jsonl stays in the state directory.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Stdout format; on a run, json is the whole manifest
    #[arg(long, default_value = "json", value_parser = ["json", "text"])]
    format: String,

    /// Progress lines on stderr; silence with --progress=false
    ///
    /// Off, stderr never names the run directory: pass --out to know it up front.
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          default_value = "true", value_parser = clap::builder::BoolishValueParser::new())]
    progress: bool,

    /// Delivered replies needed for exit 0
    ///
    /// At 1, exit 0 can hide lost members: read quorum.delivered and hint. Set it to the panel
    /// size to fail on any loss; above that it is refused.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..))]
    min_quorum: u16,

    /// Wall clock per member, in seconds [default: 900, or the task's; audit: 2400]
    ///
    /// The usual way a member is lost. What it wrote is kept, and re-running it without raising
    /// this usually loses it again. Below full --concurrency a run can last
    /// ceil(members / concurrency) times this.
    // `Option`, so a task's clock applies unless the flag is given. The range keeps out 0, which
    // sends nothing, and the `Instant + Duration` overflow.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=run::MAX_SECONDS))]
    max_seconds: Option<u64>,

    /// Members in flight at once [default: all]
    ///
    /// Lower it only for a provider's concurrency limit: queued members start as others finish,
    /// so the run can outlast one --max-seconds.
    // 0 is not a concurrency.
    #[arg(long, value_parser = clap::value_parser!(u16).range(1..))]
    concurrency: Option<u16>,

    /// Send every member over one transport
    ///
    /// Overrides a slug's direct: or openrouter: prefix. A model the transport's roster does not
    /// list is refused up front.
    #[arg(long, value_parser = ["openrouter", "anthropic", "openai"])]
    via: Option<String>,

    /// Print the plan and its cost; spend nothing
    ///
    /// No model is called. Size --max-usd above cost.observed's max (what past runs of these
    /// models were charged for inference; shown from three runs on) plus searches. worst_usd
    /// covers only members priced in advance, never OpenRouter ones. With prune, set-key,
    /// useful-yes or useful-no: shows what would change.
    // Reads history, so two identical invocations need not print the same.
    #[arg(long)]
    dry_run: bool,

    /// Stop starting rounds once reported spend reaches this, in USD
    ///
    /// Checked before each round, so overshoot is up to one in-flight round per member. Counts
    /// searches and each reply's reported or rated cost; a direct member with no rate is refused
    /// up front (add one under "rates" in council.json). Paid text is still written. A capped
    /// run exits 0 if the quorum delivered, 7 if the cap alone cost it. Default: council.json's
    /// max_usd.
    #[arg(long)]
    max_usd: Option<f64>,

    /// Output tokens per round, before escalation [task default: 64000]
    ///
    /// A truncated round escalates, up to 128000 or the model's lower published ceiling.
    // 0 would be raised to the wire's floor; above 128000 every wire 400s.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=128_000))]
    max_tokens: Option<u32>,

    /// Reasoning effort [task default: high]
    ///
    /// Lower it when the run says a member spent its output on reasoning and answered nothing.
    /// none turns reasoning off on OpenRouter and OpenAI (models that require it refuse); on
    /// Anthropic it is the lowest effort, adaptive thinking still on. minimal is sent as low.
    #[arg(long, value_parser = clap::builder::PossibleValuesParser::new(Effort::ALL))]
    effort: Option<String>,

    /// Brave lookups per member [task default: 15]
    ///
    /// Each is a billed Brave request on the member's clock; at the limit a member answers from
    /// the pages it has.
    // 0 would attach a tool that can only answer "search budget exhausted".
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=1_000))]
    max_searches: Option<u32>,

    /// Model turns per member [task default: max-searches + 6]
    ///
    /// Unset, it follows --max-searches, leaving six turns for escalation and the answer.
    // 0 is the internal "unset".
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=1_000))]
    max_steps: Option<u32>,

    /// Whether members may search Brave [task default: auto]
    ///
    /// none: answer from model knowledge alone, and no Brave key is needed. A Brave refusal
    /// (401, 402, 403) ends search for the whole run; members then answer from what they have.
    #[arg(long, value_parser = ["auto", "none"])]
    tools: Option<String>,

    /// Also write trace.jsonl and per-member transcripts
    ///
    /// trace.jsonl: queries, retries, escalations. NN-slug.transcript.json: each member's whole
    /// conversation, subject and search results included. Records never delay a reply: under
    /// load they are dropped and counted (trace_dropped).
    #[arg(long)]
    trace: bool,
}

/// Sub-commands share the subject slot; clap cannot say whether `--` was used, so one of these
/// words is reviewed only as @FILE, and with `-m` it is refused as a mistyped sub-command.
const RESERVED: &[&str] = &[
    "ls",
    "status",
    "tasks",
    "prune",
    "models",
    "doctor",
    "set-key",
    "findings",
    "useful-yes",
    "useful-no",
    // Only to refuse, naming `useful-yes` and `useful-no`.
    "useful",
];

impl CouncilArgs {
    /// The run a read sub-command names, by either spelling. Both at once is refused, printing
    /// both, since they may disagree.
    pub fn run_target(&self) -> Result<Option<std::path::PathBuf>, String> {
        let given = match (&self.out, &self.target) {
            (Some(o), Some(t)) => {
                return Err(format!(
                    "--out {} and `{t}` name the same thing; pass one of them",
                    o.display()
                ));
            }
            (Some(o), None) => o.clone(),
            (None, Some(t)) => std::path::PathBuf::from(t),
            (None, None) => return Ok(None),
        };
        // Empty is neither a run nor "the newest", which omitting it means.
        if given.as_os_str().is_empty() {
            return Err(
                "the run is an empty string; name a run id or a directory, or omit it for the \
                 newest run"
                    .into(),
            );
        }
        Ok(Some(given))
    }

    /// For `main.rs`, which refuses bad global flags in the caller's format.
    pub fn format(&self) -> &str {
        &self.format
    }
}

/// The sub-commands that act on one named run. Any other reserved word would silently drop a
/// run named with it; on `prune`, reaping the default scope instead. `useful` is here because
/// its refusal tells the caller to name a run.
const TARGETED: &[&str] = &["status", "findings", "useful-yes", "useful-no", "useful"];

/// The sub-command in the subject slot, if any. `main.rs` asks before resolving the Brave key,
/// which no sub-command needs.
pub fn subcommand(a: &CouncilArgs) -> Option<&'static str> {
    if !a.models.is_empty() {
        return None;
    }
    let w = a.subject.as_deref()?;
    RESERVED.iter().copied().find(|&r| r == w)
}

// ── Entry ────────────────────────────────────────────────────────────

/// The error line on stderr and, unless `--format text`, its JSON on stdout.
fn fail(machine: bool, kind: &str, exit: i32, message: &str) {
    note!("error: {message}");
    if machine {
        say!(
            "{}",
            serde_json::json!({"error": {"kind": kind, "exit": exit, "message": message}})
        );
    }
}

/// A refusal raised before `main`: `main.rs` validates the global flags for every command.
/// `format` is `None` for a non-council command, which answers in prose only.
pub fn refuse(format: Option<&str>, message: &str) -> ! {
    fail(
        format.is_some_and(|f| f != "text"),
        resolve::INVALID_REQUEST,
        2,
        message,
    );
    std::process::exit(2)
}

pub fn main(a: CouncilArgs, brave_base: &str, brave_key: &str, timeout: u64) -> ! {
    // Read before `a` is moved: a JSON caller gets its failure in JSON too.
    let machine = a.format != "text";
    let code = match subcommand(&a) {
        // Both spellings: `prune --out <run>` once ignored the run and deleted every old one.
        Some(w) if (a.target.is_some() || a.out.is_some()) && !TARGETED.contains(&w) => {
            Err(resolve::Refusal::from(format!(
                "`{w}` takes no run, by either spelling{}",
                if w == "prune" && !a.dry_run {
                    " — run `bx council prune --dry-run` to see what it would remove"
                } else {
                    ""
                }
            )))
        }
        Some(_) if !a.rest.is_empty() => Err(resolve::Refusal::from(format!(
            "unexpected third argument `{}`: a sub-command takes one run",
            a.rest[0]
        ))),
        Some(w) => sub::dispatch(w, &a, brave_key).map_err(resolve::Refusal::from),
        // A subject takes one positional; the usual cause is a split subject or a typo'd
        // sub-command.
        None => match a.target.as_deref() {
            Some(extra) => Err(resolve::Refusal::from(format!(
                "unexpected second argument `{extra}`: a subject takes one positional. {}",
                // After `--` every flag is a positional, and quoting is not the fix.
                if extra.starts_with('-') {
                    "Everything after `--` is a positional, so put your flags before it"
                } else {
                    "Quote it if it contains spaces, or name a sub-command first"
                }
            ))),
            None => run::run(a, brave_base, brave_key, timeout),
        },
    };
    std::process::exit(match code {
        Ok(c) => c,
        // The refusal carries its own code: 2, or 3 and 8 when a pre-flight rejects a key or
        // finds no credit.
        Err(e) => {
            fail(machine, e.kind, e.exit, &e.message);
            e.exit
        }
    })
}

#[cfg(test)]
mod tests;
