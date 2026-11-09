//! Flags and config into a concrete plan: lookups, validation, the subject and `--out` reads, and
//! the only requests sent before a run, the free pre-flights `check_models` and `check_credit`.

use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;

use super::types::Effort;
use super::wire::Served;
use super::{cfg, task, wire};

/// Where every run's records live, one directory per run, kept until an operator deletes them:
/// `$XDG_STATE_HOME/bx/council/<run-id>/`, plus the `history.jsonl` that outlives them.
pub fn state_dir() -> Result<PathBuf, String> {
    #[cfg(test)]
    if let Some(p) = STATE_DIR_OVERRIDE.with(|o| o.borrow().clone()) {
        return Ok(p);
    }
    // `state_dir()` is None on macOS and Windows; Windows' `data_dir()` roams, so not that.
    dirs::state_dir()
        .or_else(dirs::data_local_dir)
        .map(|d| d.join("bx").join("council"))
        .ok_or_else(|| "cannot determine state directory".into())
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(test)]
    if let Some(p) = HOME_OVERRIDE.with(|o| o.borrow().clone()) {
        return Some(p);
    }
    dirs::home_dir()
}

// Per-thread test overrides: `set_var` races other test threads, and XDG is unread off Linux.
#[cfg(test)]
type Slot = std::thread::LocalKey<std::cell::RefCell<Option<PathBuf>>>;

#[cfg(test)]
thread_local! {
    static HOME_OVERRIDE: std::cell::RefCell<Option<PathBuf>> =
        const { std::cell::RefCell::new(None) };
    static STATE_DIR_OVERRIDE: std::cell::RefCell<Option<PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

/// Points `slot` at `p` for this thread, until the returned guard drops.
#[cfg(test)]
fn override_for_test(slot: &'static Slot, p: &std::path::Path) -> impl Drop {
    struct Reset(&'static Slot);
    impl Drop for Reset {
        fn drop(&mut self) {
            self.0.with(|o| *o.borrow_mut() = None);
        }
    }
    slot.with(|o| *o.borrow_mut() = Some(p.to_path_buf()));
    Reset(slot)
}

#[cfg(test)]
pub(super) fn state_dir_for_test(p: &std::path::Path) -> impl Drop {
    override_for_test(&STATE_DIR_OVERRIDE, p)
}

#[cfg(all(test, unix))]
pub(super) fn home_for_test(p: &std::path::Path) -> impl Drop {
    override_for_test(&HOME_OVERRIDE, p)
}

pub fn panel_slugs(
    c: &cfg::Council,
    cli: &[String],
    from_task: &[String],
) -> Result<Vec<String>, String> {
    let raw: Vec<String> = if !cli.is_empty() {
        cli.to_vec()
    } else if !from_task.is_empty() {
        from_task.to_vec()
    } else if c.panels.contains_key("default") {
        // Expanded below, which refuses it empty.
        vec!["@default".into()]
    } else {
        cfg::DEFAULT_PANEL
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    };
    // `-m 'opus, glm'`: that space is the shell's. Interior whitespace still fails the guard.
    let raw = raw.into_iter().map(|e| e.trim().to_string());
    // Panels first, then every entry through the alias table: a panel's `opus` is `-m opus`.
    let mut out = Vec::new();
    for entry in raw {
        match entry.strip_prefix('@') {
            // An unknown or empty panel is refused, never expanded to a silently smaller panel.
            Some(name) => {
                // As typed first, so a config panel named `sota` is reachable; then canonical, so
                // `@sota` sees a config `default` as `@default` does.
                let members: Vec<String> = match c
                    .panels
                    .get(name)
                    .or_else(|| c.panels.get(cfg::canonical_panel(name)))
                {
                    Some(p) if p.is_empty() => {
                        return Err(format!(
                            "panel '@{name}' in council.json is empty; list its members or \
                             delete it"
                        ));
                    }
                    Some(p) => p.clone(),
                    // There are only a handful, so the refusal lists them.
                    None => cfg::builtin_panel(name)
                        .ok_or_else(|| {
                            let have: std::collections::BTreeSet<String> = c
                                .panels
                                .keys()
                                .map(String::as_str)
                                .chain(cfg::DEFAULT_PANELS.iter().map(|(n, _)| *n))
                                .chain(cfg::PANEL_ALIASES.iter().map(|(a, _)| *a))
                                .map(|n| format!("@{n}"))
                                .collect();
                            let have: Vec<String> = have.into_iter().collect();
                            format!("unknown panel '@{name}' (have: {})", have.join(", "))
                        })?
                        .iter()
                        .map(|s| (*s).to_string())
                        .collect(),
                };
                out.extend(members);
            }
            None => out.push(entry),
        }
    }
    let mut out: Vec<String> = out
        .into_iter()
        // Again: a config panel's members were not trimmed with the entries.
        .map(|e| cfg::alias(c, e.trim()))
        .collect::<Result<_, _>>()?;
    // After the alias table, or an alias whose value is `@fast` reaches the provider as a model.
    if let Some(nested) = out.iter().find(|e| e.starts_with('@')) {
        return Err(format!(
            "'{}' names a panel inside a panel; panels do not nest",
            shown(nested)
        ));
    }
    // Not `dedup`: panel expansion yields separated duplicates, each a second bill.
    let mut seen = std::collections::HashSet::new();
    let asked = out.len();
    out.retain(|s| seen.insert(s.clone()));
    // Several opinions asked for and one paid for is a mistake; one model named once is allowed.
    // After the alias map, so `-m opus,anthropic/claude-opus-5.5` counts as one.
    if out.len() == 1 && asked > 1 {
        return Err(format!(
            "{asked} entries name one model ({}) — name the members you want, or run \
             `bx council doctor` to see the roster and aliases",
            shown(&out[0])
        ));
    }
    // An allow-list: a slug reaches the provider, unsanitized sinks, a `hint` an agent runs, and
    // `history.jsonl` borrowed as `&str` (no `\` or `"`). Emptiness is judged on the bare slug:
    // `direct:` alone would send `"model": ""`.
    if let Some(bad) = out.iter().find(|s| {
        super::output::bare_slug(s).is_empty()
            || !s.chars().all(|c| {
                c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | '/' | ':' | '~')
            })
    }) {
        return Err(format!(
            "invalid model slug {:?} (empty, or outside a-z 0-9 and -._/:~)",
            shown(bad)
        ));
    }
    // Zero retention does not cover OpenRouter's web plugin, and members search Brave already.
    if let Some(s) = out.iter().find(|s| s.split(':').any(|v| v == "online")) {
        return Err(format!(
            "`{}`: `:online` sends the subject to OpenRouter's web plugin, outside zero data \
             retention; drop it, members already search the web",
            shown(s)
        ));
    }
    Ok(out)
}

/// The wire name and the bare slug. A `direct:`/`openrouter:` prefix wins over config, which wins
/// over the default. `direct:` is a preference: a vendor with no direct wire uses OpenRouter.
pub fn transport_for<'a>(
    c: &cfg::Council,
    slug: &'a str,
    via: Option<&'static wire::Wire>,
) -> (&'static str, &'a str) {
    // Stripped even under `--via`, which overrides the routing, not the model id.
    let bare = super::output::bare_slug(slug);
    if let Some(w) = via {
        return (w.name, bare);
    }
    if slug.starts_with("openrouter:") {
        return ("openrouter", bare);
    }
    let vendor = bare.trim_start_matches('~').split('/').next().unwrap_or("");
    // An explicit empty list means no vendor bypasses OpenRouter.
    let direct = |name: &str| match &c.direct {
        None => cfg::DEFAULT_DIRECT.contains(&name),
        Some(list) => list.iter().any(|d| d == name),
    };
    let forced = slug.starts_with("direct:");
    match vendor {
        "anthropic" if forced || direct("anthropic") => ("anthropic", bare),
        "openai" if forced || direct("openai") => ("openai", bare),
        _ => ("openrouter", bare),
    }
}

/// Refuses an empty subject (`"$UNSET"`, an empty file), a lone word naming a path and, unless
/// `-m` was given, any one-word inline one: each buys a panel reviewing nothing. A sub-command's
/// word with `-m` is a typo.
pub fn subject_of(
    arg: Option<&str>,
    max_seconds: u64,
    models_named: bool,
) -> Result<String, String> {
    const MAX: u64 = 1024 * 1024;
    /// The subject read is bounded by `--max-seconds`, and never waits longer than this.
    const MAX_SUBJECT_WAIT: Duration = Duration::from_secs(30);
    let wait = Duration::from_secs(max_seconds).min(MAX_SUBJECT_WAIT);
    let subject = match arg {
        None => {
            return Err("a subject is required (text, @FILE, or -); see `bx council -h`".into());
        }
        Some("-") => read_stdin(MAX, wait)?,
        Some(s) => match s.strip_prefix('@') {
            None if models_named
                && let Some(r) = super::RESERVED
                    .iter()
                    .find(|r| r.eq_ignore_ascii_case(s.trim())) =>
            {
                return Err(format!(
                    "`bx council {r}` takes no -m; to review the word itself, pass it as @FILE"
                ));
            }
            // A lone word naming a path is a forgotten `@`, with or without `-m`.
            None if !s.trim().is_empty()
                && !s.trim().contains(char::is_whitespace)
                && std::path::Path::new(s.trim()).exists() =>
            {
                let w = s.trim();
                return Err(format!(
                    "`{w}` is a path here; to review the file, pass @{w}"
                ));
            }
            // ASCII only: CJK and other scripts are written without spaces.
            None if !models_named
                && !s.trim().is_empty()
                && s.trim().bytes().all(|b| b.is_ascii_graphic()) =>
            {
                return Err(one_word(s.trim()));
            }
            None => s.to_string(),
            // Pipes are accepted: `@<(git diff)` is one. Bounded by `take`, since a FIFO and
            // `/dev/zero` report length 0.
            Some(path) => {
                // Non-blocking, so a FIFO with no writer reads EOF instead of hanging.
                let f = super::util::open_nonblocking(std::path::Path::new(path)).map_err(|e| {
                    let hint = if e.kind() == std::io::ErrorKind::NotFound {
                        " — to review text starting with @, pass it on stdin (-)"
                    } else {
                        ""
                    };
                    format!("{path}: {e}{hint}")
                })?;
                let w = Waiting {
                    inner: f,
                    until: std::time::Instant::now() + wait,
                };
                read_capped(w, MAX).map_err(|e| format!("{path}: {e}"))?
            }
        },
    };
    if subject.trim().is_empty() {
        return Err("the subject is empty; nothing would be reviewed".into());
    }
    Ok(subject)
}

/// Names the sub-command a mistyped word was reaching for, if any.
fn one_word(w: &str) -> String {
    const WORD: &str = "a subject is a question, not a word";
    let near = super::RESERVED
        .iter()
        .find(|r| r.eq_ignore_ascii_case(w))
        .map(|r| format!("bx council {r}"))
        .or_else(|| {
            w.eq_ignore_ascii_case("help")
                .then(|| "bx council --help".into())
        });
    match near {
        Some(cmd) => format!("did you mean `{cmd}`? {WORD}"),
        None => format!("{WORD} — quote a sentence, or pass @FILE or -"),
    }
}

/// Reads stdin under the deadline `@FILE` gets. On a thread, since a blocking read cannot be timed
/// out; it is left running, because the caller exits. A terminal is exempt: a person is typing.
fn read_stdin(max: u64, wait: Duration) -> Result<String, String> {
    let stdin = std::io::stdin();
    if std::io::IsTerminal::is_terminal(&stdin) {
        return read_capped(stdin.lock(), max).map_err(|e| e.to_string());
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        tx.send(read_capped(std::io::stdin().lock(), max).map_err(|e| e.to_string()))
            .ok();
    });
    match rx.recv_timeout(wait) {
        Ok(read) => read,
        Err(_) => Err(format!(
            "stdin was not readable in {}s; is anything writing to it?",
            wait.as_secs()
        )),
    }
}

/// A non-blocking reader made to block again, until `until`: a slow `@<(cmd)` first reads
/// `WouldBlock`, and a silent writer returns it forever. Polling, not `fcntl`: no `unsafe`.
struct Waiting<R> {
    inner: R,
    until: std::time::Instant,
}

impl<R: std::io::Read> std::io::Read for Waiting<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        loop {
            match self.inner.read(buf) {
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if std::time::Instant::now() >= self.until {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            "not readable in time; is anything writing to it?",
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                other => return other,
            }
        }
    }
}

/// Reads at most `max` bytes, refusing anything larger rather than truncating it silently.
fn read_capped(mut src: impl std::io::Read, max: u64) -> std::io::Result<String> {
    use std::io::Read;
    // Bytes first: the cut can split a character, which is a size error, not an encoding one.
    let mut buf = Vec::new();
    src.by_ref()
        .take(max.saturating_add(1))
        .read_to_end(&mut buf)?;
    if buf.len() as u64 > max {
        return Err(std::io::Error::other(format!(
            "input exceeds maximum size ({max} bytes)"
        )));
    }
    String::from_utf8(buf).map_err(std::io::Error::other)
}

/// Every fixed name a run writes into its directory; `manifest.json` first, as the likeliest.
const ARTIFACTS: [&str; 6] = [
    "manifest.json",
    "brief.md",
    "index.md",
    "status.json",
    "sources.json",
    "trace.jsonl",
];

/// Resolves `--out` and returns the path it checked, which is the one to write to. Refuses `/`,
/// `$HOME`, anything inside `.git`, a leaf that is a link or not a directory, a directory already
/// holding a run's files, and a path `sanitize` would alter. The deepest existing ancestor is
/// canonicalized, so a link in the path cannot carry the run past these checks.
pub fn check_out(p: &std::path::Path) -> Result<PathBuf, String> {
    use std::path::Component;
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| format!("cannot resolve --out: {e}"))?
            .join(p)
    };
    // Lexical, as a shell's `cd` is: the leaf usually does not exist yet.
    let mut real = PathBuf::new();
    for c in abs.components() {
        match c {
            Component::ParentDir => {
                real.pop();
            }
            Component::CurDir => {}
            c => real.push(c),
        }
    }
    // The leaf is read, not followed: a planted `run -> /their/dir` passes every rule below.
    if let Ok(m) = std::fs::symlink_metadata(&real) {
        if m.file_type().is_symlink() {
            return Err(format!(
                "--out {} is a symbolic link; pass the directory it names",
                real.display()
            ));
        }
        if !m.is_dir() {
            return Err(format!("--out {} is not a directory", real.display()));
        }
        // Every artifact name is fixed, so a run would replace these. A lock is `plan`'s to judge.
        if std::fs::symlink_metadata(real.join(super::run::LOCK)).is_err()
            && let Some(name) = ARTIFACTS
                .iter()
                .find(|n| std::fs::symlink_metadata(real.join(n)).is_ok())
        {
            return Err(format!(
                "--out {} already contains {name}; pass an empty or new directory",
                real.display()
            ));
        }
    }
    // Canonicalize the deepest ancestor that exists, and re-attach the tail.
    let mut existing = real.clone();
    let mut tail = Vec::new();
    while !existing.exists() {
        match (existing.file_name(), existing.parent()) {
            (Some(name), Some(parent)) => {
                tail.push(name.to_os_string());
                existing = parent.to_path_buf();
            }
            _ => break,
        }
    }
    if !existing.is_dir() {
        return Err(format!(
            "--out {}: {} is not a directory",
            p.display(),
            existing.display()
        ));
    }
    // Refused, not best-effort: a lexical fallback would judge a link by its name.
    let canonical = existing
        .canonicalize()
        .map_err(|e| format!("cannot resolve --out {}: {e}", existing.display()))?;
    real = tail.iter().rev().fold(canonical, |p, name| p.join(name));

    if real.parent().is_none() {
        return Err(format!("refusing to write to {}", real.display()));
    }
    // Case-insensitively, as NTFS is.
    if real
        .components()
        .any(|c| c.as_os_str().eq_ignore_ascii_case(".git"))
    {
        return Err(format!(
            "refusing to write inside a .git directory: {}",
            real.display()
        ));
    }
    // Canonical on both sides: Windows' verbatim form and a linked home differ in spelling only.
    let home = home_dir().and_then(|h| h.canonicalize().ok());
    if home.is_some_and(|h| h == real) {
        return Err("refusing to write directly to $HOME".into());
    }
    // Every published path is sanitized; one this alters would name another directory.
    match real.to_str() {
        Some(s) if super::util::sanitize(s) == s => Ok(real),
        _ => Err(format!(
            "--out {} holds invisible, control or non-UTF-8 characters; pass a plain path",
            real.display()
        )),
    }
}

/// `GET /api/v1/key`: free; refuses a rejected key or an exhausted per-key limit. The account
/// balance needs a management key, so a member's 402 still reports it. Unreachable is not a
/// refusal: that is the retry policy's job.
pub fn check_credit(base: &str, key: &str) -> Result<(), Refusal> {
    let agent = crate::api::agent(PREFLIGHT_SECS);
    let url = format!("{base}/api/v1/key");
    let headers = [("Authorization", format!("Bearer {key}"))];
    let Ok(r) = crate::api::get_raw(&agent, &url, &headers, 64 * 1024) else {
        return Ok(());
    };
    match r.status {
        401 | 403 => Err(Refusal::auth("OpenRouter rejected the API key (401/403)")),
        402 => Err(Refusal::credit(
            "OpenRouter credit exhausted (402 Payment Required)",
        )),
        200 => {
            let v = crate::query::parse_json_slice(&r.body).unwrap_or(Value::Null);
            let d = &v["data"];
            match (d["limit"].as_f64(), d["limit_remaining"].as_f64()) {
                (Some(_), Some(rem)) if rem <= 0.0 => Err(Refusal::credit(
                    "OpenRouter credit exhausted (limit_remaining is 0)",
                )),
                _ => Ok(()),
            }
        }
        _ => Ok(()),
    }
}

/// Bound on each free pre-flight GET.
const PREFLIGHT_SECS: u64 = 15;

/// A refusal from `plan`, with the exit code a member meeting the same fault mid-run gets.
pub struct Refusal {
    pub message: String,
    pub exit: i32,
    /// What to do about it, in one word, for a machine reader.
    pub kind: &'static str,
}

impl Refusal {
    fn at(exit: i32, kind: &'static str, message: &str) -> Self {
        Refusal {
            message: message.to_string(),
            exit,
            kind,
        }
    }
    fn auth(message: &str) -> Self {
        Self::at(3, "auth", message)
    }
    fn credit(message: &str) -> Self {
        Self::at(8, "credit", message)
    }
}

/// The `kind` of every exit-2 refusal; `main.rs` reports in the same shape.
pub const INVALID_REQUEST: &str = "invalid_request";

/// Everything else `plan` refuses: nothing was sent and the command itself has to change.
impl From<String> for Refusal {
    fn from(message: String) -> Self {
        Refusal {
            message,
            exit: 2,
            kind: INVALID_REQUEST,
        }
    }
}

/// Refuses a request its wire will reject, judged against each wire's own model list: the id as
/// sent, zero-retention availability on OpenRouter, the output ceiling (which it lowers, refusing
/// only below the wire's floor), and the effort. Each would cost a seat at round one. Fails open
/// on anything short of a usable list, except a rejected key.
pub fn check_models(plan: &mut [Route], zdr: bool, effort: Effort) -> Result<(), Refusal> {
    let agent = crate::api::agent(PREFLIGHT_SECS);
    let (mut misses, mut asked) = (Vec::new(), Vec::new());
    for i in 0..plan.len() {
        let w = plan[i].wire;
        let name = w.name;
        if asked.contains(&name) {
            continue;
        }
        asked.push(name);
        let (key, base) = (plan[i].key.clone(), plan[i].base.clone());
        let mut headers = vec![(w.auth.0, format!("{}{key}", w.auth.1))];
        headers.extend(w.headers.iter().map(|(k, v)| (*k, (*v).to_string())));
        // Only OpenRouter's list is filtered by retention; a direct wire's is not.
        let filtered = zdr && name == wire::OPENROUTER.name;
        let mine: Vec<(usize, String)> = plan
            .iter()
            .enumerate()
            .filter(|(_, r)| r.wire.name == name)
            .map(|(j, r)| (j, listed_id(&r.on_wire())))
            .collect();
        // A cached list is used only when it places and clears every member: a refusal is always
        // re-read live.
        let cache_clears_them = |list: &[Served]| {
            mine.iter().all(|(_, id)| {
                list.iter().find(|s| s.id == *id).is_some_and(|s| {
                    (w.rejects)(s, effort).is_none()
                        && s.max_out.is_none_or(|limit| limit >= w.min_out)
                })
            })
        };
        let mut served = cached_roster(name, &base, &key, filtered).unwrap_or_default();
        let mut public = Vec::new();
        if !cache_clears_them(&served) {
            match roster(&agent, w.served, &base, w.models, &headers, filtered) {
                Ok(mut fresh) if !fresh.is_empty() => {
                    // Live, like `fresh`: a refusal is never read from a cached list.
                    let read = w.public.and_then(|_| {
                        let or = &wire::OPENROUTER;
                        let b = cfg::base_url(or).ok()?;
                        roster(&agent, or.served, &b, or.models, &[], false).ok()
                    });
                    let lent = w.public.is_none() || read.is_some();
                    public = read.unwrap_or_default();
                    if let Some(ns) = w.public {
                        wire::lend(ns, &mut fresh, &public);
                    }
                    // Not cached unlent: bare ids would clear every member for `ROSTER_TTL`.
                    if lent {
                        cache_roster(name, &base, &key, filtered, &fresh);
                    }
                    served = fresh;
                }
                // Every member on this wire would die on it at round one. Not 403: an OpenAI
                // key without model-read scope still infers.
                Err(Some(401)) => {
                    return Err(Refusal::auth(&format!("{name} rejected the API key (401)")));
                }
                // An unreadable list is not an empty roster.
                _ => continue,
            }
        }
        // Read only to explain a miss: absent, and absent under zero retention, differ in repair.
        let mut unfiltered = None;
        for (j, want) in &mine {
            let want = want.as_str();
            let found = match served.iter().find(|s| s.id == want) {
                Some(s) => Some(s.clone()),
                // A direct API's list omits the aliases it resolves (`claude-sonnet-4-5`).
                None if name != wire::OPENROUTER.name => {
                    match model_lookup(&agent, w, &base, &headers, want) {
                        Ok(mut m) => {
                            if let (Some(ns), Some(m)) = (w.public, m.as_mut()) {
                                wire::lend(ns, std::slice::from_mut(m), &public);
                            }
                            m
                        }
                        Err(Some(401)) => {
                            return Err(Refusal::auth(&format!(
                                "{name} rejected the API key (401)"
                            )));
                        }
                        // Unreadable is not absent.
                        Err(_) => continue,
                    }
                }
                None => None,
            };
            let Some(model) = found else {
                if filtered {
                    let all = unfiltered.get_or_insert_with(|| {
                        roster(&agent, w.served, &base, w.models, &headers, false)
                            .unwrap_or_default()
                    });
                    if all.iter().any(|s| s.id == want) {
                        misses.push(format!(
                            "  {name} serves `{}` but not under zero data retention (from -m {}) \
                             — drop it, or set \"zdr\": false in council.json",
                            shown(want),
                            shown(&plan[*j].slug)
                        ));
                        continue;
                    }
                }
                // Suggested only when unambiguous: a separator, or a missing vendor prefix.
                let near: Vec<&String> = served
                    .iter()
                    .map(|s| &s.id)
                    .filter(|s| loose(s) == loose(want) || wire::bare_model(s) == want)
                    .collect();
                misses.push(format!(
                    "  {name} does not serve `{}` (from -m {}){}",
                    shown(want),
                    shown(&plan[*j].slug),
                    match near.as_slice() {
                        [one] => format!(" — did you mean `{}`?", shown(one)),
                        _ => String::new(),
                    }
                ));
                continue;
            };
            // Lowered, never raised: a smaller ceiling still works. Below the wire's floor, the
            // smallest request it answers, it cannot.
            if let Some(limit) = model.max_out {
                let route = &mut plan[*j];
                if limit < route.wire.min_out {
                    misses.push(format!(
                        "  {name} caps `{}` at {limit} output tokens, below the {} this \
                         transport must send (from -m {})",
                        shown(want),
                        route.wire.min_out,
                        shown(&plan[*j].slug)
                    ));
                    continue;
                }
                route.start_tokens = route.start_tokens.min(limit);
                route.cap = route.cap.min(limit).max(route.start_tokens);
            }
            if let Some(why) = (w.rejects)(&model, effort) {
                misses.push(format!(
                    "  {name} serves `{}` but {why} (from -m {})",
                    shown(want),
                    shown(&plan[*j].slug)
                ));
            }
        }
    }
    if misses.is_empty() {
        return Ok(());
    }
    Err(format!(
        "nothing was sent — {}:\n{}",
        if misses.len() == 1 {
            "one member cannot run as asked"
        } else {
            "these members cannot run as asked"
        },
        misses.join("\n")
    )
    .into())
}

/// How long a fetched roster is trusted: free across a `--dry-run` loop, fresh by the afternoon.
pub(super) const ROSTER_TTL: Duration = Duration::from_secs(6 * 60 * 60);

/// A name from config, `-m` or a roster, ready to print: sanitized, and bounded.
fn shown(s: &str) -> String {
    super::util::truncate(&super::util::sanitize(s), 120)
}

/// Shared with `sub::stale_rosters`, which deletes by it.
pub(super) const ROSTER_PREFIX: &str = "roster-";

/// Where a roster is cached: named for the wire, the host and the key (another account may be
/// served another list; hashed, never written), and the retention filter.
fn roster_cache(wire: &str, base: &str, key: &str, zdr_only: bool) -> Option<PathBuf> {
    let name = format!(
        "{ROSTER_PREFIX}{wire}-{}{}.json",
        super::util::sha256_short(format!("{base}\n{key}").as_bytes()),
        if zdr_only { "-zdr" } else { "" }
    );
    Some(state_dir().ok()?.join("cache").join(name))
}

/// A roster written less than [`ROSTER_TTL`] ago.
fn cached_roster(wire: &str, base: &str, key: &str, zdr_only: bool) -> Option<Vec<Served>> {
    let p = roster_cache(wire, base, key, zdr_only)?;
    if std::fs::metadata(&p)
        .ok()?
        .modified()
        .ok()?
        .elapsed()
        .ok()?
        > ROSTER_TTL
    {
        return None;
    }
    // `read_artifact`: a link at a name this program minted is a plant.
    serde_json::from_str(&super::util::read_artifact(&p, super::util::MAX_LOCAL_FILE).ok()?).ok()
}

/// Best effort: a failed write costs a request. Through `make_private_dir`, since a `--dry-run`
/// may create the state root here.
pub(super) fn cache_roster(wire: &str, base: &str, key: &str, zdr_only: bool, models: &[Served]) {
    let Some(p) = roster_cache(wire, base, key, zdr_only) else {
        return;
    };
    if p.parent()
        .is_some_and(|d| super::run::make_private_dir(d).is_ok())
        && let Ok(body) = serde_json::to_string(models)
    {
        super::output::write_private(&p, &body).ok();
    }
}

/// What one wire serves, with the constraints it publishes; OpenAI publishes only ids, so its come
/// back `None`. 8 MiB cap: OpenRouter's list is the big one. `Err` carries a non-200 status.
fn roster(
    agent: &ureq::Agent,
    read: fn(&Value) -> Option<Served>,
    base: &str,
    path: &str,
    headers: &[(&str, String)],
    zdr_only: bool,
) -> Result<Vec<Served>, Option<u16>> {
    let url = format!("{base}{path}{}", if zdr_only { "?zdr=true" } else { "" });
    let r = crate::api::get_raw(agent, &url, headers, 8 << 20).map_err(|_| None)?;
    // A non-200 is not a roster, whatever `data` its error body carries.
    if r.status != 200 {
        return Err(Some(r.status));
    }
    let v = crate::query::parse_json_slice(&r.body).map_err(|_| None)?;
    Ok(v["data"]
        .as_array()
        .ok_or(None)?
        .iter()
        .filter_map(read)
        .collect())
}

/// `GET {models}/{id}`: the one record, resolved from an alias the list omits. `Ok(None)` is an
/// answer naming no model; `Err` is no answer, with its status if one came, as [`roster`]'s.
fn model_lookup(
    agent: &ureq::Agent,
    w: &wire::Wire,
    base: &str,
    headers: &[(&str, String)],
    id: &str,
) -> Result<Option<Served>, Option<u16>> {
    let path = w.models.split('?').next().unwrap_or(w.models);
    let url = format!("{base}{path}/{id}");
    let r = crate::api::get_raw(agent, &url, headers, 1 << 20).map_err(|_| None)?;
    match r.status {
        200 => Ok(crate::query::parse_json_slice(&r.body)
            .ok()
            .and_then(|v| (w.served)(&v))),
        404 => Ok(None),
        s => Err(Some(s)),
    }
}

/// The id a roster lists: OpenRouter's routing variants, combined in any order, are not in
/// `/models`; the base and any catalog variant (`:free`) are.
fn listed_id(id: &str) -> String {
    let mut parts = id.split(':');
    let base = parts.next().unwrap_or(id);
    parts
        .filter(|v| !["nitro", "floor", "exacto"].contains(v))
        .fold(base.to_string(), |id, v| id + ":" + v)
}

/// Ids differing only in separators are one name spelled two ways: `claude-fable-5.1` on
/// OpenRouter is `claude-fable-5-1` on Anthropic.
fn loose(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

pub fn env_of(w: &str) -> &'static str {
    wire::wire_for(w).map_or(wire::OPENROUTER.env, |w| w.env)
}

/// One member's route: everything `plan` decided about it, resolved once.
pub struct Route {
    /// The spelling the operator used. Every artifact reports this.
    pub slug: String,
    /// The slug with any `direct:`/`openrouter:` prefix removed — the key rates are stored under.
    pub bare: String,
    pub wire: &'static wire::Wire,
    pub key: String,
    pub base: String,
    /// Where this member's first request starts, after the wire's floor is applied.
    pub start_tokens: u32,
    /// Where truncation escalation stops: the lower of the universal ceiling and whatever the
    /// provider publishes for this model.
    pub cap: u32,
    /// The rate, when the transport has a knowable one — see `rate_of`.
    pub rate: Option<cfg::Rate>,
}

impl Route {
    /// The id this route will actually put on the request.
    pub fn on_wire(&self) -> std::borrow::Cow<'_, str> {
        wire::model_on_wire(self.wire, &self.bare)
    }
}

/// A member's rate, on a direct wire only: an OpenRouter slug routes across endpoints at
/// different prices. Keyed by the bare id, as `run::price` is.
pub fn rate_of(c: &cfg::Council, w: &wire::Wire, bare: &str) -> Option<cfg::Rate> {
    if w.name == wire::OPENROUTER.name {
        return None;
    }
    cfg::rate_for(c, bare)
}

/// Upper bound on spend, and how many members it could not price (they add nothing, so the count
/// says what the figure covers). Every member runs every round at its cap, every billed attempt.
pub fn worst_case(c: &cfg::Council, plan: &[Route], t: &task::Task, brief: &str) -> (f64, usize) {
    let steps = f64::from(t.steps());
    // A 200 carrying an error is charged, then retried.
    let attempts = f64::from(super::member::MAX_RETRIES + 1);
    // Four bytes per token: a prose estimate; the output terms dominate.
    let seed = (brief.len() / 4) as f64;
    let result = (super::search::MAX_RESULT_CHARS / 4) as f64;
    let priced: Vec<f64> = plan
        .iter()
        .filter_map(|r| {
            let [inp, out, _] = r.rate?;
            // The escalation ceiling, not the start: truncation escalates to it.
            let per_round = f64::from(r.cap);
            // Each round re-sends the seed, every earlier output and every tool result.
            let input = seed * steps + (per_round + result) * steps * (steps + 1.0) / 2.0;
            // Every leg: `clamp_completion` admits `cap * PARALLEL_LEGS` (`-pro` threads).
            let output = per_round * steps * out * super::types::PARALLEL_LEGS as f64;
            // At the cache-write tariff (1.25x): a first round's prefix is all write.
            Some(attempts * (input * inp * 1.25 + output) / 1e6)
        })
        .collect();
    // Brave is knowable: price per call, times the search budget, times the members.
    let per_call = cfg::brave_price(c).unwrap_or(0.0);
    let brave = per_call * f64::from(t.max_searches) * plan.len() as f64;
    (
        // `fold` from +0.0: `sum` starts at -0.0, which prints "$-0.00".
        priced.iter().fold(brave, |acc, v| acc + v),
        plan.len() - priced.len(),
    )
}

#[cfg(test)]
#[path = "resolve_tests.rs"]
mod tests;
