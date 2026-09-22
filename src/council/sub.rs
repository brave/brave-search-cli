//! The sub-commands: read-only ones, plus `set-key`, `prune` and `useful-*`. They take the subject
//! slot rather than a nested clap grammar, as `bx web` does.
//!
//! `prune` is the only one that deletes: it decides on lstat alone, on the run-id format, and
//! honours `--dry-run`.

use std::io::Write;

use super::resolve::{env_of, panel_slugs, state_dir, transport_for};
use super::{CouncilArgs, cfg, output, task, util, wire};
use crate::query::build_json_body;

/// Whether this invocation wants machine output; `--format` defaults to `json`.
fn machine(a: &CouncilArgs) -> bool {
    a.format != "text"
}

pub fn dispatch(what: &str, a: &CouncilArgs, brave_key: &str) -> Result<i32, String> {
    let run =
        handler(what).ok_or_else(|| format!("'{what}' has no handler; this is a bug in bx"))?;
    run(a, brave_key)
}

/// A sub-command: the args and the Brave key `main.rs` resolved.
pub(super) type Handler = fn(&CouncilArgs, &str) -> Result<i32, String>;

/// The handler for each reserved word. `None` only if `mod.rs::RESERVED` gains a word this lacks;
/// an error then, not a panic, since `panic = "abort"` leaves no exit code or message.
pub(super) fn handler(what: &str) -> Option<Handler> {
    let h: Handler = match what {
        "tasks" => |a, _| tasks(a),
        "ls" => |a, _| ls(a),
        "prune" => |a, _| prune(a),
        "status" => |a, _| status(a),
        "findings" => |a, _| findings(a),
        "useful-yes" => |a, _| score(a, true),
        "useful-no" => |a, _| score(a, false),
        // Reserved to refuse: as a subject it would buy a panel review of one word.
        "useful" => |_, _| {
            Err(
                "say which: `bx council useful-yes` or `bx council useful-no` \
                 (name a run after it, or --out <run>, to score one other than the newest)"
                    .into(),
            )
        },
        "set-key" => |a, _| set_key(a),
        // `models` is doctor's old name, kept so a stale habit cannot buy a panel.
        "models" | "doctor" => doctor,
        _ => return None,
    };
    Some(h)
}

/// Lists the tasks, built-in and on disk, and how to write one.
fn tasks(a: &CouncilArgs) -> Result<i32, String> {
    let c = cfg::load(None)?;
    let default = c.default_task.as_deref().unwrap_or("design-review");
    let on_disk = task::on_disk_rows();
    // Built-in order (reviews before answers). An on-disk task replaces the built-in of
    // its name, as it does when resolved.
    let rows: Vec<(&str, String, &str)> = task::BUILTINS
        .iter()
        .filter(|(n, _, _)| !on_disk.iter().any(|(d, _)| d == n))
        .map(|(n, d, _)| (*n, (*d).to_string(), "builtin"))
        .chain(
            on_disk
                .iter()
                .map(|(n, d)| (n.as_str(), d.clone(), "on-disk")),
        )
        .collect();
    let dir = task::dir();
    if machine(a) {
        say!(
            "{}",
            serde_json::json!({
                "default": util::sanitize(default),
                "dir": dir.as_ref().map(|d| util::sanitize(&d.display().to_string())),
                "frontmatter": task::FRONTMATTER,
                // Sanitized before serializing: an invalid file's error echoes its raw text.
                "tasks": rows.iter().map(|(n, d, src)| serde_json::json!({
                    "name": n, "description": util::sanitize(d), "source": src,
                })).collect::<Vec<_>>(),
            })
        );
        return Ok(0);
    }
    let w = rows.iter().map(|(n, _, _)| n.len()).max().unwrap_or(0);
    for (name, what, source) in &rows {
        let mark = if *name == default { '\u{2605}' } else { ' ' };
        let what: &str = if what.is_empty() { source } else { what };
        say!("{mark} {name:w$}  {what}");
    }
    // How to add a task: where the file goes and what its frontmatter may pin.
    say!(
        "\n  Write your own: any .md file in {}\n  \
         --task matches the file name; `{{doc}}` marks where the subject goes, or it is \
         appended.\n  \
         Optional `---` frontmatter: {}\n  \
         A task carries its own panel and clock, so it is a saved review and not just a \
         prompt. Copy this from the first column — an indented `---` is not \
         frontmatter:\n\n{}",
        dir.map_or_else(
            || "(no config directory on this platform)".to_string(),
            |d| d.display().to_string()
        ),
        task::FRONTMATTER.join(", "),
        task::EXAMPLE
    );
    Ok(0)
}

/// Lists the newest runs with their state and verdict.
fn ls(a: &CouncilArgs) -> Result<i32, String> {
    let root = state_dir()?;
    let runs = run_ids(&root);
    // Unreadable means no verdicts. Clipped to the newest MAX_HISTORY, so only ~12k later
    // verdicts could push out a listed run's.
    let verdicts =
        util::read_tail(&root.join(output::SCORES), output::MAX_HISTORY).unwrap_or_default();
    let verdicts = output::scores(&verdicts.text);
    // One `run_state` per row, so a row's fields describe one moment.
    let rows: Vec<(&String, std::path::PathBuf, RunState)> = runs
        .iter()
        .rev()
        .take(LS_LIMIT)
        .map(|r| {
            let dir = root.join(r);
            let state = run_state(&dir);
            (r, dir, state)
        })
        .collect();
    if machine(a) {
        say!(
            "{}",
            serde_json::json!({
                "root": util::sanitize(&root.display().to_string()),
                "total": runs.len(),
                "shown": rows.len(),
                // `build_json_body` drops a `None`: `useful` is absent until judged.
                // `done`/`total` are `null` when the run published no count.
                "runs": rows.iter().enumerate().map(|(n, (r, dir, st))| build_json_body(&[
                    ("run", Some(r.as_str().into())),
                    // Sanitized before serializing: `serde_json` escapes an ESC to text
                    // that `say!` cannot see and a consumer decodes back.
                    ("dir", Some(util::sanitize(&dir.display().to_string()).into())),
                    ("prompt_sha", Some(subject_sha(dir).into())),
                    ("phase", Some(st.phase.into())),
                    ("done", Some(serde_json::json!(st.done))),
                    ("total", Some(serde_json::json!(st.total))),
                    ("useful", verdicts.get(dir.display().to_string().as_str()).map(|u| (*u).into())),
                    ("latest", Some((n == 0).into())),
                ])).collect::<Vec<_>>(),
            })
        );
        return Ok(0);
    }
    // Still says where runs will land.
    if rows.is_empty() {
        say!("no runs yet under {}", root.display());
    }
    for (n, (r, dir, st)) in rows.iter().enumerate() {
        // Id and path first: they are what `status` takes.
        say!(
            "{}\t{}\t{}\t{}{}{}",
            r,
            dir.display(),
            subject_sha(dir),
            st.rendered(),
            match verdicts.get(dir.display().to_string().as_str()) {
                Some(true) => " · useful",
                Some(false) => " · not useful",
                None => "",
            },
            if n == 0 { " · latest" } else { "" }
        );
    }
    if runs.len() > rows.len() {
        say!(
            "… {} more, newest first; {} of {} shown",
            runs.len() - rows.len(),
            rows.len(),
            runs.len()
        );
    }
    Ok(0)
}

/// One run's state: the named run, else the newest.
fn status(a: &CouncilArgs) -> Result<i32, String> {
    let dir = match a.run_target()? {
        Some(d) => run_dir(d)?,
        None => latest_run()?,
    };
    // `status.json` is written before the first request: without it this is not a run.
    let file = dir.join("status.json");
    util::read_artifact(&file, util::MAX_LOCAL_FILE)
        .map_err(|e| format!("{}: {e}", file.display()))?;
    // The reader `ls` uses, so the two never disagree about a run.
    let st = run_state(&dir);
    if !machine(a) {
        say!("{}\t{}", st.rendered(), dir.display());
    } else {
        say!(
            "{}",
            serde_json::json!({
                // Which run: without `--out` "the newest" can change mid-poll.
                "run": dir.file_name().map(|n| util::sanitize(&n.to_string_lossy())),
                "dir": util::sanitize(&dir.display().to_string()),
                "phase": st.phase, "done": st.done, "total": st.total,
            })
        );
    }
    Ok(0)
}

/// Splits each member's reply into its `## ` sections, so one section can be read across the panel.
///
/// Extraction, never interpretation: no clustering or severity, since that needs an inference call
/// in a command that is free. Keyed on the heading text as written; models reword headings.
fn findings(a: &CouncilArgs) -> Result<i32, String> {
    let dir = match a.run_target()? {
        Some(d) => run_dir(d)?,
        None => latest_run()?,
    };
    // The manifest names the task and each member's file. It is written last, so a killed run
    // has none: absent is read around, unreadable or malformed is an error.
    let path = dir.join("manifest.json");
    let man: Option<serde_json::Value> = match std::fs::symlink_metadata(&path) {
        Ok(_) => Some(
            crate::query::parse_json_slice(
                util::read_artifact(&path, util::MAX_LOCAL_FILE)
                    .map_err(|e| format!("{}: {e}", path.display()))?
                    .as_bytes(),
            )
            .map_err(|e| format!("{}: {e}", path.display()))?,
        ),
        // Only "not there" is absent; any other error may hide an intact manifest.
        Err(e)
            if matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
            ) =>
        {
            None
        }
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let state = man.as_ref().map_or_else(|| run_state(&dir), finished_state);
    let members: Vec<serde_json::Value> = match &man {
        Some(man) => man["members"]
            .as_array()
            .ok_or_else(|| format!("{}: `members` is not a list", path.display()))?
            .iter()
            // Rows may repeat a file and each is read, so they are bounded like `member_files`.
            .take(super::run::MAX_PANEL)
            .map(|m| {
                // Re-joined under the run: a planted manifest could name any path.
                let name = m["file"]
                    .as_str()
                    .and_then(|f| std::path::Path::new(f).file_name())
                    .map(|f| dir.join(f));
                member_row(name.as_deref(), m)
            })
            .collect(),
        // No manifest: read the files in seat order. The model is left absent; `slugify` is
        // lossy and the file's header is prose, not a field.
        None => member_files(&dir)
            .iter()
            .map(|p| member_row(Some(p), &serde_json::Value::Null))
            .collect(),
    };
    // Neither a manifest nor a reply: not a run, rather than a run with no members.
    if man.is_none() && members.is_empty() {
        return Err(format!(
            "{}: no manifest.json and no member replies — not a readable run",
            dir.display()
        ));
    }
    if !machine(a) {
        // Says why the members have no names.
        if man.is_none() {
            say!("{} — no manifest; read from the files", state.rendered());
        }
        // Headings and sizes only; the bodies are the member files. A tab or newline in a
        // manifest field would forge a column or a row.
        let flat = |v: &serde_json::Value| str_of(v).replace(['\n', '\t'], " ");
        for m in &members {
            let who = if m["model"].is_null() {
                &m["file"]
            } else {
                &m["model"]
            };
            say!("{}\t{}", flat(who), flat(&m["status"]));
            for s in m["sections"].as_array().unwrap_or(&Vec::new()) {
                let head = str_of(&s["heading"]);
                say!(
                    "  {:<48} {} chars",
                    if head.is_empty() { "(preamble)" } else { head },
                    str_of(&s["text"]).chars().count()
                );
            }
        }
        return Ok(0);
    }
    say!(
        "{}",
        serde_json::json!({
            "schema": 1,
            "run": dir.file_name().map(|n| util::sanitize(&n.to_string_lossy())),
            "dir": util::sanitize(&dir.display().to_string()),
            // Sanitized before serializing, like every field read from the run.
            "task": man.as_ref().and_then(|m| m["task"].as_str()).map(util::sanitize),
            // Says whether absent names mean a killed run.
            "phase": state.phase,
            "members": members,
        })
    );
    Ok(0)
}

/// One row of `members`: the reply's path and sections, plus what `m` (the manifest row, or `Null`)
/// says about who wrote it. Every string is sanitized. A field the manifest omits or a file that
/// cannot be read is absent, not null; `partial` is present (`true`) only for a draft.
fn member_row(path: Option<&std::path::Path>, m: &serde_json::Value) -> serde_json::Value {
    // Named when it is a regular file, not a link; read through `read_artifact`, which enforces
    // that in the open itself. A reply too large to read still keeps its path.
    let path = path.filter(|p| std::fs::symlink_metadata(p).is_ok_and(|m| m.is_file()));
    let body = path.and_then(|p| util::read_artifact(p, util::MAX_LOCAL_FILE).ok());
    build_json_body(&[
        // From the manifest, never the file's `<!-- … -->` header, which is rewritten prose.
        (
            "model",
            m["model"].as_str().map(|v| util::sanitize(v).into()),
        ),
        (
            "status",
            m["status"].as_str().map(|v| util::sanitize(v).into()),
        ),
        (
            "reason",
            m["reason"].as_str().map(|v| util::sanitize(v).into()),
        ),
        (
            "file",
            path.map(|p| util::sanitize(&p.display().to_string()).into()),
        ),
        (
            "partial",
            path.filter(|p| is_partial(p)).map(|_| true.into()),
        ),
        (
            "sections",
            body.as_deref()
                .map(|b| sections(b, !path.is_some_and(is_partial)).into()),
        ),
    ])
}

/// `write_member`'s two names for one seat: `{idx:02}-{slug}.md` and the `.md.partial` it flushes
/// each turn beside it.
fn is_partial(p: &std::path::Path) -> bool {
    p.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.ends_with(".md.partial"))
}

/// A seat's `NN-` prefix, whatever follows. Bytes, as a reply's name need not be UTF-8.
fn seat_named(n: &[u8]) -> bool {
    matches!(n, [d1, d2, b'-', ..] if d1.is_ascii_digit() && d2.is_ascii_digit())
}

/// The shape of a reply's name, `NN-*.md` or `NN-*.md.partial`, whatever its alphabet.
fn reply_shaped(n: &[u8]) -> bool {
    seat_named(n) && (n.ends_with(b".md") || n.ends_with(b".md.partial"))
}

/// Does this run hold paid output? `prune` asks before deleting. Fails closed: a directory it
/// cannot read counts.
fn holds_reply(dir: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return true;
    };
    entries.into_iter().any(|e| {
        e.map_or(true, |e| {
            paid_entry(e.file_type(), e.file_name().as_encoded_bytes())
        })
    })
}

/// Any regular file named for a seat, so `write_private`'s kept temp counts; looser than
/// `member_files`. An entry whose type cannot be read counts too.
fn paid_entry(kind: std::io::Result<std::fs::FileType>, name: &[u8]) -> bool {
    kind.map_or(true, |t| t.is_file() && seat_named(name))
}

/// A manifest-less run's replies, one per seat in seat order (`{idx:02}-{slug}.md` sorts by seat),
/// drafts included. Stricter than [`holds_reply`] and for reading only.
fn member_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut found: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        // Regular files only: `DirEntry::file_type` is lstat.
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                reply_shaped(n.as_bytes())
                    // `slugify`'s alphabet only: any other name was not written here, and could
                    // forge a text row or publish a path `sanitize` altered.
                    && n.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
            })
        })
        .collect();
    found.sort();
    // One row per seat: a draft beside its finished reply is dropped, before the truncation so it
    // cannot push a finished reply past the bound.
    let finished: std::collections::HashSet<std::path::PathBuf> =
        found.iter().filter(|p| !is_partial(p)).cloned().collect();
    // `X.md.partial` less its extension is `X.md`, as `run.rs` mints them.
    found.retain(|p| !is_partial(p) || !finished.contains(&p.with_extension("")));
    // No run has more members than a panel may hold.
    found.truncate(super::run::MAX_PANEL);
    found
}

/// A `Value`'s string, or empty.
fn str_of(v: &serde_json::Value) -> &str {
    v.as_str().unwrap_or_default()
}

/// Every `## ` section of one reply, in order; a finished reply's `<!-- model -->` header and an
/// empty preamble are dropped. A non-empty preamble is a section named `""` (it can carry a
/// caveat), and duplicate headings stay separate entries.
fn sections(body: &str, headed: bool) -> Vec<serde_json::Value> {
    // The header is ours; a `.md.partial` has none, so a comment opening one is the model's.
    let body = body
        .strip_prefix("<!--")
        .filter(|_| headed)
        .and_then(|r| r.split_once("-->"))
        .map_or(body, |(_, rest)| rest);
    // The open section is carried, not reached through `last_mut`, which would need an `expect`.
    let mut out: Vec<(String, String)> = Vec::new();
    let (mut head, mut text) = (String::new(), String::new());
    // The open fence's character and length (CommonMark): a heading inside one is code.
    let mut fence: Option<(char, usize)> = None;
    for line in body.lines() {
        // Up to three leading spaces on a fence or heading; four is an indented code block.
        let bare = line.trim_start_matches(' ');
        let block = line.len() - bare.len() <= 3;
        if block {
            let mark = bare.chars().next().filter(|c| matches!(c, '`' | '~'));
            let run = mark.map_or(0, |m| bare.chars().take_while(|c| *c == m).count());
            match (fence, mark) {
                // Closed by the same character, at least as long, then only spaces or tabs.
                (Some((open, len)), Some(m))
                    if m == open
                        && run >= len
                        && bare[run..].bytes().all(|b| matches!(b, b' ' | b'\t')) =>
                {
                    fence = None;
                }
                // A backtick info string may not hold a backtick, so ```x``` is a code span.
                (None, Some(m)) if run >= 3 && !(m == '`' && bare[run..].contains('`')) => {
                    fence = Some((m, run));
                }
                _ => {}
            }
        }
        let heading = bare.strip_prefix("## ").filter(|_| block);
        match heading.filter(|_| fence.is_none()) {
            Some(next) => {
                out.push((std::mem::take(&mut head), std::mem::take(&mut text)));
                head = next.trim().to_string();
            }
            None => {
                text.push_str(line);
                text.push('\n');
            }
        }
    }
    out.push((head, text));
    out.into_iter()
        .filter(|(head, text)| !head.is_empty() || !text.trim().is_empty())
        .map(|(head, text)| {
            // Sanitized before serializing. Leading blank lines go; the first line's indentation
            // stays, as it may be code.
            let t = text.trim_end();
            let lead = t.len() - t.trim_start().len();
            let t = &t[t[..lead].rfind('\n').map_or(0, |i| i + 1)..];
            serde_json::json!({"heading": util::sanitize(&head), "text": util::sanitize(t)})
        })
        .collect()
}

/// Records whether a run was worth its money, in `scores.jsonl` (not `history.jsonl`, which is one
/// line per member). Gated on `status.json` like `status`, so a killed run can be judged; a live
/// one is refused, as nothing in it is finished to judge. A later verdict replaces an earlier one.
fn score(a: &CouncilArgs, useful: bool) -> Result<i32, String> {
    let dir = match a.run_target()? {
        Some(d) => run_dir(d)?,
        None => latest_run()?,
    };
    if !std::fs::symlink_metadata(&dir).is_ok_and(|m| m.is_dir()) {
        return Err(format!("{} is not a run directory", dir.display()));
    }
    let status = dir.join("status.json");
    util::read_artifact(&status, util::MAX_LOCAL_FILE)
        .map_err(|e| format!("{}: {e}", status.display()))?;
    let st = run_state(&dir);
    if st.phase == "running" {
        return Err(format!(
            "{} is still running, or stopped too recently to tell; score it once it has finished",
            dir.display()
        ));
    }
    // `run` and `dir`: under `--out` the name need not be a run id. `dir` is the join key `ls`
    // builds as `root.join(id)`, so any spelling of a state-root run (`..`, a linked root) is
    // recorded as exactly that; any other directory is made absolute and lexically clean.
    let root = state_dir()?;
    let in_root = std::fs::canonicalize(&dir).ok().and_then(|real| {
        let id = real.file_name()?.to_str().filter(|n| util::is_run_id(n))?;
        (real.parent() == std::fs::canonicalize(&root).ok().as_deref()).then(|| root.join(id))
    });
    let dir = in_root.unwrap_or_else(|| {
        std::path::absolute(&dir)
            .unwrap_or(dir)
            .components()
            .collect()
    });
    let run = dir
        .file_name()
        .map_or_else(String::new, |n| util::sanitize(&n.to_string_lossy()));
    let line = serde_json::json!({
        "schema": 1, "ts": util::iso8601_now(), "run": run,
        "dir": dir.display().to_string(), "useful": useful,
    });
    // The printed copy is scrubbed; the record keeps the raw join key.
    let mut shown = line.clone();
    shown["dir"] = util::sanitize(&dir.display().to_string()).into();
    let path = root.join(output::SCORES);
    if a.dry_run {
        if machine(a) {
            say!("{shown}");
        } else {
            say!(
                "would record {} {}",
                if useful { "useful" } else { "not useful" },
                dir.display()
            );
        }
        return Ok(0);
    }
    super::run::make_private_dir(path.parent().unwrap_or(&path))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    // One `write_all` with its newline: O_APPEND is atomic per call, so two writes can interleave.
    std::io::Write::write_all(
        &mut super::run::open_append(&path).map_err(|e| format!("{}: {e}", path.display()))?,
        format!("{line}\n").as_bytes(),
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    if machine(a) {
        say!("{shown}");
    } else {
        say!(
            "{} {}",
            if useful { "useful" } else { "not useful" },
            dir.display()
        );
    }
    Ok(0)
}

/// Run directories under the state root, oldest first; `ls`, `latest_run` and `prune` share it.
/// A run is a real directory (never a link) named in the run-id format. An unreadable root yields
/// none, with a warning.
fn run_ids(root: &std::path::Path) -> Vec<String> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        // Absent is "no runs yet"; anything else must not read as that.
        Err(e) => {
            if e.kind() != std::io::ErrorKind::NotFound {
                note!("warning: {}: {e}", root.display());
            }
            return Vec::new();
        }
    };
    let mut runs: Vec<String> = entries
        .flatten()
        // `DirEntry::file_type` is lstat: a link or a file named like a run is not one.
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| util::is_run_id(n))
        .collect();
    // By the id's UTC second, which `cp` or `touch` cannot move; within a second, by the brief's
    // mtime (written once), else the directory's. Never through a link.
    runs.sort_by_cached_key(|r| {
        let dir = root.join(r);
        let started = std::fs::symlink_metadata(dir.join("brief.md"))
            .or_else(|_| std::fs::symlink_metadata(&dir))
            .and_then(|m| m.modified())
            .ok();
        (r[..15].to_string(), started, r.clone())
    });
    runs
}

/// A bare run id names the run under the state root, so `status <id>` composes with `ls`. Anything
/// else, `<id>/` included, is a path used as given.
fn run_dir(given: std::path::PathBuf) -> Result<std::path::PathBuf, String> {
    if given.to_str().is_some_and(util::is_run_id) {
        return Ok(state_dir()?.join(given));
    }
    Ok(given)
}

/// The newest run by `run_ids`' order, the one `ls` marks latest.
fn latest_run() -> Result<std::path::PathBuf, String> {
    let root = state_dir()?;
    run_ids(&root)
        .pop()
        .map(|r| root.join(r))
        .ok_or_else(|| format!("no runs under {}", root.display()))
}

/// The run's subject fingerprint (`prompt_sha`), hashed from the bytes of `brief.md` so it holds
/// for a live run and for a non-UTF-8 subject. Bounded, regular-file read: a FIFO cannot hang `ls`.
fn subject_sha(dir: &std::path::Path) -> String {
    util::read_artifact_bytes(&dir.join("brief.md"), util::MAX_LOCAL_FILE)
        .map_or_else(|_| "-".repeat(16), |b| util::sha256_short(&b))
}

/// One run's state, as fields; [`RunState::rendered`] is the terminal form. `manifest.json` is
/// written last and `status.json` before the first request.
struct RunState {
    /// `ok` | `partial` | `failed` | `running` | `stalled` | `incomplete` | `unknown`.
    phase: &'static str,
    done: Option<u64>,
    total: Option<u64>,
}

fn run_state(dir: &std::path::Path) -> RunState {
    let read = |name: &str| {
        util::read_artifact(&dir.join(name), util::MAX_LOCAL_FILE)
            .ok()
            .and_then(|s| crate::query::parse_json(&s).ok())
    };
    // A manifest present but unparseable is `unknown`, never a fall-through to `running`.
    if std::fs::symlink_metadata(dir.join("manifest.json")).is_ok() {
        return finished_state(&read("manifest.json").unwrap_or(serde_json::Value::Null));
    }
    match read("status.json").filter(serde_json::Value::is_object) {
        Some(s) => {
            // A live run rewrites this as members finish, so one older than twice the clock is a
            // dead run. The clock is capped so a planted value cannot keep it running.
            let budget = s["max_seconds"]
                .as_u64()
                .unwrap_or(super::run::DEFAULT_MAX_SECONDS)
                .min(super::run::MAX_SECONDS);
            // The distance either way: a status dated far ahead (a clock stepped back) is as
            // dead as one far behind.
            let stale = std::fs::symlink_metadata(dir.join("status.json"))
                .and_then(|m| m.modified())
                .ok()
                .map(|t| t.elapsed().unwrap_or_else(|e| e.duration()))
                .is_some_and(|age| age > std::time::Duration::from_secs(budget.saturating_mul(2)));
            RunState {
                phase: if stale { "stalled" } else { "running" },
                done: s["done"].as_u64(),
                total: s["total"].as_u64(),
            }
        }
        None => RunState {
            phase: "incomplete",
            done: None,
            total: None,
        },
    }
}

/// A finished run's state, from its manifest. The phase is chosen here, never echoed, so a planted
/// manifest cannot put a word of its own on the terminal.
fn finished_state(m: &serde_json::Value) -> RunState {
    let q = &m["quorum"];
    RunState {
        phase: match m["state"].as_str() {
            Some("ok") => "ok",
            Some("partial") => "partial",
            Some("failed") => "failed",
            _ => "unknown",
        },
        done: q["delivered"].as_u64(),
        total: q["total"].as_u64(),
    }
}

impl RunState {
    /// The terminal form: a glyph for the phase and the progress beside it.
    fn rendered(&self) -> String {
        let mark = match self.phase {
            "ok" => "\u{2713}",      // ✓
            "partial" => "\u{25d0}", // ◐
            "failed" => "\u{2717}",  // ✗
            "running" => "\u{25cf}", // ●
            "stalled" => "\u{26a0}", // ⚠
            "incomplete" => return "\u{26a0} incomplete".into(),
            _ => "?",
        };
        match (self.done, self.total) {
            (Some(d), Some(t)) => format!(
                "{mark} {d}/{t}{}",
                if self.phase == "stalled" {
                    " stalled"
                } else {
                    ""
                }
            ),
            _ => mark.to_string(),
        }
    }
}

/// How many runs `ls` prints, newest first; both forms say how many were left out.
const LS_LIMIT: usize = 20;

/// How old a run must be before `prune` may remove it. An age rather than a count, so starting a
/// few runs never evicts one still being read.
const PRUNE_AFTER: std::time::Duration = std::time::Duration::from_secs(30 * 24 * 60 * 60);

/// Removes runs older than [`PRUNE_AFTER`] that are finished or hold no paid output, never one
/// with a live lock, and sweeps expired rosters. `history.jsonl` is kept: it is the only record of
/// a pruned run, and its readers say when their window drops rows.
fn prune(a: &CouncilArgs) -> Result<i32, String> {
    let root = state_dir()?;
    let mut removed: Vec<(String, u64)> = Vec::new();
    let mut failed: Vec<String> = Vec::new();
    let mut locked: Vec<String> = Vec::new();
    let mut kept: Vec<(String, u64)> = Vec::new();
    // lstat only: a planted link must not age a run or finish it.
    let regular = |p: std::path::PathBuf| {
        std::fs::symlink_metadata(&p)
            .ok()
            .filter(|m| m.is_file())
            .map(|m| (p, m))
    };
    // Asked under the lock, so no run is writing: an unfinished run is kept only for the paid
    // output in it.
    let keeps_paid_output =
        |dir: &std::path::Path| regular(dir.join("manifest.json")).is_none() && holds_reply(dir);
    for id in run_ids(&root) {
        let dir = root.join(&id);
        // Aged by `brief.md`, written once before the first request, else by the directory,
        // whose mtime only ever moves later. A future time reads young.
        let aged = |m: std::fs::Metadata| {
            m.modified()
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age > PRUNE_AFTER)
        };
        let old = regular(dir.join("brief.md"))
            .map(|(_, m)| m)
            .or_else(|| std::fs::symlink_metadata(&dir).ok())
            .is_some_and(aged);
        if !old {
            continue;
        }
        // Measured before the lock below exists, so neither counts it.
        let bytes = dir_bytes(&dir);
        // So a partial failure can say what it destroyed.
        let before = entry_names(&dir);
        // Every run holds this lock while it writes, so taking it excludes a writer until the
        // removal. A lock older than the window is a dead run's (none outlives `MAX_SECONDS`)
        // unless its pid lives: the clock can jump. `--dry-run` only looks: creating the lock
        // moves a brief-less run's age.
        let lock = dir.join(super::run::LOCK);
        let taken = if a.dry_run {
            match std::fs::symlink_metadata(&lock) {
                Ok(_) => Err(std::io::ErrorKind::AlreadyExists.into()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(e),
            }
        } else {
            super::run::lock(&dir).map(Some)
        };
        let held = match taken {
            Ok(l) => l,
            Err(e)
                if e.kind() == std::io::ErrorKind::AlreadyExists
                    && std::fs::symlink_metadata(&lock).is_ok_and(aged)
                    && !writer_alive(&lock) =>
            {
                None
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                locked.push(id);
                continue;
            }
            Err(e) => {
                failed.push(util::sanitize(&format!("{}: {e}", lock.display())));
                continue;
            }
        };
        // Re-checked under the lock, by the brief alone: the lock just moved the directory mtime.
        if regular(dir.join("brief.md")).is_some_and(|(_, m)| !aged(m)) {
            continue;
        }
        if keeps_paid_output(&dir) {
            kept.push((id, bytes));
            continue;
        }
        if a.dry_run {
            removed.push((id, bytes));
            continue;
        }
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => {
                // The lock went with the directory; dropping it would unlink one a new run made.
                std::mem::forget(held);
                removed.push((id, bytes));
            }
            // `remove_dir_all` stops at the first entry it cannot unlink, after destroying the
            // ones before it; say which.
            Err(e) => failed.push(util::sanitize(&format!(
                "{}: {e} ({})",
                dir.display(),
                lost(&before, &dir)
            ))),
        }
    }
    // An expired roster is never read again; removing it costs one free GET.
    let cache = stale_rosters(&root.join("cache"), a.dry_run);
    let freed: u64 = removed.iter().map(|(_, b)| b).sum();
    if machine(a) {
        say!(
            "{}",
            serde_json::json!({
                "root": util::sanitize(&root.display().to_string()),
                "dry_run": a.dry_run,
                "removed": removed.iter().map(|(id, b)| serde_json::json!({
                    "run": id, "bytes": b,
                })).collect::<Vec<_>>(),
                "bytes_freed": freed,
                "stale_cache_bytes": cache,
                "kept_newer_than_days": PRUNE_AFTER.as_secs() / 86_400,
                "failed": failed,
                "locked": locked,
                "kept_unfinished": kept.iter().map(|(id, b)| serde_json::json!({
                    "run": id, "bytes": b,
                })).collect::<Vec<_>>(),
            })
        );
    } else {
        let verb = if a.dry_run { "would remove" } else { "removed" };
        for (id, bytes) in &removed {
            say!("{verb} {id} ({bytes} bytes)");
        }
        for id in &locked {
            say!("kept {id}: a run holds its lock, live or crashed");
        }
        for (id, bytes) in &kept {
            say!("kept {id}: unfinished, holds paid output ({bytes} bytes)");
        }
        say!(
            "{} run(s) {verb}, {freed} bytes{}; finished runs newer than {} days are kept{}",
            removed.len(),
            if cache > 0 {
                format!(" (plus {cache} bytes of expired roster cache)")
            } else {
                String::new()
            },
            PRUNE_AFTER.as_secs() / 86_400,
            if a.dry_run {
                " — nothing was deleted"
            } else {
                ""
            }
        );
        for f in &failed {
            note!("warning: {f}");
        }
    }
    Ok(if failed.is_empty() { 0 } else { 2 })
}

/// Deletes every cached roster past its TTL and returns the bytes freed: one level deep, regular
/// files only. A run replaces its roster atomically, so the worst race costs one free GET.
fn stale_rosters(dir: &std::path::Path, dry_run: bool) -> u64 {
    // lstat the directory first: `read_dir` would follow a `cache` link out of the root.
    let m = std::fs::symlink_metadata(dir);
    if !m.as_ref().is_ok_and(std::fs::Metadata::is_dir) {
        // Absent is a fresh install; anything else is worth a warning.
        if m.is_ok() {
            note!(
                "warning: {} is not a directory; no roster was swept",
                dir.display()
            );
        }
        return 0;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut bytes = 0;
    for e in entries.flatten() {
        let p = e.path();
        let Ok(m) = std::fs::symlink_metadata(&p) else {
            continue;
        };
        // Only the two names bx writes: the roster, and the temp `write_private` leaves when
        // interrupted (`.json` replaced by `.tmp{pid}-{nanos}`).
        let expired = p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
            let temp = n.rsplit_once(".tmp").and_then(|(_, t)| t.split_once('-'));
            let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
            n.starts_with(super::resolve::ROSTER_PREFIX)
                && (n.ends_with(".json") || temp.is_some_and(|(pid, ns)| digits(pid) && digits(ns)))
        }) && m.is_file()
            && m.modified()
                .and_then(|t| t.elapsed().map_err(std::io::Error::other))
                .is_ok_and(|age| age > super::resolve::ROSTER_TTL);
        if expired {
            match if dry_run {
                Ok(())
            } else {
                std::fs::remove_file(&p)
            } {
                Ok(()) => bytes += m.len(),
                Err(e) => note!("warning: {}: {e}", p.display()),
            }
        }
    }
    bytes
}

/// Whether the pid a lock names still runs. A reused pid keeps the run, the safe side.
fn writer_alive(lock: &std::path::Path) -> bool {
    #[cfg(unix)]
    if let Some(pid) = util::read_artifact(lock, 256).ok().and_then(|s| {
        s.strip_prefix("pid ")?
            .split_whitespace()
            .next()?
            .parse::<libc::pid_t>()
            .ok()
            .filter(|p| *p > 0)
    }) {
        // SAFETY: signal 0 sends nothing; it only asks whether `pid` exists.
        return unsafe { libc::kill(pid, 0) } == 0
            || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
    }
    #[cfg(not(unix))]
    let _ = lock;
    false
}

/// What a run holds, one level deep, so a failed removal can say what it destroyed. Raw names, so
/// a non-UTF-8 one is looked up as itself.
fn entry_names(dir: &std::path::Path) -> Vec<std::ffi::OsString> {
    std::fs::read_dir(dir).map_or_else(
        |_| Vec::new(),
        |es| es.flatten().map(|e| e.file_name()).collect(),
    )
}

/// Which of `before` no longer exists, as warning text: sanitized and bounded, since the names are
/// whatever the directory held.
fn lost(before: &[std::ffi::OsString], dir: &std::path::Path) -> String {
    /// Enough to identify what went, short enough to stay one line.
    const SHOWN: usize = 8;
    let gone: Vec<&std::ffi::OsString> = before
        .iter()
        // Gone only when it is not there: an entry it cannot stat may be intact.
        .filter(|n| {
            std::fs::symlink_metadata(dir.join(n))
                .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
        })
        .collect();
    if gone.is_empty() {
        return "nothing in it was removed".into();
    }
    let names = gone
        .iter()
        .take(SHOWN)
        .map(|n| util::sanitize(&n.to_string_lossy()))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{} of its {} entries were destroyed: {names}{}. What is left still reads as a run, with less in it",
        gone.len(),
        before.len(),
        if gone.len() > SHOWN {
            format!(", and {} more", gone.len() - SHOWN)
        } else {
            String::new()
        }
    )
}

/// The bytes of every regular file under `dir`, the tree `remove_dir_all` deletes. Links are never
/// followed (`DirEntry::metadata` is lstat), and the walk keeps its own stack, so no planted depth
/// can overflow ours.
fn dir_bytes(dir: &std::path::Path) -> u64 {
    let mut total = 0;
    let mut todo = vec![dir.to_path_buf()];
    while let Some(d) = todo.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            match e.metadata() {
                Ok(m) if m.is_dir() => todo.push(e.path()),
                Ok(m) if m.is_file() => total += m.len(),
                _ => {}
            }
        }
    }
    total
}

/// Each wire's key-signup page, read off `wire::TABLE` so it cannot drift.
fn signup() -> impl Iterator<Item = (&'static str, &'static str)> {
    wire::TABLE.iter().map(|w| (w.name, w.signup))
}

/// Saves provider keys read from stdin, never argv (world-readable, and kept in shell history).
/// Without `--via` it walks every provider at a terminal, blank to skip; `--via` sets one and
/// takes a pipe.
fn set_key(a: &CouncilArgs) -> Result<i32, String> {
    let only = a
        .via
        .as_deref()
        .map(wire::wire_for)
        .transpose()?
        .map(|w| w.name);
    let where_to = cfg::path().map_or_else(|| "the config".into(), |p| p.display().to_string());
    // Before any prompt: a dry run must not ask for a key it would discard.
    if a.dry_run {
        let asked: Vec<&str> = signup()
            .map(|(n, _)| n)
            .filter(|n| only.is_none_or(|o| o == *n))
            .collect();
        note!("would ask for {} and write {where_to}", asked.join(", "));
        if machine(a) {
            say!(
                "{}",
                serde_json::json!({
                    "dry_run": true,
                    "would_ask": asked,
                    "config": util::sanitize(&where_to),
                })
            );
        }
        return Ok(0);
    }
    // The walk needs a terminal: prompts nobody answers would hang.
    if only.is_none() && !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return Err(
            "set-key with no --via needs a terminal; pass --via, or set the env vars".into(),
        );
    }

    let c = cfg::load(None)?;
    note!("Keys are saved to {where_to}, readable only by you.");
    note!("(input is visible — to avoid it, export the env var instead)");

    let mut accepted: Vec<(&str, String)> = Vec::new();
    let mut bad = false;
    for (name, url) in signup().filter(|(n, _)| only.is_none_or(|o| o == *n)) {
        note!(
            "\n{name} — {} · {url}",
            if cfg::key_for(&c, name).is_some() {
                "already set"
            } else {
                env_of(name)
            }
        );
        std::io::stderr()
            .write_all(b"  paste key (blank to skip): ")
            .ok();
        std::io::stderr().flush().ok();
        let mut line = String::new();
        // Ends the walk, but the keys already accepted are still saved.
        if let Err(e) = std::io::BufRead::read_line(&mut std::io::stdin().lock(), &mut line) {
            note!("  not read: {e}");
            bad = true;
            break;
        }
        let key = line.trim().to_string();
        if key.is_empty() {
            continue;
        }
        // Reported, not fatal, so the keys already entered survive.
        if let Err(e) = cfg::check_key(name, &key) {
            note!("  not saved: {e}");
            bad = true;
            continue;
        }
        accepted.push((name, key));
    }

    // An empty pipe to `--via` is a failure; at a prompt, blank means "never mind".
    if accepted.is_empty() && !bad && only.is_some() {
        return Err("no key on stdin; pipe one in, or set the environment variable".into());
    }
    // One write for the whole walk, so a mistyped third key cannot lose the first two. Loaded
    // again here, not before the prompts, so an edit made while they waited is kept.
    if !accepted.is_empty() {
        let mut c = cfg::load(None)?;
        for (name, key) in &accepted {
            *key_slot(&mut c, name)
                .ok_or_else(|| format!("no key slot for {name}; this is a bug in bx"))? =
                Some(key.clone());
        }
        cfg::save(&c)?;
        note!("\nsaved {} key(s) to {where_to}", accepted.len());
        for (name, _) in &accepted {
            let var = env_of(name);
            if std::env::var(var)
                .ok()
                .and_then(crate::config::trim_non_empty)
                .is_some()
            {
                note!(
                    "  {var} is set and overrides the saved {name} key; unset it to use this one"
                );
            }
        }
    } else {
        note!("\nnothing saved");
    }
    note!("run `bx council doctor` to check");
    Ok(if bad { 2 } else { 0 })
}

/// A wire's key slot in `council.json`. No catch-all, so a new wire cannot overwrite another's key.
fn key_slot<'a>(c: &'a mut cfg::Council, wire: &str) -> Option<&'a mut Option<String>> {
    match wire {
        "openrouter" => Some(&mut c.openrouter_api_key),
        "anthropic" => Some(&mut c.anthropic_api_key),
        "openai" => Some(&mut c.openai_api_key),
        _ => None,
    }
}

/// The onboarding path, one string for both forms. The Brave key has its own command.
const SETUP: &str = "bx council set-key (inference) · bx config set-key <KEY> (brave)";

/// Diagnoses keys, panels and paths without any request; exit 2 when the panel a run would use
/// cannot run. Gathers first and renders second, so both forms describe one state.
fn doctor(a: &CouncilArgs, brave_key: &str) -> Result<i32, String> {
    let c = cfg::load(None)?;
    let via = a.via.as_deref().map(wire::wire_for).transpose()?;
    let mut brave_bad = false;
    let mut have: Vec<&str> = Vec::new();
    // "set" means present and shaped right, never verified: `doctor` is free.
    let mut keys: Vec<(&str, bool, String)> = Vec::new();
    for w in wire::TABLE.iter().map(|w| w.name) {
        match cfg::key_for(&c, w) {
            None => keys.push((w, false, "not set".into())),
            Some(k) => match cfg::check_key(w, &k) {
                Ok(()) => {
                    have.push(w);
                    keys.push((w, true, "set".into()));
                }
                // A warning unless the panel routes to it: then it lands in `missing` below.
                Err(e) => keys.push((w, false, e)),
            },
        }
    }
    // The Brave key every run needs, resolved by `main.rs` from the same sources a run reads,
    // without prompting or migrating anything.
    let brave = crate::config::trim_non_empty(brave_key.to_string());
    match &brave {
        None => keys.push(("brave", false, "not set — bx config set-key <KEY>".into())),
        Some(k) => match crate::config::validate_api_key(k) {
            Ok(()) => keys.push(("brave", true, "set".into())),
            Err(e) => {
                brave_bad = true;
                keys.push(("brave", false, e.to_string()));
            }
        },
    }
    // The effective panels: a config panel replaces the built-in of its name in place; a new
    // name is appended.
    let mut panels: Vec<(String, Vec<String>)> = cfg::DEFAULT_PANELS
        .iter()
        .map(|(n, m)| {
            let members = c.panels.get(*n).cloned();
            let members = members.unwrap_or_else(|| m.iter().map(|s| (*s).to_string()).collect());
            ((*n).to_string(), members)
        })
        .collect();
    for (name, members) in &c.panels {
        if !panels.iter().any(|(n, _)| n == name) {
            panels.push((name.clone(), members.clone()));
        }
    }

    // The effective aliases, `council.json`'s included.
    let mut aliases: Vec<String> = cfg::DEFAULT_ALIASES
        .iter()
        .map(|(a, _)| (*a).to_string())
        .chain(c.aliases.keys().cloned())
        .collect();
    aliases.sort_unstable();
    aliases.dedup();
    // Resolved by the code that uses them, so they are right on every platform.
    let paths = [
        ("config", cfg::path()),
        ("tasks", task::dir()),
        ("runs", state_dir().ok()),
    ];
    let mut missing: Vec<&'static str> = Vec::new();
    let mut members: Vec<(String, &'static str, Option<cfg::Rate>, String)> = Vec::new();
    // The task's `panel:` first, as a run resolves it. An error is reported beside the rest,
    // which is what explains it.
    let task = a.task.as_deref().or(c.default_task.as_deref());
    let panel =
        task::resolve(task.unwrap_or("design-review")).and_then(|t| panel_slugs(&c, &[], &t.panel));
    let (slugs, panel_error) = match panel {
        Ok(s) => (s, None),
        Err(e) => (Vec::new(), Some(e)),
    };
    for slug in slugs {
        // Rated by `bare`, as the run prices it, and routed with `--via`.
        let (w, bare) = transport_for(&c, &slug, via);
        let rate = cfg::rate_for(&c, bare);
        if !have.contains(&w) && !missing.contains(&w) {
            missing.push(w);
        }
        let note = match (w, rate) {
            ("openrouter", _) => "priced by the provider, after the fact".to_string(),
            (_, Some(r)) => format!(
                "${}/${} per MTok, cache reads x{} — --max-usd can bound it",
                r[0], r[1], r[2]
            ),
            _ => "no rate — --max-usd cannot bound this member".into(),
        };
        members.push((slug, w, rate, note));
    }
    // Judged against the panel that would run, not a count of keys.
    let short = brave_bad || !missing.is_empty() || brave.is_none() || panel_error.is_some();

    if machine(a) {
        say!(
            "{}",
            serde_json::json!({
                "ok": !short,
                "keys": keys.iter().map(|(n, ok, note)| serde_json::json!({
                    "provider": n, "set": ok, "detail": note,
                })).collect::<Vec<_>>(),
                "keys_checked": "shape only, never against the provider",
                "retention": {
                    "zdr": cfg::zdr(&c),
                    "by_transport": wire::TABLE.iter().map(|w| {
                        let r = wire::retention_of(w, cfg::zdr(&c));
                        serde_json::json!({
                            "transport": w.name,
                            "mechanism": r.mechanism,
                            "enforced_per_request": r.enforced_per_request,
                            "detail": r.detail,
                        })
                    }).collect::<Vec<_>>(),
                },
                // `null` when unset: not the same configuration as the default spelled out.
                "direct": c.direct.as_ref().map(|d|
                    d.iter().map(|v| util::sanitize(v)).collect::<Vec<_>>()),
                "direct_default": cfg::DEFAULT_DIRECT,
                // The operator's strings, sanitized before serializing.
                "aliases": aliases.iter().map(|a| util::sanitize(a)).collect::<Vec<_>>(),
                "panels": panels.iter().map(|(n, m)| serde_json::json!({
                    "name": util::sanitize(n),
                    "also": cfg::panel_aliases_for(n),
                    "members": m.iter().map(|s| util::sanitize(s)).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "paths": paths.iter().map(|(l, p)| serde_json::json!({
                    "what": l,
                    "path": p.as_ref().map(|p| util::sanitize(&p.display().to_string())),
                })).collect::<Vec<_>>(),
                // USD per million tokens; null where the provider prices after the fact.
                "default_panel": members.iter().map(|(slug, w, rate, note)| serde_json::json!({
                    "model": slug, "transport": w, "cost": note,
                    "priced": *w == "openrouter" || rate.is_some(),
                    "bounded_in_advance": rate.is_some(),
                    "rate_in": rate.map(|r| r[0]), "rate_out": rate.map(|r| r[1]),
                    "cache_read_multiplier": rate.map(|r| r[2]),
                })).collect::<Vec<_>>(),
                "missing_keys": missing,
                "panel_error": panel_error.as_deref().map(util::sanitize),
                // Billed per search whatever the models cost; `configured` says whose number.
                "brave_price_per_call_usd": {
                    "usd": cfg::brave_price(&c),
                    "configured": cfg::brave_price_is_configured(&c),
                },
                "setup": short.then_some(SETUP),
            })
        );
        return Ok(if short { 2 } else { 0 });
    }

    for (name, ok, note) in &keys {
        say!(
            "{} {name:<11} {note}",
            if *ok { '\u{2713}' } else { '\u{2717}' }
        );
    }
    say!("  (\"set\" = present and correctly shaped, not checked against the provider)");
    let zdr = cfg::zdr(&c);
    say!(
        "\n  retention   {}, and it means something different per transport:",
        if zdr {
            "zero requested"
        } else {
            "not requested — \"zdr\": false is set"
        }
    );
    for w in wire::TABLE {
        say!("    {:<12}{}", w.name, wire::retention_of(w, zdr).detail);
    }
    say!(
        "  direct      {}",
        match &c.direct {
            None => format!("{} (default)", cfg::DEFAULT_DIRECT.join(",")),
            Some(d) if d.is_empty() => "none — everything routes through openrouter".to_string(),
            Some(d) => d.join(","),
        }
    );
    say!(
        "  brave       {} per call{}",
        cfg::brave_price(&c).map_or_else(|| "unpriced".to_string(), |p| format!("${p}")),
        if cfg::brave_price_is_configured(&c) {
            " (yours)"
        } else {
            " (Brave's list price — set brave_price_per_call_usd for your plan)"
        }
    );
    say!("  aliases     {}", aliases.join(", "));
    for (name, list) in &panels {
        let also = cfg::panel_aliases_for(name);
        let label = if also.is_empty() {
            format!("@{name}")
        } else {
            format!("@{name} (@{})", also.join(", @"))
        };
        say!("  {label:<22} {}", list.join(", "));
    }
    for (label, path) in &paths {
        say!(
            "  {label:<11} {}",
            path.as_ref()
                .map_or_else(|| "—".into(), |p| p.display().to_string())
        );
    }
    say!("");
    for (slug, w, _, note) in &members {
        say!("  {slug:<32} {w:<11} {note}");
    }
    if let Some(e) = &panel_error {
        say!("  panel       {e}");
    }
    if !missing.is_empty() {
        say!(
            "\n  missing     {} — the default panel cannot run without {}",
            missing.join(", "),
            if missing.len() == 1 { "it" } else { "them" }
        );
    }
    if short {
        say!("  setup       {SETUP}");
    }
    Ok(if short { 2 } else { 0 })
}

#[cfg(test)]
#[path = "sub_tests.rs"]
mod tests;
