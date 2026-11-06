//! Tests for `sub.rs`, as a child module so they reach its private items.

use std::time::Duration;

use super::*;

/// A run's state comes from its artifacts, and the phase is chosen, never echoed from them.
#[test]
fn a_runs_state_is_read_from_its_artifacts_and_never_echoed_from_them() {
    let tmp = tempfile::tempdir().unwrap();
    let write = |name: &str, body: &str| {
        std::fs::write(tmp.path().join(name), body).unwrap();
    };
    // Nothing at all.
    assert!(run_state(tmp.path()).rendered().contains('\u{26a0}'));

    for (state, mark) in [
        ("ok", '\u{2713}'),
        ("partial", '\u{25d0}'),
        ("failed", '\u{2717}'),
    ] {
        write(
            "manifest.json",
            &format!(r#"{{"state":"{state}","quorum":{{"delivered":1,"total":2}}}}"#),
        );
        let read = run_state(tmp.path()).rendered();
        assert!(read.starts_with(mark), "{state} rendered as {read:?}");
        assert!(read.contains("1/2"), "{read:?}");
    }
    // A state this crate does not publish, carrying a screen-clear, is unknown.
    write("manifest.json", r#"{"state":"\u001b[2Jok"}"#);
    assert_eq!(run_state(tmp.path()).phase, "unknown");
}

/// A bare run id names the state-root run; anything else is a path used as given.
#[test]
fn a_bare_run_id_resolves_under_the_state_root_and_a_path_does_not() {
    let id = crate::council::util::run_id();
    let resolved = run_dir(std::path::PathBuf::from(&id)).unwrap();
    assert!(resolved.ends_with(&id), "{resolved:?}");
    assert!(
        resolved.parent().is_some_and(|p| p.ends_with("council")),
        "{resolved:?}"
    );
    let given = std::path::PathBuf::from("/tmp").join(&id);
    assert_eq!(run_dir(given.clone()).unwrap(), given);

    // Only the run-id format counts, not its length.
    for impostor in [
        "important-notes-backup",
        "2026090a-120000-abcdef", // a letter where the date digits go
        "20260903_120000-abcdef", // the wrong separator
    ] {
        assert_eq!(
            run_dir(std::path::PathBuf::from(impostor)).unwrap(),
            std::path::PathBuf::from(impostor),
            "{impostor} is not a run id and must be used as given"
        );
    }
}

/// `<id>/` (tab completion) and `<id>/.` name a directory relative to the cwd, not the state-root
/// run of that id.
#[test]
fn a_run_id_with_a_trailing_slash_is_a_path_not_a_state_root_run() {
    for given in ["20260101-120000-abcdef/", "20260101-120000-abcdef/."] {
        let given = std::path::PathBuf::from(given);
        assert_eq!(run_dir(given.clone()).unwrap(), given);
    }
}

/// Splitting a reply loses no paid text: the preamble, repeated headings and fenced headings all
/// stay, and only our own header goes.
#[test]
fn sections_keep_the_preamble_repeats_and_fenced_headings() {
    let body = concat!(
        "<!-- a/one -->\n\n",
        "**Disclosure:** no search tool reached me.\n\n",
        "## A. FATAL\r\n",
        "first\n",
        "```md\n## NOT A HEADING\n```\n",
        "## B. WRONG   \n",
        "### 1. a sub-heading is intra-section shape, not structure\n",
        "second \u{1b}[2Jwith a screen-clear\n",
        "## A. FATAL\n",
        "again\n",
    );
    let got = sections(body, true);
    let heads: Vec<&str> = got.iter().map(|s| s["heading"].as_str().unwrap()).collect();
    // The preamble is a section named "", and a repeated heading is its own entry.
    assert_eq!(heads, ["", "A. FATAL", "B. WRONG", "A. FATAL"], "{heads:?}");
    assert!(got[0]["text"].as_str().unwrap().starts_with("**Disclosure"));
    let a = got[1]["text"].as_str().unwrap();
    assert!(a.contains("## NOT A HEADING") && a.contains("```"), "{a}");
    assert_eq!(got[3]["text"], "again");
    // `###` stays inside its section.
    let b = got[2]["text"].as_str().unwrap();
    assert!(b.starts_with("### 1. a sub-heading"), "{b}");
    // Sanitized before serializing.
    assert!(
        !b.contains('\u{1b}') && b.contains("with a screen-clear"),
        "{b}"
    );

    // Our header goes; a reply with no headings is one section.
    let plain = sections("<!-- a/one -->\n\njust prose\n", true);
    assert_eq!(plain.len(), 1);
    assert_eq!(plain[0]["heading"], "");
    assert_eq!(plain[0]["text"], "just prose");
    assert!(sections("<!-- a/one -->\n\n", true).is_empty());
}

/// A section that opens with an indented code block keeps its first line's indentation.
#[test]
fn a_section_opening_with_indented_code_keeps_its_indentation() {
    let got = sections("## A\n\n    let x = 1;\n    y\n\n", true);
    assert_eq!(got[0]["text"], "    let x = 1;\n    y");
}

/// A `.md.partial` is the model's text alone, so a comment opening it is paid output.
#[test]
fn a_draft_opening_with_a_comment_keeps_it() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("00-a_one.md.partial");
    std::fs::write(&p, "<!-- model note -->\n## A\nx\n").unwrap();
    let row = member_row(Some(&p), &serde_json::Value::Null);
    assert_eq!(row["sections"][0]["text"], "<!-- model note -->", "{row}");
}

/// A fence is three or more backticks or tildes, closed only by the same character at least as
/// long; a heading inside one is code.
#[test]
fn only_a_matching_fence_closes_one_and_headings_inside_stay_inside() {
    let heads = |body: &str| {
        sections(body, true)
            .iter()
            .map(|s| s["heading"].as_str().unwrap_or_default().to_string())
            .collect::<Vec<_>>()
    };
    // A four-backtick block quoting a three-backtick one.
    assert_eq!(
        heads("## A\n````md\n```rust\nx\n```\n````\n## B\nb\n"),
        ["A", "B"],
        "an inner fence closed the outer block"
    );
    assert_eq!(
        heads("## A\n~~~md\n## NOT A HEADING\n~~~\n## B\nb\n"),
        ["A", "B"],
        "a heading inside a tilde fence was promoted"
    );
    // Backticks do not close a tilde fence and vice versa.
    assert_eq!(
        heads("## A\n~~~\n```\n## INSIDE\n~~~\n## B\nb\n"),
        ["A", "B"]
    );
    // Four spaces is an indented code block, not a fence.
    assert_eq!(heads("## A\n    ```\n## B\nb\n"), ["A", "B"]);
    // An unclosed fence runs to the end (CommonMark).
    assert_eq!(heads("## A\n```\nx\n## B\n"), ["A"]);
    // Up to three spaces before a heading; four is code.
    assert_eq!(heads("## A\na\n  ## B\nb\n"), ["A", "B"]);
    assert_eq!(heads("## A\na\n   ## B\nb\n"), ["A", "B"]);
    assert_eq!(heads("## A\na\n    ## B\nb\n"), ["A"]);
    // ```x``` is a code span: a backtick info string may not hold a backtick.
    assert_eq!(heads("## A\n```json``` is rejected\n## B\nb\n"), ["A", "B"]);
    assert_eq!(heads("## A\n```json\n## NOT\n```\n## B\n"), ["A", "B"]);
}

/// Only spaces or tabs may follow a closing fence (CommonMark); a no-break space leaves it open.
#[test]
fn a_fence_followed_by_a_no_break_space_does_not_close() {
    let got = sections("## A\n```\n```\u{a0}\n## NOT\n```\n## B\nb\n", true);
    let heads: Vec<&str> = got.iter().map(|s| s["heading"].as_str().unwrap()).collect();
    assert_eq!(heads, ["A", "B"]);
}

/// Within one second, a run without a brief is ordered by its directory's mtime.
#[cfg(unix)]
#[test]
fn a_run_without_a_brief_is_ordered_by_its_directory_not_by_a_flag() {
    let tmp = tempfile::tempdir().unwrap();
    let mk = |id: &str, brief: Option<std::time::SystemTime>| {
        let d = tmp.path().join(id);
        std::fs::create_dir(&d).unwrap();
        if let Some(at) = brief {
            let p = d.join("brief.md");
            std::fs::write(&p, "s").unwrap();
            std::fs::File::options()
                .write(true)
                .open(&p)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(at))
                .unwrap();
        }
        d
    };
    let now = std::time::SystemTime::now();
    // Suffixes run against start order.
    let stale = mk("20260101-120000-cccccc", None);
    std::fs::File::options()
        .write(true)
        .open(stale.join(".keep"))
        .or_else(|_| std::fs::File::create(stale.join(".keep")))
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(now - Duration::from_secs(30 * 86_400)))
        .unwrap();
    filetime_dir(&stale, now - Duration::from_secs(30 * 86_400));
    mk(
        "20260101-120000-bbbbbb",
        Some(now - Duration::from_secs(60)),
    );
    mk("20260101-120000-aaaaaa", None);

    let ids = run_ids(tmp.path());
    assert_eq!(
        ids.last().map(String::as_str),
        Some("20260101-120000-aaaaaa"),
        "the run still starting up is the newest: {ids:?}"
    );
    assert_eq!(
        ids.first().map(String::as_str),
        Some("20260101-120000-cccccc"),
        "an abandoned directory from last month is the oldest: {ids:?}"
    );
}

/// The id's UTC second outranks any file time, so a copied old run is not the latest.
#[test]
fn a_copied_old_run_is_not_the_latest_however_fresh_its_brief() {
    let tmp = tempfile::tempdir().unwrap();
    let now = std::time::SystemTime::now();
    for (id, at) in [
        ("20260101-120000-aaaaaa", now),
        ("20260201-120000-bbbbbb", now - Duration::from_secs(86_400)),
    ] {
        let d = tmp.path().join(id);
        std::fs::create_dir(&d).unwrap();
        std::fs::File::create(d.join("brief.md"))
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(at))
            .unwrap();
    }
    let ids = run_ids(tmp.path());
    assert_eq!(
        ids.last().map(String::as_str),
        Some("20260201-120000-bbbbbb"),
        "{ids:?}"
    );
}

/// A file or a link named like a run id is not a run, so it is never the latest.
#[test]
fn a_file_or_link_named_like_a_run_id_is_not_a_run() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir(tmp.path().join("20260101-120000-aaaaaa")).unwrap();
    std::fs::write(tmp.path().join("20991231-235959-bbbbbb"), "").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(
        tmp.path().join("20260101-120000-aaaaaa"),
        tmp.path().join("20991231-235959-cccccc"),
    )
    .unwrap();
    assert_eq!(run_ids(tmp.path()), ["20260101-120000-aaaaaa"]);
}

/// `set_times` on a directory, which `std::fs::File::options()` cannot open portably.
#[cfg(unix)]
fn filetime_dir(dir: &std::path::Path, at: std::time::SystemTime) {
    let d = std::fs::File::open(dir).unwrap();
    d.set_times(std::fs::FileTimes::new().set_modified(at)).ok();
}

/// Verdicts are append-only: the later row for a directory wins.
#[test]
fn a_run_scored_twice_keeps_the_later_verdict() {
    let buf = concat!(
        r#"{"schema":1,"run":"r1","dir":"/a","useful":true}"#,
        "\n",
        r#"{"schema":1,"run":"r2","dir":"/b","useful":true}"#,
        "\n",
        r#"{"schema":1,"run":"r1","dir":"/a","useful":false}"#,
        "\n",
        "not json\n",
    );
    let got = super::output::scores(buf);
    assert_eq!(got.get("/a"), Some(&false), "the later row wins");
    assert_eq!(got.get("/b"), Some(&true));
    assert_eq!(got.get("/c"), None, "an unscored run has no verdict at all");
    assert_eq!(got.len(), 2, "the junk line costs itself: {got:?}");
}

/// Verdicts join on the directory, never the run id, which two directories can share.
#[test]
fn a_verdict_for_one_directory_never_lands_on_another_of_the_same_name() {
    let buf = concat!(
        r#"{"schema":1,"run":"20260907-120000-abcdef","dir":"/elsewhere/20260907-120000-abcdef","useful":false}"#,
        "\n",
    );
    let got = super::output::scores(buf);
    assert_eq!(
        got.get("/state/bx/council/20260907-120000-abcdef"),
        None,
        "the state-root run shares a basename and was never judged: {got:?}"
    );
    assert_eq!(got.get("/elsewhere/20260907-120000-abcdef"), Some(&false));
}

/// Every Windows path, and any path with a `"`, is escaped on the way in.
#[test]
fn a_verdict_for_a_directory_json_must_escape_is_still_read() {
    let buf = concat!(
        r#"{"schema":1,"run":"r1","dir":"C:\\state\\\"r1\"","useful":true}"#,
        "\n",
    );
    let got = super::output::scores(buf);
    assert_eq!(got.get(r#"C:\state\"r1""#), Some(&true), "{got:?}");
}

/// What `prune` reports freed is every regular file it removes, nested ones included; a link is
/// worth nothing, since its target stays on disk.
#[test]
fn a_runs_size_counts_every_nested_file_and_no_link() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("brief.md"), "12345").unwrap();
    std::fs::write(tmp.path().join("manifest.json"), "678").unwrap();
    let nested = tmp.path().join("nested").join("deeper");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(nested.join("trace.jsonl"), "counted").unwrap();
    // Outside the run: only the link to it is inside.
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("big.md"), vec![b'x'; 4096]).unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(outside.path().join("big.md"), tmp.path().join("link.md"))
            .unwrap();
        std::os::unix::fs::symlink(outside.path(), tmp.path().join("link-dir")).unwrap();
    }
    assert_eq!(dir_bytes(tmp.path()), 5 + 3 + 7);
    assert_eq!(
        dir_bytes(&tmp.path().join("absent")),
        0,
        "an unreadable run is not an error"
    );
}

/// A status with no counts renders no progress.
#[test]
fn a_status_file_without_a_count_is_not_rendered_as_progress() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("status.json"), r#"{"ts":"now"}"#).unwrap();
    let read = run_state(tmp.path()).rendered();
    assert!(read.starts_with('\u{25cf}'), "{read:?}");
    assert!(!read.contains('/'), "no count may be invented: {read:?}");
}

/// A `status.json` that is not an object was not written by a run, so it is not a running one.
#[test]
fn a_status_file_that_is_not_an_object_is_not_a_running_run() {
    let tmp = tempfile::tempdir().unwrap();
    for body in ["null", "[]", r#""x""#] {
        std::fs::write(tmp.path().join("status.json"), body).unwrap();
        assert_eq!(run_state(tmp.path()).phase, "incomplete", "{body}");
    }
}

/// Staleness is the clock distance either way: a status dated far ahead is a dead run, one a
/// second ahead (a server clock slightly fast) is still live.
#[test]
fn a_status_dated_far_in_the_future_reads_stalled_and_a_second_ahead_running() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("status.json");
    std::fs::write(&p, r#"{"max_seconds":60}"#).unwrap();
    let at = |ahead: Duration| {
        std::fs::File::options()
            .write(true)
            .open(&p)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(std::time::SystemTime::now() + ahead))
            .unwrap();
        run_state(tmp.path()).phase
    };
    assert_eq!(at(Duration::from_secs(365 * 86_400)), "stalled");
    assert_eq!(at(Duration::from_secs(1)), "running");
}

/// Expired rosters (and the writer's temps) are swept; fresh ones, foreign files and look-alikes
/// stay, and a dry run only counts.
#[test]
fn expired_rosters_are_swept_and_a_dry_run_only_counts_them() {
    let tmp = tempfile::tempdir().unwrap();
    let cache = tmp.path().join("cache");
    std::fs::create_dir(&cache).unwrap();
    let put = |name: &str, ago: std::time::Duration| {
        let p = cache.join(name);
        std::fs::write(&p, "12345").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&p)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(std::time::SystemTime::now() - ago))
            .unwrap();
        p
    };
    let ttl = crate::council::resolve::ROSTER_TTL;
    let expired = put("roster-openrouter-aaaa.json", ttl + Duration::from_secs(60));
    let fresh = put("roster-anthropic-bbbb.json", Duration::from_secs(60));

    assert_eq!(
        stale_rosters(&cache, true),
        5,
        "a dry run reports its bytes"
    );
    assert!(expired.exists(), "a dry run deleted a file");

    let foreign = put("IMPORTANT.txt", ttl + Duration::from_secs(60));
    // `write_private`'s temp replaces the extension.
    let orphan = put(
        "roster-openrouter-aaaa.tmp999-1",
        ttl + Duration::from_secs(60),
    );
    let backup = put(
        "roster-openrouter-aaaa.json.bak",
        ttl + Duration::from_secs(60),
    );

    assert_eq!(stale_rosters(&cache, false), 10);
    assert!(!expired.exists());
    assert!(
        !orphan.exists(),
        "the writer's own leftover was kept forever"
    );
    assert!(backup.exists(), "an operator's backup was swept");
    assert!(fresh.exists(), "a roster still inside its window was swept");
    assert!(
        foreign.exists(),
        "a file this program never wrote was swept"
    );
    assert_eq!(
        stale_rosters(&tmp.path().join("absent"), false),
        0,
        "no cache directory yet is not an error"
    );
}

/// The sweep takes `write_private`'s real temp name and spares a name that merely contains `.tmp`.
#[test]
fn the_roster_sweep_spares_a_name_that_only_contains_tmp() {
    let tmp = tempfile::tempdir().unwrap();
    let old = std::time::SystemTime::now()
        - (crate::council::resolve::ROSTER_TTL + Duration::from_secs(60));
    let put = |name: &str| {
        let p = tmp.path().join(name);
        std::fs::write(&p, "1").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&p)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(old))
            .unwrap();
        p
    };
    let temp = put("roster-openrouter-aaaa.tmp4242-1700000000123456789");
    let spared = [
        put("roster-openrouter-aaaa.tmpl"),
        put("roster-openrouter-aaaa.tmp.bak"),
        put("roster-openrouter-aaaa.tmp42-"),
    ];
    assert_eq!(stale_rosters(tmp.path(), false), 1);
    assert!(!temp.exists());
    for p in spared {
        assert!(p.exists(), "{p:?} was swept");
    }
}

/// A `cache` that is a link is never swept, whatever it points at.
#[cfg(unix)]
#[test]
fn a_symlinked_cache_is_not_a_cache_this_program_may_sweep() {
    let tmp = tempfile::tempdir().unwrap();
    let victim = tmp.path().join("elsewhere");
    std::fs::create_dir(&victim).unwrap();
    // Named and aged like a roster of ours.
    let bait = victim.join("roster-openrouter-aaaa.json");
    std::fs::write(&bait, "12345").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&bait)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(
            std::time::SystemTime::now()
                - (crate::council::resolve::ROSTER_TTL + Duration::from_secs(60)),
        ))
        .unwrap();

    let cache = tmp.path().join("cache");
    std::os::unix::fs::symlink(&victim, &cache).unwrap();
    assert_eq!(stale_rosters(&cache, false), 0);
    assert!(bait.exists(), "the sweep followed a link out of the root");
}

/// An unparseable manifest is `unknown`, never a fall-through to `status.json`'s `running`.
#[test]
fn a_manifest_that_cannot_be_parsed_is_not_a_run_still_going() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("status.json"), r#"{"done":1,"total":4}"#).unwrap();
    std::fs::write(tmp.path().join("manifest.json"), "not json at all {{{").unwrap();
    let st = run_state(tmp.path());
    assert_eq!(st.phase, "unknown", "an unreadable manifest read as live");
    assert_eq!(st.done, None, "counts were invented from another file");
}

/// `serde_json`'s raw-value token is an ordinary key to `run_state`, as it is to `findings`.
#[test]
fn a_manifest_opening_with_the_raw_value_token_reads_like_any_other() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join("manifest.json"),
        r#"{"$serde_json::private::RawValue":"x","state":"ok","quorum":{"delivered":1,"total":1}}"#,
    )
    .unwrap();
    assert_eq!(run_state(tmp.path()).phase, "ok");
}

/// A non-UTF-8 entry still on disk is not reported destroyed.
#[cfg(unix)]
#[test]
fn a_name_that_is_not_utf8_is_not_reported_as_destroyed() {
    use std::os::unix::ffi::OsStrExt;
    let tmp = tempfile::tempdir().unwrap();
    let raw = std::ffi::OsStr::from_bytes(b"bad\xffname.md");
    std::fs::write(tmp.path().join(raw), "paid text").unwrap();
    let before = entry_names(tmp.path());
    assert_eq!(before.len(), 1);
    assert_eq!(
        lost(&before, tmp.path()),
        "nothing in it was removed",
        "a file still on disk was reported destroyed"
    );
}

/// An entry `prune` cannot stat after a failed removal may be intact, so it is not reported
/// destroyed.
#[cfg(unix)]
#[test]
fn an_entry_prune_cannot_stat_is_not_reported_as_destroyed() {
    use std::os::unix::fs::PermissionsExt;
    // SAFETY: geteuid has no preconditions.
    if unsafe { libc::geteuid() } == 0 {
        return; // root stats through any mode
    }
    let tmp = tempfile::tempdir().unwrap();
    let run = tmp.path().join("run");
    std::fs::create_dir(&run).unwrap();
    std::fs::write(run.join("00-a_one.md"), "paid text").unwrap();
    let before = entry_names(&run);
    // Readable but not searchable: every stat inside fails with EACCES.
    std::fs::set_permissions(&run, std::fs::Permissions::from_mode(0o600)).unwrap();
    let said = lost(&before, &run);
    std::fs::set_permissions(&run, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert_eq!(said, "nothing in it was removed");
}

/// `prune` fails closed on an entry it cannot classify, whatever its name.
#[test]
fn an_entry_whose_type_cannot_be_read_counts_as_paid_output() {
    assert!(paid_entry(Err(std::io::Error::other("EIO")), b"brief.md"));
}

/// A symlinked manifest is never read through, so the run is `unknown`: not `running`, so still
/// scorable, and not the state the link's target claims.
#[cfg(unix)]
#[test]
fn a_symlinked_manifest_reads_unknown_and_is_still_scorable() {
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("elsewhere.json");
    std::fs::write(
        &target,
        r#"{"state":"ok","quorum":{"delivered":1,"total":1}}"#,
    )
    .unwrap();
    std::os::unix::fs::symlink(&target, tmp.path().join("manifest.json")).unwrap();
    std::fs::write(tmp.path().join("status.json"), "{}").unwrap();
    let st = run_state(tmp.path());
    assert_eq!(st.phase, "unknown");
    assert_eq!(st.done, None);
}

/// `tasks` pads names by byte length, which is right only because every listed name is ASCII.
#[test]
fn every_listed_task_name_is_ascii() {
    assert!(task::BUILTINS.iter().all(|(n, _, _)| n.is_ascii()));
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("café.md"), "x {doc}").unwrap();
    std::fs::write(tmp.path().join("mine.md"), "x {doc}").unwrap();
    let _dir = task::dir_for_test(tmp.path());
    let names: Vec<String> = task::on_disk_rows().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["mine"]);
}

/// Fence scanning slices by byte after a run of single-byte marks, so multi-byte text on a fence
/// line cannot split a character.
#[test]
fn sections_never_panics_on_a_fence_line_of_multibyte_characters() {
    for body in [
        "## A\n```é\nx\n```\n## B\n",
        "## A\n~~~~ü~\n## B\n",
        "## A\n   é```\n## B\n",
        "## A\n``\u{2003}`\n## B\n",
        "## A\n```\n```\u{1f600}\n## B\n",
    ] {
        let _ = sections(body, true);
    }
}

/// Every wire has a key slot of its own, so saving one key never lands in another's.
#[test]
fn every_wire_in_the_table_has_its_own_key_slot() {
    let mut c = cfg::Council::default();
    for w in wire::TABLE {
        *key_slot(&mut c, w.name).unwrap() = Some(w.name.to_string());
    }
    for w in wire::TABLE {
        assert_eq!(key_slot(&mut c, w.name).unwrap().as_deref(), Some(w.name));
    }
}
