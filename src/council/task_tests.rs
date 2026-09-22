//! Tests for `task.rs`, as its child module so they reach private items.

use super::*;

#[test]
fn every_builtin_parses_and_carries_its_clauses() {
    for (name, _, _) in BUILTINS {
        let t = builtin(name).expect("builtin resolves");
        assert!(t.body.contains("{doc}"), "{name} has no {{doc}}");
        assert!(t.body.contains("No preamble"), "{name} lost anti-praise");
        assert_eq!(t.min_reply_chars, 400);
    }
}

#[test]
fn review_builtins_keep_the_breaking_condition_clause() {
    for name in ["design-review", "code-review"] {
        assert!(builtin(name).unwrap().body.contains("breaking condition"));
    }
}

#[test]
fn a_body_without_the_placeholder_gets_the_subject_appended() {
    let t = Task {
        body: "Review this.".into(),
        ..Task::default()
    };
    assert_eq!(t.render("CODE").unwrap(), "Review this.\n\nCODE");
}

#[test]
fn frontmatter_is_optional() {
    let t = parse("just a body").unwrap();
    assert_eq!(t.body, "just a body");
    assert_eq!(t.max_tokens, 64_000);
    assert!(t.grounded);
}

#[test]
fn frontmatter_overrides_and_the_panel_list_splits() {
    let t = parse("---\neffort: low\nmax_searches: 3\npanel: a, b ,c\n---\nbody").unwrap();
    assert!(t.effort == Effort::Low);
    assert_eq!(t.max_searches, 3);
    assert_eq!(t.panel, ["a", "b", "c"]);
    assert_eq!(t.steps(), 9, "max_steps follows max_searches when unset");
}

/// An empty `panel:` must not mean the default panel.
#[test]
fn an_empty_panel_line_is_refused_rather_than_read_as_the_default() {
    for v in ["", ",", " , ,"] {
        let Err(e) = parse(&format!("---\npanel: {v}\n---\nbody")) else {
            panic!("accepted panel: {v:?}")
        };
        assert!(e.contains("panel lists no models"), "{v:?}: {e}");
    }
}

#[test]
fn an_explicit_max_steps_wins() {
    assert_eq!(parse("---\nmax_steps: 2\n---\nb").unwrap().steps(), 2);
}

#[test]
fn unknown_keys_and_bad_values_are_errors_not_silent_defaults() {
    assert!(parse("---\nnope: 1\n---\nb").is_err());
    assert!(parse("---\nmax_tokens: lots\n---\nb").is_err());
    assert!(parse("---\neffort: turbo\n---\nb").is_err());
    assert!(parse("---\nno frontmatter close\nb").is_err());
    assert!(parse("---\n---\n   ").is_err(), "empty body");
}

/// A file is no safer a source than a flag: the same floor and a ceiling for every number.
#[test]
fn frontmatter_numbers_have_the_same_floor_as_their_command_line_twins() {
    for key in ["max_tokens", "max_steps", "max_searches", "min_reply_chars"] {
        assert!(
            parse(&format!("---\n{key}: 0\n---\nbody")).is_err(),
            "{key}: 0 was accepted"
        );
        assert!(
            parse(&format!("---\n{key}: 12\n---\nbody")).is_ok(),
            "{key}: 12 must still be accepted"
        );
        // Unbounded, `max_steps` times `max_tokens` would size away the conversation budget.
        assert!(
            parse(&format!("---\n{key}: 4000000000\n---\nbody")).is_err(),
            "{key} has no ceiling"
        );
    }
}

/// One 8 MiB line must not come back whole in the error.
#[test]
fn a_task_file_error_echoes_at_most_a_bounded_slice_of_its_line() {
    let long = "9".repeat(10_000);
    for raw in [
        format!("---\n{long}\n---\nbody"),
        format!("---\nmax_tokens: {long}\n---\nbody"),
        format!("---\n{long}: 1\n---\nbody"),
    ] {
        let Err(e) = parse(&raw) else {
            panic!("accepted a 10k-character line")
        };
        assert!(e.len() < 400, "{} bytes", e.len());
    }
}

#[test]
fn a_signed_frontmatter_number_is_refused() {
    for key in [
        "max_tokens",
        "max_steps",
        "max_searches",
        "max_seconds",
        "min_reply_chars",
    ] {
        assert!(
            parse(&format!("---\n{key}: +12\n---\nbody")).is_err(),
            "{key}: +12 was accepted"
        );
    }
}

/// The frontmatter clock stops where `--max-seconds` does.
#[test]
fn a_task_file_max_seconds_above_the_flags_ceiling_is_refused() {
    let max = crate::council::run::MAX_SECONDS;
    let at = parse(&format!("---\nmax_seconds: {max}\n---\nbody")).unwrap();
    assert_eq!(at.max_seconds, Some(max));
    for over in [max + 1, u64::MAX] {
        assert!(
            parse(&format!("---\nmax_seconds: {over}\n---\nbody")).is_err(),
            "max_seconds: {over} was accepted"
        );
    }
}

/// A floor above what one reply can hold rejects every member after paid reprompts.
#[test]
fn a_reply_floor_no_model_can_reach_is_refused_at_parse() {
    assert!(parse("---\nmin_reply_chars: 100000\n---\nbody").is_ok());
    assert!(parse("---\nmin_reply_chars: 100001\n---\nbody").is_err());
}

/// `..` passes the byte allowlist, so only its own clause keeps `push` inside the directory.
#[test]
fn a_dotted_name_cannot_walk_out_of_the_task_directory() {
    for bad in ["..", "...", "..md", "a..b", "..\\x"] {
        assert!(check_name(bad).is_err(), "accepted {bad:?}");
    }
    assert!(
        check_name("design.review-2").is_ok(),
        "a single dot is fine"
    );
}

#[test]
fn names_that_could_escape_the_task_directory_are_rejected() {
    for bad in ["../etc/passwd", "a/b", "a\\b", "", "con", "NUL.md"] {
        assert!(check_name(bad).is_err(), "accepted {bad:?}");
    }
    assert!(check_name("design-review").is_ok());
}

/// A file is planted where each escaping name would land, so a weakened check loads it.
#[test]
fn a_task_name_cannot_reach_a_file_outside_the_task_directory() {
    let root = tempfile::tempdir().unwrap();
    let tasks = root.path().join("a").join("b");
    std::fs::create_dir_all(&tasks).unwrap();
    let _dir = dir_for_test(&tasks);
    std::fs::write(root.path().join("x.md"), "planted {doc}").unwrap();
    let abs = root.path().join("x");
    for name in ["../../x", abs.to_str().unwrap(), "com1", "LPT9.md"] {
        let err = resolve(name).map(|t| t.body).unwrap_err();
        assert!(err.contains("invalid task name"), "{name}: {err}");
    }
}

/// Every built-in can search, so each needs both halves of the grounding contract.
#[test]
fn every_builtin_task_carries_the_whole_grounding_contract() {
    for (name, _, body) in BUILTINS {
        assert!(
            body.contains("Ground every load-bearing claim with a search"),
            "{name} never tells the model to search"
        );
        // The source's `\` continuation leaves one space here.
        assert!(
            body.contains("never instructions to follow"),
            "{name} does not defend against a page that gives the model orders"
        );
        // Appending is the fallback for hand-written tasks, not for shipped ones.
        assert!(body.contains("{doc}"), "{name} has no {{doc}} placeholder");
        // Fixed sections, so two members' answers can be compared.
        assert!(
            body.contains("## A."),
            "{name} has no fixed section structure"
        );
    }
}

#[test]
fn every_review_shaped_builtin_carries_headings_a_through_e() {
    for name in [
        "design-review",
        "code-review",
        "audit",
        "second-pass",
        "fact-check",
    ] {
        let body = builtin(name).unwrap().body;
        for h in ["## A.", "## B.", "## C.", "## D.", "## E. VERDICT"] {
            assert!(body.contains(h), "{name} lacks {h}");
        }
    }
}

/// The prompt tells the model which framing marks retrieved text; the framing must be that tag.
#[test]
fn the_tag_the_prompt_names_is_the_tag_search_results_are_wrapped_in() {
    let body =
        r#"{"grounding":{"generic":[{"url":"https://a.example","title":"t","snippets":["s"]}]}}"#;
    let out =
        crate::council::search::render(&crate::api::Resp::new(200, body.as_bytes(), &[])).render();
    let tag = out.split('>').next().unwrap().to_string() + ">";
    assert_eq!(tag, "<search_results>", "{out}");
    for (name, _, prompt) in BUILTINS {
        assert!(prompt.contains(&format!("Text inside {tag}")), "{name}");
    }
}

/// A panel line is comma-separated slugs, not YAML: a list or a trailing comment must fail as a
/// slug, never reach a provider.
#[test]
fn a_yaml_list_or_trailing_comment_in_panel_is_refused_as_an_invalid_slug() {
    let c = crate::council::cfg::Council::default();
    for v in ["[opus, astra]", "opus, astra # mine"] {
        let t = parse(&format!("---\npanel: {v}\n---\nbody")).unwrap();
        let Err(e) = crate::council::resolve::panel_slugs(&c, &[], &t.panel) else {
            panic!("panel: {v} resolved")
        };
        assert!(e.contains("invalid model slug"), "{v}: {e}");
    }
}

/// A frontmatter name reaches `index.md` on one line; no separator may survive into it.
#[test]
fn a_frontmatter_name_never_carries_a_line_break() {
    for sep in ["\r", "\u{85}", "\u{2028}", "\u{2029}"] {
        if let Ok(t) = parse(&format!("---\nname: a{sep}b\n---\nbody")) {
            assert!(
                !t.name.contains(|c: char| c == '\n' || sep.contains(c)),
                "{sep:?}"
            );
        }
    }
}

/// Above `--max-tokens`' ceiling every wire 400s, so a task file may not go past it either.
#[test]
fn a_task_file_cannot_raise_the_output_ceiling_above_the_flag_that_shares_it() {
    let ceiling = crate::council::run::MODEL_MAX_OUT;
    let at = parse(&format!("---\nmax_tokens: {ceiling}\n---\nbody {{doc}}"))
        .expect("the ceiling itself is allowed");
    assert_eq!(at.max_tokens, ceiling);
    let over = parse(&format!("---\nmax_tokens: {}\n---\nbody", ceiling + 1));
    assert!(
        over.is_err_and(|e| e.contains(&ceiling.to_string())),
        "a task file may not ask for more output than any wire accepts"
    );
}

/// Every `{doc}` is substituted, so 64 of them would turn a 1 MiB subject into a 64 MiB brief.
#[test]
fn a_task_that_multiplies_the_subject_is_refused_before_it_is_rendered() {
    let t = parse(&format!("---\nname: x\n---\n{}", "{doc}".repeat(64))).unwrap();
    let err = t.render(&"x".repeat(1024 * 1024)).unwrap_err();
    assert!(err.contains("repeats {doc} 64 times"), "{err}");
    assert!(t.render("small").is_ok());
}

/// A prefix would pick a paid prompt nobody named, so only an exact name resolves, and an unknown
/// one lists every task that exists, on disk too.
#[test]
fn a_task_name_resolves_only_exactly_and_an_unknown_one_lists_the_rest() {
    let tmp = tempfile::tempdir().unwrap();
    let _dir = dir_for_test(tmp.path());
    std::fs::write(tmp.path().join("mine.md"), "body").unwrap();
    assert_eq!(resolve("audit").unwrap().name, "audit");
    for name in ["aud", "code", "min", "zzz"] {
        let Err(unknown) = resolve(name) else {
            panic!("`{name}` is not a task name")
        };
        assert!(
            unknown.contains(&format!("unknown task '{name}'")),
            "{unknown}"
        );
        for (task, _, _) in BUILTINS {
            assert!(unknown.contains(task), "{task} missing from {unknown}");
        }
        assert!(unknown.contains("mine"), "{unknown}");
    }
}

/// Only a listed name resolves. APFS and NTFS open `audit.md` for `Audit`; a directory never lists.
#[test]
fn a_task_resolves_only_under_the_name_it_is_listed_by() {
    let tmp = tempfile::tempdir().unwrap();
    let _dir = dir_for_test(tmp.path());
    std::fs::write(tmp.path().join("audit.md"), "mine {doc}").unwrap();
    std::fs::create_dir(tmp.path().join("notes.md")).unwrap();
    for name in ["Audit", "notes"] {
        let err = resolve(name).map(|t| t.body).unwrap_err();
        assert!(err.contains(&format!("unknown task '{name}'")), "{err}");
    }
}

/// On Windows the config dir is roaming and the local one is not; one base for both.
#[test]
fn task_dir_and_council_json_share_one_base_directory() {
    let tasks = dir().expect("a config directory");
    let council = crate::council::cfg::path().expect("a config directory");
    assert_eq!(tasks.parent(), council.parent());
}

/// A task dir bx cannot search hides the operator's `audit.md`; running the built-in instead
/// would review with a prompt they replaced.
#[cfg(unix)]
#[test]
fn a_task_behind_an_unsearchable_directory_is_an_error_not_the_builtin() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = tempfile::tempdir().unwrap();
    let tasks = tmp.path().join("tasks");
    std::fs::create_dir(&tasks).unwrap();
    std::fs::write(tasks.join("audit.md"), "mine {doc}").unwrap();
    let _dir = dir_for_test(&tasks);
    for mode in [0o600, 0o000] {
        std::fs::set_permissions(&tasks, std::fs::Permissions::from_mode(mode)).unwrap();
        // Root searches any directory, so there is nothing to refuse.
        if std::fs::symlink_metadata(tasks.join("audit.md")).is_ok() {
            continue;
        }
        let err = resolve("audit").map(|t| t.body).unwrap_err();
        assert!(err.contains("audit.md"), "{mode:o}: {err}");
    }
    std::fs::set_permissions(&tasks, std::fs::Permissions::from_mode(0o700)).unwrap();
}

/// Forking `audit` by copying it out must not quietly cut its clock to the 900 s default.
#[test]
fn a_disk_task_shadowing_audit_keeps_audits_clock_unless_it_sets_one() {
    let tmp = tempfile::tempdir().unwrap();
    let _dir = dir_for_test(tmp.path());
    let builtin = builtin("audit").unwrap().max_seconds;
    assert!(builtin.is_some());
    std::fs::write(tmp.path().join("audit.md"), "mine {doc}").unwrap();
    let t = resolve("audit").unwrap();
    assert_eq!((t.body.as_str(), t.max_seconds), ("mine {doc}", builtin));
    std::fs::write(
        tmp.path().join("audit.md"),
        "---\nmax_seconds: 60\n---\nmine",
    )
    .unwrap();
    assert_eq!(resolve("audit").unwrap().max_seconds, Some(60));
    std::fs::write(tmp.path().join("other.md"), "mine").unwrap();
    assert_eq!(resolve("other").unwrap().max_seconds, None);
}

#[test]
fn a_task_file_that_does_not_parse_lists_as_invalid() {
    let tmp = tempfile::tempdir().unwrap();
    let _dir = dir_for_test(tmp.path());
    std::fs::write(tmp.path().join("broken.md"), "---\nnope: 1\n---\nbody").unwrap();
    let rows = on_disk_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, "broken");
    assert!(rows[0].1.starts_with("(invalid: "), "{}", rows[0].1);
    assert!(
        rows[0].1.contains("unknown frontmatter key"),
        "{}",
        rows[0].1
    );
}

#[test]
fn a_directory_named_like_a_task_file_is_not_a_task() {
    let tmp = tempfile::tempdir().unwrap();
    let _dir = dir_for_test(tmp.path());
    std::fs::create_dir(tmp.path().join("notes.md")).unwrap();
    std::fs::write(tmp.path().join("real.md"), "body").unwrap();
    assert_eq!(on_disk_rows(), [("real".to_string(), String::new())]);
}

/// A listed `x.MD` must load on a case-sensitive filesystem too.
#[test]
fn an_uppercase_md_task_file_that_is_listed_also_loads() {
    let tmp = tempfile::tempdir().unwrap();
    let _dir = dir_for_test(tmp.path());
    std::fs::write(
        tmp.path().join("shout.MD"),
        "---\ndescription: loud\n---\nbody\n",
    )
    .unwrap();
    assert_eq!(on_disk_rows(), [("shout".to_string(), "loud".to_string())]);
    assert_eq!(resolve("shout").unwrap().description, "loud");
}

/// An unreadable value is an error: `grounded: yes` must not quietly disable search.
#[test]
fn a_frontmatter_value_that_cannot_be_read_is_an_error_not_a_default() {
    let Err(e) = parse("---\ngrounded: yes\n---\nbody") else {
        panic!("true or false must be refused")
    };
    assert!(e.contains("true or false"), "{e}");
    let Err(e) = parse("---\nmax_seconds: 0\n---\nbody") else {
        panic!("max_seconds must be refused")
    };
    assert!(e.contains("max_seconds"), "{e}");
    let Err(e) = parse("---\neffort: turbo\n---\nbody") else {
        panic!("unknown effort must be refused")
    };
    assert!(e.contains("unknown effort"), "{e}");
    let t = parse("---\ngrounded: false\nmax_seconds: 2400\n---\nbody").unwrap();
    assert!(!t.grounded);
    assert_eq!(t.max_seconds, Some(2400));
}

/// Frontmatter is hand-written: blank lines and `#` comments are skipped.
#[test]
fn frontmatter_skips_blank_lines_and_comments() {
    let t = parse("---\n\n# which panel this is for\n\neffort: low\n---\nbody").unwrap();
    assert_eq!(t.effort.as_str(), "low");
}

/// The listing advertises `FRONTMATTER` and `parse` decides: the two must agree both ways.
#[test]
fn every_frontmatter_key_this_lists_is_one_parse_takes() {
    for key in FRONTMATTER {
        // A value each key accepts, so a refusal is about the key.
        let value = match *key {
            "grounded" => "true",
            "effort" => "high",
            "panel" => "opus",
            "name" | "description" => "x",
            _ => "1",
        };
        let raw = format!("---\n{key}: {value}\n---\nbody\n");
        assert!(
            parse(&raw).is_ok(),
            "FRONTMATTER lists `{key}` but parse rejects it"
        );
    }
    assert!(
        parse("---\nnot_a_key: 1\n---\nbody\n").is_err(),
        "parse accepts a key FRONTMATTER does not list"
    );
}

#[test]
fn a_frontmatter_key_given_twice_is_refused_even_with_the_same_value() {
    for key in FRONTMATTER {
        let value = match *key {
            "grounded" => "true",
            "effort" => "high",
            "panel" => "opus",
            "name" | "description" => "x",
            _ => "1",
        };
        let Err(e) = parse(&format!("---\n{key}: {value}\n{key}: {value}\n---\nbody")) else {
            panic!("{key} given twice was accepted")
        };
        assert!(e.contains("given twice"), "{key}: {e}");
    }
}

/// A quoted name reached the manifest and history with its quotes.
#[test]
fn a_quoted_frontmatter_value_means_the_text_inside() {
    let t = parse(
        "---\nname: \"mine\"\ndescription: 'find it'\nmax_seconds: \"60\"\npanel: \"opus, astra\"\n---\nbody",
    )
    .unwrap();
    assert_eq!(t.name, "mine");
    assert_eq!(t.description, "find it");
    assert_eq!(t.max_seconds, Some(60));
    assert_eq!(t.panel, ["opus", "astra"]);
    // Only a matching pair is a quote.
    assert_eq!(parse("---\nname: \"x'\n---\nb").unwrap().name, "\"x'");
    assert_eq!(parse("---\nname: \"\n---\nb").unwrap().name, "\"");
}

/// A task file declares the description the listing prints, sanitized for the terminal.
#[test]
fn a_task_file_can_declare_the_description_the_listing_prints() {
    let t = parse("---\ndescription: find the exploit\n---\nbody\n").unwrap();
    assert_eq!(t.description, "find the exploit");
    let esc = parse("---\ndescription: a\u{1b}[2Jb\n---\nbody\n").unwrap();
    assert!(
        !esc.description.contains('\u{1b}'),
        "an escape survived into the listing: {:?}",
        esc.description
    );
    // Absent is empty, never an error.
    assert_eq!(parse("body\n").unwrap().description, "");
}

/// An indented fence is refused, never loaded as a prompt made of YAML.
#[test]
fn frontmatter_indented_off_the_first_column_is_refused_not_swallowed() {
    // `unwrap_err` would need `Debug` on `Task`.
    let Err(err) = parse("    ---\n    panel: opus\n    ---\nbody\n") else {
        panic!("an indented fence was accepted");
    };
    assert!(err.contains("first column"), "{err}");
    assert!(parse("\t---\npanel: opus\n---\nbody\n").is_err());
    // A body that merely contains an indented `---` stays a body.
    assert_eq!(
        parse("body\n    ---\nmore\n").unwrap().body,
        "body\n    ---\nmore\n"
    );
}

/// A lone `---` never closed; a closing fence on the last line closes an empty body.
#[test]
fn a_fence_at_either_end_of_the_file_gets_the_message_for_its_fault() {
    for raw in ["---", "---\n", "---\n\n", "---\ndescription: d\n"] {
        let Err(e) = parse(raw) else {
            panic!("accepted {raw:?}")
        };
        assert_eq!(e, NEVER_CLOSED, "{raw:?}");
    }
    for raw in ["---\ndescription: d\n---", "---\n---", "---\n---\n"] {
        let Err(e) = parse(raw) else {
            panic!("accepted {raw:?}")
        };
        assert_eq!(e, "task body is empty", "{raw:?}");
    }
    // A padded or indented closing fence is no fence, never the start of a prompt.
    for raw in [
        "---\neffort: low\n--- \nbody",
        "---\neffort: low\n  ---\nbody",
    ] {
        assert!(parse(raw).is_err(), "accepted {raw:?}");
    }
}

/// The example the listing prints must parse and show what the prose claims it shows.
#[test]
fn the_example_the_listing_prints_is_a_task_that_parses() {
    let t = parse(EXAMPLE).expect("the example `bx council tasks` prints must parse");
    assert_eq!(t.panel, ["opus", "astra"]);
    assert_eq!(t.max_seconds, Some(2400));
    assert_eq!(t.max_searches, 25);
    assert!(
        !t.description.is_empty(),
        "the example declares no description"
    );
    assert!(t.body.contains("{doc}"), "{}", t.body);
    assert!(
        t.render("SUBJECT")
            .expect("the example renders")
            .contains("SUBJECT")
    );
    for key in ["description", "panel", "max_searches", "max_seconds"] {
        assert!(
            FRONTMATTER.contains(&key),
            "the example uses unlisted key {key}"
        );
    }
}

/// The size bound is the rendered length exactly, so a brief at the cap to the byte renders.
#[test]
fn the_brief_bound_measures_what_would_be_rendered() {
    let exactly = |body: &str, subject: &str| {
        let t = Task {
            name: "t".into(),
            body: body.into(),
            ..Task::default()
        };
        t.render(subject).map(|b| b.len())
    };
    let body = "{doc}";
    let subject = "x".repeat(MAX_BRIEF);
    assert_eq!(exactly(body, &subject), Ok(MAX_BRIEF));
    assert!(exactly(body, &"x".repeat(MAX_BRIEF + 1)).is_err());
    // Appended after a blank line: two bytes.
    let body = "b";
    assert_eq!(exactly(body, &"x".repeat(MAX_BRIEF - 3)), Ok(MAX_BRIEF));
    assert!(exactly(body, &"x".repeat(MAX_BRIEF - 2)).is_err());
    let e = exactly("b", &"x".repeat(MAX_BRIEF)).unwrap_err();
    assert!(e.contains("appends the subject"), "{e}");
    assert!(
        !e.contains("  "),
        "a user-visible error with a run of spaces: {e}"
    );
}

/// A resolved built-in carries the description it is listed under.
#[test]
fn a_builtin_task_resolves_with_the_description_it_is_listed_under() {
    for (name, description, _) in BUILTINS {
        let t = builtin(name).expect("a listed built-in resolves");
        assert_eq!(&t.description, description, "{name}");
        assert!(!t.description.is_empty(), "{name}");
    }
}

/// Frontmatter opens with exactly `---` on line one; every other spelling is refused.
#[test]
fn a_bom_after_a_blank_line_before_a_padded_fence_is_still_refused() {
    let Err(e) = parse("\n\u{feff}--- \ndescription: d\n---\nBody\n") else {
        panic!("a BOM hid the fence")
    };
    assert!(e.contains("frontmatter must be exactly"), "{e}");
}

#[test]
fn a_fence_that_is_not_exactly_three_dashes_on_line_one_is_refused() {
    let parse = super::parse;
    for bad in [
        "  ---\ndescription: d\n---\nBody\n",   // indented
        "--- \ndescription: d\n---\nBody\n",    // a trailing space
        "\n---\ndescription: d\n---\nBody\n",   // a blank line first
        "\n\n  --- \ndescription: d\n---\nB\n", // all three at once
    ] {
        let Err(e) = parse(bad) else {
            panic!("accepted {bad:?}")
        };
        assert!(e.contains("frontmatter must be exactly"), "{bad:?}: {e}");
    }
    let Ok(good) = parse("---\ndescription: d\n---\nBody\n") else {
        panic!("a clean fence must parse")
    };
    assert_eq!(good.description, "d");
    assert!(parse("Body that mentions --- in passing\n").is_ok());
}
