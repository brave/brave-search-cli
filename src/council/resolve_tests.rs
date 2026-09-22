//! Tests for `resolve.rs`, as a child module so they reach its private items.

use super::*;

/// `check_out` returns the normalised path it checked.
#[test]
fn an_out_path_is_normalised_through_every_component_kind() {
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().canonicalize().unwrap();
    let given = real.join("./a/../b/./c");
    assert_eq!(check_out(&given).unwrap(), real.join("b/c"));
    assert!(check_out(std::path::Path::new("/")).is_err());
}

/// Published paths are sanitized, so `re\u{AD}view` would publish `review`, another directory.
#[test]
fn an_out_path_that_sanitize_would_alter_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().canonicalize().unwrap();
    for name in ["re\u{ad}view", "re\u{200b}view", "re\u{1b}[2Jview"] {
        let e = check_out(&real.join(name)).unwrap_err();
        assert!(e.contains("pass a plain path"), "{name:?}: {e}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let raw = std::ffi::OsStr::from_bytes(b"re\xffview");
        assert!(check_out(&real.join(raw)).is_err(), "non-UTF-8");
    }
    assert!(check_out(&real.join("réview")).is_ok());
}

/// Artifact names are fixed, so a run would replace a docs site's `index.md`. Empty or new is
/// fine, and a held lock is `plan`'s to judge.
#[test]
fn out_into_a_directory_holding_a_run_artifact_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().canonicalize().unwrap().join("docs");
    assert!(check_out(&dir).is_ok(), "new");
    std::fs::create_dir(&dir).unwrap();
    assert_eq!(check_out(&dir).unwrap(), dir, "empty");
    for name in ARTIFACTS {
        std::fs::write(dir.join(name), "mine").unwrap();
        let e = check_out(&dir).unwrap_err();
        assert!(e.contains(name), "{e}");
        std::fs::remove_file(dir.join(name)).unwrap();
    }
    std::fs::write(dir.join("index.md"), "mine").unwrap();
    std::fs::write(dir.join(super::super::run::LOCK), "").unwrap();
    assert!(check_out(&dir).is_ok());
}

#[test]
fn an_oversized_cjk_subject_is_refused_for_its_size_not_its_encoding() {
    // Byte 4 is inside the second three-byte character.
    let e = read_capped("設計".as_bytes(), 3).unwrap_err();
    assert!(e.to_string().contains("exceeds maximum size"), "{e}");
    assert_eq!(read_capped("設計".as_bytes(), 6).unwrap(), "設計");
}

/// A roster is a fact about one host, trusted for `ROSTER_TTL` and no longer.
#[test]
fn a_roster_older_than_its_shelf_life_is_not_read_back() {
    let tmp = tempfile::tempdir().unwrap();
    let _root = state_dir_for_test(tmp.path());

    let models = vec![Served {
        id: "a/one".into(),
        max_out: None,
        efforts: None,
        effort_mandatory: false,
        adaptive: None,
    }];
    cache_roster("openrouter", "https://openrouter.ai", "k", true, &models);
    let fresh = cached_roster("openrouter", "https://openrouter.ai", "k", true);
    assert!(
        fresh.is_some_and(|m| m.len() == 1),
        "a fresh cache is read back"
    );
    assert!(
        cached_roster("openrouter", "http://127.0.0.1:9999", "k", true).is_none(),
        "a roster fetched from one host answered for another"
    );

    // Aged by its mtime, which is what `elapsed` reads.
    let path = roster_cache("openrouter", "https://openrouter.ai", "k", true).unwrap();
    let old = std::time::SystemTime::now() - (ROSTER_TTL + Duration::from_secs(60));
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(old))
        .unwrap();
    assert!(cached_roster("openrouter", "https://openrouter.ai", "k", true).is_none());
}

/// Another account may not be served what this one is: an `ft:` model, a gated preview.
#[test]
fn a_roster_cached_under_one_key_is_not_read_under_another() {
    let tmp = tempfile::tempdir().unwrap();
    let _root = state_dir_for_test(tmp.path());
    cache_roster("openai", "https://api.openai.com", "sk-a", false, &[]);
    assert!(cached_roster("openai", "https://api.openai.com", "sk-a", false).is_some());
    assert!(cached_roster("openai", "https://api.openai.com", "sk-b", false).is_none());
}

/// A near miss is a differing separator or a missing vendor prefix.
#[test]
fn a_near_miss_is_recognised_by_separator_or_by_a_missing_vendor_prefix() {
    assert_eq!(loose("claude-fable-5.1"), loose("claude-fable-5-1"));
    assert_ne!(loose("gpt-6-astra"), loose("gpt-6-sol"));
    assert_eq!(wire::bare_model("openai/gpt-6-astra"), "gpt-6-astra");
    assert_eq!(
        wire::bare_model("gpt-6-astra"),
        "gpt-6-astra",
        "no prefix to remove"
    );
}

/// OpenRouter combines routing variants in any order; only a catalog variant is listed.
#[test]
fn every_routing_variant_is_dropped_and_the_catalog_variant_kept() {
    for (asked, listed) in [
        ("a/b:nitro:exacto", "a/b"),
        ("a/b:nitro:free", "a/b:free"),
        ("a/b:free:floor", "a/b:free"),
        ("a/b:free", "a/b:free"),
        ("a/b", "a/b"),
    ] {
        assert_eq!(listed_id(asked), listed, "{asked}");
    }
}

/// Each wire judges the effort it will send (`minimal` goes out as `low`), not the typed word.
#[test]
fn each_wire_judges_a_model_against_the_request_it_would_build() {
    let served = |efforts: Option<Vec<&str>>, mandatory, adaptive| Served {
        id: "a/one".into(),
        max_out: None,
        efforts: efforts.map(|e| e.into_iter().map(str::to_string).collect()),
        effort_mandatory: mandatory,
        adaptive,
    };
    let or = |s: &Served, e| (wire::OPENROUTER.rejects)(s, e);
    // Nothing published: nothing to refuse.
    assert!(or(&served(None, false, None), Effort::Max).is_none());
    // Reasoning that cannot be turned off refuses `none`, and nothing else.
    assert!(or(&served(None, true, None), Effort::Off).is_some());
    assert!(or(&served(None, true, None), Effort::High).is_none());
    let three = served(Some(vec!["max", "high", "low"]), false, None);
    assert!(or(&three, Effort::Medium).is_some());
    assert!(
        or(&three, Effort::Minimal).is_none(),
        "sent as low, which is on the list"
    );
    assert!(or(&three, Effort::High).is_none());

    // Every Anthropic request asks for adaptive thinking, `none` included.
    let ant = |s: &Served, e| (wire::ANTHROPIC.rejects)(s, e);
    let old = served(Some(vec!["high"]), false, Some(false));
    assert!(ant(&old, Effort::High).is_some_and(|m| m.contains("adaptive")));
    assert!(
        ant(&served(Some(vec!["low"]), false, Some(false)), Effort::Off)
            .is_some_and(|m| m.contains("adaptive")),
        "none still asks for adaptive"
    );
    assert!(ant(&served(Some(vec!["high"]), false, Some(true)), Effort::High).is_none());
    // OpenAI sends the same `reasoning.effort`, so it is judged by the same rule, on what
    // `lend` borrowed; with nothing borrowed, nothing is refused.
    let oai = |s: &Served, e| (wire::OPENAI.rejects)(s, e);
    assert!(oai(&served(None, true, None), Effort::Off).is_some());
    assert!(oai(&three, Effort::Medium).is_some());
    assert!(oai(&three, Effort::Max).is_none(), "max is sent as max");
    assert!(oai(&served(None, false, None), Effort::Off).is_none());
}

/// OpenAI lists `none` as a level; OpenRouter's `none` disables reasoning, which only
/// `mandatory` refuses, so a Claude entry's list without it still passes there.
#[test]
fn effort_none_is_judged_against_the_list_on_openai_and_by_mandatory_on_openrouter() {
    let listed = |levels: &[&str]| Served {
        id: "a/one".into(),
        max_out: None,
        efforts: Some(levels.iter().map(|e| e.to_string()).collect()),
        effort_mandatory: false,
        adaptive: None,
    };
    let codex_mini = listed(&["high", "medium", "low"]);
    let why = (wire::OPENAI.rejects)(&codex_mini, Effort::Off).expect("none is not listed");
    assert!(why.contains("`--effort none`"), "{why}");
    assert!((wire::OPENAI.rejects)(&listed(&["none", "low"]), Effort::Off).is_none());
    assert!((wire::OPENROUTER.rejects)(&codex_mini, Effort::Off).is_none());
}

/// OpenAI's list is ids only; the public OpenRouter entry for `openai/<id>` supplies the ceiling
/// and efforts, and a model it does not list keeps nothing to judge.
#[test]
fn an_openai_model_borrows_its_constraints_from_its_public_openrouter_entry() {
    let public = [Served {
        id: "openai/gpt-x".into(),
        max_out: Some(100_000),
        efforts: Some(vec!["high".into()]),
        effort_mandatory: true,
        adaptive: None,
    }];
    let bare = |id: &str| Served {
        id: id.into(),
        ..Served::default()
    };
    let mut own = [bare("gpt-x"), bare("openai/gpt-x"), bare("gpt-y")];
    wire::lend("openai/", &mut own, &public);
    assert_eq!(own[0].max_out, Some(100_000));
    assert!(own[0].effort_mandatory);
    assert_eq!(own[0].efforts.as_deref(), Some(&["high".to_string()][..]));
    // Matched by the whole namespaced id, never by a suffix.
    assert!(own[1].efforts.is_none() && own[2].efforts.is_none());
}

/// `--effort none` sends `effort: "low"` on Anthropic, so a model taking no level is refused.
#[test]
fn anthropic_judges_effort_none_at_the_low_level_it_sends() {
    let served = |efforts: &[&str]| Served {
        id: "claude-sonnet-4-5".into(),
        max_out: None,
        efforts: Some(efforts.iter().map(|e| e.to_string()).collect()),
        effort_mandatory: false,
        adaptive: Some(true),
    };
    let ant = |s: &Served| (wire::ANTHROPIC.rejects)(s, Effort::Off);
    let why = ant(&served(&[])).expect("no effort level is accepted");
    assert!(why.contains("takes no effort level"), "{why}");
    assert!(!why.contains("(it takes: )"), "{why}");
    let why = ant(&served(&["high"])).expect("low is not on the list");
    assert!(
        why.contains("`--effort low`, which `--effort none` sends (it takes: high)"),
        "{why}"
    );
    assert!(ant(&served(&["low"])).is_none());
    assert_eq!(
        (wire::ANTHROPIC.effort)(Effort::Off)["output_config"]["effort"],
        "low",
        "the level judged is the level sent"
    );
}

/// Each wire's model list is a different document.
#[test]
fn each_wire_reads_its_own_model_list_schema() {
    // OpenRouter: `top_provider` and `reasoning`.
    let or = (wire::OPENROUTER.served)(&serde_json::json!({
        "id": "a/one", "top_provider": {"max_completion_tokens": 32000},
        "reasoning": {"mandatory": true, "supported_efforts": ["high", "low"]}
    }))
    .unwrap();
    assert_eq!((or.max_out, or.effort_mandatory), (Some(32000), true));
    assert_eq!(
        or.efforts.as_deref(),
        Some(&["high".to_string(), "low".to_string()][..])
    );

    // Anthropic: `max_tokens`, and capabilities that name the thinking types and effort levels.
    let ant = (wire::ANTHROPIC.served)(&serde_json::json!({
        "id": "claude-sonnet-4-5", "max_tokens": 64000,
        "capabilities": {"thinking": {"types": {"enabled": {"supported": true},
                                                "adaptive": {"supported": false}}},
                         "effort": {"supported": false}}
    }))
    .unwrap();
    assert_eq!(ant.max_out, Some(64000));
    assert_eq!(ant.adaptive, Some(false));
    assert_eq!(
        ant.efforts.as_deref(),
        Some(&[][..]),
        "no level is supported"
    );

    // OpenAI: an id and nothing to judge.
    let oai = (wire::OPENAI.served)(&serde_json::json!({"id": "gpt-6-astra", "object": "model"}))
        .unwrap();
    assert_eq!((oai.max_out, oai.adaptive, oai.efforts), (None, None, None));
}

/// A planted leaf link would carry every write past the rules that judge the resolved path.
#[test]
#[cfg(unix)]
fn an_out_leaf_that_is_a_symlink_is_refused_rather_than_written_through() {
    let tmp = tempfile::tempdir().unwrap();
    let real = tmp.path().canonicalize().unwrap();
    let target = real.join("target");
    std::fs::create_dir(&target).unwrap();

    let planted = real.join("planted");
    std::os::unix::fs::symlink(&target, &planted).unwrap();
    let e = check_out(&planted).unwrap_err();
    assert!(e.contains("symbolic link"), "{e}");
    assert_eq!(check_out(&target).unwrap(), target);

    // Dangling: `exists()` follows links, so this reaches the check as a missing leaf.
    let dangling = real.join("dangling");
    std::os::unix::fs::symlink(real.join("absent"), &dangling).unwrap();
    assert!(
        check_out(&dangling).is_err(),
        "a dangling leaf was accepted"
    );
    // An intermediate link is not refused: macOS `/tmp` and linked homes are ordinary.
    assert!(check_out(&planted.join("under")).is_ok());

    let file = real.join("notadir");
    std::fs::write(&file, "mine").unwrap();
    let e = check_out(&file).unwrap_err();
    assert!(e.contains("is not a directory"), "{e}");
    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        "mine",
        "it was written to"
    );
    assert!(check_out(&real.join("fresh")).is_ok());
}

/// The dry run blessed this and the real run then failed creating the directory.
#[test]
fn out_under_a_regular_file_is_refused_by_the_dry_run() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("f");
    std::fs::write(&file, "mine").unwrap();
    let e = check_out(&file.join("run")).unwrap_err();
    assert!(e.contains("is not a directory"), "{e}");
}
