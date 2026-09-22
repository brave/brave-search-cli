//! Tests for `config.rs`; a child module, so private items stay reachable.

use super::*;

/// True when no save left a temp file behind in `dir`.
#[cfg(unix)]
fn no_temps(dir: &Path) -> bool {
    fs::read_dir(dir)
        .unwrap()
        .flatten()
        .all(|e| !e.file_name().to_string_lossy().starts_with(".bx-"))
}

#[test]
fn parse_empty_object() {
    let c: Config = serde_json::from_str("{}").unwrap();
    assert!(c.api_key.is_none());
    assert!(c.base_url.is_none());
    assert!(c.timeout.is_none());
}

#[test]
fn parse_full_config() {
    let c: Config =
        serde_json::from_str(r#"{"api_key":"BSAtest","base_url":"https://x.com","timeout":60}"#)
            .unwrap();
    assert_eq!(c.api_key.as_deref(), Some("BSAtest"));
    assert_eq!(c.base_url.as_deref(), Some("https://x.com"));
    assert_eq!(c.timeout, Some(60));
}

#[test]
fn parse_partial_config() {
    let c: Config = serde_json::from_str(r#"{"timeout":10}"#).unwrap();
    assert!(c.api_key.is_none());
    assert!(c.base_url.is_none());
    assert_eq!(c.timeout, Some(10));
}

#[test]
fn parse_unknown_keys_rejected() {
    assert!(serde_json::from_str::<Config>(r#"{"api_key":"k","future_field":true}"#).is_err());
}

#[test]
fn parse_wrong_type_errors() {
    assert!(serde_json::from_str::<Config>(r#"{"timeout":"abc"}"#).is_err());
}

#[test]
fn parse_invalid_json() {
    assert!(serde_json::from_str::<Config>("{invalid").is_err());
}

#[test]
fn parse_nested_object_rejected() {
    assert!(serde_json::from_str::<Config>(r#"{"timeout":5,"nested":{"foo":1}}"#).is_err());
}

#[test]
fn serialize_round_trip() {
    let c = Config {
        api_key: Some("testkey123".into()),
        base_url: Some("https://x.com".into()),
        timeout: Some(45),
    };
    let s = serde_json::to_string_pretty(&c).unwrap();
    let c2: Config = serde_json::from_str(&s).unwrap();
    assert_eq!(c2.api_key.as_deref(), Some("testkey123"));
    assert_eq!(c2.base_url.as_deref(), Some("https://x.com"));
    assert_eq!(c2.timeout, Some(45));
}

#[test]
fn load_config_override_valid() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, r#"{"timeout":99}"#).unwrap();
    let c = load_config(Some(p.as_path())).unwrap();
    assert_eq!(c.timeout, Some(99));
}

#[test]
fn load_config_override_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, "").unwrap();
    let c = load_config(Some(p.as_path())).unwrap();
    assert!(c.api_key.is_none());
}

#[test]
fn load_config_whitespace_only_file_returns_default() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, "  \n  \n").unwrap();
    let c = load_config(Some(p.as_path())).unwrap();
    assert!(c.api_key.is_none());
    assert!(c.timeout.is_none());
}

/// An absent config is the default unless the caller named it; a named path that is missing is
/// an error, not a silent fallback to other settings.
#[test]
fn a_missing_config_is_a_default_only_when_nobody_named_it() {
    let dir = tempfile::tempdir().unwrap();
    assert!(load_config(Some(&dir.path().join("absent.json"))).is_err());
    // The default location may hold a real config; whatever is there must still parse.
    assert!(
        load_config(None).is_ok(),
        "the config at the default location does not load"
    );
}

#[test]
fn load_config_override_invalid_json_returns_err() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("bad.json");
    fs::write(&p, "{invalid").unwrap();
    assert!(load_config(Some(p.as_path())).is_err());
}

#[test]
fn load_config_override_missing_file_returns_err() {
    let p = Path::new("/nonexistent/path/config.json");
    assert!(load_config(Some(p)).is_err());
}

#[test]
fn save_and_load_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    let c = Config {
        api_key: Some("mykey12345".into()),
        base_url: None,
        timeout: Some(15),
    };
    save_config(&c, Some(p.as_path())).unwrap();
    let loaded = load_config(Some(p.as_path())).unwrap();
    assert_eq!(loaded.api_key.as_deref(), Some("mykey12345"));
    assert!(loaded.base_url.is_none());
    assert_eq!(loaded.timeout, Some(15));
}

#[test]
fn save_config_creates_parent_dirs() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("sub").join("dir").join("config.json");
    let c = Config {
        timeout: Some(1),
        ..Default::default()
    };
    save_config(&c, Some(p.as_path())).unwrap();
    assert!(p.exists());
}

#[test]
fn save_api_key_preserves_other_fields() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, r#"{"base_url":"https://x.com","timeout":20}"#).unwrap();
    save_api_key("newkey12345", Some(p.as_path())).unwrap();
    let c = load_config(Some(p.as_path())).unwrap();
    assert_eq!(c.api_key.as_deref(), Some("newkey12345"));
    assert_eq!(c.base_url.as_deref(), Some("https://x.com"));
    assert_eq!(c.timeout, Some(20));
}

#[test]
fn save_api_key_creates_new_file() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    assert!(!p.exists());
    save_api_key("newkey12345", Some(p.as_path())).unwrap();
    let c = load_config(Some(p.as_path())).unwrap();
    assert_eq!(c.api_key.as_deref(), Some("newkey12345"));
    assert!(c.base_url.is_none());
    assert!(c.timeout.is_none());
}

#[test]
fn save_api_key_validates_too_short() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    assert!(save_api_key("abc", Some(p.as_path())).is_err());
}

#[test]
fn validate_api_key_length_boundary() {
    assert!(validate_api_key("abcdefg").is_err()); // 7
    assert!(validate_api_key("abcdefgh").is_ok()); // 8
}

#[test]
fn validate_api_key_rejects_non_ascii() {
    // A pasted NBSP: the case the check exists for.
    assert!(validate_api_key("abcd\u{00a0}efgh").is_err());
    assert!(validate_api_key("clé_sécurisée").is_err());
}

#[test]
fn validate_api_key_accepts_punctuation() {
    // The rule is printable ASCII, not alphanumerics.
    assert!(validate_api_key("!abc~_-.7").is_ok());
}

#[test]
fn save_api_key_validates_whitespace() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    assert!(save_api_key("abc def ghi", Some(p.as_path())).is_err());
}

#[test]
fn save_api_key_validates_control_chars() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    assert!(save_api_key("abcdef\tgh", Some(p.as_path())).is_err());
}

#[test]
fn save_api_key_trims_whitespace() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    save_api_key("  testkey12345  ", Some(p.as_path())).unwrap();
    let c = load_config(Some(p.as_path())).unwrap();
    assert_eq!(c.api_key.as_deref(), Some("testkey12345"));
}

#[cfg(unix)]
#[test]
fn save_config_file_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    let c = Config {
        timeout: Some(1),
        ..Default::default()
    };
    save_config(&c, Some(p.as_path())).unwrap();
    let mode = fs::metadata(&p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
}

#[cfg(unix)]
#[test]
fn save_config_tightens_permissions_on_an_existing_file() {
    // `OpenOptions::mode` only applies on creation; an in-place write would keep the old mode.
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, "{}").unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(0o644)).unwrap();

    save_api_key("BSAtestkey123456", Some(p.as_path())).unwrap();

    let mode = fs::metadata(&p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "API key left in a world-readable file");
}

#[cfg(unix)]
#[test]
fn save_config_leaves_an_existing_directory_alone() {
    // `--config ./cfg.json` makes the CWD the parent, which is not ours to chmod.
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();

    save_api_key("BSAtestkey123456", Some(&dir.path().join("config.json"))).unwrap();

    let mode = fs::metadata(dir.path()).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o755, "chmodded a directory we did not create");
}

#[cfg(unix)]
#[test]
fn save_config_replaces_a_symlink_instead_of_writing_through_it() {
    let dir = tempfile::tempdir().unwrap();
    let canary = dir.path().join("canary");
    let link = dir.path().join("config.json");
    fs::write(&canary, "do not touch").unwrap();
    std::os::unix::fs::symlink(&canary, &link).unwrap();

    save_api_key("BSAtestkey123456", Some(link.as_path())).unwrap();

    assert_eq!(fs::read_to_string(&canary).unwrap(), "do not touch");
    assert!(!fs::symlink_metadata(&link).unwrap().is_symlink());
}

#[cfg(unix)]
#[test]
fn save_config_dir_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let sub = dir.path().join("sub");
    let p = sub.join("config.json");
    let c = Config::default();
    save_config(&c, Some(p.as_path())).unwrap();
    let mode = fs::metadata(&sub).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o700);
}

#[cfg(unix)]
#[test]
fn save_config_tightens_only_the_leaf_directory() {
    // Intermediate dirs keep their umask mode; umask varies, so compare with a dir made the
    // same way rather than a literal.
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let mid = dir.path().join("a");
    let leaf = mid.join("b");
    let reference = dir.path().join("reference");
    fs::create_dir(&reference).unwrap();

    save_config(&Config::default(), Some(&leaf.join("config.json"))).unwrap();

    let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&mid), mode(&reference), "chmodded an intermediate dir");
    assert_eq!(mode(&leaf), 0o700, "leaf dir not tightened");
}

#[test]
fn save_config_errors_when_the_path_has_no_parent() {
    assert!(save_config(&Config::default(), Some(Path::new("/"))).is_err());
}

#[test]
fn save_config_errors_when_the_parent_is_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("not-a-dir");
    fs::write(&file, "x").unwrap();
    assert!(save_config(&Config::default(), Some(&file.join("config.json"))).is_err());
}

#[test]
fn save_config_overwrites_an_existing_file() {
    // Not unix-only: the only coverage of rename-over-existing on Windows.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    let with_timeout = |t| Config {
        timeout: Some(t),
        ..Default::default()
    };
    save_config(&with_timeout(1), Some(p.as_path())).unwrap();
    save_config(&with_timeout(2), Some(p.as_path())).unwrap();
    assert_eq!(load_config(Some(p.as_path())).unwrap().timeout, Some(2));
}

#[cfg(unix)]
#[test]
fn save_config_replaces_a_read_only_file() {
    // `rename` needs write permission on the directory, not on the target.
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, "{}").unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(0o444)).unwrap();

    save_api_key("BSAtestkey123456", Some(p.as_path())).unwrap();

    let mode = fs::metadata(&p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
}

#[cfg(unix)]
#[test]
fn save_config_fails_cleanly_in_a_read_only_directory() {
    // The atomic write needs a writable directory. The failure must be clean: previous config
    // intact, no temp beside it.
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, "{}").unwrap();
    let chmod = |m| fs::set_permissions(dir.path(), fs::Permissions::from_mode(m)).unwrap();

    chmod(0o500);
    // Root, CAP_DAC_OVERRIDE and some container mounts ignore the mode bits entirely.
    if fs::File::create(dir.path().join(".probe")).is_ok() {
        chmod(0o700);
        return;
    }
    let saved = save_config(&Config::default(), Some(p.as_path()));
    chmod(0o700); // before the asserts: a panic must not leave the dir unremovable

    assert!(saved.is_err());
    assert_eq!(fs::read_to_string(&p).unwrap(), "{}");
    assert!(no_temps(dir.path()));
}

#[cfg(unix)]
#[test]
fn save_config_severs_a_hard_link() {
    // The rename replaces the directory entry, so the other name keeps the old inode.
    let dir = tempfile::tempdir().unwrap();
    let other = dir.path().join("other.json");
    let p = dir.path().join("config.json");
    fs::write(&other, r#"{"timeout":1}"#).unwrap();
    if fs::hard_link(&other, &p).is_err() {
        return; // no hard links on this filesystem
    }

    save_api_key("BSAtestkey123456", Some(p.as_path())).unwrap();

    assert_eq!(fs::read_to_string(&other).unwrap(), r#"{"timeout":1}"#);
}

#[cfg(unix)]
#[test]
fn save_config_replaces_a_dangling_symlink() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let link = dir.path().join("config.json");
    std::os::unix::fs::symlink(dir.path().join("gone"), &link).unwrap();

    save_api_key("BSAtestkey123456", Some(link.as_path())).unwrap();

    assert!(!fs::symlink_metadata(&link).unwrap().is_symlink());
    let mode = fs::metadata(&link).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    assert!(!dir.path().join("gone").exists(), "wrote through the link");
}

#[cfg(unix)]
#[test]
fn a_failed_save_leaves_no_temp_file_behind() {
    // The rename fails late (the target is a directory), and must leave no temp file behind.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::create_dir(&p).unwrap();

    assert!(save_config(&Config::default(), Some(p.as_path())).is_err());

    assert!(no_temps(dir.path()));
    assert!(p.is_dir(), "target was replaced");
}

#[test]
fn create_private_does_not_unlink_a_file_it_did_not_create() {
    // A name another live save holds must never be removed by this one.
    let dir = tempfile::tempdir().unwrap();
    let taken = dir.path().join(".bx-999999-1.tmp");
    fs::write(&taken, "theirs").unwrap();

    let err = create_private(&taken, "ours").unwrap_err();

    assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
    assert_eq!(fs::read_to_string(&taken).unwrap(), "theirs");
}

#[test]
fn a_leftover_temp_file_does_not_block_a_later_save() {
    // A save killed before its rename leaves one behind. It is 0600 in a 0700 dir, so it is
    // left alone: sweeping would race concurrent saves.
    let dir = tempfile::tempdir().unwrap();
    let leftover = dir.path().join(".bx-999999-1.tmp");
    fs::write(&leftover, r#"{"api_key":"BSAoldkey123456789"}"#).unwrap();

    save_api_key("BSAtestkey123456", Some(&dir.path().join("config.json"))).unwrap();

    assert!(leftover.exists());
}

#[test]
fn try_remove_file_existing() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("legacy_key");
    fs::write(&p, "test").unwrap();
    assert!(p.exists());
    try_remove_file(&p);
    assert!(!p.exists());
}

#[test]
fn try_remove_file_missing() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("nonexistent");
    try_remove_file(&p); // should not panic
}

#[test]
fn mask_long_key() {
    // Real keys are 30+ characters, so 8 shown is a safe fingerprint.
    assert_eq!(mask_key("BSA_abcdefghijklmnopqrstuvwxyz12"), "BSA_...yz12");
}

#[test]
fn mask_never_reveals_most_of_a_short_key() {
    // At 9 characters, first-4 + last-4 would leave one character secret.
    assert_eq!(mask_key("abcdefghi"), "abcd...");
    assert_eq!(mask_key("abcdefghijklmnop"), "abcd...");
    assert_eq!(mask_key("abcdefghijklmnopq"), "abcd...nopq");
}

#[test]
fn mask_never_slices_across_a_character() {
    // Both slicing arms with byte 4 inside `é` (18 and 10 bytes): without the ASCII guards
    // these panic.
    assert_eq!(mask_key("abcé_defghijklmno"), "****...");
    assert_eq!(mask_key("abcé_defg"), "****...");
}

#[test]
fn mask_hides_keys_too_short_to_fingerprint() {
    assert_eq!(mask_key("abcdefgh"), "****...");
    assert_eq!(mask_key("abcd"), "****...");
    assert_eq!(mask_key("a"), "****...");
    assert_eq!(mask_key(""), "****...");
}

#[test]
fn mask_non_ascii() {
    assert_eq!(mask_key("clé_sécurisée"), "****...");
}

#[test]
fn trim_non_empty_normal() {
    assert_eq!(trim_non_empty("hello".into()), Some("hello".into()));
}

#[test]
fn trim_non_empty_with_whitespace() {
    assert_eq!(trim_non_empty("  hello  ".into()), Some("hello".into()));
}

#[test]
fn trim_non_empty_empty() {
    assert_eq!(trim_non_empty(String::new()), None);
}

#[test]
fn trim_non_empty_whitespace_only() {
    assert_eq!(trim_non_empty("   ".into()), None);
}

#[test]
fn load_api_key_for_display_from_config() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, r#"{"api_key":"testkey12345"}"#).unwrap();
    let key = load_api_key_for_display(Some(p.as_path()));
    assert_eq!(key.as_deref(), Some("testkey12345"));
}

#[test]
fn config_path_ends_with_config_json() {
    if let Some(p) = config_path() {
        assert!(p.ends_with("config.json"));
        assert!(p.parent().unwrap().ends_with("brave-search"));
    }
}

#[test]
fn parse_non_object_root_rejected() {
    assert!(serde_json::from_str::<Config>("null").is_err());
    assert!(serde_json::from_str::<Config>(r#""hello""#).is_err());
    assert!(serde_json::from_str::<Config>("42").is_err());
}

#[test]
fn parse_timeout_negative_rejected() {
    assert!(serde_json::from_str::<Config>(r#"{"timeout":-1}"#).is_err());
}

#[cfg(unix)]
#[test]
fn load_config_override_permission_denied_returns_err() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, r#"{"timeout":10}"#).unwrap();
    fs::set_permissions(&p, fs::Permissions::from_mode(0o000)).unwrap();
    assert!(load_config(Some(p.as_path())).is_err());
    fs::set_permissions(&p, fs::Permissions::from_mode(0o644)).unwrap();
}

#[test]
fn load_config_override_non_object_returns_err() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, r#""hello""#).unwrap();
    assert!(load_config(Some(p.as_path())).is_err());
}

#[test]
fn load_config_override_empty_object() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, "{}").unwrap();
    let c = load_config(Some(p.as_path())).unwrap();
    assert!(c.api_key.is_none());
    assert!(c.timeout.is_none());
}

#[test]
fn save_api_key_drops_unknown_fields() {
    // Deliberate: `deny_unknown_fields` makes a newer bx's config unreadable, and the save
    // overwrites it with a warning.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, r#"{"timeout":42,"from_the_future":true}"#).unwrap();

    save_api_key("BSAtestkey123456", Some(p.as_path())).unwrap();

    let c = load_config(Some(p.as_path())).unwrap();
    assert_eq!(c.api_key.as_deref(), Some("BSAtestkey123456"));
    assert!(c.timeout.is_none(), "kept a field it could not parse");
}

#[test]
fn save_api_key_with_corrupt_config_still_saves() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, "{invalid").unwrap();
    save_api_key("newkey12345", Some(p.as_path())).unwrap();
    let c = load_config(Some(p.as_path())).unwrap();
    assert_eq!(c.api_key.as_deref(), Some("newkey12345"));
}

// These pass `Some(path)`, so the legacy cleanup is skipped and the real key file untouched.
#[test]
fn migrate_legacy_key_saves_to_config() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    migrate_legacy_key("testkey12345", Some(p.as_path())).unwrap();
    let c = load_config(Some(p.as_path())).unwrap();
    assert_eq!(c.api_key.as_deref(), Some("testkey12345"));
}

#[test]
fn migrate_legacy_key_preserves_other_fields() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    fs::write(&p, r#"{"timeout":42}"#).unwrap();
    migrate_legacy_key("testkey12345", Some(p.as_path())).unwrap();
    let c = load_config(Some(p.as_path())).unwrap();
    assert_eq!(c.api_key.as_deref(), Some("testkey12345"));
    assert_eq!(c.timeout, Some(42));
}

#[test]
fn migrate_legacy_key_invalid_key_returns_err() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("config.json");
    assert!(migrate_legacy_key("short", Some(p.as_path())).is_err());
    assert!(!p.exists());
}
