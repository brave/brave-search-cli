use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

// ── Config struct ────────────────────────────────────────────────────

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
}

// ── Paths ────────────────────────────────────────────────────────────

/// Returns the config directory: ~/.config/brave-search/ (or platform equivalent).
fn config_dir() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("brave-search"))
}

/// Returns the path to the JSON config file.
fn config_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("config.json"))
}

/// Returns the resolved config file path (override or default).
fn resolve_config_path(override_path: Option<&Path>) -> Option<PathBuf> {
    override_path.map(Path::to_path_buf).or_else(config_path)
}

/// Returns the path to the legacy API key file.
fn legacy_key_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("api_key"))
}

// ── Load / save ──────────────────────────────────────────────────────

/// Loads the JSON config file. Returns `Config::default()` when the default
/// path does not exist. Returns an error for any other failure.
pub fn load_config(override_path: Option<&Path>) -> Result<Config, String> {
    let is_explicit = override_path.is_some();
    let path = match resolve_config_path(override_path) {
        Some(p) => p,
        None => return Ok(Config::default()),
    };

    match fs::read_to_string(&path) {
        Ok(contents) => {
            if contents.trim().is_empty() {
                return Ok(Config::default());
            }
            serde_json::from_str::<Config>(&contents)
                .map_err(|e| format!("failed to parse {}: {e}", path.display()))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            if is_explicit {
                Err(format!("config file not found: {}", path.display()))
            } else {
                Ok(Config::default())
            }
        }
        Err(e) => Err(format!("cannot read {}: {e}", path.display())),
    }
}

/// Writes `contents` to `path`, which must not already exist, readable only by its owner
/// on unix. A failed write removes the file again, best-effort, so a partial API key is
/// not left lying around.
fn create_private(path: &Path, contents: &str) -> io::Result<()> {
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts.open(path)?;
    // sync before the caller renames: otherwise a crash can leave the rename applied and
    // the contents not — a config that is present, atomic, and empty.
    let written = file
        .write_all(contents.as_bytes())
        .and_then(|()| file.sync_all());
    if written.is_err() {
        fs::remove_file(path).ok();
    }
    written
}

/// Saves the config, readable only by its owner on unix.
fn save_config(config: &Config, override_path: Option<&Path>) -> io::Result<()> {
    let path = resolve_config_path(override_path)
        .ok_or_else(|| io::Error::other("cannot determine config directory"))?;
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other("config path has no parent directory"))?;
    // Tighten a directory we created, and the default one on every save — a restore that
    // left it 0755 lets other users list the key file. Never an existing `--config`
    // parent, which may be the working directory; an empty parent is the working
    // directory, from a bare `--config config.json`.
    #[cfg(unix)]
    let tighten = !dir.as_os_str().is_empty() && (!dir.exists() || override_path.is_none());
    fs::create_dir_all(dir)?;
    #[cfg(unix)]
    if tighten {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }

    let contents = serde_json::to_string_pretty(config).map_err(io::Error::other)?;

    // Write then rename: the rename is atomic, so an interrupted run cannot truncate the
    // config, the 0600 mode always applies because the file is one we just created, and a
    // symlink at `path` is replaced rather than written through. Pid and clock keep a
    // concurrent `set-key`, or a leftover from a killed one, from blocking this save, and
    // the name replaces the config's basename rather than extending it, so a long one
    // cannot push it past NAME_MAX.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let tmp = path.with_file_name(format!(".bx-{}-{stamp}.tmp", std::process::id()));
    create_private(&tmp, &contents)?;
    let renamed = fs::rename(&tmp, &path);
    if renamed.is_err() {
        fs::remove_file(&tmp).ok();
    }
    renamed
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Returns the trimmed string if non-empty.
pub(crate) fn trim_non_empty(s: String) -> Option<String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Loads the API key from the legacy bare `api_key` file, if it exists.
pub fn load_legacy_api_key() -> Option<String> {
    let path = legacy_key_path()?;
    fs::read_to_string(path).ok().and_then(trim_non_empty)
}

/// Best-effort removal of a file; logs to stderr on success or non-trivial failure.
fn try_remove_file(path: &Path) {
    match fs::remove_file(path) {
        Ok(()) => eprintln!("note: removed legacy {}", path.display()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => eprintln!("warning: could not remove {}: {e}", path.display()),
    }
}

/// Removes the legacy bare `api_key` file, if it exists.
fn remove_legacy_key_file() {
    if let Some(p) = legacy_key_path() {
        try_remove_file(&p);
    }
}

/// Saves an API key to the config file and removes any legacy key file.
/// On save failure the legacy file is left in place.
pub fn migrate_legacy_key(key: &str, config_path: Option<&Path>) -> io::Result<()> {
    save_api_key(key, config_path)?;
    remove_legacy_key_file();
    Ok(())
}

/// Validates that an API key looks reasonable before saving.
fn validate_api_key(key: &str) -> io::Result<()> {
    if key.len() < 8 {
        return Err(io::Error::other(
            "API key is too short (expected at least 8 characters)",
        ));
    }
    if key.bytes().any(|b| b.is_ascii_control() || b == b' ') {
        return Err(io::Error::other(
            "API key contains whitespace or control characters",
        ));
    }
    Ok(())
}

/// Saves the API key into the JSON config file (read-modify-write).
fn save_api_key(key: &str, config_path: Option<&Path>) -> io::Result<()> {
    let trimmed = key.trim();
    validate_api_key(trimmed)?;
    let mut config = load_config(config_path).unwrap_or_else(|e| {
        // Silent when nothing is there to lose: on a first `set-key` the config not
        // existing yet is the normal case, not a warning.
        if resolve_config_path(config_path).is_some_and(|p| p.exists()) {
            eprintln!("warning: {e}; other settings may be reset");
        }
        Config::default()
    });
    config.api_key = Some(trimmed.to_string());
    save_config(&config, config_path)
}

/// Masks an API key: enough to recognise which key it is, never enough to reconstruct it.
/// Never reveals half — first-4 plus last-4 needs 17 characters, first-4 alone needs 9.
fn mask_key(key: &str) -> String {
    match key.len() {
        _ if !key.is_ascii() => "****...".into(),
        17.. => format!("{}...{}", &key[..4], &key[key.len() - 4..]),
        9..=16 => format!("{}...", &key[..4]),
        _ => "****...".into(),
    }
}

/// Loads the API key from the config file, falling back to the legacy file.
fn load_api_key_for_display(config_path: Option<&Path>) -> Option<String> {
    load_config(config_path)
        .unwrap_or_else(|e| {
            eprintln!("warning: {e}");
            Config::default()
        })
        .api_key
        .and_then(trim_non_empty)
        .or_else(load_legacy_api_key)
}

// ── Onboarding ───────────────────────────────────────────────────────

const SETUP_MSG: &str = "\
No API key found. To get started:

  1. Sign up at https://api-dashboard.search.brave.com/register
  2. Choose a plan — every plan includes $5/month free credits (~1,000 free queries)
     Note: different endpoints may require different plans (e.g. Search vs Answers)
  3. Go to \"API Keys\" in the dashboard and generate a key

Then configure it (pick one):

  bx config set-key <YOUR_KEY>
  export BRAVE_SEARCH_API_KEY=<YOUR_KEY>
  bx --api-key <YOUR_KEY> web \"test query\"";

/// Prompts and reads an API key from stdin.
fn read_key_line() -> Result<String, String> {
    eprintln!("(input will be visible — to avoid, set BRAVE_SEARCH_API_KEY env var instead)");
    eprint!("Paste your API key: ");
    io::stderr().flush().ok();

    let mut line = String::new();
    io::stdin()
        .lock()
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    let key = line.trim().to_string();

    if key.is_empty() {
        return Err("no API key provided".into());
    }

    Ok(key)
}

/// Interactive onboarding when no API key is found.
pub fn onboard(config_path: Option<&Path>) -> Result<String, String> {
    eprintln!("{SETUP_MSG}");

    if !io::stdin().is_terminal() {
        return Err("no API key configured".into());
    }

    eprintln!();
    if let Some(p) = resolve_config_path(config_path) {
        eprintln!("Your key will be saved to {}", p.display());
    }
    let key = read_key_line()?;

    migrate_legacy_key(&key, config_path).map_err(|e| format!("failed to save API key: {e}"))?;
    if let Some(p) = resolve_config_path(config_path) {
        eprintln!("API key saved to {}", p.display());
    }

    Ok(key)
}

/// Prompts for an API key on stdin (TTY required).
fn prompt_api_key() -> Result<String, String> {
    if !io::stdin().is_terminal() {
        return Err("no key argument provided and stdin is not a terminal".into());
    }
    read_key_line()
}

// ── Config subcommand handler ────────────────────────────────────────

/// Handles the `config` subcommand.
pub fn handle_config(cmd: &super::ConfigCmd, config_path: Option<&Path>) {
    match cmd {
        super::ConfigCmd::SetKey { key } => {
            let resolved = match key {
                Some(k) => k.clone(),
                None => match prompt_api_key() {
                    Ok(k) => k,
                    Err(e) => {
                        eprintln!("error: {e}");
                        std::process::exit(1);
                    }
                },
            };
            match migrate_legacy_key(&resolved, config_path) {
                Ok(()) => {
                    if let Some(p) = resolve_config_path(config_path) {
                        eprintln!("API key saved to {}", p.display());
                    }
                }
                Err(e) => {
                    eprintln!("error: failed to save API key: {e}");
                    std::process::exit(1);
                }
            }
        }
        super::ConfigCmd::ShowKey => match load_api_key_for_display(config_path) {
            Some(key) => println!("{}", mask_key(&key)),
            None => {
                eprintln!("no API key configured");
                std::process::exit(1);
            }
        },
        super::ConfigCmd::Path => match resolve_config_path(config_path) {
            Some(p) => println!("{}", p.display()),
            None => {
                eprintln!("error: cannot determine config directory");
                std::process::exit(1);
            }
        },
        super::ConfigCmd::Show => {
            let config = match load_config(config_path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            };
            let api_key = config
                .api_key
                .and_then(trim_non_empty)
                .or_else(load_legacy_api_key);
            if api_key.is_none() && config.base_url.is_none() && config.timeout.is_none() {
                eprintln!("(no configuration found)");
                return;
            }
            if let Some(ref key) = api_key {
                println!("api_key = {}", mask_key(key));
            }
            if let Some(ref url) = config.base_url {
                println!("base_url = {url}");
            }
            if let Some(t) = config.timeout {
                println!("timeout = {t}");
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_object() {
        let c: Config = serde_json::from_str("{}").unwrap();
        assert!(c.api_key.is_none());
        assert!(c.base_url.is_none());
        assert!(c.timeout.is_none());
    }

    #[test]
    fn parse_full_config() {
        let c: Config = serde_json::from_str(
            r#"{"api_key":"BSAtest","base_url":"https://x.com","timeout":60}"#,
        )
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

    #[test]
    fn load_config_default_missing_returns_default() {
        load_config(None).unwrap();
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
        // `OpenOptions::mode` only applies to a file it creates, so a config that already
        // existed world-readable would have taken the API key at its old mode.
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
        // `--config ./cfg.json` makes the CWD the parent. Chmodding it to 0700 because we
        // happened to write a file in it is not ours to do.
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
    fn a_failed_save_leaves_no_temp_file_behind() {
        // The rename is what fails late — here because the target is a directory, which
        // it must still be afterwards, with no temp file left beside it.
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        fs::create_dir(&p).unwrap();

        assert!(save_config(&Config::default(), Some(p.as_path())).is_err());

        let leftovers: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .filter(|n| n.to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "left behind {leftovers:?}");
        assert_eq!(
            fs::metadata(&p).unwrap().permissions().mode() & 0o170000,
            0o40000
        );
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
        // A real Brave key is 30+ characters: 8 shown out of 32 is a safe fingerprint.
        assert_eq!(mask_key("BSA_abcdefghijklmnopqrstuvwxyz12"), "BSA_...yz12");
    }

    #[test]
    fn mask_never_reveals_most_of_a_short_key() {
        // The boundary that matters: at 9 characters, first-4 + last-4 would leave exactly
        // one character secret.
        assert_eq!(mask_key("abcdefghi"), "abcd...");
        assert_eq!(mask_key("abcdefghijklmnop"), "abcd...");
        assert_eq!(mask_key("abcdefghijklmnopq"), "abcd...nopq");
    }

    #[test]
    fn mask_hides_keys_too_short_to_fingerprint() {
        assert_eq!(mask_key("abcdefgh"), "****...");
        assert_eq!(mask_key("abcd"), "****...");
        assert_eq!(mask_key("a"), "****...");
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
    fn save_api_key_with_corrupt_config_still_saves() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        fs::write(&p, "{invalid").unwrap();
        save_api_key("newkey12345", Some(p.as_path())).unwrap();
        let c = load_config(Some(p.as_path())).unwrap();
        assert_eq!(c.api_key.as_deref(), Some("newkey12345"));
    }

    #[test]
    // The only `migrate_legacy_key` test: the others duplicated `save_api_key`'s and, by
    // reaching `remove_legacy_key_file`, deleted the developer's real key file on every
    // `cargo test` — that function resolves the real config dir, which `--config` cannot
    // redirect. This one returns before it.
    fn migrate_legacy_key_invalid_key_returns_err() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        assert!(migrate_legacy_key("short", Some(p.as_path())).is_err());
        assert!(!p.exists());
    }
}
