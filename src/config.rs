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
/// on unix. A failed write removes the file again, so no partial API key is left behind.
fn create_private(path: &Path, contents: &str) -> io::Result<()> {
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts.open(path)?;
    // sync before the caller renames, or a crash can rename an empty config into place.
    file.write_all(contents.as_bytes())
        .and_then(|()| file.sync_all())
        .inspect_err(|_| {
            fs::remove_file(path).ok();
        })
}

/// Saves the config, readable only by its owner on unix.
fn save_config(config: &Config, override_path: Option<&Path>) -> io::Result<()> {
    let path = resolve_config_path(override_path)
        .ok_or_else(|| io::Error::other("cannot determine config directory"))?;
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other("config path has no parent directory"))?;
    // Tighten the default dir every save (a restore can leave it 0755) and any dir we
    // create, never an existing `--config` parent. Probed before `create_dir_all`, which
    // would make every dir look pre-existing.
    #[cfg(unix)]
    let tighten = !dir.as_os_str().is_empty() && (override_path.is_none() || !dir.exists());
    fs::create_dir_all(dir)?;
    #[cfg(unix)]
    if tighten {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }

    let contents = serde_json::to_string_pretty(config).map_err(io::Error::other)?;

    // Temp + rename: atomic, and 0600 sticks because the file is new — `OpenOptions::mode`
    // is a no-op on an existing file. A symlink at `path` is replaced, not written through.
    // Pid is unique among live saves; the clock only guards a recycled pid's leftover.
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let tmp = path.with_file_name(format!(".bx-{}-{stamp}.tmp", std::process::id()));
    create_private(&tmp, &contents)?;
    fs::rename(&tmp, &path).inspect_err(|_| {
        fs::remove_file(&tmp).ok();
    })
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
    // `--config` cannot redirect the legacy path, so removing it while saving elsewhere
    // would delete a key the default runs still need.
    if config_path.is_none() {
        remove_legacy_key_file();
    }
    Ok(())
}

/// Validates an API key before saving. A non-ASCII byte — a non-breaking space from a
/// copy-paste — survives every later check and 401s in a way `show-key` cannot explain.
fn validate_api_key(key: &str) -> io::Result<()> {
    if key.len() < 8 {
        return Err(io::Error::other(
            "API key is too short (expected at least 8 characters)",
        ));
    }
    if !key.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(io::Error::other(
            "API key must be printable ASCII without spaces",
        ));
    }
    Ok(())
}

/// Saves the API key into the JSON config file (read-modify-write).
fn save_api_key(key: &str, config_path: Option<&Path>) -> io::Result<()> {
    let trimmed = key.trim();
    validate_api_key(trimmed)?;
    let mut config = load_config(config_path).unwrap_or_else(|e| {
        // Only a `--config` target that is genuinely absent is a normal first run; a stat
        // that fails is not, and a missing *default* config never reaches this closure.
        if config_path.is_none_or(|p| p.try_exists().unwrap_or(true)) {
            eprintln!("warning: {e}; other settings may be reset");
        }
        Config::default()
    });
    config.api_key = Some(trimmed.to_string());
    save_config(&config, config_path)
}

/// Masks an API key: at most 8 characters, and never half of one — first-4 needs 9
/// characters, first-4 plus last-4 needs 17. The ASCII guards keep the slicing safe.
fn mask_key(key: &str) -> String {
    match key.len() {
        17.. if key.is_ascii() => format!("{}...{}", &key[..4], &key[key.len() - 4..]),
        9..=16 if key.is_ascii() => format!("{}...", &key[..4]),
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
                        std::process::exit(2);
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
                    std::process::exit(2);
                }
            }
        }
        super::ConfigCmd::ShowKey => match load_api_key_for_display(config_path) {
            Some(key) => println!("{}", mask_key(&key)),
            None => {
                eprintln!("no API key configured");
                std::process::exit(2);
            }
        },
        super::ConfigCmd::Path => match resolve_config_path(config_path) {
            Some(p) => println!("{}", p.display()),
            None => {
                eprintln!("error: cannot determine config directory");
                std::process::exit(2);
            }
        },
        super::ConfigCmd::Show => {
            let config = match load_config(config_path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(2);
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

    /// True when no save left a temp file behind in `dir`.
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
    fn validate_api_key_length_boundary() {
        assert!(validate_api_key("abcdefg").is_err()); // 7
        assert!(validate_api_key("abcdefgh").is_ok()); // 8
    }

    #[test]
    fn validate_api_key_rejects_non_ascii() {
        // The paste-from-a-web-page case the check exists for.
        assert!(validate_api_key("abcd\u{00a0}efgh").is_err());
        assert!(validate_api_key("clé_sécurisée").is_err());
    }

    #[test]
    fn validate_api_key_accepts_punctuation() {
        // The rule is printable ASCII, not alphanumerics — do not tighten it further.
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
    fn save_config_tightens_only_the_leaf_directory() {
        // Intermediate dirs keep their umask mode. A literal would be flaky — umask is 022
        // on CI and 002 on many desktops — so compare with a dir created the same way.
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
        // Deliberately not unix-only: the rename-over-existing path has no other coverage
        // on Windows, which CI cross-compiles but never runs.
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
        // `rename` needs write on the directory, not on the target, so this succeeds where
        // the old truncate-in-place write failed outright.
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
        // The atomic write needs a writable directory, where the old in-place write did
        // not. Pin the failure as clean: previous config intact, no temp beside it.
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
        // The rename is what fails late — here because the target is a directory, which
        // it must still be afterwards, with no temp file left beside it.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.json");
        fs::create_dir(&p).unwrap();

        assert!(save_config(&Config::default(), Some(p.as_path())).is_err());

        assert!(no_temps(dir.path()));
        assert!(p.is_dir(), "target was replaced");
    }

    #[test]
    fn create_private_does_not_unlink_a_file_it_did_not_create() {
        // The cleanup lives here, not beside the rename, precisely so a name another live
        // save already holds is never removed by this one.
        let dir = tempfile::tempdir().unwrap();
        let taken = dir.path().join(".bx-999999-1.tmp");
        fs::write(&taken, "theirs").unwrap();

        let err = create_private(&taken, "ours").unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read_to_string(&taken).unwrap(), "theirs");
    }

    #[test]
    fn a_leftover_temp_file_does_not_block_a_later_save() {
        // A save killed before its rename leaves one behind. It is 0600 in a 0700 dir, so
        // it is left alone: sweeping the directory raced concurrent saves for no gain.
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
    fn mask_never_slices_across_a_character() {
        // Both slicing arms, with byte 4 inside `é`: without their ASCII guards these
        // panic rather than mask. Lengths are in bytes — 18 and 10.
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
        // `deny_unknown_fields` makes a config written by a newer bx unreadable, and the
        // save then overwrites it. Deliberate — the warning says so — so pin it.
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

    // These pass `Some(path)`, so the legacy cleanup is skipped and the developer's real
    // key file is never touched. Before that guard they deleted it on every `cargo test`.
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
}
