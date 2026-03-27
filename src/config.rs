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

/// Loads the JSON config file. Returns `Config::default()` if the default path
/// is missing. Returns an error if an explicit `--config` path is missing or unreadable.
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
            match serde_json::from_str::<Config>(&contents) {
                Ok(cfg) => Ok(cfg),
                Err(e) => {
                    if is_explicit {
                        return Err(format!("failed to parse {}: {e}", path.display()));
                    }
                    eprintln!("warning: failed to parse {}: {e}", path.display());
                    Ok(Config::default())
                }
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            if is_explicit {
                Err(format!("config file not found: {}", path.display()))
            } else {
                Ok(Config::default())
            }
        }
        Err(e) => {
            if is_explicit {
                return Err(format!("cannot read {}: {e}", path.display()));
            }
            eprintln!("warning: cannot read {}: {e}", path.display());
            Ok(Config::default())
        }
    }
}

/// Saves the config to a JSON file with restricted permissions.
fn save_config(config: &Config, override_path: Option<&Path>) -> io::Result<()> {
    let path = resolve_config_path(override_path)
        .ok_or_else(|| io::Error::other("cannot determine config directory"))?;
    let dir = path
        .parent()
        .ok_or_else(|| io::Error::other("config path has no parent directory"))?;
    fs::create_dir_all(dir)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }

    let contents = serde_json::to_string_pretty(config).map_err(io::Error::other)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)?;
        file.write_all(contents.as_bytes())?;
    }

    #[cfg(not(unix))]
    {
        fs::write(&path, contents)?;
    }

    Ok(())
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
    let mut config = load_config(config_path).unwrap_or_default();
    config.api_key = Some(trimmed.to_string());
    save_config(&config, config_path)
}

/// Masks an API key for display.
fn mask_key(key: &str) -> String {
    if key.is_empty() {
        return "...".into();
    }
    if !key.is_ascii() {
        "****...".into()
    } else if key.len() > 8 {
        format!("{}...{}", &key[..4], &key[key.len() - 4..])
    } else if key.len() > 4 {
        format!("{}...", &key[..4])
    } else {
        format!("{}...", &key[..1])
    }
}

/// Loads the API key from the config file, falling back to the legacy file.
fn load_api_key_for_display(config_path: Option<&Path>) -> Option<String> {
    load_config(config_path)
        .ok()
        .and_then(|c| c.api_key)
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
    let key = read_key_line()?;

    save_api_key(&key, config_path).map_err(|e| format!("failed to save API key: {e}"))?;
    remove_legacy_key_file();
    let path = resolve_config_path(config_path)
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    eprintln!("API key saved to {path}");

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
            match save_api_key(&resolved, config_path) {
                Ok(()) => {
                    remove_legacy_key_file();
                    let path = resolve_config_path(config_path)
                        .map(|p| p.display().to_string())
                        .unwrap_or_default();
                    eprintln!("API key saved to {path}");
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
        assert_eq!(mask_key("abcdefghijkl"), "abcd...ijkl");
    }

    #[test]
    fn mask_exactly_8_chars() {
        assert_eq!(mask_key("abcdefgh"), "abcd...");
    }

    #[test]
    fn mask_exactly_4_chars() {
        assert_eq!(mask_key("abcd"), "a...");
    }

    #[test]
    fn mask_exactly_1_char() {
        assert_eq!(mask_key("a"), "a...");
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
}
