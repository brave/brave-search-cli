use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

// ── Config struct ────────────────────────────────────────────────────

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub timeout: Option<u64>,
}

// ── Paths ────────────────────────────────────────────────────────────

/// Returns the config directory: ~/.config/brave-search/ (or platform equivalent).
fn config_dir() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("brave-search"))
}

/// Returns the path to the TOML config file.
fn config_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("config.toml"))
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

/// Loads the TOML config file. Returns `Config::default()` if the default path
/// is missing. Hard-exits if an explicit `--config` override cannot be read.
pub fn load_config(override_path: Option<&Path>) -> Config {
    let is_explicit = override_path.is_some();
    let path = match resolve_config_path(override_path) {
        Some(p) => p,
        None => return Config::default(),
    };

    match fs::read_to_string(&path) {
        Ok(contents) => match toml::from_str::<Config>(&contents) {
            Ok(cfg) => cfg,
            Err(e) => {
                if is_explicit {
                    eprintln!("error: failed to parse {}: {e}", path.display());
                    std::process::exit(1);
                }
                eprintln!("warning: failed to parse {}: {e}", path.display());
                Config::default()
            }
        },
        Err(e) if e.kind() == io::ErrorKind::NotFound => Config::default(),
        Err(e) => {
            if is_explicit {
                eprintln!("error: cannot read {}: {e}", path.display());
                std::process::exit(1);
            }
            Config::default()
        }
    }
}

/// Saves the config to a TOML file with restricted permissions.
pub fn save_config(config: &Config, override_path: Option<&Path>) -> io::Result<()> {
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

    let contents = toml::to_string_pretty(config).map_err(io::Error::other)?;

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

// ── String helpers ───────────────────────────────────────────────────

/// Returns the trimmed string if non-empty, reusing the allocation when possible.
pub(crate) fn trim_non_empty(s: String) -> Option<String> {
    let tl = s.trim().len();
    if tl == 0 {
        None
    } else if tl == s.len() {
        Some(s)
    } else {
        Some(s.trim().to_string())
    }
}

// ── API key helpers ──────────────────────────────────────────────────

/// Loads the API key from the legacy bare `api_key` file, if it exists.
pub fn load_legacy_api_key() -> Option<String> {
    let path = legacy_key_path()?;
    fs::read_to_string(path).ok().and_then(trim_non_empty)
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

/// Saves the API key into the TOML config file (read-modify-write).
pub fn save_api_key(key: &str, config_path: Option<&Path>) -> io::Result<()> {
    let trimmed = key.trim();
    validate_api_key(trimmed)?;
    let mut config = load_config(config_path);
    config.api_key = Some(trimmed.to_string());
    save_config(&config, config_path)
}

/// Masks an API key for display.
fn mask_key(key: &str) -> String {
    if !key.is_ascii() {
        "****...".into()
    } else if key.len() > 8 {
        format!("{}...{}", &key[..4], &key[key.len() - 4..])
    } else if key.len() > 4 {
        format!("{}...", &key[..4])
    } else {
        format!("{}...", &key[..1.min(key.len())])
    }
}

/// Loads the API key from the config file, falling back to the legacy file.
fn load_api_key_for_display(config_path: Option<&Path>) -> Option<String> {
    load_config(config_path)
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
    let key = read_key_line()?;

    save_api_key(&key, config_path).map_err(|e| format!("failed to save API key: {e}"))?;
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
            let config = load_config(config_path);
            let has_any =
                config.api_key.is_some() || config.base_url.is_some() || config.timeout.is_some();
            if !has_any {
                // Fall back to legacy key for display
                if let Some(key) = load_legacy_api_key() {
                    println!("api_key = {}", mask_key(&key));
                    return;
                }
                eprintln!("(no configuration found)");
                return;
            }
            if let Some(ref key) = config.api_key {
                let display = if key.trim().is_empty() {
                    "(empty)".into()
                } else {
                    mask_key(key)
                };
                println!("api_key = {display}");
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

    // ── Config deserialization ──

    #[test]
    fn parse_empty_string() {
        let c: Config = toml::from_str("").unwrap();
        assert!(c.api_key.is_none());
        assert!(c.base_url.is_none());
        assert!(c.timeout.is_none());
    }

    #[test]
    fn parse_full_config() {
        let c: Config =
            toml::from_str("api_key = \"BSAtest\"\nbase_url = \"https://x.com\"\ntimeout = 60\n")
                .unwrap();
        assert_eq!(c.api_key.as_deref(), Some("BSAtest"));
        assert_eq!(c.base_url.as_deref(), Some("https://x.com"));
        assert_eq!(c.timeout, Some(60));
    }

    #[test]
    fn parse_partial_config() {
        let c: Config = toml::from_str("timeout = 10\n").unwrap();
        assert!(c.api_key.is_none());
        assert!(c.base_url.is_none());
        assert_eq!(c.timeout, Some(10));
    }

    #[test]
    fn parse_unknown_keys_ignored() {
        let c: Config = toml::from_str("api_key = \"k\"\nfuture_field = true\n").unwrap();
        assert_eq!(c.api_key.as_deref(), Some("k"));
    }

    #[test]
    fn parse_wrong_type_errors() {
        assert!(toml::from_str::<Config>("timeout = \"abc\"").is_err());
        assert!(toml::from_str::<Config>("timeout = 3.14").is_err());
        assert!(toml::from_str::<Config>("api_key = true").is_err());
    }

    #[test]
    fn parse_invalid_toml_syntax() {
        assert!(toml::from_str::<Config>("timeout = ").is_err());
        assert!(toml::from_str::<Config>("[[[bad").is_err());
    }

    #[test]
    fn parse_comments_ignored() {
        let c: Config = toml::from_str("# comment\ntimeout = 5\n").unwrap();
        assert_eq!(c.timeout, Some(5));
    }

    #[test]
    fn parse_empty_string_value() {
        let c: Config = toml::from_str("api_key = \"\"\n").unwrap();
        assert_eq!(c.api_key.as_deref(), Some(""));
    }

    #[test]
    fn parse_whitespace_only_file() {
        let c: Config = toml::from_str("  \n  \n").unwrap();
        assert!(c.timeout.is_none());
    }

    #[test]
    fn parse_timeout_zero() {
        let c: Config = toml::from_str("timeout = 0\n").unwrap();
        assert_eq!(c.timeout, Some(0));
    }

    #[test]
    fn parse_timeout_i64_max() {
        let c: Config = toml::from_str("timeout = 9223372036854775807\n").unwrap();
        assert_eq!(c.timeout, Some(i64::MAX as u64));
    }

    #[test]
    fn parse_timeout_negative_errors() {
        assert!(toml::from_str::<Config>("timeout = -1").is_err());
    }

    #[test]
    fn parse_timeout_above_i64_max() {
        // toml 1.1 accepts u64 values above i64::MAX for unsigned fields
        let c: Config = toml::from_str("timeout = 9223372036854775808\n").unwrap();
        assert_eq!(c.timeout, Some(9223372036854775808));
    }

    #[test]
    fn parse_timeout_above_u64_max_errors() {
        assert!(toml::from_str::<Config>("timeout = 18446744073709551616").is_err());
    }

    #[test]
    fn parse_windows_line_endings() {
        let c: Config = toml::from_str("timeout = 5\r\napi_key = \"k\"\r\n").unwrap();
        assert_eq!(c.timeout, Some(5));
        assert_eq!(c.api_key.as_deref(), Some("k"));
    }

    #[test]
    fn parse_unicode_values() {
        let c: Config = toml::from_str("base_url = \"https://例え.jp/api\"\n").unwrap();
        assert_eq!(c.base_url.as_deref(), Some("https://例え.jp/api"));
    }

    #[test]
    fn parse_unknown_section_ignored() {
        let c: Config = toml::from_str("timeout = 5\n[unknown]\nfoo = 1\n").unwrap();
        assert_eq!(c.timeout, Some(5));
    }

    // ── Config serialization ──

    #[test]
    fn serialize_all_none_is_empty() {
        let s = toml::to_string_pretty(&Config::default()).unwrap();
        assert_eq!(s, "");
    }

    #[test]
    fn serialize_round_trip() {
        let c = Config {
            api_key: Some("testkey123".into()),
            base_url: Some("https://x.com".into()),
            timeout: Some(45),
        };
        let s = toml::to_string_pretty(&c).unwrap();
        let c2: Config = toml::from_str(&s).unwrap();
        assert_eq!(c2.api_key.as_deref(), Some("testkey123"));
        assert_eq!(c2.base_url.as_deref(), Some("https://x.com"));
        assert_eq!(c2.timeout, Some(45));
    }

    #[test]
    fn serialize_partial_omits_none() {
        let c = Config {
            timeout: Some(10),
            ..Default::default()
        };
        let s = toml::to_string_pretty(&c).unwrap();
        assert!(!s.contains("api_key"));
        assert!(!s.contains("base_url"));
        assert!(s.contains("timeout = 10"));
    }

    #[test]
    fn serialize_empty_string_value() {
        let c = Config {
            api_key: Some(String::new()),
            ..Default::default()
        };
        let s = toml::to_string_pretty(&c).unwrap();
        assert!(s.contains("api_key = \"\""));
    }

    // ── load_config ──

    #[test]
    fn load_config_override_valid() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("test.toml");
        fs::write(&p, "timeout = 99\n").unwrap();
        let c = load_config(Some(p.as_path()));
        assert_eq!(c.timeout, Some(99));
    }

    #[test]
    fn load_config_override_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("test.toml");
        fs::write(&p, "").unwrap();
        let c = load_config(Some(p.as_path()));
        assert!(c.api_key.is_none());
    }

    #[test]
    fn load_config_default_missing_returns_default() {
        let c = load_config(None);
        let _ = c;
    }

    // ── save_config + round trip ──

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        let c = Config {
            api_key: Some("mykey12345".into()),
            base_url: None,
            timeout: Some(15),
        };
        save_config(&c, Some(p.as_path())).unwrap();
        let loaded = load_config(Some(p.as_path()));
        assert_eq!(loaded.api_key.as_deref(), Some("mykey12345"));
        assert!(loaded.base_url.is_none());
        assert_eq!(loaded.timeout, Some(15));
    }

    #[test]
    fn save_config_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sub").join("dir").join("config.toml");
        let c = Config {
            timeout: Some(1),
            ..Default::default()
        };
        save_config(&c, Some(p.as_path())).unwrap();
        assert!(p.exists());
    }

    // ── save_api_key ──

    #[test]
    fn save_api_key_preserves_other_fields() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        fs::write(&p, "base_url = \"https://x.com\"\ntimeout = 20\n").unwrap();
        save_api_key("newkey12345", Some(p.as_path())).unwrap();
        let c = load_config(Some(p.as_path()));
        assert_eq!(c.api_key.as_deref(), Some("newkey12345"));
        assert_eq!(c.base_url.as_deref(), Some("https://x.com"));
        assert_eq!(c.timeout, Some(20));
    }

    #[test]
    fn save_api_key_validates_too_short() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        assert!(save_api_key("abc", Some(p.as_path())).is_err());
    }

    #[test]
    fn save_api_key_validates_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        assert!(save_api_key("abc def ghi", Some(p.as_path())).is_err());
    }

    #[test]
    fn save_api_key_validates_control_chars() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        assert!(save_api_key("abcdef\tgh", Some(p.as_path())).is_err());
    }

    #[test]
    fn save_api_key_trims_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        save_api_key("  testkey12345  ", Some(p.as_path())).unwrap();
        let c = load_config(Some(p.as_path()));
        assert_eq!(c.api_key.as_deref(), Some("testkey12345"));
    }

    #[test]
    fn save_api_key_exactly_8_chars() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        save_api_key("12345678", Some(p.as_path())).unwrap();
        let c = load_config(Some(p.as_path()));
        assert_eq!(c.api_key.as_deref(), Some("12345678"));
    }

    // ── mask_key ──

    #[test]
    fn mask_long_key() {
        assert_eq!(mask_key("abcdefghijkl"), "abcd...ijkl");
    }

    #[test]
    fn mask_exactly_9_chars() {
        assert_eq!(mask_key("abcdefghi"), "abcd...fghi");
    }

    #[test]
    fn mask_exactly_8_chars() {
        assert_eq!(mask_key("abcdefgh"), "abcd...");
    }

    #[test]
    fn mask_exactly_5_chars() {
        assert_eq!(mask_key("abcde"), "abcd...");
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

    // ── mask_key edge cases ──

    #[test]
    fn mask_empty_string() {
        assert_eq!(mask_key(""), "...");
    }

    #[test]
    fn mask_exactly_2_chars() {
        assert_eq!(mask_key("ab"), "a...");
    }

    #[test]
    fn mask_exactly_3_chars() {
        assert_eq!(mask_key("abc"), "a...");
    }

    // ── validate_api_key boundaries (via save_api_key) ──

    #[test]
    fn save_api_key_exactly_7_chars_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        assert!(save_api_key("1234567", Some(p.as_path())).is_err());
    }

    #[test]
    fn save_api_key_validates_newline() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        assert!(save_api_key("abcdef\ngh", Some(p.as_path())).is_err());
    }

    #[test]
    fn save_api_key_validates_null() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        assert!(save_api_key("abcdef\0gh", Some(p.as_path())).is_err());
    }

    // ── save_config permissions ──

    #[cfg(unix)]
    #[test]
    fn save_config_file_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
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
        let p = sub.join("config.toml");
        let c = Config::default();
        save_config(&c, Some(p.as_path())).unwrap();
        let mode = fs::metadata(&sub).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }

    // ── trim_non_empty ──

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
    fn trim_non_empty_leading_only() {
        assert_eq!(trim_non_empty("  key".into()), Some("key".into()));
    }

    #[test]
    fn trim_non_empty_trailing_only() {
        assert_eq!(trim_non_empty("key  ".into()), Some("key".into()));
    }

    // ── load_api_key_for_display ──

    #[test]
    fn load_api_key_for_display_from_config() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        fs::write(&p, "api_key = \"testkey12345\"\n").unwrap();
        let key = load_api_key_for_display(Some(p.as_path()));
        assert_eq!(key.as_deref(), Some("testkey12345"));
    }

    #[test]
    fn load_api_key_for_display_empty_config() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        fs::write(&p, "").unwrap();
        // Verify the config itself has no key
        assert!(load_config(Some(p.as_path())).api_key.is_none());
    }

    #[test]
    fn load_api_key_for_display_whitespace_key() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        fs::write(&p, "api_key = \"   \"\n").unwrap();
        // Whitespace-only key should be filtered out by trim_non_empty
        let config = load_config(Some(p.as_path()));
        assert_eq!(config.api_key.as_deref(), Some("   "));
        // But display should filter it
        let key = load_api_key_for_display(Some(p.as_path()));
        // Either None (no legacy) or a legacy key — not the whitespace one
        if let Some(ref k) = key {
            assert!(
                !k.trim().is_empty(),
                "should not return whitespace-only key"
            );
        }
    }

    // ── config_path ──

    #[test]
    fn config_path_ends_with_config_toml() {
        if let Some(p) = config_path() {
            assert!(p.ends_with("config.toml"));
            assert!(p.parent().unwrap().ends_with("brave-search"));
        }
    }
}
