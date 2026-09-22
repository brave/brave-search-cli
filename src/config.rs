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

/// Loads the JSON config file. Returns `Config::default()` when the file is empty or the
/// default path does not exist; any other failure is an error.
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
    // Tighten the default dir on every save (a restore can leave it 0755) and any dir we
    // create, never an existing `--config` parent. Probed before `create_dir_all` creates it.
    #[cfg(unix)]
    let tighten = !dir.as_os_str().is_empty() && (override_path.is_none() || !dir.exists());
    fs::create_dir_all(dir)?;
    #[cfg(unix)]
    if tighten {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }

    let contents = serde_json::to_string_pretty(config).map_err(io::Error::other)?;

    // Temp + rename: atomic, 0600 applies because the file is new, and a symlink at `path` is
    // replaced, not written through. The clock only guards a recycled pid's leftover.
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
    // `--config` does not redirect the legacy path; default runs may still need that key.
    if config_path.is_none() {
        remove_legacy_key_file();
    }
    Ok(())
}

/// Validates an API key before saving. A non-ASCII byte (a pasted NBSP) would otherwise 401 in
/// a way `show-key` cannot reveal.
pub(crate) fn validate_api_key(key: &str) -> io::Result<()> {
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
        // An absent `--config` target is a first run; a failed stat is not. (A missing
        // default config never reaches here.)
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

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
