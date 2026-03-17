use std::fs;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

/// Returns the config directory: ~/.config/brave-search/ (or platform equivalent).
fn config_dir() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("brave-search"))
}

/// Returns the path to the API key file.
pub fn key_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("api_key"))
}

/// Loads the API key from the config file, if it exists.
pub fn load_api_key() -> Option<String> {
    let path = key_path()?;
    fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
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

/// Saves the API key to the config file with restricted permissions.
pub fn save_api_key(key: &str) -> io::Result<()> {
    let trimmed = key.trim();
    validate_api_key(trimmed)?;
    let dir = config_dir().ok_or_else(|| io::Error::other("cannot determine config directory"))?;
    fs::create_dir_all(&dir)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Restrict directory to owner-only (prevent listing by other users).
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
    }

    let path = dir.join("api_key");

    // On Unix, create the file with 0o600 atomically to avoid a TOCTOU window
    // where the file is briefly world-readable between creation and chmod.
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)?;
        io::Write::write_all(&mut file, trimmed.as_bytes())?;
    }

    #[cfg(not(unix))]
    {
        fs::write(&path, trimmed)?;
    }

    Ok(())
}

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
/// If stdin is a TTY, prompts for the key. Otherwise, prints instructions and exits.
pub fn onboard() -> Result<String, String> {
    eprintln!("{SETUP_MSG}");

    if !io::stdin().is_terminal() {
        return Err("no API key configured".into());
    }

    eprintln!();
    let key = read_key_line()?;

    save_api_key(&key).map_err(|e| format!("failed to save API key: {e}"))?;
    let path = key_path()
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

/// Handles the `config` subcommand.
pub fn handle_config(cmd: &super::ConfigCmd) {
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
            match save_api_key(&resolved) {
                Ok(()) => {
                    let path = key_path()
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
        super::ConfigCmd::ShowKey => match load_api_key() {
            Some(key) => {
                // Guard against panicking on multibyte UTF-8 boundaries.
                if !key.is_ascii() {
                    println!("****...");
                } else if key.len() > 8 {
                    println!("{}...{}", &key[..4], &key[key.len() - 4..]);
                } else if key.len() > 4 {
                    println!("{}...", &key[..4]);
                } else {
                    println!("{}...", &key[..1.min(key.len())]);
                }
            }
            None => {
                eprintln!("no API key configured");
                std::process::exit(1);
            }
        },
        super::ConfigCmd::Path => match key_path() {
            Some(p) => println!("{}", p.display()),
            None => {
                eprintln!("error: cannot determine config directory");
                std::process::exit(1);
            }
        },
    }
}
