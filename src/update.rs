use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

const REPO: &str = "brave/brave-search-cli";
const RELEASES_URL: &str = "https://github.com/brave/brave-search-cli/releases";
const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
// ── Platform detection (compile-time) ────────────────────────────────

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub const PLATFORM: &str = "linux-amd64";
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub const PLATFORM: &str = "linux-arm64";
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub const PLATFORM: &str = "darwin-arm64";
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
pub const PLATFORM: &str = "windows-amd64";
#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
pub const PLATFORM: &str = "windows-arm64";

#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "aarch64"),
    all(target_os = "macos", target_arch = "aarch64"),
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "windows", target_arch = "aarch64"),
)))]
pub const PLATFORM: &str = "unsupported";

const BINARY_EXT: &str = if cfg!(windows) { ".exe" } else { "" };

// ── Update-check state file ──────────────────────────────────────────

#[derive(Debug, Default, Serialize, Deserialize)]
struct UpdateCheckState {
    last_check_epoch: Option<u64>,
    latest_version: Option<String>,
}

fn state_path() -> Option<PathBuf> {
    crate::config::config_dir().map(|d| d.join("update-check.json"))
}

fn save_state(state: &UpdateCheckState) {
    let Some(path) = state_path() else { return };
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).ok();
    }
    if let Ok(json) = serde_json::to_string(state) {
        fs::write(path, json).ok();
    }
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ── Version comparison ───────────────────────────────────────────────

/// Parses a semver string (with optional leading 'v') into (major, minor, patch).
fn parse_semver(s: &str) -> Option<(u64, u64, u64)> {
    let s = s.strip_prefix('v').unwrap_or(s);
    let mut parts = s.splitn(3, '.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    Some((major, minor, patch))
}

/// Returns true if `latest` is strictly newer than `current`.
fn is_newer(current: &str, latest: &str) -> bool {
    match (parse_semver(current), parse_semver(latest)) {
        (Some(c), Some(l)) => l > c,
        _ => false,
    }
}

// ── Version resolution ───────────────────────────────────────────────

fn update_agent(timeout_secs: u64) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .timeout_global(Some(Duration::from_secs(timeout_secs)))
            .user_agent(concat!("bx/", env!("CARGO_PKG_VERSION")))
            .build(),
    )
}

fn redirecting_agent(timeout_secs: u64) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .http_status_as_error(false)
            .max_redirects(10)
            .timeout_global(Some(Duration::from_secs(timeout_secs)))
            .user_agent(concat!("bx/", env!("CARGO_PKG_VERSION")))
            .build(),
    )
}

/// Resolves the latest release tag by following the /releases/latest redirect.
fn resolve_latest_version(timeout: u64) -> Result<String, String> {
    let url = format!("{RELEASES_URL}/latest");
    let resp = update_agent(timeout)
        .head(&url)
        .call()
        .map_err(|e| format!("network error: {e}"))?;

    let status = resp.status().as_u16();
    if status == 302 || status == 301 {
        if let Some(loc) = resp.headers().get("location") {
            let loc = loc.to_str().map_err(|_| "invalid location header")?;
            if let Some(tag) = loc.rsplit('/').next() {
                let tag = tag.trim();
                if !tag.is_empty() {
                    return Ok(tag.to_string());
                }
            }
        }
        return Err("redirect did not contain a version tag".into());
    }

    // GitHub sometimes returns 200 with the HTML page instead of redirecting
    // for HEAD requests. Fall back to the API endpoint.
    let api_url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let resp = update_agent(timeout)
        .get(&api_url)
        .header("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| format!("network error: {e}"))?;

    let status = resp.status().as_u16();
    if status != 200 {
        return Err(format!("GitHub API returned HTTP {status}"));
    }

    let body: String = resp
        .into_body()
        .read_to_string()
        .map_err(|e| format!("failed to read response: {e}"))?;

    let json: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("invalid JSON: {e}"))?;

    json["tag_name"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| "tag_name not found in response".into())
}

// ── Explicit check ───────────────────────────────────────────────────

/// Checks for updates and prints the result. Returns the exit code.
pub fn check_for_update() -> i32 {
    eprint!("Checking for updates... ");
    match resolve_latest_version(30) {
        Ok(tag) => {
            let version = tag.strip_prefix('v').unwrap_or(&tag);
            save_state(&UpdateCheckState {
                last_check_epoch: Some(now_epoch()),
                latest_version: Some(version.to_string()),
            });

            if is_newer(CURRENT_VERSION, version) {
                eprintln!("v{version} is available (current: v{CURRENT_VERSION})");
                eprintln!("Run `bx update` to upgrade.");
                0
            } else {
                eprintln!("bx v{CURRENT_VERSION} is already up to date.");
                0
            }
        }
        Err(e) => {
            eprintln!("failed");
            eprintln!("error: {e}");
            1
        }
    }
}

// ── Self-update (download + verify + replace) ────────────────────────

/// Downloads and installs the latest version. Returns the exit code.
pub fn perform_update() -> i32 {
    if PLATFORM == "unsupported" {
        eprintln!("error: no pre-built binary available for this platform");
        eprintln!("hint: build from source instead: cargo install --git https://github.com/{REPO}");
        return 1;
    }

    eprint!("Checking for updates... ");
    let tag = match resolve_latest_version(30) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed");
            eprintln!("error: {e}");
            return 1;
        }
    };

    let version = tag.strip_prefix('v').unwrap_or(&tag);
    if !is_newer(CURRENT_VERSION, version) {
        eprintln!("bx v{CURRENT_VERSION} is already up to date.");
        return 0;
    }
    eprintln!("v{version} is available (current: v{CURRENT_VERSION})");

    let binary_name = format!("bx-{version}-{PLATFORM}{BINARY_EXT}");
    let checksum_name = format!("{binary_name}.sha256");
    let release_url = format!("{RELEASES_URL}/download/{tag}");

    // Download binary
    eprintln!("Downloading {binary_name}...");
    let binary_data = match download_file(&format!("{release_url}/{binary_name}")) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("error: failed to download binary: {e}");
            return 1;
        }
    };

    // Download checksum
    eprintln!("Verifying checksum...");
    let checksum_data = match download_file(&format!("{release_url}/{checksum_name}")) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("error: failed to download checksum: {e}");
            return 1;
        }
    };

    // Verify checksum
    let expected = String::from_utf8_lossy(&checksum_data);
    let expected = expected.split_whitespace().next().unwrap_or("").trim();
    if expected.len() != 64 || !expected.chars().all(|c| c.is_ascii_hexdigit()) {
        eprintln!("error: invalid checksum format in {checksum_name}");
        return 1;
    }

    let mut hasher = Sha256::new();
    hasher.update(&binary_data);
    let actual = format!("{:x}", hasher.finalize());

    if actual != expected.to_lowercase() {
        eprintln!("error: checksum verification failed!");
        eprintln!("  expected: {expected}");
        eprintln!("  got:      {actual}");
        eprintln!("  the downloaded binary may be corrupted or tampered with");
        return 1;
    }

    // Safety: current_exe() is only used to locate the install path for self-replacement,
    // not for trust/authorization decisions. The downloaded binary is integrity-checked via
    // SHA256 above, so a manipulated path cannot cause execution of unverified content.
    // nosemgrep: rust.lang.security.current-exe.current-exe
    let current_exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: cannot determine current executable path: {e}");
            return 1;
        }
    };

    // Resolve symlinks to get the actual binary path
    let current_exe = match current_exe.canonicalize() {
        Ok(p) => p,
        Err(_) => current_exe,
    };

    eprintln!("Installing to {}...", current_exe.display());

    if let Err(e) = self_replace(&current_exe, &binary_data) {
        eprintln!("error: failed to replace binary: {e}");
        if e.kind() == io::ErrorKind::PermissionDenied {
            eprintln!(
                "hint: you may need elevated privileges, or re-install to a user-writable directory"
            );
        }
        return 1;
    }

    save_state(&UpdateCheckState {
        last_check_epoch: Some(now_epoch()),
        latest_version: Some(version.to_string()),
    });

    eprintln!("bx updated to v{version} successfully.");
    0
}

const MAX_DOWNLOAD_SIZE: u64 = 10 * 1024 * 1024; // 10 MB

fn download_file(url: &str) -> Result<Vec<u8>, String> {
    let resp = redirecting_agent(60)
        .get(url)
        .call()
        .map_err(|e| format!("network error: {e}"))?;

    let status = resp.status().as_u16();
    if status != 200 {
        return Err(format!("HTTP {status}"));
    }

    let body = resp.into_body().into_reader();
    let mut buf = Vec::new();
    body.take(MAX_DOWNLOAD_SIZE + 1)
        .read_to_end(&mut buf)
        .map_err(|e| format!("read error: {e}"))?;

    if buf.len() as u64 > MAX_DOWNLOAD_SIZE {
        return Err(format!(
            "download exceeds maximum size ({MAX_DOWNLOAD_SIZE} bytes)"
        ));
    }

    Ok(buf)
}

// ── Self-replacement ─────────────────────────────────────────────────

#[cfg(unix)]
fn self_replace(current_exe: &Path, new_binary: &[u8]) -> io::Result<()> {
    let dir = current_exe
        .parent()
        .ok_or_else(|| io::Error::other("exe has no parent directory"))?;

    // Same directory as the target so rename is same-filesystem (atomic).
    let tmp_path = dir.join(".bx.update.tmp");

    fs::write(&tmp_path, new_binary)?;
    if let Err(e) = fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o755)) {
        fs::remove_file(&tmp_path).ok();
        return Err(e);
    }

    // Atomic rename — old inode stays alive until the running process exits.
    if let Err(e) = fs::rename(&tmp_path, current_exe) {
        fs::remove_file(&tmp_path).ok();
        return Err(e);
    }
    Ok(())
}

#[cfg(windows)]
fn self_replace(current_exe: &Path, new_binary: &[u8]) -> io::Result<()> {
    let dir = current_exe
        .parent()
        .ok_or_else(|| io::Error::other("exe has no parent directory"))?;

    let old_path = dir.join(format!(
        "{}.old",
        current_exe
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    ));
    let tmp_path = dir.join(".bx.update.tmp.exe");

    // Write the new binary to a temp file
    fs::write(&tmp_path, new_binary)?;

    // Rename the currently running exe out of the way (Windows allows renaming a locked file)
    if let Err(e) = fs::rename(current_exe, &old_path) {
        // Clean up temp file on failure
        fs::remove_file(&tmp_path).ok();
        return Err(e);
    }

    // Move the new binary into place
    if let Err(e) = fs::rename(&tmp_path, current_exe) {
        // Try to restore the old binary
        fs::rename(&old_path, current_exe).ok();
        fs::remove_file(&tmp_path).ok();
        return Err(e);
    }

    // Best-effort cleanup of old binary (may fail if still locked)
    fs::remove_file(&old_path).ok();

    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn self_replace(_current_exe: &Path, _new_binary: &[u8]) -> io::Result<()> {
    Err(io::Error::other(
        "self-update not supported on this platform",
    ))
}

// ── Windows .old cleanup ─────────────────────────────────────────────

/// Cleans up any stale .old binary left behind from a previous Windows update.
#[cfg(windows)]
pub fn cleanup_old_binary() {
    // Safety: same as perform_update — path used only for cleanup, not trust decisions.
    // nosemgrep: rust.lang.security.current-exe.current-exe
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let exe = exe.canonicalize().unwrap_or(exe);
    let Some(dir) = exe.parent() else { return };
    let old_name = format!(
        "{}.old",
        exe.file_name().unwrap_or_default().to_string_lossy()
    );
    let old_path = dir.join(old_name);
    if old_path.exists() {
        fs::remove_file(&old_path).ok();
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_semver_basic() {
        assert_eq!(parse_semver("1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_semver("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(parse_semver("0.0.1"), Some((0, 0, 1)));
        assert_eq!(parse_semver("v10.20.30"), Some((10, 20, 30)));
    }

    #[test]
    fn parse_semver_invalid() {
        assert_eq!(parse_semver("1.2"), None);
        assert_eq!(parse_semver("abc"), None);
        assert_eq!(parse_semver(""), None);
        assert_eq!(parse_semver("v"), None);
        assert_eq!(parse_semver("1.2.x"), None);
    }

    #[test]
    fn is_newer_basic() {
        assert!(is_newer("1.0.0", "1.0.1"));
        assert!(is_newer("1.0.0", "1.1.0"));
        assert!(is_newer("1.0.0", "2.0.0"));
        assert!(is_newer("1.2.0", "v1.3.0"));
    }

    #[test]
    fn is_newer_not_newer() {
        assert!(!is_newer("1.0.0", "1.0.0"));
        assert!(!is_newer("1.1.0", "1.0.0"));
        assert!(!is_newer("2.0.0", "1.9.9"));
    }

    #[test]
    fn is_newer_with_v_prefix() {
        assert!(is_newer("v1.0.0", "v1.0.1"));
        assert!(is_newer("1.0.0", "v1.0.1"));
        assert!(is_newer("v1.0.0", "1.0.1"));
    }

    #[test]
    fn is_newer_invalid_versions() {
        assert!(!is_newer("abc", "1.0.0"));
        assert!(!is_newer("1.0.0", "abc"));
        assert!(!is_newer("", "1.0.0"));
    }

    #[test]
    fn platform_is_set() {
        // PLATFORM is "unsupported" on targets without pre-built binaries
        // (e.g. darwin-amd64). The value is always a static string.
        assert!(!PLATFORM.is_empty());
    }

    #[test]
    fn state_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.json");
        let state = UpdateCheckState {
            last_check_epoch: Some(1234567890),
            latest_version: Some("1.3.0".into()),
        };
        let json = serde_json::to_string(&state).unwrap();
        fs::write(&path, &json).unwrap();
        let loaded: UpdateCheckState =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(loaded.last_check_epoch, Some(1234567890));
        assert_eq!(loaded.latest_version.as_deref(), Some("1.3.0"));
    }

    #[test]
    fn state_default_is_empty() {
        let state = UpdateCheckState::default();
        assert!(state.last_check_epoch.is_none());
        assert!(state.latest_version.is_none());
    }

    #[test]
    fn checksum_verification_logic() {
        let data = b"hello world";
        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = format!("{:x}", hasher.finalize());
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn binary_ext_matches_platform() {
        if cfg!(windows) {
            assert_eq!(BINARY_EXT, ".exe");
        } else {
            assert_eq!(BINARY_EXT, "");
        }
    }
}
