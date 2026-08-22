//! `bx config set-key` against a real file: what it warns about, and what it must not.
//!
//! Unix only. These run the binary for real, and `dirs` reads the Windows config location
//! from `SHGetKnownFolderPath`, which no environment variable can redirect — so on Windows
//! the legacy-key cleanup below would delete the developer's own key file.
#![cfg(unix)]

use std::process::{Command, Output};

/// `--config` redirects where the config is written, but not where the *legacy* key file
/// is looked for and deleted — that comes from the real config dir. Hence the throwaway
/// home as well.
fn set_key(path: &std::path::Path) -> Output {
    let home = tempfile::tempdir().unwrap();
    bx_with_home(home.path())
        .args(["config", "set-key", "BSAtestkey1234567"])
        .arg("--config")
        .arg(path)
        .output()
        .unwrap()
}

fn bx_with_home(home: &std::path::Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_bx"));
    c.env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env_remove("BRAVE_SEARCH_API_KEY")
        .env_remove("BRAVE_API_KEY");
    c
}

#[test]
fn a_first_set_key_does_not_warn_about_the_file_it_creates() {
    let dir = tempfile::tempdir().unwrap();
    let out = set_key(&dir.path().join("new.json"));

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    // True of a config we failed to parse, but false and alarming on the first run.
    assert!(!stderr.contains("may be reset"), "{stderr}");
}

#[test]
fn set_key_over_a_corrupt_config_warns_that_settings_are_lost() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt.json");
    std::fs::write(&path, "{not json").unwrap();

    let out = set_key(&path);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stderr.contains("may be reset"), "{stderr}");
    // The key is saved regardless; the warning says what it cost.
    assert!(std::fs::read_to_string(&path).unwrap().contains("BSAtest"));
}

#[test]
fn the_default_config_dir_is_tightened_even_if_it_already_existed() {
    // Only the directory we own is chmodded on every save, so this cannot be tested
    // through --config. A dotfiles restore or a `mkdir -p` leaves it 0755, which lets
    // other users list the directory and see the key file's name and mtime.
    use std::os::unix::fs::PermissionsExt;
    let home = tempfile::tempdir().unwrap();

    let path = bx_with_home(home.path())
        .arg("config")
        .arg("path")
        .output()
        .unwrap();
    let path = String::from_utf8(path.stdout).unwrap();
    let dir = std::path::Path::new(path.trim())
        .parent()
        .unwrap()
        .to_path_buf();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();

    let out = bx_with_home(home.path())
        .args(["config", "set-key", "BSAtestkey1234567"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o700, "config dir left group/world readable");
    // The property that actually protects the key.
    let file_mode = std::fs::metadata(path.trim()).unwrap().permissions().mode();
    assert_eq!(file_mode & 0o777, 0o600, "key file left readable by others");
}
