//! `bx config set-key` against a real file: what it warns about, what it must not, and
//! what it is allowed to delete. Unix only — `dirs` reads the Windows config dir from
//! `SHGetKnownFolderPath`, which no environment variable can redirect.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Output;

mod common;
use common::{Bounded, Pty, bx_in, finish};

const KEY: &str = "BSAtestkey1234567";

/// The config path `bx` itself resolves under `home`: `dirs` reads `$XDG_CONFIG_HOME` on
/// Linux but `$HOME/Library` on macOS, so ask rather than guess.
fn config_path(home: &Path) -> PathBuf {
    let out = bx_in(home).args(["config", "path"]).bounded_output();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    PathBuf::from(String::from_utf8(out.stdout).unwrap().trim())
}

/// Plants the legacy bare `api_key` file `bx` migrates from, and returns its path.
fn plant_legacy_key(home: &Path, key: &str) -> PathBuf {
    let path = config_path(home).with_file_name("api_key");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, key).unwrap();
    path
}

fn set_key(path: &Path) -> Output {
    let home = tempfile::tempdir().unwrap();
    bx_in(home.path())
        .args(["config", "set-key", KEY])
        .arg("--config")
        .arg(path)
        .bounded_output()
}

fn mode_of(path: &Path) -> u32 {
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
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
fn an_empty_config_file_does_not_warn() {
    // An empty file parses as the default config, so there is nothing to lose.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.json");
    std::fs::write(&path, "").unwrap();

    let out = set_key(&path);

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(!stderr.contains("may be reset"), "{stderr}");
}

#[test]
fn an_unstattable_config_still_warns_before_failing() {
    // A config we cannot stat is not a config that is absent. `Path::exists` reports both
    // as false, which silently dropped the warning for the one case it exists for.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    std::fs::write(&path, "{not json").unwrap();
    let chmod = |m| std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(m));
    chmod(0o600).unwrap();

    // Root and CAP_DAC_OVERRIDE ignore the mode bits, as do some container mounts.
    let readable = std::fs::read_to_string(&path).is_ok();
    let out = set_key(&path);
    chmod(0o700).unwrap(); // before the asserts: a panic must not leave the dir unremovable
    if readable {
        return;
    }

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("may be reset"), "{stderr}");
    assert_eq!(out.status.code(), Some(2), "{stderr}");
}

#[test]
fn the_default_config_dir_is_tightened_even_if_it_already_existed() {
    // Only the directory we own is chmodded on every save, so this cannot be tested
    // through --config. A dotfiles restore or a `mkdir -p` leaves it 0755, which lets
    // other users list the directory and see the key file's name and mtime.
    let home = tempfile::tempdir().unwrap();
    let path = config_path(home.path());
    let dir = path.parent().unwrap();
    std::fs::create_dir_all(dir).unwrap();
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755)).unwrap();

    let out = bx_in(home.path())
        .args(["config", "set-key", KEY])
        .bounded_output();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert_eq!(mode_of(dir), 0o700, "config dir left group/world readable");
    // The property that actually protects the key.
    assert_eq!(mode_of(&path), 0o600, "key file left readable by others");
}

#[test]
fn a_bare_relative_config_path_writes_to_the_working_directory() {
    // `--config config.json` gives an empty parent path — the only caller that can, and
    // the one that would chmod the working directory if the guard were dropped.
    let home = tempfile::tempdir().unwrap();
    let cwd = tempfile::tempdir().unwrap();
    let before = mode_of(cwd.path());

    let out = bx_in(home.path())
        .args(["config", "set-key", KEY, "--config", "config.json"])
        .current_dir(cwd.path())
        .bounded_output();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert_eq!(mode_of(&cwd.path().join("config.json")), 0o600);
    assert_eq!(
        mode_of(cwd.path()),
        before,
        "chmodded the working directory"
    );
}

#[test]
fn concurrent_saves_all_succeed() {
    // A save used to sweep the config directory first, which unlinked the temp of any
    // other save already in flight and failed it with ENOENT. Distinct pids make each
    // temp name unique, so with the sweep gone every one of these must land.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.json");
    let homes: Vec<_> = (0..4).map(|_| tempfile::tempdir().unwrap()).collect();

    let running: Vec<_> = homes
        .iter()
        .enumerate()
        .map(|(i, home)| {
            bx_in(home.path())
                .args(["config", "set-key", &format!("BSAtestkey123456{i}")])
                .arg("--config")
                .arg(&path)
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    for child in running {
        let out = finish(child);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{stderr}");
    }

    // Last writer wins; which one is nobody's business, but the file must be whole.
    let saved: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert!(saved["api_key"].as_str().unwrap().starts_with("BSAtestkey"));
    assert!(
        std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .all(|e| !e.file_name().to_string_lossy().starts_with(".bx-")),
        "temp file left behind"
    );
}

#[test]
fn the_default_config_removes_the_legacy_key_file() {
    let home = tempfile::tempdir().unwrap();
    let legacy = plant_legacy_key(home.path(), "legacykey12345");

    let out = bx_in(home.path())
        .args(["config", "set-key", KEY])
        .bounded_output();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(!legacy.exists(), "legacy key file not cleaned up");
}

#[test]
fn an_explicit_config_keeps_the_legacy_key_file() {
    // `--config` cannot redirect the legacy path, so saving elsewhere must not delete it:
    // it is still the only key the user's default runs have.
    let home = tempfile::tempdir().unwrap();
    let legacy = plant_legacy_key(home.path(), "legacykey12345");
    let elsewhere = tempfile::tempdir().unwrap();
    let path = elsewhere.path().join("scratch.json");

    let out = bx_in(home.path())
        .args(["config", "set-key", KEY])
        .arg("--config")
        .arg(&path)
        .bounded_output();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(std::fs::read_to_string(&path).unwrap().contains(KEY));
    assert!(legacy.exists(), "deleted a key --config never replaced");
}

#[test]
fn a_failed_save_keeps_the_legacy_key_file() {
    // The legacy file is the fallback until the config has actually been written.
    let home = tempfile::tempdir().unwrap();
    let legacy = plant_legacy_key(home.path(), "legacykey12345");
    std::fs::create_dir(config_path(home.path())).unwrap(); // a dir where the file goes

    let out = bx_in(home.path())
        .args(["config", "set-key", KEY])
        .bounded_output();

    assert_eq!(out.status.code(), Some(2));
    assert!(legacy.exists(), "dropped the only key on a failed save");
}

#[test]
fn a_non_ascii_legacy_key_warns_but_still_works() {
    // Such a key cannot be saved any more — it would 401 unexplainably — but it is the
    // user's only one, so migration warns and the request still goes out with it.
    let home = tempfile::tempdir().unwrap();
    let legacy = plant_legacy_key(home.path(), "legacy\u{00a0}key12345");

    let out = bx_in(home.path())
        .args(["web", "q", "--base-url", "http://127.0.0.1:1"])
        .bounded_output();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("failed to migrate legacy API key"),
        "{stderr}"
    );
    // Exit 5 is "the network refused us", i.e. the key was used rather than rejected.
    assert_eq!(out.status.code(), Some(5), "{stderr}");
    assert!(legacy.exists());
}

// ── a pasted key on a terminal ───────────────────────────────────────

/// `bx config set-key` prompting on `pty`, writing to `config`.
fn prompt(pty: &Pty, config: &Path) -> std::process::Command {
    let home = config.parent().unwrap();
    let mut c = bx_in(home);
    c.args(["config", "set-key", "--config"])
        .arg(config)
        .stdin(pty.stdin());
    c
}

/// The paste is never echoed, and echo is back once the key is saved.
#[test]
fn a_pasted_key_is_hidden_then_echo_is_restored() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("c.json");
    let mut pty = Pty::new();
    let child = prompt(&pty, &config).spawn().unwrap();
    pty.wait_echo(false);
    pty.type_in(format!("{KEY}\n").as_bytes());
    let out = finish(child);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(pty.echo(), "echo left off");
    assert!(std::fs::read_to_string(&config).unwrap().contains(KEY));
    let echoed = pty.echoed();
    assert!(
        !String::from_utf8_lossy(&echoed).contains(KEY),
        "the key was echoed: {echoed:?}"
    );
}

/// A signal at the prompt restores echo and still ends the process by that signal, so a shell
/// sees the real cause. Every signal the guard installs a handler for.
#[test]
fn a_signal_at_the_prompt_restores_echo_and_still_kills() {
    use std::os::unix::process::{CommandExt, ExitStatusExt};
    for sig in [
        libc::SIGINT,
        libc::SIGTERM,
        libc::SIGHUP,
        libc::SIGQUIT,
        libc::SIGABRT,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let pty = Pty::new();
        let mut c = prompt(&pty, &dir.path().join("c.json"));
        // SAFETY: async-signal-safe. No core file from SIGQUIT or SIGABRT, and the default
        // action even under a parent that ignores `sig`, as a background job's does.
        unsafe {
            c.pre_exec(move || {
                let none = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: 0,
                };
                libc::setrlimit(libc::RLIMIT_CORE, &none);
                libc::signal(sig, libc::SIG_DFL);
                Ok(())
            });
        }
        let child = c.spawn().unwrap();
        pty.wait_echo(false);
        // SAFETY: signals our own child.
        unsafe { libc::kill(child.id() as libc::pid_t, sig) };
        let out = finish(child);
        assert_eq!(out.status.signal(), Some(sig), "{:?}", out.status);
        assert!(pty.echo(), "signal {sig} left echo off");
    }
}

/// The rest of a multi-line paste is discarded, never left for the shell to run.
#[test]
fn a_multi_line_paste_leaves_nothing_for_the_shell() {
    let dir = tempfile::tempdir().unwrap();
    let mut pty = Pty::new();
    let child = prompt(&pty, &dir.path().join("c.json")).spawn().unwrap();
    pty.wait_echo(false);
    pty.type_in(format!("{KEY}\nrm -rf ~\n").as_bytes());
    let out = finish(child);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(pty.unread(), 0, "a line was left for the shell");
}

/// An ignored hangup stays ignored, as `nohup` relies on: the prompt survives it.
#[test]
fn nohup_survives_a_hangup_at_the_prompt() {
    use std::os::unix::process::CommandExt;
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("c.json");
    let mut pty = Pty::new();
    let mut c = prompt(&pty, &config);
    // SAFETY: async-signal-safe.
    unsafe {
        c.pre_exec(|| {
            libc::signal(libc::SIGHUP, libc::SIG_IGN);
            Ok(())
        });
    }
    let child = c.spawn().unwrap();
    pty.wait_echo(false);
    // SAFETY: signals our own child.
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGHUP) };
    pty.type_in(format!("{KEY}\n").as_bytes());
    let out = finish(child);
    assert!(out.status.success(), "{:?}", out.status);
    assert!(std::fs::read_to_string(&config).unwrap().contains(KEY));
}
