//! The one way every harness spawns `bx`: nothing inherited, and no route to a real provider.
// Each harness uses a different subset.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::Duration;

pub const BX: &str = env!("CARGO_BIN_EXE_bx");

/// Refuses connections: a regression that reaches the network exits 5 instead of billing.
pub const DEAD: &str = "http://127.0.0.1:1";

/// Every base URL `bx` reads from the environment.
pub const BASE_URLS: [&str; 4] = [
    "BX_COUNCIL_OPENROUTER_BASE_URL",
    "BX_COUNCIL_ANTHROPIC_BASE_URL",
    "BX_COUNCIL_OPENAI_BASE_URL",
    "BRAVE_SEARCH_BASE_URL",
];

/// Far above the slowest deliberate wait in any harness, far below CI's job limit.
const LIMIT: Duration = Duration::from_secs(120);

/// `bx` in a sandbox rooted at `home`. A test sets any key or mock URL it needs on top.
pub fn bx_in(home: &Path) -> Command {
    let mut c = Command::new(BX);
    sandbox(&mut c, home);
    c
}

/// Clears the environment, then restores only what a process needs to start. For any
/// command, so a shell that execs `bx` gets the same sandbox. Stdio defaults to what
/// `output()` uses, so a spawn never inherits the developer's terminal.
pub fn sandbox<'a>(c: &'a mut Command, home: &Path) -> &'a mut Command {
    c.env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let keep: &[&str] = if cfg!(windows) {
        &["PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE"]
    } else {
        &["PATH"]
    };
    for var in keep {
        if let Some(v) = std::env::var_os(var) {
            c.env(var, v);
        }
    }
    for var in ["XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME"] {
        c.env(var, home);
    }
    c.env("HOME", home_for(home));
    // `dirs` reads Known Folders, not these, so a Windows spawn is never fully sandboxed.
    #[cfg(windows)]
    for var in ["APPDATA", "LOCALAPPDATA"] {
        c.env(var, home);
    }
    for var in BASE_URLS {
        c.env(var, DEAD);
    }
    c.env("NO_PROXY", "*")
}

/// `root` as `bx` spells it: build from this any path compared to one `bx` prints.
#[cfg(not(target_os = "macos"))]
pub fn bx_root(root: &Path) -> PathBuf {
    root.to_path_buf()
}

#[cfg(target_os = "macos")]
pub fn bx_root(root: &Path) -> PathBuf {
    home_for(root).join("Library/Application Support")
}

#[cfg(not(target_os = "macos"))]
fn home_for(root: &Path) -> PathBuf {
    root.to_path_buf()
}

/// macOS ignores XDG and roots config and state in `$HOME/Library/Application Support`, so
/// HOME is a stub whose one link lands in `root`: the layout every test plants on Linux.
#[cfg(target_os = "macos")]
fn home_for(root: &Path) -> PathBuf {
    let home = root.join(".home");
    let lib = home.join("Library");
    std::fs::create_dir_all(&lib).unwrap();
    if let Err(e) = std::os::unix::fs::symlink(root, lib.join("Application Support")) {
        assert_eq!(e.kind(), std::io::ErrorKind::AlreadyExists, "{e}");
    }
    home
}

/// Waits for a spawned `bx`, failing rather than hanging: libtest has no per-test timeout,
/// and a hung child otherwise becomes a CI job killed with no output.
#[track_caller]
pub fn finish(child: Child) -> Output {
    // On a thread, so the pipes drain while the clock runs: polling `try_wait` unread
    // deadlocks once a child fills the pipe buffer.
    let pid = child.id();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || tx.send(child.wait_with_output()));
    match rx.recv_timeout(LIMIT) {
        Ok(out) => out.unwrap(),
        Err(_) => {
            // The waiting thread owns the child, so kill by pid: a hung `bx` must not
            // outlive the test holding a mock's port.
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as libc::pid_t, libc::SIGKILL);
            }
            panic!("bx ({pid}) did not exit within {LIMIT:?}");
        }
    }
}

pub trait Bounded {
    /// `output()`, bounded by [`finish`]. Stdio is the caller's, piped by [`sandbox`].
    fn bounded_output(&mut self) -> Output;
}

impl Bounded for Command {
    #[track_caller]
    fn bounded_output(&mut self) -> Output {
        finish(self.spawn().unwrap())
    }
}
