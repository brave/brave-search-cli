//! Argument handling that never reaches the network. The exit-code contract itself lives
//! in `exit_codes.rs`; these two cases are about not crashing on the way to it.

use std::process::Command;

/// Isolated from the developer's config: `load_config` runs before the checks below, so a
/// malformed real config would fail these tests with an unrelated error.
fn bx(args: &[&str]) -> std::process::Output {
    let cfg = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(cfg.path(), "{}").unwrap();
    Command::new(env!("CARGO_BIN_EXE_bx"))
        .args(args)
        .arg("--config")
        .arg(cfg.path())
        .env_remove("BRAVE_SEARCH_API_KEY")
        .env_remove("BRAVE_API_KEY")
        .env_remove("BRAVE_SEARCH_BASE_URL")
        .output()
        .unwrap()
}

#[test]
fn a_huge_timeout_is_rejected_instead_of_panicking() {
    for bad in ["0", "86401", "18446744073709551615"] {
        let out = bx(&["web", "q", "--api-key", "TEST", "--timeout", bad]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "--timeout {bad}: {stderr}");
        assert!(stderr.contains("between 1 and 86400"), "{stderr}");
    }

    // The bound is inclusive, and the largest accepted value must not panic inside ureq's
    // deadline arithmetic — which is what an unbounded --timeout used to do, aborting with
    // no usable message at all.
    let out = bx(&[
        "web",
        "q",
        "--api-key",
        "TEST",
        "--timeout",
        "86400",
        "--base-url",
        "http://127.0.0.1:1",
    ]);
    assert_ne!(out.status.code(), Some(101), "panicked");
    assert_eq!(out.status.code(), Some(5), "expected a connection failure");
}

#[test]
fn a_non_utf8_argument_is_reported_not_panicked() {
    // `std::env::args()` panics on these before clap ever sees them; bx reads `args_os`.
    let out = Command::new(env!("CARGO_BIN_EXE_bx"))
        .arg(non_utf8_arg())
        .env_remove("BRAVE_SEARCH_API_KEY")
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("not valid UTF-8"), "{stderr}");
}

#[cfg(unix)]
fn non_utf8_arg() -> std::ffi::OsString {
    use std::os::unix::ffi::OsStringExt;
    std::ffi::OsString::from_vec(vec![0xff, 0xfe])
}

#[cfg(windows)]
fn non_utf8_arg() -> std::ffi::OsString {
    // Windows argv is UTF-16; an unpaired surrogate is the equivalent unrepresentable case.
    use std::os::windows::ffi::OsStringExt;
    std::ffi::OsString::from_wide(&[0xd800])
}
