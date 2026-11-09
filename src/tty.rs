//! Hides a pasted secret on a Unix terminal.

/// Echo off on a terminal stdin while held; restored on drop, and on a signal that would end the
/// process with echo still off.
#[cfg(unix)]
pub struct Hidden {
    old: [Option<libc::sigaction>; 4],
}

/// No terminal to hide on: `hide` is always `None`.
#[cfg(not(unix))]
pub enum Hidden {}

#[cfg(unix)]
static SAVED: std::sync::OnceLock<libc::termios> = std::sync::OnceLock::new();

#[cfg(unix)]
const SIGNALS: [libc::c_int; 4] = [libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGQUIT];

/// The saved terminal, unread input discarded: the rest of a multi-line paste would otherwise run
/// as shell commands. Not `TCSAFLUSH`, which on macOS first waits for output to drain.
#[cfg(unix)]
fn restore() {
    if let Some(t) = SAVED.get() {
        // SAFETY: both are async-signal-safe, on a valid `termios`.
        unsafe {
            libc::tcflush(0, libc::TCIFLUSH);
            libc::tcsetattr(0, libc::TCSANOW, t);
        }
    }
}

#[cfg(unix)]
extern "C" fn on_signal(sig: libc::c_int) {
    restore();
    // SAFETY: `SA_RESETHAND` restored the default action, so `sig` ends the process as it would
    // have; it stays blocked until this handler returns.
    unsafe { libc::raise(sig) };
}

#[cfg(unix)]
fn put_back(old: &[Option<libc::sigaction>; 4]) {
    for (sig, prev) in SIGNALS.iter().zip(old) {
        if let Some(p) = prev {
            // SAFETY: `p` is the disposition `sigaction` reported for `sig`.
            unsafe { libc::sigaction(*sig, p, std::ptr::null_mut()) };
        }
    }
}

/// Echo off while the guard lives, when stdin is a terminal; `None` leaves input visible.
#[cfg(unix)]
pub fn hide() -> Option<Hidden> {
    if !std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return None;
    }
    let mut t = std::mem::MaybeUninit::<libc::termios>::uninit();
    // SAFETY: fills `t` on success.
    if unsafe { libc::tcgetattr(0, t.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: initialised by the call above.
    let saved = *SAVED.get_or_init(|| unsafe { t.assume_init() });
    // Before hiding, so no signal finds echo off under the default action. An ignored signal
    // stays ignored: `nohup` must still survive a hangup.
    let mut old = [None; 4];
    for (sig, slot) in SIGNALS.iter().zip(&mut old) {
        // SAFETY: a zeroed `sigaction` is valid; the calls read and write only these.
        unsafe {
            let mut prev: libc::sigaction = std::mem::zeroed();
            if libc::sigaction(*sig, std::ptr::null(), &mut prev) != 0
                || prev.sa_sigaction == libc::SIG_IGN
            {
                continue;
            }
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
            sa.sa_flags = libc::SA_RESETHAND;
            libc::sigfillset(&mut sa.sa_mask);
            if libc::sigaction(*sig, &sa, std::ptr::null_mut()) == 0 {
                *slot = Some(prev);
            }
        }
    }
    let mut quiet = saved;
    quiet.c_lflag &= !(libc::ECHO | libc::ECHONL);
    // `TCSANOW`: `TCSAFLUSH` would drop a paste that beat the prompt.
    // SAFETY: a valid `termios`.
    if unsafe { libc::tcsetattr(0, libc::TCSANOW, &quiet) } != 0 {
        put_back(&old);
        return None;
    }
    Some(Hidden { old })
}

#[cfg(not(unix))]
pub fn hide() -> Option<Hidden> {
    None
}

#[cfg(unix)]
impl Drop for Hidden {
    fn drop(&mut self) {
        restore();
        put_back(&self.old);
    }
}
