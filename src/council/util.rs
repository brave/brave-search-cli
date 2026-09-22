//! Small pure helpers, the clock, and the bounded file readers. No HTTP. Crate-wide: `api` reads
//! a `Retry-After` date with the calendar below.

use std::time::{SystemTime, UNIX_EPOCH};

/// Replaces every secret with `***`, overlapping matches as one span. Redact before truncating:
/// a cut key leaves a greppable half.
///
/// The floor only stops an empty secret from mangling the text. Counted in characters, so
/// scrubbing can only shorten, which `keep_draft` relies on.
pub fn redact(s: &str, secrets: &[String]) -> String {
    // Raw and JSON-escaped: traces and transcripts are redacted after serialization.
    let keys: Vec<String> = secrets
        .iter()
        .filter(|k| k.chars().count() >= 4)
        .flat_map(|k| {
            let json = serde_json::to_string(k).unwrap_or_default();
            // Exactly the two delimiters: a key ending in `"` serializes as `"abc\""`.
            let escaped = json
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .unwrap_or_default()
                .to_string();
            [
                Some(k.clone()),
                (!escaped.is_empty() && escaped != *k).then_some(escaped),
            ]
        })
        .flatten()
        .collect();
    // Every occurrence of every spelling, overlapping ones merged, so no key keeps a tail.
    let mut hits: Vec<(usize, usize)> = Vec::new();
    for k in &keys {
        let mut from = 0;
        while let Some(i) = s[from..].find(k.as_str()) {
            let at = from + i;
            hits.push((at, at + k.len()));
            from = at + s[at..].chars().next().map_or(1, char::len_utf8);
        }
    }
    if hits.is_empty() {
        return s.to_string();
    }
    hits.sort_unstable();
    let mut out = String::with_capacity(s.len());
    let (mut done, mut i) = (0, 0);
    while i < hits.len() {
        let (mut start, mut end) = hits[i];
        while i < hits.len() && hits[i].0 <= end {
            end = end.max(hits[i].1);
            i += 1;
        }
        // Opening inside a JSON escape (`\n` before a key starting with `n`): take the backslash
        // too, or `\***` is invalid JSON.
        if s[..start].bytes().rev().take_while(|&b| b == b'\\').count() % 2 == 1 {
            start -= 1;
        }
        out.push_str(&s[done..start]);
        out.push_str("***");
        done = end;
    }
    out.push_str(&s[done..]);
    out
}

/// Truncates to `max` characters, not bytes, appending an elision note.
pub fn truncate(s: &str, max: usize) -> String {
    match s.char_indices().nth(max) {
        None => s.to_string(),
        Some((cut, _)) => format!("{}… [truncated]", &s[..cut]),
    }
}

/// Strips ANSI escapes, control characters other than `\n` and `\t`, the line separators, and
/// the bidi and zero-width formats, from every string bound for a terminal or a log.
pub fn sanitize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\u{1b}' => {
                // Other introducers lose only ESC and the next byte; what is left is inert text.
                if chars.next_if_eq(&'[').is_some() {
                    // ECMA-48 §5.4: parameters `0x20`..`0x3f`, then one final byte `@`..`~`.
                    // Terminals ignore NUL and DEL mid-sequence, so the scan does too.
                    while chars
                        .next_if(|c| matches!(c, '\u{20}'..='\u{3f}' | '\0' | '\u{7f}'))
                        .is_some()
                    {}
                    chars.next_if(|c| matches!(c, '\u{40}'..='\u{7e}'));
                } else {
                    // Only printable ASCII continues an escape; keep a `\n` or `é` after it.
                    chars.next_if(|c| matches!(c, '\u{20}'..='\u{7e}'));
                }
            }
            '\n' | '\t' => out.push(c),
            // `is_control` is C0, DEL and C1 (U+009B is a one-character CSI). Listed: the line
            // separators, which forge a break; the bidi overrides, isolates and marks, which
            // reorder text; the zero-width formats; and the Tags block, invisible ASCII a model
            // reads (it costs the subdivision flags). Kept, because real text needs them: ZWJ,
            // the variation selectors (a run of them can carry bytes), U+2061-4 and the Hangul
            // fillers.
            c if c.is_control()
                || matches!(c as u32,
                    0x00ad | 0x034f | 0x061c | 0x180e | 0x200b | 0x200e | 0x200f | 0x2028 | 0x2029
                    | 0x202a..=0x202e | 0x2060 | 0x2066..=0x206f | 0xfeff | 0xfff9..=0xfffb
                    | 0xe0000..=0xe007f) => {}
            c => out.push(c),
        }
    }
    out
}

/// Filename-safe form of a model slug: every character outside `[A-Za-z0-9._-]` becomes `_`, cut
/// at 80. Safe, not unique: the caller's index prefix prevents collisions. `:` must go: on NTFS
/// `a:free` opens an alternate data stream.
pub fn slugify(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .take(80)
        .collect()
}

/// A number spelled as digits only: `parse` also takes a leading `+`.
pub fn unsigned<T: std::str::FromStr>(s: &str) -> Option<T> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

// ── Time ─────────────────────────────────────────────────────────────

pub fn secs_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Civil date from a Unix day number (Howard Hinnant's `civil_from_days`).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// The inverse of `civil` (Howard Hinnant's `days_from_civil`). Checked: the year comes from a
/// provider's `Retry-After`.
fn days_from_civil(y: i64, m: u32, d: u32) -> Option<i64> {
    let y = y.checked_sub(i64::from(m <= 2))?;
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era.checked_mul(146_097)?
        .checked_add(doe)?
        .checked_sub(719_468)
}

/// Days in a month, so a date that cannot exist is refused rather than normalised.
fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            28 + u32::from(
                y.rem_euclid(4) == 0 && (y.rem_euclid(100) != 0 || y.rem_euclid(400) == 0),
            )
        }
    }
}

/// An IMF-fixdate (`Wed, 21 Oct 2026 07:28:00 GMT`), the form RFC 9110 requires senders to use,
/// as Unix seconds. The obsolete forms are `None`.
pub fn http_date(s: &str) -> Option<u64> {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let (_, rest) = s.trim().split_once(", ")?;
    let mut f = rest.split(' ');
    let (day, mon, year, time, zone) = (f.next()?, f.next()?, f.next()?, f.next()?, f.next()?);
    if zone != "GMT" || f.next().is_some() {
        return None;
    }
    let mut hms = time.split(':');
    let (h, mi, sec) = (hms.next()?, hms.next()?, hms.next()?);
    if hms.next().is_some() {
        return None;
    }
    let m = MONTHS.iter().position(|x| *x == mon)? as u32 + 1;
    // Calendar-valid, not merely digit-shaped: `31 Feb` and `07:60:00` are refused.
    let (day, year) = (unsigned::<u32>(day)?, unsigned::<i64>(year)?);
    if !(1..=days_in_month(year, m)).contains(&day) {
        return None;
    }
    let (h, mi, sec) = (
        unsigned::<u32>(h)?,
        unsigned::<u32>(mi)?,
        unsigned::<u32>(sec)?,
    );
    if h > 23 || mi > 59 || sec > 60 {
        return None;
    }
    let days = days_from_civil(year, m, day)?;
    let secs = i64::from(3600 * h + 60 * mi + sec);
    days.checked_mul(86_400)?.checked_add(secs)?.try_into().ok()
}

fn parts(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    let (y, mo, d) = civil(days);
    (
        y,
        mo,
        d,
        (rem / 3600) as u32,
        (rem % 3600 / 60) as u32,
        (rem % 60) as u32,
    )
}

/// UTC timestamp, `2026-09-03T11:57:39Z`.
pub fn iso8601(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = parts(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

pub fn iso8601_now() -> String {
    iso8601(secs_now())
}

/// The width of a `run_id`, so a reader can recognise one without re-deriving the format.
pub const RUN_ID_LEN: usize = 22;

/// Whether a directory name is exactly what `run_id` mints: `prune` deletes what this accepts.
pub fn is_run_id(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() == RUN_ID_LEN
        && b[8] == b'-'
        && b[15] == b'-'
        && b[..8].iter().all(u8::is_ascii_digit)
        && b[9..15].iter().all(u8::is_ascii_digit)
        && b[16..]
            .iter()
            .all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'))
}

/// `<YYYYmmdd-HHMMSS>-<6 hex>`. The suffix makes a same-second collision unlikely; the caller's
/// retry on `AlreadyExists` makes it impossible.
pub fn run_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let (y, mo, d, h, mi, s) = parts(now / 1_000_000_000);
    // splitmix64 finalizer over the low nanosecond bits, the only entropy here.
    let mut x = now ^ (u64::from(std::process::id()) << 32);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    format!(
        "{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}-{:06x}",
        (x ^ (x >> 31)) & 0xff_ffff
    )
}

// ── SHA-256 ──────────────────────────────────────────────────────────

/// First 8 bytes of SHA-256, hex: `sha256sum brief.md | cut -c1-16` reproduces it without bx.
pub fn sha256_short(data: &[u8]) -> String {
    let d = ring::digest::digest(&ring::digest::SHA256, data);
    d.as_ref()[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// The most any file this program reads back may hold: a bound, not a size.
pub const MAX_LOCAL_FILE: u64 = 8 * 1024 * 1024;

/// A tail read, and whether it is the whole file: past the cap, `observed` describes a window.
#[derive(Default, Debug)]
pub struct Tail {
    pub text: String,
    /// `false` when bytes before the window were discarded, so the rows read are a suffix.
    pub whole: bool,
}

/// The last `max` bytes of an append-only log, from a line boundary. Not [`read_local`], which
/// refuses a file over its cap, and `prune` never trims `history.jsonl`.
///
/// Judged on the handle, never through a link, like the `O_NOFOLLOW` writer. The partial first
/// line is dropped while still bytes: a seek can land mid-character, and UTF-8 decoding would
/// then lose the whole file.
pub fn read_tail(path: &std::path::Path, max: u64) -> std::io::Result<Tail> {
    use std::io::{Read, Seek};
    let mut f = open_read(path, true)?;
    let meta = judged(&f, true)?;
    // One byte before the window tells a cut on a line boundary from one mid-line.
    let from = meta.len().saturating_sub(max);
    f.seek(std::io::SeekFrom::Start(from.saturating_sub(1)))?;
    let mut buf = Vec::with_capacity(meta.len().min(max) as usize + 1);
    // Never past the stat'd length, so a record still being appended is not read; a torn last
    // line costs one row.
    f.take(meta.len() - from.saturating_sub(1))
        .read_to_end(&mut buf)?;
    if from > 0 {
        let cut = match buf.first() {
            Some(b'\n') => 1,
            // No newline: one record longer than the cap, nothing whole to return.
            _ => buf
                .iter()
                .position(|b| *b == b'\n')
                .map_or(buf.len(), |i| i + 1),
        };
        buf.drain(..cut);
    }
    // Lossy only where the file itself is corrupt, costing that record alone.
    Ok(Tail {
        text: String::from_utf8(buf)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()),
        whole: from == 0,
    })
}

/// Opens for reading, refusing a symlink at the last component when `no_follow`. `O_NONBLOCK`,
/// so a FIFO with no writer cannot hang the `open`; the caller then judges the descriptor, not
/// the name, which `lstat` then `open` would resolve twice.
fn open_read(path: &std::path::Path, no_follow: bool) -> std::io::Result<std::fs::File> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_NONBLOCK | if no_follow { libc::O_NOFOLLOW } else { 0 });
    }
    // `BACKUP_SEMANTICS` lets a directory open, so it is judged "not a regular file" as
    // elsewhere; `OPEN_REPARSE_POINT` is Windows' `O_NOFOLLOW`.
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        let reparse = if no_follow { 0x0020_0000 } else { 0 };
        o.custom_flags(0x0200_0000 | reparse); // FILE_FLAG_BACKUP_SEMANTICS
    }
    o.open(path)
}

/// What an open file must be before it is read: a regular file and, under `no_follow`, the only
/// name its inode has, since a hard link passes `O_NOFOLLOW`. `> 1`, not `!= 1`: a file renamed
/// over while open has zero links and is a correct snapshot. Unix only: the Windows link count
/// is nightly.
fn judged(f: &std::fs::File, no_follow: bool) -> std::io::Result<std::fs::Metadata> {
    let meta = f.metadata()?;
    if !meta.is_file() {
        return Err(std::io::Error::other("not a regular file"));
    }
    #[cfg(unix)]
    if no_follow {
        use std::os::unix::fs::MetadataExt;
        if meta.nlink() > 1 {
            return Err(std::io::Error::other("not an exclusively-linked file"));
        }
    }
    #[cfg(not(unix))]
    let _ = no_follow;
    Ok(meta)
}

/// Opens a file the operator named, where a link is a choice, not a plant.
pub fn open_nonblocking(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    open_read(path, false)
}

/// Bytes of a regular file, at most `max`: a FIFO or `/dev/zero` in its place must not hang or
/// exhaust the run. Pre-sized from `stat`, as `Take` gives `read_to_end` no hint.
fn read_under(path: &std::path::Path, max: u64, no_follow: bool) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let f = open_read(path, no_follow)?;
    let meta = judged(&f, no_follow)?;
    let mut buf = Vec::with_capacity(meta.len().min(max) as usize + 1);
    f.take(max.saturating_add(1)).read_to_end(&mut buf)?;
    if buf.len() as u64 > max {
        return Err(std::io::Error::other(format!(
            "file exceeds maximum size ({max} bytes)"
        )));
    }
    Ok(buf)
}

fn utf8(b: Vec<u8>) -> std::io::Result<String> {
    String::from_utf8(b).map_err(|_| std::io::Error::other("not valid UTF-8"))
}

/// Text of a file the operator chose: `council.json`, a task, a cached roster. Links followed,
/// for dotfile managers.
pub fn read_local(path: &std::path::Path, max: u64) -> std::io::Result<String> {
    utf8(read_under(path, max, false)?)
}

/// Text of a file inside a run directory, where bx minted every name, so a link is a plant:
/// `O_NOFOLLOW` refuses it in the `open` itself, leaving no `lstat` race.
pub fn read_artifact(path: &std::path::Path, max: u64) -> std::io::Result<String> {
    utf8(read_artifact_bytes(path, max)?)
}

/// The same, as bytes: a `brief.md` that is not UTF-8 still has a fingerprint.
pub fn read_artifact_bytes(path: &std::path::Path, max: u64) -> std::io::Result<Vec<u8>> {
    read_under(path, max, true)
}
