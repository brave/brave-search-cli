//! `council.json`, beside `config.json` (see [`path`]). Its own file: it holds three provider keys
//! and is hand-edited, so a typo in it cannot cost `config.json`. Written private, never 0644.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Council {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openrouter_api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic_api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_api_key: Option<String>,
    /// Vendors that bypass OpenRouter: absent means [`DEFAULT_DIRECT`], `[]` means none. Direct,
    /// the price is knowable; through the router it depends on the endpoint picked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direct: Option<Vec<String>>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub aliases: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub panels: BTreeMap<String, Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_task: Option<String>,
    /// Read through [`zdr`], never directly: absent means on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_min_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_usd: Option<f64>,
    /// `[input, output]` USD per million tokens, plus an optional cache-read multiplier on
    /// input; keyed by resolved slug. Read through [`rate_for`].
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub rates: BTreeMap<String, Vec<f64>>,
    /// OpenRouter provider slugs a model may be served by, keyed by resolved slug; empty leaves
    /// routing to the router. Endpoints for one slug differ in reliability, not only price.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub providers: BTreeMap<String, Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brave_price_per_call_usd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub referer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// Opaque, never derived: a derived `Debug` would print the keys.
impl std::fmt::Debug for Council {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Council { .. }")
    }
}

/// Which vendors bypass OpenRouter when `council.json` names none; one definition for
/// `transport_for` and `doctor`.
pub const DEFAULT_DIRECT: &[&str] = &["anthropic"];

/// Zero data retention, on unless `"zdr": false`. Enforced per request only on OpenRouter; each
/// wire's posture is its `Retention`, printed by `bx council doctor`.
pub fn zdr(c: &Council) -> bool {
    c.zdr != Some(false)
}

/// Where `council.json` lives: the local config directory, never Windows' roaming `%APPDATA%`,
/// which a domain login syncs to a server. It holds keys.
pub fn path() -> Option<PathBuf> {
    dirs::config_local_dir().map(|d| d.join("brave-search").join("council.json"))
}

pub fn load(override_path: Option<&Path>) -> Result<Council, String> {
    let Some(p) = override_path.map(Path::to_path_buf).or_else(path) else {
        return Ok(Council::default());
    };
    // Windows editors write a BOM; serde refuses one.
    match super::util::read_local(&p, super::util::MAX_LOCAL_FILE)
        .map(|s| s.trim_start_matches('\u{feff}').to_string())
    {
        Ok(s) if s.trim().is_empty() => Ok(Council::default()),
        // Position, never `{e}`: serde quotes the offending value, which may be a pasted key.
        Ok(s) => {
            let c = serde_json::from_str(&s).map_err(|e| {
                format!(
                    "failed to parse {} at line {} column {}{}",
                    p.display(),
                    e.line(),
                    e.column(),
                    unknown_key(&e.to_string())
                )
            })?;
            check(&c).map_err(|e| format!("{}: {e}", p.display()))?;
            Ok(c)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && override_path.is_none() => {
            Ok(Council::default())
        }
        Err(e) => Err(format!("cannot read {}: {e}", p.display())),
    }
}

/// Values serde accepts that would silently reroute or uncap a run.
fn check(c: &Council) -> Result<(), String> {
    let direct: Vec<&str> = super::wire::TABLE
        .iter()
        .map(|w| w.name)
        .filter(|n| *n != super::wire::OPENROUTER.name)
        .collect();
    if let Some(d) = c
        .direct
        .iter()
        .flatten()
        .find(|d| !direct.contains(&d.as_str()))
    {
        return Err(format!(
            "direct: unknown vendor \"{}\" (one of: {})",
            super::util::truncate(&super::util::sanitize(d), 40),
            direct.join(", ")
        ));
    }
    // Unpriced Brave calls are never charged against `--max-usd`.
    if c.brave_price_per_call_usd
        .is_some_and(|v| publishable_usd(v).is_none())
    {
        return Err("brave_price_per_call_usd must be a non-negative USD amount".into());
    }
    Ok(())
}

/// Names an unknown top-level key, but only one shaped like a field name, never a pasted secret.
fn unknown_key(serde_error: &str) -> String {
    let Some(rest) = serde_error.strip_prefix("unknown field `") else {
        return String::new();
    };
    match rest.split('`').next() {
        Some(k) if k.len() <= 32 && k.bytes().all(|b| b.is_ascii_lowercase() || b == b'_') => {
            format!(": unknown key \"{k}\"")
        }
        _ => ": unknown key".into(),
    }
}

/// Re-tightens the directory every save (a restore can leave it 0755); the file goes through the
/// one private writer.
pub fn save(c: &Council) -> Result<(), String> {
    let p = path().ok_or("cannot determine config directory")?;
    let dir = p.parent().ok_or("config path has no parent")?;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
            .map_err(|e| e.to_string())?;
    }
    let body = serde_json::to_string_pretty(c).map_err(|e| e.to_string())?;
    super::output::write_private(&p, &body)
}

// ── Compiled-in defaults ─────────────────────────────────────────────

/// The compiled-in panels, `-m @default` and `-m @fast` (the same idea at a fraction of the
/// cost). Version families, not dated snapshots: a baked-in snapshot expires and 404s like a
/// typo. One lab per seat, for disagreement rather than a ranking; every seat clears
/// `MODEL_MAX_OUT`, so a truncated round can escalate.
pub const DEFAULT_PANELS: &[(&str, &[&str])] = &[
    (
        "default",
        &[
            "anthropic/claude-opus-5.5",
            // Never the `-pro` variant: see @fast's first seat.
            "openai/gpt-6-astra",
            "moonshotai/kimi-k3",
            "z-ai/glm-5.3",
        ],
    ),
    (
        "fast",
        &[
            // Not `-pro`: same rate card, but it bills parallel threads over the whole input.
            "openai/gpt-5.6-luna",
            "minimax/minimax-m3",
            // Not the floating `~…-latest` alias: OpenRouter cannot certify one for zdr.
            "z-ai/glm-5.3-flash",
            "deepseek/deepseek-v4-flash",
            "qwen/qwen3.5-397b-a17b",
        ],
    ),
];

/// The panel a run uses when neither `-m` nor the task nor config names one.
pub const DEFAULT_PANEL: &[&str] = DEFAULT_PANELS[0].1;

/// Other names for a compiled-in panel: `@sota` says what `@default` is for.
pub const PANEL_ALIASES: &[(&str, &str)] = &[("sota", "default")];

/// Every other name a compiled-in panel answers to, for `doctor` to print.
pub fn panel_aliases_for(panel: &str) -> Vec<&'static str> {
    PANEL_ALIASES
        .iter()
        .filter(|(_, target)| *target == panel)
        .map(|(a, _)| *a)
        .collect()
}

/// The compiled-in name a panel alias points at. Public: config lookup has to also happen on the
/// canonical name, or `@sota` would miss a config `default`.
pub fn canonical_panel(name: &str) -> &str {
    PANEL_ALIASES
        .iter()
        .find(|(a, _)| *a == name)
        .map_or(name, |(_, target)| *target)
}

/// A compiled-in panel by name, so `-m @fast` needs no config file.
pub fn builtin_panel(name: &str) -> Option<&'static [&'static str]> {
    let name = canonical_panel(name);
    DEFAULT_PANELS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, p)| *p)
}

/// Short names for the roster. Vocabulary, so it lives in config; these are the fallback.
/// Wider than `DEFAULT_PANEL` on purpose: swapping a member should not need a full slug.
pub const DEFAULT_ALIASES: &[(&str, &str)] = &[
    ("opus", "anthropic/claude-opus-5.5"),
    ("astra", "openai/gpt-6-astra"),
    ("sol", "openai/gpt-5.6-sol"),
    ("glm", "z-ai/glm-5.3"),
    ("minimax", "minimax/minimax-m3"),
    ("qwen", "qwen/qwen3.5-397b-a17b"),
    // Both GLM tiers: glm sits in @default, glm-flash in @fast.
    ("glm-flash", "z-ai/glm-5.3-flash"),
    ("kimi", "moonshotai/kimi-k3"),
    // Dated on purpose: unsuffixed v4-pro is an older, dearer build.
    ("deepseek", "deepseek/deepseek-v4-pro-0813"),
    // @fast's members, nameable on their own.
    ("luna", "openai/gpt-5.6-luna"),
    ("deepseek-flash", "deepseek/deepseek-v4-flash"),
];

/// Compiled-in rates, for direct wires only: an OpenRouter bill is read back from `usage.cost`.
/// Valid while the vendor's price is flat in context length.
const FALLBACK_RATES: &[(&str, Rate)] = &[("anthropic/claude-opus-5.5", [4.0, 20.0, 0.05])];

/// `[input, output, cache_read]`: USD per million tokens, and the multiplier on input a cache
/// read is billed at.
pub type Rate = [f64; 3];

/// The cache-read multiplier when a configured rate names none.
const DEFAULT_CACHE_READ: f64 = 0.1;

/// Brave's list price, $5 per 1,000 requests on the Search plan (brave.com/search/api). Free
/// monthly credits are ignored, so it over-reports for a light user, the safe direction.
const FALLBACK_BRAVE_USD: f64 = 0.005;

/// What one Brave call costs. `None` only for a price `load` would have refused; an explicit `0`
/// (free credits, an internal endpoint) is kept.
pub fn brave_price(c: &Council) -> Option<f64> {
    publishable_usd(c.brave_price_per_call_usd.unwrap_or(FALLBACK_BRAVE_USD))
}

/// Whether the price above is the operator's, for `doctor`.
pub fn brave_price_is_configured(c: &Council) -> bool {
    c.brave_price_per_call_usd.is_some()
}

/// A USD figure this program will publish, the one predicate every price passes: non-negative (a
/// negative bound satisfies any `--max-usd`), finite, and finite in microdollars, since
/// `output::micro` scales by 1e6 and an infinity renders as `null`.
pub fn publishable_usd(v: f64) -> Option<f64> {
    (v.is_finite() && v >= 0.0 && (v * 1e6).is_finite()).then_some(v)
}

/// The most a configured rate may be per million tokens: above it `worst_case` overflows. Far
/// above any real price, so it only bounds a typo.
const MAX_RATE_USD: f64 = 1e6;

/// A slug's rate, configured else compiled-in. An unusable one is ignored, with a warning.
pub fn rate_for(c: &Council, slug: &str) -> Option<Rate> {
    let r = match c.rates.get(slug).map(Vec::as_slice) {
        None => {
            return FALLBACK_RATES
                .iter()
                .find(|(s, _)| *s == slug)
                .map(|(_, r)| *r);
        }
        Some(&[i, o]) => [i, o, DEFAULT_CACHE_READ],
        Some(&[i, o, read]) => [i, o, read],
        Some(_) => [f64::NAN; 3],
    };
    let ok = r
        .iter()
        .all(|v| publishable_usd(*v).is_some_and(|v| v <= MAX_RATE_USD))
        && r[2] > 0.0
        && r[2] <= 1.0;
    if !ok {
        // Said, not swallowed: an unpriced member is one `--max-usd` does not cover.
        note!(
            "warning: ignoring unusable rate for {slug}: {:?}",
            c.rates[slug]
        );
    }
    ok.then_some(r)
}

/// The endpoints a model may be served by, if the operator named any; keyed as [`rate_for`] is.
pub fn providers_for(c: &Council, slug: &str) -> Vec<String> {
    c.providers.get(slug).cloned().unwrap_or_default()
}

/// `entry` with its model chased through the alias table. Looked up without its wire prefix
/// (`direct:opus` is `opus`); the first prefix named, on the entry else a value, is kept. A cycle
/// is an error.
pub fn alias<'a>(c: &'a Council, entry: &'a str) -> Result<String, String> {
    let bare = super::output::bare_slug(entry);
    let mut wire = &entry[..entry.len() - bare.len()];
    let mut model = bare;
    let mut seen = Vec::new();
    while let Some(v) = alias_once(c, model) {
        if seen.contains(&model) {
            return Err(format!("alias '{bare}' loops back to '{model}'"));
        }
        seen.push(model);
        model = super::output::bare_slug(v);
        if wire.is_empty() {
            wire = &v[..v.len() - model.len()];
        }
    }
    Ok(format!("{wire}{model}"))
}

fn alias_once<'a>(c: &'a Council, name: &str) -> Option<&'a str> {
    c.aliases.get(name).map(String::as_str).or_else(|| {
        DEFAULT_ALIASES
            .iter()
            .find(|(a, _)| *a == name)
            .map(|(_, s)| *s)
    })
}

// ── Credentials and endpoints ────────────────────────────────────────

/// The allowlist is `Wire.host`, plus loopback for a local mock: every inference URL is
/// `base + wire.path`. The carve-out means a local process that can set a `BX_COUNCIL_*_BASE_URL`
/// can receive a key.
pub fn check_base_url(w: &super::wire::Wire, url: &str) -> Result<String, String> {
    let url = url.trim_end_matches('/');
    if url == w.host {
        return Ok(url.to_string());
    }
    if let Some(rest) = url.strip_prefix("http://") {
        let (authority, path) = rest.split_once('/').map_or((rest, ""), |(a, p)| (a, p));
        // Userinfo hides the host (`http://127.0.0.1@evil.com` is evil.com); a `?` or `#` would
        // swallow the API path appended to the base.
        if !authority.contains('@')
            && !path.contains(['?', '#'])
            && let Ok((host, port)) = crate::parse_authority(authority)
        {
            // Resolved once and pinned to the literal IP, as the Brave allowlist is: ureq
            // re-resolves a name on every connect, which is a rebinding window.
            let literal = host
                .parse::<std::net::IpAddr>()
                .map(|ip| ip.is_loopback().then(|| authority.to_string()))
                .unwrap_or_else(|_| {
                    host.eq_ignore_ascii_case("localhost")
                        .then(|| crate::resolve_localhost(host, port.unwrap_or(80)).ok())
                        .flatten()
                        .map(|ip| match port {
                            Some(p) => format!("{ip}:{p}"),
                            None => ip,
                        })
                });
            if let Some(authority) = literal {
                let sep = if path.is_empty() { "" } else { "/" };
                return Ok(format!("http://{authority}{sep}{path}"));
            }
        }
    }
    // Never echoed: userinfo, path, query and fragment can each carry a key.
    Err(format!(
        "{} base URL not allowed\nhint: allowed is {}, or http://localhost:<port>",
        w.name, w.host
    ))
}

/// Our own variables, never the vendor SDKs': those point other tools at routers and proxies.
pub fn base_url(w: &super::wire::Wire) -> Result<String, String> {
    let env = match w.name {
        "anthropic" => "BX_COUNCIL_ANTHROPIC_BASE_URL",
        "openai" => "BX_COUNCIL_OPENAI_BASE_URL",
        _ => "BX_COUNCIL_OPENROUTER_BASE_URL",
    };
    match std::env::var(env)
        .ok()
        .and_then(crate::config::trim_non_empty)
    {
        Some(u) => check_base_url(w, &u).map_err(|e| format!("{env}: {e}")),
        None => Ok(w.host.to_string()),
    }
}

/// Env var, then config file. Never a CLI flag: argv is world-readable and logged.
pub fn key_for(c: &Council, wire: &str) -> Option<String> {
    let from_file = match wire {
        "anthropic" => &c.anthropic_api_key,
        "openai" => &c.openai_api_key,
        _ => &c.openrouter_api_key,
    };
    std::env::var(super::resolve::env_of(wire))
        .ok()
        .and_then(crate::config::trim_non_empty)
        .or_else(|| from_file.clone().and_then(crate::config::trim_non_empty))
}

/// Refuses a key that is obviously another provider's, before sending leaks it.
pub fn check_key(wire: &str, key: &str) -> Result<(), String> {
    // Every other wire's prefix; an unknown wire name is checked against all of them.
    let wrong = super::wire::TABLE
        .iter()
        .filter(|w| w.name != wire)
        .map(|w| w.key_prefix)
        .find(|p| key.starts_with(p));
    match wrong {
        Some(p) => Err(format!(
            "{wire} key starts with '{p}', which is another provider's prefix"
        )),
        None if key.len() < 8 => Err(format!("{wire} key is too short")),
        // Printable ASCII: a key is a header value, and ureq checks only `authorization`, so a
        // bad `x-api-key` is sent, or fails mid-run after the panel is paid for.
        None if key.chars().any(|c| !c.is_ascii_graphic()) => Err(format!(
            "{wire} key contains a character that cannot appear in a header value"
        )),
        None => Ok(()),
    }
}

/// The two config-supplied header values, refused here rather than by ureq mid-run.
pub fn check_headers(c: &Council) -> Result<(), String> {
    for (name, v) in [("referer", &c.referer), ("title", &c.title)] {
        if v.as_deref()
            .is_some_and(|v| v.chars().any(char::is_control))
        {
            return Err(format!("council.{name} contains a control character"));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "cfg_tests.rs"]
mod tests;
