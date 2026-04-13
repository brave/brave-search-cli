//! Code signature checks for downloaded release binaries (self-update).
//!
//! Pinning is aligned with [Brave signing keys](https://brave.com/signing-keys/):
//! - **Windows:** SHA-1 thumbprints of code-signing leaf certificates (`#windows`).
//! - **macOS:** Apple Developer Team Identifier (`#macos` / Developer ID section).
//! - **Linux:** The page documents **PGP** keys for repositories, not Authenticode-style
//!   binary thumbprints. This module does not pin Linux signatures.

use std::path::Path;

/// SHA-1 thumbprints (hex, uppercase) of Brave Windows code-signing leaf certificates.
/// Keep in sync with <https://brave.com/signing-keys/#windows> when certificates rotate.
#[cfg(windows)]
const WINDOWS_ALLOWED_SIGNER_SHA1: &[&str] = &[
    "8903F2BD47465A4F0F080AA7CEEC31A31B74DE42",
    "F8AC5F11DE7E26383B7A389FC19A2613835799D7",
];

/// Apple Developer Team ID for Brave's Developer ID Application signing identity.
/// Keep in sync with <https://brave.com/signing-keys/#macos> if/when details change.
#[cfg(target_os = "macos")]
const MACOS_EXPECTED_TEAM_ID: &str = "KL8N8XSYF4";

#[cfg(windows)]
pub fn verify_release_binary(path: &Path) -> Result<(), String> {
    verify_windows(path)
}

#[cfg(target_os = "macos")]
pub fn verify_release_binary(path: &Path) -> Result<(), String> {
    verify_macos(path)
}

#[cfg(all(unix, not(target_os = "macos")))]
pub fn verify_release_binary(_path: &Path) -> Result<(), String> {
    // Linux: signing-keys page lists GPG keys for repos, not ELF code-signing thumbprints here.
    Ok(())
}

#[cfg(not(any(windows, unix)))]
pub fn verify_release_binary(_path: &Path) -> Result<(), String> {
    Ok(())
}

// ── Windows: PowerShell Get-AuthenticodeSignature ───────────────────

#[cfg(windows)]
fn verify_windows(path: &Path) -> Result<(), String> {
    use std::process::Command;

    let path = path
        .canonicalize()
        .map_err(|e| format!("cannot canonicalize path for signature check: {e}"))?;

    // Path is passed via env var to avoid shell escaping issues.
    // -LiteralPath treats $env:BX_VERIFY_PATH as a literal path (no wildcard expansion).
    let output = Command::new("powershell.exe")
        .env("BX_VERIFY_PATH", &path)
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$s = Get-AuthenticodeSignature -LiteralPath $env:BX_VERIFY_PATH; \
             $s.Status; \
             if ($s.SignerCertificate) { $s.SignerCertificate.Thumbprint }",
        ])
        .output()
        .map_err(|e| format!("powershell: failed to verify signature: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let status = lines.next().unwrap_or("").trim();
    let thumbprint = lines.next().unwrap_or("").trim();

    if status != "Valid" {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = if !stderr.trim().is_empty() {
            format!(" ({})", stderr.trim())
        } else {
            String::new()
        };
        return Err(format!(
            "Windows Authenticode verification failed (status: {status}){detail}"
        ));
    }

    if thumbprint.is_empty() {
        return Err(
            "could not read signing certificate thumbprint from Authenticode signature".into(),
        );
    }

    let ok = WINDOWS_ALLOWED_SIGNER_SHA1
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(thumbprint));
    if !ok {
        return Err(format!(
            "signing certificate SHA1 thumbprint is not in the Brave allow-list (got {thumbprint}); \
             update `bx` or see https://brave.com/signing-keys/#windows"
        ));
    }

    Ok(())
}

// ── macOS: codesign ─────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn verify_macos(path: &Path) -> Result<(), String> {
    use std::process::Command;

    let path = path
        .canonicalize()
        .map_err(|e| format!("cannot canonicalize path for signature check: {e}"))?;

    let verify = Command::new("codesign")
        .arg("--verify")
        .arg("--strict")
        .arg("--verbose=0")
        .arg(&path)
        .output()
        .map_err(|e| format!("codesign --verify: failed to run: {e}"))?;

    if !verify.status.success() {
        let stderr = String::from_utf8_lossy(&verify.stderr);
        let stdout = String::from_utf8_lossy(&verify.stdout);
        let detail = if !stderr.trim().is_empty() {
            stderr.trim().to_string()
        } else {
            stdout.trim().to_string()
        };
        return Err(format!(
            "macOS code signature verification failed: {detail}"
        ));
    }

    let display = Command::new("codesign")
        .arg("-dv")
        .arg("--verbose=4")
        .arg(&path)
        .output()
        .map_err(|e| format!("codesign -dv: failed to run: {e}"))?;

    if !display.status.success() {
        let stderr = String::from_utf8_lossy(&display.stderr);
        return Err(format!("codesign -dv failed: {}", stderr.trim()));
    }

    // codesign prints -dv details to stderr (sometimes stdout); search both.
    let stderr = String::from_utf8_lossy(&display.stderr);
    let stdout = String::from_utf8_lossy(&display.stdout);
    let team_id = parse_codesign_team_identifier(&stderr)
        .or_else(|| parse_codesign_team_identifier(&stdout))
        .ok_or_else(|| {
            format!(
                "could not find TeamIdentifier in codesign output; expected {MACOS_EXPECTED_TEAM_ID} \
                 (https://brave.com/signing-keys/)"
            )
        })?;

    if team_id != MACOS_EXPECTED_TEAM_ID {
        return Err(format!(
            "TeamIdentifier {team_id} does not match Brave expected team ID {MACOS_EXPECTED_TEAM_ID} \
             (https://brave.com/signing-keys/)"
        ));
    }

    Ok(())
}

// ── Shared parsing ──────────────────────────────────────────────────

#[cfg(any(target_os = "macos", test))]
fn parse_codesign_team_identifier(codesign_output: &str) -> Option<&str> {
    for line in codesign_output.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("TeamIdentifier=") {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #[test]
    fn parse_team_identifier_finds_value() {
        let sample = "Identifier=org.example.app\nTeamIdentifier=KL8N8XSYF4\n";
        assert_eq!(
            super::parse_codesign_team_identifier(sample),
            Some("KL8N8XSYF4")
        );
    }

    #[test]
    fn parse_team_identifier_missing() {
        assert_eq!(
            super::parse_codesign_team_identifier("Identifier=foo\nFormat=Mach-O\n"),
            None
        );
    }

    #[test]
    fn parse_team_identifier_empty_value() {
        assert_eq!(
            super::parse_codesign_team_identifier("TeamIdentifier=\n"),
            None
        );
    }

    #[test]
    fn parse_team_identifier_whitespace() {
        assert_eq!(
            super::parse_codesign_team_identifier("  TeamIdentifier=  KL8N8XSYF4  \n"),
            Some("KL8N8XSYF4")
        );
    }

    #[test]
    fn parse_team_identifier_empty_input() {
        assert_eq!(super::parse_codesign_team_identifier(""), None);
    }

    #[cfg(windows)]
    #[test]
    fn windows_allowed_thumbprints_are_nonempty_hex() {
        for t in super::WINDOWS_ALLOWED_SIGNER_SHA1 {
            assert_eq!(t.len(), 40);
            assert!(t.bytes().all(|b| b.is_ascii_hexdigit()));
        }
    }
}
