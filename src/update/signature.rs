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

/// Apple Developer Team ID for Brave’s Developer ID Application signing identity.
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

#[cfg(windows)]
use windows_sys::Win32::Security::Cryptography::{
    CERT_CONTEXT, CERT_SHA1_HASH_PROP_ID, CertGetCertificateContextProperty,
};

#[cfg(windows)]
fn verify_windows(path: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::Security::WinTrust::{
        WINTRUST_ACTION_GENERIC_VERIFY_V2, WINTRUST_DATA, WINTRUST_DATA_0, WINTRUST_FILE_INFO,
        WTD_CHOICE_FILE, WTD_REVOKE_NONE, WTD_STATEACTION_CLOSE, WTD_STATEACTION_VERIFY,
        WTD_UI_NONE, WinVerifyTrust,
    };

    let path = path
        .canonicalize()
        .map_err(|e| format!("cannot canonicalize path for signature check: {e}"))?;

    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let mut file_info = WINTRUST_FILE_INFO {
        cbStruct: std::mem::size_of::<WINTRUST_FILE_INFO>() as u32,
        pcwszFilePath: wide.as_ptr(),
        hFile: INVALID_HANDLE_VALUE,
        pgKnownSubject: std::ptr::null_mut(),
    };

    let mut action_id = WINTRUST_ACTION_GENERIC_VERIFY_V2;

    let mut data = WINTRUST_DATA {
        cbStruct: std::mem::size_of::<WINTRUST_DATA>() as u32,
        pPolicyCallbackData: std::ptr::null_mut(),
        pSIPClientData: std::ptr::null_mut(),
        dwUIChoice: WTD_UI_NONE,
        fdwRevocationChecks: WTD_REVOKE_NONE,
        dwUnionChoice: WTD_CHOICE_FILE,
        Anonymous: WINTRUST_DATA_0 {
            pFile: &mut file_info,
        },
        dwStateAction: WTD_STATEACTION_VERIFY,
        hWVTStateData: std::ptr::null_mut(),
        pwszURLReference: std::ptr::null_mut(),
        dwProvFlags: 0,
        dwUIContext: 0,
        pSignatureSettings: std::ptr::null_mut(),
    };

    // SAFETY: WinTrust FFI; `wide` and `file_info` remain valid for both calls.
    let status = unsafe {
        WinVerifyTrust(
            std::ptr::null_mut(),
            &mut action_id,
            &mut data as *mut WINTRUST_DATA as *mut core::ffi::c_void,
        )
    };

    let pin_result = if status == 0 {
        verify_windows_signer_pin(data.hWVTStateData)
    } else {
        Ok(())
    };

    data.dwStateAction = WTD_STATEACTION_CLOSE;
    unsafe {
        WinVerifyTrust(
            std::ptr::null_mut(),
            &mut action_id,
            &mut data as *mut WINTRUST_DATA as *mut core::ffi::c_void,
        );
    }

    if status != 0 {
        return Err(format!(
            "Windows Authenticode verification failed (WinVerifyTrust returned {status})"
        ));
    }

    pin_result
}

#[cfg(windows)]
fn verify_windows_signer_pin(h_state: *mut core::ffi::c_void) -> Result<(), String> {
    let Some(ctx) = leaf_cert_context_from_trust_state(h_state) else {
        return Err(
            "could not read signing certificate from WinTrust state after successful verification"
                .into(),
        );
    };

    let thumb = cert_sha1_thumbprint_hex(ctx).ok_or_else(|| {
        "could not read SHA1 thumbprint for signing certificate (pin check failed)".to_string()
    })?;

    let ok = WINDOWS_ALLOWED_SIGNER_SHA1
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(&thumb));
    if !ok {
        return Err(format!(
            "signing certificate SHA1 thumbprint is not in the Brave allow-list (got {thumb}); \
             update `bx` or see https://brave.com/signing-keys/#windows"
        ));
    }

    Ok(())
}

#[cfg(windows)]
fn leaf_cert_context_from_trust_state(
    h_state: *mut core::ffi::c_void,
) -> Option<*const CERT_CONTEXT> {
    use windows_sys::Win32::Foundation::FALSE;
    use windows_sys::Win32::Security::WinTrust::{
        WTHelperGetProvCertFromChain, WTHelperGetProvSignerFromChain, WTHelperProvDataFromStateData,
    };

    if h_state.is_null() {
        return None;
    }

    // SAFETY: `h_state` comes from WinVerifyTrust after WTD_STATEACTION_VERIFY.
    unsafe {
        let prov = WTHelperProvDataFromStateData(h_state);
        if prov.is_null() {
            return None;
        }

        let sgnr = WTHelperGetProvSignerFromChain(prov, 0, FALSE, 0);
        if sgnr.is_null() {
            return None;
        }

        let prov_cert = WTHelperGetProvCertFromChain(sgnr, 0);
        if prov_cert.is_null() {
            return None;
        }

        let c = (*prov_cert).pCert;
        if c.is_null() {
            return None;
        }
        Some(c)
    }
}

#[cfg(windows)]
fn cert_sha1_thumbprint_hex(ctx: *const CERT_CONTEXT) -> Option<String> {
    let mut cb: u32 = 0;
    unsafe {
        CertGetCertificateContextProperty(
            ctx,
            CERT_SHA1_HASH_PROP_ID,
            std::ptr::null_mut(),
            &mut cb,
        );
    }
    if cb == 0 || cb > 256 {
        return None;
    }
    let mut buf = vec![0u8; cb as usize];
    let ok = unsafe {
        CertGetCertificateContextProperty(
            ctx,
            CERT_SHA1_HASH_PROP_ID,
            buf.as_mut_ptr().cast(),
            &mut cb,
        )
    };
    if ok == 0 {
        return None;
    }
    buf.truncate(cb as usize);
    Some(
        buf.iter()
            .map(|b| format!("{b:02X}"))
            .collect::<Vec<_>>()
            .concat(),
    )
}

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

    let info = String::from_utf8_lossy(&display.stderr);
    let team_id = parse_codesign_team_identifier(&info).ok_or_else(|| {
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

#[cfg(target_os = "macos")]
fn parse_codesign_team_identifier(codesign_stderr: &str) -> Option<String> {
    for line in codesign_stderr.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("TeamIdentifier=") {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    #[test]
    fn parse_team_identifier_finds_value() {
        let sample = "Identifier=org.example.app\nTeamIdentifier=KL8N8XSYF4\n";
        assert_eq!(
            super::parse_codesign_team_identifier(sample).as_deref(),
            Some("KL8N8XSYF4")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_allowed_thumbprints_are_nonempty_hex() {
        for t in super::WINDOWS_ALLOWED_SIGNER_SHA1 {
            assert_eq!(t.len(), 40);
            assert!(t.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }
}
