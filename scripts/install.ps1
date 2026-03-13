#!/usr/bin/env pwsh
# install.ps1 - bx (Brave Search CLI) installer for Windows
#
# Downloads a pre-built Windows binary from GitHub Releases,
# verifies SHA256 checksum, and installs to a user directory.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/brave/brave-search-cli/main/scripts/install.ps1 | iex"
#
# Or with a specific version/install directory:
#   .\install.ps1 -Version v1.0.0 -InstallDir "$env:USERPROFILE\.local\bin"

[CmdletBinding()]
param(
    [string]$Version = $env:VERSION,
    [string]$InstallDir = $env:BX_INSTALL_DIR
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
# PS 5.1 defaults to SSL 3.0 + TLS 1.0; GitHub requires TLS 1.2.
# Only affects HttpWebRequest (PS 5.1); PS 7+ uses HttpClient with OS defaults.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Write-Info {
    param([string]$Message)
    Write-Host "  $Message"
}

function Fail {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Messages)
    foreach ($m in $Messages) {
        [Console]::Error.WriteLine("error: $m")
    }
    exit 1
}

function Download-File {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile
    )
    Invoke-WebRequest -Uri $Url -OutFile $OutFile -Headers @{ "User-Agent" = "bx-installer" }
}

$Repo = "brave/brave-search-cli"
$Bin = "bx.exe"
$Releases = "https://github.com/$Repo/releases"

if (-not $InstallDir) {
    $InstallDir = Join-Path $env:USERPROFILE ".local\bin"
}

# --- platform detection ---
if (-not [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
    Fail "unsupported OS: $([System.Runtime.InteropServices.RuntimeInformation]::OSDescription)" "this installer only supports Windows"
}

$osArch = $null
try {
    $osArch = [string][System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
} catch {
    $osArch = $null
}

# Some PowerShell hosts can report RuntimeInformation inconsistently.
# Fall back to common Windows architecture environment variables.
if (-not $osArch) {
    if ($env:PROCESSOR_ARCHITEW6432) {
        $osArch = $env:PROCESSOR_ARCHITEW6432
    } elseif ($env:PROCESSOR_ARCHITECTURE) {
        $osArch = $env:PROCESSOR_ARCHITECTURE
    }
}

$archProbe = if ($osArch) { $osArch.ToUpperInvariant() } else { "" }
switch -Regex ($archProbe) {
    "^(X64|AMD64)$" { $arch = "amd64"; break }
    "^ARM64$" {
        Fail "Windows arm64 binaries are not currently published" "supported platform for this script is windows-amd64"
    }
    default {
        $reportedArch = if ($osArch) { $osArch } else { "(unknown)" }
        Fail "unsupported architecture: $reportedArch" "only amd64 is currently supported on Windows"
    }
}

$platform = "windows-$arch"

# --- version resolution ---
if (-not $Version) {
    Write-Info "fetching latest version..."
    try {
        $latestResponse = Invoke-WebRequest -Uri "$Releases/latest" -Headers @{ "User-Agent" = "bx-installer" }
    } catch {
        Fail "failed to determine latest version" "set -Version vX.Y.Z manually, or check $Releases"
    }

    # PS 5.1: BaseResponse is HttpWebResponse (has ResponseUri).
    # PS 7+: BaseResponse is HttpResponseMessage (has RequestMessage.RequestUri).
    if ($latestResponse.BaseResponse.ResponseUri) {
        $latestUri = [string]$latestResponse.BaseResponse.ResponseUri.AbsoluteUri
    } else {
        $latestUri = [string]$latestResponse.BaseResponse.RequestMessage.RequestUri.AbsoluteUri
    }
    if ($latestUri -match "/tag/(?<tag>v[^/?#]+)$") {
        $Version = $Matches["tag"]
    } else {
        Fail "failed to determine latest version" "set -Version vX.Y.Z manually, or check $Releases"
    }
}

$Version = [string]$Version
$Version = $Version.Trim()

if (-not $Version) {
    Fail "version cannot be empty" "set -Version vX.Y.Z"
}

if ($Version -match "^[vV](?<num>.+)$") {
    $versionNum = $Matches["num"]
} else {
    $versionNum = $Version
}

if (-not ($versionNum -match "^\d+\.\d+\.\d+")) {
    Fail "invalid version format: $Version" "expected vX.Y.Z (e.g., v1.0.0)"
}

$Version = "v$versionNum"
$binaryName = "bx-$versionNum-$platform.exe"
$releaseUrl = "$Releases/download/$Version"

Write-Info "installing bx $Version ($platform)"

$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("bx-install-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmpDir | Out-Null

try {
    # --- download ---
    $binaryPath = Join-Path $tmpDir $binaryName

    Write-Info "downloading $binaryName..."
    try {
        Download-File -Url "$releaseUrl/$binaryName" -OutFile $binaryPath
    } catch {
        Fail "failed to download $binaryName" "check that $Version exists at $Releases"
    }

    # --- checksum verification (mandatory) ---
    Write-Info "verifying checksum..."
    $checksumFile = "$binaryName.sha256"
    $checksumPath = Join-Path $tmpDir $checksumFile
    try {
        Download-File -Url "$releaseUrl/$checksumFile" -OutFile $checksumPath
    } catch {
        Fail "failed to download $checksumFile" "cannot verify binary integrity"
    }

    $checksumContent = Get-Content -Path $checksumPath -Raw
    if (-not $checksumContent) { Fail "invalid checksum in $checksumFile" }
    $expected = ($checksumContent.Trim() -split '\s')[0].ToLowerInvariant()
    if (-not ($expected -match '^[A-Fa-f0-9]{64}$')) {
        Fail "invalid checksum in $checksumFile"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -Path $binaryPath).Hash.ToLowerInvariant()

    if ($expected -ne $actual) {
        Fail "checksum verification failed!" "expected: $expected" "got:      $actual" "the downloaded binary may be corrupted or tampered with"
    }

    # --- install ---
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $destination = Join-Path $InstallDir $Bin
    Copy-Item -Path $binaryPath -Destination $destination -Force

    # Verify binary executes.
    & $destination --version *> $null
    if ($LASTEXITCODE -ne 0) {
        Fail "installed binary failed to execute" "this may indicate a platform mismatch or a corrupted download"
    }

    # GitHub Actions support.
    if ($env:GITHUB_ACTIONS -and $env:GITHUB_PATH) {
        Add-Content -Path $env:GITHUB_PATH -Value $InstallDir
        Write-Info "added $InstallDir to `$GITHUB_PATH"
    }

    # PATH guidance only; do not auto-modify user profile.
    $normalizedInstall = $InstallDir.Trim().Trim('"').TrimEnd("\")
    $pathEntries = ($env:PATH -split ";") | ForEach-Object { $_.Trim().Trim('"').TrimEnd("\") } | Where-Object { $_ }
    if (-not ($pathEntries -contains $normalizedInstall)) {
        Write-Info ""
        Write-Info "add bx to your PATH (if not already there):"
        Write-Info "  `$env:Path = `"$InstallDir;`$env:Path`""
        Write-Info "  # optional (persist for future sessions):"
        Write-Info "  [Environment]::SetEnvironmentVariable('Path', `"$InstallDir;`$env:Path`", 'User')"
    }

    Write-Info ""
    Write-Info "bx $Version installed to $destination"
    Write-Info ""
    Write-Info "next steps:"
    Write-Info "  bx config set-key YOUR_API_KEY    # set your Brave Search API key"
    Write-Info '  bx "your search query"             # search (= bx context "...")'
}
finally {
    Remove-Item -Path $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
}
