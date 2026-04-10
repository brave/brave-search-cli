#!/bin/sh
# install.sh — bx (Brave Search CLI) installer
# https://github.com/brave/brave-search-cli
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/brave/brave-search-cli/main/scripts/install.sh | sh
#   VERSION=v1.5.0 curl -fsSL .../scripts/install.sh | sh
#
# Env: VERSION (default: latest), BX_INSTALL_DIR (default: ~/.local/bin)
# Requires: curl or wget, sha256sum or shasum
# Platforms: linux-amd64, linux-arm64, darwin-arm64
#
# Security: HTTPS + SHA256 verification. No sudo, no eval.
# Does not protect against compromised releases (would need signing).

set -eu

main() {
    REPO="brave/brave-search-cli"
    BIN="bx"
    RELEASES="https://github.com/${REPO}/releases"

    install_dir="${BX_INSTALL_DIR:-${HOME}/.local/bin}"

    error() { printf "error: %s\n" "$@" >&2; exit 1; }
    info()  { printf "  %s\n" "$@"; }
    available() { command -v "$1" >/dev/null 2>&1; }

    available curl || available wget || error "curl or wget is required"
    available sha256sum || available shasum || error "sha256sum or shasum is required"

    # curl: --proto '=https' blocks HTTP downgrades; --tlsv1.2 sets TLS floor.
    # wget: --secure-protocol=TLSv1_2 sets TLS floor. Do NOT use --https-only
    #   (only affects recursive mode, no-op for direct downloads). wget has no
    #   equivalent to --proto '=https' (cannot block HTTPS→HTTP redirects);
    #   SHA256 verification below covers this gap.
    # Prefers curl (stronger TLS controls). No cross-fallback by design.
    download() {
        if available curl; then
            curl --proto '=https' --tlsv1.2 -fsSL -o "$2" "$1"
        elif available wget; then
            wget --secure-protocol=TLSv1_2 -q -O "$2" "$1"
        fi
    }

    case "$(uname -s)" in
        Linux)  os="linux" ;;
        Darwin) os="darwin" ;;
        *)      error "unsupported OS: $(uname -s)" \
                      "only Linux and macOS are supported" ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64)  arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *)             error "unsupported architecture: $(uname -m)" \
                             "only x86_64 and arm64 are supported" ;;
    esac

    # No darwin-amd64 build — only darwin-arm64.
    if [ "$os" = "darwin" ] && [ "$arch" = "amd64" ]; then
        error "macOS x86_64 (Intel) binaries are not available" \
              "build from source instead: cargo build --release"
    fi

    platform="${os}-${arch}"

    # Extract version from /releases/latest 302 Location header (avoids API rate limit).
    if [ -z "${VERSION:-}" ]; then
        info "fetching latest version..."
        if available curl; then
            VERSION=$(curl --proto '=https' --tlsv1.2 -fsSI \
                "${RELEASES}/latest" 2>/dev/null \
                | grep -i '^location:' | sed 's|.*/tag/||;s/[[:space:]]*$//')
        elif available wget; then
            VERSION=$(wget --spider -S "${RELEASES}/latest" 2>&1 \
                | grep -i 'location:' | tail -1 | sed 's|.*/tag/||;s/[[:space:]]*$//')
        fi
        [ -n "${VERSION:-}" ] || error "failed to determine latest version" \
            "set VERSION=vX.Y.Z manually, or check ${RELEASES}"
    fi

    case "$VERSION" in
        v[0-9]*.[0-9]*.[0-9]*) ;;
        *) error "invalid version format: ${VERSION}" \
                 "expected format: vX.Y.Z (e.g., v1.0.0)" ;;
    esac

    version_num="${VERSION#v}"
    binary_name="${BIN}-${version_num}-${platform}"
    release_url="${RELEASES}/download/${VERSION}"

    info "installing ${BIN} ${VERSION} (${platform})"

    tmp_dir=$(mktemp -d) || error "failed to create temporary directory"
    # EXIT alone doesn't fire on signals in dash/ash — add HUP/INT/TERM.
    trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM

    info "downloading ${binary_name}..."
    download "${release_url}/${binary_name}" "${tmp_dir}/${binary_name}" ||
        error "failed to download ${binary_name}" \
              "check that ${VERSION} exists at ${RELEASES}"

    info "verifying checksum..."
    checksum_file="${binary_name}.sha256"
    download "${release_url}/${checksum_file}" "${tmp_dir}/${checksum_file}" ||
        error "failed to download ${checksum_file}" \
              "cannot verify binary integrity"

    expected=$(cut -d' ' -f1 < "${tmp_dir}/${checksum_file}")
    printf '%s' "$expected" | grep -qx '[0-9a-fA-F]\{64\}' ||
        error "invalid checksum in ${checksum_file}"

    if available sha256sum; then
        actual=$(sha256sum "${tmp_dir}/${binary_name}" | cut -d' ' -f1)
    else
        actual=$(shasum -a 256 "${tmp_dir}/${binary_name}" | cut -d' ' -f1)
    fi

    if [ "$expected" != "$actual" ]; then
        error "checksum verification failed!" \
              "expected: ${expected}" \
              "got:      ${actual}" \
              "the downloaded binary may be corrupted or tampered with"
    fi

    mkdir -p "$install_dir"
    # Atomic install with permissions (no cp+chmod race).
    install -m 755 "${tmp_dir}/${binary_name}" "${install_dir}/${BIN}"

    if ! "${install_dir}/${BIN}" --version >/dev/null 2>&1; then
        error "installed binary failed to execute" \
              "this may indicate a platform mismatch or a corrupted download"
    fi

    # GitHub Actions: add to $GITHUB_PATH so subsequent steps find bx.
    if [ -n "${GITHUB_ACTIONS:-}" ] && [ -n "${GITHUB_PATH:-}" ]; then
        echo "$install_dir" >> "$GITHUB_PATH"
        info "added ${install_dir} to \$GITHUB_PATH"
    fi

    # Don't modify shell configs — print PATH instructions instead.
    case ":${PATH}:" in
        *":${install_dir}:"*) ;;
        *)
            info ""
            info "add bx to your PATH (if not already there):"
            info "  export PATH=\"${install_dir}:\$PATH\""
            ;;
    esac

    info ""
    info "bx ${VERSION} installed to ${install_dir}/${BIN}"
    info ""
    info "next steps:"
    info "  bx config set-key YOUR_API_KEY    # set your Brave Search API key"
    info "  bx \"your search query\"             # search (= bx context \"...\")"
}

# Invoke main only after the entire script is received (curl|sh safety).
main "$@"
