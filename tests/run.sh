#!/usr/bin/env bash
set -uo pipefail

# ── Helpers ───────────────────────────────────────────────────────────

passed=0 failed=0
green=$'\033[32m' red=$'\033[31m' reset=$'\033[0m'

pass() { ((passed++)); printf '%s\n' "${green}PASS${reset}  $1"; }
fail() {
  ((failed++))
  printf '%s\n' "${red}FAIL${reset}  $1"
  echo "$2" | sed "s/^/      /" >&2
}

summary() {
  echo
  if [ "$failed" -eq 0 ]; then
    printf '%s\n' "${green}$passed passed${reset}, 0 failed"
  else
    printf '%s\n' "$passed passed, ${red}$failed failed${reset}"
  fi
  [ "$failed" -eq 0 ]
}

# Run command, capture stdout/stderr/exit code.
run() {
  out=$("$@" 2>"$tmp/err") && rc=0 || rc=$?
  err=$(cat "$tmp/err")
}

# Assert: exit 0, stderr empty, valid JSON, optional jq expression.
check() {
  local name=$1
  if [ $rc -ne 0 ]; then fail "$name" "exit $rc — $err"; return 1; fi
  if [ -n "$err" ]; then fail "$name" "unexpected stderr: $err"; return 1; fi
  if ! echo "$out" | jq empty >/dev/null 2>&1; then fail "$name" "invalid JSON output"; return 1; fi
  if [ -n "${2:-}" ] && ! echo "$out" | jq -e "$2" >/dev/null 2>&1; then
    fail "$name" "assertion failed: $2"; return 1
  fi
  pass "$name"
}

# ── Setup ─────────────────────────────────────────────────────────────

if ! command -v jq >/dev/null; then
  echo "error: jq is required but not found in PATH" >&2; exit 1
fi

# Build from source
cargo build --release --quiet

# Resolve binary
if [ -n "${BX:-}" ]; then :
elif [ -x ./target/release/bx ]; then BX=./target/release/bx
elif command -v bx >/dev/null; then BX=bx
else echo "error: bx binary not found. Set \$BX or run: cargo build --release" >&2; exit 1
fi

# Resolve API key
if [ -z "${BRAVE_SEARCH_API_KEY:-}" ] && [ -n "${BRAVE_API_KEY:-}" ]; then
  BRAVE_SEARCH_API_KEY="$BRAVE_API_KEY"
fi
if [ -z "${BRAVE_SEARCH_API_KEY:-}" ]; then
  for f in "${HOME}/.config/brave-search/api_key" \
           "${HOME}/Library/Application Support/brave-search/api_key" \
           "${APPDATA:-}/brave-search/api_key"; do
    if [ -f "$f" ]; then BRAVE_SEARCH_API_KEY=$(cat "$f"); break; fi
  done
fi
if [ -z "${BRAVE_SEARCH_API_KEY:-}" ]; then
  echo "error: no API key found. Set BRAVE_SEARCH_API_KEY or run: bx config set-key" >&2; exit 1
fi
export BRAVE_SEARCH_API_KEY

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "binary: $BX ($($BX --version 2>/dev/null || echo unknown))"
echo

# ── CLI basics ────────────────────────────────────────────────────────

run $BX --help
if [ $rc -ne 0 ]; then fail "cli: --help" "exit $rc — $err"
elif ! echo "$out" | grep -q "COMMAND"; then fail "cli: --help" "output missing COMMAND"
else pass "cli: --help"; fi

run $BX --version
if [ $rc -ne 0 ]; then fail "cli: --version" "exit $rc — $err"
elif ! echo "$out" | grep -qE '^bx [0-9]+\.[0-9]+'; then fail "cli: --version" "unexpected: $out"
else pass "cli: --version"; fi

run $BX config show-key
if [ $rc -ne 0 ]; then fail "cli: config show-key" "exit $rc — $err"
elif ! echo "$out" | grep -q '\.\.\.'; then fail "cli: config show-key" "expected masked key, got: $out"
else pass "cli: config show-key"; fi

run $BX config path
if [ $rc -ne 0 ]; then fail "cli: config path" "exit $rc — $err"
elif [ -z "$out" ]; then fail "cli: config path" "empty output"
else pass "cli: config path"; fi

run $BX config show
if [ $rc -ne 0 ]; then fail "cli: config show" "exit $rc — $err"
else pass "cli: config show"; fi

# ── Default subcommand (context) ─────────────────────────────────────

run $BX "tokio spawn async task" --count 3
check "default: implicit context" '.grounding.generic[0] | has("url", "snippets")'

# -- inserts context before --, making the query a positional arg
# NOTE: flags CANNOT come after -- (they'd become positionals), so no --count here
run $BX -- "tokio spawn async task"
check "default: -- passes query through" '.grounding.generic[0] | has("url", "snippets")'

# Searching for a word that matches a subcommand name
run $BX -- web
check "default: -- disambiguates subcommand name" '.grounding.generic[0] | has("url", "snippets")'

# Explicit subcommand also works for subcommand-name queries
run $BX context web
check "default: explicit context with subcommand-name query" '.grounding.generic[0] | has("url", "snippets")'

# ── Web search ────────────────────────────────────────────────────────

run $BX web "rust programming" --count 3
check "web: basic" '.web.results[0] | has("url", "title")'

run $BX web "coffee shops" --count 3 \
  --lat 47.6062 --long -122.3321 --city Seattle --state WA \
  --state-name Washington --loc-country US \
  --timezone America/Los_Angeles --postal-code 98101
check "web: all location headers" '.web.results[0] | has("url", "title")'

run $BX web "rust axum" --count 3 --goggles '$boost=3,site=docs.rs'
check "web: goggles" '.web.results | length >= 1'

# Repeatable --goggles: two flags joined with newlines
run $BX web "rust axum" --count 3 --goggles '$boost=3,site=docs.rs' --goggles '$discard,site=w3schools.com'
check "web: repeatable goggles" '.web.results | length >= 1'

# Inline \n unescaping: two rules in one value
run $BX web "rust axum" --count 3 --goggles '$boost=3,site=docs.rs\n$discard,site=w3schools.com'
check "web: goggles inline newline" '.web.results | length >= 1'

# File-based goggles via @file
echo '$boost=3,site=docs.rs' > "$tmp/test.goggle"
run $BX web "rust axum" --count 3 --goggles "@$tmp/test.goggle"
check "web: goggles @file" '.web.results | length >= 1'

run $BX web "rust axum" --count 3 --include-site docs.rs
check "web: include-site" '.web.results | length >= 1'

run $BX web "rust programming" --count 3 --exclude-site w3schools.com
check "web: exclude-site" '.web.results | length >= 1'

run $BX web "café Zürich" --count 3 --city Zürich --loc-country CH
check "web: UTF-8 query + headers" '.web.results | length >= 1'

run $BX web "restaurants" --count 3 --result-filter locations \
  --lat 47.6062 --long -122.3321 --city Seattle
check "web: result-filter locations"

# ── Context ───────────────────────────────────────────────────────────

run $BX context "tokio spawn async task" --count 3
check "context: basic" '.grounding.generic[0] | has("url", "snippets")'

run $BX context "best restaurants" --count 3 \
  --lat 40.7128 --long -74.0060 --city "New York" --state NY --loc-country US
check "context: location headers" '.grounding.generic | length >= 1'

run $BX context "axum middleware" --count 3 --goggles '$boost=3,site=docs.rs'
check "context: goggles" '.grounding.generic | length >= 1'

run $BX context "axum middleware" --count 3 --include-site docs.rs
check "context: include-site" '.grounding.generic | length >= 1'

# ── News ──────────────────────────────────────────────────────────────

run $BX news "artificial intelligence" --count 3
check "news: basic" '.results[0] | has("url", "title")'

run $BX news "technology" --count 3 --freshness pw
check "news: freshness" '.results | length >= 1'

run $BX news "technology" --count 3 --exclude-site medium.com
check "news: exclude-site" '.results | length >= 1'

# ── Images ────────────────────────────────────────────────────────────

run $BX images "sunset mountains" --count 3
check "images: basic" '.results[0] | has("url", "title")'

# ── Videos ────────────────────────────────────────────────────────────

run $BX videos "rust tutorial" --count 3
check "videos: basic" '.results[0] | has("url", "title")'

run $BX videos "news today" --count 3 --freshness pw
check "videos: freshness" '.results | length >= 1'

# ── Places ────────────────────────────────────────────────────────────

run $BX places --location "San Francisco, CA" --count 3
check "places: basic" '.results[0] | has("title")'

# ── Suggest ───────────────────────────────────────────────────────────

run $BX suggest "how to lear"
check "suggest: basic" '.results[0] | has("query")'

# ── Spellcheck ────────────────────────────────────────────────────────

run $BX spellcheck "helo wrold"
if [ $rc -ne 0 ]; then fail "spellcheck: basic" "exit $rc — $err"
elif [ "$(echo "$out" | jq -r '.results[0].query')" != "hello world" ]; then
  fail "spellcheck: basic" "expected 'hello world', got: $(echo "$out" | jq -r '.results[0].query')"
else pass "spellcheck: basic"; fi

# ── Answers ───────────────────────────────────────────────────────────

run $BX answers "what is the capital of France" --no-stream
if [ $rc -ne 0 ]; then fail "answers: non-streaming" "exit $rc — $err"
elif [ "$(echo "$out" | jq -r '.choices[0].finish_reason')" != "stop" ]; then
  fail "answers: non-streaming" "expected finish_reason=stop, got: $(echo "$out" | jq -r '.choices[0].finish_reason')"
elif ! echo "$out" | jq -e '.choices[0].message.content | length > 0' >/dev/null 2>&1; then
  fail "answers: non-streaming" "empty content"
else pass "answers: non-streaming"; fi

run $BX answers "explain what HTTP status code 418 means"
if [ $rc -ne 0 ]; then fail "answers: streaming" "exit $rc — $err"
else
  line_count=$(echo "$out" | wc -l)
  if [ "$line_count" -lt 2 ]; then
    fail "answers: streaming" "expected multiple lines, got $line_count"
  elif ! echo "$out" | jq empty >/dev/null 2>&1; then
    fail "answers: streaming" "output contains invalid JSON lines"
  else pass "answers: streaming"; fi
fi

# ── POIs (chained) ────────────────────────────────────────────────────

run $BX web "restaurants" --count 3 --result-filter locations \
  --lat 47.6062 --long -122.3321 --city Seattle
poi_id=$(echo "$out" | jq -r '.locations.results[0].id // empty')
if [ -z "$poi_id" ]; then
  fail "pois: get location ID" "no POI ID in locations response"
else
  pass "pois: get location ID"

  run $BX pois "$poi_id" --lat 47.6062 --long -122.3321
  check "pois: details" '.results[0] | has("title")'

  run $BX descriptions "$poi_id"
  check "pois: descriptions" '.results | length >= 1'
fi

# ── Error handling ────────────────────────────────────────────────────

out=$($BX --api-key invalid_key web "test" --count 1 2>"$tmp/err") && rc=0 || rc=$?
err=$(cat "$tmp/err")
if [ $rc -ne 1 ]; then fail "errors: invalid API key" "expected exit 1, got $rc — $err"
elif ! echo "$err" | grep -q "error:"; then fail "errors: invalid API key" "stderr missing 'error:': $err"
else pass "errors: invalid API key"; fi

out=$($BX --base-url https://evil.example.com web "test" 2>"$tmp/err") && rc=0 || rc=$?
err=$(cat "$tmp/err")
if [ $rc -ne 1 ]; then fail "errors: invalid base URL" "expected exit 1, got $rc"
elif ! echo "$err" | grep -q "allowlist"; then fail "errors: invalid base URL" "stderr missing 'allowlist': $err"
else pass "errors: invalid base URL"; fi

out=$($BX web "test" --include-site docs.rs --goggles '$discard' 2>"$tmp/err") && rc=0 || rc=$?
if [ $rc -ne 2 ]; then fail "errors: include-site + goggles conflict" "expected exit 2, got $rc"
else pass "errors: include-site + goggles conflict"; fi

out=$($BX web "test" --include-site docs.rs --exclude-site x.com 2>"$tmp/err") && rc=0 || rc=$?
if [ $rc -ne 2 ]; then fail "errors: include-site + exclude-site conflict" "expected exit 2, got $rc"
else pass "errors: include-site + exclude-site conflict"; fi

out=$($BX web "test" --include-site 'bad domain!' 2>"$tmp/err") && rc=0 || rc=$?
if [ $rc -ne 2 ]; then fail "errors: invalid domain" "expected exit 2, got $rc"
else pass "errors: invalid domain"; fi

# ── Validation passthrough ───────────────────────────────────────────

# count=21 was previously rejected client-side (max was 20 for web);
# now the CLI passes it through and the API decides.
run $BX web "test" --count 21
check "validation: count beyond old limit" '.web.results | length >= 1'

# count=0 should be rejected by the API (not the CLI)
out=$($BX web "test" --count 0 2>"$tmp/err") && rc=0 || rc=$?
err=$(cat "$tmp/err")
if [ $rc -ne 1 ]; then fail "validation: count=0 API rejection" "expected exit 1, got $rc"
else pass "validation: count=0 API rejection"; fi

# ── Summary ───────────────────────────────────────────────────────────

summary
