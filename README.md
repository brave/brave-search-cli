# bx — Brave Search CLI

A zero-dependency, token-efficient CLI for the [Brave Search API](https://brave.com/search/api/), built for AI agents and LLMs.

One binary, JSON in/out, no runtime dependencies. `bx context` replaces search + scrape + extract in a single call with token-budgeted output — purpose-built for RAG and LLM grounding.

## Quick Start

**macOS/Linux** (macOS 11.7.1 or later)
```bash
curl -fsSL https://raw.githubusercontent.com/brave/brave-search-cli/main/scripts/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/brave/brave-search-cli/main/scripts/install.ps1 | iex"
```

**npm** (requires Node ≥ 16)
```bash
npm install -g @brave/brave-search-cli
```

```bash
bx config set-key YOUR_API_KEY    # get a key at https://api-dashboard.search.brave.com
bx "your search query"            # shorthand for: bx context "your search query"
bx --help                         # see all commands; bx <command> --help for flags
# To search for a word matching a subcommand name: bx -- web  or  bx context "web"
```

## For AI Agents

**Use `context` by default.** It returns pre-extracted, relevance-scored web content ready for LLM prompt injection. One API call replaces the search → scrape → extract pipeline.

```bash
# RAG grounding with token budget
bx context "Python TypeError cannot unpack non-iterable NoneType" --max-tokens 4096

# Direct AI answer (OpenAI-compatible, streams by default)
bx answers "explain Rust lifetimes with examples"

# Raw web search when you need site: scoping or result filtering
bx web "site:docs.rs axum middleware" --count 5
```

### When to Use Which Command

| Your need | Command | Why |
|-----------|---------|-----|
| Look up docs, errors, code patterns | `context` | Pre-extracted text, token-budgeted |
| Get a synthesized explanation | `answers` | AI-generated, cites sources |
| Search a specific site (site:) | `web` | Supports search operators |
| Find discussions/forums | `web --result-filter discussions` | Forums often have solutions |
| Check latest versions/releases | `context` or `news --freshness pd` | Fresh info beyond training data |
| Research security vulnerabilities | `context` or `news` | CVE details, advisories |
| Boost/filter specific domains | `--goggles` on context/web/news | Custom re-ranking, no other API has this |

### Response Shapes

**`bx context`** — RAG/grounding (recommended)
```json
{
  "grounding": {
    "generic": [
      { "url": "...", "title": "...", "snippets": ["extracted content...", "..."] }
    ]
  }
}
```

**`bx answers`** — AI answer (streaming, one JSON chunk per line)
```json
{"choices": [{"delta": {"content": "Rust lifetimes ensure references..."}}]}
```

**`bx answers --no-stream`** — AI answer (single response)
```json
{"choices": [{"message": {"content": "Rust lifetimes ensure references..."}}]}
```

**`bx web`** — Full search results
```json
{
  "web": { "results": [{"title": "...", "url": "...", "description": "..."}] },
  "news": { "results": [...] },
  "videos": { "results": [...] },
  "discussions": { "results": [...] }
}
```

### Agent Workflow Examples

**Debugging an error:**
```bash
# Agent encounters: "TypeError: cannot unpack non-iterable NoneType"
bx "Python TypeError cannot unpack non-iterable NoneType" --max-tokens 4096
```

**Evaluating a dependency:**
```bash
# Is this crate maintained? Any security issues?
bx context "reqwest crate security issues maintained 2026" --threshold strict
bx news "reqwest Rust crate" --freshness pm
```

**Corrective RAG loop:**
```bash
# 1. Broad search
bx "axum middleware authentication" --max-tokens 4096
# 2. Too general? Narrow down
bx "axum middleware tower layer authentication example" --threshold strict --max-tokens 4096
# 3. Still need synthesis? Ask for an answer
bx answers "how to implement JWT auth middleware in axum" --enable-research
```

**Checking for breaking changes before upgrading:**
```bash
bx context "Next.js 15 breaking changes migration guide" --max-tokens 8192
bx news "Next.js 15 release" --freshness pm
```

**Focused search with Goggles (custom re-ranking):**
```bash
# Boost official docs, prefer /docs/ paths, suppress SEO spam
bx "Python asyncio gather vs wait" \
  --goggles '$boost=3,site=docs.python.org
/docs/$boost=3
/blog/$downrank=2
$discard,site=geeksforgeeks.org
$discard,site=w3schools.com' --max-tokens 4096
```

**Token budget control:**
```bash
bx context "topic" --max-tokens 4096 --max-tokens-per-url 1024 --max-urls 5
```

## Goggles — Custom Search Re-Ranking

Brave Goggles let you define custom re-ranking rules for search results. Boost domains, URL paths, or content patterns; downrank noise; discard SEO spam — from simple domain allow/deny lists to complex multi-rule ranking profiles. No other search API offers this. Supported on `context`, `web`, and `news`.

### Domain Shortcuts — `--include-site` / `--exclude-site`

For the common case of restricting to or excluding specific domains, use the convenience flags (available on `context`, `web`, `news`):

```bash
# Only include results from these domains (allowlist)
bx "rust axum" --include-site docs.rs --include-site github.com

# Exclude specific domains
bx web "rust tutorial" --exclude-site w3schools.com --exclude-site medium.com
```

These generate Goggles rules internally. For more advanced re-ranking (boosting, path patterns, wildcards), use `--goggles` directly. The three flags are mutually exclusive.

### Why Agents Should Use Goggles

- **Domain allow/deny lists**: Boost, downrank, or discard entire domains — but that's just the start
- **URL path patterns**: Boost `/docs/` paths, downrank `/blog/` paths, discard `/ads/` — fine-grained URL targeting with wildcards
- **Full power of `site:`** — Brave converts `site:` operators to Goggles internally, but explicit Goggles unlock the full DSL: hundreds of rules, path patterns, content matching, boost/downrank strengths — without bloating the query string
- **Keep queries clean**: A single `--goggles` parameter (or URL) replaces long `site:X OR site:Y` chains in the query, saving tokens
- **Reusable & shareable**: Host a `.goggle` file on GitHub — share among team members, agents, CI pipelines. Document your ranking preferences as code
- **Community-maintained**: Leverage existing Goggles like [Tech Blogs](https://raw.githubusercontent.com/brave/goggles-quickstart/main/goggles/tech_blogs.goggle) that are continuously updated

### Inline Rules (zero setup)

```bash
# Domain boosting — prioritize official sources for a Python error
bx "TypeError cannot unpack non-iterable NoneType" \
  --goggles '$boost=3,site=docs.python.org
$boost=2,site=github.com
$discard,site=geeksforgeeks.org
$discard,site=w3schools.com'

# Path-based boosting — prefer /docs/ over /blog/ across all sites
bx context "axum middleware tower" \
  --goggles '/docs/$boost=5
/api/$boost=3
/blog/$downrank=3' --max-tokens 4096

# Ecosystem focus — boost Rust sources for crate research
bx context "serde custom deserializer" \
  --goggles '$boost=5,site=docs.rs
$boost=5,site=crates.io
$boost=3,site=github.com' --max-tokens 4096

# Downrank blog spam in news results
bx news "npm security advisory" --freshness pd \
  --goggles '$downrank=5,site=medium.com'
```

### DSL Quick Reference

| Rule | Effect | Example |
|------|--------|---------|
| `$boost=N,site=DOMAIN` | Promote domain (N=1-10) | `$boost=3,site=docs.rs` |
| `$downrank=N,site=DOMAIN` | Demote domain (N=1-10) | `$downrank=5,site=medium.com` |
| `$discard,site=DOMAIN` | Remove domain entirely | `$discard,site=w3schools.com` |
| `/path/$boost=N` | Boost matching URL paths | `/docs/$boost=5` |
| `*pattern*$boost=N` | Wildcard URL matching | `*api*$boost=3` |
| Generic `$discard` | Allowlist mode — discard all unmatched | `$discard` (as last rule) |

Separate multiple rules with newlines. Full DSL + pattern syntax: [goggles-quickstart](https://github.com/brave/goggles-quickstart).

### From a File (`@file`) — ideal for agents

Agents can generate a `.goggle` file on the fly and reference it:

```bash
# Agent writes rules to a file, then uses it across multiple queries
cat > /tmp/rust.goggle << 'EOF'
$boost=5,site=docs.rs
$boost=5,site=crates.io
$boost=3,site=github.com
/blog/$downrank=3
$discard,site=w3schools.com
$discard,site=geeksforgeeks.org
EOF

bx context "axum middleware tower" --goggles @/tmp/rust.goggle --max-tokens 4096
bx context "serde custom deserializer" --goggles @/tmp/rust.goggle --max-tokens 4096
```

### From stdin (`@-`) — pipe generated rules

```bash
# Pipe rules from another command
echo '$boost=5,site=docs.rs
$boost=3,site=github.com' | bx web "tokio runtime" --goggles @-
```

### Hosted Goggles (reusable, shareable)

Host a `.goggle` file on GitHub/GitLab, [submit it to Brave](https://search.brave.com/goggles/create), then reference by URL:

```bash
# Use the official Tech Blogs goggle
bx web "distributed systems" \
  --goggles 'https://raw.githubusercontent.com/brave/goggles-quickstart/main/goggles/tech_blogs.goggle'
```

Community Goggles: [brave/goggles-quickstart](https://github.com/brave/goggles-quickstart) | [Discover page](https://search.brave.com/goggles/discover)

## Getting an API Key

1. Sign up at <https://api-dashboard.search.brave.com/register>
2. Choose a plan — all plans include **$5/month free credits** (~1,000 free queries)
3. Different endpoints may require different plans (e.g. Search vs Answers)
4. Go to "API Keys" in the dashboard and generate a key

## Configuration

Settings are resolved in priority order: CLI flag > environment variable > config file > default.

| Setting | CLI Flag | Env Var | Config Key | Default |
|---------|----------|---------|------------|---------|
| API key | `--api-key` | `BRAVE_SEARCH_API_KEY` | `api_key` | *(interactive prompt)* |
| Base URL | `--base-url` | `BRAVE_SEARCH_BASE_URL` | `base_url` | `https://api.search.brave.com` |
| Timeout | `--timeout` | — | `timeout` | `30` (`300` for `answers --enable-research`; max `86400`) |

Streamed `answers` responses have no total deadline — every other command keeps one. The
timeout bounds *idle* time: connecting, and then each read. A response that keeps arriving is
never cut off however long it runs, and a server that only sends keep-alives never ends. Deep
research pauses for minutes while it synthesises, so it defaults to `300` seconds of silence;
`--timeout` or a `timeout` in your config file always wins, and applies to connecting too.

**Config file** (`bx config path` to see location):
- Linux: `~/.config/brave-search/config.json`
- macOS: `~/Library/Application Support/brave-search/config.json`
- Windows: `%APPDATA%\brave-search\config.json`

```json
{
  "api_key": "BSA...",
  "base_url": "https://api.search.brave.com"
}
```

Use `--config <path>` to load a different config file (e.g. one for dev, one for prod):

```bash
bx --config ~/.config/brave-search/dev.json web "test"
```

**Security tip:** Prefer the env var or config file over `--api-key`, which is visible in process listings. Use `bx config set-key` without an argument to enter the key interactively, avoiding shell history.

Saving a key writes a new file and renames it into place: mode `0600` on Linux and macOS, with the default directory `~/.config/brave-search` kept `0700` (a directory you pass via `--config` is left as it is). Because the file is *replaced* rather than written in place, a symlink, hard link, or bind mount at the config path does not survive a save, and neither do extended attributes or ACLs on the old file — deliberate, so a key is never written through an indirection `bx` does not control. On Windows no ACL is set and the file inherits whatever `%APPDATA%\brave-search` grants — restrict that directory yourself if the machine has other users.

### Local Proxy

You can point `bx` at a local reverse proxy instead of the Brave API directly. This is useful for sandboxed environments where credentials are managed outside the sandbox.

Allowed localhost forms:
- `http://127.0.0.1[:<port>][/path]`
- `http://[::1][:<port>][/path]`
- `http://localhost[:<port>][/path]` (resolved and rewritten to a literal loopback IP)

```bash
# Point bx at a local proxy on port 8080 with path prefix /brave
BRAVE_SEARCH_BASE_URL=http://127.0.0.1:8080/brave bx "your query"

# Or in config.json
# { "base_url": "http://127.0.0.1:8080/brave" }
```

The proxy receives requests at the configured base URL (e.g. `http://127.0.0.1:8080/brave/res/v1/web/search?q=...`) and can inject the `X-Subscription-Token` header before forwarding to `api.search.brave.com`. Since `bx` still requires an API key, set a placeholder: `BRAVE_SEARCH_API_KEY=unused`.

Only loopback addresses are allowed for `http://` URLs — non-loopback IPs and arbitrary hostnames are rejected. The `localhost` hostname is resolved at startup and rewritten to a literal IP to prevent DNS rebinding.

## Commands

| Command | Description | Output Shape |
|---------|-------------|--------------|
| `context` | **RAG/LLM grounding** — pre-extracted web content | `.grounding.generic[]` → `{url, title, snippets[]}` |
| `answers` | **AI answers** — OpenAI-compatible, streaming | `.choices[0].delta.content` (stream) |
| `web` | Full web search — all result types | `.web.results[]`, `.news.results[]`, etc. |
| `news` | News articles with freshness filters | `.results[]` → `{title, url, age}` |
| `images` | Image search (up to 200 results) | `.results[]` → `{title, url, thumbnail.src}` |
| `videos` | Video search with duration/views | `.results[]` → `{title, url, video.duration}` |
| `places` | Local place/POI search (200M+ POIs) | `.results[]` → `{title, postal_address, contact}` |
| `suggest` | Autocomplete/query suggestions | `.results[]` → `{query}` |
| `spellcheck` | Spell-check a query | `.results[0].query` |
| `pois` | POI details by ID | (use IDs from `places`) |
| `descriptions` | AI-generated POI descriptions | `.results[].description` |
| `council` | **Multi-model review** — several frontier models research one subject in parallel ([details](#council)) | `.quorum.delivered`, `.members[]` → `{model, status, action, file}` |
| `config` | Manage configuration | `set-key`, `show-key`, `show`, `path` |

## Usage Examples

```bash
# Default subcommand is context
bx "tokio spawn async task example"

# RAG grounding with token budget
bx context "Python TypeError cannot unpack non-iterable NoneType" --max-tokens 4096 --threshold strict

# AI answer (streaming)
bx answers "explain Rust lifetimes with examples"

# AI answer (non-streaming)
bx answers "compare SQLx and Diesel for Rust database access" --no-stream | jq .

# Research mode (multi-step, thorough)
bx answers "what changed in React 19 vs React 18?" --enable-research

# Web search with site: scoping
bx web "site:docs.rs axum middleware" | jq '.web.results[].url'

# News from the past day
bx news "npm security advisory" --freshness pd | jq '.results[].title'

# Image search
bx images "system architecture diagram microservices" | jq '.results[].thumbnail.src'

# Autocomplete
bx suggest "how to implement" --count 10 | jq '.results[].query'

# Local places
bx places --location "San Francisco CA US" -q "coffee" | jq '.results[].title'

# Location-aware web search
bx web "restaurants near me" --lat 37.7749 --long -122.4194 --city "San Francisco"

# Filter result types
bx web "rust" --result-filter "web,discussions"
```

## Extra Parameters and Custom Endpoints

### `--extra KEY=VALUE`

Pass additional API parameters not covered by named flags. Repeatable. Merged into the JSON body (POST endpoints) or query string (GET endpoints). Warns if the key collides with an existing flag.

For POST endpoints, values are auto-typed: integers and floats become JSON numbers, `true`/`false` become JSON booleans, everything else stays a string. GET values are always strings.

```bash
bx web "rust" --extra custom_param=value
bx web "rust" --extra count=5            # overrides --count (with warning)
bx images "sunset" --extra safesearch=off
```

### `--endpoint PATH`

Override the default API path. For internal or beta Brave endpoints. Must start with `/`.

```bash
bx web "test" --endpoint /res/v2/web/search
```

## Answers Endpoint

The `answers` command supports two modes:

### Simple Mode

Pass a question directly:

```bash
bx answers "explain Rust lifetimes with examples"         # streaming (default)
bx answers "compare SQLx and Diesel" --no-stream          # single JSON response
bx answers "what changed in React 19?" --enable-research  # deep research
bx answers "Hello" --model brave                          # faster model
```

`--enable-citations`, `--enable-entities`, and `--enable-research` all require streaming, so
none combines with `--no-stream`. Citations and entities are not supported in research mode,
but are fine together. Rejected combinations exit 2, nothing sent:

```bash
bx answers "what changed in React 19?" --enable-research --enable-citations
# error: the argument '--enable-research' cannot be used with '--enable-citations'
```

### Stdin Mode

Pass `-` to read a full JSON request body from stdin:

```bash
echo '{"messages":[{"role":"user","content":"review this code for security issues"}],"model":"brave-pro"}' | bx answers -
cat request.json | bx answers -
```

## Council

`bx council` sends one plan, diff or question to several frontier models at once. Each has
live search over Brave's independent index and none sees another's reply, so you get
independent, sourced critiques instead of one model's recall. Nothing is lost quietly: a member
that fails is named with its reason and whether a retry would help, every reply you paid for is
written to disk, and `manifest.json` tells an agent what to do next.

Use it on a change you are about to ship or run unattended, a plan worth attacking before it
is built, or a question where one model's answer is not evidence.

### Quick Start

```bash
bx council doctor                  # keys, panels, pricing and file paths; free, no network
bx council @design.md --dry-run    # the plan and what your past runs cost; spends nothing
bx council @design.md              # run it: a few dollars and up to 15 minutes by default
```

`OPENROUTER_API_KEY` reaches every model. The default panel also calls Anthropic directly, so
it needs `ANTHROPIC_API_KEY` unless you pass `--via openrouter`. Members search with your Brave
key (`bx config set-key`); `--tools none` runs without search and without that key.
`bx council set-key` saves inference keys to `council.json`.

A run blocks until its last member finishes, which is longer than most agent shells wait.
Name it, background it, and poll; `--out` takes a new or empty directory:

```bash
bx council @design.md --out review-1 >review-1.json 2>review-1.log &
bx council status review-1         # phase, done, total
git diff main | bx council - --task code-review   # review a change from stdin
```

### Writing the Subject

Pass a file (`@design.md`), stdin (`-`), or a quoted sentence; anything longer than a line
belongs in a file or on stdin (1 MiB limit). Each member is billed for the subject on every
round it takes, so its size drives the bill: send what the question needs, not the
repository. The models see only that text — search is their one tool, so they cannot open
your files or run your tests.

One scoped question per council, in this order: what this is and what it costs to be wrong;
the material; what is out of scope or already decided; then a numbered list of asks. Split a
large review across several councils rather than sending one giant paste.

### Reading the Results

The run directory is named on stderr as the run starts. Start at `index.md`, where a failed
member is struck through with its reason. Beside it:

| File | Holds |
|------|-------|
| `NN-<model>.md` | One member's reply; a rejected one too, headed `REJECTED` with the reason |
| `NN-<model>.md.partial` | Paid text that was never published, kept rather than discarded |
| `manifest.json` | The run as JSON, also printed on stdout |
| `sources.json` | Every URL each member was shown |

In the manifest, read `quorum.delivered`, `members[].action` (`retry` or `none`) and `hint`,
which gives the `-m` that re-runs only the members worth retrying. Exit 0 means at least
`--min-quorum` members delivered, and the default is 1, so check `quorum.delivered` or raise
`--min-quorum`.

Delivered means a reply passed the delivery gate, not that it is right. Treat each finding like
a stranger's bug report and reproduce it before you act; two members agreeing is a filter, not
proof. After fixing, run `--task second-pass` over the fixes and the first round's findings.
`bx council findings [run]` prints a run as JSON split on its headings; `bx council ls` lists
the 20 newest runs.

### Cost and Time

- `--dry-run` prints the plan and, once a panel has three past runs, `observed`: what those
  runs were charged for inference. Size `--max-usd` above its max. `worst_usd` beside it covers
  only members priced in advance; OpenRouter members' inference is not in it.
- `--max-usd` caps reported spend, Brave lookups included. It is checked between rounds, so
  each running member can overshoot by one round, and a member it stops keeps what it wrote.
  It refuses to start while a direct member has no rate; add one under `rates` in
  `council.json`. `bx council doctor` shows how each member is priced.
- `--max-seconds` is each member's clock: 900 by default, 2400 for `audit`. Members run in
  parallel, so it bounds the run too unless you lower `--concurrency`. Raise it when you retry
  a member it stopped.
- `-m @fast` is a cheaper five-model panel that needs only `OPENROUTER_API_KEY` for inference.
- `--max-searches` (default 15 per member) bounds Brave lookups, each a billed request.
- `--effort`: lower it when a member spends its output on hidden reasoning; the run names this
  flag when that happens.

### Tasks and `council.json`

`--task` picks the prompt by its exact name. Reviews, in lettered sections ending in a
verdict: `design-review` (default), `code-review`, `audit` (every line, on a longer clock) and
`second-pass` (grades your fixes). Answers built from sources: `decide`, `ask`, `fact-check`.

Any Markdown file in the tasks directory is a task named after the file, and one named like a
built-in replaces it. Its frontmatter can pin the panel, effort, clock and search budget, so a
task is a saved review rather than a prompt. `{doc}` marks where the subject goes; without it
the subject is appended. `bx council tasks` names the directory and every frontmatter key:

```markdown
---
description: rank detection signals by discrimination x durability
panel: opus,astra
max_searches: 25
max_seconds: 2400
---
You are a threat-intelligence analyst. Rank the signals in {doc} by ...
```

`council.json` holds your own panels (`-m @name`; one named `default` replaces `@default`),
aliases, `default_task`, `max_usd`, `rates` for direct members, `direct` (which vendors bypass
OpenRouter), and `providers`, which pins a model to named OpenRouter endpoints. `bx council
doctor` prints its path; `--config` does not move it.

```json
{
  "panels": { "mine": ["opus", "kimi", "z-ai/glm-5.3"] },
  "default_task": "code-review",
  "max_usd": 10
}
```

### Privacy and Retention

Your subject goes to every model provider on the panel. Members write search queries from it
and send them to Brave, and what Brave returns goes back to the provider that asked.
`--dry-run` states this egress before you spend. Send nothing those parties should not see.

Zero data retention is requested by default; `"zdr": false` in `council.json` stops requesting
it. What a transport can enforce differs, so `bx council doctor` prints it per transport and
each manifest records what applied under `members[].retention`. While it is on, an OpenRouter
model with no zero-retention endpoint is refused before the run starts.

Runs stay in the state directory (`doctor` prints it) until `bx council prune` removes those
older than 30 days that finished or were abandoned before any reply; `--dry-run` lists them
first, and a live run is skipped. `--out` directories and `history.jsonl` (one metadata line
per member, no subject text) are never pruned. `--trace` also keeps each member's whole conversation. On Linux and macOS files are
`0600`, in directories bx creates at `0700`; on Windows no ACL is set, so restrict the
directory yourself.

## Exit Codes

| Code | Meaning | Agent action |
|------|---------|-------------|
| 0 | Success | Process results |
| 1 | The request was made; the result is unusable | Fix query/parameters |
| 2 | Usage error — nothing was sent | Fix the CLI arguments, config, or input |
| 3 | Auth/permission error (401/403) | Check API key or plan: `bx config show-key`; `council`: `bx council doctor` |
| 4 | Rate limited (429) | Retry after delay |
| 5 | Server/network error | Retry with backoff |
| 6 | `council`: quorum missed for any other reason, a network outage included | Read `members[].action`; retry those marked `retry` |
| 7 | `council`: quorum missed, and every failure was `--max-usd` | Raise the cap (`cost.max_usd`, `hint`) and re-run |
| 8 | `council`: out of credit (402) | Top up the provider account |
| 127 | npm launcher could not start bx | Reinstall, or check platform support |

`council` exits 0 once `--min-quorum` members deliver (default 1), whatever happened to the
rest, so a run the cap cut short can still exit 0. When quorum is missed it exits 3, 4, 5 or 8
if every failed member met the same kind of provider error, 7 if every failure was the cap,
and 6 otherwise.
Its free pre-flight checks can exit 3 (key rejected) or 8 (balance exhausted) before anything
is sent.

## Error Handling

Errors are printed to stderr with a human-readable summary, recovery hints, and the full JSON error body:

```
error: rate limited (429) — Request rate limit exceeded for plan.
hint: retry after a short delay, or upgrade plan for higher rate limits
{"type":"ErrorResponse","error":{"code":"RATE_LIMITED","status":429,...}}
```

Exit codes are differentiated — see [Exit Codes](#exit-codes) above. The line between
1 and 2 is whether a request was made: 2 means bx could not act on what it was given and
never contacted the API, so retrying without changing the command will not help.

## Building from Source

```bash
# Debug build
cargo build

# Release build (~2 MB)
cargo build --release
```

### Static binaries via Docker

Requires Docker, GNU Make, and a POSIX shell (`sh`/`bash`). Produces fully static, cross-compiled binaries.

On Windows, run these targets from Git Bash or WSL. Running `make` from native `cmd`/PowerShell can fail because the `Makefile` uses POSIX utilities (`grep`, `sed`, shell redirection).

```bash
make linux-amd64     # → dist/bx-<version>-linux-amd64
make linux-arm64     # → dist/bx-<version>-linux-arm64
make darwin-amd64    # → dist/bx-<version>-darwin-amd64
make darwin-arm64    # → dist/bx-<version>-darwin-arm64
make windows-amd64   # → dist/bx-<version>-windows-amd64.exe
make windows-arm64   # → dist/bx-<version>-windows-arm64.exe
make dist-all        # → all of the above
```

Both macOS binaries **require macOS 11.7.1 or later** — zig sets that deployment
target and `MACOSX_DEPLOYMENT_TARGET` does not override it. The `smoke-darwin`
job asserts it, so a toolchain bump that raises the floor fails CI.

## Testing

```bash
# End-to-end API smoke suite (requires bash + jq + BRAVE_SEARCH_API_KEY/BRAVE_API_KEY)
make test

# Unit tests (no network/API key required)
make test-unit
```
