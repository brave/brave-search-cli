mod api;
mod config;
mod council;
mod query;

use std::borrow::Cow;
use std::net::{IpAddr, ToSocketAddrs};
use std::path::Path;

use clap::{Args, Parser, Subcommand};

/// Search the Web via Brave's independent index. The only search API with its own Web index at
/// scale. Truly independent, lightning-fast, and built to power AI apps. Your queries never
/// leave Brave, except under `council`: it sends your subject, and the searches its models run
/// and the results they read, to those models' providers.
///
/// One CLI for the whole API, shaped for agents: JSON to stdout, errors to stderr, an exit code
/// per failure kind.
/// API key: https://api-dashboard.search.brave.com
///
/// WHICH COMMAND?
///   Looking up docs, APIs, errors, code patterns? → context (default, recommended)
///   Need a synthesized answer with citations? → answers
///   Need raw search results or result filtering? → web
///   Too costly to get wrong on one model's word? → council (paid: a few dollars, ~15 min)
///   Other: news, images, videos, places, suggest, spellcheck
///   Restrict to / exclude specific domains? → --include-site / --exclude-site on context/web/news
///   Custom ranking (boost docs, discard spam)? → --goggles on context/web/news
///
/// Shorthand: bx "query" = bx context "query"
/// To search for a subcommand name: bx -- web  or  bx context "web"
///
/// Quick start:
///   bx config set-key <YOUR_KEY>
///   bx "tokio spawn async task example" # RAG grounding (= bx context)
///   bx answers "how does Rust's borrow checker work?" # AI answer
///   bx web "site:docs.rs reqwest" | jq . # web search
///   bx council @design.md --dry-run # multi-model review: the plan and its cost, free
#[derive(Parser)]
#[command(name = "bx", version, verbatim_doc_comment)]
struct Cli {
    /// Path to the Brave config file (config.json)
    ///
    /// Council's own council.json is not redirected by this.
    #[arg(long, global = true, display_order = 900)]
    config: Option<std::path::PathBuf>,

    /// Brave Search API key
    ///
    /// Never an inference key; council reads those from the environment or council.json. Prefer
    /// the env var — flags show in process listings.
    #[arg(
        long,
        env = "BRAVE_SEARCH_API_KEY",
        global = true,
        display_order = 901,
        hide_env_values = true
    )]
    api_key: Option<String>,

    /// Base URL for Brave Search [default: https://api.search.brave.com]
    #[arg(
        long,
        env = "BRAVE_SEARCH_BASE_URL",
        global = true,
        display_order = 902,
        hide_env_values = true
    )]
    base_url: Option<String>,

    /// Seconds per HTTP request [default: 30; 300 with answers --enable-research; max: 86400]
    ///
    /// On council this bounds one LLM Context lookup; --max-seconds bounds a member.
    #[arg(long, global = true, display_order = 903)]
    timeout: Option<u64>,

    /// Extra Brave API parameters (KEY=VALUE, repeatable)
    ///
    /// Merged into the request body (POST) or query string (GET). Warns on collision with
    /// existing flags. POST values are auto-typed: integers, floats, true/false → JSON natives.
    /// Council refuses it rather than ignoring it.
    #[arg(long, global = true, value_name = "KEY=VALUE", display_order = 904)]
    extra: Vec<String>,

    /// Override the Brave API path (e.g. /res/v1/web/search)
    ///
    /// For internal/beta endpoints. Council refuses it rather than ignoring it.
    #[arg(long, global = true, display_order = 905)]
    endpoint: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// RAG/LLM grounding — pre-extracted web content, clean text, token-budgeted
    ///
    /// THE recommended endpoint for AI agents. Returns relevance-scored, pre-extracted
    /// web content for LLM context injection. Clean text, structured data, code snippets
    /// — ready to feed into prompts. One API call replaces search + scrape + extract.
    ///
    /// Common agent use-cases: error debugging, API/library docs, code patterns,
    /// version lookups, best practices, security advisories.
    ///
    /// Token budget: --max-tokens (total), --max-tokens-per-url (per source).
    /// Relevance: --threshold (strict|balanced|lenient).
    ///
    /// Output: .grounding.generic[] — array of {url, title, snippets[]}
    ///
    /// Examples:
    ///   bx context "Python TypeError cannot unpack non-iterable NoneType" --max-tokens 4096
    ///   bx context "tokio vs async-std Rust async runtime" --count 5 --threshold strict
    ///   bx context "how to implement retry with exponential backoff" --max-tokens 2048
    ///   bx context "axum middleware" --include-site docs.rs --include-site github.com --max-tokens 4096
    ///   bx context "axum middleware" --goggles '$boost=3,site=docs.rs' --max-tokens 4096
    #[command(verbatim_doc_comment)]
    Context(ContextArgs),

    /// AI-grounded answers — OpenAI-compatible, streaming, SOTA SimpleQA F1=94.1%
    ///
    /// Web-grounded AI answers. Streams by default (SSE, one JSON chunk per line).
    /// Use --no-stream for a single JSON response (not with the --enable-* flags).
    ///
    /// Use when you need a synthesized answer rather than raw source material.
    /// Best for: explanations, comparisons, "how does X work" questions.
    ///
    /// Simple mode: pass a question as a positional argument.
    /// Stdin mode:  pass "-" to read a full JSON request body from stdin.
    ///
    /// Models: brave-pro (default, higher quality), brave (faster).
    ///
    /// Output (stream): one JSON chunk per line, content in .choices[0].delta.content
    /// Output (no-stream): single JSON, content in .choices[0].message.content
    ///
    /// Examples:
    ///   bx answers "explain Rust lifetimes with examples"
    ///   bx answers "compare SQLx and Diesel for Rust database access" --no-stream | jq .
    ///   bx answers "what changed in React 19 vs React 18?" --enable-research
    ///   echo '{"messages":[{"role":"user","content":"review this code for security issues"}]}' | bx answers -
    #[command(verbatim_doc_comment)]
    Answers(AnswersArgs),

    /// Full web search — pages, discussions, FAQ, infobox, news, videos
    ///
    /// The kitchen-sink endpoint. Returns all result types in one call.
    /// Supports freshness filters, goggles, search operators, location awareness.
    ///
    /// Use when you need: structured result types, site:-scoped search, pagination,
    /// discussion forums, FAQ snippets, or result type filtering.
    ///
    /// Output: .web.results[], .news.results[], .videos.results[], .discussions.results[]
    ///
    /// Examples:
    ///   bx web "site:docs.rs axum middleware" | jq '.web.results[].url'
    ///   bx web "ECONNREFUSED PostgreSQL connection refused fix" --count 5
    ///   bx web "site:github.com rust-lang/rust issue borrow checker" --operators
    ///   bx web "rust error handling" --include-site docs.rs --count 5
    ///   bx web "rust error handling" --goggles '$boost=3,site=docs.rs' --count 5
    ///
    /// Result types (filter with --result-filter):
    ///   web, discussions, faq, infobox, news, videos, query, summarizer, locations
    #[command(verbatim_doc_comment)]
    Web(WebArgs),

    /// News search — articles with freshness filters (pd/pw/pm/py or date range)
    ///
    /// Output: .results[] — array of {title, url, description, age, thumbnail}
    ///
    /// Examples:
    ///   bx news "Rust language" --freshness pw | jq '.results[].title'
    ///   bx news "npm security advisory" --freshness pd
    ///   bx news "OpenAI API" --freshness pm --count 10
    ///   bx news "security vulnerability" --exclude-site medium.com
    #[command(verbatim_doc_comment)]
    News(NewsArgs),

    /// Image search — thumbnails, sources, dimensions
    ///
    /// Returns up to 200 image results with thumbnails, source URLs, and metadata.
    ///
    /// Output: .results[] — array of {title, url, thumbnail.src, properties.width/height}
    ///
    /// Examples:
    ///   bx images "system architecture diagram microservices" | jq '.results[].thumbnail.src'
    ///   bx images "Rust ownership model diagram" --count 10
    #[command(verbatim_doc_comment)]
    Images(ImagesArgs),

    /// Video search — titles, URLs, thumbnails, durations
    ///
    /// Output: .results[] — array of {title, url, thumbnail.src, video.duration}
    ///
    /// Examples:
    ///   bx videos "Rust async await tutorial" --count 5 | jq '.results[].url'
    ///   bx videos "distributed systems design" --freshness pm
    #[command(verbatim_doc_comment)]
    Videos(VideosArgs),

    /// Local place/POI search across 200M+ POIs
    ///
    /// Search by query and/or location. Provide location via --location string
    /// or --latitude/--longitude coordinates. Returns names, addresses, ratings, hours.
    /// Query is optional — you can search by location alone.
    ///
    /// Output: .results[] — array of {title, postal_address, contact}
    ///
    /// Examples:
    ///   bx places "coffee" --location "San Francisco CA US"
    ///   bx places "pizza" --latitude 37.7749 --longitude -122.4194
    ///   bx places "museums" --location "NYC" | jq '.results[].title'
    ///   bx places --location "NYC" --count 5   # no query, location-only browse
    #[command(verbatim_doc_comment)]
    Places(PlacesArgs),

    /// Autocomplete/query suggestions
    ///
    /// Output: .results[] — array of {query}
    ///
    /// Examples:
    ///   bx suggest "how to implement" --count 10 | jq '.results[].query'
    ///   bx suggest "Rust error handling" --rich
    #[command(verbatim_doc_comment)]
    Suggest(SuggestArgs),

    /// Spell-check a query — returns corrected text
    ///
    /// Output: .results[0].query — the corrected query string
    ///
    /// Examples:
    ///   bx spellcheck "kubernetse dployment" | jq '.results[0].query'
    #[command(verbatim_doc_comment)]
    Spellcheck(SpellcheckArgs),

    /// POI details by ID — hours, ratings, reviews, contact info
    ///
    /// Get detailed info for specific places using IDs from the places command.
    ///
    /// Examples:
    ///   bx pois PLACE_ID_1 PLACE_ID_2 | jq .
    #[command(verbatim_doc_comment, hide = true)]
    Pois(PoisArgs),

    /// AI-generated POI descriptions by ID
    ///
    /// Examples:
    ///   bx descriptions PLACE_ID_1 PLACE_ID_2 | jq '.results[].description'
    #[command(verbatim_doc_comment, hide = true)]
    Descriptions(DescriptionsArgs),

    /// Several frontier models attack one question at once, each searching Brave
    ///
    /// Members from different labs research your subject in parallel, each live on Brave's
    /// independent index, so claims can be checked against pages instead of recalled. Replies
    /// land side by side in one run directory, with their sources and cost. Every failure is
    /// named, with its reason and whether a retry helps, and paid text is kept even when a cap
    /// cuts it off. Use it when being wrong is costly: a change about to ship, a plan before it
    /// is built, a claim one model cannot settle. A default panel costs a few dollars and takes
    /// ~15 minutes.
    ///
    /// START
    ///   bx council doctor                  keys, panels, prices; exit 2 names what is missing
    ///   bx council @design.md --dry-run    the plan and your past cost; spends nothing
    ///   bx council @design.md --out ./r1 >r1.json 2>r1.log &
    ///   bx council status ./r1             poll it; a run outlasts most agent shell timeouts
    ///   git diff main | bx council - --task code-review --max-usd 10
    /// No --task means a red-team review; for an answer, use --task ask or decide.
    ///
    /// KEYS     @default (Anthropic, OpenAI, Moonshot, Z.ai) needs OPENROUTER_API_KEY and
    ///          ANTHROPIC_API_KEY; with --via openrouter, only the first. @fast is five cheaper
    ///          labs, all through OpenRouter. Search needs BRAVE_SEARCH_API_KEY unless
    ///          --tools none. Store them: bx council set-key (models), bx config set-key (Brave).
    ///
    /// SUBJECT  One scoped question: what it is and what being wrong costs, the material, what
    ///          is out of scope or decided, then numbered asks. Short beats long: every member
    ///          pays for the whole subject every round. Split a big review across councils.
    ///
    /// RESULT   stdout is manifest.json; stderr names the run directory. Read index.md first,
    ///          then quorum.delivered, members[].action and hint: exit 0 only means --min-quorum
    ///          (default 1) delivered. A rejected member's file keeps what it wrote, and a
    ///          .md.partial is paid text never published. Re-buy lost seats with hint's -m,
    ///          raising --max-seconds if the clock stopped them. `bx council findings` returns a
    ///          run as JSON split by heading. After fixing, --task second-pass checks the fixes
    ///          against round one's findings; every round is billed in full.
    ///
    /// TRUST    `delivered` means a reply passed the gate, not that it is right. Reproduce a
    ///          finding before acting on it; members agreeing is a filter, not proof.
    ///
    /// CONFIG   council.json holds keys, panels, aliases, rates, default_task, max_usd and zdr.
    ///          Zero data retention is requested by default and means something different per
    ///          provider: `bx council doctor` prints what, and every path council uses.
    #[command(verbatim_doc_comment)]
    Council(council::CouncilArgs),

    /// Manage configuration — set-key, show-key, show, path
    ///
    /// Config file: ~/.config/brave-search/config.json (Linux),
    /// ~/Library/Application Support/brave-search/config.json (macOS),
    /// %APPDATA%\brave-search\config.json (Windows).
    ///
    /// Examples:
    ///   bx config set-key <KEY>
    ///   bx config show-key
    ///   bx config show
    ///   bx config path
    #[command(verbatim_doc_comment)]
    Config {
        #[command(subcommand)]
        cmd: ConfigCmd,
    },
}

#[derive(Subcommand)]
pub enum ConfigCmd {
    /// Save an API key to the config file (tip: omit key to enter interactively, avoiding shell history)
    SetKey {
        /// The API key to save (omit to enter interactively)
        key: Option<String>,
    },
    /// Show the configured API key (masked)
    ShowKey,
    /// Print the config file path
    Path,
    /// Show the full configuration (API key masked)
    Show,
}

// ── Subcommand args ──────────────────────────────────────────────────

#[derive(Args)]
struct GogglesArgs {
    /// Goggles: custom re-ranking rules — boost, downrank, or discard results.
    /// Target by domain (site=) or URL path pattern (/docs/$boost=3).
    /// Actions: $boost=N (1-10), $downrank=N (1-10), $discard. One rule per line.
    /// Repeatable: --goggles '$site=docs.rs' --goggles '$discard' (joined with newlines)
    /// Inline:    --goggles '$boost=3,site=docs.python.org'  (use \n for multiple rules)
    /// File:      --goggles @rules.goggle  (reads local file, ideal for agents)
    /// Stdin:     --goggles @-  (reads from stdin)
    /// Hosted:    --goggles 'https://raw.githubusercontent.com/.../my.goggle'
    /// Unique to Brave — no other search API offers custom re-ranking.
    /// Mutually exclusive with --include-site / --exclude-site.
    /// Ref: https://github.com/brave/goggles-quickstart
    #[arg(long, verbatim_doc_comment)]
    goggles: Vec<String>,

    /// Only include results from these domains (repeatable, exclusive with --goggles / --exclude-site)
    #[arg(long, conflicts_with_all = ["goggles", "exclude_site"],
          value_parser = validate_domain)]
    include_site: Vec<String>,

    /// Exclude results from these domains (repeatable, exclusive with --goggles / --include-site)
    #[arg(long, conflicts_with_all = ["goggles", "include_site"],
          value_parser = validate_domain)]
    exclude_site: Vec<String>,
}

#[derive(Parser)]
struct WebArgs {
    /// Search query
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    q: String,

    /// Country code (e.g. US, GB, DE)
    #[arg(long)]
    country: Option<String>,

    /// Search language (e.g. en, fr, de)
    #[arg(long)]
    search_lang: Option<String>,

    /// UI language (e.g. en-US, fr-FR)
    #[arg(long)]
    ui_lang: Option<String>,

    /// Number of results
    #[arg(long)]
    count: Option<u16>,

    /// Result offset for pagination
    #[arg(long)]
    offset: Option<u16>,

    /// Safe search: off, moderate, strict
    #[arg(long, value_parser = ["off", "moderate", "strict"])]
    safesearch: Option<String>,

    /// Freshness: pd (past day), pw (past week), pm (past month), py (past year), or YYYY-MM-DDtoYYYY-MM-DD
    #[arg(long)]
    freshness: Option<String>,

    /// Text decorations (bold markers in snippets) [omit: API default, --text-decorations: enable, --text-decorations false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    text_decorations: Option<bool>,

    /// Spellcheck [omit: API default, --spellcheck: enable, --spellcheck false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    spellcheck: Option<bool>,

    /// Comma-separated result types: discussions,faq,infobox,news,query,summarizer,videos,web,locations
    #[arg(long)]
    result_filter: Option<String>,

    #[command(flatten)]
    goggles_args: GogglesArgs,

    /// Extra snippets from different parts of the page [omit: API default, --extra-snippets: enable, --extra-snippets false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    extra_snippets: Option<bool>,

    /// Units: metric or imperial
    #[arg(long, value_parser = ["metric", "imperial"])]
    units: Option<String>,

    /// Enable search operators in the query (site:, intitle:, etc.)
    /// Full list: https://search.brave.com/help/operators
    /// [omit: API default, --operators: enable, --operators false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    operators: Option<bool>,

    /// Latitude for location-aware results
    #[arg(long, allow_hyphen_values = true)]
    lat: Option<String>,

    /// Longitude for location-aware results
    #[arg(long, requires = "lat", allow_hyphen_values = true)]
    long: Option<String>,

    /// Timezone (e.g. America/New_York)
    #[arg(long)]
    timezone: Option<String>,

    /// City for location-aware results
    #[arg(long)]
    city: Option<String>,

    /// State abbreviation (e.g. CA)
    #[arg(long)]
    state: Option<String>,

    /// State name (e.g. California)
    #[arg(long)]
    state_name: Option<String>,

    /// Country for location header
    #[arg(long)]
    loc_country: Option<String>,

    /// Postal code for location-aware results
    #[arg(long)]
    postal_code: Option<String>,
}

#[derive(Parser)]
struct ImagesArgs {
    /// Search query
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    q: String,

    /// Country code
    #[arg(long)]
    country: Option<String>,

    /// Search language
    #[arg(long)]
    search_lang: Option<String>,

    /// Number of results
    #[arg(long)]
    count: Option<u16>,

    /// Safe search: off or strict
    #[arg(long, value_parser = ["off", "strict"])]
    safesearch: Option<String>,

    /// Spellcheck [omit: API default, --spellcheck: enable, --spellcheck false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    spellcheck: Option<bool>,
}

#[derive(Parser)]
struct VideosArgs {
    /// Search query
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    q: String,

    /// Country code
    #[arg(long)]
    country: Option<String>,

    /// Search language
    #[arg(long)]
    search_lang: Option<String>,

    /// UI language
    #[arg(long)]
    ui_lang: Option<String>,

    /// Number of results
    #[arg(long)]
    count: Option<u16>,

    /// Result offset for pagination
    #[arg(long)]
    offset: Option<u16>,

    /// Safe search: off, moderate, strict
    #[arg(long, value_parser = ["off", "moderate", "strict"])]
    safesearch: Option<String>,

    /// Freshness filter
    #[arg(long)]
    freshness: Option<String>,

    /// Spellcheck [omit: API default, --spellcheck: enable, --spellcheck false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    spellcheck: Option<bool>,

    /// Enable search operators in the query (site:, intitle:, etc.)
    /// Full list: https://search.brave.com/help/operators
    /// [omit: API default, --operators: enable, --operators false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    operators: Option<bool>,
}

#[derive(Parser)]
struct NewsArgs {
    /// Search query
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    q: String,

    /// Country code
    #[arg(long)]
    country: Option<String>,

    /// Search language
    #[arg(long)]
    search_lang: Option<String>,

    /// UI language
    #[arg(long)]
    ui_lang: Option<String>,

    /// Number of results
    #[arg(long)]
    count: Option<u16>,

    /// Result offset for pagination
    #[arg(long)]
    offset: Option<u16>,

    /// Safe search: off, moderate, strict
    #[arg(long, value_parser = ["off", "moderate", "strict"])]
    safesearch: Option<String>,

    /// Freshness filter
    #[arg(long)]
    freshness: Option<String>,

    /// Spellcheck [omit: API default, --spellcheck: enable, --spellcheck false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    spellcheck: Option<bool>,

    /// Extra snippets [omit: API default, --extra-snippets: enable, --extra-snippets false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    extra_snippets: Option<bool>,

    #[command(flatten)]
    goggles_args: GogglesArgs,

    /// Enable search operators in the query (site:, intitle:, etc.)
    /// Full list: https://search.brave.com/help/operators
    /// [omit: API default, --operators: enable, --operators false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    operators: Option<bool>,
}

#[derive(Parser)]
struct SuggestArgs {
    /// Search query
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    q: String,

    /// Language
    #[arg(long)]
    lang: Option<String>,

    /// Country code
    #[arg(long)]
    country: Option<String>,

    /// Number of suggestions
    #[arg(long)]
    count: Option<u16>,

    /// Rich suggestions [omit: API default, --rich: enable, --rich false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    rich: Option<bool>,
}

#[derive(Parser)]
struct SpellcheckArgs {
    /// Query to spell-check
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    q: String,

    /// Language
    #[arg(long)]
    lang: Option<String>,

    /// Country code
    #[arg(long)]
    country: Option<String>,
}

#[derive(Parser)]
struct AnswersArgs {
    /// Question to ask, or "-" to read JSON body from stdin
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    query: String,

    /// Model: brave-pro or brave
    #[arg(long, value_parser = ["brave-pro", "brave"])]
    model: Option<String>,

    /// Disable streaming (default: stream enabled)
    #[arg(long)]
    no_stream: bool,

    /// Country code
    #[arg(long)]
    country: Option<String>,

    /// Language
    #[arg(long)]
    language: Option<String>,

    /// Safe search: off, moderate, strict
    #[arg(long, value_parser = ["off", "moderate", "strict"])]
    safesearch: Option<String>,

    /// Maximum completion tokens
    #[arg(long)]
    max_completion_tokens: Option<u32>,

    /// Inline citation tags (streaming only; not in research mode)
    #[arg(long, conflicts_with = "no_stream")]
    enable_citations: bool,

    /// Inline entity tags (streaming only; not in research mode)
    #[arg(long, conflicts_with = "no_stream")]
    enable_entities: bool,

    /// Iterative deep research instead of a single search (streaming only)
    #[arg(long, conflicts_with_all = ["enable_citations", "enable_entities", "no_stream"])]
    enable_research: bool,

    /// Allow thinking in research mode
    #[arg(long)]
    research_allow_thinking: Option<bool>,

    /// Max tokens per research query
    #[arg(long)]
    research_max_tokens_per_query: Option<u32>,

    /// Max research queries
    #[arg(long)]
    research_max_queries: Option<u32>,

    /// Max research iterations
    #[arg(long)]
    research_max_iterations: Option<u32>,

    /// Max research seconds
    #[arg(long)]
    research_max_seconds: Option<u32>,

    /// Max results per research query
    #[arg(long)]
    research_max_results_per_query: Option<u32>,

    /// Search context size: low, medium, high
    #[arg(long, value_parser = ["low", "medium", "high"])]
    search_context_size: Option<String>,

    /// Approximate user city
    #[arg(long)]
    user_city: Option<String>,

    /// Approximate user country
    #[arg(long)]
    user_country: Option<String>,

    /// Approximate user region
    #[arg(long)]
    user_region: Option<String>,

    /// Approximate user timezone
    #[arg(long)]
    user_timezone: Option<String>,
}

#[derive(Parser)]
struct ContextArgs {
    /// Search query
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    q: String,

    /// Country code
    #[arg(long)]
    country: Option<String>,

    /// Search language
    #[arg(long)]
    search_lang: Option<String>,

    /// Number of results
    #[arg(long)]
    count: Option<u16>,

    /// Max URLs to include
    #[arg(long, visible_alias = "max-urls")]
    maximum_number_of_urls: Option<u32>,

    /// Max total tokens
    #[arg(long, visible_alias = "max-tokens")]
    maximum_number_of_tokens: Option<u32>,

    /// Max snippets
    #[arg(long, visible_alias = "max-snippets")]
    maximum_number_of_snippets: Option<u32>,

    /// Max tokens per URL
    #[arg(long, visible_alias = "max-tokens-per-url")]
    maximum_number_of_tokens_per_url: Option<u32>,

    /// Max snippets per URL
    #[arg(long, visible_alias = "max-snippets-per-url")]
    maximum_number_of_snippets_per_url: Option<u32>,

    /// Threshold mode: strict, balanced, lenient
    #[arg(long, visible_alias = "threshold", value_parser = ["strict", "balanced", "lenient"])]
    context_threshold_mode: Option<String>,

    #[command(flatten)]
    goggles_args: GogglesArgs,

    /// Local results [omit: API default, --enable-local: enable, --enable-local false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    enable_local: Option<bool>,

    /// Latitude for location-aware results
    #[arg(long, allow_hyphen_values = true)]
    lat: Option<String>,

    /// Longitude for location-aware results
    #[arg(long, requires = "lat", allow_hyphen_values = true)]
    long: Option<String>,

    /// Timezone
    #[arg(long)]
    timezone: Option<String>,

    /// City
    #[arg(long)]
    city: Option<String>,

    /// State abbreviation
    #[arg(long)]
    state: Option<String>,

    /// State name
    #[arg(long)]
    state_name: Option<String>,

    /// Country for location header
    #[arg(long)]
    loc_country: Option<String>,

    /// Postal code
    #[arg(long)]
    postal_code: Option<String>,
}

#[derive(Parser)]
struct PlacesArgs {
    /// Search query (optional — omit to browse by location)
    q: Option<String>,

    /// Latitude
    #[arg(long, allow_hyphen_values = true)]
    latitude: Option<String>,

    /// Longitude
    #[arg(long, requires = "latitude", allow_hyphen_values = true)]
    longitude: Option<String>,

    /// Location string (alternative to lat/long, e.g. "San Francisco CA US")
    #[arg(long)]
    location: Option<String>,

    /// Search radius in meters
    #[arg(long)]
    radius: Option<String>,

    /// Number of results
    #[arg(long)]
    count: Option<u16>,

    /// Country code
    #[arg(long)]
    country: Option<String>,

    /// Search language
    #[arg(long)]
    search_lang: Option<String>,

    /// UI language
    #[arg(long)]
    ui_lang: Option<String>,

    /// Units: metric or imperial
    #[arg(long, value_parser = ["metric", "imperial"])]
    units: Option<String>,

    /// Safe search
    #[arg(long, value_parser = ["off", "moderate", "strict"])]
    safesearch: Option<String>,

    /// Spellcheck [omit: API default, --spellcheck: enable, --spellcheck false: disable]
    #[arg(long, num_args = 0..=1, default_missing_value = "true",
          value_parser = clap::builder::BoolishValueParser::new())]
    spellcheck: Option<bool>,
}

#[derive(Parser)]
struct PoisArgs {
    /// POI IDs
    ids: Vec<String>,

    /// Search language
    #[arg(long)]
    search_lang: Option<String>,

    /// UI language
    #[arg(long)]
    ui_lang: Option<String>,

    /// Units: metric or imperial
    #[arg(long, value_parser = ["metric", "imperial"])]
    units: Option<String>,

    /// Latitude for location context
    #[arg(long, allow_hyphen_values = true)]
    lat: Option<String>,

    /// Longitude for location context
    #[arg(long, requires = "lat", allow_hyphen_values = true)]
    long: Option<String>,
}

#[derive(Parser)]
struct DescriptionsArgs {
    /// POI IDs
    ids: Vec<String>,
}

// ── Main ─────────────────────────────────────────────────────────────

const SUBCOMMANDS: &[&str] = &[
    "context",
    "answers",
    "web",
    "news",
    "images",
    "videos",
    "places",
    "suggest",
    "spellcheck",
    "pois",
    "descriptions",
    "council",
    "config",
    "help",
];

/// Injects "context" as the default subcommand when the first positional
/// argument is not a known subcommand (e.g. `bx "query"` → `bx context "query"`).
/// Use `--` to force context for queries matching subcommand names: `bx -- web`.
fn inject_default_subcommand() -> Vec<String> {
    // Safety: args[0] is not used for security decisions — we skip it (i = 1) and only
    // inspect subsequent args for subcommand routing. CWE-807 does not apply here.
    let args: Vec<String> = std::env::args().collect(); // nosemgrep: rust.lang.security.args.args
    inject_default_subcommand_impl(args)
}

/// Global flags that consume the next argument; shared with the parse-error path so the two argv
/// walks cannot drift.
const VALUE_FLAGS: &[&str] = &[
    "--api-key",
    "--base-url",
    "--timeout",
    "--config",
    "--extra",
    "--endpoint",
];

fn inject_default_subcommand_impl(mut args: Vec<String>) -> Vec<String> {
    let mut i = 1; // skip binary name
    while i < args.len() {
        if args[i] == "--" {
            // No subcommand before -- ; inject "context" so clap sees it as
            // `bx context -- <query>`, allowing disambiguation of subcommand names.
            args.insert(i, "context".to_string());
            return args;
        }

        if args[i].starts_with('-') {
            // Check if this flag consumes the next arg
            if VALUE_FLAGS.contains(&args[i].as_str()) {
                i += 2; // skip flag and its value
                continue;
            }
            // --flag=value or boolean flag: just skip
            i += 1;
            continue;
        }

        // First positional argument found
        if !SUBCOMMANDS.contains(&args[i].as_str()) {
            args.insert(i, "context".to_string());
            return args;
        }

        return args; // known subcommand, no injection
    }

    args // no positional found (e.g. `bx --help`)
}

/// Every argv token clap will weigh, in order: the binary name dropped, each value flag's value
/// skipped, and nothing past `--`, which makes the rest positional. One walk for both readers
/// below.
fn walk(args: &[String]) -> impl Iterator<Item = &str> {
    let mut value = false;
    args.iter()
        .skip(1)
        .map(String::as_str)
        .take_while(|a| *a != "--")
        .filter(move |a| {
            let was = value;
            // `!was &&`, or a value that happens to spell a flag is mistaken for one.
            value = !was && VALUE_FLAGS.contains(a);
            !was
        })
}

/// The sub-command clap will route to, read from the argv `inject_default_subcommand` produced.
/// Needed only on the parse-error path, where there is no parsed `Cli` left to ask.
fn subcommand_in(args: &[String]) -> Option<&str> {
    walk(args).find(|a| !a.starts_with('-'))
}

/// The `--format` asked for, read from raw argv because the parse failed. Absent or unreadable
/// means JSON, council's default.
fn format_in(args: &[String]) -> &str {
    let mut fmt = "json";
    let mut it = walk(args);
    while let Some(a) = it.next() {
        if let Some(v) = a.strip_prefix("--format=") {
            fmt = v;
        } else if a == "--format"
            && let Some(v) = it.next()
        {
            fmt = v;
        }
    }
    fmt
}

/// clap's error without its usage block, as the one-line `message` the JSON body carries.
fn one_line(rendered: &str) -> String {
    rendered
        .lines()
        .take_while(|l| !l.starts_with("Usage:") && !l.starts_with("For more information"))
        .map(str::trim)
        .filter(|l| !l.is_empty())
        // `council::refuse` adds its own prefix; clap's would arrive twice.
        .map(|l| l.strip_prefix("error: ").unwrap_or(l))
        .collect::<Vec<_>>()
        .join("; ")
}

/// clap's own refusals, answered like council's others: JSON unless `--format text`. Other
/// commands keep clap's prose. Help and version are not failures and keep exit 0.
fn parse_or_refuse(argv: &[String]) -> Cli {
    match Cli::try_parse_from(argv) {
        Ok(c) => c,
        Err(e) if e.use_stderr() && subcommand_in(argv) == Some("council") => {
            council::refuse(Some(format_in(argv)), &one_line(&e.render().to_string()))
        }
        Err(e) => e.exit(),
    }
}

/// Parses --extra KEY=VALUE pairs. Returns borrowed slices into the input.
fn parse_extra(extras: &[String]) -> Vec<(&str, &str)> {
    extras
        .iter()
        .map(|entry| match entry.split_once('=') {
            Some((k, v)) if !k.is_empty() => (k, v),
            _ => {
                eprintln!("error: --extra requires KEY=VALUE format, got: {entry}");
                std::process::exit(2);
            }
        })
        .collect()
}

/// Merges --extra pairs into a JSON body, exiting on error.
fn merge_extras(body: &mut serde_json::Value, extras: &[(&str, &str)]) {
    if let Err(msg) = query::merge_extra_into_json(body, extras) {
        eprintln!("error: {msg}");
        std::process::exit(2);
    }
}

/// Converts a bool to a static string for GET query parameters.
fn bool_str(v: bool) -> &'static str {
    if v { "true" } else { "false" }
}

const DEFAULT_BASE_URL: &str = "https://api.search.brave.com";
const DEFAULT_TIMEOUT: u64 = 30;
/// `--enable-research` searches server-side and can stream nothing for minutes.
const DEFAULT_RESEARCH_TIMEOUT: u64 = 300;
/// A day: nothing legitimate comes close, and a huge value overflows ureq's `Instant` maths.
const MAX_TIMEOUT: u64 = 86_400;
/// `DEFAULT_TIMEOUT` is range-checked at runtime on every call; the research default is the one
/// value `answers_timeouts` can return without passing through that check.
const _: () = assert!(DEFAULT_RESEARCH_TIMEOUT >= 1 && DEFAULT_RESEARCH_TIMEOUT <= MAX_TIMEOUT);

fn main() {
    let cli = parse_or_refuse(&inject_default_subcommand());
    let cfg_path = cli.config.as_deref();

    // Config subcommand doesn't need an API key.
    if let Command::Config { ref cmd } = cli.command {
        config::handle_config(cmd, cfg_path);
        return;
    }

    // The global flags are validated here, so council's refusals for them are built here too.
    let council_format = match &cli.command {
        Command::Council(a) => Some(a.format().to_string()),
        _ => None,
    };
    let refuse = |msg: &str| -> ! { council::refuse(council_format.as_deref(), msg) };

    let config = match config::load_config(cfg_path) {
        Ok(c) => c,
        Err(e) => refuse(&e),
    };

    let base_raw: Cow<'static, str> = cli
        .base_url
        .or(config.base_url)
        .map_or(Cow::Borrowed(DEFAULT_BASE_URL), Cow::Owned);

    let base = match check_base_url(&base_raw) {
        Ok(url) => url,
        Err(msg) => refuse(&msg),
    };

    // `answers` picks its own default per request, so it needs the unresolved value.
    let configured_timeout = cli.timeout.or(config.timeout);
    let timeout = configured_timeout.unwrap_or(DEFAULT_TIMEOUT);
    if timeout == 0 || timeout > MAX_TIMEOUT {
        refuse(&format!(
            "timeout must be between 1 and {MAX_TIMEOUT} seconds"
        ));
    }

    let ep = cli.endpoint.as_deref();
    // Before `parse_extra`, which exits in prose: council refuses these outright, in its format.
    if council_format.is_some() && (!cli.extra.is_empty() || ep.is_some()) {
        refuse("--extra and --endpoint are not supported by council");
    }
    let extras = parse_extra(&cli.extra);

    if let Some(ep) = ep
        && let Err(msg) = check_endpoint(ep)
    {
        refuse(&msg);
    }

    // Last, because it can prompt and write a config file: a bad flag is refused first. Sources
    // are trimmed, so the header check catches an interior control character as a bad argument.
    let api_key = match &cli.command {
        // Council never prompts: it refuses a missing key itself, in the caller's format, and
        // only when the run searches.
        Command::Council(_) => {
            configured_api_key(cli.api_key.clone(), config.api_key.clone()).unwrap_or_default()
        }
        _ => {
            let k = resolve_api_key(cli.api_key, config.api_key, cfg_path);
            validate_header_value("X-Subscription-Token", &k);
            k
        }
    };

    match cli.command {
        Command::Context(args) => cmd_context(&base, &api_key, args, &extras, ep, timeout),
        Command::Answers(args) => {
            cmd_answers(&base, &api_key, args, &extras, ep, configured_timeout)
        }
        Command::Web(args) => cmd_web(&base, &api_key, args, &extras, ep, timeout),
        Command::News(args) => cmd_news(&base, &api_key, args, &extras, ep, timeout),
        Command::Images(args) => cmd_images(&base, &api_key, args, &extras, ep, timeout),
        Command::Videos(args) => cmd_videos(&base, &api_key, args, &extras, ep, timeout),
        Command::Places(args) => cmd_places(&base, &api_key, args, &extras, ep, timeout),
        Command::Suggest(args) => cmd_suggest(&base, &api_key, args, &extras, ep, timeout),
        Command::Spellcheck(args) => cmd_spellcheck(&base, &api_key, args, &extras, ep, timeout),
        Command::Pois(args) => cmd_pois(&base, &api_key, args, &extras, ep, timeout),
        Command::Descriptions(args) => {
            cmd_descriptions(&base, &api_key, args, &extras, ep, timeout)
        }
        Command::Council(args) => council::main(args, &base, &api_key, timeout),
        Command::Config { .. } => unreachable!(),
    }
}

/// Allowed base URLs for the Brave Search API.
const ALLOWED_BASE_URLS: &[&str] = &[
    "https://api.search.brave.com",
    "https://api.search.brave.software",
];

/// Validates and normalizes the base URL.
///
/// **Security model (SSRF prevention):**
/// - Production/staging HTTPS URLs pass via a static allowlist.
/// - Localhost URLs (`http://` only) are allowed for local reverse-proxy setups
///   (e.g. sandboxed agents that inject credentials via a proxy).
/// - Only loopback addresses are accepted: `127.0.0.0/8` (IPv4), `::1` (IPv6),
///   or the hostname `localhost`.
///
/// **TOCTOU resistance:** the hostname `localhost` is resolved once, ALL
/// addresses are verified as loopback, then the URL is rewritten to use the
/// resolved literal IP — ureq connects directly, no rebinding window.
fn check_base_url(url: &str) -> Result<Cow<'_, str>, String> {
    let url = url.trim_end_matches('/');
    if ALLOWED_BASE_URLS.contains(&url) {
        return Ok(Cow::Borrowed(url));
    }

    let rest = url.strip_prefix("http://").ok_or_else(|| {
        format!(
            "base URL not allowed (got: {url})\n\
         hint: allowed URLs are {}, or http://localhost:<port>",
            ALLOWED_BASE_URLS.join(", ")
        )
    })?;

    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };

    if authority.is_empty() {
        return Err("base URL has empty host".into());
    }

    // Reject userinfo — prevents SSRF via authority confusion
    // (e.g. http://127.0.0.1@evil.com makes evil.com the real host).
    if authority.contains('@') {
        return Err(format!("base URL must not contain userinfo (got: {url})"));
    }

    let (host, port) = parse_authority(authority)?;

    // IpAddr parser is strict: Ipv4Addr rejects octal (0177.0.0.1), decimal
    // (2130706433), shorthand (127.1), and leading zeros. Ipv6Addr::is_loopback()
    // only matches ::1; IPv4-mapped ::ffff:127.0.0.1 returns false
    // (rust-lang/rust#69772). Both properties block common SSRF bypass techniques.
    if let Ok(ip) = host.parse::<IpAddr>() {
        return if ip.is_loopback() {
            Ok(Cow::Borrowed(url))
        } else {
            Err(format!("{ip} is not a loopback address"))
        };
    }

    // Only "localhost" is allowed — blocks DNS-service bypasses (e.g. nip.io).
    if !host.eq_ignore_ascii_case("localhost") {
        return Err(format!(
            "base URL not allowed (got: {url})\n\
             hint: allowed URLs are {}, or http://localhost:<port>",
            ALLOWED_BASE_URLS.join(", ")
        ));
    }

    // TOCTOU defense: resolve once, verify every address is loopback, rewrite
    // to the resolved literal IP. ureq connects to this IP directly — no
    // re-resolution, no DNS rebinding window.
    let ip = resolve_localhost(host, port.unwrap_or(80))?;
    Ok(Cow::Owned(match port {
        Some(p) => format!("http://{ip}:{p}{path}"),
        None => format!("http://{ip}{path}"),
    }))
}

/// Splits a URL authority into host and optional port.
///
/// IPv6 addresses must be bracketed per RFC 3986 (`[::1]:8080`); the returned
/// host is the bare address without brackets. Port 0 is rejected.
pub(crate) fn parse_authority(authority: &str) -> Result<(&str, Option<u16>), String> {
    let parse_port = |s: &str| -> Result<u16, String> {
        match s.parse::<u16>() {
            Ok(0) => Err("port 0 is not allowed".into()),
            Ok(p) => Ok(p),
            Err(_) => Err(format!("invalid port: '{s}'")),
        }
    };

    if let Some(rest) = authority.strip_prefix('[') {
        let (host, after) = rest
            .split_once(']')
            .ok_or("invalid IPv6 address: missing ']'")?;

        match after.strip_prefix(':') {
            Some(p) => Ok((host, Some(parse_port(p)?))),
            None if after.is_empty() => Ok((host, None)),
            _ => Err("invalid URL authority".into()),
        }
    } else if authority.matches(':').count() > 1 {
        // Only a bracketed literal may hold more than one colon. Unbracketed, `rsplit_once`
        // reads `::1:8080` as the loopback host `::1` on port 8080 — so it passed
        // validation and was then rejected by ureq as a network error.
        Err(format!(
            "invalid URL authority (got: {authority})\n\
             hint: bracket an IPv6 address, as in http://[::1]:<port>"
        ))
    } else {
        match authority.rsplit_once(':') {
            Some((host, p)) => Ok((host, Some(parse_port(p)?))),
            None => Ok((authority, None)),
        }
    }
}

/// Resolves a hostname and verifies ALL addresses are loopback.
///
/// Returns the first resolved address formatted for URL insertion (bracketed
/// for IPv6). Checks ALL addresses — not just the first — to guard against
/// poisoned DNS entries that mix loopback with non-loopback results.
pub(crate) fn resolve_localhost(host: &str, port: u16) -> Result<String, String> {
    let addrs: Vec<_> = (host, port)
        .to_socket_addrs()
        .map_err(|e| {
            format!(
                "could not resolve '{host}': {e}\n\
                 hint: use a literal IP instead (127.0.0.1 or [::1])"
            )
        })?
        .collect();

    if addrs.is_empty() {
        return Err(format!("'{host}' did not resolve to any address"));
    }

    for addr in &addrs {
        if !addr.ip().is_loopback() {
            return Err(format!(
                "'{host}' resolved to non-loopback address {}, rejecting for safety",
                addr.ip()
            ));
        }
    }

    let ip = addrs[0].ip();
    Ok(if ip.is_ipv4() {
        ip.to_string()
    } else {
        format!("[{ip}]")
    })
}

fn non_empty_env(var: &str) -> Option<String> {
    std::env::var(var).ok().and_then(config::trim_non_empty)
}

/// The key as configured, in the documented order, with no prompt, migration or exit, so council
/// sees `--api-key` and `--config` without onboarding.
fn configured_api_key(
    cli_api_key: Option<String>,
    config_api_key: Option<String>,
) -> Option<String> {
    set_api_key(cli_api_key, config_api_key)
        // 4. Legacy ~/.config/brave-search/api_key file, read but not migrated
        .or_else(config::load_legacy_api_key)
}

/// Steps 1–3 of the order: the sources that are pure lookups. Shared so that reporting the key
/// and resolving it cannot drift apart.
fn set_api_key(cli_api_key: Option<String>, config_api_key: Option<String>) -> Option<String> {
    // 1. --api-key flag / BRAVE_SEARCH_API_KEY env (handled by clap)
    cli_api_key
        .and_then(config::trim_non_empty)
        // 2. BRAVE_API_KEY fallback (alternate env name used by some tooling)
        .or_else(|| non_empty_env("BRAVE_API_KEY"))
        // 3. Config file (filter empty — api_key = "" deserializes as Some(""))
        .or_else(|| config_api_key.and_then(config::trim_non_empty))
}

fn resolve_api_key(
    cli_api_key: Option<String>,
    config_api_key: Option<String>,
    cfg_path: Option<&Path>,
) -> String {
    if let Some(k) = set_api_key(cli_api_key, config_api_key) {
        return k;
    }
    // 4. Legacy ~/.config/brave-search/api_key file → auto-migrate
    if let Some(k) = config::load_legacy_api_key() {
        match config::migrate_legacy_key(&k, cfg_path) {
            Ok(()) => eprintln!("note: migrated API key from legacy api_key file to config.json"),
            Err(e) => eprintln!(
                "warning: failed to migrate legacy API key: {e}; continuing with legacy api_key file"
            ),
        }
        return k;
    }
    // 5. Interactive onboarding
    match config::onboard(cfg_path) {
        Ok(k) => k,
        Err(msg) => {
            eprintln!("error: {msg}");
            std::process::exit(2);
        }
    }
}

// ── Location headers helper ──────────────────────────────────────────

struct LocationHeaders {
    lat: Option<String>,
    long: Option<String>,
    timezone: Option<String>,
    city: Option<String>,
    state: Option<String>,
    state_name: Option<String>,
    country: Option<String>,
    postal_code: Option<String>,
}

/// Defense-in-depth: validate endpoint path against a strict allowlist.
/// Allowed characters: ASCII alphanumeric, `/`, `-`, `_`, `.`
fn check_endpoint(ep: &str) -> Result<(), String> {
    if !ep.starts_with('/') {
        return Err(format!("--endpoint must start with '/', got: {ep}"));
    }
    if let Some(b) = ep
        .bytes()
        .find(|&b| !b.is_ascii_alphanumeric() && b != b'/' && b != b'-' && b != b'_' && b != b'.')
    {
        return Err(format!(
            "--endpoint contains disallowed character '{}'",
            char::from(b)
        ));
    }
    if ep.contains("//") {
        return Err("--endpoint must not contain consecutive slashes".into());
    }
    if ep.split('/').any(|s| s == "..") {
        return Err("--endpoint must not contain '..' path segments".into());
    }
    Ok(())
}

/// Defense-in-depth: reject HTTP header values containing control characters
/// (bytes < 32 or DEL). Stricter than the `http` crate which allows tab.
/// Ensures clear errors even if the underlying HTTP client changes.
/// Ref: https://docs.rs/http/latest/http/header/struct.HeaderValue.html
fn check_header_value(name: &str, value: &str) -> Result<(), String> {
    if value.bytes().any(|b| b < 32 || b == 127) {
        return Err(format!(
            "invalid header value for {name}: contains control characters"
        ));
    }
    Ok(())
}

fn validate_header_value(name: &str, value: &str) {
    if let Err(msg) = check_header_value(name, value) {
        eprintln!("error: {msg}");
        std::process::exit(2);
    }
}

fn location_header_pairs(loc: &LocationHeaders) -> Vec<(&str, &str)> {
    let mut headers = Vec::new();
    if let Some(ref v) = loc.lat {
        validate_header_value("X-Loc-Lat", v);
        headers.push(("X-Loc-Lat", v.as_str()));
    }
    if let Some(ref v) = loc.long {
        validate_header_value("X-Loc-Long", v);
        headers.push(("X-Loc-Long", v.as_str()));
    }
    if let Some(ref v) = loc.timezone {
        validate_header_value("X-Loc-Timezone", v);
        headers.push(("X-Loc-Timezone", v.as_str()));
    }
    if let Some(ref v) = loc.city {
        validate_header_value("X-Loc-City", v);
        headers.push(("X-Loc-City", v.as_str()));
    }
    if let Some(ref v) = loc.state {
        validate_header_value("X-Loc-State", v);
        headers.push(("X-Loc-State", v.as_str()));
    }
    if let Some(ref v) = loc.state_name {
        validate_header_value("X-Loc-State-Name", v);
        headers.push(("X-Loc-State-Name", v.as_str()));
    }
    if let Some(ref v) = loc.country {
        validate_header_value("X-Loc-Country", v);
        headers.push(("X-Loc-Country", v.as_str()));
    }
    if let Some(ref v) = loc.postal_code {
        validate_header_value("X-Loc-Postal-Code", v);
        headers.push(("X-Loc-Postal-Code", v.as_str()));
    }
    headers
}

// ── Site shortcuts ───────────────────────────────────────────────────

/// Validates that a domain contains only allowed characters.
fn validate_domain(s: &str) -> Result<String, String> {
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_')
    {
        Ok(s.to_string())
    } else {
        Err(format!(
            "invalid domain '{s}': only letters, digits, dots, hyphens, underscores allowed"
        ))
    }
}

/// Builds an inline Goggles string from --include-site / --exclude-site flags.
fn build_site_goggles(include: &[String], exclude: &[String]) -> Option<String> {
    if !include.is_empty() {
        let mut rules = vec!["$discard".to_string()];
        rules.extend(include.iter().map(|d| format!("$boost,site={d}")));
        Some(rules.join("\n"))
    } else if !exclude.is_empty() {
        let rules: Vec<String> = exclude
            .iter()
            .map(|d| format!("$discard,site={d}"))
            .collect();
        Some(rules.join("\n"))
    } else {
        None
    }
}

// ── Goggles resolution ───────────────────────────────────────────────

impl GogglesArgs {
    /// Resolves goggles from --goggles, --include-site, or --exclude-site.
    fn resolve(&self) -> Option<String> {
        if !self.goggles.is_empty() {
            for v in &self.goggles {
                warn_shell_expanded_goggles(v);
            }
            let parts: Vec<Cow<str>> = self.goggles.iter().map(|v| resolve_goggles(v)).collect();
            Some(parts.join("\n"))
        } else {
            build_site_goggles(&self.include_site, &self.exclude_site)
        }
    }
}

/// Warns if an inline --goggles value looks like it suffered shell variable expansion.
/// e.g. "$site=example.org" in double quotes → "=example.org" after the shell eats $site.
fn warn_shell_expanded_goggles(value: &str) {
    if value.starts_with('@') || value.starts_with("http://") || value.starts_with("https://") {
        return;
    }
    if value.starts_with('=') || value.starts_with(',') {
        eprintln!(
            "warning: --goggles value starts with '{}' — \
             this often means the shell expanded a $variable to nothing\n\
             hint: use single quotes to prevent expansion, e.g. --goggles '$site=example.org'",
            &value[..1]
        );
    }
}

/// Maximum size for file/stdin reads (goggles, answers stdin JSON).
const MAX_INPUT_SIZE: u64 = 1024 * 1024; // 1 MB

/// Resolves a --goggles value:
///   @-       → read from stdin
///   @path    → read from file
///   http(s)  → pass through (hosted goggle URL)
///   other    → inline rules (\n unescaped to newlines)
fn resolve_goggles(value: &str) -> Cow<'_, str> {
    if let Some(path) = value.strip_prefix('@') {
        if path == "-" {
            let mut buf = String::new();
            let mut limited = std::io::Read::take(std::io::stdin(), MAX_INPUT_SIZE + 1);
            if let Err(e) = std::io::Read::read_to_string(&mut limited, &mut buf) {
                eprintln!("error: failed to read goggles from stdin: {e}");
                std::process::exit(2);
            }
            if buf.len() as u64 > MAX_INPUT_SIZE {
                eprintln!("error: goggles input exceeds maximum size ({MAX_INPUT_SIZE} bytes)");
                std::process::exit(2);
            }
            Cow::Owned(buf)
        } else {
            match std::fs::metadata(path) {
                Ok(meta) if meta.len() > MAX_INPUT_SIZE => {
                    eprintln!(
                        "error: goggles file '{path}' exceeds maximum size ({MAX_INPUT_SIZE} bytes)"
                    );
                    std::process::exit(2);
                }
                Err(e) => {
                    eprintln!("error: failed to read goggles file '{path}': {e}");
                    std::process::exit(2);
                }
                _ => {}
            }
            match std::fs::read_to_string(path) {
                Ok(contents) => Cow::Owned(contents),
                Err(e) => {
                    eprintln!("error: failed to read goggles file '{path}': {e}");
                    std::process::exit(2);
                }
            }
        }
    } else if value.starts_with("http://") || value.starts_with("https://") {
        Cow::Borrowed(value)
    } else {
        unescape_inline_newlines(value)
    }
}

/// Unescapes `\n` → newline and `\\` → backslash in inline goggles.
/// Not applied to @file/@- reads (those already contain real newlines) or hosted URLs.
fn unescape_inline_newlines(s: &str) -> Cow<'_, str> {
    if !s.contains('\\') {
        return Cow::Borrowed(s);
    }
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, chars.peek()) {
            ('\\', Some(&'n')) => {
                chars.next();
                result.push('\n');
            }
            ('\\', Some(&'\\')) => {
                chars.next();
                result.push('\\');
            }
            _ => result.push(c),
        }
    }
    Cow::Owned(result)
}

// ── Command handlers ─────────────────────────────────────────────────

fn cmd_web(
    base: &str,
    key: &str,
    a: WebArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    timeout: u64,
) {
    let goggles_resolved = a.goggles_args.resolve();
    let mut body = query::build_json_body(&[
        ("country", a.country.map(Into::into)),
        ("search_lang", a.search_lang.map(Into::into)),
        ("ui_lang", a.ui_lang.map(Into::into)),
        ("count", a.count.map(Into::into)),
        ("offset", a.offset.map(Into::into)),
        ("safesearch", a.safesearch.map(Into::into)),
        ("freshness", a.freshness.map(Into::into)),
        ("text_decorations", a.text_decorations.map(Into::into)),
        ("spellcheck", a.spellcheck.map(Into::into)),
        ("goggles", goggles_resolved.map(Into::into)),
        ("extra_snippets", a.extra_snippets.map(Into::into)),
        ("units", a.units.map(Into::into)),
        ("operators", a.operators.map(Into::into)),
    ]);
    body["q"] = a.q.into();
    // POST body requires result_filter as a JSON array, not a comma-separated string.
    if let Some(ref rf) = a.result_filter {
        let arr: Vec<_> = rf
            .split(',')
            .map(|s| serde_json::Value::String(s.trim().into()))
            .collect();
        body["result_filter"] = serde_json::Value::Array(arr);
    }
    merge_extras(&mut body, extras);
    let loc = LocationHeaders {
        lat: a.lat,
        long: a.long,
        timezone: a.timezone,
        city: a.city,
        state: a.state,
        state_name: a.state_name,
        country: a.loc_country,
        postal_code: a.postal_code,
    };
    let headers = location_header_pairs(&loc);
    api::post_json(
        base,
        ep.unwrap_or("/res/v1/web/search"),
        key,
        &body,
        &headers,
        timeout,
    );
}

fn cmd_images(
    base: &str,
    key: &str,
    a: ImagesArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    timeout: u64,
) {
    let count_str = a.count.map(|v| v.to_string());
    let params: &[(&str, Option<&str>)] = &[
        ("q", Some(a.q.as_str())),
        ("country", a.country.as_deref()),
        ("search_lang", a.search_lang.as_deref()),
        ("count", count_str.as_deref()),
        ("safesearch", a.safesearch.as_deref()),
        ("spellcheck", a.spellcheck.map(bool_str)),
    ];
    let qs = query::build_query(params, extras);
    let path = format!("{}{qs}", ep.unwrap_or("/res/v1/images/search"));
    api::get(base, &path, key, timeout);
}

fn cmd_videos(
    base: &str,
    key: &str,
    a: VideosArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    timeout: u64,
) {
    let mut body = query::build_json_body(&[
        ("country", a.country.map(Into::into)),
        ("search_lang", a.search_lang.map(Into::into)),
        ("ui_lang", a.ui_lang.map(Into::into)),
        ("count", a.count.map(Into::into)),
        ("offset", a.offset.map(Into::into)),
        ("safesearch", a.safesearch.map(Into::into)),
        ("freshness", a.freshness.map(Into::into)),
        ("spellcheck", a.spellcheck.map(Into::into)),
        ("operators", a.operators.map(Into::into)),
    ]);
    body["q"] = a.q.into();
    merge_extras(&mut body, extras);
    api::post_json(
        base,
        ep.unwrap_or("/res/v1/videos/search"),
        key,
        &body,
        &[],
        timeout,
    );
}

fn cmd_news(
    base: &str,
    key: &str,
    a: NewsArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    timeout: u64,
) {
    let goggles_resolved = a.goggles_args.resolve();
    let mut body = query::build_json_body(&[
        ("country", a.country.map(Into::into)),
        ("search_lang", a.search_lang.map(Into::into)),
        ("ui_lang", a.ui_lang.map(Into::into)),
        ("count", a.count.map(Into::into)),
        ("offset", a.offset.map(Into::into)),
        ("safesearch", a.safesearch.map(Into::into)),
        ("freshness", a.freshness.map(Into::into)),
        ("spellcheck", a.spellcheck.map(Into::into)),
        ("extra_snippets", a.extra_snippets.map(Into::into)),
        ("goggles", goggles_resolved.map(Into::into)),
        ("operators", a.operators.map(Into::into)),
    ]);
    body["q"] = a.q.into();
    merge_extras(&mut body, extras);
    api::post_json(
        base,
        ep.unwrap_or("/res/v1/news/search"),
        key,
        &body,
        &[],
        timeout,
    );
}

fn cmd_suggest(
    base: &str,
    key: &str,
    a: SuggestArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    timeout: u64,
) {
    let count_str = a.count.map(|v| v.to_string());
    let params: &[(&str, Option<&str>)] = &[
        ("q", Some(a.q.as_str())),
        ("lang", a.lang.as_deref()),
        ("country", a.country.as_deref()),
        ("count", count_str.as_deref()),
        ("rich", a.rich.map(bool_str)),
    ];
    let qs = query::build_query(params, extras);
    let path = format!("{}{qs}", ep.unwrap_or("/res/v1/suggest/search"));
    api::get(base, &path, key, timeout);
}

fn cmd_spellcheck(
    base: &str,
    key: &str,
    a: SpellcheckArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    timeout: u64,
) {
    let params: &[(&str, Option<&str>)] = &[
        ("q", Some(a.q.as_str())),
        ("lang", a.lang.as_deref()),
        ("country", a.country.as_deref()),
    ];
    let qs = query::build_query(params, extras);
    let path = format!("{}{qs}", ep.unwrap_or("/res/v1/spellcheck/search"));
    api::get(base, &path, key, timeout);
}

/// The two deadlines an answers request needs: `dial` to the first byte, `read` for silence once
/// streaming (the whole budget when blocking). Research earns a longer `read`, never a longer
/// dial, or a dead host would take minutes to report. Read from the merged body, so every input
/// path agrees.
fn answers_timeouts(configured: Option<u64>, body: &serde_json::Value) -> (u64, u64) {
    let read = configured.unwrap_or(match query::json_bool(&body["enable_research"]) {
        Some(true) => DEFAULT_RESEARCH_TIMEOUT,
        _ => DEFAULT_TIMEOUT,
    });
    (configured.unwrap_or(DEFAULT_TIMEOUT), read)
}

/// The transport the merged body asks for, written back as a real boolean so the request and
/// the parser agree; `--extra` types a bare `0` as a number.
fn resolve_stream(body: &mut serde_json::Value, default: bool) -> bool {
    let stream = query::json_bool(&body["stream"]).unwrap_or(default);
    // `as_object_mut`, not `body["stream"] = …`: `IndexMut` panics on the non-object body a
    // stdin caller can supply.
    if let Some(obj) = body.as_object_mut() {
        obj.insert("stream".into(), stream.into());
    }
    stream
}

fn cmd_answers(
    base: &str,
    key: &str,
    a: AnswersArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    configured_timeout: Option<u64>,
) {
    let path = ep.unwrap_or("/res/v1/chat/completions");

    // Stdin mode: read raw JSON body from stdin when query is "-".
    if a.query == "-" {
        let mut buf = String::new();
        let mut limited = std::io::Read::take(std::io::stdin().lock(), MAX_INPUT_SIZE + 1);
        if let Err(e) = std::io::Read::read_to_string(&mut limited, &mut buf) {
            eprintln!("error: failed to read JSON from stdin: {e}");
            std::process::exit(2);
        }
        if buf.len() as u64 > MAX_INPUT_SIZE {
            eprintln!("error: stdin JSON exceeds maximum size ({MAX_INPUT_SIZE} bytes)");
            std::process::exit(2);
        }
        let mut body: serde_json::Value = match query::parse_json(&buf) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: invalid JSON on stdin: {e}");
                std::process::exit(2);
            }
        };
        merge_extras(&mut body, extras);

        let (dial, read) = answers_timeouts(configured_timeout, &body);
        if resolve_stream(&mut body, !a.no_stream) {
            api::post_json_stream(base, path, key, &body, dial, read);
        } else {
            api::post_json(base, path, key, &body, &[], read);
        }
        return;
    }

    // Simple mode: build JSON body from CLI args.
    let stream = !a.no_stream;
    let mut body = serde_json::json!({
        "messages": [{"role": "user", "content": a.query}],
        "stream": stream,
    });

    let obj = body.as_object_mut().expect("body must be a JSON object");
    for (key, val) in [
        ("model", a.model),
        ("country", a.country),
        ("language", a.language),
        ("safesearch", a.safesearch),
    ] {
        if let Some(v) = val {
            obj.insert(key.into(), v.into());
        }
    }

    if let Some(max) = a.max_completion_tokens {
        obj.insert("max_completion_tokens".into(), max.into());
    }
    if a.enable_citations {
        obj.insert("enable_citations".into(), true.into());
    }
    if a.enable_entities {
        obj.insert("enable_entities".into(), true.into());
    }
    if a.enable_research {
        obj.insert("enable_research".into(), true.into());
    }
    if let Some(v) = a.research_allow_thinking {
        obj.insert("research_allow_thinking".into(), v.into());
    }
    if let Some(v) = a.research_max_tokens_per_query {
        obj.insert(
            "research_maximum_number_of_tokens_per_query".into(),
            v.into(),
        );
    }
    if let Some(v) = a.research_max_queries {
        obj.insert("research_maximum_number_of_queries".into(), v.into());
    }
    if let Some(v) = a.research_max_iterations {
        obj.insert("research_maximum_number_of_iterations".into(), v.into());
    }
    if let Some(v) = a.research_max_seconds {
        obj.insert("research_maximum_number_of_seconds".into(), v.into());
    }
    if let Some(v) = a.research_max_results_per_query {
        obj.insert(
            "research_maximum_number_of_results_per_query".into(),
            v.into(),
        );
    }

    // web_search_options
    let has_wso = a.search_context_size.is_some()
        || a.user_city.is_some()
        || a.user_country.is_some()
        || a.user_region.is_some()
        || a.user_timezone.is_some();

    if has_wso {
        let mut wso = serde_json::Map::new();
        if let Some(size) = a.search_context_size {
            wso.insert("search_context_size".into(), size.into());
        }
        let mut approx = serde_json::Map::new();
        if let Some(v) = a.user_city {
            approx.insert("city".into(), v.into());
        }
        if let Some(v) = a.user_country {
            approx.insert("country".into(), v.into());
        }
        if let Some(v) = a.user_region {
            approx.insert("region".into(), v.into());
        }
        if let Some(v) = a.user_timezone {
            approx.insert("timezone".into(), v.into());
        }
        if !approx.is_empty() {
            let mut loc = serde_json::Map::new();
            loc.insert("approximate".into(), approx.into());
            wso.insert("user_location".into(), loc.into());
        }
        obj.insert("web_search_options".into(), wso.into());
    }

    merge_extras(&mut body, extras);

    let (dial, read) = answers_timeouts(configured_timeout, &body);
    if resolve_stream(&mut body, stream) {
        api::post_json_stream(base, path, key, &body, dial, read);
    } else {
        api::post_json(base, path, key, &body, &[], read);
    }
}

fn cmd_context(
    base: &str,
    key: &str,
    a: ContextArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    timeout: u64,
) {
    let goggles_resolved = a.goggles_args.resolve();
    let mut body = query::build_json_body(&[
        ("country", a.country.map(Into::into)),
        ("search_lang", a.search_lang.map(Into::into)),
        ("count", a.count.map(Into::into)),
        (
            "maximum_number_of_urls",
            a.maximum_number_of_urls.map(Into::into),
        ),
        (
            "maximum_number_of_tokens",
            a.maximum_number_of_tokens.map(Into::into),
        ),
        (
            "maximum_number_of_snippets",
            a.maximum_number_of_snippets.map(Into::into),
        ),
        (
            "maximum_number_of_tokens_per_url",
            a.maximum_number_of_tokens_per_url.map(Into::into),
        ),
        (
            "maximum_number_of_snippets_per_url",
            a.maximum_number_of_snippets_per_url.map(Into::into),
        ),
        (
            "context_threshold_mode",
            a.context_threshold_mode.map(Into::into),
        ),
        ("goggles", goggles_resolved.map(Into::into)),
        ("enable_local", a.enable_local.map(Into::into)),
    ]);
    body["q"] = a.q.into();
    merge_extras(&mut body, extras);
    let loc = LocationHeaders {
        lat: a.lat,
        long: a.long,
        timezone: a.timezone,
        city: a.city,
        state: a.state,
        state_name: a.state_name,
        country: a.loc_country,
        postal_code: a.postal_code,
    };
    let headers = location_header_pairs(&loc);
    api::post_json(
        base,
        ep.unwrap_or("/res/v1/llm/context"),
        key,
        &body,
        &headers,
        timeout,
    );
}

fn cmd_places(
    base: &str,
    key: &str,
    a: PlacesArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    timeout: u64,
) {
    let count_str = a.count.map(|v| v.to_string());
    let params: &[(&str, Option<&str>)] = &[
        ("q", a.q.as_deref()),
        ("latitude", a.latitude.as_deref()),
        ("longitude", a.longitude.as_deref()),
        ("location", a.location.as_deref()),
        ("radius", a.radius.as_deref()),
        ("count", count_str.as_deref()),
        ("country", a.country.as_deref()),
        ("search_lang", a.search_lang.as_deref()),
        ("ui_lang", a.ui_lang.as_deref()),
        ("units", a.units.as_deref()),
        ("safesearch", a.safesearch.as_deref()),
        ("spellcheck", a.spellcheck.map(bool_str)),
    ];
    let qs = query::build_query(params, extras);
    let path = format!("{}{qs}", ep.unwrap_or("/res/v1/local/place_search"));
    api::get(base, &path, key, timeout);
}

fn cmd_pois(
    base: &str,
    key: &str,
    a: PoisArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    timeout: u64,
) {
    if a.ids.is_empty() {
        eprintln!("error: at least one POI ID is required");
        std::process::exit(2);
    }
    let params: &[(&str, Option<&str>)] = &[
        ("search_lang", a.search_lang.as_deref()),
        ("ui_lang", a.ui_lang.as_deref()),
        ("units", a.units.as_deref()),
    ];
    let qs = query::build_query_repeated(params, &[("ids", &a.ids)], extras);
    let mut headers = Vec::new();
    if let Some(ref v) = a.lat {
        validate_header_value("X-Loc-Lat", v);
        headers.push(("X-Loc-Lat", v.as_str()));
    }
    if let Some(ref v) = a.long {
        validate_header_value("X-Loc-Long", v);
        headers.push(("X-Loc-Long", v.as_str()));
    }
    let path = format!("{}{qs}", ep.unwrap_or("/res/v1/local/pois"));
    api::get_with_headers(base, &path, key, &headers, timeout);
}

fn cmd_descriptions(
    base: &str,
    key: &str,
    a: DescriptionsArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    timeout: u64,
) {
    if a.ids.is_empty() {
        eprintln!("error: at least one POI ID is required");
        std::process::exit(2);
    }
    let qs = query::build_query_repeated(&[], &[("ids", &a.ids)], extras);
    let path = format!("{}{qs}", ep.unwrap_or("/res/v1/local/descriptions"));
    api::get(base, &path, key, timeout);
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
