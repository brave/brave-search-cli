mod api;
mod config;
mod update;

use std::borrow::Cow;
use std::net::{IpAddr, ToSocketAddrs};
use std::path::Path;

use clap::{Args, Parser, Subcommand};

/// Search the Web via Brave's independent index. The only search API with its own Web index at
/// scale. Truly independent, lightning-fast, and built to power AI apps. Private and secure, your
/// queries never leave Brave.
///
/// All commands output JSON to stdout, errors to stderr.
/// API key: https://api-dashboard.search.brave.com
///
/// WHICH COMMAND?
///   Looking up docs, APIs, errors, code patterns? → context (default, recommended)
///   Need a synthesized answer with citations? → answers
///   Need raw search results or result filtering? → web
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
#[derive(Parser)]
#[command(name = "bx", version, verbatim_doc_comment)]
struct Cli {
    /// Path to config file
    #[arg(long, global = true)]
    config: Option<std::path::PathBuf>,

    /// API key (prefer env var or config file — command-line flags are visible in process listings)
    #[arg(
        long,
        env = "BRAVE_SEARCH_API_KEY",
        global = true,
        hide_env_values = true
    )]
    api_key: Option<String>,

    /// Base URL for the API [default: https://api.search.brave.com]
    #[arg(
        long,
        env = "BRAVE_SEARCH_BASE_URL",
        global = true,
        hide_env_values = true
    )]
    base_url: Option<String>,

    /// Request timeout in seconds [default: 30]
    #[arg(long, global = true)]
    timeout: Option<u64>,

    /// Extra API parameters (KEY=VALUE, repeatable). Merged into request body
    /// (POST) or query string (GET). Warns on collision with existing flags.
    /// POST values are auto-typed: integers, floats, true/false → JSON natives.
    #[arg(long, global = true, value_name = "KEY=VALUE")]
    extra: Vec<String>,

    /// Override the API path (e.g. /res/v1/web/search). For internal/beta Brave endpoints.
    #[arg(long, global = true)]
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
    /// Use --no-stream for a single JSON response.
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

    /// Update bx to the latest version
    ///
    /// Downloads and installs the latest release from GitHub.
    /// Uses SHA256 checksum verification plus pinned code signatures (Windows thumbprints,
    /// and macOS Team ID per brave.com/signing-keys).
    ///
    /// Examples:
    ///   bx update            # download and install latest version
    ///   bx update --check    # check for updates without installing
    #[command(verbatim_doc_comment)]
    Update {
        /// Only check for updates, don't install
        #[arg(long)]
        check: bool,
    },

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

    /// Enable citations (requires streaming)
    #[arg(long)]
    enable_citations: bool,

    /// Enable entities (requires streaming)
    #[arg(long)]
    enable_entities: bool,

    /// Enable research mode (requires streaming)
    #[arg(long)]
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
    "update",
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

fn inject_default_subcommand_impl(mut args: Vec<String>) -> Vec<String> {
    // Flags that consume the next argument as a value
    const VALUE_FLAGS: &[&str] = &[
        "--api-key",
        "--base-url",
        "--timeout",
        "--config",
        "--extra",
        "--endpoint",
    ];

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

/// Parses --extra KEY=VALUE pairs. Returns borrowed slices into the input.
fn parse_extra(extras: &[String]) -> Vec<(&str, &str)> {
    extras
        .iter()
        .map(|entry| match entry.split_once('=') {
            Some((k, v)) if !k.is_empty() => (k, v),
            _ => {
                eprintln!("error: --extra requires KEY=VALUE format, got: {entry}");
                std::process::exit(1);
            }
        })
        .collect()
}

/// Merges --extra pairs into a JSON body, exiting on error.
fn merge_extras(body: &mut serde_json::Value, extras: &[(&str, &str)]) {
    if let Err(msg) = api::merge_extra_into_json(body, extras) {
        eprintln!("error: {msg}");
        std::process::exit(1);
    }
}

/// Converts a bool to a static string for GET query parameters.
fn bool_str(v: bool) -> &'static str {
    if v { "true" } else { "false" }
}

const DEFAULT_BASE_URL: &str = "https://api.search.brave.com";
const DEFAULT_TIMEOUT: u64 = 30;

fn main() {
    let cli = Cli::parse_from(inject_default_subcommand());
    let cfg_path = cli.config.as_deref();

    // Clean up stale .old binary from a previous Windows update.
    #[cfg(windows)]
    update::cleanup_old_binary();

    // Config subcommand doesn't need an API key.
    if let Command::Config { ref cmd } = cli.command {
        config::handle_config(cmd, cfg_path);
        return;
    }

    // Update subcommand doesn't need an API key.
    if let Command::Update { check } = cli.command {
        let code = if check {
            update::check_for_update()
        } else {
            update::perform_update()
        };
        std::process::exit(code);
    }

    let config = match config::load_config(cfg_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let api_key = resolve_api_key(cli.api_key, config.api_key, cfg_path);

    let base_raw: Cow<'static, str> = cli
        .base_url
        .or(config.base_url)
        .map_or(Cow::Borrowed(DEFAULT_BASE_URL), Cow::Owned);

    let base = match check_base_url(&base_raw) {
        Ok(url) => url,
        Err(msg) => {
            eprintln!("error: {msg}");
            std::process::exit(1);
        }
    };

    let timeout = cli.timeout.or(config.timeout).unwrap_or(DEFAULT_TIMEOUT);
    if timeout == 0 {
        eprintln!("error: timeout must be greater than 0");
        std::process::exit(1);
    }

    let extras = parse_extra(&cli.extra);
    let ep = cli.endpoint.as_deref();

    if let Some(ep) = ep {
        if let Err(msg) = check_endpoint(ep) {
            eprintln!("error: {msg}");
            std::process::exit(1);
        }
    }

    match cli.command {
        Command::Context(args) => cmd_context(&base, &api_key, args, &extras, ep, timeout),
        Command::Answers(args) => cmd_answers(&base, &api_key, args, &extras, ep, timeout),
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
        Command::Update { .. } | Command::Config { .. } => unreachable!(),
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
fn parse_authority(authority: &str) -> Result<(&str, Option<u16>), String> {
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
fn resolve_localhost(host: &str, port: u16) -> Result<String, String> {
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

fn resolve_api_key(
    cli_api_key: Option<String>,
    config_api_key: Option<String>,
    cfg_path: Option<&Path>,
) -> String {
    // 1. --api-key flag / BRAVE_SEARCH_API_KEY env (handled by clap)
    if let Some(k) = cli_api_key.and_then(config::trim_non_empty) {
        return k;
    }
    // 2. BRAVE_API_KEY fallback (alternate env name used by some tooling)
    if let Some(k) = non_empty_env("BRAVE_API_KEY") {
        return k;
    }
    // 3. Config file (filter empty — api_key = "" deserializes as Some(""))
    if let Some(k) = config_api_key.and_then(config::trim_non_empty) {
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
            std::process::exit(1);
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
        std::process::exit(1);
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
                std::process::exit(1);
            }
            if buf.len() as u64 > MAX_INPUT_SIZE {
                eprintln!("error: goggles input exceeds maximum size ({MAX_INPUT_SIZE} bytes)");
                std::process::exit(1);
            }
            Cow::Owned(buf)
        } else {
            match std::fs::metadata(path) {
                Ok(meta) if meta.len() > MAX_INPUT_SIZE => {
                    eprintln!(
                        "error: goggles file '{path}' exceeds maximum size ({MAX_INPUT_SIZE} bytes)"
                    );
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("error: failed to read goggles file '{path}': {e}");
                    std::process::exit(1);
                }
                _ => {}
            }
            match std::fs::read_to_string(path) {
                Ok(contents) => Cow::Owned(contents),
                Err(e) => {
                    eprintln!("error: failed to read goggles file '{path}': {e}");
                    std::process::exit(1);
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
    let mut body = api::build_json_body(&[
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
    let qs = api::build_query(params, extras);
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
    let mut body = api::build_json_body(&[
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
    let mut body = api::build_json_body(&[
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
    let qs = api::build_query(params, extras);
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
    let qs = api::build_query(params, extras);
    let path = format!("{}{qs}", ep.unwrap_or("/res/v1/spellcheck/search"));
    api::get(base, &path, key, timeout);
}

fn cmd_answers(
    base: &str,
    key: &str,
    a: AnswersArgs,
    extras: &[(&str, &str)],
    ep: Option<&str>,
    timeout: u64,
) {
    let path = ep.unwrap_or("/res/v1/chat/completions");

    // Stdin mode: read raw JSON body from stdin when query is "-".
    if a.query == "-" {
        let mut buf = String::new();
        let mut limited = std::io::Read::take(std::io::stdin().lock(), MAX_INPUT_SIZE + 1);
        if let Err(e) = std::io::Read::read_to_string(&mut limited, &mut buf) {
            eprintln!("error: failed to read JSON from stdin: {e}");
            std::process::exit(1);
        }
        if buf.len() as u64 > MAX_INPUT_SIZE {
            eprintln!("error: stdin JSON exceeds maximum size ({MAX_INPUT_SIZE} bytes)");
            std::process::exit(1);
        }
        let mut body: serde_json::Value = match serde_json::from_str(&buf) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: invalid JSON on stdin: {e}");
                std::process::exit(1);
            }
        };
        merge_extras(&mut body, extras);

        let is_stream = body["stream"].as_bool().unwrap_or(true);
        if is_stream {
            api::post_json_stream(base, path, key, &body, &[], timeout);
        } else {
            api::post_json(base, path, key, &body, &[], timeout);
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

    if stream {
        api::post_json_stream(base, path, key, &body, &[], timeout);
    } else {
        api::post_json(base, path, key, &body, &[], timeout);
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
    let mut body = api::build_json_body(&[
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
    let qs = api::build_query(params, extras);
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
        std::process::exit(1);
    }
    let params: &[(&str, Option<&str>)] = &[
        ("search_lang", a.search_lang.as_deref()),
        ("ui_lang", a.ui_lang.as_deref()),
        ("units", a.units.as_deref()),
    ];
    let qs = api::build_query_repeated(params, &[("ids", &a.ids)], extras);
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
        std::process::exit(1);
    }
    let qs = api::build_query_repeated(&[], &[("ids", &a.ids)], extras);
    let path = format!("{}{qs}", ep.unwrap_or("/res/v1/local/descriptions"));
    api::get(base, &path, key, timeout);
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn subcommands_list_matches_clap_enum() {
        let cmd = Cli::command();
        let mut clap_names: std::collections::HashSet<&str> =
            cmd.get_subcommands().map(|s| s.get_name()).collect();

        // "help" is implicitly added by clap but not returned by get_subcommands().
        // We include it in SUBCOMMANDS so `bx help` isn't treated as a search query.
        clap_names.insert("help");

        let hardcoded: std::collections::HashSet<&str> = SUBCOMMANDS.iter().copied().collect();

        let missing: Vec<&&str> = clap_names.difference(&hardcoded).collect();
        let extra: Vec<&&str> = hardcoded.difference(&clap_names).collect();

        assert!(
            missing.is_empty(),
            "Subcommands in clap but missing from SUBCOMMANDS: {missing:?}"
        );
        assert!(
            extra.is_empty(),
            "Entries in SUBCOMMANDS but not in clap: {extra:?}"
        );
    }

    #[test]
    fn check_header_value_accepts_normal_ascii() {
        assert!(check_header_value("X-Test", "New York").is_ok());
        assert!(check_header_value("X-Test", "America/Los_Angeles").is_ok());
        assert!(check_header_value("X-Test", "90210").is_ok());
        assert!(check_header_value("X-Test", "47.6062").is_ok());
    }

    #[test]
    fn check_header_value_accepts_utf8() {
        // Non-ASCII UTF-8 (accented city names) should pass our check.
        // The http crate also accepts these (obs-text per RFC 7230).
        assert!(check_header_value("X-Loc-City", "Zürich").is_ok());
        assert!(check_header_value("X-Loc-City", "São Paulo").is_ok());
    }

    #[test]
    fn check_header_value_rejects_newline() {
        assert!(check_header_value("X-Test", "evil\ninjection").is_err());
    }

    #[test]
    fn check_header_value_rejects_carriage_return() {
        assert!(check_header_value("X-Test", "evil\rinjection").is_err());
    }

    #[test]
    fn check_header_value_rejects_null() {
        assert!(check_header_value("X-Test", "evil\0injection").is_err());
    }

    // ── check_base_url tests ─────────────────────────────────────────

    #[test]
    fn check_base_url_allows_production_urls() {
        assert_eq!(
            check_base_url("https://api.search.brave.com").unwrap(),
            "https://api.search.brave.com"
        );
        assert_eq!(
            check_base_url("https://api.search.brave.software").unwrap(),
            "https://api.search.brave.software"
        );
        // Trailing slash stripped
        assert_eq!(
            check_base_url("https://api.search.brave.com/").unwrap(),
            "https://api.search.brave.com"
        );
    }

    #[test]
    fn check_base_url_accepts_ipv4_loopback() {
        assert_eq!(
            check_base_url("http://127.0.0.1:8080").unwrap(),
            "http://127.0.0.1:8080"
        );
        assert_eq!(
            check_base_url("http://127.0.0.1").unwrap(),
            "http://127.0.0.1"
        );
        // Full 127.0.0.0/8 range
        assert_eq!(
            check_base_url("http://127.255.255.255:3000").unwrap(),
            "http://127.255.255.255:3000"
        );
        // Path preserved
        assert_eq!(
            check_base_url("http://127.0.0.1:8080/brave").unwrap(),
            "http://127.0.0.1:8080/brave"
        );
        // Trailing slash stripped (prevents double-slash in URL construction)
        assert_eq!(
            check_base_url("http://127.0.0.1:8080/").unwrap(),
            "http://127.0.0.1:8080"
        );
        // Multiple trailing slashes all stripped
        assert_eq!(
            check_base_url("http://127.0.0.1:8080///").unwrap(),
            "http://127.0.0.1:8080"
        );
    }

    #[test]
    fn check_base_url_accepts_ipv6_loopback() {
        assert_eq!(
            check_base_url("http://[::1]:8080").unwrap(),
            "http://[::1]:8080"
        );
        assert_eq!(check_base_url("http://[::1]").unwrap(), "http://[::1]");
        assert_eq!(
            check_base_url("http://[::1]:8080/v1").unwrap(),
            "http://[::1]:8080/v1"
        );
    }

    #[test]
    fn check_base_url_accepts_localhost_hostname() {
        // Skip if localhost doesn't resolve (unusual CI environments)
        if ("localhost", 80u16).to_socket_addrs().is_err() {
            return;
        }
        let result = check_base_url("http://localhost:8080").unwrap();
        // TOCTOU defense: must be rewritten to a literal loopback IP
        assert!(
            !result.contains("localhost"),
            "must rewrite localhost to literal IP, got: {result}"
        );
        assert!(
            result.starts_with("http://127.") || result.starts_with("http://[::1]"),
            "must resolve to loopback, got: {result}"
        );
        // Case-insensitive
        assert!(check_base_url("http://LOCALHOST:8080").is_ok());
        // No port
        assert!(check_base_url("http://localhost").is_ok());
        // Path preserved after localhost → IP rewrite
        let with_path = check_base_url("http://localhost:8080/brave").unwrap();
        assert!(
            with_path.ends_with(":8080/brave"),
            "path must be preserved, got: {with_path}"
        );
    }

    #[test]
    fn check_base_url_port_boundaries() {
        assert!(check_base_url("http://127.0.0.1:1").is_ok());
        assert!(check_base_url("http://127.0.0.1:65535").is_ok());
        assert!(check_base_url("http://127.0.0.1:0").is_err()); // port 0
        assert!(check_base_url("http://127.0.0.1:65536").is_err()); // u16 overflow
        assert!(check_base_url("http://127.0.0.1:99999").is_err()); // u16 overflow
        assert!(check_base_url("http://127.0.0.1:").is_err()); // empty port
        assert!(check_base_url("http://[::1]:").is_err()); // empty port (IPv6)
    }

    #[test]
    fn check_base_url_rejects_non_loopback_ips() {
        assert!(check_base_url("http://192.168.1.1:8080").is_err()); // RFC 1918 private
        assert!(check_base_url("http://10.0.0.1:8080").is_err()); // RFC 1918 private
        assert!(check_base_url("http://0.0.0.0:8080").is_err()); // unspecified ≠ loopback
        assert!(check_base_url("http://[::2]:8080").is_err()); // non-loopback IPv6
        assert!(check_base_url("http://169.254.169.254:80").is_err()); // link-local / cloud metadata
        assert!(check_base_url("http://[fe80::1]:8080").is_err()); // link-local IPv6
    }

    #[test]
    fn check_base_url_rejects_ssrf_bypass_attempts() {
        // Octal notation — Rust's strict Ipv4Addr parser per RFC 6943 rejects this
        assert!(check_base_url("http://0177.0.0.1:8080").is_err());
        // Decimal IP (2130706433 = 127.0.0.1) — not a valid Ipv4Addr, not "localhost"
        assert!(check_base_url("http://2130706433:8080").is_err());
        // Shorthand (127.1) — Rust rejects non-four-octet forms
        assert!(check_base_url("http://127.1:8080").is_err());
        // Hex IP — Rust rejects, not "localhost"
        assert!(check_base_url("http://0x7f000001:8080").is_err());
        // IPv4-mapped IPv6 — Ipv6Addr::is_loopback() returns false (rust-lang/rust#69772)
        assert!(check_base_url("http://[::ffff:127.0.0.1]:8080").is_err());
        assert!(check_base_url("http://[::ffff:7f00:1]:8080").is_err());
        // Userinfo smuggling — @ makes the part after it the real host
        assert!(check_base_url("http://user:pass@127.0.0.1:8080").is_err());
        assert!(check_base_url("http://127.0.0.1@evil.com:8080").is_err());
        // DNS service bypass — not "localhost", not a valid IP literal
        assert!(check_base_url("http://127.0.0.1.nip.io:8080").is_err());
        // Percent-encoded IP — code never decodes, raw string fails both parsers
        assert!(check_base_url("http://%31%32%37.0.0.1:8080").is_err());
        // Unicode confusable 'l' (U+217C) — eq_ignore_ascii_case is ASCII-only
        assert!(check_base_url("http://\u{217C}ocalhost:8080").is_err());
    }

    #[test]
    fn check_base_url_rejects_bad_scheme_or_structure() {
        // https for localhost — must use http://
        assert!(check_base_url("https://localhost:8080").is_err());
        assert!(check_base_url("https://127.0.0.1:8080").is_err());
        assert!(check_base_url("ftp://127.0.0.1:8080").is_err());
        assert!(check_base_url("127.0.0.1:8080").is_err()); // no scheme
        assert!(check_base_url("http://").is_err()); // empty host
        assert!(check_base_url("not-a-url").is_err());
        assert!(check_base_url("").is_err());
        assert!(check_base_url("https://evil.com").is_err()); // not in allowlist
        assert!(check_base_url("http:///127.0.0.1:8080").is_err()); // triple slash → empty host
        assert!(check_base_url("http://127.0.0.1:8080?foo=bar").is_err()); // query contaminates port
        assert!(check_base_url("http://127.0.0.1:8080#frag").is_err()); // fragment contaminates port
    }

    #[test]
    fn check_base_url_rejects_malformed_ipv6() {
        assert!(check_base_url("http://[::1").is_err()); // missing ]
        assert!(check_base_url("http://[::1]garbage:8080").is_err()); // junk after bracket
        assert!(check_base_url("http://[::1%25eth0]:8080").is_err()); // zone ID
        assert!(check_base_url("http://[]:8080").is_err()); // empty brackets
    }

    #[test]
    fn check_endpoint_accepts_valid_paths() {
        assert!(check_endpoint("/res/v1/web/search").is_ok());
        assert!(check_endpoint("/res/v1/llm/context").is_ok());
        assert!(check_endpoint("/beta/v2/new-endpoint").is_ok());
        assert!(check_endpoint("/").is_ok());
        assert!(check_endpoint("/res/v1/").is_ok()); // trailing slash
        assert!(check_endpoint("/res/v1/foo..bar").is_ok()); // ".." as substring
    }

    #[test]
    fn check_endpoint_rejects_missing_slash() {
        assert!(check_endpoint("res/v1/web/search").is_err());
    }

    #[test]
    fn check_endpoint_rejects_disallowed_chars() {
        assert!(check_endpoint("/foo?bar=1").is_err()); // query string
        assert!(check_endpoint("/foo#frag").is_err()); // fragment
        assert!(check_endpoint("/foo@bar").is_err()); // authority
        assert!(check_endpoint("/foo\\bar").is_err()); // backslash
        assert!(check_endpoint("/foo%2e").is_err()); // percent encoding
        assert!(check_endpoint("/foo;bar").is_err()); // semicolon
        assert!(check_endpoint("/foo bar").is_err()); // space
        assert!(check_endpoint("/foo\nbar").is_err()); // newline
        assert!(check_endpoint("/foo\rbar").is_err()); // carriage return
        assert!(check_endpoint("/foo\0bar").is_err()); // null
    }

    #[test]
    fn check_endpoint_rejects_path_traversal() {
        assert!(check_endpoint("/../admin").is_err());
        assert!(check_endpoint("/res/../admin").is_err());
        assert!(check_endpoint("/res/v1/..").is_err());
    }

    #[test]
    fn check_endpoint_rejects_consecutive_slashes() {
        assert!(check_endpoint("//evil.com").is_err());
        assert!(check_endpoint("/res//v1").is_err());
    }

    #[test]
    fn check_endpoint_single_dot_segment_allowed() {
        assert!(check_endpoint("/res/./v1").is_ok()); // `.` is harmless (current dir)
    }

    #[test]
    fn check_endpoint_rejects_empty_and_non_ascii() {
        assert!(check_endpoint("").is_err()); // no leading `/`
        assert!(check_endpoint("/res/caf\u{00e9}").is_err()); // non-ASCII byte rejected
    }

    #[test]
    fn site_goggles_allowlist() {
        let result = build_site_goggles(&["docs.rs".into(), "github.com".into()], &[]);
        assert_eq!(
            result.unwrap(),
            "$discard\n$boost,site=docs.rs\n$boost,site=github.com"
        );
    }

    #[test]
    fn site_goggles_blocklist() {
        let result = build_site_goggles(&[], &["w3schools.com".into(), "medium.com".into()]);
        assert_eq!(
            result.unwrap(),
            "$discard,site=w3schools.com\n$discard,site=medium.com"
        );
    }

    #[test]
    fn site_goggles_empty() {
        assert!(build_site_goggles(&[], &[]).is_none());
    }

    #[test]
    fn site_goggles_single() {
        let result = build_site_goggles(&["docs.rs".into()], &[]);
        assert_eq!(result.unwrap(), "$discard\n$boost,site=docs.rs");
    }

    #[test]
    fn warn_goggles_shell_expansion_triggers() {
        // Smoke test: should not panic (warns on stderr)
        warn_shell_expanded_goggles("=example.org");
        warn_shell_expanded_goggles(",site=example.org");
    }

    #[test]
    fn warn_goggles_shell_expansion_skips() {
        // Should not warn (and not panic)
        warn_shell_expanded_goggles("$site=example.org");
        warn_shell_expanded_goggles("$boost=3,site=docs.rs");
        warn_shell_expanded_goggles("@rules.goggle");
        warn_shell_expanded_goggles("https://example.com/goggle");
        warn_shell_expanded_goggles("");
    }

    #[test]
    fn validate_domain_accepts_valid() {
        assert!(validate_domain("docs.rs").is_ok());
        assert!(validate_domain("my-site.example.com").is_ok());
        assert!(validate_domain("under_score.io").is_ok());
    }

    #[test]
    fn validate_domain_rejects_invalid() {
        assert!(validate_domain("").is_err());
        assert!(validate_domain("bad domain").is_err());
        assert!(validate_domain("evil!.com").is_err());
        assert!(validate_domain("path/slash").is_err());
        // Goggles syntax characters must be rejected to prevent injection
        assert!(validate_domain("$inject").is_err());
        assert!(validate_domain("a,b=c").is_err());
        assert!(validate_domain("a\nb").is_err());
    }

    // ── inject_default_subcommand_impl tests ────────────────────────

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn inject_normal_query() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx rust-error")),
            args("bx context rust-error")
        );
    }

    #[test]
    fn inject_known_subcommand_no_change() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx web query")),
            args("bx web query")
        );
    }

    #[test]
    fn inject_double_dash_inserts_context() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx -- web")),
            args("bx context -- web")
        );
    }

    #[test]
    fn inject_double_dash_with_normal_query() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx -- some-query")),
            args("bx context -- some-query")
        );
    }

    #[test]
    fn inject_skips_global_value_flags() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --api-key KEY query")),
            args("bx --api-key KEY context query")
        );
    }

    #[test]
    fn inject_help_no_change() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --help")),
            args("bx --help")
        );
    }

    #[test]
    fn inject_double_dash_alone() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --")),
            args("bx context --")
        );
    }

    #[test]
    fn inject_no_args() {
        assert_eq!(inject_default_subcommand_impl(args("bx")), args("bx"));
    }

    #[test]
    fn inject_subcommand_alone() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx web")),
            args("bx web")
        );
    }

    #[test]
    fn inject_skips_multiple_value_flags() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --api-key KEY --timeout 30 query")),
            args("bx --api-key KEY --timeout 30 context query")
        );
    }

    #[test]
    fn inject_value_flag_then_double_dash() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --timeout 30 -- web")),
            args("bx --timeout 30 context -- web")
        );
    }

    #[test]
    fn inject_value_flag_at_end() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --timeout")),
            args("bx --timeout")
        );
    }

    #[test]
    fn inject_unknown_flag_before_query() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --verbose query")),
            args("bx --verbose context query")
        );
    }

    #[test]
    fn inject_equals_form_flag() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --api-key=KEY query")),
            args("bx --api-key=KEY context query")
        );
    }

    // ── inject_default_subcommand_impl with --config ───────────────

    #[test]
    fn inject_skips_config_flag() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --config /tmp/c.json myquery")),
            args("bx --config /tmp/c.json context myquery")
        );
    }

    #[test]
    fn inject_config_equals_form() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --config=/tmp/c.json myquery")),
            args("bx --config=/tmp/c.json context myquery")
        );
    }

    #[test]
    fn inject_config_with_subcommand() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --config /tmp/c.json web myquery")),
            args("bx --config /tmp/c.json web myquery")
        );
    }

    #[test]
    fn inject_config_with_other_flags() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --config /tmp/c.json --timeout 60 myquery")),
            args("bx --config /tmp/c.json --timeout 60 context myquery")
        );
    }

    #[test]
    fn inject_config_dangling_at_end() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --config")),
            args("bx --config")
        );
    }

    #[test]
    fn inject_config_before_double_dash() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --config /tmp/c.json -- web")),
            args("bx --config /tmp/c.json context -- web")
        );
    }

    // ── unescape_inline_newlines tests ──────────────────────────────

    #[test]
    fn unescape_newline() {
        assert_eq!(unescape_inline_newlines("a\\nb"), "a\nb");
    }

    #[test]
    fn unescape_double_backslash() {
        assert_eq!(unescape_inline_newlines("a\\\\nb"), "a\\nb");
    }

    #[test]
    fn unescape_noop() {
        assert_eq!(unescape_inline_newlines("plain text"), "plain text");
    }

    #[test]
    fn unescape_trailing_backslash() {
        assert_eq!(unescape_inline_newlines("end\\"), "end\\");
    }

    #[test]
    fn unescape_other_escape() {
        assert_eq!(unescape_inline_newlines("a\\tb"), "a\\tb");
    }

    #[test]
    fn unescape_empty() {
        assert_eq!(unescape_inline_newlines(""), "");
    }

    #[test]
    fn unescape_only_newline() {
        assert_eq!(unescape_inline_newlines("\\n"), "\n");
    }

    #[test]
    fn unescape_newline_at_start() {
        assert_eq!(unescape_inline_newlines("\\ntext"), "\ntext");
    }

    #[test]
    fn unescape_newline_at_end() {
        assert_eq!(unescape_inline_newlines("text\\n"), "text\n");
    }

    #[test]
    fn unescape_consecutive_newlines() {
        assert_eq!(unescape_inline_newlines("a\\n\\nb"), "a\n\nb");
    }

    #[test]
    fn unescape_mixed() {
        assert_eq!(unescape_inline_newlines("a\\nb\\\\c\\nd"), "a\nb\\c\nd");
    }

    #[test]
    fn unescape_only_backslash() {
        assert_eq!(unescape_inline_newlines("\\"), "\\");
    }

    #[test]
    fn unescape_triple_backslash() {
        assert_eq!(unescape_inline_newlines("\\\\\\"), "\\\\");
    }

    #[test]
    fn unescape_triple_backslash_n() {
        assert_eq!(unescape_inline_newlines("\\\\\\n"), "\\\n");
    }

    #[test]
    fn unescape_quadruple_backslash() {
        assert_eq!(unescape_inline_newlines("\\\\\\\\"), "\\\\");
    }

    // ── parse_extra ──────────────────────────────────────────────────

    #[test]
    fn parse_extra_basic() {
        assert_eq!(parse_extra(&["key=value".into()]), vec![("key", "value")]);
    }

    #[test]
    fn parse_extra_value_with_equals() {
        assert_eq!(parse_extra(&["k=a=b".into()]), vec![("k", "a=b")]);
    }

    #[test]
    fn parse_extra_empty_value() {
        assert_eq!(parse_extra(&["key=".into()]), vec![("key", "")]);
    }

    #[test]
    fn parse_extra_multiple() {
        assert_eq!(
            parse_extra(&["a=1".into(), "b=2".into()]),
            vec![("a", "1"), ("b", "2")]
        );
    }

    #[test]
    fn parse_extra_empty_input() {
        let empty: Vec<String> = vec![];
        assert!(parse_extra(&empty).is_empty());
    }

    // ── inject_default_subcommand_impl with --extra / --endpoint ─────

    #[test]
    fn inject_extra_before_subcommand() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --extra foo=bar web query")),
            args("bx --extra foo=bar web query")
        );
    }

    #[test]
    fn inject_extra_before_query() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --extra foo=bar query")),
            args("bx --extra foo=bar context query")
        );
    }

    #[test]
    fn inject_extra_equals_form() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --extra=foo=bar query")),
            args("bx --extra=foo=bar context query")
        );
    }

    #[test]
    fn inject_multiple_extras() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --extra a=1 --extra b=2 query")),
            args("bx --extra a=1 --extra b=2 context query")
        );
    }

    #[test]
    fn inject_endpoint_before_subcommand() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --endpoint /custom web query")),
            args("bx --endpoint /custom web query")
        );
    }

    #[test]
    fn inject_endpoint_before_query() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --endpoint /custom query")),
            args("bx --endpoint /custom context query")
        );
    }

    #[test]
    fn inject_extra_at_end() {
        // --extra with no value: loop ends without panic, clap will error later
        assert_eq!(
            inject_default_subcommand_impl(args("bx --extra")),
            args("bx --extra")
        );
    }

    #[test]
    fn inject_endpoint_equals_form() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --endpoint=/custom query")),
            args("bx --endpoint=/custom context query")
        );
    }

    #[test]
    fn inject_extra_and_endpoint_combined() {
        assert_eq!(
            inject_default_subcommand_impl(args("bx --extra a=1 --endpoint /p query")),
            args("bx --extra a=1 --endpoint /p context query")
        );
    }

    #[test]
    fn inject_extra_value_looks_like_subcommand() {
        // --extra consumes "web=1" as its value, not as a subcommand
        assert_eq!(
            inject_default_subcommand_impl(args("bx --extra web=1 query")),
            args("bx --extra web=1 context query")
        );
    }
}
