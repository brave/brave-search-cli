mod api;
mod config;

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
/// Quick start:
///   bx config set-key <YOUR_KEY>
///   bx "tokio spawn async task example" # RAG grounding (= bx context)
///   bx answers "how does Rust's borrow checker work?" # AI answer
///   bx web "site:docs.rs reqwest" | jq . # web search
#[derive(Parser)]
#[command(name = "bx", version, verbatim_doc_comment)]
struct Cli {
    /// API key (prefer env var or config file — command-line flags are visible in process listings)
    #[arg(
        long,
        env = "BRAVE_SEARCH_API_KEY",
        global = true,
        hide_env_values = true
    )]
    api_key: Option<String>,

    /// Base URL for the API
    #[arg(
        long,
        env = "BRAVE_SEARCH_BASE_URL",
        default_value = "https://api.search.brave.com",
        global = true,
        hide_env_values = true
    )]
    base_url: String,

    /// Request timeout in seconds (default: 30)
    #[arg(long, global = true, default_value_t = 30)]
    timeout: u64,

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
    ///   bx "how to implement retry with exponential backoff" --max-tokens 2048
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
    ///
    /// Output: .results[] — array of {title, postal_address, contact}
    ///
    /// Examples:
    ///   bx places --location "San Francisco CA US" -q "coffee"
    ///   bx places --latitude 37.7749 --longitude -122.4194 -q "pizza"
    ///   bx places --location "NYC" -q "museums" | jq '.results[].title'
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

    /// Manage local API key — set-key, show-key, path
    ///
    /// Stores the API key in ~/.config/brave-search/api_key (Linux),
    /// ~/Library/Application Support/brave-search/api_key (macOS),
    /// %APPDATA%\brave-search\api_key (Windows).
    ///
    /// Examples:
    ///   bx config set-key <KEY>
    ///   bx config show-key
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
}

// ── Subcommand args ──────────────────────────────────────────────────

#[derive(Args)]
struct GogglesArgs {
    /// Goggles: custom re-ranking rules — boost, downrank, or discard results.
    /// Target by domain ($site=) or URL path pattern (/docs/$boost=3).
    /// Actions: $boost=N (1-10), $downrank=N (1-10), $discard. Combine with commas.
    /// Inline:  --goggles '$boost=3,site=docs.python.org'
    /// File:    --goggles @rules.goggle  (reads local file, ideal for agents)
    /// Stdin:   --goggles @-  (reads from stdin)
    /// Hosted:  --goggles 'https://raw.githubusercontent.com/.../my.goggle'
    /// Unique to Brave — no other search API offers custom re-ranking.
    /// Mutually exclusive with --include-site / --exclude-site.
    /// Ref: https://github.com/brave/goggles-quickstart
    #[arg(long, verbatim_doc_comment)]
    goggles: Option<String>,

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
    #[arg(long, default_value = "US")]
    country: String,

    /// Search language (e.g. en, fr, de)
    #[arg(long, default_value = "en")]
    search_lang: String,

    /// UI language (e.g. en-US, fr-FR)
    #[arg(long, default_value = "en-US")]
    ui_lang: String,

    /// Number of results (1-20)
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u16).range(1..=20))]
    count: u16,

    /// Result offset (0-9)
    #[arg(long, value_parser = clap::value_parser!(u16).range(0..=9))]
    offset: Option<u16>,

    /// Safe search: off, moderate, strict
    #[arg(long, default_value = "moderate", value_parser = ["off", "moderate", "strict"])]
    safesearch: String,

    /// Freshness: pd (past day), pw (past week), pm (past month), py (past year), or YYYY-MM-DDtoYYYY-MM-DD
    #[arg(long)]
    freshness: Option<String>,

    /// Enable text decorations (bold markers in snippets)
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    text_decorations: Option<String>,

    /// Enable spellcheck
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    spellcheck: Option<String>,

    /// Comma-separated result types: discussions,faq,infobox,news,query,summarizer,videos,web,locations
    #[arg(long)]
    result_filter: Option<String>,

    #[command(flatten)]
    goggles_args: GogglesArgs,

    /// Return extra snippets from different parts of the page
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    extra_snippets: Option<String>,

    /// Units: metric or imperial
    #[arg(long, value_parser = ["metric", "imperial"])]
    units: Option<String>,

    /// Enable search operators
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    operators: Option<String>,

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
    #[arg(long, default_value = "US")]
    country: String,

    /// Search language
    #[arg(long, default_value = "en")]
    search_lang: String,

    /// Number of results (1-200)
    #[arg(long, default_value_t = 50, value_parser = clap::value_parser!(u16).range(1..=200))]
    count: u16,

    /// Safe search: off or strict
    #[arg(long, default_value = "strict", value_parser = ["off", "strict"])]
    safesearch: String,

    /// Enable spellcheck
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    spellcheck: Option<String>,
}

#[derive(Parser)]
struct VideosArgs {
    /// Search query
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    q: String,

    /// Country code
    #[arg(long, default_value = "US")]
    country: String,

    /// Search language
    #[arg(long, default_value = "en")]
    search_lang: String,

    /// UI language
    #[arg(long, default_value = "en-US")]
    ui_lang: String,

    /// Number of results (1-50)
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u16).range(1..=50))]
    count: u16,

    /// Result offset (0-9)
    #[arg(long, value_parser = clap::value_parser!(u16).range(0..=9))]
    offset: Option<u16>,

    /// Safe search: off, moderate, strict
    #[arg(long, default_value = "moderate", value_parser = ["off", "moderate", "strict"])]
    safesearch: String,

    /// Freshness filter
    #[arg(long)]
    freshness: Option<String>,

    /// Enable spellcheck
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    spellcheck: Option<String>,

    /// Enable search operators
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    operators: Option<String>,
}

#[derive(Parser)]
struct NewsArgs {
    /// Search query
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    q: String,

    /// Country code
    #[arg(long, default_value = "US")]
    country: String,

    /// Search language
    #[arg(long, default_value = "en")]
    search_lang: String,

    /// UI language
    #[arg(long, default_value = "en-US")]
    ui_lang: String,

    /// Number of results (1-50)
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u16).range(1..=50))]
    count: u16,

    /// Result offset (0-9)
    #[arg(long, value_parser = clap::value_parser!(u16).range(0..=9))]
    offset: Option<u16>,

    /// Safe search: off, moderate, strict
    #[arg(long, default_value = "strict", value_parser = ["off", "moderate", "strict"])]
    safesearch: String,

    /// Freshness filter
    #[arg(long)]
    freshness: Option<String>,

    /// Enable spellcheck
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    spellcheck: Option<String>,

    /// Return extra snippets
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    extra_snippets: Option<String>,

    #[command(flatten)]
    goggles_args: GogglesArgs,

    /// Enable search operators
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    operators: Option<String>,
}

#[derive(Parser)]
struct SuggestArgs {
    /// Search query
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    q: String,

    /// Language
    #[arg(long, default_value = "en")]
    lang: String,

    /// Country code
    #[arg(long, default_value = "US")]
    country: String,

    /// Number of suggestions (1-20)
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u16).range(1..=20))]
    count: u16,

    /// Enable rich suggestions
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    rich: Option<String>,
}

#[derive(Parser)]
struct SpellcheckArgs {
    /// Query to spell-check
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    q: String,

    /// Language
    #[arg(long, default_value = "en")]
    lang: String,

    /// Country code
    #[arg(long, default_value = "US")]
    country: String,
}

#[derive(Parser)]
struct AnswersArgs {
    /// Question to ask, or "-" to read JSON body from stdin
    #[arg(value_parser = clap::builder::NonEmptyStringValueParser::new())]
    query: String,

    /// Model: brave-pro or brave
    #[arg(long, default_value = "brave-pro", value_parser = ["brave-pro", "brave"])]
    model: String,

    /// Disable streaming (default: stream enabled)
    #[arg(long)]
    no_stream: bool,

    /// Country code
    #[arg(long, default_value = "US")]
    country: String,

    /// Language
    #[arg(long, default_value = "en")]
    language: String,

    /// Safe search: off, moderate, strict
    #[arg(long, default_value = "moderate", value_parser = ["off", "moderate", "strict"])]
    safesearch: String,

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

    /// Max tokens per research query (1024-16384)
    #[arg(long)]
    research_max_tokens_per_query: Option<u32>,

    /// Max research queries (1-50)
    #[arg(long)]
    research_max_queries: Option<u32>,

    /// Max research iterations (1-5)
    #[arg(long)]
    research_max_iterations: Option<u32>,

    /// Max research seconds (1-300)
    #[arg(long)]
    research_max_seconds: Option<u32>,

    /// Max results per research query (1-60)
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
    #[arg(long, default_value = "US")]
    country: String,

    /// Search language
    #[arg(long, default_value = "en")]
    search_lang: String,

    /// Number of results (1-50)
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u16).range(1..=50))]
    count: u16,

    /// Max URLs to include (1-50)
    #[arg(long, visible_alias = "max-urls")]
    maximum_number_of_urls: Option<String>,

    /// Max total tokens (1024-32768)
    #[arg(long, visible_alias = "max-tokens")]
    maximum_number_of_tokens: Option<String>,

    /// Max snippets (1-100)
    #[arg(long, visible_alias = "max-snippets")]
    maximum_number_of_snippets: Option<String>,

    /// Max tokens per URL (512-8192)
    #[arg(long, visible_alias = "max-tokens-per-url")]
    maximum_number_of_tokens_per_url: Option<String>,

    /// Max snippets per URL (1-100)
    #[arg(long, visible_alias = "max-snippets-per-url")]
    maximum_number_of_snippets_per_url: Option<String>,

    /// Threshold mode: strict, balanced, lenient
    #[arg(long, visible_alias = "threshold", value_parser = ["strict", "balanced", "lenient"])]
    context_threshold_mode: Option<String>,

    #[command(flatten)]
    goggles_args: GogglesArgs,

    /// Enable local results
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    enable_local: Option<String>,

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
    /// Search query
    #[arg(long, short)]
    q: Option<String>,

    /// Latitude (-90 to 90)
    #[arg(long, allow_hyphen_values = true)]
    latitude: Option<String>,

    /// Longitude (-180 to 180)
    #[arg(long, requires = "latitude", allow_hyphen_values = true)]
    longitude: Option<String>,

    /// Location string (alternative to lat/long, e.g. "San Francisco CA US")
    #[arg(long)]
    location: Option<String>,

    /// Search radius in meters (0-20000)
    #[arg(long)]
    radius: Option<String>,

    /// Number of results (1-50)
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u16).range(1..=50))]
    count: u16,

    /// Country code
    #[arg(long, default_value = "US")]
    country: String,

    /// Search language
    #[arg(long, default_value = "en")]
    search_lang: String,

    /// UI language
    #[arg(long, default_value = "en-US")]
    ui_lang: String,

    /// Units: metric or imperial
    #[arg(long, default_value = "metric", value_parser = ["metric", "imperial"])]
    units: String,

    /// Safe search
    #[arg(long, default_value = "strict", value_parser = ["off", "moderate", "strict"])]
    safesearch: String,

    /// Enable spellcheck
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    spellcheck: Option<String>,
}

#[derive(Parser)]
struct PoisArgs {
    /// POI IDs (1-20)
    ids: Vec<String>,

    /// Search language
    #[arg(long, default_value = "en")]
    search_lang: String,

    /// UI language
    #[arg(long, default_value = "en-US")]
    ui_lang: String,

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
    /// POI IDs (1-20)
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
    "config",
    "help",
];

/// Injects "context" as the default subcommand when the first positional
/// argument is not a known subcommand (e.g. `bx "query"` → `bx context "query"`).
fn inject_default_subcommand() -> Vec<String> {
    // Safety: args[0] is not used for security decisions — we skip it (i = 1) and only
    // inspect subsequent args for subcommand routing. CWE-807 does not apply here.
    let args: Vec<String> = std::env::args().collect(); // nosemgrep: rust.lang.security.args.args

    // Flags that consume the next argument as a value
    const VALUE_FLAGS: &[&str] = &["--api-key", "--base-url", "--timeout"];

    let mut i = 1; // skip binary name
    while i < args.len() {
        let arg = &args[i];

        if arg == "--" {
            break;
        }

        if arg.starts_with('-') {
            // Check if this flag consumes the next arg
            if VALUE_FLAGS.contains(&arg.as_str()) {
                i += 2; // skip flag and its value
                continue;
            }
            // --flag=value or boolean flag: just skip
            i += 1;
            continue;
        }

        // First positional argument found
        if !SUBCOMMANDS.contains(&arg.as_str()) {
            let mut new_args = args.clone();
            new_args.insert(i, "context".to_string());
            return new_args;
        }

        return args; // known subcommand, no injection
    }

    args // no positional found (e.g. `bx --help`)
}

fn main() {
    let cli = Cli::parse_from(inject_default_subcommand());

    // Config subcommand doesn't need an API key.
    if let Command::Config { ref cmd } = cli.command {
        config::handle_config(cmd);
        return;
    }

    let api_key = resolve_api_key(&cli);
    let base = &cli.base_url;
    validate_base_url(base);
    let timeout = cli.timeout;

    match cli.command {
        Command::Context(args) => cmd_context(base, &api_key, args, timeout),
        Command::Answers(args) => cmd_answers(base, &api_key, args, timeout),
        Command::Web(args) => cmd_web(base, &api_key, args, timeout),
        Command::News(args) => cmd_news(base, &api_key, args, timeout),
        Command::Images(args) => cmd_images(base, &api_key, args, timeout),
        Command::Videos(args) => cmd_videos(base, &api_key, args, timeout),
        Command::Places(args) => cmd_places(base, &api_key, args, timeout),
        Command::Suggest(args) => cmd_suggest(base, &api_key, args, timeout),
        Command::Spellcheck(args) => cmd_spellcheck(base, &api_key, args, timeout),
        Command::Pois(args) => cmd_pois(base, &api_key, args, timeout),
        Command::Descriptions(args) => cmd_descriptions(base, &api_key, args, timeout),
        Command::Config { .. } => unreachable!(),
    }
}

/// Allowed base URLs for the Brave Search API.
const ALLOWED_BASE_URLS: &[&str] = &[
    "https://api.search.brave.com",
    "https://api.search.brave.software",
];

fn validate_base_url(url: &str) {
    let normalized = url.trim_end_matches('/');
    if !ALLOWED_BASE_URLS.contains(&normalized) {
        eprintln!(
            "error: base URL not in allowlist (got: {url})\n\
             hint: allowed URLs are {}",
            ALLOWED_BASE_URLS.join(", ")
        );
        std::process::exit(1);
    }
}

fn resolve_api_key(cli: &Cli) -> String {
    // 1. --api-key flag / BRAVE_SEARCH_API_KEY env (handled by clap)
    if let Some(ref key) = cli.api_key {
        return key.clone();
    }

    // 2. Config file
    if let Some(key) = config::load_api_key() {
        return key;
    }

    // 3. Interactive onboarding
    match config::onboard() {
        Ok(key) => key,
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
        self.goggles
            .as_deref()
            .map(resolve_goggles)
            .or_else(|| build_site_goggles(&self.include_site, &self.exclude_site))
    }
}

/// Maximum size for file/stdin reads (goggles, answers stdin JSON).
const MAX_INPUT_SIZE: u64 = 1024 * 1024; // 1 MB

/// Resolves a --goggles value:
///   @-       → read from stdin
///   @path    → read from file
///   other    → return as-is (inline rules or hosted URL)
fn resolve_goggles(value: &str) -> String {
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
            buf
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
                Ok(contents) => contents,
                Err(e) => {
                    eprintln!("error: failed to read goggles file '{path}': {e}");
                    std::process::exit(1);
                }
            }
        }
    } else {
        value.to_string()
    }
}

// ── Command handlers ─────────────────────────────────────────────────

fn cmd_web(base: &str, key: &str, a: WebArgs, timeout: u64) {
    let count_str = a.count.to_string();
    let offset_str = a.offset.map(|v| v.to_string());
    let goggles_resolved = a.goggles_args.resolve();
    let qs = api::build_query(&[
        ("q", Some(a.q.as_str())),
        ("country", Some(a.country.as_str())),
        ("search_lang", Some(a.search_lang.as_str())),
        ("ui_lang", Some(a.ui_lang.as_str())),
        ("count", Some(&count_str)),
        ("offset", offset_str.as_deref()),
        ("safesearch", Some(a.safesearch.as_str())),
        ("freshness", a.freshness.as_deref()),
        ("text_decorations", a.text_decorations.as_deref()),
        ("spellcheck", a.spellcheck.as_deref()),
        ("result_filter", a.result_filter.as_deref()),
        ("goggles", goggles_resolved.as_deref()),
        ("extra_snippets", a.extra_snippets.as_deref()),
        ("units", a.units.as_deref()),
        ("operators", a.operators.as_deref()),
    ]);
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
    api::get_with_headers(
        base,
        &format!("/res/v1/web/search{qs}"),
        key,
        &headers,
        timeout,
    );
}

fn cmd_images(base: &str, key: &str, a: ImagesArgs, timeout: u64) {
    let count_str = a.count.to_string();
    let qs = api::build_query(&[
        ("q", Some(a.q.as_str())),
        ("country", Some(a.country.as_str())),
        ("search_lang", Some(a.search_lang.as_str())),
        ("count", Some(&count_str)),
        ("safesearch", Some(a.safesearch.as_str())),
        ("spellcheck", a.spellcheck.as_deref()),
    ]);
    api::get(base, &format!("/res/v1/images/search{qs}"), key, timeout);
}

fn cmd_videos(base: &str, key: &str, a: VideosArgs, timeout: u64) {
    let count_str = a.count.to_string();
    let offset_str = a.offset.map(|v| v.to_string());
    let qs = api::build_query(&[
        ("q", Some(a.q.as_str())),
        ("country", Some(a.country.as_str())),
        ("search_lang", Some(a.search_lang.as_str())),
        ("ui_lang", Some(a.ui_lang.as_str())),
        ("count", Some(&count_str)),
        ("offset", offset_str.as_deref()),
        ("safesearch", Some(a.safesearch.as_str())),
        ("freshness", a.freshness.as_deref()),
        ("spellcheck", a.spellcheck.as_deref()),
        ("operators", a.operators.as_deref()),
    ]);
    api::get(base, &format!("/res/v1/videos/search{qs}"), key, timeout);
}

fn cmd_news(base: &str, key: &str, a: NewsArgs, timeout: u64) {
    let count_str = a.count.to_string();
    let offset_str = a.offset.map(|v| v.to_string());
    let goggles_resolved = a.goggles_args.resolve();
    let qs = api::build_query(&[
        ("q", Some(a.q.as_str())),
        ("country", Some(a.country.as_str())),
        ("search_lang", Some(a.search_lang.as_str())),
        ("ui_lang", Some(a.ui_lang.as_str())),
        ("count", Some(&count_str)),
        ("offset", offset_str.as_deref()),
        ("safesearch", Some(a.safesearch.as_str())),
        ("freshness", a.freshness.as_deref()),
        ("spellcheck", a.spellcheck.as_deref()),
        ("extra_snippets", a.extra_snippets.as_deref()),
        ("goggles", goggles_resolved.as_deref()),
        ("operators", a.operators.as_deref()),
    ]);
    api::get(base, &format!("/res/v1/news/search{qs}"), key, timeout);
}

fn cmd_suggest(base: &str, key: &str, a: SuggestArgs, timeout: u64) {
    let count_str = a.count.to_string();
    let qs = api::build_query(&[
        ("q", Some(a.q.as_str())),
        ("lang", Some(a.lang.as_str())),
        ("country", Some(a.country.as_str())),
        ("count", Some(&count_str)),
        ("rich", a.rich.as_deref()),
    ]);
    api::get(base, &format!("/res/v1/suggest/search{qs}"), key, timeout);
}

fn cmd_spellcheck(base: &str, key: &str, a: SpellcheckArgs, timeout: u64) {
    let qs = api::build_query(&[
        ("q", Some(a.q.as_str())),
        ("lang", Some(a.lang.as_str())),
        ("country", Some(a.country.as_str())),
    ]);
    api::get(
        base,
        &format!("/res/v1/spellcheck/search{qs}"),
        key,
        timeout,
    );
}

fn cmd_answers(base: &str, key: &str, a: AnswersArgs, timeout: u64) {
    let path = "/res/v1/chat/completions";

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
        let body: serde_json::Value = match serde_json::from_str(&buf) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: invalid JSON on stdin: {e}");
                std::process::exit(1);
            }
        };

        let is_stream = body["stream"].as_bool().unwrap_or(true);
        if is_stream {
            api::post_json_stream(base, path, key, &body, timeout);
        } else {
            api::post_json(base, path, key, &body, timeout);
        }
        return;
    }

    // Simple mode: build JSON body from CLI args.
    let stream = !a.no_stream;
    let mut body = serde_json::json!({
        "messages": [{"role": "user", "content": a.query}],
        "model": a.model,
        "stream": stream,
        "country": a.country,
        "language": a.language,
        "safesearch": a.safesearch,
    });

    let obj = body.as_object_mut().expect("body must be a JSON object");

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

    if stream {
        api::post_json_stream(base, path, key, &body, timeout);
    } else {
        api::post_json(base, path, key, &body, timeout);
    }
}

fn cmd_context(base: &str, key: &str, a: ContextArgs, timeout: u64) {
    let count_str = a.count.to_string();
    let goggles_resolved = a.goggles_args.resolve();
    let qs = api::build_query(&[
        ("q", Some(a.q.as_str())),
        ("country", Some(a.country.as_str())),
        ("search_lang", Some(a.search_lang.as_str())),
        ("count", Some(&count_str)),
        (
            "maximum_number_of_urls",
            a.maximum_number_of_urls.as_deref(),
        ),
        (
            "maximum_number_of_tokens",
            a.maximum_number_of_tokens.as_deref(),
        ),
        (
            "maximum_number_of_snippets",
            a.maximum_number_of_snippets.as_deref(),
        ),
        (
            "maximum_number_of_tokens_per_url",
            a.maximum_number_of_tokens_per_url.as_deref(),
        ),
        (
            "maximum_number_of_snippets_per_url",
            a.maximum_number_of_snippets_per_url.as_deref(),
        ),
        (
            "context_threshold_mode",
            a.context_threshold_mode.as_deref(),
        ),
        ("goggles", goggles_resolved.as_deref()),
        ("enable_local", a.enable_local.as_deref()),
    ]);
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
    api::get_with_headers(
        base,
        &format!("/res/v1/llm/context{qs}"),
        key,
        &headers,
        timeout,
    );
}

fn cmd_places(base: &str, key: &str, a: PlacesArgs, timeout: u64) {
    let count_str = a.count.to_string();
    let qs = api::build_query(&[
        ("q", a.q.as_deref()),
        ("latitude", a.latitude.as_deref()),
        ("longitude", a.longitude.as_deref()),
        ("location", a.location.as_deref()),
        ("radius", a.radius.as_deref()),
        ("count", Some(&count_str)),
        ("country", Some(a.country.as_str())),
        ("search_lang", Some(a.search_lang.as_str())),
        ("ui_lang", Some(a.ui_lang.as_str())),
        ("units", Some(a.units.as_str())),
        ("safesearch", Some(a.safesearch.as_str())),
        ("spellcheck", a.spellcheck.as_deref()),
    ]);
    api::get(
        base,
        &format!("/res/v1/local/place_search{qs}"),
        key,
        timeout,
    );
}

fn cmd_pois(base: &str, key: &str, a: PoisArgs, timeout: u64) {
    if a.ids.is_empty() {
        eprintln!("error: at least one POI ID is required");
        std::process::exit(1);
    }
    let qs = api::build_query_repeated(
        &[
            ("search_lang", Some(a.search_lang.as_str())),
            ("ui_lang", Some(a.ui_lang.as_str())),
            ("units", a.units.as_deref()),
        ],
        &[("ids", &a.ids)],
    );
    let mut headers = Vec::new();
    if let Some(ref v) = a.lat {
        validate_header_value("X-Loc-Lat", v);
        headers.push(("X-Loc-Lat", v.as_str()));
    }
    if let Some(ref v) = a.long {
        validate_header_value("X-Loc-Long", v);
        headers.push(("X-Loc-Long", v.as_str()));
    }
    api::get_with_headers(
        base,
        &format!("/res/v1/local/pois{qs}"),
        key,
        &headers,
        timeout,
    );
}

fn cmd_descriptions(base: &str, key: &str, a: DescriptionsArgs, timeout: u64) {
    if a.ids.is_empty() {
        eprintln!("error: at least one POI ID is required");
        std::process::exit(1);
    }
    let qs = api::build_query_repeated(&[], &[("ids", &a.ids)]);
    api::get(
        base,
        &format!("/res/v1/local/descriptions{qs}"),
        key,
        timeout,
    );
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
}
