//! The trust gate. Validates **delivery**, not quality: `Delivered` means something of plausible
//! size was published, never that the review is good. No deterministic check can judge that, and
//! a formatting test would be a Goodhart target.
//!
//! **The order is load-bearing**: transport signals, then shape, then substance, so a truncated
//! or errored reply is never published on the strength of its prefix.

use super::member::Report;
use super::types::{Outcome, Reason};

/// Named by both sentences that offer it, so the two cannot drift apart.
const REASONING_LEVER: &str = "lower --effort or raise --max-tokens";

/// Unicode's `Default_Ignorable_Code_Point`: invisible wherever it is not understood.
fn default_ignorable(c: char) -> bool {
    matches!(c as u32,
        0x00ad | 0x034f | 0x061c | 0x115f..=0x1160 | 0x17b4..=0x17b5 | 0x180b..=0x180f
        | 0x200b..=0x200f | 0x202a..=0x202e | 0x2060..=0x206f | 0x3164 | 0xfe00..=0xfe0f
        | 0xfeff | 0xffa0 | 0xfff0..=0xfff8 | 0x1bca0..=0x1bca3 | 0x1d173..=0x1d17a
        | 0xe0000..=0xe0fff)
}

/// Judges the loop's own report against the task's `min_reply_chars`. `e.text` must already be
/// the redacted, sanitized text that will be published: both shrink it.
pub fn evaluate(e: &Report, min_reply_chars: usize) -> Outcome {
    let rejected = |reason: Reason, error: String| Outcome::Rejected {
        reason,
        error,
        status: None,
    };

    if let Some(err) = &e.fatal {
        return Outcome::Rejected {
            reason: Reason::ProviderError,
            error: err.clone(),
            status: e.status,
        };
    }
    // With the transport signals: nothing below can turn a declined prompt into a reply.
    if e.terminated_by == "refused" {
        return Outcome::Rejected {
            reason: Reason::Refused,
            error: "provider declined to answer this prompt".into(),
            status: e.status,
        };
    }
    if e.truncated {
        // One reason, several ceilings. `action` is "none", so the sentence must name the lever
        // for this one: searches cost input, so they only help a filled context window.
        let (why, lever) = match e.terminated_by {
            "context_window" => (
                "the conversation filled the model's context window",
                "shorten the subject or lower --max-searches",
            ),
            "max_steps" => (
                "generation was still cut off on the last round it was allowed",
                "raise --max-steps or --max-tokens",
            ),
            // At the ceiling escalation climbs to: `--max-tokens` cannot go higher.
            "truncated_at_cap" => (
                "generation hit the output ceiling and stopped mid-answer",
                "lower --effort or shorten the subject",
            ),
            _ => (
                "generation hit the output ceiling and stopped mid-answer",
                REASONING_LEVER,
            ),
        };
        return rejected(Reason::Truncated, format!("{why}; {lever}"));
    }
    // Ahead of the content checks, since a scratchpad is long enough to clear any floor. Keyed on
    // the provider's channel, never on the prose.
    if e.terminated_by == "reasoning_only" {
        return rejected(
            Reason::NoAnswer,
            format!(
                "model emitted reasoning and no answer after {} round(s); {REASONING_LEVER}",
                e.rounds
            ),
        );
    }
    // Our own three stops, ahead of the content checks unless the member published a review
    // anyway: `Empty`'s "do not retry" is the wrong advice for each, and each names its lever.
    if let Some((reason, why, lever)) = match e.terminated_by {
        _ if e.answered => None,
        "budget" => Some((Reason::Budget, "--max-seconds elapsed", "")),
        "max_usd" => Some((Reason::Cost, "--max-usd reached", "")),
        // Sized from the steps and searches, not from `--max-tokens`.
        "context_budget" => Some((
            Reason::ContextBudget,
            "conversation outgrew its budget",
            "; raise --max-steps",
        )),
        _ => None,
    } {
        return rejected(reason, format!("{why} after {} round(s){lever}", e.rounds));
    }
    // Before emptiness, as the one that names a repair. Only when nothing was published and
    // every attempt errored, at least one past the model's own malformed calls: no search, or
    // one that found nothing, is not a fault, and a written review stands on `partial_grounding`.
    if !e.answered
        && e.searches > 0
        && e.search_errors == e.searches
        && e.search_errors > e.malformed
    {
        let why = match e.search_status {
            Some(429) => "Brave quota exhausted (HTTP 429)".into(),
            Some(s) => format!("Brave refused the key (HTTP {s}); check BRAVE_SEARCH_API_KEY"),
            None => "all errored".into(),
        };
        return Outcome::Rejected {
            reason: Reason::SearchFailed,
            error: format!("{} searches attempted, {why}", e.searches),
            status: e.search_status,
        };
    }
    let published = e.text.trim();
    if published.is_empty() {
        // Out of rounds, not out of words: the step cap is a lever `Empty` cannot name.
        if e.terminated_by == "max_steps" {
            return rejected(
                Reason::NoAnswer,
                format!("no text after {} round(s); raise --max-steps", e.rounds),
            );
        }
        return rejected(Reason::Empty, "no text after trim".into());
    }
    // Only what renders counts, a run of whitespace as one, so padding cannot clear the floor.
    // `sanitize` keeps some invisible characters on purpose.
    let (seen, _) =
        published
            .chars()
            .filter(|c| !default_ignorable(*c))
            .fold((0, false), |(n, gap), c| {
                let ws = c.is_whitespace();
                (n + usize::from(!(ws && gap)), ws)
            });
    if !e.answered || seen < min_reply_chars {
        // A member whose every round was thinking can arrive here under the bound that stopped
        // it; the sentence still names the lever.
        let lever = if e.draft_reasoning_only {
            format!("; every round was thinking, never an answer — {REASONING_LEVER}")
        } else if e.terminated_by == "max_steps" {
            "; raise --max-steps".into()
        } else {
            String::new()
        };
        return rejected(
            Reason::NoAnswer,
            format!(
                "loop ended after {} rounds with {seen} chars{lever}",
                e.rounds
            ),
        );
    }
    Outcome::Delivered {
        // On what failed, not on what found nothing: an empty result is a correct answer.
        partial_grounding: e.search_errors > 0,
    }
}

#[cfg(test)]
#[path = "gate_tests.rs"]
mod tests;
