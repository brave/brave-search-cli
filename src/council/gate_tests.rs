//! Tests for `gate.rs`, as its child module so they reach its private items.

use super::*;

/// A delivered member's report. Exhaustive, so a new field has to be answered for here.
fn base(text: &str) -> Report {
    Report {
        text: text.to_string(),
        usage: crate::council::types::Usage::default(),
        rounds: 5,
        retries: 0,
        searches: 3,
        brave_calls: 3,
        grounded: 3,
        search_errors: 0,
        malformed: 0,
        search_status: None,
        sources: Vec::new(),
        peak_round: 0,
        escalations: 0,
        max_tokens: 1024,
        max_tokens_reached: 1024,
        write_errors: 0,
        draft_truncated: false,
        draft_provisional: false,
        draft_reasoning_only: false,
        flushed: 0,
        draft_flushed: false,
        transcript_written: false,
        fatal: None,
        status: None,
        truncated: false,
        answered: true,
        terminated_by: "answer",
        finish: "stop",
        served_model: None,
        provider: None,
    }
}

/// `Task::default`'s `min_reply_chars`.
const FLOOR: usize = 400;

fn reason(e: &Report) -> Option<Reason> {
    match evaluate(e, FLOOR) {
        Outcome::Rejected { reason, .. } => Some(reason),
        Outcome::Delivered { .. } => None,
    }
}

#[test]
fn delivers_a_normal_review() {
    let t = "a".repeat(500);
    assert_eq!(reason(&base(&t)), None);
}

#[test]
fn ordering_puts_transport_before_shape() {
    let mut e = base("");
    e.fatal = Some("502".into());
    e.truncated = true;
    assert_eq!(reason(&e), Some(Reason::ProviderError));
}

#[test]
fn truncation_precedes_emptiness() {
    let mut e = base("");
    e.truncated = true;
    assert_eq!(reason(&e), Some(Reason::Truncated));
}

/// One reason, a lever per ceiling: `action` is "none", so the sentence must name it.
#[test]
fn a_filled_context_window_names_a_different_lever_than_a_filled_output_cap() {
    let msg = |terminated_by| {
        let mut e = base("");
        (e.truncated, e.terminated_by) = (true, terminated_by);
        match evaluate(&e, FLOOR) {
            Outcome::Rejected { reason, error, .. } => {
                assert_eq!(reason, Reason::Truncated);
                error
            }
            Outcome::Delivered { .. } => panic!("a cut-off reply is not a delivery"),
        }
    };
    // Searches cost input: the lever for a filled window, never for a filled output cap.
    let window = msg("context_window");
    assert!(
        window.contains("context window") && window.contains("--max-searches"),
        "{window}"
    );
    let cap = msg("truncated_at_cap");
    assert!(
        cap.contains("output ceiling") && cap.contains("--effort"),
        "{cap}"
    );
    // Escalation already reached the ceiling: `--max-tokens` has nowhere to go.
    assert!(!cap.contains("--max-tokens"), "{cap}");
    assert!(!cap.contains("--max-searches"), "{cap}");
    let steps = msg("max_steps");
    assert!(steps.contains("--max-steps"), "{steps}");
}

/// A scratchpad is long enough to clear any floor, so the channel decides, ahead of every
/// content check.
#[test]
fn thinking_is_not_an_answer_however_long_it_runs() {
    let mut e = base(&"a".repeat(80_496));
    (e.answered, e.terminated_by) = (false, "reasoning_only");
    assert_eq!(reason(&e), Some(Reason::NoAnswer));
    // Ahead of the search check too: the lever is the effort, not the Brave key.
    let mut errored = base(&"a".repeat(80_496));
    (errored.answered, errored.terminated_by) = (false, "reasoning_only");
    (errored.grounded, errored.search_errors) = (0, errored.searches);
    assert_eq!(reason(&errored), Some(Reason::NoAnswer));
    match evaluate(&errored, FLOOR) {
        Outcome::Rejected { error, .. } => {
            assert!(error.contains("reasoning and no answer"), "{error}")
        }
        Outcome::Delivered { .. } => panic!("published"),
    }
    match evaluate(&e, FLOOR) {
        Outcome::Rejected { error, .. } => {
            assert!(error.contains("reasoning and no answer"), "{error}");
            assert!(error.contains("--effort"), "names no lever: {error}");
        }
        Outcome::Delivered { .. } => panic!("a scratchpad was published as a review"),
    }
    // The same text on the answer channel is delivered.
    let ok = base(&"a".repeat(80_496));
    assert_eq!(reason(&ok), None);
}

/// A member whose every round was thinking, stopped by the step cap, keeps that label but still
/// names the effort lever; a merely short answer does not get it.
#[test]
fn a_scratchpad_no_bound_claimed_still_names_the_lever() {
    let mut e = base(&"a".repeat(20_496));
    (e.answered, e.terminated_by) = (false, "max_steps");
    e.draft_reasoning_only = true;
    e.rounds = 21;
    match evaluate(&e, FLOOR) {
        Outcome::Rejected { reason, error, .. } => {
            assert_eq!(reason, Reason::NoAnswer);
            assert!(error.contains("loop ended after 21 rounds"), "{error}");
            assert!(error.contains("--effort"), "names no lever: {error}");
        }
        Outcome::Delivered { .. } => panic!("a scratchpad was published as a review"),
    }
    let short = base("too short to publish");
    match evaluate(&short, FLOOR) {
        Outcome::Rejected { error, .. } => {
            assert!(error.contains("loop ended after 5 rounds"), "{error}");
            assert!(!error.contains("--effort"), "wrong lever offered: {error}");
        }
        Outcome::Delivered { .. } => panic!("published under the floor"),
    }
}

#[test]
fn zero_searches_by_choice_is_not_a_fault() {
    let t = "a".repeat(500);
    let mut e = base(&t);
    e.searches = 0;
    e.grounded = 0;
    assert_eq!(reason(&e), None);
}

/// Calls that named no query never reached Brave: a member that sent only those failed to
/// answer, not to search.
#[test]
fn a_member_that_only_sent_malformed_calls_is_no_answer_not_search_failed() {
    let mut e = base("partial");
    (e.grounded, e.search_errors, e.answered) = (0, e.searches, false);
    e.malformed = e.searches;
    assert_eq!(reason(&e), Some(Reason::NoAnswer));
}

/// A refused key fails the same on every retry, so it is not retryable and the error names the
/// key. A statusless failure still is.
#[test]
fn every_search_refused_for_the_key_names_it_and_is_not_retryable() {
    let mut e = base("");
    (e.grounded, e.search_errors, e.answered) = (0, e.searches, false);
    e.search_status = Some(401);
    match evaluate(&e, FLOOR) {
        Outcome::Rejected {
            reason,
            error,
            status,
        } => {
            assert_eq!(reason, Reason::SearchFailed);
            assert_eq!(reason.action(status), "none");
            assert!(error.contains("BRAVE_SEARCH_API_KEY"), "{error}");
        }
        Outcome::Delivered { .. } => panic!("nothing was delivered"),
    }
    e.search_status = None;
    let Outcome::Rejected { reason, status, .. } = evaluate(&e, FLOOR) else {
        panic!("nothing was delivered");
    };
    assert_eq!(reason.action(status), "retry");
}

/// A spent quota latches like a refused key: a re-run meets the same 429.
#[test]
fn a_spent_brave_quota_is_not_offered_for_retry() {
    let mut e = base("");
    (e.grounded, e.search_errors, e.answered) = (0, e.searches, false);
    e.search_status = Some(429);
    let Outcome::Rejected {
        reason,
        error,
        status,
    } = evaluate(&e, FLOOR)
    else {
        panic!("nothing was delivered");
    };
    assert!(error.contains("quota"), "{error}");
    assert_eq!(reason.action(status), "none");
}

/// Every search failing outranks emptiness, because only it names a repair.
#[test]
fn every_search_failing_outranks_emptiness_because_only_one_of_them_names_a_repair() {
    let mut all_failed = base("");
    (
        all_failed.answered,
        all_failed.search_errors,
        all_failed.grounded,
    ) = (false, 3, 0);
    assert_eq!(reason(&all_failed), Some(Reason::SearchFailed));

    // With nothing attempted, emptiness is all there is to say.
    let mut silent = base("");
    (silent.answered, silent.searches, silent.search_errors) = (false, 0, 0);
    assert_eq!(reason(&silent), Some(Reason::Empty));
}

/// Out of rounds with nothing written is not `Empty`, which says do not retry: `--max-steps`
/// helps.
#[test]
fn a_member_the_step_cap_stopped_with_nothing_written_names_max_steps_and_is_retryable() {
    let mut e = base("");
    (e.answered, e.terminated_by, e.rounds) = (false, "max_steps", 7);
    match evaluate(&e, FLOOR) {
        Outcome::Rejected { reason, error, .. } => {
            assert_eq!(reason, Reason::NoAnswer);
            assert_eq!(reason.action(None), "retry");
            assert!(
                error.contains("7 round(s)") && error.contains("--max-steps"),
                "{error}"
            );
        }
        Outcome::Delivered { .. } => panic!("nothing was delivered"),
    }
}

/// A preamble beside a search is not a review either: the step cap is still the lever.
#[test]
fn a_member_the_step_cap_stopped_holding_only_a_preamble_names_max_steps() {
    let mut e = base("I'll look up the RFC first.");
    (e.answered, e.terminated_by) = (false, "max_steps");
    let Outcome::Rejected { reason, error, .. } = evaluate(&e, FLOOR) else {
        panic!("a preamble was delivered");
    };
    assert_eq!(reason, Reason::NoAnswer);
    assert!(error.contains("raise --max-steps"), "{error}");
}

/// A full review is published even when every search errored; `partial_grounding` says so.
#[test]
fn a_complete_review_is_published_even_when_every_search_failed() {
    let t = "a".repeat(500);
    let mut e = base(&t);
    (e.grounded, e.search_errors) = (0, e.searches);
    assert_eq!(reason(&e), None, "a billed, complete review was discarded");
    assert!(matches!(
        evaluate(&e, FLOOR),
        Outcome::Delivered {
            partial_grounding: true
        }
    ));
}

/// Delivery, not content: tool-call syntax in the reply is text like any other.
#[test]
fn tool_call_markup_in_the_text_is_not_a_rejection() {
    let text = format!(
        "<tool_call>{{\"name\":\"search\"}}</tool_call>\n{}",
        "a".repeat(500)
    );
    assert_eq!(reason(&base(&text)), None);
    // A reply that is only markup still fails on length.
    assert_eq!(
        reason(&base("<tool_call>{}</tool_call>")),
        Some(Reason::NoAnswer)
    );
}

/// *Every* search, not merely one: a mix of outcomes has not shown that searching is broken.
#[test]
fn a_member_with_no_answer_and_only_some_searches_failing_is_not_a_search_failure() {
    let mut e = base("partial");
    (e.searches, e.grounded, e.search_errors, e.answered) = (3, 2, 1, false);
    assert_eq!(reason(&e), Some(Reason::NoAnswer));
    e.search_errors = 3;
    assert_eq!(reason(&e), Some(Reason::SearchFailed));
}

/// Thin evidence is a warning on a delivered review, keyed on searches that failed: an empty
/// result set is a correct answer.
#[test]
fn partial_grounding_flags_failed_searches_and_not_fruitless_ones() {
    let t = "a".repeat(500);
    let mut e = base(&t);
    (e.searches, e.grounded, e.search_errors) = (3, 1, 1);
    assert!(matches!(
        evaluate(&e, FLOOR),
        Outcome::Delivered {
            partial_grounding: true
        }
    ));
    (e.grounded, e.search_errors) = (0, 0);
    assert!(matches!(
        evaluate(&e, FLOOR),
        Outcome::Delivered {
            partial_grounding: false
        }
    ));
}

/// A byte floor would be three times looser for CJK.
#[test]
fn the_floor_counts_characters_not_bytes() {
    // 399 characters, 1,197 bytes: only a character floor rejects it.
    let short = "日".repeat(399);
    assert_eq!(reason(&base(&short)), Some(Reason::NoAnswer));
}

#[test]
fn the_floor_boundary_is_inclusive() {
    for (n, want) in [(399, Some(Reason::NoAnswer)), (400, None), (401, None)] {
        let t = "a".repeat(n);
        assert_eq!(reason(&base(&t)), want, "at {n} chars");
    }
}

/// The floor counts the trimmed text, as emptiness does.
#[test]
fn padding_does_not_carry_a_reply_over_the_floor() {
    let padded = format!("x{}", " ".repeat(399));
    assert_eq!(reason(&base(&padded)), Some(Reason::NoAnswer));
    let Outcome::Rejected { error, .. } = evaluate(&base(&padded), 400) else {
        panic!("padding was delivered as a reply");
    };
    assert!(error.contains("with 1 chars"), "{error}");
}

/// Inside the text, a run of whitespace counts as one character.
#[test]
fn interior_whitespace_does_not_carry_a_reply_over_the_floor() {
    let padded = format!("x{}y", " \n\t".repeat(133));
    let Outcome::Rejected { error, .. } = evaluate(&base(&padded), FLOOR) else {
        panic!("padding was delivered as a reply");
    };
    assert!(error.contains("with 3 chars"), "{error}");
}

/// `sanitize` keeps variation selectors and the Hangul filler; neither is visible alone.
#[test]
fn invisible_characters_do_not_carry_a_reply_over_the_floor() {
    for pad in ['\u{fe0f}', '\u{3164}', '\u{e0100}'] {
        let padded = format!("x{}", pad.to_string().repeat(399));
        assert_eq!(reason(&base(&padded)), Some(Reason::NoAnswer), "{pad:?}");
    }
}

#[test]
fn actions_are_derived_in_one_place() {
    // (reason, manifest `reason`, `action` with no status), in the gate's order. `action` is what
    // an agent acts on, and a retryable reason lands the member in `hint`.
    let table = [
        (Reason::ProviderError, "provider_error", "retry"),
        // The prompt is what was declined, and a retry re-sends the prompt.
        (Reason::Refused, "refused", "none"),
        // The same flags reach the same ceiling.
        (Reason::Truncated, "truncated", "none"),
        (Reason::Budget, "budget", "retry"),
        // A re-run spends the cap again and stops in the same place.
        (Reason::Cost, "max_usd", "none"),
        (Reason::ContextBudget, "context_budget", "retry"),
        // Nothing the caller can vary makes a reply that arrived empty arrive full.
        (Reason::Empty, "empty", "none"),
        (Reason::SearchFailed, "search_failed", "retry"),
        (Reason::NoAnswer, "no_answer", "retry"),
        // A second generation cannot repair a full disk.
        (Reason::WriteFailed, "write_failed", "none"),
    ];
    for (r, name, action) in table {
        // Exhaustive, so a new variant stops this compiling until it has a row.
        match r {
            Reason::ProviderError
            | Reason::Refused
            | Reason::Truncated
            | Reason::Budget
            | Reason::Cost
            | Reason::ContextBudget
            | Reason::Empty
            | Reason::SearchFailed
            | Reason::NoAnswer
            | Reason::WriteFailed => {}
        }
        assert_eq!((r.name(), r.action(None)), (name, action), "{name}");
    }

    // `ProviderError` covers a 503 a retry may clear and a 401 it never will.
    for (status, want) in [
        (Some(400), "none"), // the request is wrong; sending it again is wrong again
        (Some(401), "none"),
        (Some(402), "none"), // out of credit
        (Some(403), "none"),
        (Some(408), "retry"),
        (Some(429), "retry"),
        (Some(500), "retry"),
        (Some(503), "retry"),
        // A transport fault: the request never reached a server.
        (None, "retry"),
    ] {
        assert_eq!(
            Reason::ProviderError.action(status),
            want,
            "HTTP {status:?}"
        );
    }
    // The statusless reasons are decided on their own and ignore it.
    assert_eq!(Reason::Budget.action(Some(401)), "retry");
    assert_eq!(Reason::Cost.action(Some(503)), "none");
}

/// The conversation budget is sized from the step count, so that is the lever to name.
#[test]
fn a_context_budget_rejection_names_max_steps() {
    let mut e = base("");
    (e.answered, e.terminated_by) = (false, "context_budget");
    let Outcome::Rejected { reason, error, .. } = evaluate(&e, FLOOR) else {
        panic!("nothing was delivered");
    };
    assert_eq!(reason, Reason::ContextBudget);
    assert!(error.contains("--max-steps"), "{error}");
}

/// A member the clock stopped is worth retrying, so the clock is judged before the content, whose
/// `Empty` says do not retry.
#[test]
fn a_member_the_budget_stopped_is_retryable_rather_than_written_off() {
    let mut e = base("");
    e.terminated_by = "budget";
    e.answered = false;
    assert_eq!(reason(&e), Some(Reason::Budget));
    assert_eq!(Reason::Budget.action(None), "retry");

    // With a draft the member did not publish, too.
    let long = "a".repeat(500);
    let mut e = base(&long);
    (e.terminated_by, e.answered) = ("budget", false);
    assert_eq!(reason(&e), Some(Reason::Budget));

    // A transport fault still outranks the clock: it names the cause.
    let mut e = base("");
    e.terminated_by = "budget";
    e.fatal = Some("502".into());
    assert_eq!(reason(&e), Some(Reason::ProviderError));
}
