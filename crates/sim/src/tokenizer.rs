//! The deterministic ~4-chars/token rule.
//!
//! There is no real tokenizer here on purpose. The simulator needs token
//! counts that are simple, seed-free, and exactly reproducible, so one rule
//! covers both directions: incoming text counts 4 characters per token, and
//! synthesized completions emit exactly 4 ASCII characters per token.

use moonleaf_core::protocol::Message;

/// Filler vocabulary for synthesized completions.
///
/// Every entry is exactly 4 ASCII characters — the invariant the truthful
/// `usage` story rests on, enforced by test.
const FILLER: &[&str] = &[
    "the ", "and ", "for ", "but ", "not ", "you ", "all ", "can ", "her ", "was ", "one ", "our ",
];

/// Counts the prompt tokens a request's messages amount to.
///
/// The rule: total Unicode characters (`chars`, not bytes) across all
/// `content` fields, summed first, then divided by 4, rounded up.
///
/// # Panics
///
/// Panics if the count exceeds `u32::MAX` tokens — a ~17 GB prompt, far past
/// any request body the server would realistically be handed.
#[must_use]
pub fn prompt_tokens(messages: &[Message]) -> u32 {
    let total_chars: usize = messages.iter().map(|m| m.content.chars().count()).sum();
    u32::try_from(total_chars.div_ceil(4)).expect("prompt token count fits in u32")
}

/// The `index`-th filler token of a synthesized completion.
///
/// Indexes past the vocabulary wrap around, so any completion length works.
#[must_use]
pub fn token_text(index: usize) -> &'static str {
    FILLER[index % FILLER.len()]
}

#[cfg(test)]
mod tests {
    use moonleaf_core::protocol::Role;

    use super::*;

    fn message(content: &str) -> Message {
        Message {
            role: Role::User,
            content: content.to_owned(),
        }
    }

    #[test]
    fn every_filler_entry_is_four_ascii_chars() {
        for entry in FILLER {
            assert!(entry.is_ascii(), "not ASCII: {entry:?}");
            assert_eq!(entry.len(), 4, "not 4 chars: {entry:?}");
        }
    }

    #[test]
    fn token_text_cycles_through_the_vocabulary() {
        assert_eq!(token_text(0), FILLER[0]);
        assert_eq!(token_text(1), FILLER[1]);
        assert_eq!(token_text(FILLER.len()), FILLER[0]);
        assert_eq!(token_text(FILLER.len() + 1), FILLER[1]);
    }

    #[test]
    fn four_characters_count_as_one_token() {
        assert_eq!(prompt_tokens(&[message("abcd")]), 1);
    }

    #[test]
    fn a_partial_token_rounds_up() {
        assert_eq!(prompt_tokens(&[message("abcde")]), 2);
    }

    #[test]
    fn characters_sum_across_messages_before_dividing() {
        // Two 2-char messages are one 4-char prompt, not two rounded-up ones.
        assert_eq!(prompt_tokens(&[message("ab"), message("cd")]), 1);
    }

    #[test]
    fn multi_byte_characters_count_as_single_characters() {
        // Three chars but six bytes: the rule counts chars, so one token.
        assert_eq!(prompt_tokens(&[message("ééé")]), 1);
    }

    #[test]
    fn empty_content_counts_nothing() {
        assert_eq!(prompt_tokens(&[message("")]), 0);
    }

    #[test]
    fn synthesized_completions_count_back_exactly() {
        // The truthful-usage invariant: n filler tokens measure as n tokens.
        // 30 also wraps the 12-entry vocabulary more than twice.
        let text: String = (0..30).map(token_text).collect();
        assert_eq!(prompt_tokens(&[message(&text)]), 30);
    }
}
