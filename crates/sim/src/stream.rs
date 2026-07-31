//! Planning one completion stream before any of it is emitted.
//!
//! A [`StreamPlan`] is the whole response decided up front: every chunk in
//! order, plus the delay to sleep before each one. Planning is pure — no
//! clocks, no I/O — so tests can assert exact chunks and delays; the route
//! handler only replays the plan against real time.

use std::time::Duration;

use moonleaf_core::protocol::{
    ChatCompletionChunk, ChatCompletionRequest, ChunkChoice, Delta, FinishReason, Role, Usage,
};
use rand::Rng;

use crate::MODEL_ID;
use crate::injector::{Distribution, InjectorConfig};
use crate::tokenizer;

/// A fully decided completion stream: chunks in emission order.
///
/// The terminating `[DONE]` sentinel is not part of the plan — it is SSE
/// framing, not a chunk, and the emitting layer owns it.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamPlan {
    pub chunks: Vec<PlannedChunk>,
}

/// One chunk and the pause that precedes it.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedChunk {
    /// Sleep before emitting this chunk, relative to the previous one.
    pub delay: Duration,
    pub chunk: ChatCompletionChunk,
}

impl StreamPlan {
    /// Samples a complete plan for one validated request.
    ///
    /// The emitted sequence: a role-only delta after the TTFT delay, the
    /// first content chunk immediately behind it (so client-observed TTFT is
    /// the injected TTFT), one 4-char token per content chunk with a sampled
    /// gap before each, then a `finish_reason` chunk and — only under
    /// `stream_options.include_usage` — a choice-less usage chunk, both with
    /// zero delay so bookkeeping never distorts inter-chunk timing.
    ///
    /// Draws, in order: response id, output length, TTFT, one gap per
    /// content chunk after the first. The order is part of the seeded
    /// reproducibility contract — reordering draws changes every seeded run.
    ///
    /// # Panics
    ///
    /// Panics if the injector configuration is malformed (see
    /// [`Distribution::sample`]) or a sampled delay is too large for
    /// [`Duration`]. Both are config bugs, not user errors.
    #[must_use]
    pub fn sample(
        request: &ChatCompletionRequest,
        config: InjectorConfig,
        created: u64,
        rng: &mut impl Rng,
    ) -> Self {
        let id = format!("chatcmpl-{:032x}", rng.random::<u128>());
        let sampled = token_count(config.output_tokens, rng);
        let (completion_tokens, finish_reason) = length_and_reason(request, sampled);
        let count = usize::try_from(completion_tokens).expect("completion length fits in usize");

        let mut chunks = Vec::with_capacity(count + 3);

        chunks.push(PlannedChunk {
            delay: millis(config.ttft_ms.sample(rng)),
            chunk: chunk(
                &id,
                created,
                vec![choice(
                    Delta {
                        role: Some(Role::Assistant),
                        content: None,
                    },
                    None,
                )],
                None,
            ),
        });

        for index in 0..count {
            let delay = if index == 0 {
                Duration::ZERO
            } else {
                millis(config.inter_chunk_ms.sample(rng))
            };
            let delta = Delta {
                role: None,
                content: Some(tokenizer::token_text(index).to_owned()),
            };
            chunks.push(PlannedChunk {
                delay,
                chunk: chunk(&id, created, vec![choice(delta, None)], None),
            });
        }

        chunks.push(PlannedChunk {
            delay: Duration::ZERO,
            chunk: chunk(
                &id,
                created,
                vec![choice(Delta::default(), Some(finish_reason))],
                None,
            ),
        });

        if request
            .stream_options
            .is_some_and(|options| options.include_usage)
        {
            let prompt_tokens = tokenizer::prompt_tokens(&request.messages);
            chunks.push(PlannedChunk {
                delay: Duration::ZERO,
                chunk: chunk(
                    &id,
                    created,
                    Vec::new(),
                    Some(Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    }),
                ),
            });
        }

        Self { chunks }
    }
}

/// How long the completion runs and why it stops.
///
/// `ignore_eos` means no stop token ever arrives: the completion runs to
/// `max_tokens` exactly when given (the sampled length is discarded), to the
/// sampled length otherwise, and always finishes as `length`.
fn length_and_reason(request: &ChatCompletionRequest, sampled: u32) -> (u32, FinishReason) {
    match (request.ignore_eos.unwrap_or(false), request.max_tokens) {
        (true, Some(max)) => (max, FinishReason::Length),
        (true, None) => (sampled, FinishReason::Length),
        (false, Some(max)) if sampled > max => (max, FinishReason::Length),
        (false, _) => (sampled, FinishReason::Stop),
    }
}

/// Samples an output length: rounded to the nearest token, at least one.
///
/// A completion of zero tokens has no first token to time, so the floor
/// keeps every response measurable.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn token_count(distribution: Distribution, rng: &mut impl Rng) -> u32 {
    // The guard makes the `as` cast exact: the value is a whole number in
    // [1, u32::MAX], so neither truncation nor sign loss can occur.
    let rounded = distribution.sample(rng).round().max(1.0);
    if rounded >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        rounded as u32
    }
}

fn millis(ms: f64) -> Duration {
    Duration::from_secs_f64(ms / 1_000.0)
}

fn chunk(
    id: &str,
    created: u64,
    choices: Vec<ChunkChoice>,
    usage: Option<Usage>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created,
        model: MODEL_ID.to_owned(),
        choices,
        usage,
    }
}

fn choice(delta: Delta, finish_reason: Option<FinishReason>) -> ChunkChoice {
    ChunkChoice {
        index: 0,
        delta,
        finish_reason,
    }
}

#[cfg(test)]
mod tests {
    use moonleaf_core::protocol::{Message, StreamOptions};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    const CREATED: u64 = 1_700_000_000;

    fn fixed_config() -> InjectorConfig {
        InjectorConfig {
            ttft_ms: Distribution::Fixed(200.0),
            inter_chunk_ms: Distribution::Fixed(25.0),
            output_tokens: Distribution::Fixed(4.0),
        }
    }

    fn request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: MODEL_ID.to_owned(),
            messages: vec![Message {
                role: Role::User,
                content: "hi".to_owned(),
            }],
            stream: true,
            stream_options: None,
            max_tokens: None,
            temperature: None,
            seed: None,
            ignore_eos: None,
        }
    }

    fn plan(request: &ChatCompletionRequest) -> StreamPlan {
        StreamPlan::sample(
            request,
            fixed_config(),
            CREATED,
            &mut StdRng::seed_from_u64(7),
        )
    }

    /// The chunks that carry completion text, in order.
    fn contents(plan: &StreamPlan) -> Vec<&str> {
        plan.chunks
            .iter()
            .filter_map(|planned| planned.chunk.content())
            .collect()
    }

    #[test]
    fn plan_runs_role_content_finish() {
        let plan = plan(&request());

        // Role + 4 content + finish, no usage chunk without the opt-in.
        assert_eq!(plan.chunks.len(), 6);

        let role = &plan.chunks[0].chunk;
        assert_eq!(role.choices[0].delta.role, Some(Role::Assistant));
        assert_eq!(role.choices[0].delta.content, None);

        assert_eq!(contents(&plan).len(), 4);

        let finish = &plan.chunks[5].chunk;
        assert_eq!(finish.choices[0].delta, Delta::default());
        assert_eq!(finish.choices[0].finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn every_chunk_shares_the_response_identity() {
        let plan = plan(&request());

        for planned in &plan.chunks {
            assert_eq!(planned.chunk.id, plan.chunks[0].chunk.id);
            assert_eq!(planned.chunk.object, "chat.completion.chunk");
            assert_eq!(planned.chunk.created, CREATED);
            assert_eq!(planned.chunk.model, MODEL_ID);
        }
        assert!(plan.chunks[0].chunk.id.starts_with("chatcmpl-"));
    }

    #[test]
    fn delays_replay_the_injected_distributions() {
        let plan = plan(&request());

        let delays: Vec<Duration> = plan.chunks.iter().map(|planned| planned.delay).collect();

        // TTFT before the role chunk, nothing before the first content chunk,
        // one inter-chunk gap before each later token, free bookkeeping.
        assert_eq!(
            delays,
            vec![
                Duration::from_millis(200),
                Duration::ZERO,
                Duration::from_millis(25),
                Duration::from_millis(25),
                Duration::from_millis(25),
                Duration::ZERO,
            ]
        );
    }

    #[test]
    fn each_content_chunk_is_one_four_char_token() {
        for content in contents(&plan(&request())) {
            assert_eq!(content.len(), 4);
            assert!(content.is_ascii());
        }
    }

    #[test]
    fn sampled_length_past_max_tokens_cuts_to_length() {
        let request = ChatCompletionRequest {
            max_tokens: Some(2),
            ..request()
        };

        let plan = plan(&request);

        assert_eq!(contents(&plan).len(), 2);
        let finish = &plan.chunks[plan.chunks.len() - 1].chunk;
        assert_eq!(finish.choices[0].finish_reason, Some(FinishReason::Length));
    }

    #[test]
    fn sampled_length_within_max_tokens_stops_naturally() {
        let request = ChatCompletionRequest {
            max_tokens: Some(100),
            ..request()
        };

        let plan = plan(&request);

        assert_eq!(contents(&plan).len(), 4);
        let finish = &plan.chunks[plan.chunks.len() - 1].chunk;
        assert_eq!(finish.choices[0].finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn ignore_eos_runs_to_exactly_max_tokens() {
        let request = ChatCompletionRequest {
            max_tokens: Some(16),
            ignore_eos: Some(true),
            ..request()
        };

        let plan = plan(&request);

        assert_eq!(contents(&plan).len(), 16);
        let finish = &plan.chunks[plan.chunks.len() - 1].chunk;
        assert_eq!(finish.choices[0].finish_reason, Some(FinishReason::Length));
    }

    #[test]
    fn usage_reports_truthfully_when_opted_in() {
        let request = ChatCompletionRequest {
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            ..request()
        };

        let plan = plan(&request);

        let usage_chunk = &plan.chunks[plan.chunks.len() - 1];
        assert_eq!(usage_chunk.delay, Duration::ZERO);
        assert!(usage_chunk.chunk.choices.is_empty());
        // "hi" is 2 chars -> 1 prompt token under the 4-chars/token rule.
        assert_eq!(
            usage_chunk.chunk.usage,
            Some(Usage {
                prompt_tokens: 1,
                completion_tokens: 4,
                total_tokens: 5,
            })
        );

        // Every earlier chunk leaves usage unset.
        for planned in &plan.chunks[..plan.chunks.len() - 1] {
            assert_eq!(planned.chunk.usage, None);
        }
    }

    #[test]
    fn no_usage_chunk_without_the_opt_in() {
        for planned in &plan(&request()).chunks {
            assert_eq!(planned.chunk.usage, None);
        }
    }

    #[test]
    fn a_near_zero_sample_still_emits_one_token() {
        let config = InjectorConfig {
            output_tokens: Distribution::Fixed(0.0),
            ..fixed_config()
        };

        let plan = StreamPlan::sample(&request(), config, CREATED, &mut StdRng::seed_from_u64(7));

        assert_eq!(contents(&plan).len(), 1);
    }

    #[test]
    fn same_seed_reproduces_the_whole_plan() {
        let config = InjectorConfig {
            ttft_ms: Distribution::LogNormal {
                mu: 4.0,
                sigma: 0.5,
            },
            inter_chunk_ms: Distribution::Normal {
                mean: 25.0,
                std_dev: 5.0,
            },
            output_tokens: Distribution::Uniform {
                min: 10.0,
                max: 50.0,
            },
        };

        let first = StreamPlan::sample(&request(), config, CREATED, &mut StdRng::seed_from_u64(42));
        let second =
            StreamPlan::sample(&request(), config, CREATED, &mut StdRng::seed_from_u64(42));

        assert_eq!(first, second);
    }
}
