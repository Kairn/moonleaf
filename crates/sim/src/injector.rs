//! The injector engine's sampled distributions.
//!
//! The injector models nothing: TTFT, inter-chunk latency, and output length
//! are each drawn from a configured distribution and replayed onto the
//! stream.

use rand::Rng;
use rand_distr::{Distribution as _, LogNormal, Normal};

/// A sampled value source for one injector parameter.
///
/// Values are unitless: the same enum describes latencies (sampled as
/// milliseconds) and output lengths (sampled as tokens, rounded by the
/// caller). Samples are clamped to zero from below — a negative latency or
/// token count is never meaningful, and clamping rather than resampling
/// costs exactly one draw per sample, so seeded streams stay aligned no
/// matter what values come out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distribution {
    /// Every sample is exactly this value.
    Fixed(f64),
    /// Uniformly distributed over `[min, max]`.
    Uniform { min: f64, max: f64 },
    /// Normally distributed; negative samples clamp to zero.
    Normal { mean: f64, std_dev: f64 },
    /// Log-normally distributed. `mu` and `sigma` parameterize the
    /// *underlying normal*, not the samples: the median is `e^mu`.
    LogNormal { mu: f64, sigma: f64 },
}

impl Distribution {
    /// Draws one value, never below zero.
    ///
    /// # Panics
    ///
    /// Panics on malformed parameters: `min > max`, or a non-finite or
    /// negative `std_dev`/`sigma`. Whoever builds a config owns validating it before
    /// sampling starts; a panic here is a bug, not a user error.
    #[must_use]
    pub fn sample(self, rng: &mut impl Rng) -> f64 {
        let value = match self {
            Self::Fixed(value) => value,
            Self::Uniform { min, max } => rng.random_range(min..=max),
            Self::Normal { mean, std_dev } => Normal::new(mean, std_dev)
                .expect("std_dev must be finite and non-negative")
                .sample(rng),
            Self::LogNormal { mu, sigma } => LogNormal::new(mu, sigma)
                .expect("sigma must be finite and non-negative")
                .sample(rng),
        };

        value.max(0.0)
    }
}

/// Everything the injector engine samples, one distribution per parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InjectorConfig {
    /// Delay before the first content chunk, in milliseconds.
    pub ttft_ms: Distribution,
    /// Delay between consecutive content chunks, in milliseconds.
    pub inter_chunk_ms: Distribution,
    /// Completion length in tokens, before any `max_tokens` clamp.
    pub output_tokens: Distribution,
}

/// A believable first `curl`: 200 ms to first token, 25 ms between tokens
/// (40 tokens/s), 128-token completions. All `Fixed`, so an unconfigured
/// server is deterministic even without a seed.
impl Default for InjectorConfig {
    fn default() -> Self {
        Self {
            ttft_ms: Distribution::Fixed(200.0),
            inter_chunk_ms: Distribution::Fixed(25.0),
            output_tokens: Distribution::Fixed(128.0),
        }
    }
}

#[cfg(test)]
mod tests {
    // Exact float equality is the point under test: Fixed must echo its value
    // untouched, and equal seeds must reproduce bit-identical sequences.
    #![allow(clippy::float_cmp)]

    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    #[test]
    fn fixed_ignores_the_rng() {
        let mut a = StdRng::seed_from_u64(1);
        let mut b = StdRng::seed_from_u64(2);

        assert_eq!(Distribution::Fixed(42.5).sample(&mut a), 42.5);
        assert_eq!(Distribution::Fixed(42.5).sample(&mut b), 42.5);
    }

    #[test]
    fn uniform_stays_within_bounds() {
        let mut rng = StdRng::seed_from_u64(7);
        let dist = Distribution::Uniform {
            min: 10.0,
            max: 20.0,
        };

        for _ in 0..1_000 {
            let value = dist.sample(&mut rng);
            assert!((10.0..=20.0).contains(&value), "out of bounds: {value}");
        }
    }

    #[test]
    fn normal_tracks_its_mean() {
        // Seeded, so the loose tolerance cannot flake.
        let mut rng = StdRng::seed_from_u64(7);
        let dist = Distribution::Normal {
            mean: 100.0,
            std_dev: 10.0,
        };

        let mean = (0..1_000).map(|_| dist.sample(&mut rng)).sum::<f64>() / 1_000.0;

        assert!((mean - 100.0).abs() < 2.0, "sample mean drifted: {mean}");
    }

    #[test]
    fn negative_samples_clamp_to_zero() {
        let mut rng = StdRng::seed_from_u64(7);
        let dist = Distribution::Normal {
            mean: -1_000.0,
            std_dev: 1.0,
        };

        assert_eq!(dist.sample(&mut rng), 0.0);
        assert_eq!(Distribution::Fixed(-1.0).sample(&mut rng), 0.0);
    }

    #[test]
    fn lognormal_samples_are_positive() {
        let mut rng = StdRng::seed_from_u64(7);
        let dist = Distribution::LogNormal {
            mu: 4.0,
            sigma: 0.5,
        };

        for _ in 0..1_000 {
            let value = dist.sample(&mut rng);
            assert!(value > 0.0, "lognormal support is (0, inf): {value}");
        }
    }

    #[test]
    fn lognormal_matches_its_parameters_in_log_space() {
        // By definition ln(X) ~ Normal(mu, sigma), so measuring in log space
        // checks both parameters without heavy-tail noise in the way.
        let mut rng = StdRng::seed_from_u64(7);
        let dist = Distribution::LogNormal {
            mu: 4.0,
            sigma: 0.5,
        };

        let logs: Vec<f64> = (0..1_000).map(|_| dist.sample(&mut rng).ln()).collect();
        let mean = logs.iter().sum::<f64>() / 1_000.0;
        let std_dev = (logs.iter().map(|log| (log - mean).powi(2)).sum::<f64>() / 1_000.0).sqrt();

        assert!((mean - 4.0).abs() < 0.05, "log-space mean drifted: {mean}");
        assert!(
            (std_dev - 0.5).abs() < 0.05,
            "log-space std drifted: {std_dev}"
        );
    }

    #[test]
    fn same_seed_reproduces_the_sequence() {
        let dist = Distribution::Normal {
            mean: 50.0,
            std_dev: 15.0,
        };
        let mut a = StdRng::seed_from_u64(99);
        let mut b = StdRng::seed_from_u64(99);

        let first: Vec<f64> = (0..32).map(|_| dist.sample(&mut a)).collect();
        let second: Vec<f64> = (0..32).map(|_| dist.sample(&mut b)).collect();

        assert_eq!(first, second);
    }
}
