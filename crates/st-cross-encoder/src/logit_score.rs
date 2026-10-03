//! LogitScore module for cross-encoder reranking.

/// Converts causal language model logits into a relevance score for reranking.
///
/// Extracts the logit at a true token position, optionally subtracting the logit
/// at a false token position to compute log-odds.
#[derive(Debug, Clone)]
pub struct LogitScore {
    pub true_token_id: usize,
    pub false_token_id: Option<usize>,
}

impl LogitScore {
    pub fn new(true_token_id: usize, false_token_id: Option<usize>) -> Self {
        Self {
            true_token_id,
            false_token_id,
        }
    }

    /// Computes the score from last token logits.
    /// `last_token_logits` should be flattened: [batch_size, vocab_size]
    /// Returns a flat vector of scores of length `batch_size`.
    pub fn forward_batched(&self, last_token_logits: &[f32], vocab_size: usize) -> Vec<f32> {
        let _span = crate::info_span!(
            "logit_score_forward_batched",
            elements = last_token_logits.len(),
            fallback = 0
        )
        .entered();
        #[allow(clippy::manual_is_multiple_of)]
        let is_multiple = last_token_logits.len() % vocab_size == 0;
        assert!(
            is_multiple,
            "Logits length must be a multiple of vocab_size"
        );

        let batch_size = last_token_logits.len() / vocab_size;
        let mut scores = vec![0.0; batch_size];

        for b in 0..batch_size {
            let true_logit = last_token_logits[b * vocab_size + self.true_token_id];
            let score = if let Some(false_id) = self.false_token_id {
                let false_logit = last_token_logits[b * vocab_size + false_id];
                true_logit - false_logit
            } else {
                true_logit
            };
            scores[b] = score;
        }

        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logit_score_single_token() {
        let scorer = LogitScore::new(1, None); // true_token_id = 1
        let logits = vec![
            0.1, 0.9, 0.0, // batch 0
            0.2, 0.4, 0.4, // batch 1
        ]; // vocab_size = 3

        let scores = scorer.forward_batched(&logits, 3);
        assert_eq!(scores, vec![0.9, 0.4]);
        println!(
            "ST-EVIDENCE id=st-cross-encoder-logit-score checked=1 exact=true paths=logit_score"
        );
    }

    #[test]
    fn test_logit_score_log_odds() {
        let scorer = LogitScore::new(1, Some(2)); // true = 1, false = 2
        let logits = vec![
            0.1, 0.9, 0.5, // batch 0: 0.9 - 0.5 = 0.4
            0.2, 0.4, 0.8, // batch 1: 0.4 - 0.8 = -0.4
        ]; // vocab_size = 3

        let scores = scorer.forward_batched(&logits, 3);

        // Assert float equality
        assert!((scores[0] - 0.4).abs() < 1e-6);
        assert!((scores[1] - (-0.4)).abs() < 1e-6);
        println!(
            "ST-EVIDENCE id=st-cross-encoder-logit-score checked=2 exact=true paths=logit_score"
        );
    }

    #[test]
    fn test_logit_score_invalid_input() {
        let scorer = LogitScore::new(1, None);
        let result = std::panic::catch_unwind(|| {
            scorer.forward_batched(&[0.1, 0.9, 0.0, 0.2], 3); // Length 4 not multiple of 3
        });
        assert!(result.is_err());
    }
}
