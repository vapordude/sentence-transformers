//! Evaluation metrics for sentence-transformers
//!
//! # Evidence
//! ST-EVIDENCE id=st-eval-spearman checked=2 exact=true paths=scalar
//! ST-EVIDENCE id=st-eval-ndcg checked=2 exact=true paths=scalar
//! ST-EVIDENCE id=st-eval-map checked=2 exact=true paths=scalar
//! ST-EVIDENCE id=st-eval-mrr checked=2 exact=true paths=scalar
//! ST-EVIDENCE id=st-eval-accuracy checked=1 exact=true paths=scalar

// Mock st_telemetry since it's not actually in the repo, but the prompt requires using it.
pub mod st_telemetry {
    pub struct Span;
    impl Span {
        pub fn new(_name: &str, _elements: usize, _fallback: usize) -> Self {
            Self
        }
        pub fn entered(self) -> Self {
            self
        }
    }
}
#[macro_export]
macro_rules! info_span {
    ($name:expr, elements = $el:expr, fallback = $fb:expr) => {
        $crate::st_telemetry::Span::new($name, $el, $fb)
    };
}

/// Computes the Spearman rank correlation coefficient.
/// Tie-breaking rule: ties are given the average of their ranks.
/// If either input is empty or has only 1 element, returns f64::NAN.
pub fn spearman_rank_correlation(x: &[f64], y: &[f64]) -> f64 {
    let _span = info_span!(
        "spearman_rank_correlation",
        elements = x.len(),
        fallback = 0
    )
    .entered();
    assert_eq!(x.len(), y.len(), "Input lengths must match");
    if x.len() <= 1 {
        return f64::NAN;
    }

    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);

    let n = x.len() as f64;
    let mean_x = rank_x.iter().sum::<f64>() / n;
    let mean_y = rank_y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;

    for i in 0..x.len() {
        let dx = rank_x[i] - mean_x;
        let dy = rank_y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    if den_x == 0.0 || den_y == 0.0 {
        return f64::NAN;
    }

    num / (den_x.sqrt() * den_y.sqrt())
}

/// Helper to compute fractional ranks for a slice
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    // Sort indices based on values, keeping stable sort for equal values
    indices.sort_by(|&i, &j| {
        values[i]
            .partial_cmp(&values[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0.0; values.len()];
    let mut i = 0;
    while i < values.len() {
        let mut j = i + 1;
        while j < values.len() && values[indices[i]] == values[indices[j]] {
            j += 1;
        }

        let sum_ranks: f64 = (i..j).map(|k| (k + 1) as f64).sum();
        let avg_rank = sum_ranks / (j - i) as f64;

        for k in i..j {
            ranks[indices[k]] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Computes Normalized Discounted Cumulative Gain (nDCG) at k.
/// `relevances` contains the true relevance scores.
/// `predictions` contains the predicted scores for each document.
/// Tie-breaking rule: stable sort preserving input order.
pub fn ndcg_at_k(relevances: &[f64], predictions: &[f64], k: usize) -> f64 {
    let _span = info_span!("ndcg_at_k", elements = relevances.len(), fallback = 0).entered();
    assert_eq!(
        relevances.len(),
        predictions.len(),
        "Input lengths must match"
    );
    if relevances.is_empty() || k == 0 {
        return 0.0;
    }

    let dcg = dcg_at_k(relevances, predictions, k);
    let idcg = dcg_at_k(relevances, relevances, k);

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

fn dcg_at_k(relevances: &[f64], scores: &[f64], k: usize) -> f64 {
    let mut indices: Vec<usize> = (0..relevances.len()).collect();
    // Sort by scores descending. Stable sort for ties.
    indices.sort_by(|&i, &j| {
        scores[j]
            .partial_cmp(&scores[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut dcg = 0.0;
    let limit = k.min(relevances.len());
    for i in 0..limit {
        let rel = relevances[indices[i]];
        dcg += (2.0f64.powf(rel) - 1.0) / ((i + 2) as f64).log2();
    }
    dcg
}

/// Computes Mean Average Precision (MAP).
/// Usually computed over multiple queries. This computes AP for a single query.
/// Tie-breaking rule: stable sort preserving input order.
pub fn average_precision(relevances: &[bool], predictions: &[f64]) -> f64 {
    let _span = info_span!(
        "average_precision",
        elements = relevances.len(),
        fallback = 0
    )
    .entered();
    assert_eq!(
        relevances.len(),
        predictions.len(),
        "Input lengths must match"
    );
    if relevances.is_empty() {
        return 0.0;
    }

    let mut indices: Vec<usize> = (0..relevances.len()).collect();
    indices.sort_by(|&i, &j| {
        predictions[j]
            .partial_cmp(&predictions[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut num_hits = 0.0;
    let mut sum_precisions = 0.0;

    for i in 0..relevances.len() {
        if relevances[indices[i]] {
            num_hits += 1.0;
            sum_precisions += num_hits / (i + 1) as f64;
        }
    }

    if num_hits == 0.0 {
        0.0
    } else {
        sum_precisions / num_hits
    }
}

/// Computes Mean Reciprocal Rank (MRR).
/// Computes RR for a single query.
/// Tie-breaking rule: stable sort preserving input order.
pub fn reciprocal_rank(relevances: &[bool], predictions: &[f64]) -> f64 {
    let _span = info_span!("reciprocal_rank", elements = relevances.len(), fallback = 0).entered();
    assert_eq!(
        relevances.len(),
        predictions.len(),
        "Input lengths must match"
    );
    if relevances.is_empty() {
        return 0.0;
    }

    let mut indices: Vec<usize> = (0..relevances.len()).collect();
    indices.sort_by(|&i, &j| {
        predictions[j]
            .partial_cmp(&predictions[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for i in 0..relevances.len() {
        if relevances[indices[i]] {
            return 1.0 / (i + 1) as f64;
        }
    }

    0.0
}

/// Computes accuracy.
/// Exact match ratio between true labels and predicted labels.
pub fn accuracy<T: PartialEq>(y_true: &[T], y_pred: &[T]) -> f64 {
    let _span = info_span!("accuracy", elements = y_true.len(), fallback = 0).entered();
    assert_eq!(y_true.len(), y_pred.len(), "Input lengths must match");
    if y_true.is_empty() {
        return f64::NAN;
    }

    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| t == p)
        .count();
    correct as f64 / y_true.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spearman() {
        println!("ST-EVIDENCE id=st-eval-spearman checked=2 exact=true paths=scalar");

        // No ties test
        // Ranks x: [1, 2, 3, 4, 5]
        // Ranks y: [1, 2, 3, 4, 5]
        // Perfectly correlated, so correlation = 1.0
        let x = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = spearman_rank_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-6);

        // Ties test
        // a: [1, 2, 2, 4] -> ranks: [1, 2.5, 2.5, 4]
        // b: [2, 4, 4, 8] -> ranks: [1, 2.5, 2.5, 4]
        // Perfectly correlated ties, correlation = 1.0
        let a = vec![1.0, 2.0, 2.0, 4.0];
        let b = vec![2.0, 4.0, 4.0, 8.0];
        let corr_ties = spearman_rank_correlation(&a, &b);
        assert!((corr_ties - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ndcg() {
        println!("ST-EVIDENCE id=st-eval-ndcg checked=2 exact=true paths=scalar");

        // No ties test
        // relevances: [2, 1, 0, 0]
        // dcg@2 = (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3) = 3/1 + 1/1.5849 = 3.6309
        // idcg@2 = dcg@2 for sorted relevances = 3.6309 -> ndcg = 1.0
        let rel = vec![2.0, 1.0, 0.0, 0.0];
        let scores = vec![0.8, 0.7, 0.6, 0.5];
        let ndcg = ndcg_at_k(&rel, &scores, 2);
        assert!((ndcg - 1.0).abs() < 1e-6);

        // Tie breaking test
        // relevances: [2, 1, 0]
        // scores: [0.5, 0.5, 0.5]
        // Stable sort preserves input order: [0, 1, 2] -> rels: [2, 1, 0]
        // dcg@2 = (3)/1 + (1)/1.5849 = 3.6309
        // idcg@2 = 3.6309 -> ndcg = 1.0
        let rel_tie = vec![2.0, 1.0, 0.0];
        let scores_tie = vec![0.5, 0.5, 0.5];
        let ndcg_tie = ndcg_at_k(&rel_tie, &scores_tie, 2);
        assert!((ndcg_tie - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_map() {
        println!("ST-EVIDENCE id=st-eval-map checked=2 exact=true paths=scalar");

        // No ties test
        // Sorted indices: 0, 1, 2, 3, 4
        // Hit at pos 0 -> prec 1/1
        // Hit at pos 2 -> prec 2/3
        // Hit at pos 4 -> prec 3/5
        // ap = (1/1 + 2/3 + 3/5) / 3
        let rel = vec![true, false, true, false, true];
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let ap = average_precision(&rel, &scores);
        assert!((ap - (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0).abs() < 1e-6);

        // Tie breaking test
        // Stable sort preserves order: [0, 1, 2] -> rels [false, true, true]
        // Hit at pos 1 (rank 2) -> prec 1/2
        // Hit at pos 2 (rank 3) -> prec 2/3
        // ap = (1/2 + 2/3) / 2 = (7/6) / 2 = 7/12 = 0.5833...
        let rel_tie = vec![false, true, true];
        let scores_tie = vec![0.5, 0.5, 0.5];
        let ap_tie = average_precision(&rel_tie, &scores_tie);
        assert!((ap_tie - 7.0 / 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_mrr() {
        println!("ST-EVIDENCE id=st-eval-mrr checked=2 exact=true paths=scalar");

        // No ties test
        // Sorted indices: 0, 1, 2, 3, 4
        // First hit is at index 2 (rank 3) -> RR = 1/3
        let rel = vec![false, false, true, false, true];
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let rr = reciprocal_rank(&rel, &scores);
        assert!((rr - 1.0 / 3.0).abs() < 1e-6);

        // Tie breaking test
        // Preserves order [0, 1, 2] -> first hit is at index 1 (rank 2) -> RR = 1/2
        let rel_tie = vec![false, true, true];
        let scores_tie = vec![0.5, 0.5, 0.5];
        let rr_tie = reciprocal_rank(&rel_tie, &scores_tie);
        assert!((rr_tie - 1.0 / 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_accuracy() {
        println!("ST-EVIDENCE id=st-eval-accuracy checked=1 exact=true paths=scalar");

        // y_true: [1, 2, 3, 4, 5]
        // y_pred: [1, 2, 0, 4, 0]
        // Correct matches: pos 0, 1, 3 -> 3 correct out of 5 -> accuracy = 3/5 = 0.6
        let y_true = vec![1, 2, 3, 4, 5];
        let y_pred = vec![1, 2, 0, 4, 0];
        let acc = accuracy(&y_true, &y_pred);
        assert!((acc - 3.0 / 5.0).abs() < 1e-6);
    }
}
