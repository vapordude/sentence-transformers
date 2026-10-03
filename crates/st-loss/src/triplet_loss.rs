use crate::info_span;

/// Distance metric used for computing TripletLoss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean distance: ||a - b||_2
    Euclidean,
    /// Manhattan distance: ||a - b||_1
    Manhattan,
    /// Cosine distance: 1 - cosine_similarity(a, b)
    Cosine,
}

/// Computes the Triplet Loss.
///
/// This function expects flattened dense tensors.
/// `anchors`, `positives`, and `negatives` must all have a length of `batch_size * embedding_dim`.
///
/// Returns the mean loss over the batch.
pub fn triplet_loss(
    anchors: &[f32],
    positives: &[f32],
    negatives: &[f32],
    embedding_dim: usize,
    margin: f32,
    distance_metric: DistanceMetric,
) -> f32 {
    let _span = info_span!("triplet_loss").enter();

    assert!(embedding_dim > 0, "embedding_dim must be greater than 0");
    #[allow(clippy::manual_is_multiple_of)]
    {
        assert!(
            anchors.len() % embedding_dim == 0,
            "anchors length must be a multiple of embedding_dim"
        );
    }
    assert_eq!(
        anchors.len(),
        positives.len(),
        "anchors and positives must have the same length"
    );
    assert_eq!(
        anchors.len(),
        negatives.len(),
        "anchors and negatives must have the same length"
    );

    let batch_size = anchors.len() / embedding_dim;
    if batch_size == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0;

    for i in 0..batch_size {
        let anchor = &anchors[i * embedding_dim..(i + 1) * embedding_dim];
        let positive = &positives[i * embedding_dim..(i + 1) * embedding_dim];
        let negative = &negatives[i * embedding_dim..(i + 1) * embedding_dim];

        let d_pos = compute_distance(anchor, positive, distance_metric);
        let d_neg = compute_distance(anchor, negative, distance_metric);

        // loss = max(d_pos - d_neg + margin, 0)
        let loss = (d_pos - d_neg + margin).max(0.0);
        total_loss += loss;
    }

    total_loss / (batch_size as f32)
}

fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Euclidean => {
            let mut sum_sq = 0.0;
            for (x, y) in a.iter().zip(b.iter()) {
                let diff = x - y;
                sum_sq += diff * diff;
            }
            sum_sq.sqrt()
        }
        DistanceMetric::Manhattan => {
            let mut sum_abs = 0.0;
            for (x, y) in a.iter().zip(b.iter()) {
                sum_abs += (x - y).abs();
            }
            sum_abs
        }
        DistanceMetric::Cosine => {
            let mut dot_product = 0.0;
            let mut norm_a_sq = 0.0;
            let mut norm_b_sq = 0.0;

            for (x, y) in a.iter().zip(b.iter()) {
                dot_product += x * y;
                norm_a_sq += x * x;
                norm_b_sq += y * y;
            }

            let norm_a = norm_a_sq.sqrt();
            let norm_b = norm_b_sq.sqrt();

            let sim = if norm_a == 0.0 || norm_b == 0.0 {
                0.0
            } else {
                dot_product / (norm_a * norm_b)
            };

            1.0 - sim
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triplet_loss_euclidean() {
        println!("ST-EVIDENCE id=triplet_loss_euclidean_exact");
        let anchors = vec![0.0, 0.0];
        let positives = vec![1.0, 0.0];
        let negatives = vec![0.0, 2.0];

        // Euclidean distance:
        // d_pos = sqrt((0-1)^2 + (0-0)^2) = 1.0
        // d_neg = sqrt((0-0)^2 + (0-2)^2) = 2.0
        // margin = 1.5
        // loss = max(1.0 - 2.0 + 1.5, 0) = max(0.5, 0) = 0.5

        let loss = triplet_loss(
            &anchors,
            &positives,
            &negatives,
            2,
            1.5,
            DistanceMetric::Euclidean,
        );
        assert!((loss - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_triplet_loss_manhattan() {
        println!("ST-EVIDENCE id=triplet_loss_manhattan_exact");
        let anchors = vec![0.0, 0.0];
        let positives = vec![1.0, 0.0];
        let negatives = vec![0.0, 2.0];

        // Manhattan distance:
        // d_pos = |0-1| + |0-0| = 1.0
        // d_neg = |0-0| + |0-2| = 2.0
        // margin = 0.5
        // loss = max(1.0 - 2.0 + 0.5, 0) = max(-0.5, 0) = 0.0

        let loss = triplet_loss(
            &anchors,
            &positives,
            &negatives,
            2,
            0.5,
            DistanceMetric::Manhattan,
        );
        assert!((loss - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_triplet_loss_cosine() {
        println!("ST-EVIDENCE id=triplet_loss_cosine_exact");
        let anchors = vec![1.0, 0.0];
        let positives = vec![1.0, 1.0]; // norm = sqrt(2). dot = 1. sim = 1/sqrt(2). dist = 1 - 1/sqrt(2) ≈ 0.29289
        let negatives = vec![0.0, 1.0]; // norm = 1. dot = 0. sim = 0. dist = 1.0

        // d_pos = 1.0 - 1.0/sqrt(2) ≈ 0.29289
        // d_neg = 1.0
        // margin = 1.0
        // loss = max(0.29289 - 1.0 + 1.0, 0) = 0.29289

        let loss = triplet_loss(
            &anchors,
            &positives,
            &negatives,
            2,
            1.0,
            DistanceMetric::Cosine,
        );
        let expected = 1.0 - 1.0 / 2.0_f32.sqrt();
        assert!((loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_triplet_loss_empty() {
        println!("ST-EVIDENCE id=triplet_loss_empty");
        let anchors: Vec<f32> = vec![];
        let positives: Vec<f32> = vec![];
        let negatives: Vec<f32> = vec![];

        let loss = triplet_loss(
            &anchors,
            &positives,
            &negatives,
            2,
            1.0,
            DistanceMetric::Euclidean,
        );
        assert_eq!(loss, 0.0);
    }
}
