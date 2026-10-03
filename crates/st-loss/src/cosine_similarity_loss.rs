use crate::info_span;

/// Computes the Cosine Similarity Loss (MSE between cosine similarity and target).
///
/// This function expects flattened dense tensors.
/// Both `embeddings_a` and `embeddings_b` must have a length of `batch_size * embedding_dim`.
/// `targets` must have a length of `batch_size`.
///
/// Returns the mean squared error over the batch.
pub fn cosine_similarity_loss(
    embeddings_a: &[f32],
    embeddings_b: &[f32],
    targets: &[f32],
    embedding_dim: usize,
) -> f32 {
    let _span = info_span!("cosine_similarity_loss").enter();

    assert!(embedding_dim > 0, "embedding_dim must be greater than 0");
    #[allow(clippy::manual_is_multiple_of)]
    {
        assert!(
            embeddings_a.len() % embedding_dim == 0,
            "embeddings_a length must be a multiple of embedding_dim"
        );
    }
    assert_eq!(
        embeddings_a.len(),
        embeddings_b.len(),
        "embeddings_a and embeddings_b must have the same length"
    );

    let batch_size = embeddings_a.len() / embedding_dim;
    assert_eq!(
        targets.len(),
        batch_size,
        "targets must have the same length as batch_size"
    );

    if batch_size == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0;

    for i in 0..batch_size {
        let a = &embeddings_a[i * embedding_dim..(i + 1) * embedding_dim];
        let b = &embeddings_b[i * embedding_dim..(i + 1) * embedding_dim];

        let mut dot_product = 0.0;
        let mut norm_a_sq = 0.0;
        let mut norm_b_sq = 0.0;

        for j in 0..embedding_dim {
            dot_product += a[j] * b[j];
            norm_a_sq += a[j] * a[j];
            norm_b_sq += b[j] * b[j];
        }

        let norm_a = norm_a_sq.sqrt();
        let norm_b = norm_b_sq.sqrt();

        let sim = if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        };

        let target = targets[i];
        let diff = sim - target;
        total_loss += diff * diff;
    }

    total_loss / (batch_size as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_loss_basic() {
        println!("ST-EVIDENCE id=cosine_loss_basic_exact");
        let embeddings_a = vec![1.0, 0.0, 0.0, 1.0]; // 2 vectors
        let embeddings_b = vec![1.0, 0.0, 1.0, 0.0];

        // Sim of row 0: (1, 0) and (1, 0) = 1.0
        // Sim of row 1: (0, 1) and (1, 0) = 0.0
        let targets = vec![1.0, 1.0];

        // Loss for row 0: (1.0 - 1.0)^2 = 0.0
        // Loss for row 1: (0.0 - 1.0)^2 = 1.0
        // Mean loss: (0.0 + 1.0) / 2 = 0.5

        let loss = cosine_similarity_loss(&embeddings_a, &embeddings_b, &targets, 2);
        assert!((loss - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_loss_zero_norm() {
        println!("ST-EVIDENCE id=cosine_loss_zero_norm_exact");
        let embeddings_a = vec![0.0, 0.0];
        let embeddings_b = vec![1.0, 1.0];
        let targets = vec![0.5];

        // Sim should be 0.0 for zero vectors
        // Loss: (0.0 - 0.5)^2 = 0.25

        let loss = cosine_similarity_loss(&embeddings_a, &embeddings_b, &targets, 2);
        assert!((loss - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_loss_empty() {
        println!("ST-EVIDENCE id=cosine_loss_empty");
        let embeddings_a: Vec<f32> = vec![];
        let embeddings_b: Vec<f32> = vec![];
        let targets: Vec<f32> = vec![];

        let loss = cosine_similarity_loss(&embeddings_a, &embeddings_b, &targets, 2);
        assert_eq!(loss, 0.0);
    }
}
