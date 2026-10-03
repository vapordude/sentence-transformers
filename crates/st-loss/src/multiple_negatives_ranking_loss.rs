use crate::info_span;

/// Computes the Multiple Negatives Ranking Loss (also known as InfoNCE or NT-Xent loss).
///
/// This function expects flattened dense tensors and does not perform implicit broadcasting.
/// Both `embeddings_a` and `embeddings_b` must have a length of `batch_size * embedding_dim`.
///
/// Returns the mean loss over the batch.
pub fn multiple_negatives_ranking_loss(
    embeddings_a: &[f32],
    embeddings_b: &[f32],
    embedding_dim: usize,
    scale: f32,
) -> f32 {
    let _span = info_span!("multiple_negatives_ranking_loss").enter();

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
    if batch_size == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0;
    for i in 0..batch_size {
        let a = &embeddings_a[i * embedding_dim..(i + 1) * embedding_dim];

        let mut row_max = f32::NEG_INFINITY;
        let mut sim_row = Vec::with_capacity(batch_size);

        for j in 0..batch_size {
            let b = &embeddings_b[j * embedding_dim..(j + 1) * embedding_dim];
            let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let score = dot_product * scale;
            if score > row_max {
                row_max = score;
            }
            sim_row.push(score);
        }

        let mut exp_sum = 0.0;
        for &sim in &sim_row {
            exp_sum += (sim - row_max).exp();
        }

        let log_sum_exp = row_max + exp_sum.ln();
        let target_score = sim_row[i];

        let loss = log_sum_exp - target_score;
        total_loss += loss;
    }

    total_loss / (batch_size as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiple_negatives_ranking_loss_basic() {
        println!("ST-EVIDENCE id=mnr_loss_basic_exact");
        let embeddings_a = vec![1.0, 0.0, 0.0, 1.0]; // 2 vectors of dim 2
        let embeddings_b = vec![1.0, 0.0, 0.0, 1.0]; // identical

        // Sim matrix before scale: [[1, 0], [0, 1]]
        // Scale = 1.0
        // Expected loss for each row: log(exp(1) + exp(0)) - 1 = log(e + 1) - 1 ≈ 0.31326

        let loss = multiple_negatives_ranking_loss(&embeddings_a, &embeddings_b, 2, 1.0);
        let expected = (1.0_f32.exp() + 0.0_f32.exp()).ln() - 1.0;
        assert!((loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_multiple_negatives_ranking_loss_scale() {
        println!("ST-EVIDENCE id=mnr_loss_scale_exact");
        let embeddings_a = vec![1.0, 0.0, 0.0, 1.0];
        let embeddings_b = vec![1.0, 0.0, 0.0, 1.0];

        // Sim matrix before scale: [[1, 0], [0, 1]]
        // Scale = 20.0
        // Expected loss for each row: log(exp(20) + exp(0)) - 20 ≈ 0.0

        let loss = multiple_negatives_ranking_loss(&embeddings_a, &embeddings_b, 2, 20.0);
        let expected = (20.0_f32.exp() + 0.0_f32.exp()).ln() - 20.0;
        assert!((loss - expected).abs() < 1e-5);
    }

    #[test]
    fn test_multiple_negatives_ranking_loss_empty() {
        println!("ST-EVIDENCE id=mnr_loss_empty");
        let embeddings_a: Vec<f32> = vec![];
        let embeddings_b: Vec<f32> = vec![];

        let loss = multiple_negatives_ranking_loss(&embeddings_a, &embeddings_b, 2, 20.0);
        assert_eq!(loss, 0.0);
    }
}
