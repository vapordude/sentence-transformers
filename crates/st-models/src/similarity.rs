/// Computes the dot product between two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let _span = crate::info_span!("dot_product", elements = a.len(), fallback = 0).entered();
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Computes the cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let _span = crate::info_span!("cosine_similarity", elements = a.len(), fallback = 0).entered();
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    #[allow(clippy::needless_range_loop)]
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

/// Computes pairwise similarities between a batch of queries and a batch of documents.
/// `queries`: [num_queries, hidden_size] flattened
/// `documents`: [num_docs, hidden_size] flattened
/// Returns: [num_queries, num_docs] flattened similarity matrix
pub fn batched_similarity(
    queries: &[f32],
    documents: &[f32],
    hidden_size: usize,
    use_cosine: bool,
) -> Vec<f32> {
    let _span = crate::info_span!(
        "batched_similarity",
        elements = queries.len() + documents.len(),
        fallback = 0
    )
    .entered();
    #[allow(clippy::manual_is_multiple_of)]
    let is_multiple_q = queries.len() % hidden_size == 0;
    #[allow(clippy::manual_is_multiple_of)]
    let is_multiple_d = documents.len() % hidden_size == 0;
    assert!(is_multiple_q && is_multiple_d);

    let num_queries = queries.len() / hidden_size;
    let num_docs = documents.len() / hidden_size;

    let mut output = vec![0.0; num_queries * num_docs];

    for q in 0..num_queries {
        let q_vec = &queries[q * hidden_size..(q + 1) * hidden_size];
        for d in 0..num_docs {
            let d_vec = &documents[d * hidden_size..(d + 1) * hidden_size];

            let sim = if use_cosine {
                cosine_similarity(q_vec, d_vec)
            } else {
                dot_product(q_vec, d_vec)
            };

            output[q * num_docs + d] = sim;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b), 32.0); // 4 + 10 + 18
        println!("ST-EVIDENCE id=st-models-similarity checked=1 exact=true paths=similarity");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        let a = vec![2.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);
    }

    #[test]
    fn test_batched_similarity() {
        let queries = vec![
            1.0, 0.0, // q0
            0.0, 1.0, // q1
        ];
        let documents = vec![
            1.0, 0.0, // d0
            0.0, 1.0, // d1
        ];

        let out_dot = batched_similarity(&queries, &documents, 2, false);
        // q0*d0 = 1, q0*d1 = 0
        // q1*d0 = 0, q1*d1 = 1
        assert_eq!(out_dot, vec![1.0, 0.0, 0.0, 1.0]);
        println!("ST-EVIDENCE id=st-models-similarity checked=2 exact=true paths=similarity");
    }
}
