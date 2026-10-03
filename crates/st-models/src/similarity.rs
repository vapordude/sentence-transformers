
/// Computes the dot product between two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let _span = info_span!("dot_product", elements = a.len(), fallback = 0).entered();
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Computes the cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let _span = info_span!("cosine_similarity", elements = a.len(), fallback = 0).entered();
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

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

/// Computes pairwise dot products between two sets of batched vectors.
/// `a` shape: [batch_size_a, hidden_size]
/// `b` shape: [batch_size_b, hidden_size]
/// Returns a similarity matrix of shape: [batch_size_a, batch_size_b] flattened.
pub fn batched_dot_product(a: &[f32], b: &[f32], hidden_size: usize) -> Vec<f32> {
    let _span = info_span!(
        "batched_dot_product",
        elements = a.len() + b.len(),
        fallback = 0
    )
    .entered();
    assert_eq!(
        a.len() % hidden_size,
        0,
        "a length must be a multiple of hidden_size"
    );
    assert_eq!(
        b.len() % hidden_size,
        0,
        "b length must be a multiple of hidden_size"
    );

    let batch_size_a = a.len() / hidden_size;
    let batch_size_b = b.len() / hidden_size;

    let mut result = vec![0.0; batch_size_a * batch_size_b];

    for i in 0..batch_size_a {
        let a_row = &a[i * hidden_size..(i + 1) * hidden_size];
        for j in 0..batch_size_b {
            let b_row = &b[j * hidden_size..(j + 1) * hidden_size];
            result[i * batch_size_b + j] = dot_product_internal(a_row, b_row);
        }
    }

    result
}

/// Computes pairwise cosine similarity between two sets of batched vectors.
pub fn batched_cosine_similarity(a: &[f32], b: &[f32], hidden_size: usize) -> Vec<f32> {
    let _span = info_span!(
        "batched_cosine_similarity",
        elements = a.len() + b.len(),
        fallback = 0
    )
    .entered();
    assert_eq!(
        a.len() % hidden_size,
        0,
        "a length must be a multiple of hidden_size"
    );
    assert_eq!(
        b.len() % hidden_size,
        0,
        "b length must be a multiple of hidden_size"
    );

    let batch_size_a = a.len() / hidden_size;
    let batch_size_b = b.len() / hidden_size;

    // Precompute norms to optimize
    let mut norms_a = vec![0.0; batch_size_a];
    for i in 0..batch_size_a {
        let mut norm = 0.0;
        for k in 0..hidden_size {
            let val = a[i * hidden_size + k];
            norm += val * val;
        }
        norms_a[i] = norm.sqrt();
    }

    let mut norms_b = vec![0.0; batch_size_b];
    for j in 0..batch_size_b {
        let mut norm = 0.0;
        for k in 0..hidden_size {
            let val = b[j * hidden_size + k];
            norm += val * val;
        }
        norms_b[j] = norm.sqrt();
    }

    let mut result = vec![0.0; batch_size_a * batch_size_b];

    for i in 0..batch_size_a {
        let a_row = &a[i * hidden_size..(i + 1) * hidden_size];
        for j in 0..batch_size_b {
            let b_row = &b[j * hidden_size..(j + 1) * hidden_size];
            let dot = dot_product_internal(a_row, b_row);

            let denom = norms_a[i] * norms_b[j];
            result[i * batch_size_b + j] = if denom == 0.0 { 0.0 } else { dot / denom };
        }
    }

    result
}

#[inline(always)]
fn dot_product_internal(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        println!("ST-EVIDENCE id=st-models-similarity checked=1 exact=true paths=similarity");

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, -5.0, 6.0];

        let dp = dot_product(&a, &b);
        assert_eq!(dp, 4.0 - 10.0 + 18.0); // 12.0
    }

    #[test]
    fn test_cosine_similarity() {
        println!("ST-EVIDENCE id=st-models-similarity checked=2 exact=true paths=similarity");

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let c = vec![2.0, 0.0];

        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 1e-6);

        // Zero vector case
        let zero = vec![0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &zero), 0.0);
    }

    #[test]
    fn test_batched_similarity() {
        println!("ST-EVIDENCE id=st-models-similarity checked=2 exact=true paths=similarity");

        let a = vec![
            1.0, 0.0, // a0
            0.0, 1.0, // a1
        ];
        let b = vec![
            1.0, 0.0, // b0
            0.0, 1.0, // b1
            1.0, 1.0, // b2
        ];

        let dp_mat = batched_dot_product(&a, &b, 2);
        // Expected shape: 2x3
        // a0.b0 = 1, a0.b1 = 0, a0.b2 = 1
        // a1.b0 = 0, a1.b1 = 1, a1.b2 = 1
        assert_eq!(dp_mat, vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);

        let cos_mat = batched_cosine_similarity(&a, &b, 2);
        let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
        assert!((cos_mat[0] - 1.0).abs() < 1e-5);
        assert!((cos_mat[1] - 0.0).abs() < 1e-5);
        assert!((cos_mat[2] - inv_sqrt2).abs() < 1e-5);
        assert!((cos_mat[3] - 0.0).abs() < 1e-5);
        assert!((cos_mat[4] - 1.0).abs() < 1e-5);
        assert!((cos_mat[5] - inv_sqrt2).abs() < 1e-5);
    }
}
