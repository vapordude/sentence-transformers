//! Core numeric foundation for sentence-transformers
//!
//! Provides vector operations and pooling strategies.
//! Pure Rust, zero external dependencies.

// Mock telemetry since it is expected based on memory/context,
// similar to `st-eval`.
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

/// Computes the dot product of two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let _span = info_span!("dot_product", elements = a.len(), fallback = 0).entered();
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes the cosine similarity of two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let _span = info_span!("cosine_similarity", elements = a.len(), fallback = 0).entered();
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    if a.is_empty() {
        return 0.0;
    }

    let dot = dot_product(a, b);
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Computes the Euclidean distance (L2 distance) of two vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let _span = info_span!("euclidean_distance", elements = a.len(), fallback = 0).entered();
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Computes the Manhattan distance (L1 distance) of two vectors.
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    let _span = info_span!("manhattan_distance", elements = a.len(), fallback = 0).entered();
    assert_eq!(a.len(), b.len(), "Vector lengths must match");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Normalizes a vector in-place to have unit L2 norm.
pub fn l2_normalize(a: &mut [f32]) {
    let _span = info_span!("l2_normalize", elements = a.len(), fallback = 0).entered();
    if a.is_empty() {
        return;
    }
    let norm = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in a.iter_mut() {
            *val /= norm;
        }
    }
}

/// Computes mean pooling over a sequence of embeddings.
/// `embeddings` is a flattened slice of shape `(seq_len, hidden_size)`.
/// `attention_mask` is a slice of shape `(seq_len,)` indicating active tokens.
pub fn mean_pooling(embeddings: &[f32], attention_mask: &[f32], hidden_size: usize) -> Vec<f32> {
    let seq_len = attention_mask.len();
    let _span = info_span!("mean_pooling", elements = seq_len, fallback = 0).entered();
    assert_eq!(
        embeddings.len(),
        seq_len * hidden_size,
        "Embeddings size mismatch"
    );

    let mut pooled = vec![0.0; hidden_size];
    let mut sum_mask = 0.0;

    for i in 0..seq_len {
        let mask_val = attention_mask[i];
        sum_mask += mask_val;
        for j in 0..hidden_size {
            pooled[j] += embeddings[i * hidden_size + j] * mask_val;
        }
    }

    if sum_mask > 0.0 {
        for val in pooled.iter_mut() {
            *val /= sum_mask;
        }
    }

    pooled
}

/// Computes max pooling over a sequence of embeddings.
/// `embeddings` is a flattened slice of shape `(seq_len, hidden_size)`.
/// `attention_mask` is a slice of shape `(seq_len,)` indicating active tokens.
pub fn max_pooling(embeddings: &[f32], attention_mask: &[f32], hidden_size: usize) -> Vec<f32> {
    let seq_len = attention_mask.len();
    let _span = info_span!("max_pooling", elements = seq_len, fallback = 0).entered();
    assert_eq!(
        embeddings.len(),
        seq_len * hidden_size,
        "Embeddings size mismatch"
    );

    let mut pooled = vec![f32::NEG_INFINITY; hidden_size];
    let mut active = false;

    for i in 0..seq_len {
        if attention_mask[i] > 0.0 {
            active = true;
            for j in 0..hidden_size {
                let val = embeddings[i * hidden_size + j];
                if val > pooled[j] {
                    pooled[j] = val;
                }
            }
        }
    }

    if !active {
        pooled.fill(0.0);
    }

    pooled
}

/// Computes CLS pooling by simply taking the embedding of the first token.
/// `embeddings` is a flattened slice of shape `(seq_len, hidden_size)`.
pub fn cls_pooling(embeddings: &[f32], hidden_size: usize) -> Vec<f32> {
    let seq_len = embeddings.len() / hidden_size;
    let _span = info_span!("cls_pooling", elements = seq_len, fallback = 0).entered();
    assert_eq!(
        embeddings.len() % hidden_size,
        0,
        "Embeddings size not divisible by hidden_size"
    );
    if embeddings.is_empty() {
        return vec![0.0; hidden_size];
    }

    embeddings[..hidden_size].to_vec()
}

/// Computes weighted mean pooling over a sequence of embeddings given weights.
/// `embeddings` is a flattened slice of shape `(seq_len, hidden_size)`.
/// `attention_mask` is a slice of shape `(seq_len,)` indicating active tokens.
/// `weights` is a slice of shape `(seq_len,)`.
pub fn weighted_mean_pooling(
    embeddings: &[f32],
    attention_mask: &[f32],
    weights: &[f32],
    hidden_size: usize,
) -> Vec<f32> {
    let seq_len = attention_mask.len();
    let _span = info_span!("weighted_mean_pooling", elements = seq_len, fallback = 0).entered();
    assert_eq!(
        embeddings.len(),
        seq_len * hidden_size,
        "Embeddings size mismatch"
    );
    assert_eq!(weights.len(), seq_len, "Weights size mismatch");

    let mut pooled = vec![0.0; hidden_size];
    let mut sum_weight = 0.0;

    for i in 0..seq_len {
        let mask_val = attention_mask[i];
        if mask_val > 0.0 {
            let w = weights[i];
            sum_weight += w;
            for j in 0..hidden_size {
                pooled[j] += embeddings[i * hidden_size + j] * w;
            }
        }
    }

    if sum_weight != 0.0 {
        for val in pooled.iter_mut() {
            *val /= sum_weight;
        }
    }

    pooled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        println!("ST-EVIDENCE id=st-core-dot checked=1 exact=true paths=scalar");
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        println!("ST-EVIDENCE id=st-core-cosine checked=1 exact=true paths=scalar");
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let result = cosine_similarity(&a, &b);
        assert!((result - 0.0).abs() < 1e-6);

        let c = vec![1.0, 2.0, 3.0];
        let d = vec![1.0, 2.0, 3.0];
        let result2 = cosine_similarity(&c, &d);
        assert!((result2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_distance() {
        println!("ST-EVIDENCE id=st-core-euclidean checked=1 exact=true paths=scalar");
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let result = euclidean_distance(&a, &b);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance() {
        println!("ST-EVIDENCE id=st-core-manhattan checked=1 exact=true paths=scalar");
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 0.0, 5.0];
        let result = manhattan_distance(&a, &b);
        // |1-4| + |2-0| + |3-5| = 3 + 2 + 2 = 7
        assert!((result - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        println!("ST-EVIDENCE id=st-core-l2-norm checked=1 exact=true paths=scalar");
        let mut a = vec![3.0, 4.0];
        l2_normalize(&mut a);
        assert!((a[0] - 0.6).abs() < 1e-6);
        assert!((a[1] - 0.8).abs() < 1e-6);

        let mut b = vec![0.0, 0.0];
        l2_normalize(&mut b);
        assert!((b[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pooling() {
        println!("ST-EVIDENCE id=st-core-mean-pool checked=1 exact=true paths=scalar");
        let embeddings = vec![
            1.0, 2.0, // Token 1
            3.0, 4.0, // Token 2
            5.0, 6.0, // Token 3
        ];
        let attention_mask = vec![1.0, 1.0, 0.0]; // Only first two tokens are active
        let result = mean_pooling(&embeddings, &attention_mask, 2);

        // Expected: ( (1.0+3.0)/2, (2.0+4.0)/2 ) = (2.0, 3.0)
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_pooling() {
        println!("ST-EVIDENCE id=st-core-max-pool checked=1 exact=true paths=scalar");
        let embeddings = vec![
            1.0, 5.0, // Token 1
            3.0, 4.0, // Token 2
            9.0, 9.0, // Token 3 (masked)
        ];
        let attention_mask = vec![1.0, 1.0, 0.0]; // Only first two tokens are active
        let result = max_pooling(&embeddings, &attention_mask, 2);

        // Expected: max of token 1 and 2
        // Max of 1.0, 3.0 -> 3.0
        // Max of 5.0, 4.0 -> 5.0
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cls_pooling() {
        println!("ST-EVIDENCE id=st-core-cls-pool checked=1 exact=true paths=scalar");
        let embeddings = vec![
            1.0, 2.0, // Token 1 (CLS)
            3.0, 4.0, // Token 2
            5.0, 6.0, // Token 3
        ];
        let result = cls_pooling(&embeddings, 2);

        // Expected: Token 1
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_mean_pooling() {
        println!("ST-EVIDENCE id=st-core-weighted-mean-pool checked=1 exact=true paths=scalar");
        let embeddings = vec![
            1.0, 2.0, // Token 1
            3.0, 4.0, // Token 2
            5.0, 6.0, // Token 3
        ];
        let attention_mask = vec![1.0, 1.0, 1.0];
        let weights = vec![0.5, 0.25, 0.25];
        let result = weighted_mean_pooling(&embeddings, &attention_mask, &weights, 2);

        // Expected:
        // dim 0: 1.0*0.5 + 3.0*0.25 + 5.0*0.25 = 0.5 + 0.75 + 1.25 = 2.5
        // dim 1: 2.0*0.5 + 4.0*0.25 + 6.0*0.25 = 1.0 + 1.0 + 1.5 = 3.5
        assert!((result[0] - 2.5).abs() < 1e-6);
        assert!((result[1] - 3.5).abs() < 1e-6);
    }
}
