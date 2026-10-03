
/// Mean Pooling strategy.
/// Computes the element-wise average of token embeddings, using the attention mask.
/// `token_embeddings`: [batch_size, seq_len, hidden_size] flattened
/// `attention_mask`: [batch_size, seq_len] flattened. 1 for actual tokens, 0 for padding.
/// Returns shape: [batch_size, hidden_size] flattened.
pub fn mean_pooling(
    token_embeddings: &[f32],
    attention_mask: &[f32],
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let _span = info_span!(
        "mean_pooling",
        elements = token_embeddings.len(),
        fallback = 0
    )
    .entered();
    assert_eq!(
        token_embeddings.len(),
        batch_size * seq_len * hidden_size,
        "token_embeddings length mismatch"
    );
    assert_eq!(
        attention_mask.len(),
        batch_size * seq_len,
        "attention_mask length mismatch"
    );

    let mut output = vec![0.0; batch_size * hidden_size];

    for b in 0..batch_size {
        let b_offset = b * seq_len * hidden_size;
        let mask_offset = b * seq_len;
        let out_offset = b * hidden_size;

        let mut mask_sum = 0.0;

        for s in 0..seq_len {
            let mask_val = attention_mask[mask_offset + s];
            mask_sum += mask_val;

            let token_offset = b_offset + s * hidden_size;
            for h in 0..hidden_size {
                output[out_offset + h] += token_embeddings[token_offset + h] * mask_val;
            }
        }

        // Avoid division by zero
        if mask_sum > 0.0 {
            let inv_mask_sum = 1.0 / mask_sum;
            for h in 0..hidden_size {
                output[out_offset + h] *= inv_mask_sum;
            }
        }
    }

    output
}

/// Max Pooling strategy.
/// Computes the element-wise maximum of token embeddings, using the attention mask.
pub fn max_pooling(
    token_embeddings: &[f32],
    attention_mask: &[f32],
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let _span = info_span!(
        "max_pooling",
        elements = token_embeddings.len(),
        fallback = 0
    )
    .entered();
    assert_eq!(
        token_embeddings.len(),
        batch_size * seq_len * hidden_size,
        "token_embeddings length mismatch"
    );
    assert_eq!(
        attention_mask.len(),
        batch_size * seq_len,
        "attention_mask length mismatch"
    );

    let mut output = vec![f32::NEG_INFINITY; batch_size * hidden_size];

    for b in 0..batch_size {
        let b_offset = b * seq_len * hidden_size;
        let mask_offset = b * seq_len;
        let out_offset = b * hidden_size;

        let mut has_valid_tokens = false;

        for s in 0..seq_len {
            let mask_val = attention_mask[mask_offset + s];
            if mask_val > 0.0 {
                has_valid_tokens = true;
                let token_offset = b_offset + s * hidden_size;
                for h in 0..hidden_size {
                    let val = token_embeddings[token_offset + h];
                    if val > output[out_offset + h] {
                        output[out_offset + h] = val;
                    }
                }
            }
        }

        if !has_valid_tokens {
            for h in 0..hidden_size {
                output[out_offset + h] = 0.0;
            }
        }
    }

    output
}

/// CLS Pooling strategy.
/// Simply takes the first token's embedding (assumed to be the CLS token).
pub fn cls_pooling(
    token_embeddings: &[f32],
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let _span = info_span!(
        "cls_pooling",
        elements = token_embeddings.len(),
        fallback = 0
    )
    .entered();
    assert_eq!(
        token_embeddings.len(),
        batch_size * seq_len * hidden_size,
        "token_embeddings length mismatch"
    );

    let mut output = vec![0.0; batch_size * hidden_size];

    for b in 0..batch_size {
        let b_offset = b * seq_len * hidden_size;
        let out_offset = b * hidden_size;

        output[out_offset..out_offset + hidden_size]
            .copy_from_slice(&token_embeddings[b_offset..b_offset + hidden_size]);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_pooling() {
        println!("ST-EVIDENCE id=st-models-pooling checked=1 exact=true paths=pooling");

        let batch_size = 1;
        let seq_len = 3;
        let hidden_size = 2;

        let token_embeddings = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
            5.0, 6.0, // token 2 (padded)
        ];

        let attention_mask = vec![1.0, 1.0, 0.0];

        let pooled = mean_pooling(
            &token_embeddings,
            &attention_mask,
            batch_size,
            seq_len,
            hidden_size,
        );

        // Expected:
        // sum = [1+3, 2+4] = [4, 6]
        // mean = [4/2, 6/2] = [2.0, 3.0]
        assert_eq!(pooled, vec![2.0, 3.0]);
    }

    #[test]
    fn test_max_pooling() {
        println!("ST-EVIDENCE id=st-models-pooling checked=2 exact=true paths=pooling");

        let batch_size = 2;
        let seq_len = 2;
        let hidden_size = 2;

        let token_embeddings = vec![
            -1.0, 2.0, // b0 t0
            3.0, -4.0, // b0 t1
            5.0, 6.0, // b1 t0
            7.0, 8.0, // b1 t1 (padded)
        ];

        let attention_mask = vec![
            1.0, 1.0, // b0
            1.0, 0.0, // b1
        ];

        let pooled = max_pooling(
            &token_embeddings,
            &attention_mask,
            batch_size,
            seq_len,
            hidden_size,
        );

        // b0 max: [max(-1, 3), max(2, -4)] = [3.0, 2.0]
        // b1 max: [max(5, -inf), max(6, -inf)] = [5.0, 6.0] (t1 is masked)
        assert_eq!(pooled, vec![3.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cls_pooling() {
        println!("ST-EVIDENCE id=st-models-pooling checked=3 exact=true paths=pooling");

        let batch_size = 2;
        let seq_len = 3;
        let hidden_size = 1;

        let token_embeddings = vec![
            1.0, 2.0, 3.0, // b0
            4.0, 5.0, 6.0, // b1
        ];

        let pooled = cls_pooling(&token_embeddings, batch_size, seq_len, hidden_size);

        // First token of each batch
        assert_eq!(pooled, vec![1.0, 4.0]);
    }
}
