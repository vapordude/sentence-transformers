/// Mean Pooling strategy.
/// Computes the element-wise average of token embeddings, using the attention mask.
/// `token_embeddings`: [batch_size, seq_len, hidden_size] flattened
/// `attention_mask`: [batch_size, seq_len] flattened
pub fn mean_pooling(
    token_embeddings: &[f32],
    attention_mask: &[u8],
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let _span = crate::info_span!(
        "mean_pooling",
        elements = token_embeddings.len(),
        fallback = 0
    )
    .entered();
    assert_eq!(token_embeddings.len(), batch_size * seq_len * hidden_size);
    assert_eq!(attention_mask.len(), batch_size * seq_len);

    let mut output = vec![0.0; batch_size * hidden_size];

    for b in 0..batch_size {
        let mut sum_mask = 0.0;
        let mut row_sums = vec![0.0; hidden_size];

        for s in 0..seq_len {
            let mask_val = attention_mask[b * seq_len + s] as f32;
            sum_mask += mask_val;

            for h in 0..hidden_size {
                let val = token_embeddings[(b * seq_len + s) * hidden_size + h];
                row_sums[h] += val * mask_val;
            }
        }

        let sum_mask = sum_mask.max(1e-9); // Prevent division by zero
        for h in 0..hidden_size {
            output[b * hidden_size + h] = row_sums[h] / sum_mask;
        }
    }

    output
}

/// Max Pooling strategy.
pub fn max_pooling(
    token_embeddings: &[f32],
    attention_mask: &[u8],
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let _span = crate::info_span!(
        "max_pooling",
        elements = token_embeddings.len(),
        fallback = 0
    )
    .entered();
    assert_eq!(token_embeddings.len(), batch_size * seq_len * hidden_size);
    assert_eq!(attention_mask.len(), batch_size * seq_len);

    let mut output = vec![f32::NEG_INFINITY; batch_size * hidden_size];

    for b in 0..batch_size {
        let mut has_unmasked = false;
        for s in 0..seq_len {
            if attention_mask[b * seq_len + s] == 1 {
                has_unmasked = true;
                for h in 0..hidden_size {
                    let val = token_embeddings[(b * seq_len + s) * hidden_size + h];
                    let out_idx = b * hidden_size + h;
                    if val > output[out_idx] {
                        output[out_idx] = val;
                    }
                }
            }
        }

        // If all tokens are masked, output 0.0
        if !has_unmasked {
            for h in 0..hidden_size {
                output[b * hidden_size + h] = 0.0;
            }
        }
    }

    output
}

/// CLS Pooling strategy.
/// Simply takes the embedding of the first token (usually the [CLS] token).
pub fn cls_pooling(
    token_embeddings: &[f32],
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let _span = crate::info_span!(
        "cls_pooling",
        elements = token_embeddings.len(),
        fallback = 0
    )
    .entered();
    assert_eq!(token_embeddings.len(), batch_size * seq_len * hidden_size);

    let mut output = vec![0.0; batch_size * hidden_size];

    for b in 0..batch_size {
        for h in 0..hidden_size {
            output[b * hidden_size + h] = token_embeddings[b * seq_len * hidden_size + h];
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_pooling() {
        // 1 batch, 2 seq, 2 hidden
        let token_embeddings = vec![
            1.0, 2.0, // seq 0
            3.0, 4.0, // seq 1
        ];
        let attention_mask = vec![1, 0]; // Mask second token

        let output = mean_pooling(&token_embeddings, &attention_mask, 1, 2, 2);
        assert_eq!(output, vec![1.0, 2.0]);
        println!("ST-EVIDENCE id=st-models-pooling checked=1 exact=true paths=pooling");
    }

    #[test]
    fn test_max_pooling() {
        let token_embeddings = vec![
            1.0, 5.0, // seq 0
            3.0, 4.0, // seq 1
        ];
        let attention_mask = vec![1, 1];

        let output = max_pooling(&token_embeddings, &attention_mask, 1, 2, 2);
        assert_eq!(output, vec![3.0, 5.0]);
        println!("ST-EVIDENCE id=st-models-pooling checked=2 exact=true paths=pooling");
    }

    #[test]
    fn test_cls_pooling() {
        let token_embeddings = vec![
            1.0, 5.0, // seq 0
            3.0, 4.0, // seq 1
        ];

        let output = cls_pooling(&token_embeddings, 1, 2, 2);
        assert_eq!(output, vec![1.0, 5.0]);
        println!("ST-EVIDENCE id=st-models-pooling checked=3 exact=true paths=pooling");
    }
}
