use crate::info_span;
use crate::similarity::pairwise_token_similarity;

/// Computes the ColBERT / MaxSim score between a query and a document.
///
/// For each query token, finds the maximum similarity to any document token.
/// The overall score is the sum of these maximum similarities.
///
/// `query_embeddings`: Shape (N, D) where N is the number of query tokens.
/// `doc_embeddings`: Shape (M, D) where M is the number of document tokens.
/// `query_mask`: Optional shape (N,), where 1 means keep and 0 means ignore. Uses u8.
/// `doc_mask`: Optional shape (M,), where 1 means keep and 0 means ignore. Uses u8.
/// `length_normalize`: If true, divides the score by the number of unmasked query tokens.
pub fn colbert_maxsim_score(
    query_embeddings: &[f32],
    doc_embeddings: &[f32],
    dim: usize,
    query_mask: Option<&[u8]>,
    doc_mask: Option<&[u8]>,
    length_normalize: bool,
) -> f32 {
    let _span = info_span!("colbert_maxsim_score").enter();

    if query_embeddings.is_empty() || doc_embeddings.is_empty() || dim == 0 {
        return 0.0;
    }

    #[allow(clippy::manual_is_multiple_of)]
    if query_embeddings.len() % dim != 0 || doc_embeddings.len() % dim != 0 {
        panic!("Embedding length is not a multiple of the dimension");
    }

    let n_query_tokens = query_embeddings.len() / dim;
    let n_doc_tokens = doc_embeddings.len() / dim;

    if let Some(qm) = query_mask {
        if qm.len() != n_query_tokens {
            panic!("query_mask length does not match number of query tokens");
        }
    }
    if let Some(dm) = doc_mask {
        if dm.len() != n_doc_tokens {
            panic!("doc_mask length does not match number of document tokens");
        }
    }

    let similarities = pairwise_token_similarity(query_embeddings, doc_embeddings, dim);

    let mut total_score = 0.0;
    let mut valid_query_tokens = 0;

    #[allow(clippy::needless_range_loop)]
    for i in 0..n_query_tokens {
        // Skip masked query tokens
        if let Some(qm) = query_mask {
            if qm[i] == 0 {
                continue;
            }
        }

        valid_query_tokens += 1;
        let mut max_sim = f32::NEG_INFINITY;
        let mut has_valid_doc_token = false;

        for j in 0..n_doc_tokens {
            // Skip masked document tokens
            if let Some(dm) = doc_mask {
                if dm[j] == 0 {
                    continue;
                }
            }
            has_valid_doc_token = true;
            let sim = similarities[i * n_doc_tokens + j];
            if sim > max_sim {
                max_sim = sim;
            }
        }

        if has_valid_doc_token {
            total_score += max_sim;
        } else {
            // If all document tokens are masked, the maxsim for this query token is effectively 0
            // Or we could panic, but adding 0 is more robust.
        }
    }

    if length_normalize && valid_query_tokens > 0 {
        total_score / (valid_query_tokens as f32)
    } else {
        total_score
    }
}
