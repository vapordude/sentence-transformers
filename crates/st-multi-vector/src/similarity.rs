use crate::info_span;

/// Computes the pairwise token similarity matrix between query and document token embeddings.
///
/// `query_embeddings`: Shape (N, D) where N is the number of query tokens.
/// `doc_embeddings`: Shape (M, D) where M is the number of document tokens.
///
/// Returns:
/// A vector of length N*M representing the flattened N x M similarity matrix,
/// where element (i * M + j) is the dot product of query token i and document token j.
pub fn pairwise_token_similarity(
    query_embeddings: &[f32],
    doc_embeddings: &[f32],
    dim: usize,
) -> Vec<f32> {
    let _span = info_span!("pairwise_token_similarity").enter();

    if query_embeddings.is_empty() || doc_embeddings.is_empty() || dim == 0 {
        return vec![];
    }

    #[allow(clippy::manual_is_multiple_of)]
    if query_embeddings.len() % dim != 0 || doc_embeddings.len() % dim != 0 {
        panic!("Embedding length is not a multiple of the dimension");
    }

    let n_query_tokens = query_embeddings.len() / dim;
    let n_doc_tokens = doc_embeddings.len() / dim;

    let mut similarities = vec![0.0; n_query_tokens * n_doc_tokens];

    #[allow(clippy::needless_range_loop)]
    for i in 0..n_query_tokens {
        for j in 0..n_doc_tokens {
            let mut dot_product = 0.0;
            for k in 0..dim {
                dot_product += query_embeddings[i * dim + k] * doc_embeddings[j * dim + k];
            }
            similarities[i * n_doc_tokens + j] = dot_product;
        }
    }

    similarities
}
