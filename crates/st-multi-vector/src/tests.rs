use crate::colbert::colbert_maxsim_score;
use crate::similarity::pairwise_token_similarity;

#[test]
fn test_pairwise_token_similarity() {
    println!("ST-EVIDENCE id=pairwise_sim exactness");
    let query_embeddings = vec![1.0, 2.0, 3.0, 4.0];
    let doc_embeddings = vec![1.0, 1.0, -1.0, 2.0, 0.0, 1.0];
    let dim = 2;
    let expected = vec![
        1.0 * 1.0 + 2.0 * 1.0,  // q0, d0 = 3.0
        1.0 * -1.0 + 2.0 * 2.0, // q0, d1 = 3.0
        1.0 * 0.0 + 2.0 * 1.0,  // q0, d2 = 2.0
        3.0 * 1.0 + 4.0 * 1.0,  // q1, d0 = 7.0
        3.0 * -1.0 + 4.0 * 2.0, // q1, d1 = 5.0
        3.0 * 0.0 + 4.0 * 1.0,  // q1, d2 = 4.0
    ];
    let result = pairwise_token_similarity(&query_embeddings, &doc_embeddings, dim);
    assert_eq!(result, expected);
}

#[test]
fn test_pairwise_token_similarity_panic() {
    println!("ST-EVIDENCE id=pairwise_sim_panic invalid_len");
    let result =
        std::panic::catch_unwind(|| pairwise_token_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0], 2));
    assert!(result.is_err());
}

#[test]
fn test_colbert_maxsim_score() {
    println!("ST-EVIDENCE id=maxsim_exactness exactness");
    // q0: [1.0, 2.0]
    // q1: [3.0, 4.0]
    let query_embeddings = vec![1.0, 2.0, 3.0, 4.0];
    // d0: [1.0, 1.0] -> q0_sim = 3.0, q1_sim = 7.0
    // d1: [-1.0, 2.0] -> q0_sim = 3.0, q1_sim = 5.0
    // d2: [0.0, 1.0] -> q0_sim = 2.0, q1_sim = 4.0
    let doc_embeddings = vec![1.0, 1.0, -1.0, 2.0, 0.0, 1.0];
    let dim = 2;

    // q0 max = max(3.0, 3.0, 2.0) = 3.0
    // q1 max = max(7.0, 5.0, 4.0) = 7.0
    // total = 3.0 + 7.0 = 10.0
    let result_unnorm =
        colbert_maxsim_score(&query_embeddings, &doc_embeddings, dim, None, None, false);
    assert_eq!(result_unnorm, 10.0);

    let result_norm =
        colbert_maxsim_score(&query_embeddings, &doc_embeddings, dim, None, None, true);
    assert_eq!(result_norm, 10.0 / 2.0); // 5.0
}

#[test]
fn test_colbert_maxsim_score_with_mask() {
    println!("ST-EVIDENCE id=maxsim_mask exactness");
    let query_embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let doc_embeddings = vec![1.0, 1.0, -1.0, 2.0, 0.0, 1.0];
    let dim = 2;

    // Mask query token 1, mask document token 0
    let query_mask: Vec<u8> = vec![1, 0, 1];
    let doc_mask: Vec<u8> = vec![0, 1, 1];

    // q0: [1.0, 2.0]
    // q2: [5.0, 6.0]
    // Valid docs: d1 [-1.0, 2.0], d2 [0.0, 1.0]
    //
    // q0 -> d1: 1.0*-1.0 + 2.0*2.0 = 3.0
    // q0 -> d2: 1.0*0.0 + 2.0*1.0 = 2.0
    // q0 max = 3.0
    //
    // q2 -> d1: 5.0*-1.0 + 6.0*2.0 = 7.0
    // q2 -> d2: 5.0*0.0 + 6.0*1.0 = 6.0
    // q2 max = 7.0
    //
    // total = 10.0
    let result_unnorm = colbert_maxsim_score(
        &query_embeddings,
        &doc_embeddings,
        dim,
        Some(&query_mask),
        Some(&doc_mask),
        false,
    );
    assert_eq!(result_unnorm, 10.0);

    let result_norm = colbert_maxsim_score(
        &query_embeddings,
        &doc_embeddings,
        dim,
        Some(&query_mask),
        Some(&doc_mask),
        true,
    );
    // 2 valid query tokens
    assert_eq!(result_norm, 10.0 / 2.0);
}

#[test]
fn test_colbert_maxsim_score_panic() {
    println!("ST-EVIDENCE id=maxsim_panic invalid_len");
    let result = std::panic::catch_unwind(|| {
        colbert_maxsim_score(&[1.0, 2.0, 3.0], &[1.0, 2.0], 2, None, None, false)
    });
    assert!(result.is_err());
}
