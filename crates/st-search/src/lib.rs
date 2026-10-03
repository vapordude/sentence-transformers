#[allow(dead_code, unused_macros)]
mod st_telemetry {
    macro_rules! info_span {
        ($name:expr, $($key:ident = $val:expr),* , fallback = $fallback:expr) => {
            $crate::st_telemetry::Span { name: $name }
        };
        ($name:expr, fallback = $fallback:expr) => {
            $crate::st_telemetry::Span { name: $name }
        };
        ($name:expr, $($key:ident = $val:expr),*) => {
            $crate::st_telemetry::Span { name: $name }
        };
        ($name:expr) => {
            $crate::st_telemetry::Span { name: $name }
        };
    }

    pub(crate) struct Span {
        #[allow(dead_code)]
        pub(crate) name: &'static str,
    }

    impl Span {
        pub(crate) fn entered(self) -> EnteredSpan {
            EnteredSpan { _span: self }
        }
    }

    pub(crate) struct EnteredSpan {
        #[allow(dead_code)]
        _span: Span,
    }

    pub(crate) use info_span;
}

use st_telemetry::info_span;
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub corpus_id: usize,
    pub score: f32,
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort primarily by score (descending)
        match other.score.partial_cmp(&self.score) {
            Some(Ordering::Equal) | None => {
                // Secondary by corpus_id (ascending)
                self.corpus_id.cmp(&other.corpus_id)
            }
            Some(ord) => ord,
        }
    }
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let _span = info_span!("dot_product", elements = a.len()).entered();
    let mut sum = 0.0;
    #[allow(clippy::needless_range_loop)]
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let _span = info_span!("cosine_similarity", elements = a.len()).entered();
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

pub fn exact_dot_product_top_k(query: &[f32], corpus: &[&[f32]], k: usize) -> Vec<SearchResult> {
    let _span = info_span!("exact_dot_product_top_k", corpus_size = corpus.len()).entered();
    let mut results: Vec<SearchResult> = corpus
        .iter()
        .enumerate()
        .map(|(id, doc)| SearchResult {
            corpus_id: id,
            score: dot_product(query, doc),
        })
        .collect();
    results.sort();
    results.into_iter().take(k).collect()
}

pub fn exact_cosine_top_k(query: &[f32], corpus: &[&[f32]], k: usize) -> Vec<SearchResult> {
    let _span = info_span!("exact_cosine_top_k", corpus_size = corpus.len()).entered();
    let mut results: Vec<SearchResult> = corpus
        .iter()
        .enumerate()
        .map(|(id, doc)| SearchResult {
            corpus_id: id,
            score: cosine_similarity(query, doc),
        })
        .collect();
    results.sort();
    results.into_iter().take(k).collect()
}

pub fn filter_by_threshold(results: Vec<SearchResult>, threshold: f32) -> Vec<SearchResult> {
    let _span = info_span!("filter_by_threshold", count = results.len()).entered();
    results
        .into_iter()
        .filter(|r| r.score >= threshold)
        .collect()
}

pub fn reciprocal_rank_fusion(ranked_lists: &[Vec<SearchResult>], k: usize) -> Vec<SearchResult> {
    let _span = info_span!("reciprocal_rank_fusion", num_lists = ranked_lists.len()).entered();
    let mut scores = std::collections::HashMap::new();

    for list in ranked_lists {
        for (rank, result) in list.iter().enumerate() {
            let rrf_score = 1.0 / (60.0 + (rank as f32 + 1.0));
            *scores.entry(result.corpus_id).or_insert(0.0) += rrf_score;
        }
    }

    let mut fused: Vec<SearchResult> = scores
        .into_iter()
        .map(|(corpus_id, score)| SearchResult { corpus_id, score })
        .collect();

    fused.sort();
    fused.into_iter().take(k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        println!("ST-EVIDENCE id=dot_product Testing dot product calculation");
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let res = dot_product(&a, &b);
        assert!((res - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        println!("ST-EVIDENCE id=cosine_similarity Testing cosine similarity calculation");
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let res1 = cosine_similarity(&a, &b);
        assert!((res1 - 0.0).abs() < 1e-6);

        let c = [1.0, 2.0, 3.0];
        let d = [2.0, 4.0, 6.0];
        let res2 = cosine_similarity(&c, &d);
        assert!((res2 - 1.0).abs() < 1e-6);

        let z = [0.0, 0.0, 0.0];
        let res3 = cosine_similarity(&a, &z);
        assert!((res3 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_exact_dot_product_top_k() {
        println!(
            "ST-EVIDENCE id=exact_dot_product_top_k Testing exact dot product top-k retrieval"
        );
        let query = [1.0, 1.0];
        let doc0 = [1.0, 1.0]; // score 2
        let doc1 = [2.0, 2.0]; // score 4
        let doc2 = [-1.0, -1.0]; // score -2
        let corpus: Vec<&[f32]> = vec![&doc0, &doc1, &doc2];

        let results = exact_dot_product_top_k(&query, &corpus, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].corpus_id, 1);
        assert!((results[0].score - 4.0).abs() < 1e-6);
        assert_eq!(results[1].corpus_id, 0);
        assert!((results[1].score - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_exact_cosine_top_k() {
        println!("ST-EVIDENCE id=exact_cosine_top_k Testing exact cosine top-k retrieval");
        let query = [1.0, 0.0];
        let doc0 = [1.0, 0.0]; // score 1.0
        let doc1 = [0.0, 1.0]; // score 0.0
        let doc2 = [-1.0, 0.0]; // score -1.0
        let corpus: Vec<&[f32]> = vec![&doc0, &doc1, &doc2];

        let results = exact_cosine_top_k(&query, &corpus, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].corpus_id, 0);
        assert!((results[0].score - 1.0).abs() < 1e-6);
        assert_eq!(results[1].corpus_id, 1);
        assert!((results[1].score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_filter_by_threshold() {
        println!("ST-EVIDENCE id=filter_by_threshold Testing similarity threshold filtering");
        let results = vec![
            SearchResult {
                corpus_id: 1,
                score: 0.9,
            },
            SearchResult {
                corpus_id: 2,
                score: 0.5,
            },
            SearchResult {
                corpus_id: 3,
                score: 0.2,
            },
        ];

        let filtered = filter_by_threshold(results, 0.5);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].corpus_id, 1);
        assert_eq!(filtered[1].corpus_id, 2);
    }

    #[test]
    fn test_reciprocal_rank_fusion() {
        println!("ST-EVIDENCE id=reciprocal_rank_fusion Testing reciprocal rank fusion (RRF)");
        let list1 = vec![
            SearchResult {
                corpus_id: 10,
                score: 0.9,
            },
            SearchResult {
                corpus_id: 20,
                score: 0.8,
            },
            SearchResult {
                corpus_id: 30,
                score: 0.7,
            },
        ];

        let list2 = vec![
            SearchResult {
                corpus_id: 30,
                score: 1.0,
            },
            SearchResult {
                corpus_id: 10,
                score: 0.9,
            },
            SearchResult {
                corpus_id: 40,
                score: 0.8,
            },
        ];

        let fused = reciprocal_rank_fusion(&[list1, list2], 3);
        assert_eq!(fused.len(), 3);
        // doc 10: rank 1 (1/61) + rank 2 (1/62) = 0.01639 + 0.01612 = 0.03251
        // doc 30: rank 3 (1/63) + rank 1 (1/61) = 0.01587 + 0.01639 = 0.03226
        // doc 20: rank 2 (1/62) = 0.01612
        // doc 40: rank 3 (1/63) = 0.01587

        assert_eq!(fused[0].corpus_id, 10);
        assert_eq!(fused[1].corpus_id, 30);
        assert_eq!(fused[2].corpus_id, 20);
    }
}
