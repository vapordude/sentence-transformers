use std::collections::HashMap;

/// Mock telemetry module
pub mod st_telemetry {
    #[allow(dead_code)]
    pub struct Span;

    #[macro_export]
    macro_rules! info_span {
        ($name:expr, $($field:tt)*) => {
            $crate::st_telemetry::Span
        };
        ($name:expr) => {
            $crate::st_telemetry::Span
        };
    }
}

/// Represents a sparse vector, typically for SPLADE-style representations.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseVector {
    /// Indices of non-zero elements, expected to be sorted in ascending order.
    pub indices: Vec<usize>,
    /// Values corresponding to the non-zero indices.
    pub values: Vec<f32>,
}

impl SparseVector {
    /// Creates a new SparseVector.
    ///
    /// # Panics
    /// Panics if `indices` and `values` have different lengths.
    pub fn new(indices: Vec<usize>, values: Vec<f32>) -> Self {
        let _span = info_span!("SparseVector::new", indices_len = indices.len());
        assert_eq!(
            indices.len(),
            values.len(),
            "indices and values must have the same length"
        );
        Self { indices, values }
    }
}

/// Computes the dot product of two sparse vectors.
///
/// Both vectors must have their indices sorted in ascending order.
pub fn sparse_dot_product(a: &SparseVector, b: &SparseVector) -> f32 {
    let _span = info_span!("sparse_dot_product");
    let mut i = 0;
    let mut j = 0;
    let mut dot = 0.0;

    while i < a.indices.len() && j < b.indices.len() {
        if a.indices[i] == b.indices[j] {
            dot += a.values[i] * b.values[j];
            i += 1;
            j += 1;
        } else if a.indices[i] < b.indices[j] {
            i += 1;
        } else {
            j += 1;
        }
    }

    dot
}

/// A simple inverted index for sparse vectors.
#[derive(Debug, Clone, Default)]
pub struct InvertedIndex {
    /// Maps a token index to a list of (document_id, weight).
    pub posting_lists: HashMap<usize, Vec<(usize, f32)>>,
}

impl InvertedIndex {
    /// Creates a new, empty inverted index.
    pub fn new() -> Self {
        let _span = info_span!("InvertedIndex::new");
        Self {
            posting_lists: HashMap::new(),
        }
    }

    /// Adds a document represented as a sparse vector to the index.
    pub fn add_document(&mut self, doc_id: usize, vec: &SparseVector) {
        let _span = info_span!("InvertedIndex::add_document", doc_id = doc_id);
        for (i, &idx) in vec.indices.iter().enumerate() {
            let weight = vec.values[i];
            self.posting_lists
                .entry(idx)
                .or_default()
                .push((doc_id, weight));
        }
    }

    /// Retrieves the posting list for a given token index.
    pub fn get_posting_list(&self, token_idx: usize) -> Option<&Vec<(usize, f32)>> {
        let _span = info_span!("InvertedIndex::get_posting_list", token_idx = token_idx);
        self.posting_lists.get(&token_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_creation() {
        let vec = SparseVector::new(vec![1, 5, 9], vec![0.5, 0.8, 1.2]);
        assert_eq!(vec.indices, vec![1, 5, 9]);
        assert_eq!(vec.values, vec![0.5, 0.8, 1.2]);
        println!("ST-EVIDENCE id=sparse_vector_creation_success");
    }

    #[test]
    fn test_sparse_vector_creation_panic() {
        let result = std::panic::catch_unwind(|| {
            SparseVector::new(vec![1, 5], vec![0.5, 0.8, 1.2]);
        });
        assert!(
            result.is_err(),
            "SparseVector::new should panic on length mismatch"
        );
        println!("ST-EVIDENCE id=sparse_vector_creation_panic");
    }

    #[test]
    fn test_sparse_dot_product() {
        let a = SparseVector::new(vec![1, 3, 5, 7], vec![1.0, 2.0, 3.0, 4.0]);
        let b = SparseVector::new(vec![0, 3, 5, 9], vec![0.5, 1.5, 2.5, 3.5]);

        // Intersecting indices: 3 and 5
        // Products: 2.0 * 1.5 + 3.0 * 2.5 = 3.0 + 7.5 = 10.5
        let dot = sparse_dot_product(&a, &b);
        assert!((dot - 10.5).abs() < f32::EPSILON);
        println!("ST-EVIDENCE id=sparse_dot_product_computation");
    }

    #[test]
    fn test_inverted_index() {
        let mut index = InvertedIndex::new();

        let doc1 = SparseVector::new(vec![1, 4, 8], vec![0.1, 0.4, 0.8]);
        let doc2 = SparseVector::new(vec![4, 8, 10], vec![0.2, 0.5, 1.0]);

        index.add_document(100, &doc1);
        index.add_document(101, &doc2);

        let list_1 = index.get_posting_list(1).unwrap();
        assert_eq!(list_1.len(), 1);
        assert_eq!(list_1[0], (100, 0.1));

        let list_4 = index.get_posting_list(4).unwrap();
        assert_eq!(list_4.len(), 2);
        assert_eq!(list_4[0], (100, 0.4));
        assert_eq!(list_4[1], (101, 0.2));

        let list_99 = index.get_posting_list(99);
        assert!(list_99.is_none());

        println!("ST-EVIDENCE id=inverted_index_operations");
    }
}
