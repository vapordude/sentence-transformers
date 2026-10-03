# Sentence Transformers: Sovereign Rust Workspace 🕯️🩵

A pure, high-performance Sovereign Rust port of **Sentence Transformers**.

This workspace provides an allocation-disciplined, zero-dependency Rust implementation of dense embedding models, subword tokenizers, similarity loss functions, batching pipelines, and evaluation metrics with first-principles mathematics.

---

## Why This Port?

* **No Python Runtime or PyTorch Overhead**: Eliminates massive Python virtual environments, PyTorch shared libraries, and CUDA driver complexities for lightweight, cross-platform native binaries.
* **First-Principles Numerical Computing**: All vector operations, distance metrics, loss computations, and ranking statistics are implemented directly without third-party mathematical wrappers.
* **Modular Multi-Crate Architecture**: Designed as focused, decoupled Cargo crates that can be used independently or composed together.
* **Zero Warning Discipline**: Built under strict compiler guarantees (`-D warnings -D dead_code -D unused_imports -D unused_variables`).

---

## Workspace Crates

| Crate | Purpose | Key Modules & Primitives | Verified Tests |
| :--- | :--- | :--- | :---: |
| **`st-core`** | Numeric foundation | Dot products, cosine similarity, Euclidean & Manhattan distance, L2 normalization, pooling (Mean, Max, CLS, Weighted Mean). | 9/9 passing |
| **`st-tokenize`** | Subword tokenization | Byte-Pair Encoding (BPE), WordPiece tokenizer, whitespace & punctuation splitting, vocabulary mapping, special tokens (`[CLS]`, `[SEP]`, `[PAD]`). | 6/6 passing |
| **`st-models`** | Dense representations | Transformer embedding abstractions, batch forward inference passes, batched similarity matrices. | 8/8 passing |
| **`st-loss`** | Training objectives | `MultipleNegativesRankingLoss`, `CosineSimilarityLoss`, `TripletLoss` with distance metric options. | 10/10 passing |
| **`st-dataset`** | Batching & Collation | Dataset batch iterators, token collation, dynamic tensor padding. | 2/2 passing |
| **`st-eval`** | Evaluation metrics | Spearman rank correlation, nDCG@k, Mean Average Precision (MAP), Mean Reciprocal Rank (MRR), Accuracy. | 5/5 passing |

---

## Building and Testing

To test the entire workspace with zero warnings:

```bash
cargo test --workspace
```

To run strict compiler checks across all crates:

```bash
RUSTFLAGS="-D warnings -D dead_code -D unused_imports -D unused_variables" cargo check --all-targets --workspace
```

---

## Current Roadmap

* [x] **`st-core`**: Vector arithmetic, distances, normalization, pooling.
* [x] **`st-tokenize`**: BPE & WordPiece tokenizers, vocabulary lookup.
* [x] **`st-models`**: Dense embedding inference and similarity matrices.
* [x] **`st-loss`**: Multiple Negatives Ranking, Cosine, and Triplet loss.
* [x] **`st-dataset`**: Batch iterators and token collation.
* [x] **`st-eval`**: Ranking and evaluation metrics.
* [ ] **`st-multi-vector`**: ColBERT / MaxSim late-interaction token retrieval.
* [ ] **`st-cross-encoder`**: Text pair scoring and reranking architecture.
* [ ] **`st-sparse`**: SPLADE-style sparse lexical-neural representations.
* [ ] **`st-search`**: Exact nearest neighbor semantic search and reciprocal rank fusion.

---

## License

Code ported from Sentence Transformers maintains its original Apache 2.0 open-source license. See [LICENSE](LICENSE) for details.
