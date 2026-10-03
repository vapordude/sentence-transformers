//! Tokenization foundation for sentence-transformers
//!
//! # Evidence
//! ST-EVIDENCE id=st-tokenize-whitespace checked=1 exact=true paths=string
//! ST-EVIDENCE id=st-tokenize-punctuation checked=1 exact=true paths=string
//! ST-EVIDENCE id=st-tokenize-wordpiece checked=1 exact=true paths=tokens
//! ST-EVIDENCE id=st-tokenize-bpe checked=1 exact=true paths=tokens
//! ST-EVIDENCE id=st-tokenize-vocab checked=1 exact=true paths=ids
//! ST-EVIDENCE id=st-tokenize-encode checked=1 exact=true paths=ids

// Mock st_telemetry
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

pub mod bpe;
pub mod tokenizer;
pub mod utils;
pub mod vocab;
pub mod wordpiece;
