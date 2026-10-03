//! Rust cross-encoder architecture.
//!
//! # Evidence
//! ST-EVIDENCE id=st-cross-encoder-heads checked=2 exact=true paths=heads
//! ST-EVIDENCE id=st-cross-encoder-logit-score checked=2 exact=true paths=logit_score

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

pub mod heads;
pub mod logit_score;
