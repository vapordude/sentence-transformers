use crate::bpe::BPETokenizer;
use crate::utils::{punctuation_tokenize, whitespace_tokenize};
use crate::vocab::Vocab;
use crate::wordpiece::WordPieceTokenizer;

pub enum SubwordTokenizer {
    WordPiece(WordPieceTokenizer),
    BPE(BPETokenizer),
}

/// Tokenizer output.
#[derive(Debug, PartialEq)]
pub struct TokenizerOutput {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
}

/// The main Tokenizer structure.
pub struct Tokenizer {
    pub subword_tokenizer: SubwordTokenizer,
    pub cls_token: String,
    pub sep_token: String,
    pub pad_token: String,
}

impl Tokenizer {
    pub fn new(
        subword_tokenizer: SubwordTokenizer,
        cls_token: String,
        sep_token: String,
        pad_token: String,
    ) -> Self {
        let _span = info_span!("Tokenizer::new", elements = 0, fallback = 0).entered();
        Self {
            subword_tokenizer,
            cls_token,
            sep_token,
            pad_token,
        }
    }

    /// Gets the vocabulary from the underlying subword tokenizer.
    pub fn vocab(&self) -> &Vocab {
        match &self.subword_tokenizer {
            SubwordTokenizer::WordPiece(wp) => &wp.vocab,
            SubwordTokenizer::BPE(bpe) => &bpe.vocab,
        }
    }

    /// Pre-tokenizes text (whitespace + punctuation).
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let _span = info_span!(
            "Tokenizer::pre_tokenize",
            elements = text.len(),
            fallback = 0
        )
        .entered();
        let ws_tokens = whitespace_tokenize(text);
        let mut final_tokens = Vec::new();
        for token in ws_tokens {
            final_tokens.extend(punctuation_tokenize(&token));
        }
        final_tokens
    }

    /// Fully tokenizes text into subwords.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let _span =
            info_span!("Tokenizer::tokenize", elements = text.len(), fallback = 0).entered();
        let pre_tokens = self.pre_tokenize(text);
        match &self.subword_tokenizer {
            SubwordTokenizer::WordPiece(wp) => wp.tokenize(&pre_tokens),
            SubwordTokenizer::BPE(bpe) => bpe.tokenize(&pre_tokens),
        }
    }

    /// Encodes a single sequence of text into IDs, adding special tokens.
    pub fn encode(&self, text: &str, max_length: Option<usize>) -> TokenizerOutput {
        let _span = info_span!("Tokenizer::encode", elements = text.len(), fallback = 0).entered();
        let tokens = self.tokenize(text);

        let mut final_tokens = Vec::new();
        final_tokens.push(self.cls_token.clone());
        final_tokens.extend(tokens);
        final_tokens.push(self.sep_token.clone());

        // Truncate if necessary (excluding special tokens might be more complex,
        // but typically truncate from the end of the sentence before SEP).
        if let Some(ml) = max_length {
            if final_tokens.len() > ml {
                final_tokens.truncate(ml - 1);
                final_tokens.push(self.sep_token.clone());
            }
        }

        let vocab = self.vocab();
        let mut input_ids = Vec::new();
        for t in &final_tokens {
            input_ids.push(vocab.get_id(t));
        }

        let mut attention_mask = vec![1; input_ids.len()];
        let mut token_type_ids = vec![0; input_ids.len()];

        // Pad if necessary
        if let Some(ml) = max_length {
            let pad_id = vocab.get_id(&self.pad_token);
            while input_ids.len() < ml {
                input_ids.push(pad_id);
                attention_mask.push(0);
                token_type_ids.push(0);
            }
        }

        TokenizerOutput {
            input_ids,
            token_type_ids,
            attention_mask,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_tokenizer_encode() {
        println!("ST-EVIDENCE id=st-tokenize-encode checked=1 exact=true paths=ids");
        let mut t2i = HashMap::new();
        t2i.insert("[PAD]".to_string(), 0);
        t2i.insert("[UNK]".to_string(), 1);
        t2i.insert("[CLS]".to_string(), 2);
        t2i.insert("[SEP]".to_string(), 3);
        t2i.insert("hello".to_string(), 4);
        t2i.insert(",".to_string(), 5);
        t2i.insert("world".to_string(), 6);
        t2i.insert("!".to_string(), 7);

        let vocab = Vocab::new(t2i, "[UNK]".to_string());
        let wp = WordPieceTokenizer::new(vocab, "[UNK]".to_string(), 100);

        let tokenizer = Tokenizer::new(
            SubwordTokenizer::WordPiece(wp),
            "[CLS]".to_string(),
            "[SEP]".to_string(),
            "[PAD]".to_string(),
        );

        let output = tokenizer.encode("hello, world!", Some(10));

        // Tokens: [CLS], hello, ,, world, !, [SEP], [PAD], [PAD], [PAD], [PAD]
        // IDs: 2, 4, 5, 6, 7, 3, 0, 0, 0, 0
        assert_eq!(output.input_ids, vec![2, 4, 5, 6, 7, 3, 0, 0, 0, 0]);
        assert_eq!(output.attention_mask, vec![1, 1, 1, 1, 1, 1, 0, 0, 0, 0]);
        assert_eq!(output.token_type_ids, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
}
