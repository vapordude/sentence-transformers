use crate::vocab::Vocab;

/// WordPiece tokenizer algorithm.
/// Longest-match-first strategy.
pub struct WordPieceTokenizer {
    pub vocab: Vocab,
    pub unk_token: String,
    pub max_input_chars_per_word: usize,
}

impl WordPieceTokenizer {
    pub fn new(vocab: Vocab, unk_token: String, max_input_chars_per_word: usize) -> Self {
        let _span = info_span!("WordPieceTokenizer::new", elements = 0, fallback = 0).entered();
        Self {
            vocab,
            unk_token,
            max_input_chars_per_word,
        }
    }

    /// Tokenizes a single word into WordPiece tokens.
    pub fn tokenize_word(&self, word: &str) -> Vec<String> {
        let _span = info_span!("tokenize_word", elements = word.len(), fallback = 0).entered();
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > self.max_input_chars_per_word {
            return vec![self.unk_token.clone()];
        }

        let mut is_bad = false;
        let mut start = 0;
        let mut sub_tokens = Vec::new();

        while start < chars.len() {
            let mut end = chars.len();
            let mut cur_substr = None;

            while start < end {
                let mut substr: String = chars[start..end].iter().collect();
                if start > 0 {
                    substr = format!("##{}", substr);
                }

                if self.vocab.token_to_id.contains_key(&substr) {
                    cur_substr = Some(substr);
                    break;
                }
                end -= 1;
            }

            if let Some(sub) = cur_substr {
                sub_tokens.push(sub);
                start = end;
            } else {
                is_bad = true;
                break;
            }
        }

        if is_bad {
            vec![self.unk_token.clone()]
        } else {
            sub_tokens
        }
    }

    /// Tokenizes a sequence of whitespace-separated tokens.
    pub fn tokenize(&self, tokens: &[String]) -> Vec<String> {
        let _span = info_span!(
            "WordPieceTokenizer::tokenize",
            elements = tokens.len(),
            fallback = 0
        )
        .entered();
        let mut output_tokens = Vec::new();
        for token in tokens {
            output_tokens.extend(self.tokenize_word(token));
        }
        output_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_wordpiece_tokenize() {
        println!("ST-EVIDENCE id=st-tokenize-wordpiece checked=1 exact=true paths=tokens");
        let mut t2i = HashMap::new();
        t2i.insert("[UNK]".to_string(), 0);
        t2i.insert("un".to_string(), 1);
        t2i.insert("##aff".to_string(), 2);
        t2i.insert("##able".to_string(), 3);
        t2i.insert("cat".to_string(), 4);

        let vocab = Vocab::new(t2i, "[UNK]".to_string());
        let wp = WordPieceTokenizer::new(vocab, "[UNK]".to_string(), 100);

        let tokens = wp.tokenize(&[
            "unaffable".to_string(),
            "cat".to_string(),
            "unknown".to_string(),
        ]);
        assert_eq!(tokens, vec!["un", "##aff", "##able", "cat", "[UNK]"]);
    }
}
