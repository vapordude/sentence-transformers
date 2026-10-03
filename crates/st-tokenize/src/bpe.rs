use std::collections::HashMap;

use crate::vocab::Vocab;

/// Byte-Pair Encoding (BPE) tokenizer algorithm.
pub struct BPETokenizer {
    pub vocab: Vocab,
    pub bpe_ranks: HashMap<(String, String), u32>,
    pub unk_token: String,
}

impl BPETokenizer {
    pub fn new(vocab: Vocab, bpe_ranks: HashMap<(String, String), u32>, unk_token: String) -> Self {
        let _span = info_span!(
            "BPETokenizer::new",
            elements = bpe_ranks.len(),
            fallback = 0
        )
        .entered();
        Self {
            vocab,
            bpe_ranks,
            unk_token,
        }
    }

    /// Gets pairs of adjacent symbols in a word.
    fn get_pairs(word: &[String]) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        if word.len() < 2 {
            return pairs;
        }
        for i in 0..word.len() - 1 {
            pairs.push((word[i].clone(), word[i + 1].clone()));
        }
        pairs
    }

    /// Tokenizes a single word into BPE subwords.
    pub fn tokenize_word(&self, word: &str) -> Vec<String> {
        let _span = info_span!("bpe_tokenize_word", elements = word.len(), fallback = 0).entered();

        // Initial split into characters (usually represented by strings or bytes in actual BPE).
        // For simplicity in text, we map each char to a String. In GPT2, this involves a byte encoder.
        let mut current_word: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        if current_word.is_empty() {
            return vec![];
        }

        loop {
            let pairs = Self::get_pairs(&current_word);
            if pairs.is_empty() {
                break;
            }

            // Find the pair with the lowest rank
            let mut min_rank = u32::MAX;
            let mut best_pair = None;

            for pair in pairs {
                if let Some(&rank) = self.bpe_ranks.get(&pair) {
                    if rank < min_rank {
                        min_rank = rank;
                        best_pair = Some(pair);
                    }
                }
            }

            // If no pairs can be merged, we are done
            let best_pair = match best_pair {
                Some(p) => p,
                None => break,
            };

            // Merge the best pair
            let mut new_word = Vec::new();
            let mut i = 0;
            while i < current_word.len() {
                if i < current_word.len() - 1
                    && current_word[i] == best_pair.0
                    && current_word[i + 1] == best_pair.1
                {
                    new_word.push(format!("{}{}", best_pair.0, best_pair.1));
                    i += 2;
                } else {
                    new_word.push(current_word[i].clone());
                    i += 1;
                }
            }
            current_word = new_word;
        }

        // Map final splits to vocab, or UNK if not present
        current_word
            .into_iter()
            .map(|w| {
                if self.vocab.token_to_id.contains_key(&w) {
                    w
                } else {
                    self.unk_token.clone()
                }
            })
            .collect()
    }

    /// Tokenizes a sequence of whitespace-separated tokens.
    pub fn tokenize(&self, tokens: &[String]) -> Vec<String> {
        let _span = info_span!(
            "BPETokenizer::tokenize",
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

    #[test]
    fn test_bpe_tokenize() {
        println!("ST-EVIDENCE id=st-tokenize-bpe checked=1 exact=true paths=tokens");
        let mut t2i = HashMap::new();
        t2i.insert("[UNK]".to_string(), 0);
        t2i.insert("h".to_string(), 1);
        t2i.insert("e".to_string(), 2);
        t2i.insert("l".to_string(), 3);
        t2i.insert("o".to_string(), 4);
        t2i.insert("he".to_string(), 5);
        t2i.insert("ll".to_string(), 6);
        t2i.insert("hello".to_string(), 7);

        let vocab = Vocab::new(t2i, "[UNK]".to_string());

        let mut ranks = HashMap::new();
        ranks.insert(("h".to_string(), "e".to_string()), 0); // merge h, e -> he
        ranks.insert(("l".to_string(), "l".to_string()), 1); // merge l, l -> ll
        ranks.insert(("he".to_string(), "ll".to_string()), 2); // merge he, ll -> hell
                                                               // 'o' merges with 'hell' is not ranked

        let bpe = BPETokenizer::new(vocab, ranks, "[UNK]".to_string());

        let tokens = bpe.tokenize(&["hello".to_string()]);
        // After merges: [h, e, l, l, o] -> [he, l, l, o] -> [he, ll, o] -> [hell, o]
        // "hell" is not in vocab, maps to [UNK]. "o" is in vocab.
        assert_eq!(tokens, vec!["[UNK]", "o"]);
    }
}
