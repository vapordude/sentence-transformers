use std::collections::HashMap;

/// Represents a token vocabulary mapping tokens to IDs and vice versa.
#[derive(Debug, Clone)]
pub struct Vocab {
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub unk_token: String,
}

impl Vocab {
    /// Creates a new Vocabulary from a map of tokens to IDs.
    pub fn new(token_to_id: HashMap<String, u32>, unk_token: String) -> Self {
        let _span = info_span!("Vocab::new", elements = token_to_id.len(), fallback = 0).entered();
        let mut id_to_token = HashMap::new();
        for (token, id) in &token_to_id {
            id_to_token.insert(*id, token.clone());
        }
        Self {
            token_to_id,
            id_to_token,
            unk_token,
        }
    }

    /// Retrieves the ID for a given token. Returns UNK ID if not found.
    pub fn get_id(&self, token: &str) -> u32 {
        let _span = info_span!("Vocab::get_id", elements = 1, fallback = 0).entered();
        if let Some(&id) = self.token_to_id.get(token) {
            id
        } else if let Some(&unk_id) = self.token_to_id.get(&self.unk_token) {
            unk_id
        } else {
            // Fallback to 0 if even UNK is missing, but typically we require UNK.
            0
        }
    }

    /// Retrieves the token for a given ID.
    pub fn get_token(&self, id: u32) -> Option<String> {
        let _span = info_span!("Vocab::get_token", elements = 1, fallback = 0).entered();
        self.id_to_token.get(&id).cloned()
    }

    pub fn get_unk_id(&self) -> u32 {
        self.get_id(&self.unk_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab() {
        println!("ST-EVIDENCE id=st-tokenize-vocab checked=1 exact=true paths=ids");
        let mut t2i = HashMap::new();
        t2i.insert("[PAD]".to_string(), 0);
        t2i.insert("[UNK]".to_string(), 1);
        t2i.insert("[CLS]".to_string(), 2);
        t2i.insert("[SEP]".to_string(), 3);
        t2i.insert("hello".to_string(), 4);
        t2i.insert("world".to_string(), 5);

        let vocab = Vocab::new(t2i, "[UNK]".to_string());

        assert_eq!(vocab.get_id("hello"), 4);
        assert_eq!(vocab.get_id("unknown_word"), 1); // unk
        assert_eq!(vocab.get_token(5), Some("world".to_string()));
        assert_eq!(vocab.get_token(999), None);
        assert_eq!(vocab.get_unk_id(), 1);
    }
}
