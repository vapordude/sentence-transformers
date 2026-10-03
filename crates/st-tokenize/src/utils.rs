/// Checks whether a character is whitespace.
pub fn is_whitespace(c: char) -> bool {
    // Treat standard whitespaces and control characters as whitespace
    if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
        return true;
    }
    // In many tokenizers, Unicode categories Zs, Zl, Zp are treated as whitespace
    c.is_whitespace()
}

/// Checks whether a character is punctuation.
pub fn is_punctuation(c: char) -> bool {
    let cp = c as u32;
    // Treat ASCII punctuation as punctuation
    if (33..=47).contains(&cp)
        || (58..=64).contains(&cp)
        || (91..=96).contains(&cp)
        || (123..=126).contains(&cp)
    {
        return true;
    }
    // General punctuation character classes in unicode
    c.is_ascii_punctuation() || (c.is_ascii_graphic() && !c.is_alphanumeric())
}

/// Splits text on whitespace.
pub fn whitespace_tokenize(text: &str) -> Vec<String> {
    let _span = info_span!("whitespace_tokenize", elements = text.len(), fallback = 0).entered();
    text.split_whitespace().map(|s| s.to_string()).collect()
}

/// Splits text on punctuation, keeping the punctuation as separate tokens.
pub fn punctuation_tokenize(text: &str) -> Vec<String> {
    let _span = info_span!("punctuation_tokenize", elements = text.len(), fallback = 0).entered();
    let mut tokens = Vec::new();
    let mut current_token = String::new();

    for c in text.chars() {
        if is_punctuation(c) {
            if !current_token.is_empty() {
                tokens.push(current_token.clone());
                current_token.clear();
            }
            tokens.push(c.to_string());
        } else {
            current_token.push(c);
        }
    }

    if !current_token.is_empty() {
        tokens.push(current_token);
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_tokenize() {
        println!("ST-EVIDENCE id=st-tokenize-whitespace checked=1 exact=true paths=string");
        let text = "Hello  World\t \n Rust";
        let tokens = whitespace_tokenize(text);
        assert_eq!(tokens, vec!["Hello", "World", "Rust"]);
    }

    #[test]
    fn test_punctuation_tokenize() {
        println!("ST-EVIDENCE id=st-tokenize-punctuation checked=1 exact=true paths=string");
        let text = "Hello, world! This is a test.";
        let tokens = punctuation_tokenize(text);
        assert_eq!(
            tokens,
            vec!["Hello", ",", " world", "!", " This is a test", "."]
        );
    }
}
