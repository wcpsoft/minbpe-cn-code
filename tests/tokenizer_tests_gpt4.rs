#[cfg(all(test, feature = "gpt4"))]
mod tests {
    use std::collections::HashSet;
    use tiktoken_rs::cl100k_base;

    use minbpe::GPT4Tokenizer;
    use minbpe::RegexTokenizerTrait;
    use minbpe::Token;

    use minbpe::test_common::{unpack, TEST_STRINGS};

   // 测试我们的分词器是否与官方的 GPT-4 分词器匹配
    fn test_gpt4_tiktoken_equality_inner(text: String) {
        let special_tokens: HashSet<&str> = HashSet::new();

        let text = unpack(&text).unwrap();
        println!(
            "test_gpt4_tiktoken_equality_inner: text length is: {:?}",
            text.len()
        );
        use std::time::Instant;

        let enc = cl100k_base().unwrap();

        let tiktoken_start = Instant::now();
        let tiktoken_ids = enc.encode(&text, special_tokens);
        let tiktoken_tokens: Vec<Token> = tiktoken_ids.iter().map(|&id| id as Token).collect();
        let tiktoken_duration = tiktoken_start.elapsed();
        println!("TikToken encoding took: {:?}", tiktoken_duration);

        let tokenizer = GPT4Tokenizer::new();

        let gpt4_start = Instant::now();
        let gpt4_tokenizer_tokens = tokenizer.encode(&text);
        let gpt4_duration = gpt4_start.elapsed();
        println!("GPT4 encoding took: {:?}", gpt4_duration);

        assert_eq!(
            tiktoken_tokens.len(),
            gpt4_tokenizer_tokens.len(),
            "Token vectors are of different lengths: {} expected, but found {}",
            tiktoken_tokens.len(),
            gpt4_tokenizer_tokens.len()
        );
        assert_eq!(
            tiktoken_tokens, gpt4_tokenizer_tokens,
            "Token vectors do not match"
        );
    }

    #[test]
    fn test_gpt4_tiktoken_equality() {
        // 初始化 GPT4Tokenizer 静态数据
        GPT4Tokenizer::initialize();

        for text in TEST_STRINGS.iter() {
            println!("test_gpt4_tiktoken_equality: testing with text: {:?}", text);
            let text = unpack(text).unwrap();
            test_gpt4_tiktoken_equality_inner(text);
        }
    }
}
