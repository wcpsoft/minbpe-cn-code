#[cfg(test)]
mod tests {
    use minbpe::test_common::{LLAMA_TEXT, SPECIAL_TOKENS};
    use minbpe::AllowedSpecial;
    use minbpe::BasicTokenizer;
    use minbpe::Loadable;
    use minbpe::RegexTokenizerStruct;
    use minbpe::RegexTokenizerTrait;
    use minbpe::Saveable;
    use minbpe::Token;
    use minbpe::Trainable;

    use indexmap::IndexMap;
    use tempfile::tempdir;
    // 快速单元测试，遵循维基百科的例子：
    // https://en.wikipedia.org/wiki/Byte_pair_encoding
    //
    // 根据维基百科，对输入字符串 "aaabdaaabac" 运行 BPE：
    //
    // 经过 3 次合并后，结果字符串为 "XdXac"：
    //
    // 其中：
    // X = ZY
    // Y = ab
    // Z = aa
    //
    // 请注意，对于我们来说，a=97, b=98, c=99, d=100（ASCII 值）
    // 因此 Z 将是 256，Y 将是 257，X 将是 258。
    //
    // 所以我们期望输出的 id 列表为 [258, 100, 258, 97, 99]

    fn test_wikipedia_example_inner(tokenizer: &mut Box<dyn Trainable>) {
        let text = "aaabdaaabac";
        tokenizer.train(text, 256 + 3, false);
        let ids = tokenizer.encode(text);
        assert_eq!(ids, [258, 100, 258, 97, 99]);
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_wikipedia_example() {
        let tokenizers: Vec<Box<dyn Trainable>> = vec![
            Box::new(BasicTokenizer::new()),
            Box::<RegexTokenizerStruct>::default(),
        ];

        for mut tokenizer in tokenizers {
            test_wikipedia_example_inner(&mut tokenizer);
        }
    }

    fn test_save_load_inner(special_tokens: &IndexMap<String, Token>) {
        // 选取一段稍微复杂一些的文本并训练分词器
        let text = LLAMA_TEXT;
        // 创建一个 Tokenizer 并进行 64 次合并
        let mut tokenizer = RegexTokenizerStruct::default();
        tokenizer.train(text, 256 + 64, false);
        // 在训练之后做这件事感觉有些奇怪，这不是设置的一部分
        tokenizer.set_special_tokens(special_tokens.clone());

        // 验证 decode(encode(x)) == x
        let encoded = tokenizer.encode_special(text, AllowedSpecial::All);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);

        // 验证 save/load 是否按预期工作；保存分词器
        let dir = tempdir().unwrap();
        tokenizer.save(dir.path(), "test_tokenizer_tmp");

        // 重新加载分词器
        let mut tokenizer = RegexTokenizerStruct::default();
        let model_file = dir.path().join("test_tokenizer_tmp.model");
        tokenizer.load(&model_file);

        // 验证 decode(encode(x)) == x
        assert_eq!(tokenizer.decode(&encoded), text);
        assert_eq!(
            tokenizer.decode(&tokenizer.encode_special(text, AllowedSpecial::All)),
            text
        );
        assert_eq!(tokenizer.encode_special(text, AllowedSpecial::All), encoded);
    }

    #[test]
    fn test_save_load() {
        let special_tokens = IndexMap::new();
        test_save_load_inner(&special_tokens);
        let special_tokens = &SPECIAL_TOKENS;
        test_save_load_inner(special_tokens);
    }
}
