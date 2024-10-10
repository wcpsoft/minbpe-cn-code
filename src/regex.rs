use fancy_regex::Regex;
use indexmap::IndexMap;
use std::collections::HashSet;

use crate::{get_max_entry, Loadable, Saveable, Trainable};
use crate::{get_stats, merge, update_stats, Token, Tokenizer};

/// 主要的 GPT 文本分割模式，详见
/// https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

pub const GPT2_SPLIT_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// 指定在编码过程中如何处理特殊令牌。
///
/// 此枚举用于控制 `encode_special` 函数在文本中遇到特殊令牌时的行为。
///
/// # 变体
///
/// - `All`: 允许所有特殊令牌进行编码。
///   特殊令牌将根据其对应的令牌 ID 进行编码。
///
/// - `None`: 在编码过程中忽略所有特殊令牌。
///   特殊令牌将被视为普通文本，并使用标准编码过程进行编码。
///
/// - `NoneRaise`: 如果在编码过程中遇到任何特殊令牌，则引发错误。
///   这是 `tiktoken` 库的默认行为。
///
/// - `Set(HashSet<String>)`: 仅允许提供的 `HashSet` 中指定的特殊令牌。
///   未包含在集合中的特殊令牌将被视为普通文本，并使用标准编码过程进行编码。
///
/// # 示例
///
/// ```
/// use minbpe::AllowedSpecial;
/// use std::collections::HashSet;
///
/// // Allow all special tokens
/// let allowed_all = AllowedSpecial::All;
///
/// // Ignore all special tokens
/// let allowed_none = AllowedSpecial::None;
///
/// // Raise an error if any special token is encountered
/// let allowed_none_raise = AllowedSpecial::NoneRaise;
///
/// // Allow only specific special tokens
/// let custom_set = HashSet::from(["<|endoftext|>".to_string(), "<|startoftext|>".to_string()]);
/// let allowed_custom = AllowedSpecial::Set(custom_set);
/// ```
pub enum AllowedSpecial {
    All,
    None,
    NoneRaise,
    Set(HashSet<String>),
}

pub trait RegexTokenizerTrait: Tokenizer {
    fn encode_chunk_inner(&self, text_bytes: &[u8]) -> Vec<Token> {
        let merges = self.merges();
        let mut ids: Vec<Token> = text_bytes.iter().map(|&b| b as Token).collect();
        while ids.len() >= 2 {
            // 找到合并索引最小一对元素
            let stats = get_stats(&ids);

            let pair_opt = stats
                .keys()
                .filter_map(|&pair| merges.get(&pair).map(|_| pair))
                .min_by_key(|&pair| merges[&pair]);

            match pair_opt {
                None => break, // 如果没有更多的合并可用，则中断循环
                Some(pair) => {
                    // 否则，合并最佳对（最低合并索引
                    let idx = merges[&pair];
                    ids = merge(&ids, pair, idx);
                }
            };
        }
        ids
    }

    fn encode_chunk(&self, text_bytes: &[u8]) -> Vec<Token> {
        self.encode_chunk_inner(text_bytes)
    }

    // fn pattern(&self) -> &str;
    // fn set_pattern(&mut self, pattern: &str);

    fn compiled_pattern(&self) -> &Regex;

    // fn special_tokens(&self) -> &IndexMap<String, Token>;
    // fn set_special_tokens(&mut self, special_tokens: IndexMap<String, Token>);

    fn inverse_special_tokens(&self) -> &IndexMap<Token, String>;

    // fn merges(&self) -> &IndexMap<(Token, Token), Token>;
    // fn set_merges(&mut self, merges: IndexMap<(Token, Token), Token>);

    // fn vocab(&self) -> &IndexMap<Token, Vec<u8>>;
    // fn set_vocab(&mut self, vocab: IndexMap<Token, Vec<u8>>);

    // fn train(&mut self, text: &str, vocab_size: Token, verbose: bool);
    // fn decode(&self, ids: &[Token]) -> String;
    // fn encode(&self, text: &str) -> Vec<Token>;

    fn decode(&self, ids: &[Token]) -> String {
        let mut part_bytes = Vec::new();
        for &idx in ids {
            if let Some(bytes) = self.vocab().get(&idx) {
                part_bytes.extend_from_slice(bytes);
            } else if let Some(special_token) = self.inverse_special_tokens().get(&idx) {
                part_bytes.extend_from_slice(special_token.as_bytes());
            } else {
                panic!("Invalid token id: {}", idx);
            }
        }
        String::from_utf8_lossy(&part_bytes).into_owned()
    }

    fn encode(&self, text: &str) -> Vec<Token> {
        self.encode_special(text, AllowedSpecial::NoneRaise)
    }

    /// 编码时忽略所有特殊令牌。
    fn encode_ordinary(&self, text: &str) -> Vec<Token> {
        let text_chunks: Vec<&str> = self
            .compiled_pattern()
            .find_iter(text)
            .map(|m| {
                let matched = m.unwrap();
                &text[matched.start()..matched.end()]
            })
            .collect();
        let mut ids = Vec::new();
        for chunk in text_chunks {
            let chunk_bytes = chunk.as_bytes();
            let chunk_ids = self.encode_chunk(chunk_bytes);
            ids.extend(chunk_ids);
        }
        ids
    }

    /// 将给定文本编码为令牌 ID，并处理特殊令牌。
    ///
    /// 与 `encode_ordinary` 不同，此函数根据 `allowed_special` 参数处理特殊令牌。
    ///
    /// # 参数
    ///
    /// * `text` - 要编码的文本。
    /// * `allowed_special` - 指定如何处理特殊令牌。它可以是以下之一：
    ///   - `AllowedSpecial::All`: 允许所有特殊令牌。
    ///   - `AllowedSpecial::None`: 忽略所有特殊令牌。
    ///   - `AllowedSpecial::NoneRaise`: 如果在文本中遇到任何特殊令牌，则引发错误。
    ///     这是 `tiktoken` 库的默认行为。
    ///   - `AllowedSpecial::Set(HashSet<String>)`: 自定义允许的特殊令牌集。
    ///
    /// # 异常
    ///
    /// 如果 `allowed_special` 设置为 `AllowedSpecial::NoneRaise` 并且在文本中遇到任何特殊令牌，则引发异常。
    fn encode_special(&self, text: &str, allowed_special: AllowedSpecial) -> Vec<Token> {
        let special = match allowed_special {
            AllowedSpecial::All => self.special_tokens().clone(),
            AllowedSpecial::None => IndexMap::new(),
            AllowedSpecial::NoneRaise => {
                assert!(
                    self.special_tokens()
                        .keys()
                        .all(|token| !text.contains(token)),
                    "Special token found in text"
                );
                IndexMap::new()
            }
            AllowedSpecial::Set(special_tokens) => {
                let mut special = IndexMap::new();
                for token in special_tokens {
                    if let Some(&idx) = self.special_tokens().get(&token) {
                        special.insert(token, idx);
                    }
                }
                special
            }
        };

        if special.is_empty() {
            return self.encode_ordinary(text);
        }

        let special_pattern = "(".to_string()
            + &special
                .keys()
                .map(|k| regex::escape(k))
                .collect::<Vec<String>>()
                .join("|")
            + ")";

        let re = fancy_regex::Regex::new(&special_pattern).unwrap();
        let mut last_end = 0;
        let mut special_chunks = Vec::new();
        for m in re.find_iter(text) {
            let m = m.unwrap();
            // 将匹配之间的文本存储在 special_chunks
            special_chunks.push(&text[last_end..m.start()]);
            // 将匹配的文本存储在 special_chunks
            special_chunks.push(&text[m.start()..m.end()]);
            last_end = m.end();
        }
        let remaining = &text[last_end..];
        if !remaining.is_empty() {
            special_chunks.push(remaining);
        }

        let mut ids = Vec::new();
        for part in special_chunks {
            if let Some(&idx) = special.get(part) {
                ids.push(idx);
            } else {
                ids.extend(self.encode_ordinary(part));
            }
        }
        ids
    }
}

/// 最小化的（字节级）字对编码分词器。
///
/// 算法上遵循 GPT 分词器：
/// https://github.com/openai/gpt-2/blob/master/src/encoder.py
///
/// 与 `BasicTokenizer` 不同：
/// - `RegexTokenizer` 处理可选的正则表达式分割模式。
/// - `RegexTokenizer` 处理可选的特殊令牌。
///
/// # 示例
///
/// ```
/// use fancy_regex::Regex;
/// use minbpe::base::Loadable;
/// use minbpe::base::Tokenizer;
/// use minbpe::base::Trainable;
/// use minbpe::RegexTokenizerStruct;
/// use minbpe::RegexTokenizerTrait;
/// use minbpe::AllowedSpecial;
/// use indexmap::IndexMap;
///
/// let pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
/// let mut tokenizer = RegexTokenizerStruct::new(pattern.to_string());
/// let special_tokens = IndexMap::from([("<|endoftext|>".to_string(), 100257)]);
/// tokenizer.set_special_tokens(special_tokens);
///
/// let text = "Hello, world! This is a test.";
/// let vocab_size = 256 + 10;
/// let verbose = true;
///
/// tokenizer.train(text, vocab_size, verbose);
///
/// let encoded = tokenizer.encode_special(text, AllowedSpecial::NoneRaise);
/// let decoded = RegexTokenizerTrait::decode(&tokenizer, &encoded);
///
/// assert_eq!(text, decoded);
/// ```
pub struct RegexTokenizerStruct {
    pattern: String,
    compiled_pattern: Regex,
    special_tokens: IndexMap<String, Token>,
    inverse_special_tokens: IndexMap<Token, String>,
    merges: IndexMap<(Token, Token), Token>,
    vocab: IndexMap<Token, Vec<u8>>,
}

impl Default for RegexTokenizerStruct {
    fn default() -> Self {
        Self::new(GPT4_SPLIT_PATTERN.to_string())
    }
}

impl RegexTokenizerStruct {
    fn make(pattern: String) -> Self {
        let compiled_pattern = Regex::new(&pattern).unwrap();

        RegexTokenizerStruct {
            pattern,
            compiled_pattern,
            special_tokens: IndexMap::new(),
            inverse_special_tokens: IndexMap::new(),
            merges: IndexMap::new(),
            vocab: IndexMap::new(),
        }
    }

    pub fn new(pattern: String) -> Self {
        Self::make(pattern)
    }
}

impl Tokenizer for RegexTokenizerStruct {
    fn special_tokens(&self) -> &IndexMap<String, Token> {
        &self.special_tokens
    }

    fn merges(&self) -> &IndexMap<(Token, Token), Token> {
        &self.merges
    }

    fn vocab(&self) -> &IndexMap<Token, Vec<u8>> {
        &self.vocab
    }

    fn decode(&self, ids: &[Token]) -> String {
        // 转发到由 RegexTokenizerTrait 提供的默认实现
        <Self as RegexTokenizerTrait>::decode(self, ids)
    }

    fn encode(&self, text: &str) -> Vec<Token> {
        // 转发到由 RegexTokenizerTrait 提供的默认实现
        <Self as RegexTokenizerTrait>::encode(self, text)
    }
}

impl Trainable for RegexTokenizerStruct {
    fn train(&mut self, text: &str, vocab_size: Token, verbose: bool) {
        assert!(vocab_size >= 256, "词汇表大小必须至少为256");
        let num_merges = vocab_size - 256;

        // 将文本分割成块
        let text_chunks: Vec<&str> = self
            .compiled_pattern()
            .find_iter(text)
            .map(|m| {
                let matched = m.unwrap();
                &text[matched.start()..matched.end()]
            })
            .collect();

        // 输入文本预处理
        let mut ids: Vec<Vec<Token>> = text_chunks
            .iter()
            .map(|chunk| chunk.as_bytes().iter().map(|b| *b as Token).collect())
            .collect();

        // 迭代地合并最常见的对以创建新令牌
        let mut merges: IndexMap<(Token, Token), Token> = IndexMap::new();
        let mut vocab: IndexMap<Token, Vec<u8>> =
            (0..256).map(|idx| (idx, vec![idx as u8])).collect();

        for i in 0..num_merges {
            // 统计每个连续对出现的次数
            let mut stats = IndexMap::new();
            for chunk_ids in &ids {
                update_stats(chunk_ids, &mut stats);
            }

            // 查找计数最高的对
            let pair = get_max_entry(&stats).unwrap().0;

           // 创建新令牌：为其分配下一个可用的 id
            let idx = 256 + i;

            // 将 ids 中所有 pair 的出现替换为 idx
            ids = ids
                .iter()
                .map(|chunk_ids| merge(chunk_ids, *pair, idx))
                .collect();

            // 保存合并
            merges.insert(*pair, idx);
            vocab.insert(
                idx,
                [vocab[&pair.0].clone(), vocab[&pair.1].clone()].concat(),
            );

            // 打印进度
            if verbose {
                println!(
                    "merge {}/{}: {:?} -> {} ({:?}) had {} occurrences",
                    i + 1,
                    num_merges,
                    pair,
                    idx,
                    vocab[&idx],
                    stats[pair]
                );
            }
        }

       // 保存实例变量
        self.merges = merges;
        // FIXME: 应该是 build_vocab(&self.special_tokens, &self.merges);
        self.vocab = vocab;
    }
}

impl Saveable for RegexTokenizerStruct {
    fn pattern(&self) -> &str {
        &self.pattern
    }
}

impl Loadable for RegexTokenizerStruct {
    fn set_pattern(&mut self, pattern: &str) {
        self.pattern = pattern.to_string();
        self.compiled_pattern = Regex::new(pattern).unwrap();
    }

    fn set_special_tokens(&mut self, special_tokens: IndexMap<String, Token>) {
        self.special_tokens = special_tokens.clone();
        self.inverse_special_tokens = special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
    }

    fn set_merges(&mut self, merges: IndexMap<(Token, Token), Token>) {
        self.merges = merges;
    }

    fn set_vocab(&mut self, vocab: IndexMap<Token, Vec<u8>>) {
        self.vocab = vocab;
    }
}

impl RegexTokenizerTrait for RegexTokenizerStruct {
    fn compiled_pattern(&self) -> &Regex {
        &self.compiled_pattern
    }

    fn inverse_special_tokens(&self) -> &IndexMap<Token, String> {
        &self.inverse_special_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use std::collections::HashSet;

    #[test]
    fn test_pattern_matching() {
        let text = "Hello, world! <|endoftext|>";

        let pattern = "(<\\|endoftext\\|>)";
        let re = fancy_regex::Regex::new(pattern).unwrap();

        let mut last_end = 0;
        let mut special_chunks = Vec::new();
        for m in re.find_iter(text) {
            let m = m.unwrap();
            // 将匹配之间的文本推入
            special_chunks.push(&text[last_end..m.start()]);
            // 将匹配的文本推入
            special_chunks.push(&text[m.start()..m.end()]);
            last_end = m.end();
        }
        let remaining = &text[last_end..];
        if !remaining.is_empty() {
            special_chunks.push(remaining);
        }
    }

    #[test]
    fn test_encode_special() {
        let mut tokenizer = RegexTokenizerStruct::default();
        tokenizer.train("Hello, world! Goodbye, world!, So long...", 256 + 10, true);

        let text = "Hello, world! <|endoftext|>";

        let special_tokens = IndexMap::from([("<|endoftext|>".to_string(), 100257)]);
        tokenizer.set_special_tokens(special_tokens);

        let encoded_all = tokenizer.encode_special(text, AllowedSpecial::All);
        let encoded_none = tokenizer.encode_special(text, AllowedSpecial::None);

        let custom_set = HashSet::from(["<|endoftext|>".to_string()]);
        let encoded_custom = tokenizer.encode_special(text, AllowedSpecial::Set(custom_set));

        assert!(encoded_all.contains(&100257));
        assert!(!encoded_none.contains(&100257));
        assert!(encoded_custom.contains(&100257));
    }

    #[test]
    #[should_panic]
    fn test_encode_special_panic() {
        let mut tokenizer = RegexTokenizerStruct::default();
        let text = "Hello, world! <|endofext|>";

        let special_tokens = IndexMap::from([("<|endofext|>".to_string(), 100257)]);
        tokenizer.set_special_tokens(special_tokens);

        // This should panic
        let _ = tokenizer.encode_special(text, AllowedSpecial::NoneRaise);
    }
}
