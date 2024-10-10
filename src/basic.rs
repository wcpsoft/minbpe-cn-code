use indexmap::IndexMap;

use crate::base::{
    get_max_entry, get_stats, merge, Loadable, Saveable, Token, Tokenizer, Trainable,
};

/// 最小化的（字节级）字对编码分词器。
///
/// 算法上遵循 GPT 分词器：
/// https://github.com/openai/gpt-2/blob/master/src/encoder.py
///
/// 但是：
/// - 不处理正则表达式分割模式。
/// - 不处理任何特殊令牌。
///
/// # 示例
///
/// ```
/// use minbpe::BasicTokenizer;
/// use minbpe::Tokenizer;
/// use minbpe::Trainable;
///
/// let mut tokenizer = BasicTokenizer::new();
/// let text = "Hello, world!";
/// let vocab_size = 256;
/// let verbose = true;
///
/// tokenizer.train(text, vocab_size, verbose);
/// let encoded = tokenizer.encode(text);
/// let decoded = tokenizer.decode(&encoded);
///
/// assert_eq!(text, decoded);
/// ```
pub struct BasicTokenizer {
    special_tokens: IndexMap<String, Token>,
    merges: IndexMap<(Token, Token), Token>,
    vocab: IndexMap<Token, Vec<u8>>,
}

impl BasicTokenizer {
    pub fn new() -> Self {
        BasicTokenizer {
            special_tokens: IndexMap::new(),
            merges: IndexMap::new(),
            vocab: IndexMap::new(),
        }
    }
}

impl Default for BasicTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for BasicTokenizer {
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
       // 给定 ids（整数列表），返回 Rust 字符串
        let text_bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&idx| self.vocab[&idx].clone())
            .collect();
        String::from_utf8_lossy(&text_bytes).into_owned()
    }

    fn encode(&self, text: &str) -> Vec<Token> {
       // 给定一个字符串 text，返回令牌 id
        let text_bytes = text.as_bytes();
        let mut ids: Vec<Token> = text_bytes.iter().map(|&b| b as Token).collect();
        while ids.len() >= 2 {
           // 查找具有最低合并索引的对
            let stats = get_stats(&ids);

            let pair_opt = stats
                .keys()
                .filter_map(|&pair| self.merges.get(&pair).map(|_| pair))
                .min_by_key(|&pair| self.merges[&pair]);

            match pair_opt {
                // 如果没有更多的合并可用，退出循环
                None => break,
                Some(pair) => {
                    // 否则，合并最佳对（最低合并索引）
                    let idx = self.merges[&pair];
                    ids = merge(&ids, pair, idx);
                }
            };
        }
        ids
    }
}

impl Trainable for BasicTokenizer {
    fn train(&mut self, text: &str, vocab_size: Token, verbose: bool) {
        assert!(vocab_size >= 256, "词汇表大小必须至少为256");
        let num_merges = vocab_size - 256;

       // 输入文本预处理
        let text_bytes = text.as_bytes();
        let mut ids: Vec<Token> = text_bytes.iter().map(|&b| b as Token).collect();

        // 迭代地合并最常见的对以创建新令牌
        let mut merges: IndexMap<(Token, Token), Token> = IndexMap::new();
        let mut vocab: IndexMap<Token, Vec<u8>> =
            (0..256).map(|idx| (idx, vec![idx as u8])).collect();
        for i in 0..num_merges {
            // 统计每个连续对出现的次数
            let stats = get_stats(&ids);
            // 查找计数最高的对
            let pair = get_max_entry(&stats).unwrap().0;
            // 创建新令牌：为其分配下一个可用的 id
            let idx = 256 + i;
            // 将 ids 中所有 pair 的出现替换为 idx
            ids = merge(&ids, *pair, idx);
           // 保存合并
            merges.insert(*pair, idx);
            vocab.insert(
                idx,
                [vocab[&pair.0].clone(), vocab[&pair.1].clone()].concat(),
            );
            // 打印 进度
            if verbose {
                println!(
                    "合并 {}/{}: {:?} -> {} ({:?}) 出现了 {} 次",
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
        self.vocab = vocab;
    }
}

impl Saveable for BasicTokenizer {
    fn pattern(&self) -> &str {
        ""
    }
}

impl Loadable for BasicTokenizer {
    fn set_pattern(&mut self, pattern: &str) {
        let temp = pattern.trim();

        if !temp.is_empty() {
            panic!("不能设置非空模式")
        }
    }

    fn set_special_tokens(&mut self, special_tokens: IndexMap<String, Token>) {
        self.special_tokens = special_tokens;
    }

    fn set_merges(&mut self, merges: IndexMap<(Token, Token), Token>) {
        self.merges = merges;
    }

    fn set_vocab(&mut self, vocab: IndexMap<Token, Vec<u8>>) {
        self.vocab = vocab;
    }
}
