//! 包含基本的 Tokenizer 结构体和一些常用的辅助函数。
//! 基本结构体还包括了（通用的）保存/加载功能。
//! 可以对接口更加严格，例如将所有正则表达式部分隔离到 RegexTokenizer 中，但为了简化起见，做了一些妥协。


use std::io::Write;
use std::path::Path;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use indexmap::IndexMap;

/// 用于支持最多 2^31 个不同令牌的类型。它是有符号的，以防分词器需要使用负值表示特殊令牌。
pub type Token = i32;

/// 用于支持最多 2^64 次任何令牌对出现的计数类型。
pub type Count = u64;

/// 分词器需要实现的基本特质。
pub trait Tokenizer {
    fn special_tokens(&self) -> &IndexMap<String, Token>;

    fn merges(&self) -> &IndexMap<(Token, Token), Token>;

    fn vocab(&self) -> &IndexMap<Token, Vec<u8>>;

    /// 分词器可以将字符串编码为整数列表。
    fn encode(&self, text: &str) -> Vec<Token>;

    /// 分词器可以将整数列表解码为字符串。
    fn decode(&self, ids: &[Token]) -> String;
}

/// 可以进行训练的分词器。
pub trait Trainable: Tokenizer {
    /// 从 `text` 中训练一个大小为 `vocab_size` 的不同令牌词汇表。
    fn train(&mut self, text: &str, vocab_size: Token, verbose: bool);
}

pub trait Saveable: Tokenizer {
    fn pattern(&self) -> &str;

    /// 保存分词器的模型和词汇表到两个文件：
    /// - `file_prefix.model`：用于加载分词器的模型文件。
    /// - `file_prefix.vocab`：供检查的人类可读版本的词汇表。
    ///
    /// 这一做法受到（但不等同于）SentencePiece 模型保存的启发。
    ///
    /// # 参数
    ///
    /// * `dir` - 输出目录的路径。
    /// * `prefix` - 输出文件名的前缀。
    ///
    /// # 示例
    ///
    /// # use tempfile::tempdir;
    /// use minbpe::Saveable;
    /// use minbpe::Tokenizer;
    /// use minbpe::BasicTokenizer;
    /// let tokenizer = BasicTokenizer::new();
    /// let dir = tempdir().unwrap();
    /// let path = dir.path();
    /// tokenizer.save(&path, "prefix");
    ///
    fn save(&self, dir: &Path, prefix: &str) {
        // 写入模型文件（用于稍后加载分词器）
        let model_file_path = dir.join(format!("{}.model", prefix));
        let mut model_file = File::create(model_file_path).expect("无法创建.model文件");

       // 写入版本、模式和合并
        writeln!(model_file, "minbpe v1").expect("无法写入文件");
        writeln!(model_file, "{}", self.pattern()).expect("无法写入文件");

        // 写入特殊令牌（首先是数量，然后是每个令牌及其索引）
        writeln!(model_file, "{}", self.special_tokens().len())
            .expect("无法写入文件");
        for (special, idx) in self.special_tokens() {
            writeln!(model_file, "{} {}", special, idx).expect("无法写入文件");
        }

        let mut merges: Vec<(&(Token, Token), &Token)> = self.merges().iter().collect();
        merges.sort_by_key(|&k| k.1);

        // 写入合并字典
        for (token_pair, _new_token) in merges {
            writeln!(model_file, "{} {}", token_pair.0, token_pair.1)
                .expect("无法写入文件");
        }

        // 写入词汇表文件（提供给人工检查，可读性较好）
        let vocab_file_path = dir.join(format!("{}.vocab", prefix));
        let mut vocab_file = File::create(vocab_file_path).expect("无法创建文件");

        // 反转合并字典以便于查找
        let inverted_merges: IndexMap<Token, (Token, Token)> = self
            .merges()
            .iter()
            .map(|((idx1, idx2), idx)| (*idx, (*idx1, *idx2)))
            .collect();

        let vocab = self.vocab();

        for (idx, token) in vocab {
            // 渲染令牌，将无效的 UTF-8 序列替换为替换字符。
            let s = render_token(token);

            if let Some((idx0, idx1)) = inverted_merges.get(idx) {
                // 如果令牌有子项，则将其渲染为合并。
                let s0 = render_token(&vocab[idx0]);
                let s1 = render_token(&vocab[idx1]);
                writeln!(vocab_file, "[{}][{}] -> [{}] {}", s0, s1, s, idx)
                    .expect("无法写入文件");
            } else {
                // 否则，它是一个叶令牌（前 256 个字节之一）。
                writeln!(vocab_file, "[{}] {}", s, idx).expect("无法写入文件");
            }
        }
    }
}

pub trait Loadable: Tokenizer {
    fn set_pattern(&mut self, pattern: &str);
    fn set_special_tokens(&mut self, special_tokens: IndexMap<String, Token>);
    fn set_merges(&mut self, merges: IndexMap<(Token, Token), Token>);
    fn set_vocab(&mut self, vocab: IndexMap<Token, Vec<u8>>);

    /// 从文件加载分词器的模型。
    ///
    /// 这是 `save` 的逆操作，但仅针对模型文件。
    ///
    /// # 参数
    ///
    /// * `model_file` - 模型文件的路径。
    ///
    /// # 异常
    ///
    /// 如果模型文件没有 ".model" 扩展名或文件格式无效，则会引发异常。
    ///
    /// # 示例
    ///
    /// use std::path::PathBuf;
    /// use minbpe::Loadable;
    /// use minbpe::Tokenizer;
    /// use minbpe::BasicTokenizer;
    /// let mut tokenizer = BasicTokenizer::new();
    /// let model_path = PathBuf::from("examples/basic_example.model");
    /// tokenizer.load(&model_path);
    ///
    fn load(&mut self, model_file: &Path) {
        assert!(
            model_file.extension().map_or(false, |ext| ext == "model"),
            "模型文件必须有 .model 扩展名"
        );

        let mut merges: IndexMap<(Token, Token), Token> = IndexMap::new();
        let mut special_tokens: IndexMap<String, Token> = IndexMap::new();
        let mut idx: Token = 256;

        let file = File::open(model_file).expect("无法打开模型文件");
        let reader = BufReader::new(file);

        let lines: Vec<String> = reader
            .lines()
            .map(|line| line.expect("无法从模型文件中读取行"))
            .collect();

        let mut line_iter = lines.iter();

        if let Some(version) = line_iter.next() {
            assert_eq!(version, "minbpe v1", "无效的模型文件版本");
        } else {
            panic!("模型文件中缺少版本行");
        }

        // 检查 Tokenizer 是否支持 Pattern。
        if let Some(pattern) = line_iter.next() {
            self.set_pattern(pattern);
        } else {
            panic!("模型文件中缺少模式行");
        }

        if let Some(num_special_str) = line_iter.next() {
            let num_special = num_special_str
                .parse::<Token>()
                .expect("无效的特殊令牌数量");

            // FIXME: 检查 Tokenizer 是否支持特殊令牌。
            // FIXME: 确保其值 >= 0，因为 Token 类型是有符号的。
            // FIXME: 强制执行一个小于 2^31 的合理最大值。
            for _ in 0..num_special {
                if let Some(special_line) = line_iter.next() {
                    let mut parts = special_line.split_whitespace();
                    let special = parts.next().expect("缺少特殊令牌").to_string();
                    let special_idx = parts
                        .next()
                        .expect("缺少特殊令牌索引")
                        .parse::<Token>()
                        .expect("无效的特殊令牌索引");
                    special_tokens.insert(special, special_idx);
                } else {
                    panic!("模型文件中缺少特殊令牌行");
                }
            }
        } else {
            panic!("模型文件中缺少特殊令牌数量行");
        }

        for merge_line in line_iter {
            let mut parts = merge_line.split_whitespace();
            let idx1 = parts
                .next()
                .expect("缺少第一个索引")
                .parse::<Token>()
                .expect("无效的第一个索引");
            let idx2 = parts
                .next()
                .expect("缺少第二个索引")
                .parse::<Token>()
                .expect("无效的第二个索引");
            merges.insert((idx1, idx2), idx);
            idx += 1;
        }

        let vocab = build_vocab(&special_tokens, &merges);

        self.set_special_tokens(special_tokens);
        self.set_merges(merges);
        self.set_vocab(vocab);
    }
}

/// 分词器的附加操作。
/// 给定一个整数切片，返回一个新的 `IndexMap`，其中包含连续对的计数。
///
/// 示例：
///
/// # use indexmap::IndexMap;
/// # use minbpe::get_stats;
/// let ids = vec![1, 2, 3, 1, 2];
/// let counts = get_stats(&ids);
/// assert_eq!(counts, IndexMap::from([((1, 2), 2), ((2, 3), 1), ((3, 1), 1)]));
///
pub fn get_stats(ids: &[Token]) -> IndexMap<(Token, Token), Count> {
    let mut counts = IndexMap::new();
    update_stats(ids, &mut counts);
    counts
}

/// 使用给定整数切片中的连续对计数更新现有的 `IndexMap`。
///
/// 示例：
///
/// # use indexmap::IndexMap;
/// # use minbpe::update_stats;
/// let ids = vec![1, 2, 3, 1, 2];
/// let mut existing_counts = IndexMap::from([((1, 2), 1), ((2, 3), 1)]);
/// update_stats(&ids, &mut existing_counts);
/// assert_eq!(existing_counts, IndexMap::from([((1, 2), 3), ((2, 3), 2), ((3, 1), 1)]));
///
pub fn update_stats(ids: &[Token], counts: &mut IndexMap<(Token, Token), Count>) {
    for pair in ids.windows(2) {
        let pair = (pair[0], pair[1]);
        *counts.entry(pair).or_insert(0) += 1;
    }
}

/// 给定一个连续对计数的 `IndexMap`，返回计数最高的对。这种方法保留了 `IndexMap` 维护的对的插入顺序，返回计数最高的第一个插入的对。
pub fn get_max_entry(stats: &IndexMap<(Token, Token), Count>) -> Option<(&(Token, Token), &Count)> {
    let mut max_entry = None;

    for entry in stats.iter() {
        match max_entry {
            None => max_entry = Some(entry),
            Some((_, max_count)) => {
                let (_, count) = entry;
                if count > max_count {
                    max_entry = Some(entry);
                }
            }
        }
    }

    max_entry
}

/// 在给定的切片中合并连续出现的整数对，并用新的整数替换它们。
///
/// 参数：
/// - `ids`: 要合并的 Tokens 切片。
/// - `pair`: 要替换的连续整数对。
/// - `new_id`: 用于替换连续对的新整数。
///
/// 返回：
/// 一个新的 `Vec<Token>`，包含合并后的 Tokens。
///
/// 示例：
///
/// # use minbpe::merge;
/// let ids = vec![1, 2, 3, 1, 2];
/// let pair = (1, 2);
/// let new_id = 4;
/// let merged = merge(&ids, pair, new_id);
/// assert_eq!(merged, vec![4, 3, 4]);
/// ```
pub fn merge(ids: &[Token], pair: (Token, Token), new_id: Token) -> Vec<Token> {
    let mut new_ids = Vec::with_capacity(ids.len());
    let mut i = 0;

    while i < ids.len() {
        if i < ids.len() - 1 && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            new_ids.push(new_id);
            i += 2;
        } else {
            new_ids.push(ids[i]);
            i += 1;
        }
    }

    new_ids
}

/// 词汇表是从合并简单且确定地派生出来的。
pub fn build_vocab(
    special_tokens: &IndexMap<String, Token>,
    merges: &IndexMap<(Token, Token), Token>,
) -> IndexMap<Token, Vec<u8>> {
    let mut vocab: IndexMap<Token, Vec<u8>> = (0..256).map(|idx| (idx, vec![idx as u8])).collect();

    for ((p0, p1), idx) in merges {
        let mut token = vocab[p0].clone();
        token.extend_from_slice(&vocab[p1]);
        vocab.insert(*idx, token);
    }

    for (special, idx) in special_tokens {
        vocab.insert(*idx, special.as_bytes().to_vec());
    }

    vocab
}

/// 将给定字符串中的控制字符替换为它们的 Unicode 逃逸序列。
///
/// 控制字符是指会扭曲输出的字符，如换行符 (`\n`) 或其他属于 Unicode 类别 "C"（其他）的字符。
///
/// 参考资料：
/// - [https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117](https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117)
/// - [http://www.unicode.org/reports/tr44/#GC_Values_Table](http://www.unicode.org/reports/tr44/#GC_Values_Table)
///
/// 参数：
/// - `s`: 要处理的字符串。
///
/// 返回：
/// 一个新的 `String`，其中控制字符被替换为它们的 Unicode 逃逸序列。
///
/// 示例：
///
/// # use minbpe::tokenizer::replace_control_characters;
/// let s = "Hello\nWorld\u{7}!";
/// let result = replace_control_characters(s);
/// assert_eq!(result, "Hello\u000aWorld\u0007!");
///
fn replace_control_characters(s: &str) -> String {
    let mut chars = String::with_capacity(s.len());

    for ch in s.chars() {
        if ch.is_control() {
            let escaped = format!("\\u{:04x}", ch as u32);
            chars.push_str(&escaped);
        } else {
            chars.push(ch);
        }
    }

    chars
}

/// 以美观的方式打印令牌，通过将其解码为 UTF-8 并转义控制字符。
///
/// 参数：
/// - `token`: 作为字节切片的令牌。
///
/// 返回：
/// 一个 `String` 表示的令牌，其中控制字符已被转义。
///
/// 示例：
///
/// ```ignore
/// # use minbpe::tokenizer::render_token;
/// let token = b"Hello\nWorld\x07!";
/// let result = render_token(token);
/// assert_eq!(result, "Hello\\u000aWorld\\u0007!");
/// ```
fn render_token(token: &[u8]) -> String {
    let s = String::from_utf8_lossy(token);
    replace_control_characters(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace_control_characters() {
        let s = "Hello\nWorld\u{7}!";
        let result = replace_control_characters(s);
        assert_eq!(result, "Hello\\u000aWorld\\u0007!");
    }

    #[test]
    fn test_render_token() {
        let token = b"Hello\nWorld\x07!";
        let result = render_token(token);
        assert_eq!(result, "Hello\\u000aWorld\\u0007!");
    }

    #[test]
    fn test_indexmap_order() {
        let input_data: Vec<((Token, Token), Count)> = vec![
            ((0, 0), 2),
            ((1, 1), 12),
            ((2, 2), 18),
            ((3, 3), 11),
            ((4, 4), 1),
            ((5, 5), 9),
            ((6, 6), 99),
            ((7, 7), 7),
            ((8, 8), 20),
            ((9, 9), 99),
            ((10, 10), 99),
            ((11, 11), 99),
            ((12, 12), 4),
            ((13, 13), 99),
            ((14, 14), 19),
            ((15, 15), 99),
            ((16, 16), 5),
            ((17, 17), 99),
            ((18, 18), 99),
            ((19, 19), 7),
        ];

        let expected_max_key: (Token, Token) = (6, 6);

        let stats: IndexMap<(Token, Token), Count> = IndexMap::from_iter(input_data.clone());

        let keys: Vec<_> = stats.keys().collect();
        let input_keys: Vec<_> = input_data.iter().map(|(k, _)| k).collect();

        assert_eq!(keys, input_keys, "键不是按插入顺序排列的");

        let entries: Vec<_> = stats.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(
            entries,
            input_data.as_slice(),
            "条目不是按插入顺序排列的"
        );

        let max_entry = get_max_entry(&stats);

        let pair = max_entry.expect("统计信息为空");

        assert_eq!(*pair.0, expected_max_key);
    }
}
