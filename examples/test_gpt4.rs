use minbpe::GPT4Tokenizer;
use minbpe::RegexTokenizerTrait;

fn main() {
    let text = "\u{1e01b}%SΣ";

    // 预初始化分词器
    let start = std::time::Instant::now();
    GPT4Tokenizer::initialize();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer 静态初始化完成于：{:?}",
        duration
    );

    // 获取分词器的默认实例
    let start = std::time::Instant::now();
    let tokenizer = GPT4Tokenizer::default();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer 默认实例构造完成于: {:?}",
        duration
    );

    // 编码字符串
    let start = std::time::Instant::now();
    let tokens = tokenizer.encode(text);
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer 对 {} 个字符的字符串编码完成于： {:?}",
        text.len(),
        duration
    );

    // 打印生成的令牌
    println!("{:?}", tokens);
}
