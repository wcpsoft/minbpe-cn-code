use minbpe::GPT4Tokenizer;
use minbpe::RegexTokenizerTrait;

fn main() {
    let text = "\u{1e01b}%SΣ";

    // 预初始化分词器
    let start = std::time::Instant::now();
    GPT4Tokenizer::initialize();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer static initialization completed in: {:?}",
        duration
    );

    // 获取分词器的默认实例
    let start = std::time::Instant::now();
    let tokenizer = GPT4Tokenizer::default();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer default instance construction completed in: {:?}",
        duration
    );

    // 编码字符串
    let start = std::time::Instant::now();
    let tokens = tokenizer.encode(text);
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer encoding of {} character string completed in: {:?}",
        text.len(),
        duration
    );

    // 打印生成的令牌
    println!("{:?}", tokens);
}
