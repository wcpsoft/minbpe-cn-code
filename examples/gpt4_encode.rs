use std::fs;
use std::path::PathBuf;

use minbpe::GPT4Tokenizer;
use minbpe::RegexTokenizerTrait;

fn main() -> std::io::Result<()> {
    let file_path = PathBuf::from("tests/taylorswift.txt");

    // 预初始化分词器
    println!("预初始化分词器...");
    let start = std::time::Instant::now();
    GPT4Tokenizer::initialize();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer 静态初始化完成于: {:?}",
        duration
    );

    // 获取分词器的默认实例
    println!("获取 GPT4Tokenizer 的默认实例...");
    let start = std::time::Instant::now();
    let tokenizer = GPT4Tokenizer::default();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer 默认实例构造完成于: {:?}",
        duration
    );

    // 读取输入文件
    println!("读取文件: {:?}...", file_path);
    let start = std::time::Instant::now();
    let text = fs::read_to_string(file_path)?;
    let duration = start.elapsed();
    println!(
        "读取 {} 字符处理完成于: {:?}",
        text.len(),
        duration
    );

    // 对编码过程进行计时，可选。
    let start = std::time::Instant::now();
    let tokens = tokenizer.encode(&text);
    let duration = start.elapsed();

    println!("编码于: {:?}", duration);
    println!("生成了 {} 个编码后的令牌长度：", tokens.len());

    Ok(())
}
