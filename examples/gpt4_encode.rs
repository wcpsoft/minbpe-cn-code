use std::fs;
use std::path::PathBuf;

use minbpe::GPT4Tokenizer;
use minbpe::RegexTokenizerTrait;

fn main() -> std::io::Result<()> {
    let file_path = PathBuf::from("tests/taylorswift.txt");

    // 预初始化分词器
    println!("Pre-initializing the tokenizer...");
    let start = std::time::Instant::now();
    GPT4Tokenizer::initialize();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer static initialization completed in: {:?}",
        duration
    );

    // 获取分词器的默认实例
    println!("Getting a default instance of GPT4Tokenizer...");
    let start = std::time::Instant::now();
    let tokenizer = GPT4Tokenizer::default();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer default instance construction completed in: {:?}",
        duration
    );

    // 读取输入文件
    println!("Reading file: {:?}...", file_path);
    let start = std::time::Instant::now();
    let text = fs::read_to_string(file_path)?;
    let duration = start.elapsed();
    println!(
        "Reading {} characters completed in: {:?}",
        text.len(),
        duration
    );

    // 对编码过程进行计时，可选。
    let start = std::time::Instant::now();
    let tokens = tokenizer.encode(&text);
    let duration = start.elapsed();

    println!("Encoding completed in: {:?}", duration);
    println!("Produced {} encoded tokens", tokens.len());

    Ok(())
}
