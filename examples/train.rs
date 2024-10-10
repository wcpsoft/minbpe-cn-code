use std::fs;
use std::path::Path;
use std::time::Instant;

use minbpe::BasicTokenizer;
use minbpe::RegexTokenizerStruct;
use minbpe::Saveable;
use minbpe::Tokenizer;
use minbpe::Trainable;

fn main() {
    let text = fs::read_to_string("tests/taylorswift.txt").expect("没有读到文件");

    fs::create_dir_all("models").expect("无法创建models目录");

    let basic = BasicTokenizer::new();
    let regex = RegexTokenizerStruct::default();

    fn doit<T: Tokenizer + Trainable + Saveable>(tokenizer: T, name: &str, text: &str) {
        let mut tokenizer = tokenizer;
        tokenizer.train(text, 512, true);

        let dir = Path::new("models").to_path_buf();
        tokenizer.save(&dir, name);
    }

    let t0 = Instant::now();
    doit(basic, "basic", &text);
    doit(regex, "regex", &text);
    let t1 = Instant::now();

    let duration = t1.duration_since(t0);
    println!("训练耗时 {:.2} 秒", duration.as_secs_f64());
}
