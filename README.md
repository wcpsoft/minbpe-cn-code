# `minbpe-rs` 

> minbpe-rs是对Andrej Karpathy编写的minbpe的rust版实现。

[![minbpe-rs crate](https://img.shields.io/crates/v/minbpe.svg)](https://crates.io/crates/minbpe)
[![minbpe-rs documentation](https://docs.rs/minbpe/badge.svg)](https://docs.rs/minbpe)


## 快速入门

使用`cargo new`创建一个新项目minbpe-test。

```shell
cargo new minbpe-test
```

在创建的项目的`Cargo.toml`中添加依赖项minbpe，注意查看官方库最新版本[`crates.io`](https://crates.io/crates/minbpe)

```toml
[dependencies]
minbpe = "0.1"
```

在`src/main.rs`中添加以下代码，以实现一个简单基础的tokenizer实现算法，并保存在`models`文件夹中模型名称为`demo`。

```rust
use std::path::Path;
use minbpe::{BasicTokenizer, Saveable, Tokenizer, Trainable};

fn main() {
    let text = "aaabdaaabac" ;
    let mut tokenizer = BasicTokenizer::new() ;
    tokenizer.train( text , 256 + 3 , false ) ;
    println!( "{:?}" , tokenizer.encode(text) ) ;
    println!( "{:?}" , tokenizer.decode( &[258, 100, 258, 97, 99] ) ) ;
    tokenizer.save( Path::new( "models" ) , "demo" ) ;
}
```

执行 `cargo run`查看运行结果

```shell
cargo run
```

运行结果如下：
```shell
   ...
   Compiling minbpe-test v0.1.0 (~/minbpe-test)
    Finished dev [unoptimized + debuginfo] target(s) in 15.71s
     Running `target/debug/minbpe-test`
[258, 100, 258, 97, 99]
"aaabdaaabac"
```


## 开源协议

minbpe-rs是[MIT](LICENSE-MIT)许可协议或[Apache-2.0](LICENSE-APACHE)许可协议许可，根据需要自己选择。


## 贡献

除非你明确声明相反，任何你有意提交并包含在本项目中的贡献，根据 Apache-2.0 许可证的定义，都将按照上述条款进行双重许可，不附加任何额外的条款或条件。
