# 如何构建自己的GPT-4分词器？

### 第一步 创建基础分词器
编写一个`BasicTokenizer`功能，将具备以下三个核心功能：
- `def train(self, text, vocab_size, verbose=False)`: 训练词表，返回词表大小
- `def encode(self, text)`: 将文本编码为ID序列
- `def decode(self, ids)`: 将ID序列解码为文本

使用你喜欢的任何文本训练你的分词器，并可视化合并后的tokens。它们看起来合理吗？你可以使用的一个默认测试文件是 `tests/taylorswift.txt`。

### 第二步 对训练文本进行预处理，通过正则表达式拆分文本，输入训练文本根据训练业务先进行质量评价及处理。（如语料库是JSON格式需要提取关键语料作为训练语料）

将你的 `BasicTokenizer` 转换为 `RegexTokenizer`，它接受一个正则表达式模式，并按照 GPT-4 的方式拆分文本。分别处理各个部分，然后将结果拼接起来。重新训练你的分词器，并比较转换前后的结果。你应该会看到现在没有跨类别的tokens（数字、字母、标点符号、多个空格）。使用 GPT-4 的模式：

```
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```
以上`GPT4_SPLIT_PATTERN`正则表达式用于匹配GPT-4的分词模式。其含义如下：
- `'(?i:[sdmt]|ll|ve|re)'`: 匹配单个单词，包括's'、'd'、'm'、't'、'll'、've'、're'等，并忽略大小写。
- `[^\r\n\p{L}\p{N}]?+\p{L}+`: 匹配单个单词，包括字母。
- `\p{N}{1,3}`: 匹配单个数字，包括1到3个数字。
- ` ?[^\s\p{L}\p{N}]++[\r\n]*`: 匹配单个单词，包括标点符号。
- `\s*[\r\n]`: 匹配单个换行符。
- `\s+(?!\S)`: 匹配单个空格，但不是行首。
- `\s+`: 匹配单个空格，但不是行尾。

### 第三步 对分词器的token序列进行合并，使用GPT-4的合并规则，并展示你的分词器在`encode`和`decode`方面产生相同的结果，与[tiktoken](https://github.com/openai/tiktoken)匹配。

你现在可以加载 GPT-4 分词器的合并结果，并展示你的分词器在 `encode` 和 `decode` 方面产生相同的结果，与 [tiktoken](https://github.com/openai/tiktoken) 匹配。

```
# 执行逻辑如下：
import tiktoken
enc = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
ids = enc.encode("hello world!!!? 你好,南京市长江大桥~！欢迎您参观南京市长江大桥。 😉") #获取token
text = enc.decode(ids)  #反向转换文本
```

不幸的是，你会遇到两个问题：

1. 从 GPT-4 分词器中恢复原始合并结果并不容易。你可以轻松地恢复我们称之为 `vocab` 的内容，以及他们存储在 `enc._mergeable_ranks` 中的内容。自由复制粘贴 `minbpe/gpt4.py` 中的 `recover_merges` 函数，该函数接受这些排名并返回原始合并结果。如果你想了解这个函数的工作原理，请阅读 [这篇](https://github.com/openai/tiktoken/issues/60) 和 [这篇](https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306) 文章。基本上，在某些条件下，只需要存储父节点（及其排名）并去掉精确的子节点合并细节就足够了。
2. 第二个问题是，GPT-4 分词器出于某种原因对其原始字节进行了置换。它将这种置换存储在可合并排名的前 256 个元素中，因此你可以相对简单地恢复这个字节置换，即 `byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}`。在编码和解码过程中，你需要相应地调整字节顺序。如果你遇到困难，可以参考 `minbpe/gpt4.py` 文件以获取提示。

### 第四步 增加处理特殊tokens的能力 如：<|endoftext|> 

（可选，令人烦恼，不一定有用）增加处理特殊tokens的能力。这样你就可以在存在特殊tokens的情况下匹配 `tiktoken` 的输出，例如：

```
import tiktoken
enc = tiktoken.get_encoding("cl100k_base") # 这是GPT-4分词器
ids = enc.encode("<|endoftext|>hello world", allowed_special="all")
```

如果没有 `allowed_special`，`tiktoken` 会出错。


### 第五步 拓展分词器使用其他语言编码如Unicode。

如果你已经做到了这一步，你现在已经是 LLM 分词的专家了！遗憾的是，你还没有完全完成任务，因为很多非 OpenAI 的 LLM（例如 Llama、Mistral）使用 [sentencepiece](https://github.com/google/sentencepiece)。主要区别在于 sentencepiece 直接在 Unicode 代码点上运行 BPE，而不是在 UTF-8 编码的字节上。自由探索 sentencepiece（祝你好运，它不太漂亮），如果你真的有时间和精力，可以将你的 BPE 重写为基于 Unicode 代码点，并匹配 Llama 2 分词器。
