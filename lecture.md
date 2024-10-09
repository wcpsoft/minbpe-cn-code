# LLM标记语言模型原理演讲稿
大家好，今天我们将讨论大型语言模型(LLM)中的标记化。遗憾的是，标记化是一个相对复杂和粗糙的组成部分的状态的艺术 LLM，但是有必要了解一些细节，因为 LLM 的许多缺点，可能归因于神经网络或其他神秘的实际上追溯到标记化。

### 首先介绍一下：字符级别的分词


那么什么是分词呢？实际上，在我们之前的视频 [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) 中，我们已经介绍了分词，但那只是一个非常简单、朴素的字符级别版本。当你查看该视频的 [Google Colab](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) 时，你会发现我们从训练数据 ([Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)) 开始，这在 Python 中只是一个大的字符串：

```
First Citizen: Before we proceed any further, hear me speak.

All: Speak, speak.

First Citizen: You are all resolved rather to die than to famish?

All: Resolved. resolved.

First Citizen: First, you know Caius Marcius is chief enemy to the people.

All: We know't, we know't.
```

但是我们如何将字符串输入到语言模型中呢？我们看到，我们首先通过构建整个训练集中所有可能字符的词汇表来实现这一点：

```python
# 这里是文本中出现的所有唯一字符

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# 65
```

然后创建一个查照表，根据上述词汇表在单个字符和整数之间进行转换。这个查照表只是一个 Python 字典：

```python
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
# encoder: take a string, output a list of integers
encode = lambda s: [stoi[c] for c in s]
# decoder: take a list of integers, output a string
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))

# [46, 47, 47, 1, 58, 46, 43, 56, 43]
# hii there
```

一旦我们将字符串转换为整数序列，我们看到每个整数被用作索引，以访问一个二维的可训练参数嵌入表。因为我们有一个词汇表大小 `vocab_size=65`，
```python
class BigramLanguageModel(nn.Module):

def __init__(self, vocab_size):
	super().__init__()
	self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

def forward(self, idx, targets=None):
	tok_emb = self.token_embedding_table(idx) # (B,T,C)
```

在这里，整数“提取”嵌入表中的一行，这一行就是表示该令牌的向量。然后这个向量作为相应时间步的输入传入 Transformer。

### 使用 BPE 算法进行分词的“字符块”

对于字符级别的语言模型来说，这在朴素设置下是很好的。但在实际的最先进语言模型中，人们使用更复杂的方案来构建这些令牌词汇表。特别是，这些方案不是在字符级别上工作，而是在字符块级别上工作。构建这些块词汇表的方法是使用诸如 **字节对编码**（BPE）算法等算法，我们将在下面详细讨论。

暂时转向这种方法的历史发展，2019 年 OpenAI 发表的 [GPT-2 论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)《语言模型是无监督多任务学习者》普及了使用字节级 BPE 算法进行语言模型分词的方法。请参阅第 2.2 节“输入表示”，其中描述并解释了该算法。在这节的末尾，你会看到他们提到：

> *词汇表扩展到了 50,257 个。我们还将上下文大小从 512 增加到 1024 个令牌，并使用更大的批量大小 512。*

回想一下，在 Transformer 的注意力层中，每个令牌都会关注先前序列中的有限列表中的令牌。论文中提到 GPT-2 模型的上下文长度为 1024 个令牌，比 GPT-1 的 512 有所增加。换句话说，令牌是输入到 LLM 的基本“原子”。分词过程是将 Python 中的原始字符串转换为令牌列表，反之亦然。作为另一个流行的示例，如果你查看 [Llama 2](https://arxiv.org/abs/2307.09288) 论文并搜索“token”，你会找到 63 个相关条目。例如，论文声称他们训练了 2 万亿个令牌等。

### 分词复杂性的简要介绍

在深入实现细节之前，让我们简单说明一下理解分词过程的重要性。分词是大型语言模型（LLM）中许多奇怪现象的核心，我建议你不要忽视它。很多看似与神经网络架构相关的问题实际上可以追溯到分词。这里有一些例子：

- 为什么 LLM 不能拼写单词？**分词**。
- 为什么 LLM 不能完成像反转字符串这样简单的字符串处理任务？**分词**。
- 为什么 LLM 在非英语语言（如日语）上的表现更差？**分词**。
- 为什么 LLM 在简单算术上表现不佳？**分词**。
- 为什么 GPT-2 在 Python 编程时遇到更多不必要的麻烦？**分词**。
- 为什么我的 LLM 在看到字符串 " " 时突然停止？**分词**。
- 为什么我会收到关于“尾随空格”的奇怪警告？**分词**。
- 为什么我在询问 “SolidGoldMagikarp” 时 LLM 会出问题？**分词**。
- 为什么在使用 LLM 时我更倾向于使用 YAML 而不是 JSON？**分词**。
- 为什么 LLM 实际上不是端到端的语言建模？**分词**。
- 痛苦的真正根源是什么？**分词**。

我们将在视频的最后回到这些问题。

### 分词的视觉预览

接下来，让我们加载这个 [分词网页应用](https://tiktokenizer.vercel.app)。这个网页应用的优点在于分词过程在你的浏览器中实时运行，你可以轻松地在输入框中输入一些文本字符串，并在右侧看到分词结果。顶部显示我们当前正在使用 `gpt2` 分词器，并且可以看到我们粘贴的字符串被分成了 300 个令牌。它们以不同的颜色明确显示：

![tiktokenizer](assets/tiktokenizer.png)

例如，字符串 "Tokenization" 被编码成令牌 30642 和随后的令牌 1634。令牌 " is"（注意这是三个字符，包括前面的空格，这一点很重要！）的索引是 318。请注意空格的存在，因为它确实存在于字符串中并且必须与其他字符一起进行分词，但在可视化中通常为了清晰而省略。你可以在应用底部切换其可视化。同样，令牌 " at" 的索引是 379，" the" 的索引是 262 等等。

接下来，我们有一个简单的算术示例。在这里，我们可以看到数字可能被分词器不一致地分解。例如，数字 127 是一个三字符的单个令牌，但数字 677 被分成两个令牌：令牌 " 6"（再次注意前面的空格！）和令牌 "77"。我们依赖于大型语言模型来理解这种任意性。它必须在其参数内部并在训练过程中学习这两个令牌（" 6" 和 "77" 实际上组合成数字 677）。同样，如果 LLM 预测这个求和的结果是数字 804，它必须在两个时间步输出：首先发出令牌 " 8"，然后发出令牌 "04"。注意所有这些分割看起来都是完全任意的。在下面的例子中，我们可以看到 1275 是 "12" 跟着 "75"，6773 实际上是两个令牌 " 6" 和 "773"，8041 是 " 8" 和 "041"。

（待续...）
（TODO: 如果无法自动从视频生成，则可能会继续此部分内容 :））
单的算术示例。在这里，我们可以看到数字可能被分词器不一致地分解。例如，数字 127 是一个三字符的单个令牌，但数字 677 被分成两个令牌：令牌 " 6"（再次注意前面的空格！）和令牌 "77"。我们依赖于大型语言模型来理解这种任意性。它必须在其参数内部并在训练过程中学习这两个令牌（" 6" 和 "77" 实际上组合成数字 677）。同样，如果 LLM 预测这个求和的结果是数字 8