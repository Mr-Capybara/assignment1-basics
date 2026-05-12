# Byte-Pair Encoding (BPE) Tokenizer 中文说明

本文按 `cs336_assignment1_basics.pdf` 第 2 节的结构整理，覆盖 BPE 相关正文、提示、例子、problem 以及解答要点。这里的 BPE 指 **byte-level BPE tokenizer**：先把任意 Unicode 字符串表示成 UTF-8 字节序列，再在字节序列上训练和应用 BPE。

## 2 Byte-Pair Encoding (BPE) Tokenizer

作业第一部分要求训练并实现一个 **byte-level byte-pair encoding (BPE)** tokenizer。

核心流程是：

1. 任意 Unicode 字符串先编码成字节序列。
2. 在字节序列上训练 BPE 词表和 merge 规则。
3. 之后用 tokenizer 把文本字符串编码成整数 token ID 序列，用于语言模型训练。

byte-level BPE 的好处是：

- 因为任意文本都能表示成字节，所以没有 out-of-vocabulary 问题。
- 相比纯 byte tokenizer，BPE 会把常见字节片段合并成更长 token，从而缩短序列长度。

低资源提示：即使用本地 Apple Silicon 或 CPU，只要实现正确且高效，也能训练小型语言模型生成较流畅的 TinyStories 风格文本。官方 staff solution 在 M4 Max 36GB RAM 上，MPS 约 5 分钟内可训练出合理模型，CPU 约 30 分钟。

## 2.1 The Unicode Standard

Unicode 是一种文本编码标准，它把字符映射到整数 code point。截至 Unicode 17.0，标准包含 172 种文字系统中的 159,801 个字符。

例子：

```python
>>> ord("牛")
29275
>>> chr(29275)
"牛"
```

- `ord()`：把单个 Unicode 字符转成整数 code point。
- `chr()`：把整数 code point 转回对应字符。

### Problem (unicode1): Understanding Unicode

#### (a) `chr(0)` 返回什么 Unicode 字符？

**答案：** `chr(0)` 返回 Unicode code point U+0000，也就是空字符 NUL。

#### (b) 这个字符的字符串表示 `__repr__()` 和打印表示有什么不同？

**答案：** `repr(chr(0))` 会显示为 `'\x00'`，也就是可见的转义形式；`print(chr(0))` 会打印真实 NUL 控制字符，通常在终端里不可见。

#### (c) 当这个字符出现在文本中会发生什么？

示例：

```python
>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

**答案：** NUL 字符会真实存在于字符串中并占一个字符位置，但打印时通常不可见，因此视觉上可能像什么都没有发生，只是文本中间包含了一个不可见控制字符。

## 2.2 Unicode Encodings

Unicode 定义的是字符到 code point 的映射。如果直接在 Unicode code point 上训练 tokenizer，词表会非常大且稀疏，大约 150K 项，而且很多字符很少见。

因此，作业选择先用 Unicode encoding 把字符转成字节序列。Unicode 标准定义了三种编码：

- UTF-8
- UTF-16
- UTF-32

其中 UTF-8 是互联网上最常用的编码，覆盖超过 98% 的网页。

Python 中：

```python
test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)
print(list(utf8_encoded))
print(utf8_encoded.decode("utf-8"))
```

要点：

- `str.encode("utf-8")`：Unicode 字符串 -> UTF-8 bytes。
- 遍历 `bytes` 对象会得到 0 到 255 的整数。
- `bytes.decode("utf-8")`：UTF-8 bytes -> Unicode 字符串。
- 一个 Unicode 字符不一定对应一个字节。例如 `"hello! こんにちは!"` 字符串长度是 13，但 UTF-8 编码后是 23 个字节。

把 Unicode code point 序列转成 UTF-8 bytes 后，原本 21-bit、约 150K 有效值的问题变成了 256 个 byte 值的问题。byte-level tokenizer 的初始词表只需要覆盖 0 到 255，因此天然不会有 OOV。

### Problem (unicode2): Unicode Encodings

#### (a) 为什么更偏好在 UTF-8 bytes 上训练 tokenizer，而不是 UTF-16 或 UTF-32？

**答案：** UTF-8 对 ASCII 和常见英文文本更紧凑，且是互联网文本的主流编码；UTF-16 和 UTF-32 往往会引入更多零字节或固定宽度冗余，使序列更长或统计更稀疏，不利于压缩和 tokenizer 训练。

#### (b) 为什么下面这个 UTF-8 解码函数是错误的？给出一个输入例子。

错误函数：

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

**例子：**

```python
"牛".encode("utf-8")
# b'\xe7\x89\x9b'
```

**答案：** UTF-8 中一个字符可能由多个字节组成，例如 `"牛"` 是三个字节 `b'\xe7\x89\x9b'`；错误函数逐字节解码，每个单独字节都不是完整 UTF-8 字符，因此会抛出 `UnicodeDecodeError`，而不是得到 `"牛"`。

#### (c) 给出一个无法解码成任何 Unicode 字符的两字节序列。

**例子：**

```python
b"\x80\x80"
```

**答案：** `0x80` 是 UTF-8 continuation byte，不能作为合法 UTF-8 字符的起始字节；两个 continuation byte 连在一起没有合法起始字节，因此不能解码成 Unicode 字符。

## 2.3 Subword Tokenization

纯 byte-level tokenization 可以避免 OOV，但会导致序列过长。一个 10 个词的句子，在 word-level LM 中可能只有 10 个 token，但在 byte 或 character-level 模型中可能有 50 个甚至更多 token。

长序列的问题：

- 每步训练计算量更大。
- 语言模型需要处理更长距离依赖，训练更困难。

Subword tokenization 是 word-level 和 byte-level 之间的折中：

- byte-level 初始词表只有 256 项。
- subword tokenizer 用更大的词表换取更短的输入序列。
- 如果 `b"the"` 经常出现，把它加入词表后，原本 3 个 byte token 可以变成 1 个 token。

BPE 的作用是选择哪些 subword 加入词表。BPE 是一种压缩算法，迭代地找到最频繁的相邻 token pair，把它们合并成一个新的 token。它倾向于把高频片段合并，从而最大化输入序列压缩率。

本作业实现的是 byte-level BPE：

- 词表项可以是单个 byte，也可以是多个 byte merge 后的 bytes。
- 训练 BPE tokenizer 指的是构造词表和 merge 列表。

## 2.4 BPE Tokenizer Training

BPE tokenizer 训练有三步：

1. 初始化词表。
2. pre-tokenization。
3. 计算 BPE merges。

### Vocabulary initialization

tokenizer 词表是 bytestring token 到整数 ID 的一一映射。因为本作业是 byte-level BPE，初始词表就是全部 256 个 byte 值。

通常可表示成：

```python
vocab: dict[int, bytes] = {
    0: b"\x00",
    1: b"\x01",
    ...
    255: b"\xff",
}
```

如果有 special tokens，也要加入词表，并分配固定 ID。

### Pre-tokenization

理论上，可以直接在整个语料上统计相邻 byte pair，然后每次合并最高频 pair。但这样很慢，因为每次 merge 后都需要重新扫一遍语料。

另一个问题是，如果直接跨整个语料合并 byte，可能学到很多只差标点的 token，例如 `dog!` 和 `dog.`，它们会得到完全不同 ID，虽然语义很接近。

因此需要 pre-tokenization：先做粗粒度切分，再在每个 pre-token 内统计和合并 byte pair。

例子：如果 pre-token `"text"` 出现 10 次，那么统计相邻 byte pair 时，`t` 和 `e` 的共现可以直接增加 10，而不必每次扫描原始文本。

因为是 byte-level BPE，每个 pre-token 最终表示为 UTF-8 bytes 序列。

原始 BPE 论文中，pre-tokenization 只是按空格切分：

```python
s.split(" ")
```

这种方式仍见于一些基于 SentencePiece 的 tokenizer，例如 Llama 1/2 tokenizer。

现代 tokenizer 多使用来自 GPT-2 的 regex pre-tokenizer。本作业使用如下正则：

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

示例：

```python
import regex as re

re.findall(PAT, "some text that i'll pre-tokenize")
# ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

实现时应使用 `re.finditer`，避免把所有 pre-token 一次性存入列表，从而减少内存开销。

### Compute BPE merges

把输入文本 pre-tokenize 并转换成 UTF-8 bytes 后，就可以训练 BPE merges。

高层算法：

1. 统计所有 pre-token 内的相邻 token pair 频率。
2. 找出频率最高的 pair `(A, B)`。
3. 把每个 pre-token 中出现的 `(A, B)` 替换成合并 token `AB`。
4. 把新 token `AB` 加入词表。
5. 记录 merge `(A, B)`。
6. 重复，直到达到目标词表大小或没有可合并 pair。

注意事项：

- 不考虑跨 pre-token 边界的 pair。
- 最终词表大小 = 初始 256 byte token + special tokens + merge 次数。
- 如果多个 pair 频率并列最高，按字典序选择更大的 pair。

并列规则示例：

```python
max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
# ("BA", "A")
```

也就是说，如果这些 pair 频率一样，选择 `("BA", "A")`。

脚注要点：原始 BPE 论文包含 end-of-word token，但本作业的 byte-level BPE 不添加 end-of-word token，因为空格和标点本身都在 byte 词表中。merge 规则会自然学到词边界。

### Special tokens

special token 用于编码元信息，例如文档边界或序列结束标记。典型例子是：

```text
<|endoftext|>
```

要求：

- encoding 时 special token 永远不能被拆成多个 token。
- special token 必须加入词表，并有固定 token ID。
- 训练 BPE merge 统计时，special token 是硬边界，不能跨 special token 合并。

### Example (bpe_example): BPE training example

语料：

```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

special token：

```text
<|endoftext|>
```

#### Vocabulary

初始词表包含：

- special token `<|endoftext|>`
- 256 个 byte 值

#### Pre-tokenization

为了突出 merge 过程，例子中假设 pre-tokenization 只是按空白切分。统计得到：

```python
{
    "low": 5,
    "lower": 2,
    "widest": 3,
    "newest": 6,
}
```

实际实现中可以表示成：

```python
dict[tuple[bytes, ...], int]
```

例如：

```python
{
    (b"l", b"o", b"w"): 5,
    ...
}
```

Python 没有单独的 byte 类型，单个 byte 也用 `bytes` 对象表示。

#### Merges

第一轮统计相邻 pair：

```text
lo: 7
ow: 7
we: 8
er: 2
wi: 3
id: 3
de: 3
es: 9
st: 9
ne: 6
ew: 6
```

`("e", "s")` 和 `("s", "t")` 都出现 9 次，并列最高。按字典序选择更大的 pair，所以先 merge：

```text
s t -> st
```

merge 后 pre-token 变为：

```python
{
    (b"l", b"o", b"w"): 5,
    (b"l", b"o", b"w", b"e", b"r"): 2,
    (b"w", b"i", b"d", b"e", b"st"): 3,
    (b"n", b"e", b"w", b"e", b"st"): 6,
}
```

第二轮最高频 pair 是 `(e, st)`，频率为 9，merge 成 `est`。

继续训练，最终 merge 序列为：

```text
s t
e st
o w
l ow
w est
n e
ne west
w i
wi d
wid est
low e
lowe r
```

如果只做 6 次 merge，则 merges 为：

```text
s t
e st
o w
l ow
w est
n e
```

词表额外包含：

```text
st, est, ow, low, west, ne
```

此时单词 `newest` 会被编码为：

```text
ne, west
```

## 2.5 Experimenting with BPE Tokenizer Training

这一节要求在 TinyStories 和 OpenWebText 上训练 byte-level BPE tokenizer。

训练前建议先查看 TinyStories 数据内容，理解数据风格。

### Parallelizing pre-tokenization

pre-tokenization 通常是主要瓶颈。可以使用 Python 标准库 `multiprocessing` 并行化。

并行化建议：

- 把语料切成多个 chunk。
- chunk 边界应落在 special token 开始处。
- starter code 在 `cs336_basics/pretokenization_example.py`，也可从作业链接直接使用。

这样切分总是有效的，因为不希望跨文档边界 merge。作业场景中可以假设能用 `<|endoftext|>` 进行切分，不需要处理一个巨大语料完全没有 `<|endoftext|>` 的极端情况。

### Removing special tokens before pre-tokenization

在用 regex pattern 进行 pre-tokenization 之前，应从 corpus 或 chunk 中移除 special tokens。

更准确地说：

- 要按 special tokens split 文本。
- 对 split 后的普通文本片段分别做 pre-tokenization。
- special token 是训练时的硬边界。
- special token 本身不参与 merge 统计。

例子：

```text
[Doc 1]<|endoftext|>[Doc 2]
```

应拆成：

```text
[Doc 1]
[Doc 2]
```

分别 pre-tokenize，禁止跨 `<|endoftext|>` 合并。

实现可用：

```python
re.split(...)
```

delimiter 可以由 special tokens 组成：

```python
"|".join(re.escape(tok) for tok in special_tokens)
```

注意必须使用 `re.escape`，因为 special token 里可能包含 `|` 等 regex 特殊字符。

测试 `test_train_bpe_special_tokens` 会检查这一点：special token 应加入词表，但其他普通词表项中不应包含 `b"<|"` 这类来自 special token 的片段。

### Optimizing the merging step

朴素 BPE 实现很慢，因为每次 merge 都重新遍历所有 byte pair 来找最高频 pair。

优化思路：

- merge 之后，只有和被 merge pair 相邻、重叠的 pair count 会改变。
- 可以缓存所有 pair 的计数，并在每轮 merge 后增量更新相关 pair。
- 这能显著加速。

限制：

- BPE merge 过程本身在 Python 中不容易并行化。
- 更现实的并行化重点是 pre-tokenization。

### Low-Resource Tip: Profiling

应使用 profiling 工具定位瓶颈，例如：

- `cProfile`
- `py-spy`

优化时应优先处理 profile 显示的耗时热点，而不是凭感觉修改。

### Low-Resource Tip: Downscaling

不要一开始就直接在完整 TinyStories 上训练。建议先用小数据集调试，例如 TinyStories validation set：

- validation set 大约 22K documents。
- full training set 大约 2.12M documents。

downscaling 的一般原则：

- debug dataset 要足够大，能暴露和完整配置相同的瓶颈。
- 但也要足够小，方便快速迭代。

### Problem (train_bpe): BPE Tokenizer Training

#### 题目要求

实现一个函数：给定输入文本文件路径，训练 byte-level BPE tokenizer。

输入至少包括：

```python
input_path: str
vocab_size: int
special_tokens: list[str]
```

含义：

- `input_path`：BPE tokenizer 训练文本路径。
- `vocab_size`：最终最大词表大小，包括初始 256 byte token、special tokens、merge 产生的新 token。
- `special_tokens`：加入词表的特殊字符串。训练时把它们当成硬边界，禁止跨它们 merge，但它们不参与 merge 统计。

输出：

```python
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

含义：

- `vocab`：token ID 到 token bytes 的映射。
- `merges`：训练产生的 BPE merges，按创建顺序排列；每项 `(token1, token2)` 表示把 `token1 + token2` 合并。

测试入口：

```python
tests/adapters.py::run_train_bpe
```

测试命令：

```bash
uv run pytest tests/test_train_bpe.py
```

#### 解答要点

一个正确实现应满足：

1. 初始化 256 个单 byte token。
2. 把 `special_tokens` 编码成 UTF-8 bytes 后加入词表。
3. 读取训练文本。
4. 先按 special tokens split，避免跨 special token 合并。
5. 对普通文本片段用 GPT-2 regex pre-tokenizer：

   ```python
   PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
   ```

6. 把每个 pre-token 编码为 UTF-8 bytes，并表示为 `tuple[bytes, ...]`。
7. 统计 pre-token 频率，避免保存所有重复 pre-token。
8. 每轮统计 pair 频率，选择频率最高的 pair；并列时选择字典序更大的 pair。
9. 合并所有 pre-token 中该 pair 的出现位置。
10. 把新 bytes token 加入词表并把 pair 追加到 `merges`。
11. 重复直到达到 `vocab_size` 或没有可合并 pair。

性能上，pre-tokenization 应尽量并行；merge 阶段可通过缓存 pair counts 和增量更新优化。

可选高级实现：可以把训练关键部分用 C++、Rust 等系统语言实现，但要注意 Python 内存复制成本、构建说明，以及 GPT-2 regex 在很多 regex engine 中支持不好。官方说明中提到 Oniguruma 支持负向前瞻且速度尚可；Python `regex` 包甚至更快。

### Problem (train_bpe_tinystories): BPE Training on TinyStories

#### (a) 题目要求

在 TinyStories 数据集上训练 byte-level BPE tokenizer：

- 最大词表大小：10,000。
- 添加 special token：`<|endoftext|>`。
- 把训练得到的 vocab 和 merges 序列化到磁盘。
- 报告训练耗时、内存占用、词表中最长 token，以及它是否合理。

资源限制：

- 不用 GPU。
- 时间不超过 30 分钟。
- RAM 不超过 30GB。

Hint：

使用 multiprocessing 做 pre-tokenization，并利用两点，应能把训练时间压到 2 分钟以内：

1. `<|endoftext|>` 在数据文件中分隔文档。
2. `<|endoftext|>` 在应用 BPE merges 前作为特殊情况处理。

#### 解答要点

这道题的数值答案依赖机器、实现和实际下载的数据。回答应包含如下信息：

```text
我在 TinyStories 上训练了 vocab_size=10000 的 byte-level BPE，并加入 <|endoftext|>。
训练耗时约 [填入时间]，峰值内存约 [填入内存]；最长 token 是 [填入最长 token]。
它通常应是 TinyStories 中高频的完整词、词组片段、带空格前缀的短语，或数据集中反复出现的模板化字符串，因此如果它来自常见故事语言模式，就是合理的。
```

如果最长 token 包含 `<|endoftext|>` 的片段，说明 special token 处理错了，因为 special token 不应参与 merge 统计。

#### (b) Profile 代码：训练过程中哪部分最耗时？

**解答要点：** 需要用 `cProfile` 或 `py-spy` 基于自己的实现测量。通常最耗时的是 pre-tokenization 和 pair count/merge 更新；如果没有并行化 pre-tokenization，它往往是最大瓶颈。优化后，merge 阶段的 pair 统计或增量更新可能成为主要耗时。

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText

#### (a) 题目要求

在 OpenWebText 上训练 byte-level BPE tokenizer：

- 最大词表大小：32,000。
- 把 vocab 和 merges 序列化到磁盘。
- 报告词表中最长 token，以及它是否合理。

资源限制：

- 不用 GPU。
- 时间不超过 12 小时。
- RAM 不超过 100GB。

#### 解答要点

这道题也依赖实际训练结果。回答应包含：

```text
我在 OpenWebText 上训练了 vocab_size=32000 的 byte-level BPE。
最长 token 是 [填入最长 token]。
它是否合理取决于该 token 是否对应 OpenWebText 中高频重复片段，例如 URL 片段、HTML/网页残留、常见英文短语、空格前缀词组或其他网络文本模式。
```

如果最长 token 是明显的网页模板、URL 子串或重复格式，也可能合理，因为 OpenWebText 来自网页语料，数据分布比 TinyStories 更杂。

#### (b) 比较 TinyStories tokenizer 和 OpenWebText tokenizer

**解答要点：** TinyStories tokenizer 更偏儿童故事域，会学到简单叙事、常见角色、短句和故事模板中的高频片段；OpenWebText tokenizer 覆盖面更广，会学到更多网页文本、通用英文、URL/HTML 残留、专有名词和长尾模式。通常 OpenWebText tokenizer 泛化到开放域文本更好，而 TinyStories tokenizer 在 TinyStories 域内可能压缩更有效。

## 2.6 BPE Tokenizer: Encoding and Decoding

前面训练得到的是：

- `vocab`
- `merges`

接下来要实现 tokenizer，用它把文本编码成 token IDs，并把 token IDs 解码回文本。

## 2.6.1 Encoding text

BPE encoding 与训练过程类似。

### Step 1: Pre-tokenize

先用同样的 regex pre-tokenizer 切分输入字符串，再把每个 pre-token 表示为 UTF-8 bytes 序列。

每个 pre-token 独立处理，不允许跨 pre-token 边界 merge。

### Step 2: Apply the merges

按训练时产生 merges 的顺序，把 merge 规则应用到每个 pre-token 上。

也就是说，越早训练出的 merge 优先级越高。

### Example (bpe_encoding): BPE encoding example

输入：

```text
the cat ate
```

词表：

```python
{
    0: b" ",
    1: b"a",
    2: b"c",
    3: b"e",
    4: b"h",
    5: b"t",
    6: b"th",
    7: b" c",
    8: b" a",
    9: b"the",
    10: b" at",
}
```

merges：

```python
[
    (b"t", b"h"),
    (b" ", b"c"),
    (b" ", b"a"),
    (b"th", b"e"),
    (b" a", b"t"),
]
```

pre-tokenizer 把输入切成：

```python
["the", " cat", " ate"]
```

处理 `"the"`：

```python
[b"t", b"h", b"e"]
-> [b"th", b"e"]
-> [b"the"]
-> [9]
```

处理 `" cat"`：

```python
[b" ", b"c", b"a", b"t"]
-> [b" c", b"a", b"t"]
-> [7, 1, 5]
```

处理 `" ate"`：

```python
[b" ", b"a", b"t", b"e"]
-> [b" a", b"t", b"e"]
-> [b" at", b"e"]
-> [10, 3]
```

最终编码结果：

```python
[9, 7, 1, 5, 10, 3]
```

### Special tokens

Tokenizer 在 encoding 时必须正确处理用户提供的 special tokens。

要求：

- special token 出现在文本中时，应整体映射为一个 token ID。
- 不应被 regex pre-tokenizer 或 BPE merge 拆开。
- 如果 special tokens 重叠，例如：

  ```python
  ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
  ```

  应优先匹配更长的 special token，避免把长 special token 错拆成短 special token。

### Memory considerations

对于无法一次性放入内存的大文件，需要流式 tokenization。

目标：

- 不把整个文件读入内存。
- 分块处理，保持内存复杂度接近常数。
- 但必须确保 token 不跨 chunk 边界，否则流式结果会和一次性 tokenize 整个文本不同。

因此 `encode_iterable` 需要小心处理 chunk 边界。可行策略是按行或按安全边界处理，并保证 special token 和 pre-token 不被错误切断；或者维护缓冲区，只在确定不会跨边界的位置输出 token。

## 2.6.2 Decoding text

解码 token IDs 到文本的过程：

1. 用 `vocab` 查每个 ID 对应的 bytes。
2. 拼接所有 bytes。
3. 用 UTF-8 解码成 Unicode 字符串。

注意：任意 ID 序列不一定能拼成合法 UTF-8 bytes。遇到非法 bytes 时，应该用 Unicode replacement character 替换，也就是 U+FFFD。

Python 实现：

```python
text = b"".join(byte_tokens).decode("utf-8", errors="replace")
```

`errors="replace"` 会自动把非法 UTF-8 片段替换为 `�`。

### Problem (tokenizer): Implementing the tokenizer

#### 题目要求

实现一个 `Tokenizer` 类，给定词表和 merges 后，能编码和解码文本，并支持用户提供的 special tokens。

推荐接口：

```python
def __init__(self, vocab, merges, special_tokens=None)
```

参数：

- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `special_tokens: list[str] | None = None`

如果 special token 不在词表中，应追加到词表。

```python
@classmethod
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
```

从序列化后的 vocab 和 merges 文件构造 tokenizer。

```python
def encode(self, text: str) -> list[int]
```

把输入文本编码成 token ID 序列。

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```

给定字符串 iterable，例如文件句柄，惰性地产生 token IDs，用于大文件的内存高效 tokenization。

```python
def decode(self, ids: list[int]) -> str
```

把 token IDs 解码回文本。

测试入口：

```python
tests/adapters.py::get_tokenizer
```

测试命令：

```bash
uv run pytest tests/test_tokenizer.py
```

#### 解答要点

一个正确实现应满足：

1. 构造 `id -> bytes` 的 vocab。
2. 构造 `bytes -> id` 的反向词表。
3. 把 `merges` 转成 priority map，例如：

   ```python
   merge_rank = {pair: rank for rank, pair in enumerate(merges)}
   ```

4. encoding 时先处理 special tokens，并保证 special token 整体输出一个 ID。
5. 普通文本片段使用同一个 GPT-2 regex pre-tokenizer。
6. 每个 pre-token 先拆成单 byte bytes 序列，例如 `b"abc"` -> `[b"a", b"b", b"c"]`。
7. 在 pre-token 内按 merge rank 反复合并最高优先级 pair，直到没有可应用 merge。
8. 把最终 bytes token 映射为 ID。
9. decoding 时拼接 bytes 并用 `decode("utf-8", errors="replace")`。
10. `encode_iterable` 应惰性产生 ID，避免把整个大文件读入内存。

测试会覆盖：

- 空字符串 roundtrip。
- ASCII 和 Unicode roundtrip。
- 与 GPT-2 `tiktoken` 的结果一致性。
- special token 不被拆分。
- 重叠 special tokens 的最长匹配。
- 大文件 iterable encoding 的内存使用。

## 2.7 Experiments

### Problem (tokenizer_experiments): Experiments with tokenizers

#### (a) 采样 TinyStories 和 OpenWebText 各 10 个文档，分别用对应 tokenizer 编码，压缩率是多少？

要求：

- TinyStories tokenizer：之前训练的 10K vocab。
- OpenWebText tokenizer：之前训练的 32K vocab。
- 指标：compression ratio，即 bytes/token。

计算方式：

```python
compression_ratio = 原始 UTF-8 字节数 / token 数
```

**解答要点：** 需要基于实际采样文档计算。回答格式：

```text
TinyStories tokenizer 在 TinyStories 样本上的压缩率约为 [x] bytes/token；
OpenWebText tokenizer 在 OpenWebText 样本上的压缩率约为 [y] bytes/token。
较高的 bytes/token 表示 tokenizer 用更少 token 表示同样文本，压缩效果更好。
```

#### (b) 如果用 TinyStories tokenizer tokenize OpenWebText 样本，会发生什么？

**解答要点：** 通常压缩率会下降，即 bytes/token 变小、token 数增加，因为 TinyStories tokenizer 的词表适配儿童故事域，不擅长 OpenWebText 中的网页文本、URL、专有名词、技术词汇和多样风格。定性上会看到更多文本被拆成较短 subword 或 byte-level 片段。

#### (c) 估计 tokenizer 吞吐量；tokenize 825GB 的 Pile 需要多久？

计算方式：

```python
throughput = 处理字节数 / 处理时间  # bytes/second
time_for_pile = 825 * 1024**3 / throughput
```

如果按十进制 GB，也可以用：

```python
time_for_pile = 825 * 10**9 / throughput
```

回答时应说明使用的是二进制 GiB 还是十进制 GB。

**解答要点：**

```text
我的 tokenizer 吞吐量约为 [x] bytes/s。
按 825GB 计算，tokenize the Pile 约需 [t] 秒，也就是约 [t/3600] 小时。
```

#### (d) 把 TinyStories 和 OpenWebText 的 train/dev 数据编码成整数 token ID 序列，并解释为什么 `uint16` 合适。

**答案：** 如果词表大小不超过 65,536，`uint16` 可以表示所有 token ID；本作业的 TinyStories tokenizer 是 10K vocab，OpenWebText tokenizer 是 32K vocab，都小于 65,536。因此用 `uint16` 比 `int32` 或 `int64` 更省磁盘和内存，同时仍能完整表示 token IDs。

推荐序列化方式：

```python
import numpy as np

ids = np.array(token_ids, dtype=np.uint16)
np.save("tokens.npy", ids)
```

后续训练语言模型时，可以直接从这些预编码 token ID 序列中采样 batch。
