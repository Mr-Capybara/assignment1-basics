# 3 Transformer Language Model Architecture

## 3 Transformer 语言模型架构

语言模型的输入是一批整数 token ID 序列，也就是形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`；输出是一批在词表上的归一化概率分布，也就是形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor，其中每个输入 token 对应的预测分布都是对下一个词的预测。训练语言模型时，我们用这些 next-word 预测来计算真实下一个词和预测下一个词之间的交叉熵损失。推理生成文本时，我们取最终时间步，也就是序列最后一项，对应的 next-word 预测分布来生成下一个 token，例如取概率最高的 token、从分布中采样等；然后把生成的 token 加入输入序列，并重复这一过程。

在作业的这一部分，你将从头构建这个 Transformer 语言模型。我们会先给出模型的高层描述，然后逐步详细说明各个组件。

## 3.1 Transformer LM

给定一个 token ID 序列，Transformer 语言模型会使用输入嵌入把 token ID 转换为稠密向量，将嵌入后的 token 传入 `num_layers` 个 Transformer block，然后应用一个学习得到的线性投影，即“输出嵌入”或“LM head”，产生预测下一个 token 的 logits。示意图见图 1。

**图 1：Transformer 语言模型概览。**

流程为：

`Inputs -> Token Embedding -> num_layers Transformer Blocks -> Norm -> Linear (Output Embedding) -> Softmax -> Output Probabilities`

其中 `num_layers` 表示 Transformer block 的层数。

**图 2：pre-norm Transformer block。**

输入张量形状为：

```text
(batch_size, seq_len, d_model)
```

pre-norm block 内部包含：

- `Norm`
- `Causal Multi-Head Self-Attention w/ RoPE`
- `Add`
- `Norm`
- `Position-Wise Feed-Forward`
- `Add`

输出张量形状为：

```text
(batch_size, seq_len, d_model)
```

### Token Embeddings

在最开始的步骤中，Transformer 会把一批 token ID 序列嵌入为一个向量序列，这些向量包含 token 身份的信息，也就是图 1 中的红色块。

更具体地说，给定一个 token ID 序列，Transformer 语言模型使用 token embedding layer 产生一个向量序列。每个 embedding layer 接收一个形状为 `(batch_size, sequence_length)` 的整数张量，并产生一个形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

### Pre-norm Transformer Block

嵌入之后，激活值会经过若干结构相同的神经网络层处理。标准的 decoder-only Transformer 语言模型由 `num_layers` 个相同的层构成，这些层通常称为 Transformer “blocks”。每个 Transformer block 接收形状为 `(batch_size, sequence_length, d_model)` 的输入，并返回形状为 `(batch_size, sequence_length, d_model)` 的输出。每个 block 都会跨序列聚合信息，也就是通过 self-attention，并对信息做非线性变换，也就是通过 feed-forward layers。

经过 `num_layers` 个 Transformer blocks 之后，我们会取最终激活值，并把它转换为词表上的一个分布。

我们将实现 “pre-norm” Transformer block，详见第 3.4 节。它还需要在最后一个 Transformer block 之后使用 layer normalization，后面会详细说明，以确保输出被适当地缩放。

在这个 normalization 之后，我们将使用一个标准的学习得到的线性变换，把 Transformer blocks 的输出转换为预测下一个 token 的 logits，例如可参见 A. Radford et al. [7] 的等式 2。

## 3.2 注：Batching、Einsum 和高效计算

在整个 Transformer 中，我们会对许多 batch-like 输入执行相同的计算。下面是几个例子：

- batch 中的元素：我们会对每个 batch element 应用同一个 Transformer forward 操作。
- 序列长度：像 RMSNorm 和 feed-forward 这样的 “position-wise” 操作，会对序列中的每个位置以相同方式运行。
- attention heads：在 “multi-headed” attention 操作中，attention 操作会跨 attention heads 批量执行。

我们需要一种符合人体工学的方式来执行这些操作，使其既能充分利用 GPU，又容易阅读和理解。许多 PyTorch 操作可以在张量开头接收额外的 “batch-like” 维度，并在这些维度上高效地重复或广播操作。

举例来说，假设我们正在做一个 position-wise、batched 操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的“数据张量” `D`，并且想和一个形状为 `(d_model, d_model)` 的矩阵 `A` 做 batched vector-matrix multiply。在这种情况下，`D @ A` 会执行 batched matrix multiply，这是 PyTorch 中高效的 primitive，其中 `(batch_size, sequence_length)` 维度会被批量处理。

因此，假设你的函数可能接收额外的 batch-like 维度，并把这些维度保持在 PyTorch shape 的开头，是很有帮助的。为了以这种方式组织张量，它们可能需要通过许多步 `view`、`reshape` 和 `transpose` 来调整形状。这会有点麻烦，而且经常会让代码在做什么、张量形状是什么变得难以阅读。

一个更符合人体工学的选择是在 `torch.einsum` 中使用 einsum notation，或者使用与框架无关的库，例如 `einops` 或 `einx`。两个关键操作是 `einsum` 和 `rearrange`：`einsum` 可以对输入张量的任意维度做 tensor contractions；`rearrange` 可以重新排序、拼接和拆分任意维度。事实证明，机器学习中几乎所有操作都是某种维度调整和张量收缩的组合，偶尔再加上一个通常逐点的非线性函数。这意味着使用 einsum notation 可以让很多代码更易读、更灵活。

我们强烈建议在本课程中学习并使用 einsum notation。之前没有接触过 einsum notation 的同学应该使用 `einops`，文档见 here；已经熟悉 `einops` 的同学应该学习更通用的 `einx`，见 here。[^4] 我们提供的环境中已经安装了这两个包。

下面给出一些 einsum notation 的使用示例。这些示例是 `einops` 文档的补充，你应该先阅读其文档。

### Example (einstein_example1): 使用 einops.einsum 做 batched matrix multiplication

```python
import torch
from einops import rearrange, einsum

## Basic implementation
Y = D @ A.T
# Hard to tell the input and output shapes and what they mean.
# What shapes can D and A have, and do any of these have unexpected behavior?

## Einsum is self-documenting and robust
#                          D                A     ->          Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

## Or, a batched version where D can have any leading dimensions but A is constrained.
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

### Example (einstein_example2): 使用 einops.rearrange 做 broadcasted operations

我们有一批图像，并且希望对每张图像根据某个缩放因子生成 10 个变暗版本：

```python
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## Reshape and multiply
dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

## Or in one go:
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

### Example (einstein_example3): 使用 einops.rearrange 做 pixel mixing

假设我们有一批图像，表示为形状为 `(batch, height, width, channel)` 的张量；我们想对图像的所有像素执行一个线性变换，但这个变换应该对每个 channel 独立发生。我们的线性变换表示为一个形状为 `(height * width, height * width)` 的矩阵 `B`。

```python
channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
```

改用 `einops`：

```python
height = width = 32

## Rearrange replaces clunky torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)
```

或者，如果你想更激进一些：使用 `einx.dot`，即 `einops.einsum` 的 `einx` 等价物，一次完成：

```python
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

这里第一种实现可以通过在前后添加注释来说明输入和输出形状，但这很笨重，也容易引入 bug。使用 einsum notation 时，文档就是实现本身。

Einsum notation 可以处理任意输入 batching dimensions，同时还有一个关键好处：它具有自说明性。在使用 einsum notation 的代码中，输入和输出张量的相关形状清晰得多。对于其余张量，你可以考虑使用 Tensor 类型提示，例如使用 `jaxtyping` 库，它并不专属于 JAX。

我们会在 assignment 2 中更多讨论使用 einsum notation 的性能含义，但就目前而言，你只需要知道它们几乎总是优于替代写法。

### 3.2.1 数学记号和内存顺序

许多机器学习论文在记号中使用 row vectors，这得到的表示方式与 NumPy 和 PyTorch 默认使用的 row-major memory ordering 很匹配。使用 row vectors 时，线性变换写作：

$$
y = xW^\top
\tag{1}
$$

其中 row-major 的 $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$，row-vector 的 $x \in \mathbb{R}^{1 \times d_\text{in}}$。注意，这使我们可以通过增加 $x$ 的最外层维度来批量化输入，也就是说，可以用矩阵输入 $X \in \mathbb{R}^{\text{batch} \times d_\text{in}}$ 替代向量输入 $x$。

在线性代数中，更常见的是使用 column vectors，此时线性变换写作：

$$
y = Wx
\tag{2}
$$

其中 row-major 的 $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$，column-vector 的 $x \in \mathbb{R}^{d_\text{in}}$。在这种设定下，如果要批量化输入，batch 维度必须放在 $x$ 的最后，因此 $x$ 需要被替换为矩阵 $\tilde{X} \in \mathbb{R}^{d_\text{in} \times \text{batch}}$。

在本作业中，我们的数学记号将主要使用 column vectors，因为数学通常遵循这种记号。你应该记住：如果想使用普通矩阵乘法记号，由于 PyTorch 使用 row-major memory ordering，你必须像等式 1 中 row-vector 约定那样，用转置来应用矩阵。如果你使用 `einsum` 执行线性代数操作，只要正确标注各轴，这应该不成问题。顺便说一句，Matlab、Julia 和 Fortran 等其他语言或线性代数包都使用 column-major memory ordering，这意味着 batching dimensions 放在最后；但 Python 及其相关包采用了 C 标准的 row-major ordering。

## 3.3 基本构建块：Linear 和 Embedding 模块

### 3.3.1 参数初始化

有效训练神经网络往往需要仔细初始化模型参数；糟糕的初始化可能导致不良行为，例如梯度消失或梯度爆炸。Pre-norm transformers 对初始化异常稳健，但初始化仍然会显著影响训练速度和收敛性。由于这个作业已经很长，我们会把细节留到 assignment 3；这里先给出一些近似初始化，大多数情况下应该工作良好。目前请使用：

- Linear weights：$\mathcal{N}(\mu = 0, \sigma^2 = \frac{2}{d_\text{in}+d_\text{out}})$，截断到 $[-3\sigma, 3\sigma]$。
- Embedding：$\mathcal{N}(\mu = 0, \sigma^2 = 1)$，截断到 $[-3, 3]$。
- RMSNorm：$\mathbf{1}$。

你应该使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

### 3.3.2 Linear Module

Linear layers 是 Transformer 和一般神经网络的基本构建块。首先，你将实现自己的 `Linear` class，它继承自 `torch.nn.Module` 并执行线性变换：

$$
y = Wx
\tag{3}
$$

注意，遵循大多数现代 LLM 的做法，我们不包含 bias term。

### Problem (linear): 实现 linear module（1 分）

**Deliverable：** 实现一个继承自 `torch.nn.Module` 的 `Linear` class，用于执行线性变换。你的实现应该遵循 PyTorch 内置 `nn.Linear` module 的接口，区别是没有 bias argument 或 parameter。我们推荐以下接口：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
```

构造一个 linear transformation module。这个函数应该接收以下参数：

- `in_features: int`：输入的最后一个维度。
- `out_features: int`：输出的最后一个维度。
- `device: torch.device | None = None`：存放参数的设备。
- `dtype: torch.dtype | None = None`：参数的数据类型。

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

对输入应用线性变换。

确保做到：

- 继承 `nn.Module`。
- 调用父类构造函数。
- 将你的参数构造并存储为 $W$，而不是 $W^\top$，并把它放入 `nn.Parameter`。
- 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`。

初始化时，使用上面的设置，并用 `torch.nn.init.trunc_normal_` 初始化权重。

为了测试你的 `Linear` module，请在 `[adapters.run_linear]` 中实现 test adapter。adapter 应该把给定权重加载到你的 `Linear` module 中。你可以为此使用 `Module.load_state_dict`。然后运行：

```bash
uv run pytest -k test_linear
```

### 3.3.3 Embedding Module

如上所述，Transformer 的第一层是 embedding layer，它把整数 token ID 映射到维度为 `d_model` 的向量空间。我们将实现一个自定义的 `Embedding` class，继承自 `torch.nn.Module`，因此你不应该使用 `nn.Embedding`。`forward` 方法应该使用形状为 `(batch_size, sequence_length)` 的 token ID `torch.LongTensor`，索引形状为 `(vocab_size, d_model)` 的 embedding matrix，为每个 token ID 选择 embedding vector。

### Problem (embedding): 实现 embedding module（1 分）

**Deliverable：** 实现继承自 `torch.nn.Module` 的 `Embedding` class，并执行 embedding lookup。你的实现应该遵循 PyTorch 内置 `nn.Embedding` module 的接口。我们推荐以下接口：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```

构造一个 embedding module。这个函数应该接收以下参数：

- `num_embeddings: int`：词表大小。
- `embedding_dim: int`：embedding vectors 的维度，即 $d_\text{model}$。
- `device: torch.device | None = None`：存放参数的设备。
- `dtype: torch.dtype | None = None`：参数的数据类型。

```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

查找给定 token IDs 的 embedding vectors。

确保做到：

- 继承 `nn.Module`。
- 调用父类构造函数。
- 将你的 embedding matrix 初始化为 `nn.Parameter`。
- 存储 embedding matrix 时，让 `d_model` 成为最后一个维度。
- 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`。

同样，使用上面的初始化设置，并用 `torch.nn.init.trunc_normal_` 初始化权重。

为了测试你的实现，请实现 test adapter `[adapters.run_embedding]`。然后运行：

```bash
uv run pytest -k test_embedding
```

## 3.4 Pre-Norm Transformer Block

每个 Transformer block 有两个 sub-layers：multi-head self-attention mechanism 和 position-wise feed-forward network，参见 [A. Vaswani et al., 2017] 第 3.1 节。

在原始 Transformer 论文中，模型在两个 sub-layers 周围分别使用 residual connection，然后进行 layer normalization。这个架构通常称为 “post-norm” Transformer，因为 layer normalization 应用于 sub-layer 的输出。然而，多项工作发现，把 layer normalization 从每个 sub-layer 的输出移动到每个 sub-layer 的输入，并在最终 Transformer block 之后额外添加一个 layer normalization，可以改善 Transformer 的训练稳定性 [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]。图 2 给出了这个 “pre-norm” Transformer block 的可视化表示。每个 Transformer block sub-layer 的输出随后会通过 residual connection 加到 sub-layer 输入上，参见 A. Vaswani et al. [8] 第 5.4 节。对 pre-norm 的一种直觉是：从 input embeddings 到 Transformer 最终输出之间存在一条干净的、不经过任何 normalization 的 “residual stream”，据称这会改善梯度流动。这个 pre-norm Transformer 现在已经成为今天语言模型使用的标准，例如 GPT-3、LLaMA、PaLM 等，因此我们将实现这个变体。我们会依次介绍并实现 pre-norm Transformer block 的各个组件。

### 3.4.1 Root Mean Square Layer Normalization

A. Vaswani et al. [8] 的原始 Transformer 实现使用 layer normalization [J. L. Ba et al., 2016] 来归一化激活。跟随 H. Touvron et al. [12]，我们会使用 root mean square layer normalization，也就是 RMSNorm，参见 B. Zhang et al. [13] 的等式 4。给定一个激活向量 $a \in \mathbb{R}^{d_\text{model}}$，RMSNorm 会按如下方式重新缩放每个激活 $a_i$：

$$
\operatorname{RMSNorm}(a_i) = \frac{a_i}{\operatorname{RMS}(a)} g_i
\tag{4}
$$

其中：

$$
\operatorname{RMS}(a) =
\sqrt{
\frac{1}{d_\text{model}}
\sum_{i=1}^{d_\text{model}} a_i^2
+ \epsilon
}
$$

这里 $g_i$ 是可学习的 “gain” parameter，总共有 `d_model` 个这样的参数；$\epsilon$ 是一个超参数，通常固定为 `1e-5`。

你应该把输入 upcast 到 `torch.float32`，以防止对输入平方时发生 overflow。总体而言，你的 `forward` 方法应该类似如下：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# Your code here performing RMSNorm
...
result = ...
# Return the result in the original dtype
return result.to(in_dtype)
```

### Problem (rmsnorm): Root Mean Square Layer Normalization（1 分）

**Deliverable：** 将 RMSNorm 实现为一个 `torch.nn.Module`。我们推荐以下接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

构造 RMSNorm module。这个函数应该接收以下参数：

- `d_model: int`：模型的 hidden dimension。
- `eps: float = 1e-5`：用于数值稳定性的 epsilon value。
- `device: torch.device | None = None`：存放参数的设备。
- `dtype: torch.dtype | None = None`：参数的数据类型。

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，并返回同样形状的张量。

**Note：** 记得在执行 normalization 之前把输入 upcast 到 `torch.float32`，之后再 downcast 回原始 dtype，如上所述。

为了测试你的实现，请在 `[adapters.run_rmsnorm]` 中实现 test adapter。然后运行：

```bash
uv run pytest -k test_rmsnorm
```

### 3.4.2 Position-Wise Feed-Forward Network

**图 3：比较 SiLU，也称 Swish，和 ReLU activation functions。**

图中的曲线包括：

- `SiLU: f(x) = x σ(x)`
- `Identity: f(x) = x`
- `ReLU: f(x) = max(0, x)`

在原始 Transformer 论文中，参见 A. Vaswani et al. [8] 第 3.3 节，Transformer feed-forward network 由两个线性变换组成，中间有一个 ReLU activation，$\operatorname{ReLU}(x) = \max(0, x)$。在这个原始架构中，内部 feed-forward layer 的维度通常是输入维度的 4 倍。

然而，与这个原始设计相比，现代语言模型倾向于采用两个主要改动：使用另一种 activation function，并引入 gating mechanism。具体来说，我们将实现 “SwiGLU” activation function；它被 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024] 等 LLM 采用，结合了 SiLU，也常称为 Swish，activation 和一种称为 Gated Linear Unit，GLU，的 gating mechanism。跟随 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 以来的大多数现代 LLM，我们还会省略 linear layers 中有时使用的 bias terms。

SiLU 或 Swish activation function [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义如下：

$$
\operatorname{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
\tag{5}
$$

如图 3 所示，SiLU activation function 与 ReLU activation function 类似，但在 0 附近平滑。

Gated Linear Units，GLUs，最初由 Y. N. Dauphin et al. [19] 定义为：一个通过 sigmoid function 的线性变换与另一个线性变换之间的 element-wise product：

$$
\operatorname{GLU}(x, W_1, W_2) =
\sigma(W_1 x) \odot W_2 x
\tag{6}
$$

其中 $\odot$ 表示 element-wise multiplication。Gated Linear Units 被认为可以“通过为梯度提供一条线性路径，同时保留非线性能力，来减少深层架构中的 vanishing gradient problem”。

把 SiLU/Swish 和 GLU 组合起来，就得到 SwiGLU；我们将在 feed-forward networks 中使用它：

$$
\operatorname{FFN}(x)
= \operatorname{SwiGLU}(x, W_1, W_2, W_3)
= W_2\left(\operatorname{SiLU}(W_1 x) \odot W_3 x\right)
\tag{7}
$$

其中 $x \in \mathbb{R}^{d_\text{model}}$，$W_1, W_3 \in \mathbb{R}^{d_\text{ff} \times d_\text{model}}$，$W_2 \in \mathbb{R}^{d_\text{model} \times d_\text{ff}}$。按照规范，$d_\text{ff} = \frac{8}{3}d_\text{model}$。在具体实现中，为了硬件效率，可以把它舍入到附近的 64 的倍数。

N. Shazeer [20] 最早提出把 SiLU/Swish activation 与 GLUs 结合，并通过实验证明 SwiGLU 在语言建模任务上优于 ReLU 和不带 gating 的 SiLU 等 baseline。作业后面你会比较 SwiGLU 和 SiLU。虽然我们已经提到一些关于这些组件的启发式论证，而且相关论文提供了更多支持证据，但保持经验主义视角是有益的：Shazeer 论文中有一句现在很有名的话：

> “我们无法解释为什么这些架构似乎有效；和其他一切一样，我们把它们的成功归因于神圣的仁慈。”

### Problem (positionwise_feedforward): 实现 position-wise feed-forward network（2 分）

**Deliverable：** 实现 SwiGLU feed-forward network，它由 SiLU activation function 和 GLU 组成。

**Note：** 在这个特定情况下，为了数值稳定性，你可以在实现中自由使用 `torch.sigmoid`。

你应该在实现中把 $d_\text{ff}$ 设置为大约 $\frac{8}{3} \times d_\text{model}$，同时确保内部 feed-forward layer 的维度是 64 的倍数，以便充分利用硬件。为了用我们提供的测试检查你的实现，你需要实现 test adapter `[adapters.run_swiglu]`。然后运行：

```bash
uv run pytest -k test_swiglu
```

### 3.4.3 Relative Positional Embeddings

为了向模型注入位置信息，我们将实现 Rotary Position Embeddings [J. Su et al., 2021]，通常称为 RoPE。对于一个在 token 位置 $i$ 的 query token $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^{d}$，我们会应用一个成对旋转矩阵 $R^i$，得到 $q'^{(i)} = R^i q^{(i)} = R^i W_q x^{(i)}$。这里，$R^i$ 会把 embedding 元素对 $q^{(i)}_{2k-1:2k}$ 当作二维向量，按角度 $\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$ 旋转，其中 $k \in \{1, \ldots, d/2\}$，$\Theta$ 是某个常数。因此，我们可以把 $R^i$ 看成大小为 $d \times d$ 的 block-diagonal matrix，其中包含 blocks $R^i_k$，$k \in \{1, \ldots, d/2\}$，并且：

$$
R^i_k =
\begin{pmatrix}
\cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\
\sin(\theta_{i,k}) & \cos(\theta_{i,k})
\end{pmatrix}
\tag{8}
$$

于是得到完整的旋转矩阵：

$$
R^i =
\begin{pmatrix}
R^i_1 & 0 & 0 & \cdots & 0 \\
0 & R^i_2 & 0 & \cdots & 0 \\
0 & 0 & R^i_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & R^i_{d/2}
\end{pmatrix}
\tag{9}
$$

其中 0 表示 $2 \times 2$ 的零矩阵。虽然可以构造完整的 $d \times d$ 矩阵，但一个好的解法应该利用这个矩阵的性质，更高效地实现这个变换。由于我们只关心给定序列中 token 的相对旋转，我们可以在不同层和不同 batch 之间复用为 $\cos(\theta_{i,k})$ 和 $\sin(\theta_{i,k})$ 计算出的值。如果你想优化，可以使用一个被所有层引用的 RoPE module；它可以在 `init` 中用 `self.register_buffer(persistent=False)` 创建一个二维预计算的 sin 和 cos buffer，而不是用 `nn.Parameter`，因为我们不希望学习这些固定的 cosine 和 sine 值。我们对 $q^{(i)}$ 做的完全相同的旋转过程，也会应用于 $k^{(j)}$，并用对应的 $R^j$ 旋转。注意，这一层没有 learnable parameters。

### Problem (rope): 实现 RoPE（2 分）

**Deliverable：** 实现一个 `RotaryPositionalEmbedding` class，把 RoPE 应用于输入张量。

推荐以下接口：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

构造 RoPE module，并在需要时创建 buffers。

- `theta: float`：RoPE 的 $\Theta$ 值。
- `d_k: int`：query 和 key vectors 的维度。
- `max_seq_len: int`：会输入的最大 sequence length。
- `device: torch.device | None = None`：存放 buffer 的设备。

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

处理形状为 `(..., seq_len, d_k)` 的输入张量，并返回同样形状的张量。注意，你应该容忍 $x$ 带有任意数量的 batch dimensions。你应该假设 token positions 是一个形状为 `(..., seq_len)` 的张量，指定 $x$ 沿 sequence dimension 的 token positions。

你应该使用 token positions 沿 sequence dimension 切片你的可能预计算过的 cos 和 sin tensors。

为了测试你的实现，请完成 `[adapters.run_rope]`，并确保它通过：

```bash
uv run pytest -k test_rope
```

### 3.4.4 Scaled Dot-Product Attention

现在我们将实现 A. Vaswani et al. [8] 第 3.2.1 节描述的 scaled dot-product attention。作为预备步骤，Attention 操作的定义会使用 softmax；softmax 是一个把未归一化分数向量转换为归一化分布的操作：

$$
\operatorname{softmax}(v)_i =
\frac{\exp(v_i)}
{\sum_{j=1}^{n}\exp(v_j)}
\tag{10}
$$

注意，对较大的值，$\exp(v_i)$ 可能变成 `inf`，此时 `inf / inf = NaN`。我们可以利用 softmax 操作对所有输入加上任意常数 $c$ 不变这一性质来避免这个问题。通常，我们会从 $v$ 的所有元素中减去 $v$ 的最大项，使新的最大项为 0。你现在将使用这个技巧来实现 softmax，以获得数值稳定性。

### Problem (softmax): 实现 softmax（1 分）

**Deliverable：** 写一个函数，对张量应用 softmax 操作。你的函数应该接收两个参数：一个张量和一个维度 $i$，并对输入张量的第 $i$ 个维度应用 softmax。输出张量应该与输入张量形状相同，但第 $i$ 个维度现在会包含一个归一化概率分布。使用从第 $i$ 个维度的所有元素中减去该维度最大值的技巧，以避免数值稳定性问题。

为了测试你的实现，请完成 `[adapters.run_softmax]`，并确保它通过：

```bash
uv run pytest -k test_softmax_matches_pytorch
```

现在，我们可以用数学方式定义 Attention 操作：

$$
\operatorname{Attention}(Q, K, V)
= \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\tag{11}
$$

其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$。这里 $Q$、$K$ 和 $V$ 都是这个操作的输入；注意它们不是 learnable parameters。

**Masking：** 有时 mask attention operation 的输出会很方便。mask 的形状应该为 $M \in \{\text{True}, \text{False}\}^{n \times m}$，这个 boolean matrix 的每一行 $i$ 表示 query $i$ 应该 attend 到哪些 keys。按照惯例，并且稍微有点令人困惑，位置 $(i, j)$ 上的 `True` 表示 query $i$ 会 attend to key $j$；`False` 表示 query 不会 attend to key。换句话说，在值为 `True` 的 $(i, j)$ pair 处，“信息会流动”。例如，考虑一个 entries 为 `[[True, True, False]]` 的 $1 \times 3$ mask matrix。这个单独的 query vector 只会 attend 到前两个 keys。

从计算上说，使用 masking 比在 subsequences 上计算 attention 高效得多。我们可以通过取 pre-softmax values，也就是 $QK^\top/\sqrt{d_k}$，并向 mask matrix 中任何值为 `False` 的 entry 加上 $-\infty$ 来做到这一点。

### Problem (scaled_dot_product_attention): 实现 scaled dot-product attention（5 分）

**Deliverable：** 实现 scaled dot-product attention function。你的实现应该处理形状为 `(batch_size, ..., seq_len, d_k)` 的 keys 和 queries，以及形状为 `(batch_size, ..., seq_len, d_v)` 的 values，其中 `...` 表示任意数量的其他 batch-like dimensions，如果存在的话。实现应该返回形状为 `(batch_size, ..., seq_len, d_v)` 的输出。关于 batch-like dimensions 的讨论见第 3.2 节。

你的实现还应该支持一个用户可选提供的 boolean mask，形状为 `(seq_len, seq_len)`。mask 值为 `True` 的位置的 attention probabilities 应该合计为 1，mask 值为 `False` 的位置的 attention probabilities 应该为 0。

为了对照我们提供的测试检查你的实现，你需要在 `[adapters.run_scaled_dot_product_attention]` 中实现 test adapter。下面的命令会在三阶输入张量上测试你的实现：

```bash
uv run pytest -k test_scaled_dot_product_attention
```

下面的命令会在四阶输入张量上测试你的实现：

```bash
uv run pytest -k test_4d_scaled_dot_product_attention
```

### 3.4.5 Causal Multi-Head Self-Attention

我们将实现 A. Vaswani et al. [8] 第 3.2.2 节描述的 multi-head self-attention。回忆一下，从数学上说，应用 multi-head attention 的操作定义如下：

$$
\operatorname{MultiHead}(Q, K, V)
= \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)
\tag{12}
$$

其中：

$$
\text{head}_i =
\operatorname{Attention}(Q_i, K_i, V_i)
\tag{13}
$$

$Q_i$、$K_i$、$V_i$ 分别是 $Q$、$K$ 和 $V$ 的 embedding dimension 上第 $i \in \{1, \ldots, h\}$ 个大小为 $d_k$ 或 $d_v$ 的 slice。这里的 Attention 是第 3.4.4 节定义的 scaled dot-product attention 操作。由此我们可以形成 multi-head self-attention 操作：

$$
\operatorname{MultiHeadSelfAttention}(x)
= W_O \operatorname{MultiHead}(W_Qx, W_Kx, W_Vx)
\tag{14}
$$

这里，learnable parameters 为 $W_Q \in \mathbb{R}^{hd_k \times d_\text{model}}$，$W_K \in \mathbb{R}^{hd_k \times d_\text{model}}$，$W_V \in \mathbb{R}^{hd_v \times d_\text{model}}$，以及 $W_O \in \mathbb{R}^{d_\text{model} \times hd_v}$。由于 $Q$、$K$ 和 $V$ 会在 multi-head attention 操作中被切片，我们可以把 $W_Q$、$W_K$ 和 $W_V$ 看作在 output dimension 上为每个 head 分开。当你实现好以后，key、value 和 query projections 总共应该由三次 matrix multiplies 计算出来。[^5]

#### Causal masking

你的实现应该防止模型 attend 到序列中的未来 token。换句话说，如果模型收到 token 序列 $t_1, \ldots, t_n$，而我们想为前缀 $t_1, \ldots, t_i$，其中 $i < n$，计算 next-word predictions，那么模型不应该能够访问，也就是 attend to，位置 $t_{i+1}, \ldots, t_n$ 的 token representations；因为它在推理生成文本时无法访问这些 tokens，而且这些未来 tokens 会泄露真实下一个词身份的信息，使语言建模预训练目标变得平凡。对于输入 token 序列 $t_1, \ldots, t_n$，我们可以朴素地通过运行 $n$ 次 multi-head self-attention 来阻止访问未来 tokens，其中每次对应序列中的一个唯一 prefix。相反，我们会使用 causal attention masking，它允许 token $i$ attend 到序列中的所有位置 $j \le i$。你可以使用 `torch.triu` 或 broadcasted index comparison 构造这个 mask，并且应该利用第 3.4.4 节中 scaled dot-product attention 实现已经支持 attention masking 这一点。

#### Applying RoPE

RoPE 应该应用于 query 和 key vectors，但不应用于 value vectors。此外，head dimension 应该作为 batch dimension 来处理，因为在 multi-head attention 中，attention 会对每个 head 独立应用。这意味着应该对每个 head 的 query 和 key vectors 应用完全相同的 RoPE rotation。

### Problem (multihead_self_attention): 实现 causal multi-head self-attention（5 分）

**Deliverable：** 将 causal multi-head self-attention 实现为一个 `torch.nn.Module`。你的实现至少应该接收以下参数：

- `d_model: int`：Transformer block inputs 的维度。
- `num_heads: int`：multi-head self-attention 中使用的 heads 数。

按照 A. Vaswani et al. [8]，设置 $d_k = d_v = \frac{d_\text{model}}{h}$。为了对照我们提供的测试检查你的实现，请实现 test adapter `[adapters.run_multihead_self_attention]`。然后运行：

```bash
uv run pytest -k test_multihead_self_attention
```

## 3.5 完整的 Transformer LM

我们先组装 Transformer block，回看图 2 会有帮助。一个 Transformer block 包含两个 “sub-layers”：一个用于 multihead self attention，另一个用于 SwiGLU feed-forward network。在每个 sub-layer 中，我们先执行 RMSNorm，然后执行主要操作，也就是 MHA/FF，最后加入 residual connection。

具体来说，Transformer block 的前半部分，也就是第一个 “sub-layer”，应该实现以下更新，从输入 $x$ 产生输出 $y$：

$$
y = x + \operatorname{MultiHeadSelfAttention}(\operatorname{RMSNorm}(x))
\tag{15}
$$

### Problem (transformer_block): 实现 Transformer block（3 分）

按照第 3.4 节描述并在图 2 中展示的方式，实现 pre-norm Transformer block。你的 Transformer block 至少应该接收以下参数。

- `d_model: int`：Transformer block inputs 的维度。
- `num_heads: int`：multi-head self-attention 中使用的 heads 数。
- `d_ff: int`：position-wise feed-forward inner layer 的维度。

为了测试你的实现，请实现 adapter `[adapters.run_transformer_block]`。然后运行：

```bash
uv run pytest -k test_transformer_block
```

**Deliverable：** 通过提供测试的 Transformer block code。

现在我们按照图 1 的高层图示把 blocks 组合起来。遵循第 3.1 中关于 embedding 的描述，把它送入 `num_layers` 个 Transformer blocks，然后传入最终 layer norm 和 LM head，得到词表上的未归一化分布，也就是 logits。

### Problem (transformer_lm): 实现 Transformer LM（3 分）

现在是把所有东西组合起来的时候。按照第 3.1 节描述并在图 1 中展示的方式，实现 Transformer language model。至少，你的实现应该接收前面所有 Transformer block 的构造参数，以及以下额外参数：

- `vocab_size: int`：词表大小，用于确定 token embedding matrix 的维度。
- `context_length: int`：最大 context length，用于确定 RoPE sin 和 cos buffer 的维度。
- `num_layers: int`：使用的 Transformer blocks 数量。

为了对照我们提供的测试检查你的实现，你首先需要实现 test adapter `[adapters.run_transformer_lm]`。然后运行：

```bash
uv run pytest -k test_transformer_lm
```

**Deliverable：** 一个通过上述测试的 Transformer LM module。

### Resource accounting

理解 Transformer 各部分如何消耗计算和内存是很有用的。我们会逐步做一些基本的 “FLOPs accounting”。Transformer 中绝大多数 FLOPs 都来自 matrix multiplies，因此我们的核心方法很简单：

1. 写出 Transformer forward pass 中所有的 matrix multiplies。
2. 把每个 matrix multiply 转换为所需 FLOPs。

对于第二步，下面这个事实会很有用：

**Rule：** 给定 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$，matrix-matrix product $AB$ 需要 $2mnp$ FLOPs。

要理解这一点，注意 $(AB)[i, j] = A[i, :] \cdot B[:, j]$，这个 dot product 需要 $n$ 次加法和 $n$ 次乘法，即 $2n$ FLOPs。由于 matrix-matrix product $AB$ 有 $m \times p$ 个 entries，因此 FLOPs 总数为 $(2n)(mp) = 2mnp$。

现在，在做下一个 problem 之前，逐一检查你的 Transformer block 和 Transformer LM 的每个组件，列出所有 matrix multiplies 及其对应的 FLOPs cost，会很有帮助。

### Problem (transformer_accounting): Transformer LM resource accounting（5 分）

**(a)** 考虑一个使用我们作业架构的 GPT-2 XL 大小的模型，它有以下配置：

- `vocab_size`: 50,257
- `context_length`: 1,024
- `num_layers`: 48
- `d_model`: 1,600
- `num_heads`: 25
- `d_ff`: 4,288，即最接近 $\frac{8}{3} \times 1,600$ 的 64 的倍数。

假设我们用这个配置构建模型。这个模型会有多少 trainable parameters？假设每个参数都用 single-precision floating point 表示，仅加载这个模型需要多少内存？

**Deliverable：** 一到两句话的回答。

**解答：** 本作业架构没有 bias，且 token embedding 和 LM head 不共享权重，因此参数量为

```text
2 * vocab_size * d_model
+ num_layers * (4 * d_model^2 + 3 * d_model * d_ff + 2 * d_model)
+ d_model
```

代入 GPT-2 XL-shaped 配置得到 `1,640,452,800` 个 trainable parameters，约 `1.64B`。每个参数使用 single-precision floating point，即 4 bytes，仅加载参数需要 `6,561,811,200` bytes，约 `6.56 GB`，也就是约 `6.11 GiB`。

**(b)** 找出完成一次 GPT-2 XL-shaped model forward pass 所需的 matrix multiplies。这些 matrix multiplies 总共需要多少 FLOPs？假设我们的输入序列有 `context_length` 个 tokens。

**Deliverable：** 一个 matrix multiplies 列表，包含描述，以及所需 FLOPs 总数。

**解答：** 只统计 matrix multiplies，并假设 batch size 为 1。记 `V = vocab_size`、`n = context_length`、`L = num_layers`、`d = d_model`、`f = d_ff`。GPT-2 XL-shaped 配置为 `V=50,257, n=1,024, L=48, d=1,600, f=4,288`。

| 组件 | FLOPs 公式 | GPT-2 XL-shaped FLOPs |
| --- | ---: | ---: |
| Q/K/V projections | `L * 3 * 2 * n * d^2` | `754,974,720,000` |
| attention scores `QK^T` | `L * 2 * n^2 * d` | `161,061,273,600` |
| attention values `Attn * V` | `L * 2 * n^2 * d` | `161,061,273,600` |
| attention output projection | `L * 2 * n * d^2` | `251,658,240,000` |
| SwiGLU FFN `W1/W2/W3` | `L * 6 * n * d * f` | `2,023,332,249,600` |
| LM head | `2 * n * d * V` | `164,682,137,600` |
| Total | sum | `3,516,769,894,400` |

因此一次满长 `1024` tokens 的 GPT-2 XL-shaped forward pass 约需要 `3.52e12` FLOPs。

**(c)** 基于上面的分析，模型哪些部分需要最多 FLOPs？

**Deliverable：** 一到两句话的回答。

**解答：** 最大的 FLOPs 来源是 SwiGLU FFN，约占总 FLOPs 的 `57.53%`。其次是 attention 的 Q/K/V projections，约占 `21.47%`；在 `context_length=1024` 时，attention scores 和 attention values 这两个 `n^2` 项各自只占约 `4.58%`。

**(d)** 使用 GPT-2 small、GPT-2 medium 和 GPT-2 large 重复你的分析。其中 GPT-2 small 为 12 layers、768 `d_model`、12 heads；GPT-2 medium 为 24 layers、1024 `d_model`、16 heads；GPT-2 large 为 36 layers、1280 `d_model`、20 heads。随着模型大小增加，Transformer LM 的哪些部分在总 FLOPs 中所占比例增加或减少？

**Deliverable：** 对每个模型，给出 model components 及其相关 FLOPs 的 breakdown，表示为总 forward pass FLOPs 的比例。此外，用一到两句话描述模型大小变化如何改变各组件 FLOPs 的占比。

**解答：** 这里沿用作业规则，将 `d_ff` 取为最接近 `(8 / 3) * d_model` 的 64 的倍数。因此 small/medium/large 的 `d_ff` 分别是 `2048/2752/3392`，context length 都取 `1024`。

| 模型 | 总 FLOPs | QKV proj | score | value | output proj | FFN | LM head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 small | `291,648,307,200` | `14.91%` | `6.63%` | `6.63%` | `4.97%` | `39.76%` | `27.10%` |
| GPT-2 medium | `830,172,299,264` | `18.62%` | `6.21%` | `6.21%` | `6.21%` | `50.05%` | `12.70%` |
| GPT-2 large | `1,768,530,903,040` | `20.49%` | `5.46%` | `5.46%` | `6.83%` | `54.30%` | `7.45%` |
| GPT-2 XL | `3,516,769,894,400` | `21.47%` | `4.58%` | `4.58%` | `7.16%` | `57.53%` | `4.68%` |

随着模型变宽变深，FFN 和 projection 的占比上升，因为它们主要随 `d_model^2` 或 `d_model * d_ff` 增长。LM head 只随 `d_model * vocab_size` 线性增长，因此占比快速下降；在 context length 固定为 1024 时，attention 的二次项占比也整体下降。

**(e)** 取 GPT-2 XL，并把 context length 增加到 16,384。一次 forward pass 的总 FLOPs 会如何变化？模型组件 FLOPs 的相对贡献会如何变化？

**Deliverable：** 一到两句话的回答。

**解答：** 将 GPT-2 XL 的 context length 从 `1024` 增加到 `16,384` 后，总 FLOPs 变为 `133,577,729,638,400`，约 `1.34e14` FLOPs，是原来的约 `37.98` 倍。此时 attention scores 和 attention values 各占约 `30.87%`，合计约 `61.73%`，成为主要计算成本；原因是这两项随 `n^2` 增长，而 projection、FFN 和 LM head 只随 `n` 线性增长。

[^4]: 值得注意的是，虽然 `einops` 有大量支持，`einx` 并没有经过同等程度的实战检验。如果你发现 `einx` 有任何限制或 bug，可以随时退回到 `einops` 配合一些普通 PyTorch 的写法。

[^5]: 作为一个 stretch goal，可以尝试把 key、query 和 value projections 合并到一个单一 weight matrix 中，这样只需要一次 matrix multiply。
