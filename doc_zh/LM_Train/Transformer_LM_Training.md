# 4 Training a Transformer LM

本文是 `cs336_assignment1_basics.pdf` 中 **4 Training a Transformer LM** 部分的中文转换版，范围从第 4 节开头到 `4.5 Gradient clipping` 结束。原文中的公式、problem、deliverable、脚注和关键实现要求均保留；额外补充的解释会以“补充解释”标出，帮助理解优化器、学习率调度和梯度裁剪。

## 4 Training a Transformer LM

现在我们已经有了预处理数据的步骤，也就是 tokenizer；也有了模型，也就是 Transformer。剩下的工作是构建所有支持训练的代码。这包括以下内容：

- **Loss**：需要定义损失函数，也就是 cross-entropy。
- **Optimizer**：需要定义优化器来最小化这个损失，这里是 AdamW。
- **Training loop**：需要所有支撑训练的基础设施，包括加载数据、保存 checkpoints，以及管理训练过程。

## 4.1 Cross-entropy loss

回忆一下，Transformer 语言模型会对每个长度为 `m + 1` 的序列 `x`，以及每个 `i = 1, ..., m`，定义一个分布：

```text
p_theta(x_{i+1} | x_{1:i})
```

给定一个训练集 `D`，其中包含长度为 `m + 1` 的序列，我们定义标准的 cross-entropy，也就是 negative log-likelihood loss：

```math
\ell(\theta; D)
= \frac{1}{|D|} \sum_{x \in D} \frac{1}{m} \sum_{i=1}^{m}
-\log p_\theta(x_{i+1} \mid x_{1:i}).
\tag{16}
```

注意，Transformer 的一次 forward pass 会同时给出所有 `i = 1, ..., m` 的：

```text
p_theta(x_{i+1} | x_{1:i})
```

具体来说，Transformer 会在每个位置 `i` 计算 logits：

```math
o_i \in \mathbb{R}^{\text{vocab_size}}
```

于是有：[^6]

```math
p(x_{i+1} \mid x_{1:i})
= \operatorname{softmax}(o_i)[x_{i+1}]
= \frac{\exp(o_i[x_{i+1}])}
{\sum_{a=1}^{\text{vocab_size}} \exp(o_i[a])}.
\tag{17}
```

Cross-entropy loss 通常是相对于 logits 向量

```math
o_i \in \mathbb{R}^{\text{vocab_size}}
```

和 target `x_{i+1}` 定义的。[^7]

和 softmax 一样，实现 cross-entropy loss 时需要小心数值问题。

### Problem (cross_entropy): Implement cross-entropy（1 分）

**Deliverable：** 写一个函数来计算 cross-entropy loss。这个函数接收预测 logits `o_i` 和 targets `x_{i+1}`，并计算：

```math
\ell_i = -\log \operatorname{softmax}(o_i)[x_{i+1}].
```

你的函数应该处理以下事项：

- 为了数值稳定性，减去最大元素。
- 尽可能消去 `log` 和 `exp`。
- 处理任意额外的 batch 维度，并返回 batch 上的平均值。和第 3.2 节一样，我们假设 batch-like 维度总是放在前面，位于 vocabulary size 维度之前。

实现 `[adapters.run_cross_entropy]`，然后运行：

```bash
uv run pytest -k test_cross_entropy
```

来测试你的实现。

**实现说明：** 代码实现见 `cs336_basics/training.py` 中的 `cross_entropy`，adapter 见 `tests/adapters.py` 中的 `run_cross_entropy`。实现没有先显式计算 softmax，而是使用稳定的 log-sum-exp 形式：

```text
CE(logits, target) = logsumexp(logits) - logits[target]
```

其中 `logsumexp` 通过先减去最大 logit 来避免 `exp` 溢出。`torch.gather` 用来取出每个样本 target class 对应的 logit，因此该实现可以处理任意 batch-like 前缀维度，只要求 vocabulary 维度在最后。

### 补充解释：为什么 cross-entropy 要用 logits 直接算？

直接按定义写：

```text
-log(softmax(logits)[target])
```

数学上没错，但数值上容易出问题。因为 softmax 里有 `exp(logit)`，当 logit 很大时，`exp` 可能上溢为 `inf`。稳定做法是使用恒等变形：

```math
-\log \operatorname{softmax}(o)[y]
= -o[y] + \log \sum_j \exp(o[j]).
```

然后再用 “减最大值” 技巧：

```math
\log \sum_j \exp(o[j])
= m + \log \sum_j \exp(o[j] - m),
\quad m = \max_j o[j].
```

这样 `exp(o[j] - m)` 的最大值就是 `exp(0)=1`，不会因为大正数而溢出。这也是 PyTorch 的 `cross_entropy` 和 `logsumexp` 背后的核心思路。

## Perplexity

Cross-entropy 足够用于训练，但在评估模型时，我们还希望报告 perplexity。对于一个长度为 `m` 的序列，如果我们遭受的 cross-entropy losses 是：

```text
ell_1, ..., ell_m
```

那么 perplexity 定义为：

```math
\operatorname{perplexity}
= \exp\left(\frac{1}{m} \sum_{i=1}^{m} \ell_i\right).
\tag{18}
```

### 补充解释：perplexity 怎么理解？

Perplexity 可以粗略理解为“模型在每一步平均有多少个同样可能的选择”。如果平均 cross-entropy 是 `log(10)`，那么 perplexity 就是 `10`，表示模型平均像是在 10 个同等可能的 token 中做选择。Perplexity 越低，通常说明模型越确定、预测越好。

## 4.2 The SGD Optimizer

现在我们有了 loss function，接下来开始探索 optimizers。最简单的基于梯度的优化器是 Stochastic Gradient Descent，也就是 SGD。

我们从随机初始化的参数开始：

```math
\theta_0
```

然后对每一步 `t = 0, ..., T - 1`，执行如下更新：

```math
\theta_{t+1}
\leftarrow
\theta_t - \alpha_t \nabla L(\theta_t; B_t),
\tag{19}
```

其中 `B_t` 是从数据集 `D` 中采样出的一个随机 batch，learning rate `alpha_t` 和 batch size `|B_t|` 是 hyperparameters。

### 补充解释：优化器到底在做什么？

训练的目标是让 loss 变小。反向传播会告诉我们每个参数当前应该朝哪个方向变化，也就是梯度：

```text
p.grad
```

优化器负责根据这些梯度真正修改参数：

```text
p.data <- updated value
```

SGD 的想法最直接：沿着负梯度方向走一步。learning rate 决定这一步走多远。太小会学得慢，太大可能越过低点甚至发散。

## 4.2.1 Implementing SGD in PyTorch

为了实现自己的优化器，我们会继承 PyTorch 的：

```python
torch.optim.Optimizer
```

一个 `Optimizer` 子类必须实现两个方法。

```python
def __init__(self, params, ...)
```

这个方法应该初始化优化器。这里的 `params` 是需要优化的参数集合；也可能是 parameter groups，如果用户希望对模型的不同部分使用不同 hyperparameters，例如不同 learning rates。确保把 `params` 传给父类的 `__init__` 方法，父类会存储这些参数，供 `step` 使用。你可以根据优化器需要接收额外参数，例如 learning rate 是很常见的参数，并将它们作为一个字典传给父类构造函数。字典的 keys 是你为这些参数选择的名字，也就是字符串。

```python
def step(self)
```

这个方法应该执行一次参数更新。在 training loop 中，它会在 backward pass 之后被调用，所以你可以访问最后一个 batch 上的 gradients。这个方法应该遍历每个 parameter tensor `p`，并原地修改它们，也就是设置 `p.data`。`p.data` 保存和该 parameter 关联的 tensor，而 `p.grad` 表示 loss 对该 parameter 的梯度，如果它存在的话。

PyTorch optimizer API 有一些细节，因此用例子解释会更容易。为了让例子更丰富，我们会实现 SGD 的一个轻微变体：learning rate 会随着训练衰减，从初始 learning rate `alpha` 开始，随着时间推移逐渐采用更小的步长：

```math
\theta_{t+1}
= \theta_t - \frac{\alpha}{\sqrt{t + 1}}
\nabla L(\theta_t; B_t).
\tag{20}
```

下面是这个版本的 SGD 作为 PyTorch `Optimizer` 的实现方式：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss
```

在 `__init__` 中，我们把参数以及默认 hyperparameters 传给父类构造函数。参数可能以 groups 的形式传入，每组都有不同 hyperparameters。如果参数只是一个由 `torch.nn.Parameter` 对象构成的集合，那么父类构造函数会创建一个单独的 group，并为它分配默认 hyperparameters。

然后在 `step` 中，我们遍历每个 parameter group，再遍历该 group 中的每个 parameter，并应用等式 20。这里，我们把 iteration number 保存在每个 parameter 关联的 state 中：先读取这个值，用它进行梯度更新，然后再更新它。

API 规定用户可能传入一个 callable `closure`，用于在 optimizer step 前重新计算 loss。我们不会在本作业使用的优化器中需要它，但为了遵守 API，需要把它加上。

为了看到这个优化器如何工作，可以使用下面这个最小 training loop 示例：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad()  # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean()  # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward()  # Run backward pass, which computes gradients.
    opt.step()  # Run optimizer step.
```

这就是 training loop 的典型结构：在每次迭代中，我们计算 loss，然后运行一次 optimizer step。训练语言模型时，可学习参数来自模型，在 PyTorch 中可以通过 `m.parameters()` 得到这个集合。loss 会在一个采样出的数据 batch 上计算，但 training loop 的基本结构是一样的。

### 补充解释：`zero_grad()` 为什么必须放在每一步前面？

PyTorch 默认会把梯度累加到 `p.grad` 上，而不是每次 backward 自动清零。这对 gradient accumulation 有用，但普通训练循环里，如果忘记调用：

```python
opt.zero_grad()
```

那么当前 batch 的梯度会和之前 batch 的梯度混在一起，导致实际更新方向错误。因此常见顺序是：

```text
zero_grad -> forward -> loss -> backward -> step
```

### Problem (learning_rate_tuning): Tuning the learning rate（1 分）

正如我们将看到的，learning rate 是最影响训练的 hyperparameters 之一。

让我们在上面的 toy example 中实践这一点。使用另外三个 learning rate 值运行上面的 SGD 示例：

```text
1e1, 1e2, 1e3
```

只运行 10 个 training iterations。对于这些 learning rates，loss 会发生什么？它下降得更快、更慢，还是发散，也就是在训练过程中增加？

**Deliverable：** 用一到两句话回答你观察到的行为。

**观察结果：** 在这个 toy example 中，`1e1` 比 `1` 下降得明显更快；`1e2` 第一步会越过最小点但因为学习率按 `1 / sqrt(t + 1)` 衰减，后续很快收敛到接近 0；`1e3` 会让 loss 迅速增大并发散。这个实验说明 learning rate 太小会慢，适当增大会加速，但过大时更新会不稳定。

## 4.3 AdamW

现代语言模型通常不会用 SGD 训练，而是使用更复杂的 optimizers。最近使用的大多数优化器都是 Adam optimizer [D. P. Kingma et al., 2015] 的派生版本。

我们会使用 AdamW [I. Loshchilov et al., 2019]，它在近期工作中被广泛使用。AdamW 对 Adam 做了一个修改：通过加入 weight decay 改善 regularization。weight decay 的含义是，在每次迭代中，把参数向 0 拉近。AdamW 的关键点是，这个 weight decay 和 gradient update 是 decoupled，也就是解耦的。

我们会按照 I. Loshchilov et al. [23] 的 algorithm 2 来实现 AdamW。

AdamW 是 stateful 的：对每个参数，它会跟踪一阶矩和二阶矩的 running estimate。因此，AdamW 会使用额外内存来换取更好的稳定性和收敛性。

除了 learning rate `alpha`，AdamW 还有一对 hyperparameters：

```math
(\beta_1, \beta_2)
```

它们控制 moment estimates 的更新；还有一个 weight decay rate：

```math
\lambda
```

典型应用会把：

```math
(\beta_1, \beta_2)
```

设为：

```text
(0.9, 0.999)
```

但像 LLaMA [H. Touvron et al., 2023] 和 GPT-3 [T. B. Brown et al., 2020] 这样的大语言模型，通常使用：

```text
(0.9, 0.95)
```

算法可以写成如下形式，其中 `epsilon` 是一个很小的值，例如 `1e-8`，用于在 `v` 中出现极小值时改善数值稳定性：

### Algorithm 1: AdamW Optimizer

```text
1  init(theta)                         # 初始化可学习参数
2  m <- 0                              # 一阶矩向量初值；形状与 theta 相同
3  v <- 0                              # 二阶矩向量初值；形状与 theta 相同
4  for t = 1, ..., T do
5      Sample batch of data B_t
6      g <- grad_theta ell(theta; B_t) # 计算 loss 的梯度
7      alpha_t <- alpha * sqrt(1 - beta_2^t) / (1 - beta_1^t)
                                             # 计算第 t 次迭代调整后的 alpha
8      theta <- theta - alpha * lambda * theta
                                             # 应用 weight decay
9      m <- beta_1 * m + (1 - beta_1) * g
                                             # 更新一阶矩估计
10     v <- beta_2 * v + (1 - beta_2) * g^2
                                             # 更新二阶矩估计
11     theta <- theta - alpha_t * m / (sqrt(v) + epsilon)
                                             # 应用 moment-adjusted weight updates
12 end for
```

注意，`t` 从 1 开始。现在你将实现这个优化器。

### 补充解释：AdamW 的 `m` 和 `v` 是什么？

可以把 AdamW 理解成“带记忆的 SGD”：

- `m` 是梯度的一阶矩，也就是梯度的指数滑动平均。它类似 momentum，让更新方向不要被单个 batch 的噪声剧烈影响。
- `v` 是梯度平方的指数滑动平均。它估计每个参数的梯度尺度，让梯度大的参数步子自动变小，梯度小的参数步子相对变大。
- `beta_1` 和 `beta_2` 控制这两个滑动平均“记多久”。越接近 1，历史影响越长。
- `alpha_t` 中的校正项是 bias correction。因为 `m` 和 `v` 初始化为 0，训练初期它们会偏小，所以要用 `1 - beta_1^t` 和 `1 - beta_2^t` 做修正。

AdamW 和普通 Adam 的关键区别在 weight decay：AdamW 是先按 `theta <- theta - alpha * lambda * theta` 直接衰减参数，再做 Adam 的梯度更新。这样 weight decay 不会混入 moment estimates。

### Problem (adamw): Implement AdamW（2 分）

**Deliverable：** 将 AdamW optimizer 实现为 `torch.optim.Optimizer` 的子类。你的 class 应该在 `__init__` 中接收 learning rate `alpha`，以及 `beta`、`epsilon` 和 `lambda` hyperparameters。

为了帮助你保存 state，基础 `Optimizer` class 提供了一个 dictionary：

```python
self.state
```

它会把 `nn.Parameter` 对象映射到一个 dictionary，后者存储该 parameter 所需的任意信息。对于 AdamW，这些信息就是 moment estimates。

实现 `[adapters.get_adamw_cls]`，并确保它通过：

```bash
uv run pytest -k test_adamw
```

**实现说明：** 代码实现见 `cs336_basics/training.py` 中的 `AdamW`，adapter 见 `tests/adapters.py` 中的 `get_adamw_cls`。实现要点如下：

- 每个 parameter 在 `self.state[parameter]` 中保存 `step`、`exp_avg` 和 `exp_avg_sq`。
- `exp_avg` 对应 AdamW 算法中的一阶矩 `m`，`exp_avg_sq` 对应二阶矩 `v`。
- weight decay 使用 decoupled 形式，先直接执行 `parameter <- parameter - lr * weight_decay * parameter`，不把 weight decay 混入梯度。
- moment update 后使用 bias correction：

```text
step_size = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
```

再执行：

```text
parameter <- parameter - step_size * m / (sqrt(v) + eps)
```

### Problem (adamw_accounting): Resource accounting for training with AdamW（2 分）

现在我们计算运行 AdamW 需要多少内存和计算量。假设我们对每个 tensor 都使用 `float32`。

**(a)** 运行 AdamW 需要多少 peak memory？请根据 parameters、activations、gradients 和 optimizer state 的内存使用来分解你的答案。用 `batch_size` 和模型 hyperparameters 表示你的答案：

```text
vocab_size, context_length, num_layers, d_model, num_heads
```

假设：

```math
d_{\text{ff}} = \frac{8}{3} \times d_{\text{model}}.
```

为了简化，在计算 activations 的内存使用时，只考虑以下组件：

- Transformer block
  - RMSNorm(s)
  - Multi-head self-attention sublayer：`QKV` projections、`QK^T` matrix multiply、softmax、values 的 weighted sum、output projection。
  - Position-wise feed-forward，也就是 SwiGLU：`W_1`、`W_2`、gate branch 上的 SiLU、element-wise product、`W_3`。
- final RMSNorm
- output embedding
- logits 上的 cross-entropy

**Deliverable：** 分别给出 parameters、activations、gradients 和 optimizer state 的代数表达式，以及 total。

**(b)** 对 GPT-2 XL-shaped model 实例化你的答案，得到一个只依赖 `batch_size` 的表达式。在 `80GB` memory 内，最大能使用的 batch size 是多少？

**Deliverable：** 一个形如：

```text
a * batch_size + b
```

的表达式，其中 `a` 和 `b` 是数值；以及一个表示最大 batch size 的数字。

**(c)** 运行 AdamW 的一步需要多少 FLOPs？

**Deliverable：** 一个代数表达式，并附上简短理由。

**(d)** Model FLOPs utilization，简称 MFU，定义为 observed throughput，也就是 tokens per second，相对于硬件理论峰值 FLOP throughput 的比例 [A. Chowdhery et al., 2022]。

一块 NVIDIA H100 GPU 对 “float32” operations 的理论峰值是 `495 teraFLOP/s`。这里的 “float32” 实际上是 TensorFloat-32，而 TensorFloat-32 在现实中是 “bfloat19”。假设你能达到 `50%` MFU，那么在单张 H100 上，用 batch size `1024` 训练 GPT-2 XL `400K` steps 需要多长时间？根据 J. Kaplan et al. [25] 和 J. Hoffmann et al. [26]，假设 backward pass 的 FLOPs 是 forward pass 的两倍。

**Deliverable：** 训练需要的小时数，并附上简短理由。

**解答：** 记：

```text
B = batch_size
V = vocab_size
n = context_length
L = num_layers
d = d_model
h = num_heads
f = d_ff = (8 / 3) * d
```

本作业架构没有 bias，且 token embedding 与 LM head 不共享权重。参数元素个数为：

```text
P = 2 * V * d + L * (4 * d^2 + 3 * d * f + 2 * d) + d
```

代入 `f = (8 / 3) * d` 后：

```text
P = 2 * V * d + L * (12 * d^2 + 2 * d) + d
```

因此各部分内存为：

```text
parameters      = 4 * P bytes
gradients       = 4 * P bytes
optimizer state = 8 * P bytes      # AdamW 的 m 和 v 两份状态
```

activation 元素数按题目指定组件估算。每个 Transformer block 中：

```text
RMSNorms:       2 * B * n * d
MHA:            5 * B * n * d + 2 * B * h * n^2
SwiGLU FFN:     4 * B * n * f + B * n * d
```

所以每层 activation 元素数为：

```text
8 * B * n * d + 4 * B * n * f + 2 * B * h * n^2
```

代入 `f = (8 / 3) * d`：

```text
(56 / 3) * B * n * d + 2 * B * h * n^2
```

再加上 final RMSNorm、output embedding 和 cross-entropy on logits：

```text
activations
= 4 * B * [L * ((56 / 3) * n * d + 2 * h * n^2) + n * d + 2 * n * V] bytes
```

总 peak memory 估算为：

```text
total
= 16 * P
+ 4 * B * [L * ((56 / 3) * n * d + 2 * h * n^2) + n * d + 2 * n * V] bytes
```

对 GPT-2 XL-shaped 配置 `V=50257, n=1024, L=48, d=1600, h=25`，并按本题假设 `f=(8/3)d`，有：

```text
P = 1,635,537,600
non-activation memory = 26.1686016 GB
activation memory     = 16.356614144 GB * B
total memory          = 16.356614144 GB * B + 26.1686016 GB
```

在 `80GB` 内存限制下：

```text
B_max = floor((80 - 26.1686016) / 16.356614144) = 3
```

所以最大 batch size 为 `3`。如果按 `GiB` 计算，表达式约为：

```text
15.2333 GiB * B + 24.3714 GiB
```

最大 batch size 仍为 `3`。

AdamW optimizer step 本身对每个参数只做常数次逐元素操作。粗略计数时，weight decay 约 `2P`，一阶矩更新约 `3P`，二阶矩更新约 `4P`，最终参数更新约 `5P`，总计约：

```text
14 * P FLOPs
```

这是 `O(P)`，通常远小于 Transformer forward/backward 的矩阵乘法成本。

训练时间估算使用第 3 节 GPT-2 XL-shaped 的满长 forward FLOPs：

```text
F_forward = 3,516,769,894,400
```

一个训练 step 包含 forward 和 backward；题目假设 backward 是 forward 的 2 倍，因此：

```text
F_train_step = 3 * batch_size * F_forward
```

H100 的有效吞吐为：

```text
0.5 * 495e12 = 247.5e12 FLOP/s
```

当 `batch_size=1024`、训练 `400K` steps 时：

```text
time
= 400000 * 3 * 1024 * 3,516,769,894,400 / (247.5e12)
= 17,460,229.68 seconds
= 4,850.06 hours
```

也就是约 `202` 天。这个估算忽略了 optimizer step、数据加载、checkpoint、通信等额外开销。

### 补充解释：为什么训练内存比推理内存大很多？

推理时主要需要模型参数和少量中间结果。训练时则至少还需要：

- forward activations：反向传播要用它们计算梯度，所以不能马上丢掉。
- gradients：每个参数都要保存一个同形状的梯度。
- optimizer state：AdamW 对每个参数保存 `m` 和 `v` 两份 state。

因此如果参数本身占 `P` bytes，那么仅 AdamW state 就大约再加 `2P`，gradients 再加 `P`。不算 activations，训练也已经接近 `4P`。这就是为什么训练同一个模型所需显存远高于只加载模型做推理。

## 4.4 Learning rate scheduling

能让 loss 最快下降的 learning rate 往往会在训练过程中变化。在训练 Transformers 时，通常会使用 learning rate schedule：训练开始时使用较大的 learning rate，在初期进行更快更新；随着模型训练推进，慢慢衰减到较小值。[^8]

在这个作业中，我们会实现训练 LLaMA [H. Touvron et al., 2023] 时使用的 cosine annealing schedule。

Scheduler 本质上只是一个函数：它接收当前 step `t` 和其他相关参数，例如初始和最终 learning rates，然后返回 step `t` 的 gradient update 应该使用的 learning rate。最简单的 schedule 是常数函数，它对任意 `t` 都返回同一个 learning rate。

Cosine annealing learning rate schedule 接收以下参数：

1. 当前 iteration `t`
2. 最大 learning rate `alpha_max`
3. 最小，也就是最终 learning rate `alpha_min`
4. warm-up iterations 的数量 `T_w`
5. cosine annealing 的最终 iteration `T_c`

第 `t` 次迭代的 learning rate 定义如下。

**Warm-up：** 如果 `t < T_w`，那么：

```math
\alpha_t = \frac{t}{T_w}\alpha_{\max}.
```

**Cosine annealing：** 如果 `T_w <= t <= T_c`，那么：

```math
\alpha_t
= \alpha_{\min}
+ \frac{1}{2}
\left(
1 + \cos\left(\frac{t - T_w}{T_c - T_w}\pi\right)
\right)
(\alpha_{\max} - \alpha_{\min}).
```

**Post-annealing：** 如果 `t > T_c`，那么：

```math
\alpha_t = \alpha_{\min}.
```

### 补充解释：warmup 和 cosine decay 分别解决什么问题？

训练刚开始时，模型参数是随机的，梯度也可能不稳定。如果一上来就用最大的 learning rate，容易让参数更新过猛。因此 warmup 会让 learning rate 从 0 或很小的值线性升到 `alpha_max`。

训练中后期，模型已经接近较好的区域，继续用很大的步长可能导致 loss 来回震荡。Cosine decay 会平滑地把 learning rate 从 `alpha_max` 降到 `alpha_min`，让后期更新更细致。

它的曲线大致分三段：

```text
线性升高 -> 余弦平滑下降 -> 保持最小值
```

### Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup（1 分）

写一个函数，接收：

```text
t, alpha_max, alpha_min, T_w, T_c
```

并根据上面定义的 scheduler 返回 learning rate：

```text
alpha_t
```

然后实现 `[adapters.get_lr_cosine_schedule]`，并确保它通过：

```bash
uv run pytest -k test_get_lr_cosine_schedule
```

**实现说明：** 代码实现见 `cs336_basics/training.py` 中的 `get_lr_cosine_schedule`，adapter 见 `tests/adapters.py` 中的 `run_get_lr_cosine_schedule`。实现严格分成三段：

- `it < warmup_iters`：线性 warmup。
- `warmup_iters <= it <= cosine_cycle_iters`：cosine annealing。
- `it > cosine_cycle_iters`：固定为 `min_learning_rate`。

## 4.5 Gradient clipping

训练过程中，我们有时会遇到一些训练样本，它们产生很大的 gradients，从而让训练变得不稳定。为了缓解这个问题，实践中经常使用的一种技术是 gradient clipping。

它的想法是在每次 backward pass 之后、optimizer step 之前，对 gradient 的 norm 强制加一个限制。

给定所有参数的 gradient：

```text
g
```

我们计算它的 `l2`-norm：

```math
\|g\|_2
```

如果这个 norm 小于最大值：

```math
M
```

那么保持 `g` 不变。否则，我们把 `g` 缩小，缩放因子为：

```math
\frac{M}{\|g\|_2 + \epsilon}
```

这里会加一个很小的 `epsilon`，例如 `1e-6`，用于数值稳定性。注意，缩放后的 norm 会略小于 `M`。

### 补充解释：gradient clipping 不是改变 loss，而是限制更新幅度

Gradient clipping 发生在：

```text
loss.backward()
```

之后，以及：

```text
optimizer.step()
```

之前。它不会改变 forward 计算和 loss 的定义，只是把已经算出来的梯度按比例缩小。

本作业要求的是 global norm clipping，也就是把所有参数的 gradients 看成一个很长的向量，先算整体 `l2` norm，再统一缩放。它不是分别对每个参数 tensor 单独裁剪。

### Problem (gradient_clipping): Implement gradient clipping（1 分）

写一个函数来实现 gradient clipping。你的函数应该接收一个 parameters 列表和一个最大的 `l2`-norm。它应该原地修改每个 parameter 的 gradient。使用：

```text
epsilon = 1e-6
```

也就是 PyTorch 默认值。

然后实现 adapter：

```python
[adapters.run_gradient_clipping]
```

并确保它通过：

```bash
uv run pytest -k test_gradient_clipping
```

**实现说明：** 代码实现见 `cs336_basics/training.py` 中的 `gradient_clipping`，adapter 见 `tests/adapters.py` 中的 `run_gradient_clipping`。实现会先收集所有非空 gradients，计算 global L2 norm：

```text
total_norm = sqrt(sum_i ||grad_i||_2^2)
```

如果 `total_norm <= max_l2_norm`，则不修改梯度；否则所有梯度共享同一个缩放因子：

```text
clip_coef = max_l2_norm / (total_norm + 1e-6)
```

并原地执行：

```text
grad <- grad * clip_coef
```

## 脚注

[^6]: `o_i[k]` 表示向量 `o_i` 在索引 `k` 处的值。

[^7]: 这对应于 target `x_{i+1}` 上的 Dirac delta distribution 与预测的 `softmax(o_i)` distribution 之间的 cross-entropy。

[^8]: 有时也常见一种 schedule，其中 learning rate 会重新升高，也就是 restarts，用来帮助越过 local minima。
