# 5 Training loop

我们现在终于可以把目前已经构建的主要组件组合起来：tokenized data、model 和 optimizer。

## 5.1 Data Loader

Tokenized data，也就是例如你在 `tokenizer_experiments` 中准备的数据，是一个单独的 token 序列：

```math
x = (x_1, \ldots, x_n).
```

即使源数据可能由相互独立的 documents 组成，例如不同网页或源代码文件，常见做法也是把它们拼接成一个单独的 token 序列，并在它们之间加入 delimiter，例如 `<|endoftext|>` token。

Data loader 会把这个长序列转换成一串 batches。每个 batch 包含 `B` 个长度为 `m` 的序列，并且配有对应的 next tokens，next tokens 的长度同样为 `m`。例如，当 `B = 1, m = 3` 时：

```math
([x_2, x_3, x_4], [x_3, x_4, x_5])
```

就是一个可能的 batch。

以这种方式加载数据会从几个方面简化训练。首先，任何满足：

```math
1 \le i \le n - m
```

的位置都能给出一个合法的训练序列，因此 sampling training sequences 非常直接。其次，由于所有 training sequences 都具有相同长度，所以不需要 pad input sequences，这会提升硬件利用率，也能通过增大 batch size `B` 来进一步提升利用率。最后，我们也不需要为了采样训练数据而把整个数据集完整加载进内存，这使得处理那些原本可能无法放入内存的大数据集变得容易。

### Problem (data_loading): Implement data loading（2 分）

**Deliverable：** 写一个函数，它接收：

- 一个 numpy array `x`，这是一个包含 token IDs 的 integer array。
- `batch_size`
- `context_length`
- 一个 PyTorch device string，例如 `'cpu'` 或 `'cuda:0'`

并返回一对 tensors：sampled input sequences 和对应的 next-token targets。

这两个 tensors 都应该具有形状：

```text
(batch_size, context_length)
```

其中包含 token IDs，并且都应该被放到请求的 device 上。

为了对照我们提供的测试检查你的实现，你首先需要实现 test adapter：

```python
[adapters.run_get_batch]
```

然后运行：

```bash
uv run pytest -k test_get_batch
```

来测试你的实现。

**实现说明：** 代码实现见 `cs336_basics/training.py` 中的 `get_batch`，adapter 见 `tests/adapters.py` 中的 `run_get_batch`。实现会在合法范围：

```text
0 <= start < len(dataset) - context_length
```

内随机采样 `batch_size` 个起点。每个起点产生一对训练样本：

```text
x = dataset[start : start + context_length]
y = dataset[start + 1 : start + context_length + 1]
```

然后把它们转换成 `torch.long` tensor，并直接放到用户指定的 device 上。实现使用 numpy 高级索引一次性取出所有窗口，因此对普通 `np.ndarray` 和 `np.memmap` 都可用。

### Low-Resource Tip: Data loading on CPU or Apple Silicon

如果你计划在 CPU 或 Apple Silicon 上训练你的 LM，你需要把数据移动到正确的 device；类似地，后面也应该对模型使用同一个 device。

如果你在 CPU 上，可以使用 `'cpu'` device string；如果你在 Apple Silicon，也就是 M 系列芯片上，可以使用 `'mps'` device string。

关于 MPS，可以参考以下资源：

- https://docs.pytorch.org/docs/stable/mps.html
- https://docs.pytorch.org/docs/stable/notes/mps.html
- https://developer.apple.com/documentation/metalperformanceshaders

如果数据集太大，无法加载进内存，该怎么办？我们可以使用一个名为 `mmap` 的 Unix system call。它会把磁盘上的文件映射到 virtual memory，并且只有当某个 memory location 被访问时，才会 lazy load 对应的文件内容。因此，你可以“假装”整个数据集都在内存中。

Numpy 通过 `np.memmap` 实现这一机制；如果你最初用 `np.save` 保存数组，也可以在 `np.load` 中使用：

```python
mmap_mode="r"
```

这会返回一个类似 numpy array 的对象，在你访问 entries 时按需加载它们。

在训练期间从数据集，也就是一个 numpy array，中采样时，请确保以 memory-mapped mode 加载 dataset。具体方式取决于你如何保存数组，可以使用 `np.memmap`，也可以使用 `np.load(..., mmap_mode="r")`。还要确保你指定的 `dtype` 与正在加载的 array 匹配。

显式验证 memory-mapped data 看起来是否正确会很有帮助，例如检查它是否没有包含超过预期 vocabulary size 的值。

## 5.2 Checkpointing

除了加载数据之外，我们还需要在训练过程中保存模型。运行 jobs 时，我们经常希望能够恢复一个中途停止的 training run，例如因为 job 超时、机器故障等原因导致停止。即使一切顺利，我们也可能希望之后访问中间模型，例如用于事后研究 training dynamics，或者从不同训练阶段的模型中采样。

Checkpoint 应该包含恢复训练所需的所有 states。最基本地，我们当然希望能恢复 model weights。如果使用 stateful optimizer，例如 AdamW，那么还需要保存 optimizer 的 state；在 AdamW 的例子中，这包括 moment estimates。最后，为了恢复 learning rate schedule，我们还需要知道训练停止时的 iteration number。

PyTorch 让保存这些信息变得容易。每个 `nn.Module` 都有：

```python
state_dict()
```

方法，它会返回一个包含所有 learnable weights 的 dictionary；之后我们可以用对应的：

```python
load_state_dict()
```

方法恢复这些 weights。任何 `torch.optim.Optimizer` 也同样如此。

最后，`torch.save(obj, dest)` 可以把一个 object dump 到 file path 或 file-like object 中。例如，这个 object 可以是一个 dictionary，其中一些 values 是 tensors，也可以包含普通 Python objects，例如 integers。之后可以用：

```python
torch.load(src)
```

把它重新加载回内存。

### Problem (checkpointing): Implement model checkpointing（1 分）

实现以下两个函数来加载和保存 checkpoints。

```python
def save_checkpoint(model, optimizer, iteration, out)
```

这个函数应该把来自 `model`、`optimizer` 和 `iteration` 的所有 state dump 到 file-like object `out` 中。你可以使用 model 和 optimizer 的 `state_dict` 方法来获取它们的相关 states，并使用：

```python
torch.save(obj, out)
```

把 `obj` dump 到 `out` 中。PyTorch 在这里既支持 path，也支持 file-like object。一个典型选择是让 `obj` 成为一个 dictionary，但你可以使用任何你想要的格式，只要之后能加载你的 checkpoint 即可。

这个函数期望以下参数：

```python
model: torch.nn.Module
optimizer: torch.optim.Optimizer
iteration: int
out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
```

```python
def load_checkpoint(src, model, optimizer)
```

这个函数应该从 `src`，也就是 path 或 file-like object，加载 checkpoint，然后从 checkpoint 中恢复 model 和 optimizer states。你的函数应该返回保存到 checkpoint 中的 iteration number。你可以使用：

```python
torch.load(src)
```

来恢复你在 `save_checkpoint` 实现中保存的内容，并使用 model 和 optimizer 的 `load_state_dict` 方法，把它们恢复到之前的 states。

这个函数期望以下参数：

```python
src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
model: torch.nn.Module
optimizer: torch.optim.Optimizer
```

实现 adapters：

```python
[adapters.run_save_checkpoint]
[adapters.run_load_checkpoint]
```

并确保它们通过：

```bash
uv run pytest -k test_checkpointing
```

**实现说明：** 代码实现见 `cs336_basics/training.py` 中的 `save_checkpoint` 和 `load_checkpoint`，adapter 见 `tests/adapters.py` 中的 `run_save_checkpoint` 和 `run_load_checkpoint`。保存格式是一个普通 dictionary：

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iteration": iteration,
}
```

保存时使用 `torch.save(checkpoint, out)`；加载时使用 `torch.load(src, map_location="cpu")` 读取，再分别调用：

```python
model.load_state_dict(...)
optimizer.load_state_dict(...)
```

最后返回 checkpoint 中保存的 `iteration`，用于恢复 training loop 的 step 计数和 learning rate schedule。

## 5.3 Training loop

现在，终于到了把你实现的所有组件组合到主 training script 中的时候。让启动具有不同 hyperparameters 的 training runs 变得容易，会在之后带来回报。例如，可以让 training script 通过 command-line arguments 接收这些参数；后续你会多次运行训练，研究不同选择如何影响训练。

### Problem (training_together): Put it together（4 分）

**Deliverable：** 写一个 script，运行 training loop，在用户提供的输入上训练你的模型。具体来说，我们建议你的 training script 至少支持以下能力：

- 能够配置和控制各种 model 和 optimizer hyperparameters。
- 使用 `np.memmap` 对大型 training 和 validation datasets 做 memory-efficient loading。
- 将 checkpoints 序列化到用户提供的 path。
- 周期性地记录 training 和 validation performance，例如记录到 console，或者记录到 Weights and Biases 这样的外部服务。[^9]

**实现说明：** 训练脚本实现见 `scripts/train_lm.py`。它支持通过命令行配置主要 model hyperparameters 和 optimizer hyperparameters，包括：

- `vocab_size`
- `context_length`
- `d_model`
- `num_layers`
- `num_heads`
- `d_ff`
- `rope_theta`
- `batch_size`
- AdamW 的 learning rate、betas、epsilon、weight decay
- warmup steps、total steps、gradient clipping norm

数据加载使用：

```python
np.load(path, mmap_mode="r")
```

因此 tokenized `.npy` 文件不会被一次性完整读入内存。每一步训练的基本流程是：

```text
设置当前 step 的 cosine learning rate
-> get_batch 采样 x/y
-> model(x)
-> cross_entropy(logits, y)
-> backward
-> gradient_clipping
-> optimizer.step
```

脚本会按 `--log-every` 输出训练 loss，按 `--eval-every` 在验证集上估计 validation loss，按 `--save-every` 保存 checkpoint。它也支持 `--resume-from` 从 checkpoint 恢复训练；如果传入 `--wandb-project`，会把训练和验证指标记录到 Weights and Biases。

# 6 Generating text

现在我们已经可以训练模型，最后还需要一个能力：从模型生成文本。

回忆一下，language model 接收一个长度为 `sequence_length` 的 integer sequence，这个 sequence 也可以是 batched 的；然后它会生成一个大小为：

```text
(sequence_length, vocab_size)
```

的 matrix。序列中的每个元素都会对应一个 probability distribution，用来预测该位置之后的下一个 token。

现在我们会写一些函数，把它转换成用于生成新序列的 sampling scheme。

## Softmax

按照标准约定，language model 的输出是最后一个 linear layer 的输出，也就是 logits。因此，我们必须通过 softmax operation 把它转换成一个 normalized probability。这个 softmax operation 我们之前已经在等式 10 中见过。

## Decoding

为了从模型生成文本，也就是 decode，我们会向模型提供一个 prefix tokens 序列，也就是 prompt；然后要求模型生成一个 vocabulary 上的 probability distribution，用来预测序列中的下一个 token。接着，我们会从这个 vocabulary items 上的分布中采样，从而决定下一个 output token。

更具体地说，decoding process 的一步应该接收一个序列：

```math
x_{1 \ldots t}
```

并通过以下等式返回一个 token：

```math
x_{t+1}
```

```math
P(x_{t+1} = i \mid x_{1 \ldots t})
= \frac{\exp(v_i)}{\sum_j \exp(v_j)}
\tag{21}
```

其中：

```math
v = \operatorname{TransformerLM}(x_{1 \ldots t})_t
\in \mathbb{R}^{\text{vocab_size}}.
\tag{22}
```

这里的 `TransformerLM` 是我们的模型。它接收长度为 `sequence_length` 的 sequence 作为输入，并生成一个大小为：

```text
(sequence_length, vocab_size)
```

的 matrix。由于我们要预测第 `t` 个位置之后的 next token，所以取这个 matrix 的最后一个元素。

这样，我们就得到了一个基础 decoder：不断从这些 one-step conditionals 中采样，并把上一步生成的 output token 追加到下一次 decoding timestep 的输入中。这个过程一直重复，直到生成 end-of-sequence token：

```text
<|endoftext|>
```

或者达到用户指定的最大生成 token 数。

## Decoder tricks

我们会用小模型做实验，而小模型有时会生成质量很低的文本。有两个简单的 decoder tricks 可以帮助修复这些问题。

首先，在 **temperature scaling** 中，我们使用 temperature parameter：

```math
\tau
```

修改 softmax。新的 softmax 是：

```math
\operatorname{softmax}(v, \tau)_i
= \frac{\exp(v_i / \tau)}
{\sum_{j=1}^{\text{vocab_size}} \exp(v_j / \tau)}.
\tag{23}
```

注意，当设置：

```math
\tau \to 0
```

时，`v` 中最大的元素会占据主导地位，softmax 的输出会变成一个集中在最大元素上的 one-hot vector。

第二个技巧是 **nucleus sampling**，也叫 **top-p sampling**。在这个技巧中，我们通过截断 low-probability tokens 来修改 sampling distribution。

令 `q` 是一个 probability distribution，它来自一个 temperature-scaled softmax，大小为 `vocab_size`。带有 hyperparameter `p` 的 nucleus sampling 会根据以下等式产生 next token：

```math
P(x_{t+1} = i \mid q)
=
\begin{cases}
\dfrac{q_i}{\sum_{j \in V(p)} q_j}, & \text{if } i \in V(p) \\
0, & \text{otherwise}
\end{cases}
\tag{24}
```

其中 `V(p)` 是满足以下条件的最小 index set：

```math
\sum_{j \in V(p)} q_j \ge p.
```

你可以很容易地计算这个量：先按大小对 probability distribution `q` 进行排序，然后不断选择最大的 vocabulary elements，直到达到目标水平 `p`。

### Problem (decoding): Decoding（3 分）

**Deliverable：** 实现一个函数，用于从你的 language model 中 decode。我们建议你支持以下 features：

- 为用户提供的 prompt 生成 completions。也就是说，接收某个：

```math
x_{1 \ldots t}
```

并采样 completion，直到遇到 `<|endoftext|>` token。

- 允许用户控制 generated tokens 的最大数量。
- 给定一个 desired temperature value，在 sampling 之前，对预测的 next-token distributions 应用 softmax temperature scaling。
- Top-`p` sampling，也称为 nucleus sampling [A. Holtzman et al., 2020]，给定一个用户指定的 threshold value。

**实现说明：** 解码相关实现见 `cs336_basics/training.py` 中的三个函数：

- `sample_next_token`：对单步 logits 做 temperature scaling 和可选 top-p 过滤，然后采样一个 token。
- `generate_token_ids`：接收 prompt token IDs，自回归生成 token IDs。
- `generate_text`：对文本 prompt 做 `tokenizer.encode -> generate_token_ids -> tokenizer.decode` 的封装。

`sample_next_token` 中，`temperature=0` 时退化为 greedy decoding，也就是直接取最大 logit 的 token。`temperature>0` 时先计算：

```text
softmax(logits / temperature)
```

如果启用 top-p，则先按概率从大到小排序，选择累计概率达到阈值 `p` 所需的最小 token 集合，再把这个集合内的概率重新归一化并采样。

`generate_token_ids` 每次只取模型输出的最后一个位置：

```python
logits = model(input_ids)[0, -1]
```

作为 next-token distribution。生成出的 token 会追加到当前序列后面，并用于下一步输入。如果当前序列长度超过 `context_length`，实现只保留最近的 `context_length` 个 token 作为模型输入；如果生成了 `end_token_id`，则提前停止。

## 脚注

[^9]: wandb.ai
