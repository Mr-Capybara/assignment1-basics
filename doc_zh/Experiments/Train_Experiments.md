# 7 Experiments

现在是时候把所有内容组合起来，在一个 pretraining dataset 上训练小型 language models 了。

## 7.1 How to Run Experiments and Deliverables

理解 Transformer 各个架构组件背后原因的最好方式，就是亲自修改它并运行实验。没有什么能替代 hands-on experience。

为此，能够快速、一致地进行实验，并记录你做过什么，是很重要的。为了快速实验，我们会在一个小规模模型和简单数据集 TinyStories 上运行许多实验。这个模型大约有 17M total parameters。为了保证实验一致，你需要系统地 ablate components 并改变 hyperparameters。为了保留记录，我们会要求你提交 experiment log，以及与每个实验相关的 learning curves。

为了能够提交 loss curves，请确保周期性地评估 validation losses，并同时记录 gradient steps 数量和 wall-clock times。你可能会发现 Weights and Biases 之类的 logging infrastructure 很有帮助。

### Problem (experiment_log): Experiment logging（3 分）

对于你的 training 和 evaluation code，请创建 experiment tracking infrastructure，让你能够根据 gradient steps 和 wall-clock time 跟踪 experiments 和 loss curves。

**Deliverable：** 用于实验的 logging infrastructure code，以及一个 experiment log。这个 experiment log 是一个文档，记录你在本节下面 assignment problems 中尝试过的所有内容。

**本仓库实现：** `scripts/train_lm.py` 提供了两套日志：

- 本地 JSONL 日志：传入 `--metrics-path runs/tinystories_base/metrics.jsonl` 后，每次 `train`、`eval`、`checkpoint`、`divergence`、`finished` 都会追加一行 JSON，字段包括 `experiment_name`、`step`、`elapsed_seconds`、loss、perplexity、learning rate 等。JSONL 适合后续用 pandas 画 step 曲线和 wall-clock 曲线。
- 可选 W&B 日志：传入 `--wandb-project <project>` 后，会同步记录 train/valid loss、perplexity 和 learning rate。

推荐每个实验使用独立的 `--experiment-name` 和输出目录，例如：

```powershell
python scripts/train_lm.py `
  --experiment-name tinystories_base_lr3e-4 `
  --train-data data/tinystories_train.npy `
  --valid-data data/tinystories_valid.npy `
  --checkpoint-path runs/tinystories_base/checkpoint.pt `
  --metrics-path runs/tinystories_base/metrics.jsonl `
  --vocab-size 10000
```

## 7.2 TinyStories

我们会从一个非常简单的数据集开始：TinyStories，见 R. Eldan et al. [1]。模型会在这个数据集上较快训练，并且我们可以观察到一些有趣行为。获取这个数据集的说明见第 1 节。下面是这个数据集中的一个示例。

### Example (tinystories_example): TinyStories 中的一个示例

从前，有一个名叫 Ben 的小男孩。Ben 喜欢探索他周围的世界。他看到了许多很神奇的东西，比如商店里展出的漂亮花瓶。有一天，Ben 走过商店时，遇到了一个非常特别的花瓶。当 Ben 看到它时，他惊呆了！他说：“哇，那真是一个非常神奇的花瓶！我可以买它吗？” 店主微笑着说：“当然可以。你可以把它带回家，给你所有朋友看看它有多神奇！” 于是 Ben 把花瓶带回家，并为它感到非常自豪！他把朋友们叫过来，给他们看这个神奇的花瓶。他所有的朋友都觉得这个花瓶很漂亮，并且不敢相信 Ben 有多幸运。这就是 Ben 如何在商店里找到一个神奇花瓶的故事！

## 7.2.1 Hyperparameter tuning

我们会告诉你一些非常基础的起始 hyperparameters，并要求你为其他 hyperparameters 找到一些表现良好的设置。

**Vocab size 10000.** 典型 vocabulary sizes 通常在几万到几十万之间。你应该改变这个值，并观察 vocabulary 和 model behavior 如何变化。

**Context length 256.** 像 TinyStories 这样的简单数据集可能不需要很长的 sequence lengths，但对于后面的 OpenWebText 数据，你可能想改变这个值。尝试改变 context length，并观察它对 per-iteration runtime 和 final perplexity 的影响。

**d_model 512.** 这比许多 small Transformer papers 使用的 768 dimensions 稍小，但它会让训练更快。

**d_ff 1344.** 这大约是：

```math
\frac{8}{3} d_{\text{model}}
```

同时又是 64 的倍数，这对 GPU performance 有好处。

**RoPE theta parameter Θ 10000.**

**Number of layers and heads 4 layers, 16 heads.** 这些设置合起来会得到大约 17M non-embedding parameters，这是一个相当小的 Transformer。

**Total tokens processed 327,680,000.** 你的：

```text
batch size × total step count × context length
```

应该大致等于这个值。

你应该通过一些 trial and error，为下面这些其他 hyperparameters 找到良好的默认值：

- learning rate
- learning rate warmup
- 其他 AdamW hyperparameters，也就是 `beta_1`、`beta_2`、`epsilon`
- weight decay

你可以在 D. P. Kingma et al. [22] 中找到这些 hyperparameters 的一些典型选择。

## 7.2.2 Putting it together

现在你可以把所有内容组合起来：获得一个训练好的 BPE tokenizer，对 training dataset 进行 tokenization，然后在你写好的 training loop 中运行它。

**重要说明：** 如果你的实现正确且高效，上面的 hyperparameters 在 1 张 B200 GPU 上应该产生大约 20-30 分钟的 runtime。如果你的 runtime 长得多，请检查并确保 dataloading、checkpointing 或 validation loss code 没有成为 runtime bottleneck，并且你的实现已经正确 batched。

## 7.2.3 Tips and tricks for debugging model architectures

我们强烈建议你熟悉 IDE 内置的 debugger，例如 VSCode 或 Zed。相比用 print statements 调试，这会节省你的时间。如果你使用 text editor，可以使用类似 `ipdb` 的工具。

调试 model architectures 时，还有一些其他好的实践：

- 开发任何 neural net architecture 时，一个常见的第一步是 overfit 到单个 minibatch。如果你的实现正确，你应该能够很快把 training loss 降到接近 0。
- 在各种 model components 中设置 debug breakpoints，并检查 intermediate tensors 的 shapes，确保它们符合你的预期。
- 监控 activations、model weights 和 gradients 的 norms，确保它们没有 exploding 或 vanishing。

### Problem (learning_rate): Tune the learning rate（2 B200 hrs）（3 分）

Learning rate 是最重要的 hyperparameters 之一。使用你已经训练的 base model，回答以下问题。

**(a)** 对 learning rates 做一次 hyperparameter sweep，并报告 final losses；如果 optimizer diverges，则记录 divergence。

**Deliverable：** 与多个 learning rates 相关的 learning curves。解释你的 hyperparameter search strategy。

**Deliverable：** 一个在 TinyStories 上 validation loss，也就是 per-token loss，不超过 `1.45` 的模型。

**本仓库实现：** learning rate sweep 不需要改代码，只需要多次运行 `scripts/train_lm.py` 并改变 `--lr-max`、`--lr-min` 和 `--warmup-steps`。脚本会把每次运行的最终状态、train loss、valid loss 和 wall-clock time 写入 `--metrics-path`。如果 loss 变成 `nan` 或 `inf`，训练会记录 `divergence` 事件并停止，便于在 experiment log 中标注“发散”。

建议搜索策略：

1. 固定 base model、batch size、total steps，先粗扫 `1e-4`、`3e-4`、`1e-3`。
2. 找到发散边界后，在边界下方做细扫，例如 `3e-4`、`5e-4`、`7e-4`。
3. 每次 sweep 都让 `--total-steps` 等于 cosine decay 的周期，也就是脚本默认使用的 `cosine_cycle_iters=args.total_steps`。

示例：

```powershell
python scripts/train_lm.py `
  --experiment-name tinystories_lr5e-4 `
  --train-data data/tinystories_train.npy `
  --valid-data data/tinystories_valid.npy `
  --checkpoint-path runs/tinystories_lr5e-4/checkpoint.pt `
  --metrics-path runs/tinystories_lr5e-4/metrics.jsonl `
  --vocab-size 10000 `
  --lr-max 5e-4 `
  --lr-min 5e-5 `
  --warmup-steps 500
```

### Low-Resource Tip: Train for a few steps on CPU or Apple Silicon

如果你在 `cpu` 或 `mps` 上运行，那么你应该把 total tokens processed 数量减少到：

```text
40,000,000
```

这足以生成相当流畅的文本。你也可以把目标 validation loss 从 `1.45` 提高到 `2.00`。

使用我们的 solution code，在一块 M4 Max 芯片和 36 GB RAM 上，用调好的 learning rate，设置：

```text
batch size × total step count × context length = 32 × 5000 × 256 = 40,960,000 tokens
```

在 CPU 上需要 1 小时 22 分钟，在 MPS 上需要 36 分钟。在 step 5000 时，我们达到 validation loss `1.80`。

一些额外提示：

- 当使用 `N` 个 training steps 时，我们建议调整 cosine learning rate decay schedule，让它恰好在 step `N` 终止 decay，也就是到达 minimum learning rate。
- 当使用 `mps` 时，不要使用 TF32 kernels。也就是说，不要像在 cuda devices 上可能会做的那样设置：

```python
torch.set_float32_matmul_precision("high")
```

我们曾在 MPS 上启用 TF32 kernels，torch version `2.9.0`，并发现 backend 有时会使用 silently broken kernels，导致 unstable training。

- 你可以通过用 `torch.compile` JIT-compile 模型来加速训练。具体来说：
  - 在 `cpu` 上，用下面方式 compile 模型：

```python
model = torch.compile(model)
```

  - 在 `mps` 上，可以用下面方式在一定程度上优化 backward pass：

```python
model = torch.compile(model, backend="aot_eager")
```

截至 torch version `2.9.0`，MPS 不支持使用 Inductor 进行 compilation。

**(b)** 经验法则认为，最好的 learning rate 位于 “edge of stability”。请研究 learning rates 开始 diverge 的点与你的最佳 learning rate 之间的关系。

**Deliverable：** 一组逐渐增大的 learning rate 的 learning curves，其中至少包含一次 divergent run，并分析这与 convergence rates 的关系。

现在，让我们改变 batch size，看看训练会发生什么。Batch sizes 很重要：它们可以通过执行更大的 matrix multiplies，让我们从 GPUs 中获得更高效率。但我们是否总是希望 batch sizes 越大越好？让我们运行一些实验来找出答案。

### Problem (batch_size_experiment): Batch size variations（1 B200 hr）（1 分）

把 batch size 从 1 一直改变到 GPU memory limit。中间至少尝试几个 batch sizes，包括 64 和 128 这样的典型大小。

**Deliverable：** 使用不同 batch sizes 运行得到的 learning curves。必要时应该重新优化 learning rates。

**Deliverable：** 用几句话讨论你对 batch sizes 及其对 training 影响的发现。

**本仓库实现：** batch size sweep 使用同一个训练脚本的 `--batch-size` 参数。作业要求 total tokens processed 大致一致，因此当改变 batch size 时，应同步调整 `--total-steps`，让

```text
batch_size × total_steps × context_length
```

保持在同一量级。例如 context length 为 256 时：

- batch size 32、total steps 5000，对应约 40.96M tokens；
- batch size 64、total steps 2500，对应约 40.96M tokens；
- batch size 128、total steps 1250，对应约 40.96M tokens。

如果 batch size 变大后 loss 下降变慢，可以适当提高 `--lr-max` 重新搜索；如果出现发散，就降低 learning rate 或增加 warmup。

有了解码器之后，我们现在可以生成文本了！我们会从模型生成文本，看看它有多好。作为参考，你应该得到至少和下面示例一样好的 outputs。

### Example (ts_generate_example): TinyStories language model 的 sample output

从前，有一个名叫 Lily 的漂亮女孩。她喜欢吃口香糖，尤其是那块大的黑色口香糖。有一天，Lily 的妈妈让她帮忙做晚饭。Lily 非常兴奋！她喜欢帮助妈妈。Lily 的妈妈为晚饭做了一大锅汤。Lily 非常开心，说：“谢谢你，妈妈！我爱你。” 她帮妈妈把汤倒进一个大碗里。晚饭后，Lily 的妈妈做了一些美味的汤。Lily 很喜欢！她说：“谢谢你，妈妈！这个汤真好喝！” 她的妈妈微笑着说：“我很高兴你喜欢它，Lily。” 她们完成了烹饪，并继续一起做饭。故事结束。

### Low-Resource Tip: Generate text on CPU or Apple Silicon

如果你使用的是处理 40M tokens 的 low-resource configuration，你应该会看到生成结果仍然像英语，但没有上面那么流畅。例如，我们在 40M tokens 上训练的 TinyStories language model 的 sample output 如下：

从前，有一个名叫 Sue 的小女孩。Sue 有一颗她非常喜欢的牙齿。那是他最好的头。有一天，Sue 去散步，遇到了一只瓢虫！他们成了好朋友，并一起在小路上玩。

“嘿，Polly！我们出去吧！” Tim 说。Sue 看着天空，发现很难找到一种跳闪亮舞的方式。她微笑着，同意帮助那个会说话的！

当 Sue 看着天空移动时，它是什么。她

下面是精确的问题陈述和我们的要求。

### Problem (generate): Generate text（1 分）

使用你的 decoder 和 trained checkpoint，报告你的模型生成的文本。你可能需要调节 decoder parameters，例如 temperature、top-p 等，来获得流畅 outputs。

**Deliverable：** 至少 256 tokens 的 text dump，或者生成到第一个 `<|endoftext|>` token 为止；并对这个 output 的 fluency 做简短评论，同时说明至少两个影响这个 output 好坏的因素。

**本仓库实现：** `cs336_basics/training.py` 中实现了 `sample_next_token`、`generate_token_ids` 和 `generate_text`；`scripts/generate_lm.py` 提供从 checkpoint 生成文本的命令行入口。生成时必须使用与训练时一致的模型结构参数，例如 `--no-rope`、`--ffn-type silu`、`--norm-position post` 等。

示例：

```powershell
python scripts/generate_lm.py `
  --checkpoint-path runs/tinystories_base/checkpoint.pt `
  --bpe-pickle artifacts/tinystories_bpe/bpe.pkl `
  --prompt "Once upon a time," `
  --max-new-tokens 256 `
  --temperature 0.8 `
  --top-p 0.9
```

影响生成质量的主要因素包括：训练集/验证集 loss 是否足够低、temperature 是否过高或过低、top-p 截断是否太激进、prompt 是否贴近训练分布，以及模型是否在足够 tokens 上训练过。

## 7.3 Ablations and architecture modification

理解 Transformer 的最好方式，是实际修改它并观察它如何表现。我们现在会做几个简单的 ablations 和 modifications。

## Ablation 1: layer normalization

人们常说，layer normalization 对 Transformer training 的稳定性很重要。但也许我们想冒险一点。让我们从每个 Transformer block 中移除 RMSNorm，看看会发生什么。

### Problem (layer_norm_ablation): Remove RMSNorm and train（0.5 B200 hrs）（1 分）

从你的 Transformer 中移除所有 RMSNorms 并训练。在之前的 optimal learning rate 下会发生什么？能否通过使用更低的 learning rate 获得稳定性？

**Deliverable：** 一条移除 RMSNorms 后训练的 learning curve，以及一条使用最佳 learning rate 的 learning curve。

**Deliverable：** 用几句话评论 RMSNorm 的影响。

**本仓库实现：** `TransformerLM` 增加了 `use_rmsnorm` 开关；训练脚本中用 `--no-rmsnorm` 移除 block 内的 `ln1`、`ln2` 和最终 `ln_final`。实现上使用 `IdentityNorm` 原样返回输入，因此模型主体代码仍保持同一条路径。

示例：

```powershell
python scripts/train_lm.py `
  --experiment-name tinystories_no_rmsnorm `
  --train-data data/tinystories_train.npy `
  --valid-data data/tinystories_valid.npy `
  --checkpoint-path runs/tinystories_no_rmsnorm/checkpoint.pt `
  --metrics-path runs/tinystories_no_rmsnorm/metrics.jsonl `
  --vocab-size 10000 `
  --no-rmsnorm `
  --lr-max 1e-4
```

实验时先复用 base model 的最佳 learning rate，若发散，再逐步降低 `--lr-max` 并延长 warmup。

现在让我们研究另一个初看似乎随意的 layer normalization 选择。Pre-norm Transformer blocks 定义为：

```math
z = x + \operatorname{MultiHeadSelfAttention}(\operatorname{RMSNorm}(x))
\tag{25}
```

```math
y = z + \operatorname{FFN}(\operatorname{RMSNorm}(z)).
\tag{26}
```

这是对原始 Transformer architecture 为数不多的 “consensus” modifications 之一。原始 Transformer 使用 post-norm 方法：

```math
z = \operatorname{RMSNorm}(x + \operatorname{MultiHeadSelfAttention}(x))
\tag{27}
```

```math
y = \operatorname{RMSNorm}(z + \operatorname{FFN}(z)).
\tag{28}
```

让我们回到 post-norm 方法，看看会发生什么。

### Problem (pre_norm_ablation): Implement post-norm and train（0.5 B200 hrs）（1 分）

把你的 pre-norm Transformer implementation 修改成 post-norm。使用 post-norm model 训练，看看会发生什么。

**Deliverable：** 一条 post-norm Transformer 的 learning curve，并与 pre-norm 的 learning curve 进行比较。

**本仓库实现：** `TransformerBlock` 支持 `norm_position="pre"` 和 `"post"`。默认是 pre-norm；训练脚本传入 `--norm-position post` 后使用

```math
z = \operatorname{RMSNorm}(x + \operatorname{MHA}(x)), \quad
y = \operatorname{RMSNorm}(z + \operatorname{FFN}(z))
```

对应的 post-norm 路径。

示例：

```powershell
python scripts/train_lm.py `
  --experiment-name tinystories_post_norm `
  --train-data data/tinystories_train.npy `
  --valid-data data/tinystories_valid.npy `
  --checkpoint-path runs/tinystories_post_norm/checkpoint.pt `
  --metrics-path runs/tinystories_post_norm/metrics.jsonl `
  --vocab-size 10000 `
  --norm-position post
```

我们看到，layer normalization 对 Transformer 行为有重大影响，而且 layer normalization 的位置也很重要。

## Ablation 2: position embeddings

接下来，我们会研究 position embeddings 对模型性能的影响。具体来说，我们会比较 base model，也就是使用 RoPE 的模型，和完全不包含 position embeddings 的模型，也就是 NoPE。

事实证明，decoder-only transformers，也就是我们已经实现的带有 causal mask 的 transformers，理论上可以在没有显式提供 position embeddings 的情况下推断 relative 或 absolute position information [Y.-H. H. Tsai et al., 2019; A. Kazemnejad et al., 2023]。现在我们会通过实验测试 NoPE 相比 RoPE 表现如何。

### Problem (no_pos_emb): Implement NoPE（0.5 B200 hrs）（1 分）

修改你的 Transformer implementation，移除 RoPE，完全去掉 position embedding information，并观察会发生什么。

**Deliverable：** 一条比较 RoPE 和 NoPE 表现的 learning curve。

**本仓库实现：** attention 层本来只在 `theta` 和 `max_seq_len` 都存在时创建 RoPE。现在 `TransformerLM` 增加 `use_rope` 开关，训练脚本用 `--no-rope` 完全不创建 RoPE 模块，Q/K 不再注入显式位置旋转。

示例：

```powershell
python scripts/train_lm.py `
  --experiment-name tinystories_nope `
  --train-data data/tinystories_train.npy `
  --valid-data data/tinystories_valid.npy `
  --checkpoint-path runs/tinystories_nope/checkpoint.pt `
  --metrics-path runs/tinystories_nope/metrics.jsonl `
  --vocab-size 10000 `
  --no-rope
```

## Ablation 3: SwiGLU vs. SiLU

接下来，我们会跟随 N. Shazeer [20]，通过比较 SwiGLU feed-forward networks 与使用 SiLU activations 但不使用 gated linear unit，也就是 GLU，的 feed-forward networks，来测试 feed-forward network 中 gating 的重要性：

```math
\operatorname{FFN}_{\text{SiLU}}(x)
= W_2 \operatorname{SiLU}(W_1 x).
\tag{29}
```

回忆一下，在我们的 SwiGLU 实现中，我们把 inner feed-forward layer 的维度设置为大约：

```math
d_{\text{ff}} = \frac{8}{3} d_{\text{model}}
```

同时确保：

```math
d_{\text{ff}} \bmod 64 = 0
```

以便使用 GPU tensor cores。在这个 ablation baseline 中，你的 FFN implementation 应该改为设置：

```math
d^{\text{SiLU}}_{\text{ff}} = 4 \times d_{\text{model}}
```

这样可以让参数量与默认 SwiGLU feed-forward network 大致匹配，因为默认 SwiGLU 有三个 weight matrices，而这个 SiLU baseline 只有两个。

### Problem (swiglu_ablation): SwiGLU vs. SiLU（0.5 B200 hrs）（1 分）

**Deliverable：** 一条比较 SwiGLU 和 SiLU feed-forward networks 表现的 learning curve，其中二者参数量应大致匹配。

**Deliverable：** 用几句话讨论你的发现。

**本仓库实现：** `cs336_basics/model.py` 中新增 `SiLUFeedForward`，公式为 `W2(SiLU(W1 x))`。训练脚本默认 `--ffn-type swiglu`；使用 `--ffn-type silu` 时，如果没有显式传入 `--d-ff`，脚本会自动设置 `d_ff = 4 * d_model`，以便和默认 SwiGLU 的参数量大致匹配。

示例：

```powershell
python scripts/train_lm.py `
  --experiment-name tinystories_silu_ffn `
  --train-data data/tinystories_train.npy `
  --valid-data data/tinystories_valid.npy `
  --checkpoint-path runs/tinystories_silu_ffn/checkpoint.pt `
  --metrics-path runs/tinystories_silu_ffn/metrics.jsonl `
  --vocab-size 10000 `
  --ffn-type silu
```

### Low-Resource Tip: Online students with limited GPU resources should test modifications on TinyStories

在 assignment 的剩余部分，我们会转向一个规模更大、噪声更高的 web dataset，也就是 OpenWebText，实验 architecture modifications，并且可选地向 course leaderboard 提交结果。

在 OpenWebText 上把 LM 训练到 fluency 需要很长时间，因此我们建议 GPU access 有限的 online students 继续在 TinyStories 上测试 modifications，并使用 validation loss 作为评估性能的 metric。

## 7.4 Running on OpenWebText

现在我们会转向一个更标准的 pretraining dataset，它来自 web crawl。我们也提供了 OpenWebText [A. Gokaslan et al., 2019] 的一个小样本，形式是单个 text file；如何访问这个文件见第 1 节。

下面是 OpenWebText 中的一个示例。注意这段文本要真实、复杂、多样得多。你可能想浏览一下 training dataset，以了解 web-scraped corpus 的训练数据是什么样子。

### Example (owt_example): OWT 中的一个示例

Baseball Prospectus 的技术总监 Harry Pavlidis 在聘用 Jonathan Judge 时冒了一次险。

Pavlidis 知道，正如 Alan Schwarz 在 *The Numbers Game* 中所写，“在美国文化中，没有哪个角落比棒球运动员的表现被更精确地计数、更热情地量化。” 只需这里那里点几下，你就能发现 Noah Syndergaard 的 fastball 在飞向本垒板的途中每分钟旋转超过 2,100 次；Nelson Cruz 在 2016 年符合资格的 hitters 中拥有全联盟最高的平均 exit velocity；以及许多其他看起来像是从电子游戏或科幻小说中撕出来的小知识。不断上涨的数据海洋赋予了棒球文化中一个越来越重要的角色力量：analytical hobbyist。

这种赋权也带来了额外审视，不仅审视 measurements，也审视 measurements 背后的人和 publications。对于 Baseball Prospectus，Pavlidis 非常清楚 quantitative imperfection 会伴随 backlash。他也知道，这个网站的 catching metrics 需要重新设计，而且这项工作需要一个有学问的头脑，也就是一个能够处理复杂 statistical modeling problems 的人来完成。

“他让我们感到震惊。” Harry Pavlidis

Pavlidis 有一种直觉：Judge “懂这个”，这种直觉来自 Judge 的写作，以及他们在一次网站赞助的球场活动中的互动。[…]

**Note：** 你可能需要为这个实验重新调节 hyperparameters，例如 learning rate 或 batch size。

### Problem (main_experiment): Experiment on OWT（2 B200 hrs）（2 分）

使用与 TinyStories 相同的 model architecture 和 total training iterations，在 OpenWebText 上训练你的 language model。这个模型表现如何？

**Deliverable：** 你的 language model 在 OpenWebText 上的 learning curve。描述它与 TinyStories losses 的区别：我们应该如何解释这些 losses？

**Deliverable：** 从 OpenWebText LM 生成的文本，格式与 TinyStories outputs 相同。这个文本的 fluency 如何？为什么即使我们使用与 TinyStories 相同的 model 和 compute budget，output quality 仍然更差？

**本仓库实现：** OWT 训练仍然使用 `scripts/train_lm.py`，只需要把 `--train-data` 和 `--valid-data` 指向 OpenWebText tokenized `.npy`。模型结构和 TinyStories 保持一致；learning rate、batch size 可以按 Note 重新调参。

示例：

```powershell
python scripts/train_lm.py `
  --experiment-name owt_base `
  --train-data data/owt_train.npy `
  --valid-data data/owt_valid.npy `
  --checkpoint-path runs/owt_base/checkpoint.pt `
  --metrics-path runs/owt_base/metrics.jsonl `
  --vocab-size 10000
```

生成文本仍然使用 `scripts/generate_lm.py`，只需把 `--checkpoint-path` 换成 OWT 模型 checkpoint。解释 loss 时要注意：OWT 文本来源更复杂、主题和风格更多样，token 分布熵更高；因此同样的模型规模和训练 token budget 下，OWT validation loss 通常高于 TinyStories，生成质量也更不稳定。

## 7.5 Your own modification + leaderboard

恭喜你走到这里。你几乎完成了！现在你将尝试改进 Transformer architecture，并看看你的 hyperparameters 和 architecture 与班上其他同学相比如何。

### Rules for the leaderboard

除了以下限制之外，没有其他限制。

**Runtime：** 你的 submission 在 B200 上最多可以运行 45 分钟。如果你使用 SLURM 或 Modal，可能希望在 submission script 中强制执行这一限制。

**Data：** 你只能使用我们提供的 OpenWebText training dataset。

除此之外，你可以自由尝试任何你想做的事情。

如果你在寻找一些可以实现的想法，可以查看下面这些资源：

- State-of-the-art open-source LLM families，例如 Llama 3 [A. Grattafiori et al., 2024] 或 Qwen 2.5 [A. Yang et al., 2024]。
- NanoGPT speedrun repository，也就是 `github.com/KellerJordan/modded-nanogpt`。在这个仓库中，community members 发布了许多用于 “speedrunning” small-scale language model pretraining 的有趣 modifications。例如，一个可以追溯到原始 Transformer paper 的常见修改，是把 input 和 output embeddings 的 weights 绑定在一起，见 A. Vaswani et al. [8] 的 Section 3.4 和 A. Chowdhery et al. [16] 的 Section 2。如果你尝试 weight tying，可能需要降低 embedding/LM head initialization 的 standard deviation。

在尝试完整的 45 分钟运行之前，你会希望先在 OpenWebText 的小子集上，或者在 TinyStories 上测试这些修改。

需要提醒的是，我们确实注意到，你可能会发现一些在这个 leaderboard 中效果很好的 modifications，并不一定能 generalize 到更大规模的 pretraining。我们会在课程的 scaling laws unit 中进一步探索这个想法。

### Problem (leaderboard): Leaderboard（10 B200 hrs）（6 分）

你将按照上面的 leaderboard rules 训练一个模型，目标是在 `0.75 B200-hours` 内最小化你的 language model 的 validation loss。

**Deliverable：** 记录到的 final validation loss；一条相关的 learning curve，清楚显示 wall-clock-time x-axis 小于 45 分钟；以及你做了什么的描述。我们期望 leaderboard submission 至少超过 naive baseline，也就是 loss `5.0`。提交到 leaderboard 的地址是：

```text
github.com/stanford-cs336/assignment1-basics-leaderboard
```

**本仓库实现：** 训练脚本提供两个 leaderboard 相关能力：

- `--max-runtime-minutes 45`：达到 45 分钟后自动停止并保存 checkpoint，JSONL 中会记录 `finished` 事件和停止原因。
- `--tie-embeddings`：把 input embedding 和 output LM head 权重绑定在一起，这是讲义提示中允许尝试的 architecture modification。

示例：

```powershell
python scripts/train_lm.py `
  --experiment-name leaderboard_tied_emb `
  --train-data data/owt_train.npy `
  --valid-data data/owt_valid.npy `
  --checkpoint-path runs/leaderboard_tied_emb/checkpoint.pt `
  --metrics-path runs/leaderboard_tied_emb/metrics.jsonl `
  --vocab-size 10000 `
  --max-runtime-minutes 45 `
  --tie-embeddings
```

提交前建议先在 TinyStories 或 OWT 子集上短跑，确认没有发散、checkpoint 能加载、`scripts/generate_lm.py` 能正常生成，再跑完整 45 分钟版本。
