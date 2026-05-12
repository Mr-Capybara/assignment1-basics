from __future__ import annotations

import math

import torch
from einops import einsum, rearrange
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))

        # 按作业要求使用无 bias 线性层，并用截断正态初始化 W 而不是 W.T。
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 权重存成 (d_out, d_in)，row-major 输入需要乘 weight.T。
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))

        # embedding matrix 的最后一维是 d_model，token id 直接索引第一维。
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 归一化统计量用 float32 计算更稳，最后再回到输入 dtype。
        x_float = x.float()
        rms_inv = torch.rsqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
        normalized = (x_float * rms_inv).to(dtype=x.dtype)
        return normalized * self.weight


class IdentityNorm(nn.Module):
    """用于消融实验的“空归一化层”。

    Section 7 要求移除所有 RMSNorm。为了让 TransformerBlock 的代码结构保持统一，
    这里提供一个没有参数、只原样返回输入的模块。这样训练脚本只需要切换
    `use_rmsnorm=False`，不用在 forward 里到处写 if/else。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: W2(SiLU(W1 x) * W3 x)，逐位置作用在最后一维。
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """不带门控分支的 SiLU FFN，用于和 SwiGLU 做 ablation。

    公式对应作业中的 FFN_SiLU(x) = W2(SiLU(W1 x))。注意它只有两组权重，
    因此为了和默认 SwiGLU 的参数量大致匹配，实验时通常设置
    d_ff_silu = 4 * d_model，而默认 SwiGLU 使用约 8/3 * d_model。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 先减去最大值，避免 exp 在大 logits 上溢出。
    shifted = x - torch.amax(x, dim=dim, keepdim=True)
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = einsum(q, k, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        # mask=True 表示信息可以流动；False 的位置在 softmax 前置为 -inf。
        scores = scores.masked_fill(~mask.to(device=scores.device, dtype=torch.bool), -torch.inf)

    attention_weights = softmax(scores, dim=-1)
    return einsum(attention_weights, v, "... query key, ... key d_v -> ... query d_v")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = positions[:, None] * inv_freq[None, :]

        # RoPE 的正余弦表是固定常量，不参与训练，也不需要写入 checkpoint。
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = token_positions.to(device=x.device, dtype=torch.long)
        if torch.any(token_positions >= self.max_seq_len) or torch.any(token_positions < 0):
            raise ValueError("token_positions must be in [0, max_seq_len).")

        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)

        # x 可能多一个 head 维；把 cos/sin 在 sequence 前补 singleton 维用于广播。
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = (
            RotaryPositionalEmbedding(theta=theta, d_k=self.d_head, max_seq_len=max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]

        q = rearrange(self.q_proj(x), "... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq (head d_head) -> ... head seq d_head", head=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq (head d_head) -> ... head seq d_head", head=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 下三角 causal mask：第 i 个 query 只能看见 j <= i 的 key。
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attended = scaled_dot_product_attention(q, k, v, causal_mask)
        attended = rearrange(attended, "... head seq d_head -> ... seq (head d_head)")
        return self.output_proj(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        norm_position: str = "pre",
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if norm_position not in {"pre", "post"}:
            raise ValueError('norm_position must be either "pre" or "post".')
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError('ffn_type must be either "swiglu" or "silu".')

        self.norm_position = norm_position
        norm_cls = RMSNorm if use_rmsnorm else IdentityNorm
        self.ln1 = norm_cls(d_model, device=device, dtype=dtype) if use_rmsnorm else norm_cls()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            # NoPE ablation：完全不创建 RoPE 模块，Q/K 直接进入 attention。
            max_seq_len=max_seq_len if use_rope else None,
            theta=theta if use_rope else None,
            device=device,
            dtype=dtype,
        )
        self.ln2 = norm_cls(d_model, device=device, dtype=dtype) if use_rmsnorm else norm_cls()
        self.ffn = (
            SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
            if ffn_type == "swiglu"
            else SiLUFeedForward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_position == "pre":
            # pre-norm：先归一化再进入子层。残差路径上始终保留未归一化的 x，
            # 这通常让深层 Transformer 的梯度传播更稳定。
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            return x + self.ffn(self.ln2(x))

        # post-norm：先做子层和 residual，再对相加后的结果归一化。
        # 这是原始 Transformer 的位置安排，也是 Section 7 要求比较的 ablation。
        x = self.ln1(x + self.attn(x, token_positions=token_positions))
        return self.ln2(x + self.ffn(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        norm_position: str = "pre",
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        tie_embeddings: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.norm_position = norm_position
        self.use_rmsnorm = use_rmsnorm
        self.use_rope = use_rope
        self.ffn_type = ffn_type
        self.tie_embeddings = tie_embeddings

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    norm_position=norm_position,
                    use_rmsnorm=use_rmsnorm,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else IdentityNorm()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        if tie_embeddings:
            # input embedding 和 output projection 的权重形状都是 (vocab_size, d_model)，
            # 因而可以直接共享同一个 Parameter。这样既减少参数量，也实现了
            # leaderboard 提示中提到的 weight tying。
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError("sequence length cannot exceed context_length.")

        token_positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        return self.lm_head(self.ln_final(x))
