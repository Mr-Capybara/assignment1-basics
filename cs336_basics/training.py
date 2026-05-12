from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """对最后一维 vocabulary logits 计算平均 cross-entropy。

    inputs 的形状是 (..., vocab_size)，targets 的形状是 (...)。
    这里不先显式计算 softmax，避免大 logits 进入 exp 后发生数值溢出。
    """
    targets = targets.to(device=inputs.device, dtype=torch.long)

    # logsumexp(x) = max(x) + log(sum(exp(x - max(x))))，这是稳定版 CE 的核心。
    max_logits = torch.amax(inputs, dim=-1, keepdim=True)
    shifted_logits = inputs - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1)) + max_logits.squeeze(-1)

    # gather 取出每个样本 target class 对应的 logit，支持任意 batch-like 前缀维度。
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    losses = log_sum_exp - target_logits
    return torch.mean(losses)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """按所有参数梯度的 global L2 norm 做梯度裁剪。

    如果整体梯度范数超过 max_l2_norm，则所有梯度共享同一个缩放因子；
    没有梯度的参数会被跳过。
    """
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return

    # global norm 是把所有梯度展平成一个长向量后的 L2 norm。
    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(grad.detach(), ord=2) for grad in grads])
    )
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef >= 1:
        return

    for grad in grads:
        grad.mul_(clip_coef.to(device=grad.device, dtype=grad.dtype))


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                grad = parameter.grad
                state = self.state[parameter]
                if len(state) == 0:
                    # AdamW 需要为每个参数保存一阶矩 m 和二阶矩 v。
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # decoupled weight decay：直接衰减参数，不把它混入梯度矩估计。
                parameter.add_(parameter, alpha=-lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias correction：修正 m/v 从 0 初始化导致的训练早期偏小问题。
                step_size = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                denom = exp_avg_sq.sqrt().add_(eps)
                parameter.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """带 warmup 的 cosine annealing 学习率调度。"""
    if it < warmup_iters:
        if warmup_iters == 0:
            return max_learning_rate
        return it / warmup_iters * max_learning_rate

    if it <= cosine_cycle_iters:
        if cosine_cycle_iters == warmup_iters:
            return min_learning_rate
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_weight = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_weight * (max_learning_rate - min_learning_rate)

    return min_learning_rate


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从一维 token ID 数组中随机采样语言模型训练 batch。

    每个样本使用 dataset[start : start + context_length] 作为输入，
    使用向右平移一位的 dataset[start + 1 : start + context_length + 1] 作为标签。
    dataset 可以是普通 numpy array，也可以是 np.memmap。
    """
    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D array of token IDs.")
    if len(dataset) <= context_length:
        raise ValueError("dataset must contain at least context_length + 1 tokens.")

    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    offsets = np.arange(context_length)

    # 高级索引一次性取出所有窗口；对 memmap 也只会读取实际访问的片段。
    input_batch = dataset[starts[:, None] + offsets[None, :]]
    target_batch = dataset[starts[:, None] + offsets[None, :] + 1]

    x = torch.as_tensor(input_batch, dtype=torch.long, device=device)
    y = torch.as_tensor(target_batch, dtype=torch.long, device=device)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """保存恢复训练所需的最小状态：模型、优化器和迭代步数。"""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """加载 checkpoint，并把模型和优化器恢复到保存时的状态。"""
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """从单步 logits 中采样一个 token，支持 temperature 和 top-p。

    logits 必须是一维张量，形状为 (vocab_size,)。
    temperature 越小，分布越尖锐；temperature=0 时退化为 greedy argmax。
    """
    if logits.ndim != 1:
        raise ValueError("logits must be a 1D tensor of shape (vocab_size,).")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")

    if temperature == 0:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p is not None and top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 取达到 top_p 阈值所需的最小 token 集合，并保留越过阈值的第一个 token。
        cutoff = int(torch.searchsorted(cumulative_probs, torch.tensor(top_p, device=logits.device)).item())
        cutoff = min(cutoff, sorted_probs.shape[0] - 1)
        keep = torch.arange(sorted_probs.shape[0], device=logits.device) <= cutoff
        probs_to_sample = sorted_probs[keep]
        probs_to_sample = probs_to_sample / probs_to_sample.sum()
        sampled_position = torch.multinomial(probs_to_sample, num_samples=1, generator=generator)
        return int(sorted_indices[keep][sampled_position].item())

    sampled_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(sampled_token.item())


@torch.no_grad()
def generate_token_ids(
    model: torch.nn.Module,
    prompt_token_ids: list[int] | torch.Tensor,
    max_new_tokens: int,
    context_length: int | None = None,
    end_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """基于 prompt token IDs 自回归生成 token IDs。

    如果生成长度超过模型 context_length，只保留最近的 context_length 个 token
    作为下一步输入；返回值包含原始 prompt 和新生成的 token。
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    was_training = model.training
    model.eval()
    model_device = device if device is not None else next(model.parameters()).device

    if isinstance(prompt_token_ids, torch.Tensor):
        generated = prompt_token_ids.detach().to(dtype=torch.long, device=model_device).flatten().tolist()
    else:
        generated = list(prompt_token_ids)

    if not generated:
        raise ValueError("prompt_token_ids must contain at least one token.")

    for _ in range(max_new_tokens):
        context = generated[-context_length:] if context_length is not None else generated
        input_ids = torch.tensor(context, dtype=torch.long, device=model_device).unsqueeze(0)
        logits = model(input_ids)[0, -1]
        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        generated.append(next_token)
        if end_token_id is not None and next_token == end_token_id:
            break

    if was_training:
        model.train()
    return generated


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    context_length: int | None = None,
    end_token: str = "<|endoftext|>",
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """对文本 prompt 做 encode -> generate -> decode 的便捷封装。"""
    prompt_token_ids = tokenizer.encode(prompt)
    end_token_id = None
    if end_token is not None and hasattr(tokenizer, "token_to_id"):
        end_token_id = tokenizer.token_to_id.get(end_token.encode("utf-8"))

    output_ids = generate_token_ids(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        end_token_id=end_token_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
        generator=generator,
    )
    return tokenizer.decode(output_ids)
