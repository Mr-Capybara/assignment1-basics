from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

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
