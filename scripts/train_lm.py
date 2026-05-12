from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    """解析训练脚本参数。

    这个脚本既服务基础训练，也服务 assignment 第 7 节的实验：
    - learning-rate / batch-size sweep：直接改 optimizer 和 batch 参数；
    - ablation：通过 norm、RoPE、FFN 相关开关切换模型结构；
    - experiment log：用 JSONL 文件持续记录 step、wall-clock time 和 loss。
    """

    parser = argparse.ArgumentParser(description="Train a CS336 Transformer language model.")

    data_group = parser.add_argument_group("data and checkpoints")
    data_group.add_argument("--train-data", type=Path, required=True, help="训练集 token IDs 的 .npy 文件。")
    data_group.add_argument("--valid-data", type=Path, required=True, help="验证集 token IDs 的 .npy 文件。")
    data_group.add_argument("--checkpoint-path", type=Path, required=True, help="checkpoint 保存路径。")
    data_group.add_argument("--resume-from", type=Path, default=None, help="从已有 checkpoint 继续训练。")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--vocab-size", type=int, required=True)
    model_group.add_argument("--context-length", type=int, default=256)
    model_group.add_argument("--d-model", type=int, default=512)
    model_group.add_argument("--num-layers", type=int, default=4)
    model_group.add_argument("--num-heads", type=int, default=16)
    model_group.add_argument(
        "--d-ff",
        type=int,
        default=None,
        help=(
            "FFN hidden size。默认值按结构自动选择：SwiGLU 使用约 8/3*d_model 且向下取 64 的倍数；"
            "SiLU ablation 使用 4*d_model。"
        ),
    )
    model_group.add_argument("--rope-theta", type=float, default=10000.0)

    ablation_group = parser.add_argument_group("section 7 architecture switches")
    ablation_group.add_argument(
        "--norm-position",
        choices=("pre", "post"),
        default="pre",
        help="pre 为默认 pre-norm；post 用于 pre_norm_ablation。若同时 --no-rmsnorm，则该开关退化为残差顺序差异。",
    )
    ablation_group.add_argument(
        "--no-rmsnorm",
        action="store_true",
        help="移除所有 RMSNorm，包括 block 内的 ln1/ln2 和最终 ln_final。用于 layer_norm_ablation。",
    )
    ablation_group.add_argument(
        "--no-rope",
        action="store_true",
        help="禁用 RoPE，不使用任何显式位置编码。用于 no_pos_emb/NoPE ablation。",
    )
    ablation_group.add_argument(
        "--ffn-type",
        choices=("swiglu", "silu"),
        default="swiglu",
        help="默认 SwiGLU；silu 表示无门控 FFN，用于 swiglu_ablation。",
    )
    ablation_group.add_argument(
        "--tie-embeddings",
        action="store_true",
        help="共享 token embedding 和 LM head 权重，可作为 leaderboard 自选改动。",
    )

    optim_group = parser.add_argument_group("optimization")
    optim_group.add_argument("--batch-size", type=int, default=32)
    optim_group.add_argument("--total-steps", type=int, default=5000)
    optim_group.add_argument("--lr-max", type=float, default=3e-4)
    optim_group.add_argument("--lr-min", type=float, default=3e-5)
    optim_group.add_argument("--warmup-steps", type=int, default=500)
    optim_group.add_argument("--weight-decay", type=float, default=0.01)
    optim_group.add_argument("--beta1", type=float, default=0.9)
    optim_group.add_argument("--beta2", type=float, default=0.95)
    optim_group.add_argument("--eps", type=float, default=1e-8)
    optim_group.add_argument("--max-grad-norm", type=float, default=1.0)

    runtime_group = parser.add_argument_group("runtime and logging")
    runtime_group.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    runtime_group.add_argument("--seed", type=int, default=0)
    runtime_group.add_argument("--experiment-name", type=str, default="cs336-lm")
    runtime_group.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="JSONL metrics 文件；每行是一条 train/eval/checkpoint/divergence 事件。",
    )
    runtime_group.add_argument("--log-every", type=int, default=100)
    runtime_group.add_argument("--eval-every", type=int, default=500)
    runtime_group.add_argument("--eval-iters", type=int, default=20)
    runtime_group.add_argument("--save-every", type=int, default=1000)
    runtime_group.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=None,
        help="达到该 wall-clock 时间后停止并保存 checkpoint，便于 leaderboard 的 45 分钟限制。",
    )
    runtime_group.add_argument(
        "--compile-backend",
        choices=("none", "default", "aot_eager"),
        default="none",
        help="可选 torch.compile。CPU 常用 default；MPS 可尝试 aot_eager。",
    )
    runtime_group.add_argument("--wandb-project", type=str, default=None)
    return parser.parse_args()


def resolve_d_ff(args: argparse.Namespace) -> int:
    """根据 FFN 类型补齐默认 d_ff。

    作业默认 SwiGLU 的 d_ff 约为 8/3*d_model，并取 64 的倍数来贴合 GPU tensor cores。
    SiLU ablation 少一个门控矩阵，所以用 4*d_model 让参数量与 SwiGLU 大致相当。
    """

    if args.d_ff is not None:
        return args.d_ff
    if args.ffn_type == "silu":
        return 4 * args.d_model

    rounded_down = int((8 * args.d_model / 3) // 64) * 64
    return max(64, rounded_down)


def set_random_seed(seed: int) -> None:
    """固定 numpy 和 torch 的随机源，方便复现实验曲线。"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """统计可训练参数数量；weight tying 时共享 Parameter 只计算一次。"""

    unique_parameters = {id(parameter): parameter for parameter in model.parameters()}
    return sum(parameter.numel() for parameter in unique_parameters.values() if parameter.requires_grad)


def perplexity_from_loss(loss: float) -> float:
    """把平均 token-level cross entropy 转成 perplexity。

    loss 很大时直接 exp 可能溢出；这里做一个保守截断，仅用于日志展示。
    """

    return math.exp(min(loss, 20.0))


class ExperimentLogger:
    """极轻量的 JSONL 实验日志。

    W&B 很适合画曲线，但作业还要求提交 experiment log。JSONL 的好处是：
    - 训练中途崩溃时，已经写出的每一行仍然是完整记录；
    - 后续可以用 pandas/脚本按 experiment_name、step、elapsed_seconds 画曲线；
    - 不依赖外部服务，在离线环境里也能复现实验记录。
    """

    def __init__(self, path: Path | None, experiment_name: str, start_time: float) -> None:
        self.path = path
        self.experiment_name = experiment_name
        self.start_time = start_time
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, step: int, **payload: Any) -> None:
        if self.path is None:
            return

        record = {
            "event": event,
            "experiment_name": self.experiment_name,
            "step": step,
            "elapsed_seconds": time.time() - self.start_time,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def build_model(args: argparse.Namespace, device: torch.device) -> TransformerLM:
    """按命令行参数创建模型。

    默认配置对应 assignment 的 base Transformer；第 7 节的 ablation 只改变这里传入
    TransformerLM 的几个显式开关，因此每次实验的结构差异清晰可查。
    """

    return TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_position=args.norm_position,
        use_rmsnorm=not args.no_rmsnorm,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
        tie_embeddings=args.tie_embeddings,
        device=device,
    )


def maybe_compile_model(model: TransformerLM, backend: str) -> torch.nn.Module:
    """可选地调用 torch.compile。

    注意 checkpoint 始终保存未 compile 的 raw model。compile 后返回的是包装模块，
    它适合 forward/backward 加速，但直接保存它的 state_dict 会引入额外 key 前缀。
    """

    if backend == "none":
        return model
    if backend == "default":
        return torch.compile(model)
    return torch.compile(model, backend=backend)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    eval_iters: int,
) -> float:
    """用随机 batch 估计验证 loss。

    验证集通常很大，完整扫描会明显拖慢第 7 节的 sweep。这里抽取固定数量的 batch
    近似估计 validation loss，既能观察学习曲线趋势，又不会让 eval 成为瓶颈。
    """

    was_training = model.training
    model.eval()

    losses: list[float] = []
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        losses.append(cross_entropy(model(x), y).item())

    if was_training:
        model.train()
    return sum(losses) / len(losses)


def log_to_wandb(wandb_run: Any, metrics: dict[str, float | int]) -> None:
    """W&B 是可选依赖；集中封装后主循环更容易读。"""

    if wandb_run is not None:
        wandb_run.log(metrics)


def main() -> None:
    args = parse_args()
    args.d_ff = resolve_d_ff(args)
    set_random_seed(args.seed)

    device = torch.device(args.device)

    # mmap_mode="r" 让 numpy 按需从磁盘读取 token，适合大规模 tokenized 数据。
    # get_batch 只会索引当前 batch 需要的窗口，不会把完整数据复制进内存。
    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")

    raw_model = build_model(args, device)
    optimizer = AdamW(
        raw_model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from, raw_model, optimizer)
        print(f"resumed from {args.resume_from} at step {start_step}")

    model = maybe_compile_model(raw_model, args.compile_backend)

    wandb_run = None
    if args.wandb_project is not None:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=args.experiment_name, config=vars(args))

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    max_runtime_seconds = None if args.max_runtime_minutes is None else args.max_runtime_minutes * 60
    logger = ExperimentLogger(args.metrics_path, args.experiment_name, start_time)
    parameter_count = count_parameters(raw_model)

    logger.log("config", start_step, config=vars(args), parameter_count=parameter_count)
    print(
        "config "
        f"name={args.experiment_name} params={parameter_count:,} "
        f"d_ff={args.d_ff} norm={args.norm_position} "
        f"rmsnorm={not args.no_rmsnorm} rope={not args.no_rope} ffn={args.ffn_type}"
    )

    raw_model.train()
    last_iteration = start_step
    stopped_reason = "completed"

    for step in range(start_step, args.total_steps):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=args.lr_max,
            min_learning_rate=args.lr_min,
            warmup_iters=args.warmup_steps,
            cosine_cycle_iters=args.total_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 语言模型训练样本是相邻 token 窗口：
        # x = tokens[t : t+context_length]，y = tokens[t+1 : t+context_length+1]。
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        optimizer.zero_grad()
        loss = cross_entropy(model(x), y)

        if not torch.isfinite(loss):
            stopped_reason = "diverged"
            last_iteration = step
            loss_value = float(loss.detach().cpu().item())
            print(f"step={step} loss={loss_value} diverged")
            logger.log("divergence", step, train_loss=loss_value, lr=lr)
            log_to_wandb(wandb_run, {"train/loss": loss_value, "lr": lr, "step": step})
            break

        loss.backward()

        # global norm clipping 对小模型 sweep 很实用：它不能修复错误 learning rate，
        # 但可以减少偶发的大梯度尖峰，让不同实验之间的曲线更可比较。
        if args.max_grad_norm > 0:
            gradient_clipping(raw_model.parameters(), args.max_grad_norm)

        optimizer.step()
        iteration = step + 1
        last_iteration = iteration

        if iteration % args.log_every == 0:
            train_loss = loss.item()
            train_ppl = perplexity_from_loss(train_loss)
            elapsed = time.time() - start_time
            print(
                f"step={iteration} train_loss={train_loss:.4f} "
                f"train_ppl={train_ppl:.2f} lr={lr:.3e} elapsed={elapsed:.1f}s"
            )
            logger.log("train", iteration, train_loss=train_loss, train_ppl=train_ppl, lr=lr)
            log_to_wandb(wandb_run, {"train/loss": train_loss, "train/ppl": train_ppl, "lr": lr, "step": iteration})

        if iteration % args.eval_every == 0:
            valid_loss = estimate_loss(
                model=model,
                dataset=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_iters=args.eval_iters,
            )
            valid_ppl = perplexity_from_loss(valid_loss)
            print(f"step={iteration} valid_loss={valid_loss:.4f} valid_ppl={valid_ppl:.2f}")
            logger.log("eval", iteration, valid_loss=valid_loss, valid_ppl=valid_ppl, lr=lr)
            log_to_wandb(wandb_run, {"valid/loss": valid_loss, "valid/ppl": valid_ppl, "step": iteration})

        if iteration % args.save_every == 0:
            save_checkpoint(raw_model, optimizer, iteration, args.checkpoint_path)
            print(f"saved checkpoint to {args.checkpoint_path}")
            logger.log("checkpoint", iteration, checkpoint_path=args.checkpoint_path)

        if max_runtime_seconds is not None and time.time() - start_time >= max_runtime_seconds:
            stopped_reason = "max_runtime"
            print(f"reached max runtime at step={iteration}; saving checkpoint")
            break

    save_checkpoint(raw_model, optimizer, last_iteration, args.checkpoint_path)
    logger.log("finished", last_iteration, stopped_reason=stopped_reason, checkpoint_path=args.checkpoint_path)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
