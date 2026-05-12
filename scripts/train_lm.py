from __future__ import annotations

import argparse
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Train a CS336 Transformer language model.")

    parser.add_argument("--train-data", type=Path, required=True, help="Path to a .npy array of train token IDs.")
    parser.add_argument("--valid-data", type=Path, required=True, help="Path to a .npy array of valid token IDs.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Where to save checkpoints.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Optional checkpoint to resume from.")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--lr-max", type=float, default=3e-4)
    parser.add_argument("--lr-min", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--wandb-project", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    """用若干随机 batch 估计验证 loss，避免一次性扫完整验证集。"""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        losses.append(cross_entropy(model(x), y).item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # mmap_mode="r" 让 numpy 按需从磁盘读取 token，适合大规模 tokenized 数据。
    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        print(f"resumed from {args.resume_from} at step {start_step}")

    wandb_run = None
    if args.wandb_project is not None:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    model.train()
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

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        if args.max_grad_norm > 0:
            gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        iteration = step + 1
        if iteration % args.log_every == 0:
            elapsed = time.time() - start_time
            print(f"step={iteration} train_loss={loss.item():.4f} lr={lr:.3e} elapsed={elapsed:.1f}s")
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item(), "lr": lr, "step": iteration})

        if iteration % args.eval_every == 0:
            valid_loss = estimate_loss(
                model=model,
                dataset=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=str(device),
                eval_iters=args.eval_iters,
            )
            print(f"step={iteration} valid_loss={valid_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"valid/loss": valid_loss, "step": iteration})

        if iteration % args.save_every == 0:
            save_checkpoint(model, optimizer, iteration, args.checkpoint_path)
            print(f"saved checkpoint to {args.checkpoint_path}")

    save_checkpoint(model, optimizer, args.total_steps, args.checkpoint_path)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
