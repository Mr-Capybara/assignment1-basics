from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import generate_text


def parse_args() -> argparse.Namespace:
    """解析从 checkpoint 生成文本所需的参数。"""

    parser = argparse.ArgumentParser(description="Generate text from a trained CS336 Transformer LM.")

    io_group = parser.add_argument_group("checkpoint and tokenizer")
    io_group.add_argument("--checkpoint-path", type=Path, required=True, help="训练脚本保存的 checkpoint。")
    io_group.add_argument(
        "--bpe-pickle",
        type=Path,
        default=None,
        help="推荐：train_bpe_tinystories.py 保存的 bpe.pkl，能无损保留 bytes vocab/merges。",
    )
    io_group.add_argument("--vocab-path", type=Path, default=None, help="可选：BPE vocab.json 或 GPT-2 vocab.json。")
    io_group.add_argument("--merges-path", type=Path, default=None, help="可选：BPE merges.txt。")
    io_group.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    io_group.add_argument("--output-path", type=Path, default=None, help="可选：把生成文本写入文件。")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--vocab-size", type=int, default=None, help="默认从 tokenizer vocab 推断。")
    model_group.add_argument("--context-length", type=int, default=256)
    model_group.add_argument("--d-model", type=int, default=512)
    model_group.add_argument("--num-layers", type=int, default=4)
    model_group.add_argument("--num-heads", type=int, default=16)
    model_group.add_argument("--d-ff", type=int, default=None)
    model_group.add_argument("--rope-theta", type=float, default=10000.0)

    # 这些开关必须和训练 checkpoint 时的结构保持一致，否则 state_dict 维度或 key 会对不上。
    ablation_group = parser.add_argument_group("architecture switches")
    ablation_group.add_argument("--norm-position", choices=("pre", "post"), default="pre")
    ablation_group.add_argument("--no-rmsnorm", action="store_true")
    ablation_group.add_argument("--no-rope", action="store_true")
    ablation_group.add_argument("--ffn-type", choices=("swiglu", "silu"), default="swiglu")
    ablation_group.add_argument("--tie-embeddings", action="store_true")

    gen_group = parser.add_argument_group("generation")
    gen_group.add_argument("--prompt", type=str, default="Once upon a time,")
    gen_group.add_argument("--max-new-tokens", type=int, default=256)
    gen_group.add_argument("--temperature", type=float, default=0.8)
    gen_group.add_argument("--top-p", type=float, default=0.9)
    gen_group.add_argument("--end-token", type=str, default="<|endoftext|>")
    gen_group.add_argument("--seed", type=int, default=0)
    gen_group.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_d_ff(args: argparse.Namespace) -> int:
    """和 train_lm.py 保持一致的 d_ff 默认规则。"""

    if args.d_ff is not None:
        return args.d_ff
    if args.ffn_type == "silu":
        return 4 * args.d_model
    rounded_down = int((8 * args.d_model / 3) // 64) * 64
    return max(64, rounded_down)


def build_model(args: argparse.Namespace, vocab_size: int, device: torch.device) -> TransformerLM:
    """创建与训练时结构一致的模型骨架，再加载 checkpoint 权重。"""

    return TransformerLM(
        vocab_size=vocab_size,
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


def load_tokenizer(args: argparse.Namespace) -> Tokenizer:
    """加载 tokenizer。

    优先使用 bpe.pkl：它直接保存 Python bytes，不会丢失空格、换行等 token。
    vocab.json + merges.txt 主要用于兼容 GPT-2 风格文件；如果 merges.txt 来自本仓库
    的 latin-1 可读格式，含空格 token 的 merge 行在文本上可能有歧义。
    """

    if args.bpe_pickle is not None:
        with args.bpe_pickle.open("rb") as file:
            artifact = pickle.load(file)
        return Tokenizer(artifact["vocab"], artifact["merges"], special_tokens=args.special_tokens)

    if args.vocab_path is None or args.merges_path is None:
        raise ValueError("Please provide either --bpe-pickle or both --vocab-path and --merges-path.")
    return Tokenizer.from_files(args.vocab_path, args.merges_path, special_tokens=args.special_tokens)


def main() -> None:
    args = parse_args()
    args.d_ff = resolve_d_ff(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    tokenizer = load_tokenizer(args)
    vocab_size = args.vocab_size or len(tokenizer.vocab)

    model = build_model(args, vocab_size=vocab_size, device=device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    # generate_text 内部会把模型切到 eval 模式，并用 @torch.no_grad() 禁用梯度记录。
    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        end_token=args.end_token,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    print(text)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
