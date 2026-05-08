"""在 TinyStories 数据集上训练 BPE 分词器。

用法：
    uv run python scripts/train_bpe_tinystories.py \
        --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 \
        --output-dir artifacts/tinystories_bpe
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

from cs336_basics.tokenizer import train_bpe


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("train_bpe")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on TinyStories.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
        help="输入语料路径（TinyStories 训练集）",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="目标词表大小（讲义推荐 10000）",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="+",
        default=["<|endoftext|>"],
        help="特殊 token 列表",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tinystories_bpe"),
        help="输出目录（保存 vocab.json / merges.txt / bpe.pkl）",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="预分词并行进程数，默认取 os.cpu_count()",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭 tqdm 进度条（适合重定向日志时使用）",
    )
    return parser.parse_args()


def save_artifacts(
    output_dir: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    logger: logging.Logger,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 原生 pickle：保留 bytes，最适合后续 Tokenizer 直接加载
    pkl_path = output_dir / "bpe.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    logger.info("已保存 pickle: %s", pkl_path)

    # 2) 人类可读的 JSON vocab（id -> latin-1 字符串，保留全部字节）
    vocab_json_path = output_dir / "vocab.json"
    with vocab_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {str(token_id): token_bytes.decode("latin-1") for token_id, token_bytes in vocab.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("已保存词表: %s (size=%d)", vocab_json_path, len(vocab))

    # 3) merges.txt（每行 "left right"，latin-1 编码以保留原始字节）
    merges_path = output_dir / "merges.txt"
    with merges_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left.decode('latin-1')} {right.decode('latin-1')}\n")
    logger.info("已保存 merges: %s (count=%d)", merges_path, len(merges))


def main() -> None:
    args = parse_args()
    logger = setup_logger()

    if not args.input.exists():
        raise FileNotFoundError(
            f"找不到输入语料：{args.input}\n"
            "请先按 README 下载 TinyStories 到 data/ 目录。"
        )

    file_size_mb = args.input.stat().st_size / (1024 * 1024)
    logger.info("输入文件: %s (%.1f MB)", args.input, file_size_mb)
    logger.info("目标词表大小: %d", args.vocab_size)
    logger.info("特殊 token: %s", args.special_tokens)

    logger.info("开始训练 BPE ...")
    start = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
        show_progress=not args.no_progress,
    )
    elapsed = time.perf_counter() - start
    logger.info("训练完成，用时 %.1f s (%.2f min)", elapsed, elapsed / 60)
    logger.info("最终词表大小: %d, merges 条数: %d", len(vocab), len(merges))

    # 简单报告：最长的 10 个 token（通常是高频子词）
    longest = sorted(vocab.values(), key=len, reverse=True)[:10]
    logger.info("最长 10 个 token: %s", [t.decode("utf-8", errors="replace") for t in longest])

    save_artifacts(args.output_dir, vocab, merges, logger)
    logger.info("全部完成 ✅")


if __name__ == "__main__":
    main()
