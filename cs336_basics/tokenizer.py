from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from functools import lru_cache
from os import PathLike
from pathlib import Path

import regex as re


GPT2_PRETOKENIZE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 reversible byte-to-unicode display mapping for serialized vocab files."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs], strict=True))


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in self.token_to_id:
                token_id = len(self.vocab)
                self.vocab[token_id] = special_token_bytes
                self.token_to_id[special_token_bytes] = token_id

        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.pretokenize_regex = re.compile(GPT2_PRETOKENIZE_PATTERN)
        self.special_token_regex = self._compile_special_token_regex(self.special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        byte_decoder = {value: key for key, value in _bytes_to_unicode().items()}

        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            serialized_vocab = json.load(vocab_file)

        if all(isinstance(value, int) for value in serialized_vocab.values()):
            vocab = {
                token_id: bytes(byte_decoder[char] for char in token)
                for token, token_id in serialized_vocab.items()
            }
        else:
            vocab = {int(token_id): bytes(token_bytes) for token_id, token_bytes in serialized_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                parts = line.rstrip("\n").split(" ")
                if len(parts) != 2:
                    continue
                left, right = parts
                merges.append(
                    (
                        bytes(byte_decoder[char] for char in left),
                        bytes(byte_decoder[char] for char in right),
                    )
                )

        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _compile_special_token_regex(special_tokens: list[str]) -> re.Pattern[str] | None:
        if not special_tokens:
            return None
        alternatives = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
        return re.compile("|".join(alternatives))

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for piece, is_special in self._split_special_tokens(text):
            if is_special:
                token_ids.append(self.token_to_id[piece.encode("utf-8")])
            else:
                token_ids.extend(self._encode_ordinary_text(piece))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            stable_prefix_length = self._stable_prefix_length(buffer)
            if stable_prefix_length == 0:
                continue

            yield from self.encode(buffer[:stable_prefix_length])
            buffer = buffer[stable_prefix_length:]

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def _split_special_tokens(self, text: str) -> Iterator[tuple[str, bool]]:
        if self.special_token_regex is None:
            if text:
                yield text, False
            return

        start = 0
        for match in self.special_token_regex.finditer(text):
            if match.start() > start:
                yield text[start : match.start()], False
            yield match.group(0), True
            start = match.end()

        if start < len(text):
            yield text[start:], False

    def _encode_ordinary_text(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for match in self.pretokenize_regex.finditer(text):
            token_bytes = match.group(0).encode("utf-8")
            for bpe_token in self._bpe(token_bytes):
                token_ids.append(self.token_to_id[bpe_token])
        return token_ids

    @staticmethod
    def _stable_prefix_length(text: str) -> int:
        """Return a prefix length that avoids splitting GPT-2 pre-tokens across chunks."""
        for i in range(len(text) - 1, -1, -1):
            if text[i].isspace():
                start = i
                while start > 0 and text[start - 1].isspace():
                    start -= 1
                return start
        return 0

    @lru_cache(maxsize=100_000)
    def _bpe(self, token: bytes) -> tuple[bytes, ...]:
        parts = tuple(bytes([byte]) for byte in token)
        if len(parts) < 2:
            return parts

        while True:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None
            for pair in zip(parts, parts[1:], strict=False):
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                return parts

            parts = self._merge_pair(parts, best_pair)

    @staticmethod
    def _merge_pair(parts: tuple[bytes, ...], pair_to_merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
        merged: list[bytes] = []
        i = 0
        while i < len(parts):
            if i < len(parts) - 1 and parts[i] == pair_to_merge[0] and parts[i + 1] == pair_to_merge[1]:
                merged.append(parts[i] + parts[i + 1])
                i += 2
            else:
                merged.append(parts[i])
                i += 1
        return tuple(merged)


def train_bpe(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = _initial_vocab(special_tokens)
    if vocab_size <= len(vocab):
        return {token_id: vocab[token_id] for token_id in range(vocab_size)}, []

    pretoken_counts = _count_pretokens(input_path, special_tokens)
    words = {word_id: pretoken for word_id, pretoken in enumerate(pretoken_counts)}
    word_counts = {word_id: count for word_id, count in enumerate(pretoken_counts.values())}
    pair_counts, pair_to_word_ids = _initialize_pair_statistics(words, word_counts)
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        affected_word_ids = list(pair_to_word_ids[best_pair])
        for word_id in affected_word_ids:
            old_word = words[word_id]
            count = word_counts[word_id]
            _remove_word_pairs(old_word, count, word_id, pair_counts, pair_to_word_ids)

            new_word = Tokenizer._merge_pair(old_word, best_pair)
            words[word_id] = new_word
            _add_word_pairs(new_word, count, word_id, pair_counts, pair_to_word_ids)

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

    return vocab, merges


def _initial_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    for special_token in special_tokens:
        special_token_bytes = special_token.encode("utf-8")
        if special_token_bytes not in vocab.values():
            vocab[len(vocab)] = special_token_bytes

    for byte in range(256):
        byte_token = bytes([byte])
        if byte_token not in vocab.values():
            vocab[len(vocab)] = byte_token

    return vocab


def _count_pretokens(
    input_path: str | PathLike[str],
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    pretokenizer = re.compile(GPT2_PRETOKENIZE_PATTERN)
    text = Path(input_path).read_text(encoding="utf-8")
    pretoken_counts: Counter[tuple[bytes, ...]] = Counter()

    for segment in _split_on_special_tokens(text, special_tokens):
        for match in pretokenizer.finditer(segment):
            token_bytes = match.group(0).encode("utf-8")
            if token_bytes:
                pretoken_counts[tuple(bytes([byte]) for byte in token_bytes)] += 1

    return dict(pretoken_counts)


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    alternatives = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    return re.split("|".join(alternatives), text)


def _initialize_pair_statistics(
    words: dict[int, tuple[bytes, ...]],
    word_counts: dict[int, int],
) -> tuple[Counter[tuple[bytes, bytes]], defaultdict[tuple[bytes, bytes], set[int]]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_word_ids: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for word_id, word in words.items():
        _add_word_pairs(word, word_counts[word_id], word_id, pair_counts, pair_to_word_ids)
    return pair_counts, pair_to_word_ids


def _add_word_pairs(
    word: tuple[bytes, ...],
    count: int,
    word_id: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_word_ids: defaultdict[tuple[bytes, bytes], set[int]],
) -> None:
    seen_pairs: set[tuple[bytes, bytes]] = set()
    for pair in zip(word, word[1:], strict=False):
        pair_counts[pair] += count
        if pair not in seen_pairs:
            pair_to_word_ids[pair].add(word_id)
            seen_pairs.add(pair)


def _remove_word_pairs(
    word: tuple[bytes, ...],
    count: int,
    word_id: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair_to_word_ids: defaultdict[tuple[bytes, bytes], set[int]],
) -> None:
    seen_pairs: set[tuple[bytes, bytes]] = set()
    for pair in zip(word, word[1:], strict=False):
        pair_counts[pair] -= count
        if pair_counts[pair] == 0:
            del pair_counts[pair]

        if pair not in seen_pairs:
            pair_to_word_ids[pair].discard(word_id)
            if not pair_to_word_ids[pair]:
                del pair_to_word_ids[pair]
            seen_pairs.add(pair)
