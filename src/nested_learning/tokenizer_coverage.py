from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict

from .tokenizer import SentencePieceTokenizer


def compute_tokenizer_coverage_stats(
    tokenizer_path: Path,
    sample_file: Path,
    max_lines: int = 10_000,
) -> Dict[str, object]:
    """
    Compute tokenizer coverage statistics on a representative text sample.

    Returns a JSON-serialisable dictionary; shared by both the coverage CLI and
    the regression guard so they cannot drift apart silently.
    """

    tokenizer = SentencePieceTokenizer(tokenizer_path)
    total_words = 0
    total_tokens = 0
    total_chars = 0
    processed_lines = 0
    word_token_lengths: list[int] = []
    piece_lengths = Counter()

    with sample_file.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_lines:
                break
            stripped = line.strip()
            if not stripped:
                continue
            processed_lines += 1
            total_chars += len(stripped)
            words = stripped.split()
            if not words:
                continue
            total_words += len(words)
            encoded = tokenizer.encode(stripped, add_bos=False, add_eos=False)
            ids = encoded.tolist()
            total_tokens += len(ids)
            for word in words:
                word_tokens = tokenizer.encode(word, add_bos=False, add_eos=False).tolist()
                if not word_tokens:
                    continue
                word_token_lengths.append(len(word_tokens))
            for token_id in ids:
                piece = tokenizer.processor.id_to_piece(token_id)
                piece_lengths[len(piece)] += 1

    if total_words == 0 or not word_token_lengths:
        raise ValueError("Sample produced no words; double-check the sample_file path.")

    avg_tokens_per_word = total_tokens / total_words if total_words else 0.0
    pct_single_token = sum(1 for length in word_token_lengths if length == 1) / len(word_token_lengths)
    pct_two_or_less = sum(1 for length in word_token_lengths if length <= 2) / len(word_token_lengths)

    return {
        "tokenizer": str(tokenizer_path),
        "sample_file": str(sample_file),
        "lines_processed": processed_lines,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "avg_tokens_per_word": avg_tokens_per_word,
        "pct_single_token_words": pct_single_token,
        "pct_two_or_less_tokens_words": pct_two_or_less,
        "avg_chars_per_word": total_chars / total_words,
        "piece_length_histogram": dict(piece_lengths.most_common(20)),
    }
