#!/usr/bin/env python
from __future__ import annotations

from collections import deque
import os
from pathlib import Path
from typing import Optional

import typer
from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect_langs
from tqdm import tqdm

DetectorFactory.seed = 0

app = typer.Typer(add_completion=False, help="Filter datasets by language/length and deduplicate lines.")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def is_target_language(text: str, target_lang: str, threshold: float) -> bool:
    try:
        langs = detect_langs(text)
    except LangDetectException:
        return False
    return any(lang.lang == target_lang and lang.prob >= threshold for lang in langs)


@app.command()
def main(
    dataset: str = typer.Option(..., help="HF dataset name, e.g. HuggingFaceFW/fineweb"),
    subset: Optional[str] = typer.Option(None, help="Optional dataset subset/config name."),
    split: str = typer.Option("train", help="Dataset split."),
    text_column: str = typer.Option("text", help="Column containing text."),
    target_lang: str = typer.Option("en", help="Language code to keep."),
    lang_threshold: float = typer.Option(0.80, help="Minimum probability for language detection."),
    min_chars: int = typer.Option(200, help="Minimum character count."),
    max_chars: int = typer.Option(10000, help="Maximum character count."),
    output_path: Path = typer.Option(Path("data/filtered/output.jsonl"), help="Destination JSONL file."),
    dedup_window: int = typer.Option(50000, help="Number of recent hashes to retain for deduplication."),
    limit: Optional[int] = typer.Option(None, help="Optional limit on records processed."),
    streaming: bool = typer.Option(True, help="Use HF streaming mode."),
    data_files: Optional[str] = typer.Option(None, help="Optional data_files argument (e.g., local text file)."),
    force_exit: bool = typer.Option(False, help="Force os._exit(0) to avoid async finalization issues."),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    load_kwargs = {}
    if data_files is not None:
        load_kwargs["data_files"] = data_files
    dataset_obj = load_dataset(dataset, subset, split=split, streaming=streaming, **load_kwargs)
    iterator = dataset_obj if streaming else iter(dataset_obj)
    seen_hashes = set()
    hash_queue = deque()
    kept = 0
    total = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for row in tqdm(iterator, desc="Filtering dataset"):
            total += 1
            text = row.get(text_column)
            if not isinstance(text, str):
                continue
            normalized = normalize_text(text)
            if len(normalized) < min_chars or len(normalized) > max_chars:
                continue
            if not is_target_language(normalized, target_lang, lang_threshold):
                continue
            hashed = hash(normalized)
            if hashed in seen_hashes:
                continue
            writer.write(normalized + "\n")
            kept += 1
            seen_hashes.add(hashed)
            hash_queue.append(hashed)
            if len(hash_queue) > dedup_window:
                old_hash = hash_queue.popleft()
                seen_hashes.discard(old_hash)
            if limit and kept >= limit:
                break
    typer.echo(f"[Filter] Processed={total} kept={kept} -> {output_path}")
    if force_exit:
        os._exit(0)


if __name__ == "__main__":
    app()
