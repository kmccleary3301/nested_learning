#!/usr/bin/env python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import sentencepiece as spm
import typer
from datasets import load_dataset
from tqdm import tqdm

app = typer.Typer(add_completion=False, help="Shard datasets into tokenized numpy binaries.")


@dataclass
class ShardConfig:
    name: str
    dataset: str
    split: str = "train"
    subset: str | None = None
    text_column: str = "text"
    tokenizer_path: Path = Path()
    seq_len: int = 2048
    sequences_per_shard: int = 1024
    output_dir: Path = Path("data/shards")
    eos_id: int = -1
    max_records: Optional[int] = None
    data_files: Optional[str] = None


def shard_dataset(config: ShardConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    processor = spm.SentencePieceProcessor(model_file=str(config.tokenizer_path))
    eos = config.eos_id if config.eos_id >= 0 else processor.eos_id()
    load_kwargs = {}
    if config.data_files is not None:
        load_kwargs["data_files"] = config.data_files
    ds = load_dataset(config.dataset, config.subset, split=config.split, streaming=True, **load_kwargs)

    buffer: List[int] = []
    sequences: List[List[int]] = []
    shard_idx = 0
    records = 0
    sequences_total = 0
    tokens_total = 0

    for row in tqdm(ds, desc=f"Sharding {config.name}", unit="record"):
        text = row.get(config.text_column)
        if not isinstance(text, str):
            continue
        tokens = processor.encode(text)
        tokens.append(eos)
        tokens_total += len(tokens)
        buffer.extend(tokens)
        records += 1
        while len(buffer) >= config.seq_len:
            seq = buffer[: config.seq_len]
            buffer = buffer[config.seq_len :]
            sequences.append(seq)
            sequences_total += 1
            if len(sequences) >= config.sequences_per_shard:
                _write_shard(sequences, config.output_dir, shard_idx)
                shard_idx += 1
                sequences = []
        if config.max_records and records >= config.max_records:
            break
    if sequences:
        _write_shard(sequences, config.output_dir, shard_idx)
        shard_idx += 1

    stats = {
        "name": config.name,
        "dataset": config.dataset,
        "subset": config.subset,
        "records": records,
        "sequences": sequences_total,
        "tokens": tokens_total,
        "shards": shard_idx,
        "output_dir": str(config.output_dir),
    }
    typer.echo(
        f"[Shard] {config.name}: records={records} sequences={sequences_total} shards={shard_idx} -> {config.output_dir}"
    )
    return stats


@app.command()
def main(
    dataset: str = typer.Option("roneneldan/TinyStories", help="HF dataset name."),
    split: str = typer.Option("train", help="Dataset split."),
    subset: Optional[str] = typer.Option(None, help="Optional dataset subset/config."),
    text_column: str = typer.Option("text", help="Text column."),
    tokenizer_path: Path = typer.Option(..., help="Path to SentencePiece model."),
    seq_len: int = typer.Option(2048, help="Sequence length (tokens per sample)."),
    sequences_per_shard: int = typer.Option(1024, help="Number of sequences per shard."),
    output_dir: Path = typer.Option(Path("data/shards"), help="Directory for shard files."),
    eos_id: int = typer.Option(-1, help="EOS token id (defaults to tokenizer default)."),
    max_records: Optional[int] = typer.Option(None, help="Optional max records to process."),
    name: Optional[str] = typer.Option(None, help="Friendly name for logging."),
    log_file: Optional[Path] = typer.Option(
        Path("data/mixtures/shard_stats.json"), help="Where to save shard stats JSON."
    ),
    data_files: Optional[str] = typer.Option(None, help="Optional data_files argument."),
) -> None:
    config = ShardConfig(
        name=name or dataset.split("/")[-1],
        dataset=dataset,
        split=split,
        subset=subset,
        text_column=text_column,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        sequences_per_shard=sequences_per_shard,
        output_dir=output_dir,
        eos_id=eos_id,
        max_records=max_records,
        data_files=data_files,
    )
    stats = shard_dataset(config)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(json.dumps(stats, indent=2))
        typer.echo(f"[Shard] Stats logged to {log_file}")


def _write_shard(sequences: List[List[int]], output_dir: Path, shard_idx: int) -> None:
    array = np.asarray(sequences, dtype=np.int32)
    target = output_dir / f"shard_{shard_idx:05d}.npy"
    np.save(target, array)


if __name__ == "__main__":
    app()
