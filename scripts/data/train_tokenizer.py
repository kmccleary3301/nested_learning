#!/usr/bin/env python
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm
import typer
import yaml
from datasets import load_dataset

app = typer.Typer(add_completion=False, help="Train a SentencePiece tokenizer from HF datasets.")


@dataclass
class DatasetSpec:
    name: str
    dataset: str
    split: str = "train"
    subset: str | None = None
    text_column: str = "text"
    sample_limit: int = 100_000


def _load_specs_from_manifest(manifest: Path) -> List[DatasetSpec]:
    data = yaml.safe_load(manifest.read_text())
    entries = data.get("datasets", data)
    specs = []
    for entry in entries:
        specs.append(
            DatasetSpec(
                name=entry.get("name") or entry["dataset"].split("/")[-1],
                dataset=entry["dataset"],
                split=entry.get("split", "train"),
                subset=entry.get("subset"),
                text_column=entry.get("text_column", "text"),
                sample_limit=entry.get("sample_limit", 100_000),
            )
        )
    return specs


def _write_samples(spec: DatasetSpec, handle) -> int:
    ds = load_dataset(spec.dataset, spec.subset, split=spec.split, streaming=True)
    count = 0
    for row in ds:
        text = row.get(spec.text_column)
        if not isinstance(text, str):
            continue
        handle.write(text.replace("\n", " ") + "\n")
        count += 1
        if spec.sample_limit > 0 and count >= spec.sample_limit:
            break
    return count


@app.command()
def main(
    dataset: str = typer.Option("roneneldan/TinyStories", help="HF dataset name (ignored if manifest set)."),
    split: str = typer.Option("train", help="Dataset split (ignored if manifest set)."),
    text_column: str = typer.Option("text", help="Text column (ignored if manifest set)."),
    sample_limit: int = typer.Option(100_000, help="Sample limit per dataset (ignored if manifest set)."),
    vocab_size: int = typer.Option(32_000, help="SentencePiece vocabulary size."),
    model_type: str = typer.Option("unigram", help="SentencePiece model type."),
    character_coverage: float = typer.Option(0.9995, help="Character coverage target."),
    output_dir: Path = typer.Option(Path("artifacts/tokenizer"), help="Directory for tokenizer artifacts."),
    manifest: Optional[Path] = typer.Option(None, help="YAML manifest describing multiple datasets."),
    log_file: Optional[Path] = typer.Option(
        Path("data/mixtures/tokenizer_samples.json"), help="Where to log dataset sample stats."
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        _load_specs_from_manifest(manifest)
        if manifest is not None
        else [
            DatasetSpec(
                name=dataset.split("/")[-1],
                dataset=dataset,
                split=split,
                text_column=text_column,
                sample_limit=sample_limit,
            )
        ]
    )
    model_prefix = output_dir / f"spm_{vocab_size}_{model_type}"
    stats = []
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        typer.echo(f"[Tokenizer] Writing samples to {tmp_path}")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for spec in specs:
                typer.echo(f"[Tokenizer] Streaming {spec.name} ({spec.dataset}) limit={spec.sample_limit}")
                count = _write_samples(spec, handle)
                stats.append({"name": spec.name, "dataset": spec.dataset, "samples": count})
    typer.echo(f"[Tokenizer] Training SentencePiece -> {model_prefix}")
    spm.SentencePieceTrainer.train(
        input=str(tmp_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        input_sentence_size=sum(s["samples"] for s in stats),
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )
    typer.echo(f"[Tokenizer] Saved model to {model_prefix}.model")
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_payload = {"model": str(model_prefix), "datasets": stats}
        log_file.write_text(json.dumps(log_payload, indent=2))
        typer.echo(f"[Tokenizer] Logged sample stats to {log_file}")


if __name__ == "__main__":
    app()
