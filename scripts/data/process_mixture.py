#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import typer
import yaml

from shard_corpus import ShardConfig, shard_dataset

app = typer.Typer(add_completion=False, help="Process a dataset manifest to shard multiple corpora.")


@app.command()
def main(
    manifest: Path = typer.Argument(..., help="YAML manifest describing datasets."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece model to tokenize with."),
    log_file: Path = typer.Option(Path("data/mixtures/mixture_stats.json"), help="Output stats JSON."),
) -> None:
    data = yaml.safe_load(manifest.read_text())
    datasets = data.get("datasets", data)
    stats: List[Dict[str, Any]] = []
    for entry in datasets:
        name = entry["name"]
        config = ShardConfig(
            name=name,
            dataset=entry["dataset"],
            split=entry.get("split", "train"),
            subset=entry.get("subset"),
            text_column=entry.get("text_column", "text"),
            tokenizer_path=tokenizer_path,
            seq_len=entry.get("seq_len", 2048),
            sequences_per_shard=entry.get("sequences_per_shard", 1024),
            output_dir=Path(entry.get("output_dir", f"data/shards/{name}")),
            eos_id=entry.get("eos_id", -1),
            max_records=entry.get("max_records"),
            data_files=entry.get("data_files"),
        )
        stats.append(shard_dataset(config))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps({"manifest": str(manifest), "stats": stats}, indent=2))
    typer.echo(f"[Mixture] Logged stats for {len(stats)} datasets -> {log_file}")


if __name__ == "__main__":
    app()
