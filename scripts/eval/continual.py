#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
import typer
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nested_learning.data import TokenShardDataset, collate_batch
from nested_learning.training import build_model_from_cfg

app = typer.Typer(add_completion=False, help="Continual learning evaluation harness.")


def load_segments(yaml_path: Path) -> List[Dict[str, str]]:
    payload = yaml.safe_load(yaml_path.read_text())
    return payload.get("segments", [])


def evaluate_segment(model, dataloader: DataLoader, device: torch.device, max_batches: int | None) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for batch in dataloader:
        tokens = batch.to(device)
        with torch.no_grad():
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                reduction="sum",
            )
        total_loss += loss.item()
        total_tokens += tokens[:, 1:].numel()
        batches += 1
        if max_batches and batches >= max_batches:
            break
    return total_loss / total_tokens if total_tokens > 0 else float("nan")


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config for HOPE."),
    checkpoints: List[Path] = typer.Option(..., help="Ordered list of checkpoints (chronological)."),
    segments_yaml: Path = typer.Option(..., help="YAML describing shard directories per segment."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece model path (unused for now)."),
    batch_size: int = typer.Option(4, help="Batch size for evaluation."),
    max_batches: int = typer.Option(50, help="Max batches per segment (0 = entire dataset)."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/continual_results.json")),
) -> None:
    segments = load_segments(segments_yaml)
    if not segments:
        raise typer.BadParameter("No segments found in YAML.")

    cfg = OmegaConf.load(config)
    device_obj = torch.device(device)
    results = []

    for step_idx, ckpt_path in enumerate(checkpoints):
        state = torch.load(ckpt_path, map_location="cpu")
        model = build_model_from_cfg(cfg.model)
        model.load_state_dict(state["model"] if "model" in state else state)
        model = model.to(device_obj)

        segment_losses = {}
        for segment in segments:
            name = segment["name"]
            shards_dir = Path(segment["shards_dir"])
            dataset = TokenShardDataset(shards_dir)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
            loss = evaluate_segment(model, loader, device_obj, None if max_batches <= 0 else max_batches)
            segment_losses[name] = loss

        results.append({"checkpoint": str(ckpt_path), "segment_losses": segment_losses})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    typer.echo(f"[Continual] Saved results to {output}")


if __name__ == "__main__":
    app()
