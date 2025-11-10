#!/usr/bin/env python
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import typer
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nested_learning.data import TokenShardDataset, collate_batch
from nested_learning.training import build_model_from_cfg

app = typer.Typer(add_completion=False, help="Continual-learning evaluation harness (forgetting curves).")


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    cfg = OmegaConf.load(config_path)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    return model.to(device).eval()


def evaluate_shard_dir(
    model,
    shard_dir: str,
    device: torch.device,
    batch_size: int,
    max_batches: Optional[int],
) -> float:
    dataset = TokenShardDataset(shard_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    total_loss = 0.0
    total_tokens = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        tokens = batch.to(device)
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += tokens[:, 1:].numel()
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config."),
    checkpoints: List[Path] = typer.Option(..., help="Ordered checkpoints (one per stage)."),
    segments_yaml: Path = typer.Option(..., help="YAML describing evaluation segments with shard dirs."),
    batch_size: int = typer.Option(8, help="Evaluation batch size."),
    max_batches: int = typer.Option(50, help="Max batches per segment (0 = entire dataset)."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu", help="Device."),
    output: Path = typer.Option(Path("eval/continual_results.json"), help="Output JSON with metrics."),
) -> None:
    segment_cfg = yaml.safe_load(segments_yaml.read_text())
    segments = segment_cfg["segments"]
    torch_device = torch.device(device)

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    best_loss: Dict[str, float] = {seg["name"]: float("inf") for seg in segments}

    for ckpt_path in checkpoints:
        ckpt_name = ckpt_path.stem
        model = load_model(config, ckpt_path, torch_device)
        ckpt_metrics: Dict[str, Dict[str, float]] = {}
        for seg in segments:
            seg_name = seg["name"]
            shard_dir = seg["shards_dir"]
            batches = seg.get("max_batches", max_batches)
            loss = evaluate_shard_dir(model, shard_dir, torch_device, batch_size, None if batches == 0 else batches)
            ppl = math.exp(loss) if loss == loss else float("nan")
            ckpt_metrics[seg_name] = {"loss": loss, "ppl": ppl}
            best_loss[seg_name] = min(best_loss[seg_name], loss)
        all_results[ckpt_name] = ckpt_metrics

    final_ckpt = checkpoints[-1].stem
    forgetting = {
        seg["name"]: all_results[final_ckpt][seg["name"]]["loss"] - best_loss[seg["name"]]
        for seg in segments
    }

    payload = {
        "checkpoints": [str(path) for path in checkpoints],
        "segments": segments,
        "metrics": all_results,
        "forgetting": forgetting,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    typer.echo(f"[Continual] Saved metrics to {output}")


if __name__ == "__main__":
    app()
