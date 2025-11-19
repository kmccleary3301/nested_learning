#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import torch
import typer
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nested_learning.data import TokenShardDataset, collate_batch
from nested_learning.memorize import (
    MemorizeConfig,
    memorize_tokens,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(add_completion=False, help="Continual learning evaluation harness.")


def load_segments(yaml_path: Path) -> List[Dict[str, str]]:
    payload = yaml.safe_load(yaml_path.read_text())
    return payload.get("segments", [])


def evaluate_segment(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    memorize_cfg: MemorizeConfig,
) -> tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss_base = 0.0
    total_loss_mem = 0.0
    total_tokens = 0
    batches = 0
    path_stats: Dict[str, float] = defaultdict(float)
    base_state: Dict[str, torch.Tensor] | None = None
    if memorize_cfg.enabled and memorize_cfg.reset:
        base_state = snapshot_state_dict(model)
    for batch in dataloader:
        tokens = batch.to(device)
        with torch.no_grad():
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                reduction="sum",
            )
        total_loss_base += loss.item()
        token_count = tokens[:, 1:].numel()
        if memorize_cfg.enabled:
            stats = memorize_tokens(model, tokens, memorize_cfg)
            for key, value in stats.items():
                path_stats[key] += value
            with torch.no_grad():
                logits_mem = model(tokens)
                loss_mem = torch.nn.functional.cross_entropy(
                    logits_mem[:, :-1].reshape(-1, logits_mem.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    reduction="sum",
                )
            total_loss_mem += loss_mem.item()
            if memorize_cfg.reset and base_state is not None:
                restore_state_dict(model, base_state)
        else:
            total_loss_mem += loss.item()
        total_tokens += tokens[:, 1:].numel()
        batches += 1
        if max_batches and batches >= max_batches:
            break
    base_ce = total_loss_base / total_tokens if total_tokens > 0 else float("nan")
    mem_ce = total_loss_mem / total_tokens if total_tokens > 0 else float("nan")
    return base_ce, mem_ce, path_stats


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
    memorize: bool = typer.Option(False, help="Enable memorization while evaluating segments."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per batch."),
    memorize_no_reset: bool = typer.Option(True, help="Keep memory between segments by default."),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm needed to memorize a batch."
    ),
    memorize_paths: str = typer.Option(
        "all",
        help="Comma-separated memory paths to update (e.g., 'titan,cms_fast'); use 'all' for default behavior.",
    ),
) -> None:
    segments = load_segments(segments_yaml)
    if not segments:
        raise typer.BadParameter("No segments found in YAML.")

    cfg = OmegaConf.load(config)
    cfg = unwrap_config(cfg)
    device_obj = torch.device(device)
    results = []

    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=False,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
    )

    for step_idx, ckpt_path in enumerate(checkpoints):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model_from_cfg(cfg.model)
        state_dict = state["model"] if "model" in state else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                "[continual] Warning: state_dict mismatch "
                f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
            )
        model = model.to(device_obj)

        segment_losses = {}
        baseline_losses = {}
        segment_stats = {}
        for segment in segments:
            name = segment["name"]
            shards_dir = Path(segment["shards_dir"])
            dataset = TokenShardDataset(shards_dir)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
            base_loss, mem_loss, stats = evaluate_segment(
                model,
                loader,
                device_obj,
                None if max_batches <= 0 else max_batches,
                memorize_cfg,
            )
            baseline_losses[name] = base_loss
            segment_losses[name] = mem_loss
            if stats:
                segment_stats[name] = stats

        entry = {"checkpoint": str(ckpt_path), "segment_losses": segment_losses}
        if memorize_cfg.enabled:
            entry["segment_baseline_losses"] = baseline_losses
            entry["segment_memorize_delta"] = {
                name: baseline_losses[name] - segment_losses[name] for name in segment_losses
            }
            if segment_stats:
                entry["memorize_stats"] = segment_stats
        results.append(entry)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    typer.echo(f"[Continual] Saved results to {output}")


if __name__ == "__main__":
    app()
