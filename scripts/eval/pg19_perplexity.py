#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from datasets import load_dataset
from omegaconf import OmegaConf

from nested_learning.memorize import MemorizeConfig
from nested_learning.memorize import memorize_sequence, restore_state_dict, snapshot_state_dict
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(add_completion=False, help="Compute PG-19 perplexity for a checkpoint.")


def load_model(config: Path, checkpoint: Path, device: torch.device):
    cfg = OmegaConf.load(config)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[pg19] Warning: mismatch missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def _nll_for_text(model, tokenizer, text: str, device: torch.device, max_seq: int) -> tuple[float, int] | None:
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    if tokens.size(0) < 2:
        return None
    if tokens.size(0) > max_seq:
        tokens = tokens[:max_seq]
    tokens = tokens.to(device).unsqueeze(0)
    with torch.no_grad():
        logits = model(tokens)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = tokens[:, 1:]
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return -gathered.sum().item(), targets.numel()


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config."),
    checkpoint: Path = typer.Option(..., help="Checkpoint path."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece model path."),
    max_samples: int = typer.Option(64, help="Number of PG-19 samples."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/pg19_perplexity.json")),
    context_tokens: int = typer.Option(2048, help="Truncate text to this many tokens before scoring."),
    memorize: bool = typer.Option(False, help="Apply test-time memorization to each excerpt."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per excerpt."),
    memorize_no_reset: bool = typer.Option(False, help="Retain memory between excerpts."),
    memorize_paths: str = typer.Option(
        "all",
        help="Comma-separated memory paths to update during memorization (e.g., 'titan,cms_fast').",
    ),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required before memorizing an excerpt."
    ),
) -> None:
    torch_device = torch.device(device)
    model = load_model(config, checkpoint, torch_device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    dataset = load_dataset("pg19", split="test", streaming=True, trust_remote_code=True).shuffle(seed=42)
    total_tokens = 0
    total_nll_base = 0.0
    total_nll_mem = 0.0
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
    base_state = snapshot_state_dict(model) if memorize_cfg.enabled and memorize_cfg.reset else None
    processed = 0
    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break
        text = sample.get("text") or sample.get("passage")
        if not text:
            continue
        nll_tot = _nll_for_text(model, tokenizer, text, torch_device, context_tokens)
        if nll_tot is None:
            continue
        nll_base, tokens_seen = nll_tot
        total_nll_base += nll_base
        total_tokens += tokens_seen
        if memorize_cfg.enabled:
            memorize_sequence(model, tokenizer, text[:1024], torch_device, memorize_cfg)
            nll_mem = _nll_for_text(model, tokenizer, text, torch_device, context_tokens)
            if nll_mem is not None:
                total_nll_mem += nll_mem[0]
            if memorize_cfg.reset and base_state is not None:
                restore_state_dict(model, base_state)
        else:
            total_nll_mem += nll_base
        processed += 1
    ppl_base = float(torch.exp(torch.tensor(total_nll_base / max(1, total_tokens))))
    ppl_mem = float(torch.exp(torch.tensor(total_nll_mem / max(1, total_tokens))))
    payload = {
        "samples": processed,
        "tokens": total_tokens,
        "ppl_base": ppl_base,
        "ppl_memorize": ppl_mem,
        "ppl_delta": ppl_base - ppl_mem,
    }
    if memorize_cfg.enabled:
        payload["memorize_paths"] = "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        payload["memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    typer.echo(f"[pg19] Saved perplexity to {output}")


if __name__ == "__main__":
    app()
