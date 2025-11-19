#!/usr/bin/env python
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import typer
from omegaconf import OmegaConf

from nested_learning.memorize import MemorizeConfig, memorize_sequence, restore_state_dict, snapshot_state_dict
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(add_completion=False, help="Synthetic passkey evaluation (LongBench-style).")

PROMPT_TEMPLATE = (
    "{filler}\nRemember that the passkey for this document is {key}. "
    "Later we will ask about it.\nQuestion: What is the passkey?\nAnswer:"
)


def load_model(config: Path, checkpoint: Path, device: torch.device):
    cfg = OmegaConf.load(config)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[passkey] Warning: mismatch missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def make_prompt(context_tokens: int, key: str) -> str:
    sentences = [f"This is filler sentence number {idx}." for idx in range(context_tokens)]
    random.shuffle(sentences)
    filler = " ".join(sentences)
    return PROMPT_TEMPLATE.format(filler=filler, key=key)


def logprob(model, tokenizer, prompt: str, answer: str, device: torch.device) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    answer_ids = tokenizer.encode(" " + answer, add_bos=False, add_eos=True)
    tokens = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tokens)
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = tokens[:, 1:]
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        prompt_len = prompt_ids.numel()
        return gathered[:, prompt_len - 1 :].sum().item()


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config."),
    checkpoint: Path = typer.Option(..., help="Checkpoint path."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer."),
    samples: int = typer.Option(64, help="Number of synthetic prompts."),
    filler_sentences: int = typer.Option(200, help="Number of filler sentences (controls length)."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/passkey_results.json")),
    memorize: bool = typer.Option(False, help="Enable memorization before answering."),
    memorize_steps: int = typer.Option(1, help="Memorization iterations."),
    memorize_no_reset: bool = typer.Option(False, help="Retain memory between prompts."),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required before memorizing a prompt."
    ),
    memorize_paths: str = typer.Option(
        "all",
        help="Comma-separated memory paths to update (e.g., 'titan,cms_fast'); use 'all' for unrestricted paths.",
    ),
) -> None:
    torch_device = torch.device(device)
    model = load_model(config, checkpoint, torch_device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=True,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
    )
    base_state = snapshot_state_dict(model) if memorize_cfg.enabled and memorize_cfg.reset else None
    correct_base = 0
    correct_mem = 0
    path_stats: dict[str, float] = {}
    for _ in range(samples):
        key = f"PASSKEY-{random.randint(1000, 9999)}"
        prompt = make_prompt(filler_sentences, key)
        distractor = f"PASSKEY-{random.randint(1000, 9999)}"
        lp_true = logprob(model, tokenizer, prompt, key, torch_device)
        lp_false = logprob(model, tokenizer, prompt, distractor, torch_device)
        correct_base += int(lp_true > lp_false)
        if memorize_cfg.enabled:
            stats = memorize_sequence(model, tokenizer, prompt, torch_device, memorize_cfg)
            for k, v in stats.items():
                path_stats[k] = path_stats.get(k, 0.0) + v
            lp_true_mem = logprob(model, tokenizer, prompt, key, torch_device)
            lp_false_mem = logprob(model, tokenizer, prompt, distractor, torch_device)
            correct_mem += int(lp_true_mem > lp_false_mem)
            if memorize_cfg.reset and base_state is not None:
                restore_state_dict(model, base_state)
        else:
            correct_mem += int(lp_true > lp_false)
    base_acc = correct_base / samples if samples else 0.0
    mem_acc = correct_mem / samples if samples else 0.0
    result = {
        "samples": samples,
        "filler_sentences": filler_sentences,
        "accuracy_base": base_acc,
        "accuracy_memorize": mem_acc,
        "accuracy_delta": mem_acc - base_acc,
        "path_stats": path_stats,
    }
    if memorize_cfg.enabled:
        result["memorize_paths"] = (
            "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        )
        if memorize_cfg.surprise_threshold is not None:
            result["memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    typer.echo(f"[passkey] Saved results to {output}")


if __name__ == "__main__":
    app()
