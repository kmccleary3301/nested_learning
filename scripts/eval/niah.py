#!/usr/bin/env python
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import typer
from omegaconf import OmegaConf
from tqdm import tqdm

from nested_learning.model import HOPEModel
from nested_learning.memorize import (
    MemorizeConfig,
    memorize_sequence,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.training import build_model_from_cfg, unwrap_config
from nested_learning.tokenizer import SentencePieceTokenizer

app = typer.Typer(add_completion=False, help="Needle-in-a-haystack evaluation scaffolding.")


def load_model(config_path: Path, checkpoint: Path, device: torch.device) -> HOPEModel:
    cfg = OmegaConf.load(config_path)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[eval] Warning: state_dict mismatch "
            f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
        )
    return model.to(device).eval()


def make_prompt(needle: str, filler_tokens: int) -> str:
    filler_chunks = ["This is filler sentence number {}.".format(i) for i in range(filler_tokens)]
    random.shuffle(filler_chunks)
    haystack = " ".join(filler_chunks)
    prompt = f"{haystack} Remember that the secret key is {needle}. Later you might be asked about it. "
    prompt += "Now answer the question truthfully. What is the secret key? Answer:"
    return prompt


def logprob_answer(model: HOPEModel, tokenizer: SentencePieceTokenizer, prompt: str, answer: str, device: torch.device) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    answer_ids = tokenizer.encode(" " + answer, add_bos=False)
    inputs = torch.cat([prompt_ids, answer_ids], dim=0).to(device)
    with torch.no_grad():
        logits = model(inputs.unsqueeze(0))
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = inputs.unsqueeze(0)[:, 1:]
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        prompt_len = prompt_ids.numel()
        answer_logprob = gathered[0, prompt_len - 1 :].sum().item()
    return answer_logprob


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra config path."),
    checkpoint: Path = typer.Option(..., help="Checkpoint to evaluate."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer path."),
    context_lengths: List[int] = typer.Option([2048, 4096, 8192], help="Context lengths to probe."),
    samples_per_length: int = typer.Option(50, help="Samples per context length."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/niah_results.json")),
    memorize: bool = typer.Option(False, help="Enable test-time memorization for each prompt."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per prompt."),
    memorize_use_correct_answer: bool = typer.Option(False, help="Include correct key when memorizing."),
    memorize_no_reset: bool = typer.Option(False, help="Retain memory between samples."),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required to trigger memorization."
    ),
    memorize_paths: str = typer.Option(
        "all",
        help="Comma-separated memory paths to update (e.g., 'titan,cms_fast'); use 'all' for no restriction.",
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
        use_correct_answer=memorize_use_correct_answer,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
    )
    base_state: Dict[str, torch.Tensor] | None = None
    results = {}
    path_stats: Dict[str, float] = defaultdict(float)
    for length in context_lengths:
        correct_base = 0
        correct_mem = 0
        for _ in tqdm(range(samples_per_length), desc=f"NIAH@{length}"):
            needle = f"KEY-{random.randint(1000, 9999)}"
            prompt = make_prompt(needle, filler_tokens=max(1, length // 128))
            distractor = f"KEY-{random.randint(1000, 9999)}"
            logprob_true_base = logprob_answer(model, tokenizer, prompt, needle, torch_device)
            logprob_false_base = logprob_answer(model, tokenizer, prompt, distractor, torch_device)
            correct_base += int(logprob_true_base > logprob_false_base)
            if memorize_cfg.enabled:
                if memorize_cfg.reset and base_state is None:
                    base_state = snapshot_state_dict(model)
                memorize_text = prompt
                if memorize_cfg.use_correct_answer:
                    memorize_text = f"{prompt} {needle}"
                stats = memorize_sequence(model, tokenizer, memorize_text, torch_device, memorize_cfg)
                for key, value in stats.items():
                    path_stats[key] += value
                logprob_true_mem = logprob_answer(model, tokenizer, prompt, needle, torch_device)
                logprob_false_mem = logprob_answer(model, tokenizer, prompt, distractor, torch_device)
                correct_mem += int(logprob_true_mem > logprob_false_mem)
                if memorize_cfg.enabled and memorize_cfg.reset and base_state is not None:
                    restore_state_dict(model, base_state)
            else:
                correct_mem += int(logprob_true_base > logprob_false_base)
        base_acc = correct_base / samples_per_length if samples_per_length else 0.0
        mem_acc = correct_mem / samples_per_length if samples_per_length else 0.0
        results[f"niah_{length}"] = mem_acc
        if memorize_cfg.enabled:
            results[f"niah_{length}_baseline_accuracy"] = base_acc
            results[f"niah_{length}_memorize_accuracy"] = mem_acc
            results[f"niah_{length}_memorize_delta"] = mem_acc - base_acc
    if memorize_cfg.enabled:
        for key, value in path_stats.items():
            results[f"niah_{key}"] = value
        results["niah_memorize_paths"] = (
            "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        )
        if memorize_cfg.surprise_threshold is not None:
            results["niah_memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    typer.echo(f"[Eval] Saved NIAH metrics to {output}")


if __name__ == "__main__":
    app()
