#!/usr/bin/env python
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import torch
import typer
from omegaconf import OmegaConf
from tqdm import tqdm

from nested_learning.model import HOPEModel
from nested_learning.training import build_model_from_cfg
from nested_learning.tokenizer import SentencePieceTokenizer

app = typer.Typer(add_completion=False, help="Needle-in-a-haystack evaluation scaffolding.")


def load_model(config_path: Path, checkpoint: Path, device: torch.device) -> HOPEModel:
    cfg = OmegaConf.load(config_path)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu")
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
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
) -> None:
    torch_device = torch.device(device)
    model = load_model(config, checkpoint, torch_device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    results = {}
    for length in context_lengths:
        correct = 0
        for _ in tqdm(range(samples_per_length), desc=f"NIAH@{length}"):
            needle = f"KEY-{random.randint(1000, 9999)}"
            prompt = make_prompt(needle, filler_tokens=max(1, length // 128))
            logprob_true = logprob_answer(model, tokenizer, prompt, needle, torch_device)
            distractor = f"KEY-{random.randint(1000, 9999)}"
            logprob_false = logprob_answer(model, tokenizer, prompt, distractor, torch_device)
            if logprob_true > logprob_false:
                correct += 1
        accuracy = correct / samples_per_length
        results[f"niah_{length}"] = accuracy
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    typer.echo(f"[Eval] Saved NIAH metrics to {output}")


if __name__ == "__main__":
    app()
