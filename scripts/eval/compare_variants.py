#!/usr/bin/env python
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import typer
from omegaconf import OmegaConf
from tqdm import tqdm

from nested_learning.device import resolve_device
from nested_learning.memorize import (
    MemorizeConfig,
    memorize_sequence,
    memorize_tokens,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(
    add_completion=False, help="Compare long-context metrics across two model variants."
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: Path
    checkpoint: Path


def _load_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    cfg = OmegaConf.load(spec.config)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[compare] {spec.name}: state_dict mismatch "
            f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
        )
    return model.to(device).eval()


def _logprob_answer(
    model: torch.nn.Module,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
    *,
    fast_state=None,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    answer_ids = tokenizer.encode(" " + answer, add_bos=False)
    inputs = torch.cat([prompt_ids, answer_ids], dim=0).to(device)
    with torch.no_grad():
        logits = (
            model(inputs.unsqueeze(0), fast_state=fast_state)
            if fast_state is not None
            else model(inputs.unsqueeze(0))
        )
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = inputs.unsqueeze(0)[:, 1:]
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        prompt_len = prompt_ids.numel()
        answer_logprob = gathered[0, prompt_len - 1 :].sum().item()
    return float(answer_logprob)


def _memorize_prompt_answer_only(
    model: torch.nn.Module,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
    memorize_cfg: MemorizeConfig,
    *,
    fast_state=None,
) -> Dict[str, float]:
    """
    Memorize using gradients for the answer tokens only.

    This avoids updating on long filler/haystack tokens when `use_correct_answer=True`,
    which otherwise makes the comparison noisy for randomly-initialized checkpoints.
    """
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    answer_ids = tokenizer.encode(" " + answer, add_bos=False)
    inputs = torch.cat([prompt_ids, answer_ids], dim=0).to(device)
    batch = inputs.unsqueeze(0)
    teach_mask = torch.zeros((1, batch.size(1)), device=device)
    start = max(0, prompt_ids.numel() - 1)
    end = min(batch.size(1), start + answer_ids.numel())
    teach_mask[:, start:end] = 1.0
    return memorize_tokens(
        model,
        batch,
        memorize_cfg,
        fast_state=fast_state,
        teach_mask=teach_mask,
    )


def _make_passkey_prompt(*, filler_sentences: int, key: str) -> str:
    sentences = [f"This is filler sentence number {idx}." for idx in range(filler_sentences)]
    random.shuffle(sentences)
    filler = " ".join(sentences)
    return (
        f"{filler}\nRemember that the passkey for this document is {key}. "
        "Later we will ask about it.\nQuestion: What is the passkey?\nAnswer:"
    )


def _run_passkey(
    model: torch.nn.Module,
    tokenizer: SentencePieceTokenizer,
    device: torch.device,
    *,
    samples: int,
    filler_sentences: int,
    memorize_cfg: MemorizeConfig,
) -> Dict[str, Any]:
    base_state: Dict[str, torch.Tensor] | None = None
    fast_state = None
    if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset:
        base_state = snapshot_state_dict(model)

    correct_base = 0
    correct_mem = 0
    true_lp_base_sum = 0.0
    false_lp_base_sum = 0.0
    margin_base_sum = 0.0
    true_lp_mem_sum = 0.0
    false_lp_mem_sum = 0.0
    margin_mem_sum = 0.0
    path_stats: Dict[str, float] = {}
    for _ in tqdm(range(samples), desc="passkey"):
        key = f"PASSKEY-{random.randint(1000, 9999)}"
        prompt = _make_passkey_prompt(filler_sentences=filler_sentences, key=key)
        distractor = f"PASSKEY-{random.randint(1000, 9999)}"

        lp_true = _logprob_answer(model, tokenizer, prompt, key, device, fast_state=fast_state)
        lp_false = _logprob_answer(
            model, tokenizer, prompt, distractor, device, fast_state=fast_state
        )
        correct_base += int(lp_true > lp_false)
        true_lp_base_sum += lp_true
        false_lp_base_sum += lp_false
        margin_base_sum += lp_true - lp_false

        if memorize_cfg.enabled:
            if memorize_cfg.use_fast_state:
                if fast_state is None or memorize_cfg.reset:
                    if not hasattr(model, "init_fast_state"):
                        raise RuntimeError("Model does not support fast state memorization")
                    fast_state = model.init_fast_state()
                if memorize_cfg.use_correct_answer:
                    stats = _memorize_prompt_answer_only(
                        model,
                        tokenizer,
                        prompt,
                        key,
                        device,
                        memorize_cfg,
                        fast_state=fast_state,
                    )
                else:
                    stats = memorize_sequence(
                        model,
                        tokenizer,
                        prompt,
                        device,
                        memorize_cfg,
                        fast_state=fast_state,
                    )
                for k, v in stats.items():
                    path_stats[k] = path_stats.get(k, 0.0) + v
                lp_true_mem = _logprob_answer(
                    model, tokenizer, prompt, key, device, fast_state=fast_state
                )
                lp_false_mem = _logprob_answer(
                    model, tokenizer, prompt, distractor, device, fast_state=fast_state
                )
                correct_mem += int(lp_true_mem > lp_false_mem)
                true_lp_mem_sum += lp_true_mem
                false_lp_mem_sum += lp_false_mem
                margin_mem_sum += lp_true_mem - lp_false_mem
            else:
                memorize_text = prompt if not memorize_cfg.use_correct_answer else f"{prompt} {key}"
                stats = memorize_sequence(model, tokenizer, memorize_text, device, memorize_cfg)
                for k, v in stats.items():
                    path_stats[k] = path_stats.get(k, 0.0) + v
                lp_true_mem = _logprob_answer(model, tokenizer, prompt, key, device)
                lp_false_mem = _logprob_answer(model, tokenizer, prompt, distractor, device)
                correct_mem += int(lp_true_mem > lp_false_mem)
                true_lp_mem_sum += lp_true_mem
                false_lp_mem_sum += lp_false_mem
                margin_mem_sum += lp_true_mem - lp_false_mem
                if memorize_cfg.reset and base_state is not None:
                    restore_state_dict(model, base_state)
        else:
            correct_mem += int(lp_true > lp_false)
            true_lp_mem_sum += lp_true
            false_lp_mem_sum += lp_false
            margin_mem_sum += lp_true - lp_false

    denom = float(samples) if samples else 1.0
    base_acc = correct_base / denom
    mem_acc = correct_mem / denom
    payload: Dict[str, Any] = {
        "samples": samples,
        "filler_sentences": filler_sentences,
        "accuracy_base": base_acc,
        "accuracy_memorize": mem_acc,
        "accuracy_delta": mem_acc - base_acc,
        "mean_logprob_true_base": true_lp_base_sum / denom,
        "mean_logprob_true_memorize": true_lp_mem_sum / denom,
        "mean_logprob_true_delta": (true_lp_mem_sum - true_lp_base_sum) / denom,
        "mean_logprob_false_base": false_lp_base_sum / denom,
        "mean_logprob_false_memorize": false_lp_mem_sum / denom,
        "mean_logprob_false_delta": (false_lp_mem_sum - false_lp_base_sum) / denom,
        "mean_margin_base": margin_base_sum / denom,
        "mean_margin_memorize": margin_mem_sum / denom,
        "mean_margin_delta": (margin_mem_sum - margin_base_sum) / denom,
    }
    if memorize_cfg.enabled:
        payload["memorize_paths"] = (
            "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        )
        if memorize_cfg.surprise_threshold is not None:
            payload["memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
        payload["memorize_use_correct_answer"] = bool(memorize_cfg.use_correct_answer)
        if path_stats:
            payload["memorize_stats"] = path_stats
    return payload


def _make_niah_prompt(*, needle: str, filler_tokens: int) -> str:
    filler_chunks = ["This is filler sentence number {}.".format(i) for i in range(filler_tokens)]
    random.shuffle(filler_chunks)
    haystack = " ".join(filler_chunks)
    prompt = (
        f"{haystack} Remember that the secret key is {needle}. Later you might be asked about it. "
    )
    prompt += "Now answer the question truthfully. What is the secret key? Answer:"
    return prompt


def _run_niah(
    model: torch.nn.Module,
    tokenizer: SentencePieceTokenizer,
    device: torch.device,
    *,
    context_lengths: List[int],
    samples_per_length: int,
    memorize_cfg: MemorizeConfig,
) -> Dict[str, Any]:
    base_state: Dict[str, torch.Tensor] | None = None
    fast_state = None
    if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset:
        base_state = snapshot_state_dict(model)

    results: Dict[str, Any] = {}
    path_stats: Dict[str, float] = {}
    for length in context_lengths:
        correct_base = 0
        correct_mem = 0
        true_lp_base_sum = 0.0
        false_lp_base_sum = 0.0
        margin_base_sum = 0.0
        true_lp_mem_sum = 0.0
        false_lp_mem_sum = 0.0
        margin_mem_sum = 0.0
        for _ in tqdm(range(samples_per_length), desc=f"niah@{length}"):
            needle = f"KEY-{random.randint(1000, 9999)}"
            prompt = _make_niah_prompt(needle=needle, filler_tokens=max(1, length // 128))
            distractor = f"KEY-{random.randint(1000, 9999)}"

            lp_true_base = _logprob_answer(
                model, tokenizer, prompt, needle, device, fast_state=fast_state
            )
            lp_false_base = _logprob_answer(
                model, tokenizer, prompt, distractor, device, fast_state=fast_state
            )
            correct_base += int(lp_true_base > lp_false_base)
            true_lp_base_sum += lp_true_base
            false_lp_base_sum += lp_false_base
            margin_base_sum += lp_true_base - lp_false_base

            if memorize_cfg.enabled:
                if memorize_cfg.use_fast_state:
                    if fast_state is None or memorize_cfg.reset:
                        if not hasattr(model, "init_fast_state"):
                            raise RuntimeError("Model does not support fast state memorization")
                        fast_state = model.init_fast_state()
                    if memorize_cfg.use_correct_answer:
                        stats = _memorize_prompt_answer_only(
                            model,
                            tokenizer,
                            prompt,
                            needle,
                            device,
                            memorize_cfg,
                            fast_state=fast_state,
                        )
                    else:
                        stats = memorize_sequence(
                            model,
                            tokenizer,
                            prompt,
                            device,
                            memorize_cfg,
                            fast_state=fast_state,
                        )
                    for k, v in stats.items():
                        path_stats[k] = path_stats.get(k, 0.0) + v
                    lp_true_mem = _logprob_answer(
                        model, tokenizer, prompt, needle, device, fast_state=fast_state
                    )
                    lp_false_mem = _logprob_answer(
                        model, tokenizer, prompt, distractor, device, fast_state=fast_state
                    )
                    correct_mem += int(lp_true_mem > lp_false_mem)
                    true_lp_mem_sum += lp_true_mem
                    false_lp_mem_sum += lp_false_mem
                    margin_mem_sum += lp_true_mem - lp_false_mem
                else:
                    memorize_text = (
                        prompt if not memorize_cfg.use_correct_answer else f"{prompt} {needle}"
                    )
                    stats = memorize_sequence(model, tokenizer, memorize_text, device, memorize_cfg)
                    for k, v in stats.items():
                        path_stats[k] = path_stats.get(k, 0.0) + v
                    lp_true_mem = _logprob_answer(model, tokenizer, prompt, needle, device)
                    lp_false_mem = _logprob_answer(model, tokenizer, prompt, distractor, device)
                    correct_mem += int(lp_true_mem > lp_false_mem)
                    true_lp_mem_sum += lp_true_mem
                    false_lp_mem_sum += lp_false_mem
                    margin_mem_sum += lp_true_mem - lp_false_mem
                    if memorize_cfg.reset and base_state is not None:
                        restore_state_dict(model, base_state)
            else:
                correct_mem += int(lp_true_base > lp_false_base)
                true_lp_mem_sum += lp_true_base
                false_lp_mem_sum += lp_false_base
                margin_mem_sum += lp_true_base - lp_false_base

        base_acc = correct_base / samples_per_length if samples_per_length else 0.0
        mem_acc = correct_mem / samples_per_length if samples_per_length else 0.0
        results[f"niah_{length}_baseline_accuracy"] = base_acc
        results[f"niah_{length}_memorize_accuracy"] = mem_acc
        results[f"niah_{length}_memorize_delta"] = mem_acc - base_acc
        denom = float(samples_per_length) if samples_per_length else 1.0
        results[f"niah_{length}_mean_logprob_true_base"] = true_lp_base_sum / denom
        results[f"niah_{length}_mean_logprob_true_memorize"] = true_lp_mem_sum / denom
        results[f"niah_{length}_mean_logprob_true_delta"] = (
            true_lp_mem_sum - true_lp_base_sum
        ) / denom
        results[f"niah_{length}_mean_margin_base"] = margin_base_sum / denom
        results[f"niah_{length}_mean_margin_memorize"] = margin_mem_sum / denom
        results[f"niah_{length}_mean_margin_delta"] = (margin_mem_sum - margin_base_sum) / denom

    if memorize_cfg.enabled:
        results["memorize_paths"] = (
            "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        )
        if memorize_cfg.surprise_threshold is not None:
            results["memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
        results["memorize_use_correct_answer"] = bool(memorize_cfg.use_correct_answer)
        if path_stats:
            results["memorize_stats"] = path_stats
    return results


@app.command()
def main(
    a_config: Path = typer.Option(..., help="Hydra config for model A."),
    a_checkpoint: Path = typer.Option(..., help="Checkpoint for model A."),
    b_config: Path = typer.Option(..., help="Hydra config for model B."),
    b_checkpoint: Path = typer.Option(..., help="Checkpoint for model B."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer path."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/compare_variants.json")),
    seed: int = typer.Option(0, help="PRNG seed for prompt generation."),
    smoke: bool = typer.Option(False, help="Use tiny settings for quick sanity checks."),
    passkey_samples: int = typer.Option(64, help="Passkey prompts per model."),
    passkey_filler_sentences: int = typer.Option(200, help="Filler sentences for passkey."),
    niah_context_lengths: List[int] = typer.Option(
        [2048, 4096, 8192], help="Context lengths for NIAH."
    ),
    niah_samples_per_length: int = typer.Option(50, help="Samples per NIAH length."),
    memorize: bool = typer.Option(False, help="Enable test-time memorization for both models."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per prompt."),
    memorize_use_correct_answer: bool = typer.Option(
        False, help="Append ground truth during memorization."
    ),
    memorize_no_reset: bool = typer.Option(False, help="Retain memory between samples."),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required to trigger memorization."
    ),
    memorize_layers: str = typer.Option(
        "all",
        help=(
            "Comma-separated layer indices to update during memorization "
            "(e.g., '11' or '0,11'), or 'last', or 'all'."
        ),
    ),
    memorize_paths: str = typer.Option(
        "all",
        help=(
            "Comma-separated memory paths to update (e.g., 'titan,cms_fast'); "
            "use 'all' for no restriction."
        ),
    ),
) -> None:
    random.seed(seed)
    torch_device = resolve_device(device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)

    if smoke:
        passkey_samples = min(passkey_samples, 8)
        passkey_filler_sentences = min(passkey_filler_sentences, 20)
        niah_context_lengths = [256]
        niah_samples_per_length = min(niah_samples_per_length, 8)

    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())

    layers_raw = memorize_layers.strip().lower()
    if layers_raw == "all":
        allowed_layers = None
    elif layers_raw == "last":
        allowed_layers = (-1,)
    else:
        parsed: list[int] = []
        for part in memorize_layers.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.append(int(part))
        allowed_layers = tuple(parsed) if parsed else None
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=memorize_use_correct_answer,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
        layers=allowed_layers,
    )

    spec_a = ModelSpec(name="A", config=a_config, checkpoint=a_checkpoint)
    spec_b = ModelSpec(name="B", config=b_config, checkpoint=b_checkpoint)
    model_a = _load_model(spec_a, torch_device)
    model_b = _load_model(spec_b, torch_device)

    payload: Dict[str, Any] = {
        "seed": seed,
        "device": str(torch_device),
        "tokenizer_path": str(tokenizer_path),
        "a": {"config": str(a_config), "checkpoint": str(a_checkpoint)},
        "b": {"config": str(b_config), "checkpoint": str(b_checkpoint)},
        "memorize": {
            "enabled": memorize_cfg.enabled,
            "steps": memorize_cfg.steps,
            "reset": memorize_cfg.reset,
            "use_correct_answer": bool(memorize_cfg.use_correct_answer),
            "paths": "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths),
            "surprise_threshold": memorize_cfg.surprise_threshold,
        },
    }

    payload["a"]["passkey"] = _run_passkey(
        model_a,
        tokenizer,
        torch_device,
        samples=passkey_samples,
        filler_sentences=passkey_filler_sentences,
        memorize_cfg=memorize_cfg,
    )
    payload["b"]["passkey"] = _run_passkey(
        model_b,
        tokenizer,
        torch_device,
        samples=passkey_samples,
        filler_sentences=passkey_filler_sentences,
        memorize_cfg=memorize_cfg,
    )

    payload["a"]["niah"] = _run_niah(
        model_a,
        tokenizer,
        torch_device,
        context_lengths=niah_context_lengths,
        samples_per_length=niah_samples_per_length,
        memorize_cfg=memorize_cfg,
    )
    payload["b"]["niah"] = _run_niah(
        model_b,
        tokenizer,
        torch_device,
        context_lengths=niah_context_lengths,
        samples_per_length=niah_samples_per_length,
        memorize_cfg=memorize_cfg,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    typer.echo(f"[compare] Saved comparison to {output}")


if __name__ == "__main__":
    app()
