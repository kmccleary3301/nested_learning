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

from nested_learning.memorize import (
    MemorizeConfig,
    memorize_sequence,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(add_completion=False, help="RULER-ish NIAH suite (multiple retrieval variants).")


def load_model(config_path: Path, checkpoint: Path, device: torch.device):
    cfg = OmegaConf.load(config_path)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[niah_suite] Warning: state_dict mismatch "
            f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
        )
    return model.to(device).eval()


def _logprob_answer(
    model,
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
        return float(gathered[0, prompt_len - 1 :].sum().item())


def _filler_sentences(count: int) -> List[str]:
    return [f"This is filler sentence number {idx}." for idx in range(count)]


def _ensure_prompt_length(
    tokenizer: SentencePieceTokenizer,
    *,
    base_lines: List[str],
    target_tokens: int,
    rng: random.Random,
    max_filler: int = 50_000,
) -> str:
    filler = []
    filler_count = max(1, target_tokens // 32)
    while True:
        filler = _filler_sentences(filler_count)
        rng.shuffle(filler)
        prompt = "\n".join([*filler, *base_lines])
        token_len = int(tokenizer.encode(prompt, add_bos=True).numel())
        if token_len >= target_tokens:
            return prompt
        if filler_count >= max_filler:
            return prompt
        missing = target_tokens - token_len
        filler_count += max(1, missing // 16)


@dataclass(frozen=True)
class VariantCase:
    prompt: str
    answer: str
    distractor: str


def _case_single_needle(rng: random.Random) -> VariantCase:
    needle = f"KEY-{rng.randint(1000, 9999)}"
    prompt_lines = [
        f"Remember that the secret key is {needle}.",
        "Later you might be asked about it.",
        "Question: What is the secret key?",
        "Answer:",
    ]
    distractor = f"KEY-{rng.randint(1000, 9999)}"
    return VariantCase(prompt="\n".join(prompt_lines), answer=needle, distractor=distractor)


def _case_multi_needle(rng: random.Random, *, needles: int) -> VariantCase:
    keys = [f"KEY-{rng.randint(1000, 9999)}" for _ in range(max(2, needles))]
    query_idx = rng.randrange(len(keys))
    prompt_lines = ["Memorize the following secret keys:"]
    for idx, key in enumerate(keys, start=1):
        prompt_lines.append(f"Key {idx}: {key}.")
    prompt_lines.extend(
        [
            f"Question: What is Key {query_idx + 1}?",
            "Answer:",
        ]
    )
    distractor = f"KEY-{rng.randint(1000, 9999)}"
    return VariantCase(
        prompt="\n".join(prompt_lines), answer=keys[query_idx], distractor=distractor
    )


def _case_kv_single(rng: random.Random) -> VariantCase:
    key = f"ITEM-{rng.randint(100, 999)}"
    value = f"VALUE-{rng.randint(1000, 9999)}"
    prompt_lines = [
        "Memorize this key-value pair:",
        f"{key} -> {value}.",
        f"Question: What is the value for {key}?",
        "Answer:",
    ]
    distractor = f"VALUE-{rng.randint(1000, 9999)}"
    return VariantCase(prompt="\n".join(prompt_lines), answer=value, distractor=distractor)


def _case_kv_multi(rng: random.Random, *, pairs: int) -> VariantCase:
    pairs = max(2, pairs)
    keys = [f"ITEM-{rng.randint(100, 999)}" for _ in range(pairs)]
    values = [f"VALUE-{rng.randint(1000, 9999)}" for _ in range(pairs)]
    query_idx = rng.randrange(pairs)
    prompt_lines = ["Memorize the following key-value pairs:"]
    for k, v in zip(keys, values, strict=True):
        prompt_lines.append(f"{k} -> {v}.")
    prompt_lines.extend(
        [
            f"Question: What is the value for {keys[query_idx]}?",
            "Answer:",
        ]
    )
    distractor = f"VALUE-{rng.randint(1000, 9999)}"
    return VariantCase(
        prompt="\n".join(prompt_lines), answer=values[query_idx], distractor=distractor
    )


def _case_positioned_needle(rng: random.Random, *, position: str) -> VariantCase:
    needle = f"KEY-{rng.randint(1000, 9999)}"
    prompt_lines = [
        f"Remember that the secret key is {needle}.",
        "Question: What is the secret key?",
        "Answer:",
    ]
    distractor = f"KEY-{rng.randint(1000, 9999)}"
    return VariantCase(prompt="\n".join(prompt_lines), answer=needle, distractor=distractor)


def _variant_cases(rng: random.Random, *, variant: str) -> VariantCase:
    if variant == "single_needle":
        return _case_single_needle(rng)
    if variant == "multi_needle":
        return _case_multi_needle(rng, needles=4)
    if variant == "kv_single":
        return _case_kv_single(rng)
    if variant == "kv_multi":
        return _case_kv_multi(rng, pairs=6)
    if variant in {"needle_early", "needle_mid", "needle_late"}:
        pos = variant.split("_", 1)[1]
        return _case_positioned_needle(rng, position=pos)
    raise ValueError(f"Unknown variant: {variant}")


def _evaluate_variant(
    model,
    tokenizer: SentencePieceTokenizer,
    device: torch.device,
    *,
    variant: str,
    context_tokens: int,
    samples: int,
    rng: random.Random,
    memorize_cfg: MemorizeConfig,
) -> Dict[str, Any]:
    base_state: Dict[str, torch.Tensor] | None = None
    fast_state = None
    if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset:
        base_state = snapshot_state_dict(model)

    correct_base = 0
    correct_mem = 0
    path_stats: Dict[str, float] = {}
    for _ in tqdm(range(samples), desc=f"{variant}@{context_tokens}"):
        case = _variant_cases(rng, variant=variant)
        if variant in {"needle_early", "needle_mid", "needle_late"}:
            memory_line, question_line, answer_line = case.prompt.split("\n", 2)
            if variant == "needle_early":
                ratio = 0.1
            elif variant == "needle_late":
                ratio = 0.9
            else:
                ratio = 0.5
            filler_count = max(1, context_tokens // 32)
            while True:
                filler = _filler_sentences(filler_count)
                rng.shuffle(filler)
                insert_at = int(ratio * max(1, len(filler)))
                insert_at = max(0, min(insert_at, len(filler)))
                with_memory = filler[:insert_at] + [memory_line] + filler[insert_at:]
                prompt = "\n".join([*with_memory, question_line, answer_line])
                token_len = int(tokenizer.encode(prompt, add_bos=True).numel())
                if token_len >= context_tokens:
                    break
                filler_count += max(1, (context_tokens - token_len) // 16)
        else:
            prompt = _ensure_prompt_length(
                tokenizer,
                base_lines=[case.prompt],
                target_tokens=context_tokens,
                rng=rng,
            )
        lp_true_base = _logprob_answer(
            model, tokenizer, prompt, case.answer, device, fast_state=fast_state
        )
        lp_false_base = _logprob_answer(
            model, tokenizer, prompt, case.distractor, device, fast_state=fast_state
        )
        correct_base += int(lp_true_base > lp_false_base)
        if memorize_cfg.enabled:
            memorize_text = (
                prompt if not memorize_cfg.use_correct_answer else f"{prompt} {case.answer}"
            )
            if memorize_cfg.use_fast_state:
                if fast_state is None or memorize_cfg.reset:
                    if not hasattr(model, "init_fast_state"):
                        raise RuntimeError("Model does not support fast state memorization")
                    fast_state = model.init_fast_state()
                stats = memorize_sequence(
                    model, tokenizer, memorize_text, device, memorize_cfg, fast_state=fast_state
                )
                for k, v in stats.items():
                    path_stats[k] = path_stats.get(k, 0.0) + v
                lp_true_mem = _logprob_answer(
                    model, tokenizer, prompt, case.answer, device, fast_state=fast_state
                )
                lp_false_mem = _logprob_answer(
                    model, tokenizer, prompt, case.distractor, device, fast_state=fast_state
                )
                correct_mem += int(lp_true_mem > lp_false_mem)
            else:
                stats = memorize_sequence(model, tokenizer, memorize_text, device, memorize_cfg)
                for k, v in stats.items():
                    path_stats[k] = path_stats.get(k, 0.0) + v
                lp_true_mem = _logprob_answer(model, tokenizer, prompt, case.answer, device)
                lp_false_mem = _logprob_answer(model, tokenizer, prompt, case.distractor, device)
                correct_mem += int(lp_true_mem > lp_false_mem)
                if memorize_cfg.reset and base_state is not None:
                    restore_state_dict(model, base_state)
        else:
            correct_mem += int(lp_true_base > lp_false_base)

    base_acc = correct_base / samples if samples else 0.0
    mem_acc = correct_mem / samples if samples else 0.0
    payload: Dict[str, Any] = {
        "variant": variant,
        "context_tokens": context_tokens,
        "samples": samples,
        "baseline_accuracy": base_acc,
        "memorize_accuracy": mem_acc,
        "memorize_delta": mem_acc - base_acc,
    }
    if memorize_cfg.enabled:
        payload["memorize_paths"] = (
            "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        )
        payload["memorize_use_correct_answer"] = bool(memorize_cfg.use_correct_answer)
        if memorize_cfg.surprise_threshold is not None:
            payload["memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
        if path_stats:
            payload["memorize_stats"] = path_stats
    return payload


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra config path."),
    checkpoint: Path = typer.Option(..., help="Checkpoint to evaluate."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer path."),
    context_tokens: List[int] = typer.Option(
        [2048, 4096, 8192], help="Target prompt token lengths."
    ),
    samples_per_length: int = typer.Option(50, help="Samples per (variant, length)."),
    variants: List[str] = typer.Option(
        [
            "single_needle",
            "multi_needle",
            "kv_single",
            "kv_multi",
            "needle_early",
            "needle_mid",
            "needle_late",
        ],
        help="Variant names to run.",
    ),
    seed: int = typer.Option(0, help="Random seed."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/niah_suite_results.json")),
    smoke: bool = typer.Option(False, help="Tiny settings for quick sanity checks."),
    memorize: bool = typer.Option(False, help="Enable test-time memorization for each prompt."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per prompt."),
    memorize_use_correct_answer: bool = typer.Option(
        False, help="Append ground truth during memorization."
    ),
    memorize_no_reset: bool = typer.Option(False, help="Retain memory between samples."),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required to trigger memorization."
    ),
    memorize_paths: str = typer.Option(
        "all",
        help=(
            "Comma-separated memory paths to update (e.g., 'titan,cms_fast'); "
            "use 'all' for no restriction."
        ),
    ),
) -> None:
    rng = random.Random(seed)
    torch_device = torch.device(device)
    model = load_model(config, checkpoint, torch_device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)

    if smoke:
        context_tokens = [256]
        samples_per_length = min(samples_per_length, 8)
        variants = ["single_needle", "kv_single"]

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

    results: List[Dict[str, Any]] = []
    for variant in variants:
        for length in context_tokens:
            results.append(
                _evaluate_variant(
                    model,
                    tokenizer,
                    torch_device,
                    variant=variant,
                    context_tokens=length,
                    samples=samples_per_length,
                    rng=rng,
                    memorize_cfg=memorize_cfg,
                )
            )

    payload = {
        "seed": seed,
        "device": str(torch_device),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "tokenizer_path": str(tokenizer_path),
        "variants": variants,
        "context_tokens": context_tokens,
        "samples_per_length": samples_per_length,
        "memorize": {
            "enabled": memorize_cfg.enabled,
            "steps": memorize_cfg.steps,
            "reset": memorize_cfg.reset,
            "use_correct_answer": bool(memorize_cfg.use_correct_answer),
            "paths": "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths),
            "surprise_threshold": memorize_cfg.surprise_threshold,
        },
        "results": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    typer.echo(f"[niah_suite] Saved results to {output}")


if __name__ == "__main__":
    app()
