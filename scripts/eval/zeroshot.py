#!/usr/bin/env python
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import typer
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from nested_learning.memorize import (
    MemorizeConfig,
    memorize_sequence,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.training import build_model_from_cfg, unwrap_config
from nested_learning.tokenizer import SentencePieceTokenizer

app = typer.Typer(add_completion=False, help="Zero-shot evaluation harness for HOPE.")
HF_DATASET_KWARGS = {"trust_remote_code": True}


def load_model(config_path: Path, checkpoint: Path, device: torch.device):
    cfg = OmegaConf.load(config_path)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[eval] Warning: state_dict mismatch "
            f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
        )
    return model.to(device).eval()


def score_text(model, tokenizer: SentencePieceTokenizer, text: str, device: torch.device) -> float:
    tokens = tokenizer.encode(text)
    tokens = tokens.to(device)
    with torch.no_grad():
        logits = model(tokens.unsqueeze(0))
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = tokens.unsqueeze(0)[:, 1:]
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return gathered.sum().item()


def evaluate_multiple_choice(
    task_name: str,
    dataset_iter: Iterable[dict],
    build_texts_fn: Callable[[dict], Tuple[str, List[str], int]],
    tokenizer: SentencePieceTokenizer,
    model,
    device: torch.device,
    max_samples: int | None,
    memorize_cfg: MemorizeConfig,
) -> Dict[str, float]:
    correct = 0
    total = 0
    base_state: Dict[str, torch.Tensor] | None = None
    for sample in tqdm(dataset_iter, desc=task_name.upper()):
        prompt, texts, answer_idx = build_texts_fn(sample)
        if memorize_cfg.enabled:
            if memorize_cfg.reset and base_state is None:
                base_state = snapshot_state_dict(model)
            memorize_text = prompt
            if memorize_cfg.use_correct_answer:
                memorize_text = f"{prompt} {texts[answer_idx]}".strip()
            memorize_sequence(model, tokenizer, memorize_text, device, memorize_cfg.steps)
        scores = [score_text(model, tokenizer, t, device) for t in texts]
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        if pred == answer_idx:
            correct += 1
        total += 1
        if memorize_cfg.enabled and memorize_cfg.reset and base_state is not None:
            restore_state_dict(model, base_state)
        if max_samples and total >= max_samples:
            break
    accuracy = correct / total if total else 0.0
    return {f"{task_name}_accuracy": accuracy, f"{task_name}_samples": total}


def build_piqa_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = sample["goal"].strip()
    options = [sample["sol1"].strip(), sample["sol2"].strip()]
    texts = [f"{prompt} {opt}" for opt in options]
    target = sample["label"]
    return prompt, texts, target


def eval_piqa(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("piqa", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "piqa", dataset, build_piqa_texts, tokenizer, model, device, max_samples, memorize_cfg
    )


def build_hellaswag_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = f"{sample['ctx_a'].strip()} {sample['ctx_b'].strip()}".strip()
    endings = [ending.strip() for ending in sample["endings"]]
    texts = [f"{prompt} {ending}" for ending in endings]
    label = sample["label"]
    target = int(label) if not isinstance(label, int) else label
    return prompt, texts, target


def eval_hellaswag(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("hellaswag", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "hellaswag",
        dataset,
        build_hellaswag_texts,
        tokenizer,
        model,
        device,
        max_samples,
        memorize_cfg,
    )


def build_winogrande_texts(sample: dict) -> Tuple[str, List[str], int]:
    sentence = sample["sentence"]
    options = [sample["option1"].strip(), sample["option2"].strip()]
    texts = [sentence.replace("_", opt) for opt in options]
    target = int(sample["answer"]) - 1
    return sentence, texts, target


def eval_winogrande(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "winogrande",
        dataset,
        build_winogrande_texts,
        tokenizer,
        model,
        device,
        max_samples,
        memorize_cfg,
    )


def build_arc_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = sample["question"].strip()
    choice_texts = sample["choices"]["text"]
    labels = sample["choices"]["label"]
    texts = [f"{prompt} {choice.strip()}" for choice in choice_texts]
    target = labels.index(sample["answerKey"])
    return prompt, texts, target


def eval_arc(model, tokenizer, device, max_samples, difficulty: str, memorize_cfg: MemorizeConfig) -> Dict[str, float]:
    dataset = load_dataset("ai2_arc", difficulty, split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        f"arc_{difficulty.lower()}",
        dataset,
        build_arc_texts,
        tokenizer,
        model,
        device,
        max_samples,
        memorize_cfg,
    )


def build_boolq_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = f"{sample['passage'].strip()}\nQuestion: {sample['question'].strip()}\nAnswer:"
    texts = [f"{prompt} yes", f"{prompt} no"]
    target = 0 if sample["answer"] else 1
    return prompt, texts, target


def eval_boolq(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("boolq", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "boolq", dataset, build_boolq_texts, tokenizer, model, device, max_samples, memorize_cfg
    )


def build_siqa_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = f"Context: {sample['context'].strip()} Question: {sample['question'].strip()} Answer:"
    options = [sample["answerA"].strip(), sample["answerB"].strip(), sample["answerC"].strip()]
    texts = [f"{prompt} {opt}" for opt in options]
    target = int(sample["label"]) - 1
    return prompt, texts, target


def eval_siqa(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("social_i_qa", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "siqa", dataset, build_siqa_texts, tokenizer, model, device, max_samples, memorize_cfg
    )


def build_commonsenseqa_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = sample["question"].strip()
    choice_texts = sample["choices"]["text"]
    labels = sample["choices"]["label"]
    texts = [f"{prompt} {choice.strip()}" for choice in choice_texts]
    target = labels.index(sample["answerKey"])
    return prompt, texts, target


def eval_commonsenseqa(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("commonsense_qa", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "commonsenseqa",
        dataset,
        build_commonsenseqa_texts,
        tokenizer,
        model,
        device,
        max_samples,
        memorize_cfg,
    )


def build_openbookqa_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = sample["question_stem"].strip()
    choice_texts = sample["choices"]["text"]
    labels = sample["choices"]["label"]
    texts = [f"{prompt} {choice.strip()}" for choice in choice_texts]
    target = labels.index(sample["answerKey"])
    return prompt, texts, target


def eval_openbookqa(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("openbookqa", "main", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "openbookqa",
        dataset,
        build_openbookqa_texts,
        tokenizer,
        model,
        device,
        max_samples,
        memorize_cfg,
    )


TASK_EVALUATORS = {
    "piqa": eval_piqa,
    "hellaswag": eval_hellaswag,
    "winogrande": eval_winogrande,
    "arc_easy": lambda model, tok, dev, n, mem: eval_arc(model, tok, dev, n, "ARC-Easy", mem),
    "arc_challenge": lambda model, tok, dev, n, mem: eval_arc(model, tok, dev, n, "ARC-Challenge", mem),
    "boolq": eval_boolq,
    "siqa": eval_siqa,
    "commonsenseqa": eval_commonsenseqa,
    "openbookqa": eval_openbookqa,
}


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config path."),
    checkpoint: Path = typer.Option(..., help="Checkpoint file (state dict)."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece model path."),
    tasks: str = typer.Option("piqa", help="Comma-separated list of tasks or 'all'."),
    max_samples: int = typer.Option(500, help="Max samples per task (0 = entire split)."),
    output: Path = typer.Option(Path("eval/zeroshot_results.json"), help="Output JSON file."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run eval on."),
    list_tasks: bool = typer.Option(False, "--list-tasks", help="List available tasks and exit."),
    memorize: bool = typer.Option(False, help="Enable test-time memorization updates."),
    memorize_steps: int = typer.Option(1, help="Number of memorize passes per sample."),
    memorize_use_correct_answer: bool = typer.Option(
        False, help="When memorizing, include the correct answer text (for ablations)."
    ),
    memorize_no_reset: bool = typer.Option(False, help="If set, retain memorization across samples."),
) -> None:
    available = list(TASK_EVALUATORS.keys())
    if list_tasks:
        typer.echo("Available tasks: " + ", ".join(available))
        raise typer.Exit(0)

    selected_tasks = available if tasks.lower() == "all" else [t.strip().lower() for t in tasks.split(",")]
    torch_device = torch.device(device)
    model = load_model(config, checkpoint, torch_device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=memorize_use_correct_answer,
    )

    results: Dict[str, float] = {}
    for task in selected_tasks:
        evaluator = TASK_EVALUATORS.get(task)
        if evaluator is None:
            raise ValueError(f"Unsupported task '{task}'. Valid tasks: {available}")
        metrics = evaluator(
            model,
            tokenizer,
            torch_device,
            None if max_samples <= 0 else max_samples,
            memorize_cfg,
        )
        results.update(metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    typer.echo(f"[Eval] Saved metrics for tasks {selected_tasks} -> {output}")


if __name__ == "__main__":
    app()
