#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
import typer
from omegaconf import OmegaConf

from nested_learning.continual_classification import (
    ClassificationExample,
    load_banking77,
    load_clinc_oos,
    load_dbpedia14,
)
from nested_learning.continual_streaming import (
    ContinualEvalConfig,
    build_streaming_tasks,
    evaluate_continual_classification,
)
from nested_learning.memorize import MemorizeConfig
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(
    add_completion=False,
    help="Class-incremental continual-learning harness (CLINC/Banking/DBpedia).",
)


def _load_local_jsonl(path: Path) -> List[ClassificationExample]:
    examples: List[ClassificationExample] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        examples.append(ClassificationExample(text=str(row["text"]), label=str(row["label"])))
    return examples


def _load_examples(
    dataset: str, *, split: str, max_samples: int | None
) -> List[ClassificationExample]:
    dataset = dataset.strip().lower()
    if dataset == "clinc":
        return load_clinc_oos(split=split, max_samples=max_samples).examples
    if dataset == "banking77":
        return load_banking77(split=split, max_samples=max_samples).examples
    if dataset == "dbpedia14":
        return load_dbpedia14(split=split, max_samples=max_samples).examples
    raise typer.BadParameter("dataset must be one of: clinc, banking77, dbpedia14")


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config path."),
    checkpoint: Path = typer.Option(..., help="Checkpoint path."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer path."),
    dataset: str = typer.Option("clinc", help="Dataset: clinc | banking77 | dbpedia14."),
    split: str = typer.Option("test", help="HF split to load."),
    local_jsonl: Path = typer.Option(
        None,
        help="Optional local JSONL (each line: {'text':..., 'label':...}); bypasses HF datasets.",
    ),
    task_size: int = typer.Option(10, help="Number of classes per task."),
    train_per_label: int = typer.Option(25, help="Streaming examples per label."),
    eval_per_label: int = typer.Option(25, help="Eval examples per label."),
    seed: int = typer.Option(0, help="Label/task shuffle seed."),
    task_aware: bool = typer.Option(True, help="Restrict candidates to current task labels."),
    max_samples: int = typer.Option(
        0, help="Max dataset samples (0 = no limit); recommended for smoke runs."
    ),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/continual_classification.json")),
    smoke: bool = typer.Option(False, help="Tiny settings for quick sanity checks."),
    memorize: bool = typer.Option(False, help="Enable test-time memorization during streaming."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per example."),
    memorize_no_reset: bool = typer.Option(
        True, help="Keep memory across examples/tasks by default."
    ),
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
    torch_device = torch.device(device)
    cfg = OmegaConf.load(config)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[continual_cls] Warning: state_dict mismatch "
            f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
        )
    model = model.to(torch_device).eval()
    tokenizer = SentencePieceTokenizer(tokenizer_path)

    resolved_max = None if max_samples <= 0 else int(max_samples)
    if smoke:
        resolved_max = 500
        task_size = min(task_size, 3)
        train_per_label = min(train_per_label, 2)
        eval_per_label = min(eval_per_label, 2)

    if local_jsonl is not None:
        examples = _load_local_jsonl(local_jsonl)
    else:
        examples = _load_examples(dataset, split=split, max_samples=resolved_max)

    eval_cfg = ContinualEvalConfig(
        task_size=task_size,
        seed=seed,
        train_per_label=train_per_label,
        eval_per_label=eval_per_label,
        task_aware=task_aware,
    )
    tasks = build_streaming_tasks(examples, cfg=eval_cfg)

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

    result, meta = evaluate_continual_classification(
        model,
        tokenizer,
        tasks,
        torch_device,
        cfg=eval_cfg,
        memorize_cfg=memorize_cfg,
    )
    payload = {
        "dataset": dataset if local_jsonl is None else str(local_jsonl),
        "split": split,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "tokenizer_path": str(tokenizer_path),
        "device": str(torch_device),
        "tasks": [
            {"task_id": t.task_id, "labels": t.labels, "train": len(t.train), "eval": len(t.eval)}
            for t in tasks
        ],
        "result": {
            "avg_accuracy_final": result.avg_accuracy_final,
            "avg_forgetting": result.avg_forgetting,
            "per_task_forgetting": result.per_task_forgetting,
            "task_accuracy_matrix": result.task_accuracy_matrix,
        },
        "meta": meta,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    typer.echo(f"[continual_cls] Saved results to {output}")


if __name__ == "__main__":
    app()
