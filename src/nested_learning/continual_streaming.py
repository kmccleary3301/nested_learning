from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

from .continual_classification import ClassificationExample, unique_labels
from .memorize import MemorizeConfig, memorize_sequence
from .tokenizer import SentencePieceTokenizer


@dataclass(frozen=True)
class StreamingTask:
    task_id: int
    labels: List[str]
    train: List[ClassificationExample]
    eval: List[ClassificationExample]


@dataclass(frozen=True)
class ContinualEvalConfig:
    task_size: int = 10
    seed: int = 0
    train_per_label: int = 50
    eval_per_label: int = 50
    prompt_template: str = "Text: {text}\nLabel:"
    label_template: str = "{label}"
    task_aware: bool = True


def _logprob_completion(
    model,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    *,
    fast_state=None,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    completion_ids = tokenizer.encode(" " + completion, add_bos=False)
    tokens = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tokens, fast_state=fast_state) if fast_state is not None else model(tokens)
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = tokens[:, 1:]
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        prompt_len = prompt_ids.numel()
        return float(gathered[0, prompt_len - 1 :].sum().item())


def predict_label(
    model,
    tokenizer: SentencePieceTokenizer,
    text: str,
    candidates: Sequence[str],
    device: torch.device,
    *,
    prompt_template: str,
    label_template: str,
    fast_state=None,
) -> str:
    if not candidates:
        raise ValueError("predict_label requires at least one candidate label")
    prompt = prompt_template.format(text=text)
    best_label = candidates[0]
    best_score = -math.inf
    for label in candidates:
        label_str = label_template.format(label=label)
        score = _logprob_completion(
            model, tokenizer, prompt, label_str, device, fast_state=fast_state
        )
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def _balanced_split(
    examples: Sequence[ClassificationExample],
    *,
    labels: Sequence[str],
    train_per_label: int,
    eval_per_label: int,
) -> tuple[List[ClassificationExample], List[ClassificationExample]]:
    train: List[ClassificationExample] = []
    eval_: List[ClassificationExample] = []
    counts_train: Dict[str, int] = {lbl: 0 for lbl in labels}
    counts_eval: Dict[str, int] = {lbl: 0 for lbl in labels}
    for ex in examples:
        lbl = ex.label
        if lbl not in counts_train:
            continue
        if counts_train[lbl] < train_per_label:
            train.append(ex)
            counts_train[lbl] += 1
        elif counts_eval[lbl] < eval_per_label:
            eval_.append(ex)
            counts_eval[lbl] += 1
        if all(v >= train_per_label for v in counts_train.values()) and all(
            v >= eval_per_label for v in counts_eval.values()
        ):
            break
    return train, eval_


def build_streaming_tasks(
    examples: Sequence[ClassificationExample],
    *,
    cfg: ContinualEvalConfig,
    label_order: Sequence[str] | None = None,
) -> List[StreamingTask]:
    labels = list(label_order) if label_order is not None else unique_labels(examples)
    if label_order is None:
        import random

        rng = random.Random(cfg.seed)
        rng.shuffle(labels)
    if cfg.task_size <= 0:
        raise ValueError("task_size must be positive")
    tasks: List[StreamingTask] = []
    for task_id, start in enumerate(range(0, len(labels), cfg.task_size)):
        task_labels = labels[start : start + cfg.task_size]
        if not task_labels:
            break
        task_examples = [ex for ex in examples if ex.label in set(task_labels)]
        train, eval_ = _balanced_split(
            task_examples,
            labels=task_labels,
            train_per_label=cfg.train_per_label,
            eval_per_label=cfg.eval_per_label,
        )
        tasks.append(
            StreamingTask(task_id=task_id, labels=list(task_labels), train=train, eval=eval_)
        )
    return tasks


@dataclass(frozen=True)
class ContinualEvalResult:
    task_accuracy_matrix: List[List[float]]
    per_task_forgetting: List[float]
    avg_accuracy_final: float
    avg_forgetting: float


def evaluate_continual_classification(
    model,
    tokenizer: SentencePieceTokenizer,
    tasks: Sequence[StreamingTask],
    device: torch.device,
    *,
    cfg: ContinualEvalConfig,
    memorize_cfg: MemorizeConfig,
) -> tuple[ContinualEvalResult, Dict[str, Any]]:
    """
    Streaming class-incremental evaluation using generative classification + optional
    test-time memorization.

    - If `memorize_cfg.enabled`, each training example is memorized by appending the correct
      label string.
    - Accuracy is computed after each task on each task's eval set, producing a task-accuracy
      matrix.
    """
    meta_snapshot: Dict[str, torch.Tensor] | None = None
    if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset:
        from .memorize import snapshot_state_dict  # local import to avoid cycles

        meta_snapshot = snapshot_state_dict(model)

    fast_state = None
    if memorize_cfg.enabled and memorize_cfg.use_fast_state:
        if not hasattr(model, "init_fast_state"):
            raise RuntimeError("Model does not support fast state memorization")
        fast_state = model.init_fast_state()

    task_acc: List[List[float]] = [[float("nan") for _ in tasks] for _ in tasks]
    best_acc: List[float] = [0.0 for _ in tasks]

    memorize_stats_total: Dict[str, float] = {}

    def _eval_task(task_idx: int) -> float:
        task = tasks[task_idx]
        candidates = (
            task.labels
            if cfg.task_aware
            else [lbl for t in tasks[: current_task + 1] for lbl in t.labels]
        )
        if not task.eval:
            return float("nan")
        correct = 0
        for ex in task.eval:
            pred = predict_label(
                model,
                tokenizer,
                ex.text,
                candidates,
                device,
                prompt_template=cfg.prompt_template,
                label_template=cfg.label_template,
                fast_state=fast_state,
            )
            correct += int(pred == ex.label)
        return correct / len(task.eval) if task.eval else float("nan")

    for current_task, task in enumerate(tasks):
        # Online "training" on this task's examples via optional memorization.
        for ex in task.train:
            candidates = (
                task.labels
                if cfg.task_aware
                else [lbl for t in tasks[: current_task + 1] for lbl in t.labels]
            )
            _ = predict_label(
                model,
                tokenizer,
                ex.text,
                candidates,
                device,
                prompt_template=cfg.prompt_template,
                label_template=cfg.label_template,
                fast_state=fast_state,
            )
            if memorize_cfg.enabled:
                prompt = cfg.prompt_template.format(text=ex.text)
                target = cfg.label_template.format(label=ex.label)
                memorize_text = f"{prompt} {target}"
                if memorize_cfg.use_fast_state and memorize_cfg.reset:
                    fast_state = model.init_fast_state()
                stats = memorize_sequence(
                    model, tokenizer, memorize_text, device, memorize_cfg, fast_state=fast_state
                )
                for k, v in stats.items():
                    memorize_stats_total[k] = memorize_stats_total.get(k, 0.0) + v
                if (
                    (not memorize_cfg.use_fast_state)
                    and memorize_cfg.reset
                    and meta_snapshot is not None
                ):
                    from .memorize import restore_state_dict  # local import to avoid cycles

                    restore_state_dict(model, meta_snapshot)

        # Evaluate on all tasks seen so far.
        for task_idx in range(current_task + 1):
            acc = _eval_task(task_idx)
            task_acc[task_idx][current_task] = acc
            if not math.isnan(acc):
                best_acc[task_idx] = max(best_acc[task_idx], acc)

    final_accs = [task_acc[i][-1] for i in range(len(tasks)) if not math.isnan(task_acc[i][-1])]
    avg_accuracy_final = sum(final_accs) / len(final_accs) if final_accs else float("nan")

    per_task_forgetting: List[float] = []
    for i in range(len(tasks)):
        last = task_acc[i][-1]
        if math.isnan(last):
            per_task_forgetting.append(float("nan"))
            continue
        per_task_forgetting.append(best_acc[i] - last)
    valid_forgetting = [f for f in per_task_forgetting if not math.isnan(f)]
    avg_forgetting = (
        sum(valid_forgetting) / len(valid_forgetting) if valid_forgetting else float("nan")
    )

    result = ContinualEvalResult(
        task_accuracy_matrix=task_acc,
        per_task_forgetting=per_task_forgetting,
        avg_accuracy_final=avg_accuracy_final,
        avg_forgetting=avg_forgetting,
    )
    meta = {
        "task_size": cfg.task_size,
        "train_per_label": cfg.train_per_label,
        "eval_per_label": cfg.eval_per_label,
        "task_aware": cfg.task_aware,
        "prompt_template": cfg.prompt_template,
        "label_template": cfg.label_template,
        "memorize_stats": memorize_stats_total,
    }
    return result, meta
