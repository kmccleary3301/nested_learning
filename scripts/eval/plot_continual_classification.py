#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(
    add_completion=False, help="Plot continual classification task matrix + forgetting bars."
)


@app.command()
def main(
    continual_json: Path = typer.Option(
        ..., help="Output JSON from scripts/eval/continual_classification.py"
    ),
    output: Path = typer.Option(Path("reports/plots/continual_classification.png")),
    title: str = typer.Option("Continual Classification", help="Plot title"),
) -> None:
    payload = json.loads(continual_json.read_text())
    tasks = payload.get("tasks", [])
    matrix = payload.get("result", {}).get("task_accuracy_matrix", [])
    forgetting = payload.get("result", {}).get("per_task_forgetting", [])

    task_ids: List[str] = [str(t.get("task_id", idx)) for idx, t in enumerate(tasks)]
    data = np.array(matrix, dtype=np.float32)
    mask = np.isnan(data)
    masked = np.ma.array(data, mask=mask)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    im = ax0.imshow(masked, vmin=0.0, vmax=1.0, cmap="viridis")
    ax0.set_title(f"{title} – Task Accuracy Matrix")
    ax0.set_xlabel("After Task")
    ax0.set_ylabel("Eval Task")
    ax0.set_xticks(range(len(task_ids)))
    ax0.set_yticks(range(len(task_ids)))
    ax0.set_xticklabels(task_ids, rotation=90)
    ax0.set_yticklabels(task_ids)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04, label="Accuracy")

    f = (
        np.array(forgetting, dtype=np.float32)
        if forgetting
        else np.zeros((len(task_ids),), dtype=np.float32)
    )
    ax1.bar(range(len(task_ids)), f)
    ax1.set_title("Forgetting per Task")
    ax1.set_xlabel("Task")
    ax1.set_ylabel("Max - Final Acc")
    ax1.set_xticks(range(len(task_ids)))
    ax1.set_xticklabels(task_ids, rotation=90)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    typer.echo(f"[plot] Wrote {output}")


if __name__ == "__main__":
    app()
