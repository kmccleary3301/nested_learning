#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import typer


app = typer.Typer(add_completion=False, help="Plot continual-learning forgetting curves.")


@app.command()
def main(
    continual_json: Path = typer.Option(..., help="Path to eval/continual_*.json output."),
    output: Path = typer.Option(Path("reports/plots/continual_forgetting.png")),
    segment: str = typer.Option(None, help="Specific segment to plot (default: all)."),
) -> None:
    data = json.loads(continual_json.read_text())
    checkpoints = []
    baseline = []
    memorize = []
    for entry in data:
        checkpoints.append(entry.get("checkpoint"))
        seg_losses = entry.get("segment_losses", {})
        base_losses = entry.get("segment_baseline_losses", seg_losses)
        key = segment or next(iter(seg_losses))
        baseline.append(base_losses.get(key))
        memorize.append(seg_losses.get(key))
    plt.figure(figsize=(8, 4))
    plt.plot(checkpoints, baseline, label="baseline CE", marker="o")
    plt.plot(checkpoints, memorize, label="memorize CE", marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cross-entropy")
    plt.title(f"Continual forgetting ({segment or 'default segment'})")
    plt.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)
    typer.echo(f"[plot] Saved plot to {output}")


if __name__ == "__main__":
    app()
