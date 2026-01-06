#!/usr/bin/env python
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import typer

app = typer.Typer(add_completion=False, help="Plot NIAH suite accuracy vs context length.")


@app.command()
def main(
    niah_suite_json: Path = typer.Option(..., help="Output JSON from scripts/eval/niah_suite.py"),
    output: Path = typer.Option(Path("reports/plots/niah_suite.png")),
    title: str = typer.Option("NIAH Suite", help="Plot title"),
) -> None:
    payload = json.loads(niah_suite_json.read_text())
    results = payload.get("results", [])
    grouped: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for row in results:
        variant = str(row.get("variant", "unknown"))
        length = int(row.get("context_tokens", 0))
        base = float(row.get("baseline_accuracy", 0.0))
        mem = float(row.get("memorize_accuracy", base))
        grouped[variant].append((length, base, mem))

    variants = sorted(grouped.keys())
    ncols = 2
    nrows = (len(variants) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3, 3 * nrows)), squeeze=False)
    axes_flat = axes.flatten()
    for ax, variant in zip(axes_flat, variants, strict=False):
        series = sorted(grouped[variant], key=lambda t: t[0])
        xs = [t[0] for t in series]
        base = [t[1] for t in series]
        mem = [t[2] for t in series]
        ax.plot(xs, base, label="baseline")
        ax.plot(xs, mem, label="memorize")
        ax.set_title(variant)
        ax.set_xlabel("context_tokens")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.legend()

    for ax in axes_flat[len(variants) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    typer.echo(f"[plot] Wrote {output}")


if __name__ == "__main__":
    app()
