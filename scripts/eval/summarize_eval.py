#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import typer

app = typer.Typer(add_completion=False, help="Summarize eval JSONs into a small markdown table.")


def _flatten_numeric(obj: Any, *, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric(v, prefix=key))
        return out
    if isinstance(obj, list):
        # Avoid exploding large lists; only summarize scalar numeric lists.
        if obj and all(isinstance(v, (int, float)) for v in obj):
            out[prefix] = float(sum(float(v) for v in obj) / len(obj))
        return out
    if isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def _expand_keys(flat: Dict[str, float], keys: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    for key in keys:
        key = key.strip()
        if not key:
            continue
        if key.endswith("*"):
            prefix = key[:-1]
            matches = sorted(k for k in flat.keys() if k.startswith(prefix))
            resolved.extend(matches)
        else:
            resolved.append(key)
    # De-duplicate while preserving order.
    seen = set()
    ordered: List[str] = []
    for k in resolved:
        if k in seen:
            continue
        seen.add(k)
        ordered.append(k)
    return ordered


def _render_table(rows: List[Tuple[str, Dict[str, float]]], keys: List[str]) -> str:
    header = ["file", *keys]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for name, flat in rows:
        cells = [name]
        for key in keys:
            value = flat.get(key)
            if value is None:
                cells.append("")
            else:
                cells.append(f"{value:.6g}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


@app.command()
def main(
    inputs: List[Path] = typer.Option(..., help="Eval JSON files to summarize."),
    keys: List[str] = typer.Option(
        [],
        help=(
            "Dotted numeric keys to include (supports '*' suffix prefix expansion). "
            "If omitted, uses a small default set."
        ),
    ),
    output: Path = typer.Option(Path("eval/summary.md"), help="Markdown output path."),
) -> None:
    rows: List[Tuple[str, Dict[str, float]]] = []
    for path in inputs:
        payload = json.loads(path.read_text())
        flat = _flatten_numeric(payload)
        rows.append((path.name, flat))

    if not rows:
        raise typer.BadParameter("No input files provided.")

    if not keys:
        # Reasonable defaults across our eval scripts.
        keys = [
            "accuracy",
            "accuracy_base",
            "accuracy_memorize",
            "accuracy_delta",
            "avg_accuracy_final",
            "avg_forgetting",
        ]

    expanded = _expand_keys(rows[0][1], keys)
    for _name, flat in rows[1:]:
        expanded = sorted(set(expanded) | set(_expand_keys(flat, keys)))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_table(rows, expanded))
    typer.echo(f"[summary] Wrote {output}")


if __name__ == "__main__":
    app()
