#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from nested_learning.tokenizer_coverage import compute_tokenizer_coverage_stats


app = typer.Typer(add_completion=False, help="Compute tokenizer coverage stats on a text sample.")


@app.command()
def main(
    tokenizer_path: Path = typer.Option(..., help="SentencePiece model path."),
    sample_file: Path = typer.Option(..., help="Text file with representative lines."),
    max_lines: int = typer.Option(10000, help="Maximum lines to process."),
    output: Optional[Path] = typer.Option(None, help="Optional JSON output path."),
) -> None:
    try:
        result = compute_tokenizer_coverage_stats(tokenizer_path, sample_file, max_lines=max_lines)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = json.dumps(result, indent=2)
    typer.echo(payload)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)


if __name__ == "__main__":
    app()
