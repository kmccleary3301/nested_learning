#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from nested_learning.tokenizer_coverage import compute_tokenizer_coverage_stats


app = typer.Typer(
    add_completion=False,
    help="Regress coverage stats against a recorded baseline to catch tokenizer drift.",
)


@app.command()
def main(
    baseline: Path = typer.Option(
        ...,
        help="Reference JSON produced by scripts/data/check_tokenizer_coverage.py.",
    ),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer to evaluate."),
    sample_file: Path = typer.Option(..., help="Representative text sample."),
    max_lines: int = typer.Option(10_000, help="Maximum lines to consume."),
    avg_tokens_tolerance: float = typer.Option(
        0.05,
        help="Allowed increase in avg tokens per word before failing.",
    ),
    single_token_drop_tolerance: float = typer.Option(
        0.02,
        help="Allowed decrease in pct_single_token_words before failing.",
    ),
    two_token_drop_tolerance: float = typer.Option(
        0.02,
        help="Allowed decrease in pct_two_or_less_tokens_words before failing.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Optional path to write the freshly computed coverage JSON.",
    ),
) -> None:
    if not baseline.exists():
        raise typer.BadParameter(f"Baseline JSON {baseline} was not found.")
    baseline_stats = json.loads(baseline.read_text())
    current_stats = compute_tokenizer_coverage_stats(tokenizer_path, sample_file, max_lines=max_lines)
    violations: list[str] = []

    delta_avg = current_stats["avg_tokens_per_word"] - baseline_stats["avg_tokens_per_word"]
    if delta_avg > avg_tokens_tolerance:
        violations.append(
            f"avg_tokens_per_word regressed by {delta_avg:.4f} (limit {avg_tokens_tolerance:.4f})."
        )

    delta_single = baseline_stats["pct_single_token_words"] - current_stats["pct_single_token_words"]
    if delta_single > single_token_drop_tolerance:
        violations.append(
            f"pct_single_token_words dropped by {delta_single:.4f} (limit {single_token_drop_tolerance:.4f})."
        )

    delta_two = baseline_stats["pct_two_or_less_tokens_words"] - current_stats["pct_two_or_less_tokens_words"]
    if delta_two > two_token_drop_tolerance:
        violations.append(
            f"pct_two_or_less_tokens_words dropped by {delta_two:.4f} (limit {two_token_drop_tolerance:.4f})."
        )

    payload = json.dumps(current_stats, indent=2)
    typer.echo("# Tokenizer coverage guard")
    typer.echo(f"- Baseline: {baseline}")
    typer.echo(f"- Tokenizer: {tokenizer_path}")
    typer.echo(f"- Sample: {sample_file}")
    typer.echo(payload)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)

    if violations:
        typer.echo("Guard failed:")
        for violation in violations:
            typer.echo(f"  - {violation}")
        raise typer.Exit(code=1)

    typer.echo("Guard passed: tokenizer coverage within tolerance.")


if __name__ == "__main__":
    app()
