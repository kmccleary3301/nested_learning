#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import typer

from nested_learning.training import verify_checkpoint_integrity


app = typer.Typer(help="Verify checkpoint metadata hashes, config, and RNG sidecars.")


@app.command()
def main(
    checkpoint: Path = typer.Option(..., help="Path to checkpoint .pt file."),
) -> None:
    metadata = verify_checkpoint_integrity(checkpoint)
    typer.echo(f"[verify] {checkpoint} OK (step {metadata.get('step')})")
    typer.echo(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    app()
