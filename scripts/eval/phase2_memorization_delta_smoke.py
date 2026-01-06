#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from nested_learning.levels import LevelSpec
from nested_learning.memorize import MemorizeConfig, memorize_tokens
from nested_learning.model import HOPEModel, ModelConfig

app = typer.Typer(
    add_completion=False,
    help=(
        "CPU-friendly smoke: show HOPE-Attention adapts via CMS updates while Transformer does not."
    ),
)


def _build_model(*, variant: str, vocab_size: int, dim: int, layers: int, heads: int) -> HOPEModel:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = (LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),)
    cfg = ModelConfig(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=layers,
        heads=heads,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
        block_variant=variant,
    )
    return HOPEModel(cfg).eval()


def _run_once(
    *,
    variant: str,
    tokens: torch.Tensor,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model = _build_model(
        variant=variant,
        vocab_size=int(tokens.max().item() + 1),
        dim=16,
        layers=1,
        heads=4,
    ).to(tokens.device)
    fast_state = model.init_fast_state()
    with torch.no_grad():
        before = model(tokens, fast_state=fast_state).detach()
    cfg = MemorizeConfig(enabled=True, steps=1, use_fast_state=True, paths=("cms_fast",))
    stats = memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    with torch.no_grad():
        after = model(tokens, fast_state=fast_state).detach()
    return {
        "delta_mean_abs": float((after - before).abs().mean().item()),
        "outputs_identical": bool(torch.allclose(before, after, atol=0.0, rtol=0.0)),
        "cms_fast_update_events": float(stats.get("cms_fast_update_events", 0.0)),
        "cms_fast_updates": float(stats.get("cms_fast_updates", 0.0)),
        "titan_update_events": float(stats.get("titan_update_events", 0.0)),
    }


@app.command()
def main(
    seed: int = typer.Option(0, help="Torch RNG seed (affects weights)."),
    vocab_size: int = typer.Option(32, help="Synthetic vocab size."),
    seq_len: int = typer.Option(16, help="Token sequence length."),
    batch_size: int = typer.Option(1, help="Batch size."),
    device: str = typer.Option("cpu", help="cpu or cuda:<idx>."),
    output: Path = typer.Option(
        Path("eval/phase2_memorization_delta_smoke.json"), help="Where to write results."
    ),
) -> None:
    torch_device = torch.device(device)
    token_gen = torch.Generator(device="cpu").manual_seed(1337)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), generator=token_gen).to(
        torch_device
    )
    results = {
        "seed": int(seed),
        "vocab_size": int(vocab_size),
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "hope_attention": _run_once(variant="hope_attention", tokens=tokens, seed=seed),
        "transformer": _run_once(variant="transformer", tokens=tokens, seed=seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    typer.echo(f"[phase2] wrote {output}")


if __name__ == "__main__":
    app()

