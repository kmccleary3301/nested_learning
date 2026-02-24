from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import torch
import typer
from omegaconf import OmegaConf

from .capabilities import collect_runtime_capabilities
from .config_utils import compose_config
from .device import resolve_device
from .training import build_model_from_cfg

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Nested Learning CLI (training, diagnostics, and smoke checks).",
)


def _resolve_cli_device(device: str) -> torch.device:
    lowered = device.strip().lower()
    if lowered == "auto":
        caps = collect_runtime_capabilities()
        return resolve_device(caps.default_device)
    return resolve_device(device)


@app.command("doctor")
def doctor(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON only."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional path for writing doctor output JSON.",
            dir_okay=False,
            writable=True,
        ),
    ] = None,
) -> None:
    """Inspect runtime capabilities for backend/device compatibility."""
    payload = collect_runtime_capabilities().to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if as_json:
        typer.echo(rendered)
        return

    typer.echo("Runtime Doctor")
    typer.echo(f"python: {payload['python_version']}")
    typer.echo(f"platform: {payload['platform']} ({payload['machine']})")
    typer.echo(f"torch: {payload['torch_version']}")
    typer.echo(f"default_device: {payload['default_device']}")
    typer.echo(
        "cuda_available: {available} ({count} device(s))".format(
            available=payload["cuda_available"],
            count=payload["cuda_device_count"],
        )
    )
    for name in payload["cuda_devices"]:
        typer.echo(f"  - {name}")
    typer.echo(f"mps_available: {payload['mps_available']} (built={payload['mps_built']})")
    typer.echo(f"distributed_available: {payload['distributed_available']}")
    typer.echo(f"compile_available: {payload['compile_available']}")
    typer.echo(
        "sdpa backends: flash={flash} mem_efficient={mem} math={math}".format(
            flash=payload["sdpa_flash_available"],
            mem=payload["sdpa_mem_efficient_available"],
            math=payload["sdpa_math_available"],
        )
    )
    typer.echo(f"dtype support: bf16={payload['bf16_supported']} fp16={payload['fp16_supported']}")
    if payload["warnings"]:
        typer.echo("warnings:")
        for warning in payload["warnings"]:
            typer.echo(f"  - {warning}")


@app.command("smoke")
def smoke(
    config_name: Annotated[
        str,
        typer.Option("--config-name", "-c", help="Hydra config name (e.g. pilot, hope/mid)."),
    ] = "pilot_smoke",
    override: Annotated[
        list[str] | None,
        typer.Option(
            "--override",
            "-O",
            help="Hydra override(s), may be passed multiple times.",
        ),
    ] = None,
    config_dir: Annotated[
        Path | None,
        typer.Option(
            "--config-dir",
            help="Optional explicit config directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Device string for smoke pass (cpu, cuda:0, mps, auto).",
        ),
    ] = "cpu",
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Synthetic smoke batch size."),
    ] = 1,
    seq_len: Annotated[
        int,
        typer.Option("--seq-len", help="Synthetic smoke sequence length."),
    ] = 32,
) -> None:
    """Run a lightweight forward-pass smoke test with composed config."""
    cfg = compose_config(config_name, overrides=override or [], config_dir=config_dir)
    model_cfg = cfg.model
    torch_device = _resolve_cli_device(device)
    model = build_model_from_cfg(model_cfg).to(torch_device)
    model.eval()

    vocab_size = int(model_cfg.vocab_size)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch_device)
    with torch.no_grad():
        outputs = model(tokens)
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "config_name": config_name,
                "device": str(torch_device),
                "batch_size": batch_size,
                "seq_len": seq_len,
                "logits_shape": list(logits.shape),
                "dtype": str(logits.dtype),
            },
            sort_keys=True,
        )
    )


@app.command("train")
def train(
    config_name: Annotated[
        str,
        typer.Option("--config-name", "-c", help="Hydra config name for training."),
    ] = "pilot",
    override: Annotated[
        list[str] | None,
        typer.Option("--override", "-O", help="Hydra override(s), may be passed multiple times."),
    ] = None,
    config_dir: Annotated[
        Path | None,
        typer.Option(
            "--config-dir",
            help="Optional explicit config directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="Override cfg.train.device (e.g. cpu, cuda:1, auto).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print resolved config and exit."),
    ] = False,
) -> None:
    """Launch a local (single-process) training loop."""
    from .training import run_training_loop

    cfg = compose_config(config_name, overrides=override or [], config_dir=config_dir)
    if device is not None:
        cfg.train.device = device
    if dry_run:
        typer.echo(OmegaConf.to_yaml(cfg))
        return
    train_device = _resolve_cli_device(str(cfg.train.device))
    run_training_loop(cfg, device=train_device, distributed=False, dist_ctx=None)


@app.command("audit")
def audit(
    config_name: Annotated[
        str,
        typer.Option("--config-name", "-c", help="Hydra config name to audit."),
    ] = "pilot_paper_faithful",
    override: Annotated[
        list[str] | None,
        typer.Option("--override", "-O", help="Hydra override(s), may be passed multiple times."),
    ] = None,
    config_dir: Annotated[
        Path | None,
        typer.Option(
            "--config-dir",
            help="Optional explicit config directory.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ] = None,
) -> None:
    """Run static architecture checks on a composed config."""
    cfg = compose_config(config_name, overrides=override or [], config_dir=config_dir)
    model = build_model_from_cfg(cfg.model)
    has_embed = hasattr(model, "embed")
    has_lm_head = hasattr(model, "lm_head")
    tied_weights = False
    if has_embed and has_lm_head:
        embed = getattr(model, "embed")
        lm_head = getattr(model, "lm_head")
        tied_weights = bool(embed.weight.data_ptr() == lm_head.weight.data_ptr())

    report = {
        "status": "ok",
        "config_name": config_name,
        "model_type": str(cfg.model.get("type", "hope")),
        "block_variant": str(cfg.model.get("block_variant", "hope_hybrid")),
        "surprise_metric": str(cfg.model.get("surprise_metric", "l2")),
        "surprise_threshold": cfg.model.get("surprise_threshold"),
        "teach_scale": float(cfg.model.get("teach_scale", 1.0)),
        "teach_clip": float(cfg.model.get("teach_clip", 0.0)),
        "freeze_backbone": bool(cfg.model.get("freeze_backbone", False)),
        "has_embed": has_embed,
        "has_lm_head": has_lm_head,
        "lm_tied_to_embedding": tied_weights,
    }
    typer.echo(json.dumps(report, sort_keys=True))
