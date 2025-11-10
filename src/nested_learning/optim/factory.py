from __future__ import annotations

from typing import Any, Dict

from .deep import DeepMomentum


def build_optimizer(config: Dict[str, Any]) -> DeepMomentum:
    opt_type = config.get("type", "deep_momentum").lower()
    if opt_type != "deep_momentum":
        raise ValueError(f"Unsupported optimizer type {opt_type}")
    params = config.get("params", {})
    return DeepMomentum(**params)
