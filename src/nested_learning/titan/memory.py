from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from ..assoc_memory import AssocMemory


@dataclass
class TitanMemoryConfig:
    dim: int
    hidden_multiplier: int = 4
    layers: int = 2
    activation: str = "gelu"


def _activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "silu":
        return nn.SiLU()
    msg = f"Unsupported activation {name}"
    raise ValueError(msg)


class TitanMemory(AssocMemory):
    """Simplified TITAN-style associative memory."""

    def __init__(self, config: TitanMemoryConfig):
        super().__init__()
        self.config = config
        hidden = config.dim * config.hidden_multiplier
        blocks = []
        activation = _activation(config.activation)
        for layer_idx in range(config.layers - 1):
            blocks.extend(
                [nn.Linear(config.dim if layer_idx == 0 else hidden, hidden), activation]
            )
        blocks.append(nn.Linear(hidden if config.layers > 1 else config.dim, config.dim))
        self.net = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(config.dim)
        self.grad_clip = 1.0

    def forward(self, query: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn = self.net(query)
        if self.training and self.grad_clip > 0:
            with torch.no_grad():
                norm = attn.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norm / self.grad_clip, min=1.0)
            attn = attn / scale
        return self.norm(attn)

    def surprise(self, residual: torch.Tensor) -> torch.Tensor:
        return residual.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def update(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        error_signal: torch.Tensor | None = None,
        lr: float = 1e-3,
    ) -> None:
        with torch.enable_grad():
            key_detached = key.detach().requires_grad_(True)
            prediction = self.forward(key_detached)
            target = value.detach()
            if error_signal is None:
                loss = torch.mean((prediction - target) ** 2)
            else:
                loss = torch.mean(error_signal * prediction)
        grads = torch.autograd.grad(loss, list(self.net.parameters()), retain_graph=False)
        for param, grad in zip(self.net.parameters(), grads, strict=False):
            if grad is None:
                continue
            param.add_(grad, alpha=-lr)

    @torch.no_grad()
    def apply_deltas(self, deltas: Dict[str, torch.Tensor], scale: float = 1.0) -> None:
        for name, tensor in deltas.items():
            target = dict(self.named_parameters()).get(name)
            if target is None:
                continue
            target.add_(tensor, alpha=scale)
