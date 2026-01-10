from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from .levels import LevelSpec, ensure_level_specs


class CMSBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_multiplier: int = 4,
        activation: str = "gelu",
        grad_clip: float = 1.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        hidden = dim * hidden_multiplier
        act: nn.Module
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            act = nn.GELU()
        norm: nn.Module = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.net = nn.Sequential(
            norm,
            nn.Linear(dim, hidden),
            act,
            nn.Linear(hidden, dim),
        )
        self.grad_clip = grad_clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        delta = self.net(x)
        if self.training and self.grad_clip > 0:
            with torch.no_grad():
                norm = delta.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norm / self.grad_clip, min=1.0)
            delta = delta / scale
        return x + delta


class CMS(nn.Module):
    """Continuum Memory System with multi-frequency updates."""

    def __init__(
        self,
        *,
        dim: int,
        levels: Sequence[LevelSpec],
        hidden_multiplier: int = 4,
        activation: str = "gelu",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        ordered = ensure_level_specs(levels)
        self.level_specs: Sequence[LevelSpec] = tuple(ordered)
        self.blocks = nn.ModuleDict(
            {
                spec.name: CMSBlock(
                    dim,
                    hidden_multiplier=hidden_multiplier,
                    activation=activation,
                    grad_clip=1.0,
                    use_layernorm=use_layernorm,
                )
                for spec in self.level_specs
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        current = x
        inputs: Dict[str, torch.Tensor] = {}
        outputs: Dict[str, torch.Tensor] = {}
        for spec in self.level_specs:
            block = self.blocks[spec.name]
            inputs[spec.name] = current
            current = block(current)
            outputs[spec.name] = current
        if return_intermediates:
            return current, inputs, outputs
        return current
