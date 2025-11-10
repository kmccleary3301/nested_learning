from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from .levels import LevelClock, LevelSpec, ensure_level_specs


class CMSBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_multiplier: int = 4,
        activation: str = "gelu",
        grad_clip: float = 1.0,
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
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
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
    ) -> None:
        super().__init__()
        ordered = ensure_level_specs(levels)
        self.clock = LevelClock(ordered)
        self.blocks = nn.ModuleDict(
            {
                spec.name: CMSBlock(
                    dim,
                    hidden_multiplier=hidden_multiplier,
                    activation=activation,
                    grad_clip=1.0,
                )
                for spec in ordered
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
        for spec in self.clock.levels_in_frequency_order():
            block = self.blocks[spec.name]
            inputs[spec.name] = current
            current = block(current)
            outputs[spec.name] = current
        if return_intermediates:
            return current, inputs, outputs
        return current

    def maybe_update(
        self,
        *,
        inputs: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        error_signals: Optional[Dict[str, torch.Tensor]] = None,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        magnitudes: Dict[str, float] = {}
        for spec in self.clock.levels_in_frequency_order():
            name = spec.name
            if not self.clock.should_update(name):
                continue
            block = self.blocks[name]
            block_input = inputs[name]
            block_output = outputs[name]
            delta_target = (
                error_signals[name]
                if error_signals and name in error_signals
                else block_output - block_input
            )
            with torch.enable_grad():
                temp_inp = block_input.detach().requires_grad_(True)
                temp_out = block(temp_inp)
                loss = torch.mean((temp_out - (temp_inp + delta_target.detach())) ** 2)
            grads = torch.autograd.grad(loss, list(block.parameters()), retain_graph=False, allow_unused=True)
            total_norm = 0.0
            with torch.no_grad():
                for param, grad in zip(block.parameters(), grads):
                    if grad is None:
                        continue
                    param.add_(grad, alpha=-lr)
                    total_norm += grad.norm().item()
            magnitudes[name] = total_norm
            self.clock.record_update(name)
        self.clock.tick()
        return magnitudes
