from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

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
        self.level_specs: Sequence[LevelSpec] = tuple(ordered)
        self._spec_map: Dict[str, LevelSpec] = {spec.name: spec for spec in ordered}
        self.blocks = nn.ModuleDict(
            {
                spec.name: CMSBlock(
                    dim,
                    hidden_multiplier=hidden_multiplier,
                    activation=activation,
                    grad_clip=1.0,
                )
                for spec in self.level_specs
            }
        )
        self._chunk_buffers: Dict[str, list[dict[str, torch.Tensor]]] = {
            spec.name: [] for spec in self.level_specs
        }
        self.last_update_stats: Dict[str, Dict[str, float]] = {}

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

    def accumulate_chunks(
        self,
        *,
        inputs: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        error_signals: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
        ready: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for spec in self.level_specs:
            name = spec.name
            block_input = inputs[name]
            block_output = outputs[name]
            delta_target = (
                error_signals[name]
                if error_signals and name in error_signals
                else block_output - block_input
            )
            self._chunk_buffers[name].append(
                {
                    "input": block_input.detach(),
                    "target": (block_input + delta_target).detach(),
                }
            )
            chunk_size = spec.update_period
            if len(self._chunk_buffers[name]) < chunk_size:
                continue
            entries = [self._chunk_buffers[name].pop(0) for _ in range(chunk_size)]
            chunk_inputs = self._pad_and_cat([entry["input"] for entry in entries])
            chunk_targets = self._pad_and_cat([entry["target"] for entry in entries])
            ready[name] = (chunk_inputs, chunk_targets)
        return ready

    @staticmethod
    def _pad_and_cat(tensors: list[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            raise ValueError("No tensors to stack")
        seq_lens = {tensor.shape[1] for tensor in tensors}
        if len(seq_lens) == 1:
            return torch.cat(tensors, dim=0)
        max_len = max(seq_lens)
        padded = []
        for tensor in tensors:
            if tensor.shape[1] == max_len:
                padded.append(tensor)
                continue
            pad_len = max_len - tensor.shape[1]
            pad_shape = (tensor.shape[0], pad_len, tensor.shape[2])
            pad = torch.zeros(
                pad_shape,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            padded.append(torch.cat([tensor, pad], dim=1))
        return torch.cat(padded, dim=0)
