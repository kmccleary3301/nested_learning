from __future__ import annotations

import torch
import torch.nn as nn


class SelfModifier(nn.Module):
    """Learns parameter updates conditioned on key/value/error signals."""

    def __init__(self, dim: int, hidden_multiplier: int = 4):
        super().__init__()
        hidden = dim * hidden_multiplier
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        error_signal: torch.Tensor,
    ) -> torch.Tensor:
        concat = torch.cat([key, value, error_signal], dim=-1)
        return self.net(concat)
