from __future__ import annotations

import torch
import torch.nn as nn


class SelfModifier(nn.Module):
    """
    Learns parameter updates conditioned on key/value/error signals.
    
    Note: In this implementation, we predict a 'target modification' (delta to the error signal)
    rather than directly predicting weight deltas (Delta W). Mathematically, modifying the
    target y to (y + delta) in the inner optimization step:
        L = || f(x) - (y + delta) ||^2
    results in a gradient update that is shifted by the gradient of delta.
    This is functionally equivalent to a 'Learned Optimization Step' or 'Hypernetwork'
    that modulates the update direction, but is more efficient to implement for
    large memory modules than generating O(d^2) weight parameters directly.
    """

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
