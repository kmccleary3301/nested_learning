from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .backbones import AttentionConfig, SelfAttention


@dataclass
class TransformerBlockConfig:
    dim: int
    heads: int
    mlp_hidden_multiplier: int = 4
    activation: str = "gelu"
    qk_l2_norm: bool = False
    local_conv_window: int | None = None


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_multiplier: int = 4,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        hidden = dim * hidden_multiplier
        if activation == "relu":
            act: nn.Module = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            act,
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        x = self.norm(x)
        return residual + self.net(x)


class TransformerBlock(nn.Module):
    """
    Baseline Transformer block: Attention -> MLP (no TITAN/CMS learning updates).

    This is used for Phase 2 comparisons (HOPE-Attention vs standard Transformer).
    """

    def __init__(self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = SelfAttention(
            AttentionConfig(
                dim=config.dim,
                heads=config.heads,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
            )
        )
        self.mlp = FeedForward(
            config.dim,
            hidden_multiplier=config.mlp_hidden_multiplier,
            activation=config.activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
        fast_state=None,
    ) -> torch.Tensor:
        _ = (teach_signal, surprise_value, fast_state)
        return self.mlp(self.attn(x))

    def set_surprise_threshold(self, threshold: float | None) -> None:
        _ = threshold

    def set_allowed_levels(self, allowed) -> None:
        _ = allowed
