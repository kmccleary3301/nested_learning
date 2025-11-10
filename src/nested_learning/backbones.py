from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AttentionConfig:
    dim: int
    heads: int
    dropout: float = 0.0


class SelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.dim,
            num_heads=config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(residual + self.dropout(attn_output))
