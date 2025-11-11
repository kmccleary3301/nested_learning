from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn

from .hope.block import HOPEBlock, HOPEBlockConfig
from .levels import LevelSpec


@dataclass
class ModelConfig:
    vocab_size: int
    dim: int
    num_layers: int
    heads: int
    titan_level: LevelSpec
    cms_levels: Sequence[LevelSpec]
    optimizers: Dict[str, dict] | None = None
    teach_scale: float = 1.0
    teach_clip: float = 0.0
    teach_schedule: Dict[str, float] | None = None


class HOPEModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.base_teach_scale = config.teach_scale
        self.base_teach_clip = config.teach_clip
        self._runtime_teach_scale = config.teach_scale
        self._runtime_teach_clip = config.teach_clip
        block_config = HOPEBlockConfig(
            dim=config.dim,
            heads=config.heads,
            titan_level=config.titan_level,
            cms_levels=config.cms_levels,
            optimizer_configs=config.optimizers or {},
        )
        self.blocks = nn.ModuleList([HOPEBlock(block_config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Weight tying keeps the LM head gradient aligned with the embedding space.
        self.lm_head.weight = self.embed.weight
        self._latest_update_metrics: Dict[str, float] = {}

    def set_teach_runtime(self, *, scale: float | None = None, clip: float | None = None) -> None:
        if scale is not None:
            self._runtime_teach_scale = scale
        if clip is not None:
            self._runtime_teach_clip = clip

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            scaled_signal = None
            if teach_signal is not None:
                scaled_signal = teach_signal * self._runtime_teach_scale
                if self._runtime_teach_clip > 0:
                    norm = scaled_signal.norm(dim=-1, keepdim=True)
                    scale = torch.clamp(norm / self._runtime_teach_clip, min=1.0)
                    scaled_signal = scaled_signal / scale
            x = block(x, teach_signal=scaled_signal)
        x = self.norm(x)
        logits = self.lm_head(x)
        if teach_signal is not None:
            self._latest_update_metrics = self._gather_block_stats()
        return logits

    def _gather_block_stats(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for idx, block in enumerate(self.blocks):
            if hasattr(block, "pop_update_stats"):
                stats = block.pop_update_stats()
                for level_name, payload in stats.items():
                    prefix = f"layer{idx}.{level_name}"
                    for key, value in payload.items():
                        metrics[f"{prefix}.{key}"] = value
        return metrics

    def pop_update_metrics(self) -> Dict[str, float]:
        metrics = self._latest_update_metrics
        self._latest_update_metrics = {}
        return metrics
