from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from ..backbones import AttentionConfig, SelfAttention
from ..levels import LevelSpec
from ..optim.manager import LevelConfig, LevelOptimizerManager
from ..titan.memory import TitanMemory, TitanMemoryConfig
from ..hope.self_mod import SelfModifier


@dataclass
class TitanOnlyModelConfig:
    vocab_size: int
    dim: int
    num_layers: int
    heads: int
    titan_level: LevelSpec
    optimizers: Dict[str, dict] | None = None
    teach_scale: float = 1.0
    teach_clip: float = 0.0
    teach_schedule: Dict[str, float] | None = None
    titan_hidden_multiplier: int = 4
    activation: str = "gelu"
    self_mod_hidden: int = 4
    self_mod_lr: float = 1e-3
    surprise_threshold: float | None = None


class TitanOnlyBlock(nn.Module):
    def __init__(self, config: TitanOnlyModelConfig):
        super().__init__()
        self.config = config
        self.surprise_threshold: float | None = None
        self.enabled: bool = True
        self.attn = SelfAttention(AttentionConfig(dim=config.dim, heads=config.heads))
        titan_config = TitanMemoryConfig(
            dim=config.dim,
            hidden_multiplier=config.titan_hidden_multiplier,
            activation=config.activation,
        )
        self.titan_memory = TitanMemory(titan_config)
        self.self_modifier = SelfModifier(config.dim, hidden_multiplier=config.self_mod_hidden)
        self.dropout = nn.Dropout(0.0)
        self.norm = nn.LayerNorm(config.dim)
        level_config = LevelConfig(
            specs=[config.titan_level],
            optimizer_configs=config.optimizers or {},
            default_lr=config.self_mod_lr,
        )
        self.level_manager = LevelOptimizerManager(level_config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(x)
        mem_out = self.titan_memory(attn_out)
        combined = attn_out + mem_out
        if teach_signal is not None:
            self._update_titan(attn_out, mem_out, teach_signal)
        self.level_manager.tick()
        return self.norm(combined)

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def _update_titan(
        self,
        attn_out: torch.Tensor,
        mem_out: torch.Tensor,
        teach_signal: torch.Tensor,
    ) -> None:
        level_name = self.config.titan_level.name
        if not self.enabled:
            return
        if not self.level_manager.should_update(level_name):
            return
        if self.surprise_threshold is not None:
            surprise_value = float(teach_signal.norm())
            if surprise_value < self.surprise_threshold:
                return
            return
        pooled_key = attn_out.mean(dim=1)
        pooled_value = mem_out.mean(dim=1)
        pooled_error = teach_signal.mean(dim=1)
        modifier = self.self_modifier(
            key=pooled_key.detach(),
            value=pooled_value.detach(),
            error_signal=pooled_error.detach(),
        )
        context_vec = attn_out.detach().mean(dim=(0, 1))
        with torch.enable_grad():
            query = attn_out.detach().requires_grad_(True)
            target = (teach_signal.detach() + modifier.unsqueeze(1)).detach()
            prediction = self.titan_memory(query)
            loss = nn.functional.mse_loss(prediction, target)
        self.level_manager.optimize(level_name, self.titan_memory, loss, context=context_vec)
        # Pop metrics to avoid stale entries even if we do not log them yet.
        self.level_manager.pop_last_metrics(level_name)


class TitanOnlyModel(nn.Module):
    def __init__(self, config: TitanOnlyModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([TitanOnlyBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._runtime_teach_scale = config.teach_scale
        self._runtime_teach_clip = config.teach_clip
        self._surprise_threshold: float | None = None
        self._updates_enabled: bool = True
        self.set_surprise_threshold(config.surprise_threshold)

    def set_teach_runtime(self, *, scale: float | None = None, clip: float | None = None) -> None:
        if scale is not None:
            self._runtime_teach_scale = scale
        if clip is not None:
            self._runtime_teach_clip = clip

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self._surprise_threshold = threshold
        for block in self.blocks:
            block.set_surprise_threshold(threshold)

    def get_surprise_threshold(self) -> float | None:
        return self._surprise_threshold

    def set_allowed_update_levels(self, levels: set[str] | None) -> None:
        enabled = True
        if levels is not None and "titan" not in levels and len(levels) > 0:
            enabled = False
        self._updates_enabled = enabled
        for block in self.blocks:
            block.set_enabled(enabled)

    def get_allowed_update_levels(self) -> set[str] | None:
        if self._updates_enabled:
            return {"titan"}
        return set()

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
                    with torch.no_grad():
                        norm = scaled_signal.norm(dim=-1, keepdim=True)
                        scale = torch.clamp(norm / self._runtime_teach_clip, min=1.0)
                    scaled_signal = scaled_signal / scale
            x = block(x, teach_signal=scaled_signal)  # type: ignore[arg-type]
        x = self.norm(x)
        return self.lm_head(x)
