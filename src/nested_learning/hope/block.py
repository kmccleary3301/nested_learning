from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import AttentionConfig, SelfAttention
from ..cms import CMS
from ..levels import LevelSpec
from ..optim.manager import LevelConfig, LevelOptimizerManager
from ..titan.memory import TitanMemory, TitanMemoryConfig
from .self_mod import SelfModifier


@dataclass
class HOPEBlockConfig:
    dim: int
    heads: int
    titan_level: LevelSpec
    cms_levels: Sequence[LevelSpec]
    titan_hidden_multiplier: int = 4
    cms_hidden_multiplier: int = 4
    activation: str = "gelu"
    self_mod_hidden: int = 4
    self_mod_lr: float = 1e-3
    optimizer_configs: Dict[str, dict] = field(default_factory=dict)


class HOPEBlock(nn.Module):
    def __init__(self, config: HOPEBlockConfig):
        super().__init__()
        self.config = config
        self.last_update_stats: Dict[str, Dict[str, float]] = {}
        self.surprise_threshold: float | None = None
        self.allowed_levels: Set[str] | None = None
        self.attn = SelfAttention(AttentionConfig(dim=config.dim, heads=config.heads))
        titan_config = TitanMemoryConfig(
            dim=config.dim,
            hidden_multiplier=config.titan_hidden_multiplier,
            activation=config.activation,
        )
        self.titan_memory = TitanMemory(titan_config)
        self.cms = CMS(
            dim=config.dim,
            levels=config.cms_levels,
            hidden_multiplier=config.cms_hidden_multiplier,
            activation=config.activation,
        )
        self.self_modifier = SelfModifier(config.dim, hidden_multiplier=config.self_mod_hidden)
        self.dropout = nn.Dropout(0.0)
        specs = [config.titan_level, *config.cms_levels]
        level_config = LevelConfig(
            specs=specs,
            optimizer_configs=config.optimizer_configs,
            default_lr=config.self_mod_lr,
        )
        self.level_manager = LevelOptimizerManager(level_config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(x)
        mem_out = self.titan_memory(attn_out)
        combined = attn_out + mem_out
        cms_result = self.cms(combined, return_intermediates=True)
        cms_out, cms_inputs, cms_outputs = cms_result
        if teach_signal is not None:
            self._update_titan(attn_out, mem_out, teach_signal, surprise_value)
            self._update_cms(cms_inputs, cms_outputs, teach_signal, surprise_value)
        self.level_manager.tick()
        return cms_out

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def set_allowed_levels(self, allowed: Set[str] | None) -> None:
        self.allowed_levels = allowed.copy() if allowed is not None else None

    def _update_titan(
        self,
        attn_out: torch.Tensor,
        mem_out: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        level_name = self.config.titan_level.name
        if not self._is_level_allowed("titan"):
            return
        if not self.level_manager.should_update(level_name):
            return
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
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
            loss = F.mse_loss(prediction, target)
        magnitude = self.level_manager.optimize(level_name, self.titan_memory, loss, context=context_vec)
        extra_metrics = self.level_manager.pop_last_metrics(level_name)
        stats = {"grad_norm": magnitude, "gate_hit": 1.0}
        if surprise_value is not None:
            stats["surprise_value"] = surprise_value
        stats.update(extra_metrics)
        self.last_update_stats[f"titan.{level_name}"] = stats

    def _update_cms(
        self,
        cms_inputs: dict[str, torch.Tensor],
        cms_outputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        delta = teach_signal.detach().mean(dim=1, keepdim=True)
        error_signals = {spec.name: delta for spec in self.config.cms_levels}
        chunk_map = self.cms.accumulate_chunks(
            inputs=cms_inputs,
            outputs=cms_outputs,
            error_signals=error_signals,
        )
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self.level_manager.should_update(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            chunk = chunk_map.get(level_name)
            if chunk is None:
                continue
            chunk_inputs, chunk_targets = chunk
            with torch.enable_grad():
                chunk_inp = chunk_inputs.detach().requires_grad_(True)
                prediction = self.cms.blocks[level_name](chunk_inp)
                loss = F.mse_loss(prediction, chunk_targets)
            context_vec = chunk_inputs.mean(dim=(0, 1))
            magnitude = self.level_manager.optimize(
                level_name,
                self.cms.blocks[level_name],
                loss,
                context=context_vec,
            )
            extra_metrics = self.level_manager.pop_last_metrics(level_name)
            stats_payload = {
                "grad_norm": magnitude,
                "chunk_samples": float(chunk_inputs.shape[0]),
                "gate_hit": 1.0,
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            stats_payload.update(extra_metrics)
            self.last_update_stats[f"cms.{level_name}"] = stats_payload
            self.cms.consume_chunk(level_name)

    def pop_update_stats(self) -> Dict[str, Dict[str, float]]:
        stats = self.last_update_stats
        self.last_update_stats = {}
        return stats

    def _passes_surprise(self, surprise_value: float | None) -> bool:
        if self.surprise_threshold is None:
            return True
        if surprise_value is None:
            return False
        return surprise_value >= self.surprise_threshold

    def _is_level_allowed(self, level_name: str) -> bool:
        if self.allowed_levels is None:
            return True
        return level_name in self.allowed_levels or (
            level_name.startswith("titan") and "titan" in self.allowed_levels
        )

    def _record_gate(self, level_name: str, *, hit: bool) -> None:
        stats_key = f"gate.{level_name}"
        self.last_update_stats.setdefault(stats_key, {})
        self.last_update_stats[stats_key]["gate_hit"] = 1.0 if hit else 0.0
