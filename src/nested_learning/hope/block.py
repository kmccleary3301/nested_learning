from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

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
    ) -> torch.Tensor:
        attn_out = self.attn(x)
        mem_out = self.titan_memory(attn_out)
        combined = attn_out + mem_out
        cms_result = self.cms(combined, return_intermediates=True)
        cms_out, cms_inputs, cms_outputs = cms_result
        if teach_signal is not None:
            self._update_titan(attn_out, mem_out, teach_signal)
            self._update_cms(cms_inputs, cms_outputs, teach_signal)
        self.level_manager.tick()
        return cms_out

    def _update_titan(
        self,
        attn_out: torch.Tensor,
        mem_out: torch.Tensor,
        teach_signal: torch.Tensor,
    ) -> None:
        level_name = self.config.titan_level.name
        if not self.level_manager.should_update(level_name):
            return
        pooled_key = attn_out.mean(dim=1)
        pooled_value = mem_out.mean(dim=1)
        pooled_error = teach_signal.mean(dim=1)
        modifier = self.self_modifier(
            key=pooled_key.detach(),
            value=pooled_value.detach(),
            error_signal=pooled_error.detach(),
        )
        with torch.enable_grad():
            query = attn_out.detach().requires_grad_(True)
            target = (teach_signal.detach() + modifier.unsqueeze(1)).detach()
            prediction = self.titan_memory(query)
            loss = F.mse_loss(prediction, target)
        self.level_manager.optimize(level_name, self.titan_memory, loss)

    def _update_cms(
        self,
        cms_inputs: dict[str, torch.Tensor],
        cms_outputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
    ) -> None:
        delta = teach_signal.detach().mean(dim=1, keepdim=True)
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self.level_manager.should_update(level_name):
                continue
            block = self.cms.blocks[level_name]
            with torch.enable_grad():
                block_input = cms_inputs[level_name].detach().requires_grad_(True)
                target = (cms_outputs[level_name].detach() + delta).detach()
                prediction = block(block_input)
                loss = F.mse_loss(prediction, target)
            self.level_manager.optimize(level_name, block, loss)
