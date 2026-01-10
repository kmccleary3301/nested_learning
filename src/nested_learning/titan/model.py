from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, cast

import torch
import torch.nn as nn

from ..backbones import AttentionConfig, SelfAttention
from ..fast_state import BlockFastState, ModelFastState, build_block_fast_state
from ..functional import (
    call_with_deltas,
    call_with_params,
    grads_to_dict,
    params_with_deltas,
    require_grad_params,
)
from ..hope.self_mod import SelfModifier
from ..levels import LevelSpec
from ..optim.manager import LevelConfig, LevelOptimizerManager
from ..titan.memory import TitanMemory, TitanMemoryConfig


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
    qk_l2_norm: bool = False
    local_conv_window: int | None = None
    titan_hidden_multiplier: int = 4
    activation: str = "gelu"
    self_mod_hidden: int = 4
    self_mod_lr: float = 1e-3
    surprise_threshold: float | None = None
    surprise_metric: str = "l2"
    freeze_backbone: bool = False


class TitanOnlyBlock(nn.Module):
    def __init__(self, config: TitanOnlyModelConfig):
        super().__init__()
        self.config = config
        self.surprise_threshold: float | None = None
        self.surprise_metric: str = "l2"
        self.enabled: bool = True
        self.attn = SelfAttention(
            AttentionConfig(
                dim=config.dim,
                heads=config.heads,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
            )
        )
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
        surprise_value: float | None = None,
        fast_state: BlockFastState | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(x)
        if fast_state is None:
            mem_out = self.titan_memory(attn_out)
        else:
            if fast_state.titan_params is None:
                raise ValueError(
                    "fast_state.titan_params is required for TitanOnlyBlock fast-state forward"
                )
            mem_out = call_with_deltas(self.titan_memory, fast_state.titan_params, attn_out)
        combined = attn_out + mem_out
        if teach_signal is not None:
            if fast_state is None:
                self._update_titan(attn_out, mem_out, teach_signal, surprise_value)
            else:
                self._update_titan_fast(fast_state, attn_out, mem_out, teach_signal, surprise_value)
        if fast_state is None:
            self.level_manager.tick()
        else:
            fast_state.level_manager.tick()
        return self.norm(combined)

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def set_surprise_metric(self, metric: str) -> None:
        self.surprise_metric = str(metric).strip().lower()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def _passes_surprise(self, surprise_value: float | None) -> bool:
        if self.surprise_threshold is None:
            return True
        if surprise_value is None:
            return False
        return surprise_value >= self.surprise_threshold

    def _update_titan(
        self,
        attn_out: torch.Tensor,
        mem_out: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        level_name = self.config.titan_level.name
        if not self.enabled:
            return
        if not self.level_manager.should_update(level_name):
            return
        if not self._passes_surprise(surprise_value):
            return
        # Use full sequence for granular updates (Critique P1)
        # Note: We intentionally do not pool over dim=1 (sequence) here.
        modifier = self.self_modifier(
            key=attn_out.detach(),
            value=mem_out.detach(),
            error_signal=teach_signal.detach(),
        )
        context_vec = attn_out.detach().mean(dim=(0, 1))
        with torch.enable_grad():
            query = attn_out.detach()
            target = (teach_signal.detach() + modifier).detach()
            base_params = {name: param for name, param in self.titan_memory.named_parameters()}
            params_req = require_grad_params(base_params)
            prediction = call_with_params(self.titan_memory, params_req, query)
            loss_terms = nn.functional.mse_loss(prediction, target, reduction="none")
            active = teach_signal.detach().abs().sum(dim=-1, keepdim=True) > 0
            mask = active.float()
            if self.surprise_threshold is not None and self.surprise_metric == "l2":
                norms = teach_signal.norm(dim=-1, keepdim=True)
                mask = mask * (norms >= self.surprise_threshold).float()
            loss = (loss_terms * mask).sum() / mask.sum().clamp(min=1.0)

        grads = torch.autograd.grad(
            loss,
            tuple(params_req.values()),
            retain_graph=False,
            allow_unused=True,
        )
        grads_dict = grads_to_dict(params_req, grads)
        self.level_manager.apply_module_grads(
            level_name,
            self.titan_memory,
            grads_dict,
            context=context_vec,
            force=True,
        )
        # Pop metrics to avoid stale entries even if we do not log them yet.
        self.level_manager.pop_last_metrics(level_name)

    def _update_titan_fast(
        self,
        fast_state: BlockFastState,
        attn_out: torch.Tensor,
        mem_out: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        level_name = self.config.titan_level.name
        if not self.enabled:
            return
        if not fast_state.level_manager.should_update(level_name):
            return
        if not self._passes_surprise(surprise_value):
            return
        if fast_state.titan_params is None:
            return
        modifier = self.self_modifier(
            key=attn_out.detach(),
            value=mem_out.detach(),
            error_signal=teach_signal.detach(),
        )
        context_vec = attn_out.detach().mean(dim=(0, 1))
        base_params = fast_state.titan_params
        forward_params = params_with_deltas(self.titan_memory, base_params)
        params_req = require_grad_params(forward_params)
        with torch.enable_grad():
            query = attn_out.detach()
            target = (teach_signal.detach() + modifier).detach()
            prediction = call_with_params(self.titan_memory, params_req, query)
            loss_terms = nn.functional.mse_loss(prediction, target, reduction="none")
            active = teach_signal.detach().abs().sum(dim=-1, keepdim=True) > 0
            mask = active.float()
            if self.surprise_threshold is not None and self.surprise_metric == "l2":
                norms = teach_signal.norm(dim=-1, keepdim=True)
                mask = mask * (norms >= self.surprise_threshold).float()
            loss = (loss_terms * mask).sum() / mask.sum().clamp(min=1.0)
        grads = torch.autograd.grad(
            loss,
            tuple(params_req.values()),
            retain_graph=False,
            allow_unused=True,
        )
        grads_dict = grads_to_dict(params_req, grads)
        updated, _magnitude = fast_state.level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=False,
        )
        fast_state.titan_params = updated
        fast_state.level_manager.pop_last_metrics(level_name)


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
        self._surprise_metric = "l2"
        self._updates_enabled: bool = True
        self.set_surprise_metric(config.surprise_metric)
        self.set_surprise_threshold(config.surprise_threshold)
        if config.freeze_backbone:
            self.freeze_backbone()

    def set_teach_runtime(self, *, scale: float | None = None, clip: float | None = None) -> None:
        if scale is not None:
            self._runtime_teach_scale = scale
        if clip is not None:
            self._runtime_teach_clip = clip

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self._surprise_threshold = threshold
        for block in self.blocks:
            cast(TitanOnlyBlock, block).set_surprise_threshold(threshold)

    def get_surprise_threshold(self) -> float | None:
        return self._surprise_threshold

    def set_surprise_metric(self, metric: str) -> None:
        normalized = str(metric).strip().lower()
        allowed = {"l2", "loss", "logit_entropy"}
        if normalized not in allowed:
            raise ValueError(
                f"Unsupported surprise_metric={metric!r}; expected one of {sorted(allowed)}"
            )
        self._surprise_metric = normalized
        for block in self.blocks:
            cast(TitanOnlyBlock, block).set_surprise_metric(normalized)

    def get_surprise_metric(self) -> str:
        return self._surprise_metric

    def set_allowed_update_levels(self, levels: set[str] | None) -> None:
        enabled = True
        if levels is not None and "titan" not in levels and len(levels) > 0:
            enabled = False
        self._updates_enabled = enabled
        for block in self.blocks:
            cast(TitanOnlyBlock, block).set_enabled(enabled)

    def get_allowed_update_levels(self) -> set[str] | None:
        if self._updates_enabled:
            return {"titan"}
        return set()

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        fast_state: ModelFastState | None = None,
        surprise_value: float | None = None,
    ) -> torch.Tensor:
        require_external = self._surprise_metric in {"loss", "logit_entropy"}
        if require_external and self._surprise_threshold is not None:
            if teach_signal is not None and surprise_value is None:
                raise ValueError(
                    f"surprise_metric={self._surprise_metric} requires passing surprise_value "
                    "when model.surprise_threshold is set."
                )
        x = self.embed(tokens)
        if fast_state is not None and len(fast_state.blocks) != len(self.blocks):
            raise ValueError("fast_state.blocks length does not match model.blocks")
        base_surprise = surprise_value
        for idx, block in enumerate(self.blocks):
            scaled_signal = None
            if teach_signal is not None:
                scaled_signal = teach_signal * self._runtime_teach_scale
                if self._runtime_teach_clip > 0:
                    with torch.no_grad():
                        norm = scaled_signal.norm(dim=-1, keepdim=True)
                        scale = torch.clamp(norm / self._runtime_teach_clip, min=1.0)
                    scaled_signal = scaled_signal / scale
            block_surprise = base_surprise
            if (
                scaled_signal is not None
                and base_surprise is None
                and self._surprise_metric == "l2"
            ):
                block_surprise = float(scaled_signal.norm(dim=-1).mean().item())
            block_state = None if fast_state is None else fast_state.blocks[idx]
            x = block(  # type: ignore[arg-type]
                x,
                teach_signal=scaled_signal,
                surprise_value=block_surprise,
                fast_state=block_state,
            )
        x = self.norm(x)
        return self.lm_head(x)

    def freeze_backbone(self) -> None:
        """
        Freeze shared transformer components; leave TITAN memory/trainable paths active.
        """
        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        for block in self.blocks:
            typed_block = cast(TitanOnlyBlock, block)
            for p in typed_block.attn.parameters():
                p.requires_grad = False

    def init_fast_state(self) -> ModelFastState:
        states = []
        for block in self.blocks:
            typed_block = cast(TitanOnlyBlock, block)
            specs = [typed_block.config.titan_level]
            state = build_block_fast_state(
                titan_module=typed_block.titan_memory,
                cms_blocks={},
                specs=specs,
                optimizer_configs=typed_block.config.optimizers or {},
                default_lr=typed_block.config.self_mod_lr,
            )
            states.append(state)
        return ModelFastState(blocks=states)
