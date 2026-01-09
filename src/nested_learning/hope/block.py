from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import AttentionConfig, SelfAttention
from ..cms import CMS
from ..fast_state import BlockFastState
from ..functional import (
    call_with_deltas,
    call_with_params,
    grads_to_dict,
    params_with_deltas,
    require_grad_params,
)
from ..levels import LevelSpec
from ..optim.manager import LevelConfig, LevelOptimizerManager
from ..titan.memory import TitanMemory, TitanMemoryConfig
from ..titan.self_modifying import SelfModifyingTitans, SelfModifyingTitansConfig
from .self_mod import SelfModifier


def _chunk_loss(
    prediction: torch.Tensor,
    delta_target: torch.Tensor,
    mask_f: torch.Tensor,
    *,
    reduction: str,
) -> torch.Tensor:
    target = (prediction.detach() - delta_target).detach()
    diff_sq = (prediction - target).pow(2)
    masked = diff_sq * mask_f
    if reduction == "mean":
        return masked.sum() / mask_f.sum().clamp(min=1.0)
    if reduction == "sum":
        return masked.sum()
    raise ValueError(f"Unsupported cms_chunk_reduction={reduction}")


def _min_update_period(levels: Sequence[LevelSpec]) -> int:
    periods = [int(spec.update_period) for spec in levels if int(spec.update_period) > 0]
    return min(periods) if periods else 1


@dataclass
class _CmsBuffer:
    inputs: list[torch.Tensor]
    teach: list[torch.Tensor]
    active: list[torch.Tensor]
    count: int = 0


def _pop_buffer_chunk(
    buffer: _CmsBuffer,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("count must be positive")
    result_inputs: list[torch.Tensor] = []
    result_teach: list[torch.Tensor] = []
    result_active: list[torch.Tensor] = []
    remaining = count
    while remaining > 0:
        first = buffer.inputs[0]
        chunk_len = first.size(1)
        take = min(remaining, chunk_len)
        src_inputs = buffer.inputs[0]
        src_teach = buffer.teach[0]
        src_active = buffer.active[0]
        result_inputs.append(src_inputs[:, :take])
        result_teach.append(src_teach[:, :take])
        result_active.append(src_active[:, :take])
        if take == chunk_len:
            buffer.inputs.pop(0)
            buffer.teach.pop(0)
            buffer.active.pop(0)
        else:
            buffer.inputs[0] = src_inputs[:, take:]
            buffer.teach[0] = src_teach[:, take:]
            buffer.active[0] = src_active[:, take:]
        remaining -= take
    return (
        torch.cat(result_inputs, dim=1),
        torch.cat(result_teach, dim=1),
        torch.cat(result_active, dim=1),
    )


@dataclass
class HOPEBlockConfig:
    dim: int
    heads: int
    titan_level: LevelSpec
    cms_levels: Sequence[LevelSpec]
    titan_hidden_multiplier: int = 4
    cms_hidden_multiplier: int = 4
    activation: str = "gelu"
    qk_l2_norm: bool = False
    local_conv_window: int | None = None
    self_mod_hidden: int = 4
    self_mod_lr: float = 1e-3
    cms_chunk_reduction: str = "sum"
    cms_online_updates: bool = True
    cms_flush_partial_at_end: bool = False
    optimizer_configs: Dict[str, dict] = field(default_factory=dict)


@dataclass
class HOPEAttentionBlockConfig:
    dim: int
    heads: int
    cms_levels: Sequence[LevelSpec]
    cms_hidden_multiplier: int = 4
    activation: str = "gelu"
    qk_l2_norm: bool = False
    local_conv_window: int | None = None
    self_mod_lr: float = 1e-3
    cms_chunk_reduction: str = "sum"
    cms_online_updates: bool = True
    cms_flush_partial_at_end: bool = False
    optimizer_configs: Dict[str, dict] = field(default_factory=dict)


class HOPEAttentionBlock(nn.Module):
    """
    Paper-defined HOPE-Attention variant: softmax attention followed by CMS.

    Reference: Nested Learning paper, HOPE-Attention note under Eqs. 94–97.
    """

    def __init__(self, config: HOPEAttentionBlockConfig):
        super().__init__()
        self.config = config
        self.last_update_stats: Dict[str, Dict[str, float]] = {}
        self.surprise_threshold: float | None = None
        self.allowed_levels: Set[str] | None = None
        self.attn = SelfAttention(
            AttentionConfig(
                dim=config.dim,
                heads=config.heads,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
            )
        )
        self.cms = CMS(
            dim=config.dim,
            levels=config.cms_levels,
            hidden_multiplier=config.cms_hidden_multiplier,
            activation=config.activation,
        )
        level_config = LevelConfig(
            specs=config.cms_levels,
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
        fast_state: BlockFastState | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(x)
        if fast_state is None:
            if teach_signal is not None and self.config.cms_online_updates:
                cms_out = self._cms_forward_online(attn_out, teach_signal, surprise_value)
            else:
                cms_result = self.cms(attn_out, return_intermediates=True)
                cms_out, cms_inputs, cms_outputs = cms_result
                if teach_signal is not None:
                    self._update_cms(cms_inputs, cms_outputs, teach_signal, surprise_value)
            self.level_manager.tick()
            return cms_out
        if teach_signal is not None and self.config.cms_online_updates:
            cms_out = self._cms_forward_online_fast(
                attn_out, fast_state, teach_signal, surprise_value
            )
        else:
            cms_out, cms_inputs = self._cms_forward_fast(attn_out, fast_state)
            if teach_signal is not None:
                self._update_cms_fast(fast_state, cms_inputs, teach_signal, surprise_value)
        fast_state.level_manager.tick()
        return cms_out

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def set_allowed_levels(self, allowed: Set[str] | None) -> None:
        self.allowed_levels = allowed.copy() if allowed is not None else None

    def pop_update_stats(self) -> Dict[str, Dict[str, float]]:
        stats = self.last_update_stats
        self.last_update_stats = {}
        return stats

    def _cms_forward_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current = x
        inputs: dict[str, torch.Tensor] = {}
        for spec in self.config.cms_levels:
            level_name = spec.name
            inputs[level_name] = current
            params = fast_state.cms_params[level_name]
            current = call_with_deltas(self.cms.blocks[level_name], params, current)
        return current, inputs

    def _cms_forward_online(
        self,
        x: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                current = self.cms.blocks[level_name](current)
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk(
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)

    def _cms_forward_online_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                params = fast_state.cms_params[level_name]
                current = call_with_deltas(self.cms.blocks[level_name], params, current)
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk_fast(
                        fast_state,
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)

    def _update_cms_fast(
        self,
        fast_state: BlockFastState,
        cms_inputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _is_level_allowed(self, level_name: str) -> bool:
        if self.allowed_levels is None:
            return True
        return level_name in self.allowed_levels

    def _passes_surprise(self, surprise_value: float | None) -> bool:
        if self.surprise_threshold is None:
            return True
        if surprise_value is None:
            return False
        return surprise_value >= self.surprise_threshold

    def _record_gate(self, level_name: str, *, hit: bool) -> None:
        stats_key = f"gate.{level_name}"
        self.last_update_stats.setdefault(stats_key, {})
        self.last_update_stats[stats_key]["gate_hit"] = 1.0 if hit else 0.0

    def _update_cms(
        self,
        cms_inputs: dict[str, torch.Tensor],
        cms_outputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _update_cms_chunk(
        self,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        with torch.enable_grad():
            prediction = self.cms.blocks[level_name](chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        context_vec = chunk_inputs.mean(dim=(0, 1))
        magnitude = self.level_manager.optimize(
            level_name,
            self.cms.blocks[level_name],
            loss,
            context=context_vec,
            force=True,
        )
        self.level_manager.pop_last_metrics(level_name)
        return magnitude

    def _update_cms_chunk_fast(
        self,
        fast_state: BlockFastState,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        base_params = fast_state.cms_params[level_name]
        forward_params = params_with_deltas(self.cms.blocks[level_name], base_params)
        params_req = require_grad_params(forward_params)
        with torch.enable_grad():
            prediction = call_with_params(self.cms.blocks[level_name], params_req, chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        grads = torch.autograd.grad(
            loss,
            tuple(params_req.values()),
            retain_graph=False,
            allow_unused=True,
        )
        grads_dict = grads_to_dict(params_req, grads)
        context_vec = chunk_inputs.mean(dim=(0, 1))
        updated, magnitude = fast_state.level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=True,
        )
        fast_state.cms_params[level_name] = updated
        fast_state.level_manager.pop_last_metrics(level_name)
        return magnitude


@dataclass
class HOPESelfModBlockConfig:
    dim: int
    cms_levels: Sequence[LevelSpec]
    cms_hidden_multiplier: int = 4
    activation: str = "gelu"
    qk_l2_norm: bool = True
    cms_flush_partial_at_end: bool = False
    selfmod_adaptive_q: bool = False
    selfmod_local_conv_window: int | None = 4
    eta_scale: float = 1e-3
    selfmod_chunk_size: int = 1
    selfmod_chunk_size_memory: int | None = None
    selfmod_objective: str = "l2"
    selfmod_stopgrad_vhat: bool = True
    selfmod_use_rank1_precond: bool = True
    selfmod_use_alpha: bool = True
    selfmod_use_skip: bool = True
    selfmod_momentum: float = 0.0
    selfmod_online_updates: bool = True
    self_mod_lr: float = 1e-3
    cms_chunk_reduction: str = "sum"
    cms_online_updates: bool = True
    optimizer_configs: Dict[str, dict] = field(default_factory=dict)


class HOPESelfModBlock(nn.Module):
    """
    Paper-defined HOPE block (Eqs. 94–97): self-modifying Titans followed by CMS.

    Fast-state is required for in-context self-mod updates.
    """

    def __init__(self, config: HOPESelfModBlockConfig):
        super().__init__()
        self.config = config
        self.last_update_stats: Dict[str, Dict[str, float]] = {}
        self.surprise_threshold: float | None = None
        self.allowed_levels: Set[str] | None = None
        self.selfmod = SelfModifyingTitans(
            SelfModifyingTitansConfig(
                dim=config.dim,
                eta_scale=config.eta_scale,
                chunk_size_other=config.selfmod_chunk_size,
                chunk_size_memory=config.selfmod_chunk_size_memory,
                objective=config.selfmod_objective,
                stopgrad_vhat=config.selfmod_stopgrad_vhat,
                use_rank1_precond=config.selfmod_use_rank1_precond,
                use_alpha=config.selfmod_use_alpha,
                use_skip=config.selfmod_use_skip,
                momentum=config.selfmod_momentum,
                qk_l2_norm=config.qk_l2_norm,
                adaptive_q=config.selfmod_adaptive_q,
                local_conv_window=config.selfmod_local_conv_window,
            )
        )
        self.cms = CMS(
            dim=config.dim,
            levels=config.cms_levels,
            hidden_multiplier=config.cms_hidden_multiplier,
            activation=config.activation,
        )
        level_config = LevelConfig(
            specs=config.cms_levels,
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
        fast_state: BlockFastState | None = None,
    ) -> torch.Tensor:
        if fast_state is None:
            # Differentiable read path (used for the outer loss).
            o = self.selfmod(x)
            # Explicit update pass (typically called under `torch.no_grad()` after backward).
            if teach_signal is not None and self.config.selfmod_online_updates:
                self.selfmod.apply_updates_inplace(x)
            if teach_signal is not None and self.config.cms_online_updates:
                cms_out = self._cms_forward_online(o, teach_signal, surprise_value)
            else:
                cms_out, cms_inputs, cms_outputs = self.cms(o, return_intermediates=True)
                if teach_signal is not None:
                    self._update_cms(cms_inputs, cms_outputs, teach_signal, surprise_value)
            self.level_manager.tick()
            return cms_out

        if fast_state.selfmod_state is None:
            raise ValueError("fast_state.selfmod_state is required for hope_selfmod variant")
        if self.config.selfmod_online_updates and teach_signal is not None:
            o, updated = self.selfmod.forward_with_updates(x, fast_state.selfmod_state)
            fast_state.selfmod_state = updated
        else:
            o = self.selfmod.forward_with_state(x, fast_state.selfmod_state)
        if teach_signal is not None and self.config.cms_online_updates:
            cms_out = self._cms_forward_online_fast(o, fast_state, teach_signal, surprise_value)
        else:
            cms_out, cms_inputs = self._cms_forward_fast(o, fast_state)
            if teach_signal is not None:
                self._update_cms_fast(fast_state, cms_inputs, teach_signal, surprise_value)
        fast_state.level_manager.tick()
        return cms_out

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def set_allowed_levels(self, allowed: Set[str] | None) -> None:
        self.allowed_levels = allowed.copy() if allowed is not None else None

    def pop_update_stats(self) -> Dict[str, Dict[str, float]]:
        stats = self.last_update_stats
        self.last_update_stats = {}
        return stats

    def _cms_forward_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current = x
        inputs: dict[str, torch.Tensor] = {}
        for spec in self.config.cms_levels:
            level_name = spec.name
            inputs[level_name] = current
            params = fast_state.cms_params[level_name]
            current = call_with_deltas(self.cms.blocks[level_name], params, current)
        return current, inputs

    def _cms_forward_online(
        self,
        x: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                current = self.cms.blocks[level_name](current)
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk(
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)

    def _cms_forward_online_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                params = fast_state.cms_params[level_name]
                current = call_with_deltas(self.cms.blocks[level_name], params, current)
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk_fast(
                        fast_state,
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)

    def _is_level_allowed(self, level_name: str) -> bool:
        if self.allowed_levels is None:
            return True
        return level_name in self.allowed_levels

    def _passes_surprise(self, surprise_value: float | None) -> bool:
        if self.surprise_threshold is None:
            return True
        if surprise_value is None:
            return False
        return surprise_value >= self.surprise_threshold

    def _record_gate(self, level_name: str, *, hit: bool) -> None:
        stats_key = f"gate.{level_name}"
        self.last_update_stats.setdefault(stats_key, {})
        self.last_update_stats[stats_key]["gate_hit"] = 1.0 if hit else 0.0

    def _update_cms(
        self,
        cms_inputs: dict[str, torch.Tensor],
        cms_outputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _update_cms_fast(
        self,
        fast_state: BlockFastState,
        cms_inputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _update_cms_chunk(
        self,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        with torch.enable_grad():
            prediction = self.cms.blocks[level_name](chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        context_vec = chunk_inputs.mean(dim=(0, 1))
        magnitude = self.level_manager.optimize(
            level_name,
            self.cms.blocks[level_name],
            loss,
            context=context_vec,
            force=True,
        )
        self.level_manager.pop_last_metrics(level_name)
        return magnitude

    def _update_cms_chunk_fast(
        self,
        fast_state: BlockFastState,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        base_params = fast_state.cms_params[level_name]
        forward_params = params_with_deltas(self.cms.blocks[level_name], base_params)
        params_req = require_grad_params(forward_params)
        with torch.enable_grad():
            prediction = call_with_params(self.cms.blocks[level_name], params_req, chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        grads = torch.autograd.grad(
            loss,
            tuple(params_req.values()),
            retain_graph=False,
            allow_unused=True,
        )
        grads_dict = grads_to_dict(params_req, grads)
        context_vec = chunk_inputs.mean(dim=(0, 1))
        updated, magnitude = fast_state.level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=True,
        )
        fast_state.cms_params[level_name] = updated
        fast_state.level_manager.pop_last_metrics(level_name)
        return magnitude


class HOPEBlock(nn.Module):
    def __init__(self, config: HOPEBlockConfig):
        super().__init__()
        self.config = config
        self.last_update_stats: Dict[str, Dict[str, float]] = {}
        self.surprise_threshold: float | None = None
        self.allowed_levels: Set[str] | None = None
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
        fast_state: BlockFastState | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(x)
        if fast_state is None:
            mem_out = self.titan_memory(attn_out)
            combined = attn_out + mem_out
            if teach_signal is not None and self.config.cms_online_updates:
                cms_out = self._cms_forward_online(combined, teach_signal, surprise_value)
                self._update_titan(attn_out, mem_out, teach_signal, surprise_value)
            else:
                cms_result = self.cms(combined, return_intermediates=True)
                cms_out, cms_inputs, cms_outputs = cms_result
                if teach_signal is not None:
                    self._update_titan(attn_out, mem_out, teach_signal, surprise_value)
                    self._update_cms(cms_inputs, cms_outputs, teach_signal, surprise_value)
            self.level_manager.tick()
            return cms_out

        if fast_state.titan_params is None:
            raise ValueError("fast_state.titan_params is required for HOPEBlock fast-state forward")
        mem_out = call_with_deltas(self.titan_memory, fast_state.titan_params, attn_out)
        combined = attn_out + mem_out
        if teach_signal is not None and self.config.cms_online_updates:
            cms_out = self._cms_forward_online_fast(
                combined, fast_state, teach_signal, surprise_value
            )
            self._update_titan_fast(fast_state, attn_out, mem_out, teach_signal, surprise_value)
        else:
            cms_out, cms_inputs = self._cms_forward_fast(combined, fast_state)
            if teach_signal is not None:
                self._update_titan_fast(fast_state, attn_out, mem_out, teach_signal, surprise_value)
                self._update_cms_fast(fast_state, cms_inputs, teach_signal, surprise_value)
        fast_state.level_manager.tick()
        return cms_out

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def set_allowed_levels(self, allowed: Set[str] | None) -> None:
        self.allowed_levels = allowed.copy() if allowed is not None else None

    def _cms_forward_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current = x
        inputs: dict[str, torch.Tensor] = {}
        for spec in self.config.cms_levels:
            level_name = spec.name
            inputs[level_name] = current
            params = fast_state.cms_params[level_name]
            current = call_with_deltas(self.cms.blocks[level_name], params, current)
        return current, inputs


    def _cms_forward_online(
        self,
        x: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                current = self.cms.blocks[level_name](current)
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk(
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)

    def _cms_forward_online_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                params = fast_state.cms_params[level_name]
                current = call_with_deltas(self.cms.blocks[level_name], params, current)
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk_fast(
                        fast_state,
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)
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
        # Use full sequence for granular updates (Critique P1)
        # Note: We intentionally do not pool over dim=1 (sequence) here.
        # teach_signal is (B, T, D), attn_out is (B, T, D)
        modifier = self.self_modifier(
            key=attn_out.detach(),
            value=mem_out.detach(),
            error_signal=teach_signal.detach(),
        )
        context_vec = attn_out.detach().mean(dim=(0, 1))

        with torch.enable_grad():
            query = attn_out.detach()
            target = (modifier - teach_signal.detach()).detach()
            prediction = self.titan_memory(query)
            loss_terms = F.mse_loss(prediction, target, reduction="none")
            active = teach_signal.detach().abs().sum(dim=-1, keepdim=True) > 0
            mask = active.float()
            if self.surprise_threshold is not None:
                norms = teach_signal.norm(dim=-1, keepdim=True)
                mask = mask * (norms >= self.surprise_threshold).float()
            loss = (loss_terms * mask).sum() / mask.sum().clamp(min=1.0)

        magnitude = self.level_manager.optimize(
            level_name, self.titan_memory, loss, context=context_vec
        )
        extra_metrics = self.level_manager.pop_last_metrics(level_name)
        stats = {"grad_norm": magnitude, "gate_hit": 1.0}
        if surprise_value is not None:
            stats["surprise_value"] = surprise_value
        stats.update(extra_metrics)
        self.last_update_stats[f"titan.{level_name}"] = stats

    def _update_titan_fast(
        self,
        fast_state: BlockFastState,
        attn_out: torch.Tensor,
        mem_out: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        level_name = self.config.titan_level.name
        if not self._is_level_allowed("titan"):
            return
        if not fast_state.level_manager.should_update(level_name):
            return
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
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
            target = (modifier - teach_signal.detach()).detach()
            prediction = call_with_params(self.titan_memory, params_req, query)
            loss_terms = F.mse_loss(prediction, target, reduction="none")
            active = teach_signal.detach().abs().sum(dim=-1, keepdim=True) > 0
            mask = active.float()
            if self.surprise_threshold is not None:
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
        updated, magnitude = fast_state.level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=False,
        )
        fast_state.titan_params = updated
        extra_metrics = fast_state.level_manager.pop_last_metrics(level_name)
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
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _update_cms_fast(
        self,
        fast_state: BlockFastState,
        cms_inputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _update_cms_chunk(
        self,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        with torch.enable_grad():
            prediction = self.cms.blocks[level_name](chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        context_vec = chunk_inputs.mean(dim=(0, 1))
        magnitude = self.level_manager.optimize(
            level_name,
            self.cms.blocks[level_name],
            loss,
            context=context_vec,
            force=True,
        )
        self.level_manager.pop_last_metrics(level_name)
        return magnitude

    def _update_cms_chunk_fast(
        self,
        fast_state: BlockFastState,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        base_params = fast_state.cms_params[level_name]
        forward_params = params_with_deltas(self.cms.blocks[level_name], base_params)
        params_req = require_grad_params(forward_params)
        with torch.enable_grad():
            prediction = call_with_params(self.cms.blocks[level_name], params_req, chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        grads = torch.autograd.grad(
            loss,
            tuple(params_req.values()),
            retain_graph=False,
            allow_unused=True,
        )
        grads_dict = grads_to_dict(params_req, grads)
        context_vec = chunk_inputs.mean(dim=(0, 1))
        updated, magnitude = fast_state.level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=True,
        )
        fast_state.cms_params[level_name] = updated
        fast_state.level_manager.pop_last_metrics(level_name)
        return magnitude

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
