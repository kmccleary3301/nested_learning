from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .fast_state import ModelFastState, build_block_fast_state
from .hope.block import (
    HOPEAttentionBlock,
    HOPEAttentionBlockConfig,
    HOPEBlock,
    HOPEBlockConfig,
    HOPESelfModBlock,
    HOPESelfModBlockConfig,
)
from .levels import LevelSpec
from .transformer import TransformerBlock, TransformerBlockConfig


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
    gradient_checkpointing: bool = False
    surprise_threshold: float | None = None
    freeze_backbone: bool = False
    qk_l2_norm: bool = False
    local_conv_window: int | None = None
    self_mod_lr: float = 1e-3
    self_mod_hidden: int = 4
    self_mod_chunk_size: int = 1
    self_mod_chunk_size_memory: int | None = None
    self_mod_objective: str = "l2"
    self_mod_stopgrad_vhat: bool = True
    self_mod_use_rank1_precond: bool = True
    self_mod_momentum: float = 0.0
    transformer_mlp_hidden_multiplier: int = 4
    transformer_activation: str = "gelu"
    block_variant: str = "hope_hybrid"


class HOPEModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.base_teach_scale = config.teach_scale
        self.base_teach_clip = config.teach_clip
        self._runtime_teach_scale = config.teach_scale
        self._runtime_teach_clip = config.teach_clip
        self.gradient_checkpointing = config.gradient_checkpointing
        self._surprise_threshold = config.surprise_threshold
        self._allowed_update_levels: set[str] | None = None
        variant = str(config.block_variant).strip().lower()
        if variant == "hope_attention":
            block_config = HOPEAttentionBlockConfig(
                dim=config.dim,
                heads=config.heads,
                cms_levels=config.cms_levels,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
                self_mod_lr=config.self_mod_lr,
                optimizer_configs=config.optimizers or {},
            )
            self.blocks = nn.ModuleList(
                [HOPEAttentionBlock(block_config) for _ in range(config.num_layers)]
            )
        elif variant == "hope_hybrid":
            block_config = HOPEBlockConfig(
                dim=config.dim,
                heads=config.heads,
                titan_level=config.titan_level,
                cms_levels=config.cms_levels,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
                self_mod_lr=config.self_mod_lr,
                self_mod_hidden=config.self_mod_hidden,
                optimizer_configs=config.optimizers or {},
            )
            self.blocks = nn.ModuleList([HOPEBlock(block_config) for _ in range(config.num_layers)])
        elif variant == "hope_selfmod":
            block_config = HOPESelfModBlockConfig(
                dim=config.dim,
                cms_levels=config.cms_levels,
                qk_l2_norm=config.qk_l2_norm,
                eta_scale=config.self_mod_lr,
                selfmod_chunk_size=config.self_mod_chunk_size,
                selfmod_chunk_size_memory=config.self_mod_chunk_size_memory,
                selfmod_objective=config.self_mod_objective,
                selfmod_stopgrad_vhat=config.self_mod_stopgrad_vhat,
                selfmod_use_rank1_precond=config.self_mod_use_rank1_precond,
                selfmod_momentum=config.self_mod_momentum,
                self_mod_lr=config.self_mod_lr,
                optimizer_configs=config.optimizers or {},
            )
            self.blocks = nn.ModuleList(
                [HOPESelfModBlock(block_config) for _ in range(config.num_layers)]
            )
        elif variant == "transformer":
            block_config = TransformerBlockConfig(
                dim=config.dim,
                heads=config.heads,
                mlp_hidden_multiplier=config.transformer_mlp_hidden_multiplier,
                activation=config.transformer_activation,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
            )
            self.blocks = nn.ModuleList(
                [TransformerBlock(block_config) for _ in range(config.num_layers)]
            )
        else:
            raise ValueError(
                f"Unsupported block_variant={config.block_variant!r}; expected one of "
                "['hope_attention', 'hope_hybrid', 'hope_selfmod', 'transformer']"
            )
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Weight tying keeps the LM head gradient aligned with the embedding space.
        self.lm_head.weight = self.embed.weight
        self._latest_update_metrics: Dict[str, float] = {}
        self.set_surprise_threshold(self._surprise_threshold)
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
            block.set_surprise_threshold(threshold)

    def get_surprise_threshold(self) -> float | None:
        return self._surprise_threshold

    def set_allowed_update_levels(self, levels: set[str] | None) -> None:
        self._allowed_update_levels = levels.copy() if levels is not None else None
        for block in self.blocks:
            block.set_allowed_levels(self._allowed_update_levels)

    def get_allowed_update_levels(self) -> set[str] | None:
        return None if self._allowed_update_levels is None else self._allowed_update_levels.copy()

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        fast_state: ModelFastState | None = None,
    ) -> torch.Tensor:
        x = self.embed(tokens)
        surprise_value: float | None = None
        if teach_signal is not None:
            surprise_value = float(teach_signal.norm(dim=-1).mean().item())
        if fast_state is not None and len(fast_state.blocks) != len(self.blocks):
            raise ValueError("fast_state.blocks length does not match model.blocks")
        for idx, block in enumerate(self.blocks):
            block_state = None if fast_state is None else fast_state.blocks[idx]
            scaled_signal = None
            if teach_signal is not None:
                scaled_signal = teach_signal * self._runtime_teach_scale
                if self._runtime_teach_clip > 0:
                    norm = scaled_signal.norm(dim=-1, keepdim=True)
                    scale = torch.clamp(norm / self._runtime_teach_clip, min=1.0)
                    scaled_signal = scaled_signal / scale

            def block_call(
                hidden: torch.Tensor,
                *,
                blk=block,
                sig=scaled_signal,
                st=block_state,
            ) -> torch.Tensor:
                return blk(
                    hidden,
                    teach_signal=sig,
                    surprise_value=surprise_value,
                    fast_state=st,
                )

            if self.training and self.gradient_checkpointing:
                x = checkpoint(block_call, x, use_reentrant=False)
            else:
                x = block_call(x)
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

    def init_fast_state(self) -> ModelFastState:
        states = []
        for block in self.blocks:
            if isinstance(block, HOPEBlock):
                specs = [block.config.titan_level, *block.config.cms_levels]
                state = build_block_fast_state(
                    titan_module=block.titan_memory,
                    cms_blocks=dict(block.cms.blocks.items()),
                    specs=specs,
                    optimizer_configs=block.config.optimizer_configs,
                    default_lr=block.config.self_mod_lr,
                )
                states.append(state)
            elif isinstance(block, HOPEAttentionBlock):
                specs = list(block.config.cms_levels)
                state = build_block_fast_state(
                    titan_module=None,
                    cms_blocks=dict(block.cms.blocks.items()),
                    specs=specs,
                    optimizer_configs=block.config.optimizer_configs,
                    default_lr=block.config.self_mod_lr,
                )
                states.append(state)
            elif isinstance(block, HOPESelfModBlock):
                specs = list(block.config.cms_levels)
                state = build_block_fast_state(
                    titan_module=None,
                    cms_blocks=dict(block.cms.blocks.items()),
                    selfmod_module=block.selfmod,
                    specs=specs,
                    optimizer_configs=block.config.optimizer_configs,
                    default_lr=block.config.self_mod_lr,
                )
                states.append(state)
            elif isinstance(block, TransformerBlock):
                state = build_block_fast_state(
                    titan_module=None,
                    cms_blocks={},
                    specs=(),
                    optimizer_configs={},
                    default_lr=0.0,
                )
                states.append(state)
            else:
                raise TypeError(f"Unsupported block type for fast state: {type(block)}")
        return ModelFastState(blocks=states)

    def freeze_backbone(self) -> None:
        """
        Freeze the shared transformer spine (embeddings, attention blocks, norm, LM head).
        HOPE/TITAN/CMS memories remain trainable for adapter-style finetuning.
        """
        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        for block in self.blocks:
            if hasattr(block, "attn"):
                for p in block.attn.parameters():
                    p.requires_grad = False
