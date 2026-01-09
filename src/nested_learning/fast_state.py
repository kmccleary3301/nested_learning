from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, cast

import torch
from torch import nn

from .optim.manager import LevelConfig, LevelOptimizerManager
from .titan.self_modifying import SelfModifyingTitansState

ParamDict = Dict[str, torch.Tensor]


def init_module_deltas(module: nn.Module) -> ParamDict:
    """
    Initialize a per-parameter "fast state" delta dict for meta+delta fast state.

    The fast state stores *deltas* (initialized to 0) rather than detached parameter clones so that
    forward passes can use `meta_param + delta`, allowing outer gradients to flow to meta params
    while keeping online updates as stop-grad writes into the delta tensors.
    """

    return {name: torch.zeros_like(param).detach() for name, param in module.named_parameters()}


@dataclass
class BlockFastState:
    titan_params: ParamDict | None
    cms_params: Dict[str, ParamDict]
    level_manager: LevelOptimizerManager
    selfmod_state: SelfModifyingTitansState | None = None


def build_block_fast_state(
    *,
    titan_module: nn.Module | None,
    cms_blocks: Dict[str, nn.Module],
    selfmod_module: nn.Module | None = None,
    specs,
    optimizer_configs: Dict[str, dict],
    default_lr: float,
) -> BlockFastState:
    titan_params = None
    if titan_module is not None:
        titan_params = init_module_deltas(titan_module)
    cms_params = {name: init_module_deltas(block) for name, block in cms_blocks.items()}
    level_cfg = LevelConfig(specs=specs, optimizer_configs=optimizer_configs, default_lr=default_lr)
    level_manager = LevelOptimizerManager(level_cfg)
    selfmod_state = None
    if selfmod_module is not None:
        init_fn = getattr(selfmod_module, "init_fast_state", None)
        if callable(init_fn):
            selfmod_state = cast(SelfModifyingTitansState, init_fn())
    return BlockFastState(
        titan_params=titan_params,
        cms_params=cms_params,
        level_manager=level_manager,
        selfmod_state=selfmod_state,
    )


@dataclass
class ModelFastState:
    blocks: list[BlockFastState]
