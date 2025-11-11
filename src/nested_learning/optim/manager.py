from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch
from torch import nn

from ..levels import LevelClock, LevelSpec
from .factory import build_optimizer


@dataclass
class LevelConfig:
    specs: Iterable[LevelSpec]
    optimizer_configs: Dict[str, dict]
    default_lr: float


class LevelOptimizerManager:
    def __init__(self, config: LevelConfig):
        self.clock = LevelClock(config.specs)
        self.learning_rates: Dict[str, float] = {}
        self.optimizers = {}
        for spec in config.specs:
            key = spec.optimizer_key or "default"
            optim_cfg = config.optimizer_configs.get(key, {"type": "deep_momentum", "params": {}})
            lr = optim_cfg.get("lr", config.default_lr)
            params_cfg = optim_cfg.get("params", {})
            optimizer = build_optimizer({"type": optim_cfg.get("type", "deep_momentum"), "params": params_cfg})
            self.optimizers[spec.name] = optimizer
            self.learning_rates[spec.name] = lr

    def should_update(self, level: str) -> bool:
        return self.clock.should_update(level)

    def optimize(
        self,
        level: str,
        module: nn.Module,
        loss: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
    ) -> float:
        if not self.should_update(level):
            return 0.0
        params: Tuple[torch.nn.Parameter, ...] = tuple(module.parameters())
        if not params:
            return 0.0
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        optimizer = self.optimizers[level]
        lr = self.learning_rates[level]
        total_norm = 0.0
        with torch.no_grad():
            for param, grad in zip(params, grads, strict=True):
                if grad is None:
                    continue
                update = optimizer(grad, context=context)
                param.add_(update, alpha=-lr)
                total_norm += grad.norm().item()
        self.clock.record_update(level)
        return total_norm

    def tick(self) -> None:
        self.clock.tick()
