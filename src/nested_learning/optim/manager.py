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
        self._last_metrics: Dict[str, Dict[str, float]] = {}
        for spec in config.specs:
            key = spec.optimizer_key or "default"
            optim_cfg = config.optimizer_configs.get(key, {"type": "deep_momentum", "params": {}})
            lr = optim_cfg.get("lr", config.default_lr)
            params_cfg = optim_cfg.get("params", {})
            optimizer = build_optimizer(
                {"type": optim_cfg.get("type", "deep_momentum"), "params": params_cfg}
            )
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
        force: bool = False,
    ) -> float:
        if (not force) and (not self.should_update(level)):
            return 0.0
        named_params: Tuple[Tuple[str, torch.nn.Parameter], ...] = tuple(
            (name, param) for name, param in module.named_parameters() if param.requires_grad
        )
        if not named_params:
            return 0.0
        params = tuple(param for _, param in named_params)
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        optimizer = self.optimizers[level]
        lr = self.learning_rates[level]
        total_norm = 0.0
        with torch.no_grad():
            for (name, param), grad in zip(named_params, grads, strict=True):
                if grad is None:
                    continue
                update = optimizer(grad, context=context, param_key=name)
                param.add_(update, alpha=-lr)
                total_norm += grad.norm().item()
        self.clock.record_update(level)
        metrics = getattr(optimizer, "last_metrics", None)
        if metrics:
            self._last_metrics[level] = dict(metrics)
        else:
            self._last_metrics[level] = {}
        return total_norm

    def tick(self) -> None:
        self.clock.tick()

    def pop_last_metrics(self, level: str) -> Dict[str, float]:
        return self._last_metrics.pop(level, {})

    def apply_grads(
        self,
        level: str,
        params: Dict[str, torch.Tensor],
        grads: Dict[str, torch.Tensor],
        *,
        context: torch.Tensor | None = None,
        force: bool = False,
    ) -> tuple[Dict[str, torch.Tensor], float]:
        if (not force) and (not self.should_update(level)):
            return params, 0.0
        optimizer = self.optimizers[level]
        lr = self.learning_rates[level]
        updated: Dict[str, torch.Tensor] = {}
        total_norm = 0.0
        with torch.no_grad():
            for name, param in params.items():
                grad = grads.get(name)
                if grad is None:
                    updated[name] = param
                    continue
                update = optimizer(grad, context=context, param_key=name)
                updated[name] = (param - lr * update).detach()
                total_norm += grad.norm().item()
        self.clock.record_update(level)
        metrics = getattr(optimizer, "last_metrics", None)
        if metrics:
            self._last_metrics[level] = dict(metrics)
        else:
            self._last_metrics[level] = {}
        return updated, total_norm
