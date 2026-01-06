from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DeepMomentumState:
    grad_avg: Optional[torch.Tensor] = None
    sq_avg: Optional[torch.Tensor] = None


class DeepMomentum(nn.Module):
    """Implements momentum variants described in the NL paper."""

    def __init__(
        self,
        *,
        beta: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        variant: str = "preconditioned",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.beta2 = beta2
        self.eps = eps
        self.variant = variant
        self.state: dict[str, DeepMomentumState] = {}
        self.nonlinearity = nn.Tanh() if variant in {"dmgd", "muon"} else nn.Identity()
        self.last_metrics: dict[str, float] = {}

    def reset_state(self) -> None:
        self.state.clear()

    def _precondition(self, grad: torch.Tensor, state: DeepMomentumState) -> torch.Tensor:
        if state.sq_avg is None or state.sq_avg.shape != grad.shape:
            state.sq_avg = torch.zeros_like(grad)
        state.sq_avg.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = state.sq_avg.sqrt().add_(self.eps)
        return grad / denom

    def _nl_precondition(
        self,
        grad: torch.Tensor,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        metrics: dict[str, float] = {"ctx_norm": 0.0, "proj_norm": 0.0}
        if context is None:
            return grad, metrics
        ctx = context
        if ctx.ndim > 1:
            ctx = ctx.reshape(-1, ctx.shape[-1]).mean(dim=0)
        ctx_norm = torch.norm(ctx)
        metrics["ctx_norm"] = ctx_norm.item()

        if ctx_norm > 0:
            unit = ctx / (ctx_norm + self.eps)
            # Project grad orthogonal to context (rank-1 projector).
            projection = (grad * unit).sum(dim=-1, keepdim=True) * unit
            update = grad - projection
            metrics["proj_norm"] = torch.norm(update).item()
            return update, metrics
        return grad, metrics

    def forward(  # type: ignore[override]
        self,
        grad: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        param_key: str | None = None,
    ) -> torch.Tensor:
        key = param_key or "__default__"
        state = self.state.get(key)
        if state is None:
            state = DeepMomentumState()
            self.state[key] = state
        if state.grad_avg is None or state.grad_avg.shape != grad.shape:
            state.grad_avg = torch.zeros_like(grad)
        self.last_metrics = {}
        update = grad
        if self.variant in {"preconditioned", "muon"}:
            update = self._precondition(grad, state)
        if self.variant == "l2_objective":
            update = grad + 0.1 * torch.mean(grad, dim=-1, keepdim=True)
        if self.variant == "nl_l2_precond":
            update, metrics = self._nl_precondition(grad, context)
            self.last_metrics.update(metrics)
        if self.variant in {"dmgd", "muon"}:
            update = self.nonlinearity(update)
        state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)
        return state.grad_avg
