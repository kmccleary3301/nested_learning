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
        self.state = DeepMomentumState()
        self.nonlinearity = nn.Tanh() if variant in {"dmgd", "muon"} else nn.Identity()
        self.last_metrics: dict[str, float] = {}

    def reset_state(self) -> None:
        self.state = DeepMomentumState()

    def _precondition(self, grad: torch.Tensor) -> torch.Tensor:
        if self.state.sq_avg is None or self.state.sq_avg.shape != grad.shape:
            self.state.sq_avg = torch.zeros_like(grad)
        self.state.sq_avg.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = self.state.sq_avg.sqrt().add_(self.eps)
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
        
        # Fix: Removed orthogonalization which was zeroing out updates parallel to context.
        # For now, we return the gradient as-is (Standard GD behavior).
        # Future work: Implement exact Eq 29 update: W_new = W_old(I - xx^T) - eta*grad
        
        return grad, metrics

    def forward(self, grad: torch.Tensor, *, context: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        if self.state.grad_avg is None or self.state.grad_avg.shape != grad.shape:
            self.state.grad_avg = torch.zeros_like(grad)
        self.last_metrics = {}
        update = grad
        if self.variant in {"preconditioned", "muon"}:
            update = self._precondition(grad)
        if self.variant == "l2_objective":
            update = grad + 0.1 * torch.mean(grad, dim=-1, keepdim=True)
        if self.variant == "nl_l2_precond":
            update, metrics = self._nl_precondition(grad, context)
            self.last_metrics.update(metrics)
        if self.variant in {"dmgd", "muon"}:
            update = self.nonlinearity(update)
        self.state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)
        return self.state.grad_avg
