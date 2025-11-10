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

    def reset_state(self) -> None:
        self.state = DeepMomentumState()

    def _precondition(self, grad: torch.Tensor) -> torch.Tensor:
        if self.state.sq_avg is None or self.state.sq_avg.shape != grad.shape:
            self.state.sq_avg = torch.zeros_like(grad)
        self.state.sq_avg.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = self.state.sq_avg.sqrt().add_(self.eps)
        return grad / denom

    def forward(self, grad: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.state.grad_avg is None or self.state.grad_avg.shape != grad.shape:
            self.state.grad_avg = torch.zeros_like(grad)
        update = grad
        if self.variant in {"preconditioned", "muon"}:
            update = self._precondition(grad)
        if self.variant == "l2_objective":
            update = grad + 0.1 * torch.mean(grad, dim=-1, keepdim=True)
        if self.variant in {"dmgd", "muon"}:
            update = self.nonlinearity(update)
        self.state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)
        return self.state.grad_avg
