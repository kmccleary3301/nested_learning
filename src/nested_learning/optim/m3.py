from __future__ import annotations

from typing import Iterable

import torch


def _newton_schulz(matrix: torch.Tensor, steps: int, eps: float = 1e-6) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz expects a 2D matrix")
    dtype = matrix.dtype
    device = matrix.device
    m, n = matrix.shape
    x = matrix
    norm = torch.linalg.norm(x)
    x = x / (norm + eps)
    eye = torch.eye(n, device=device, dtype=dtype)
    for _ in range(steps):
        x = 0.5 * x @ (3.0 * eye - x.T @ x)
    return x


def _orthogonalize(tensor: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    if tensor.ndim < 2:
        return tensor
    mat = tensor.reshape(tensor.shape[0], -1)
    ortho = _newton_schulz(mat, steps=steps, eps=eps)
    return ortho.reshape_as(tensor)


class M3(torch.optim.Optimizer):
    """
    Multi-scale Momentum Muon (M3) optimizer (Nested Learning paper, Algorithm 1).

    This is a paper-faithful implementation for 2D weight tensors:
      - M1: fast momentum
      - M2: slow momentum (updated every `slow_chunk` steps)
      - V: second moment
      - O1/O2: Newton-Schulz orthogonalized momenta
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta3: float = 0.9,
        alpha: float = 1.0,
        eps: float = 1e-8,
        ns_steps: int = 3,
        slow_chunk: int = 100,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            ns_steps=ns_steps,
            slow_chunk=slow_chunk,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            beta3 = group["beta3"]
            alpha = group["alpha"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            slow_chunk = group["slow_chunk"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["slow_buffer"] = torch.zeros_like(p)
                    state["o2"] = torch.zeros_like(p)
                state["step"] += 1
                m1 = state["m1"]
                m2 = state["m2"]
                v = state["v"]
                slow_buffer = state["slow_buffer"]

                m1.add_(grad, alpha=beta1)
                v.addcmul_(grad, grad, value=beta2)
                slow_buffer.add_(grad)

                o1 = _orthogonalize(m1, steps=ns_steps, eps=eps)
                o2 = state["o2"]
                denom = v.sqrt().add_(eps)
                update = (o1 + alpha * o2) / denom
                p.add_(update, alpha=-lr)

                if slow_chunk > 0 and state["step"] % slow_chunk == 0:
                    # Paper Algorithm 1 uses the updated slow momentum term in the *next* chunk.
                    # Compute it after applying the current step update to avoid off-by-one usage.
                    m2.add_(slow_buffer, alpha=beta3)
                    slow_buffer.zero_()
                    state["o2"] = _orthogonalize(m2, steps=ns_steps, eps=eps)
        return loss
