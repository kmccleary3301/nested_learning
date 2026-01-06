from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import grad, vmap


@dataclass(frozen=True)
class SelfModifyingTitansConfig:
    dim: int
    eta_scale: float = 1e-3
    chunk_size_other: int = 1
    chunk_size_memory: int | None = None
    objective: str = "l2"
    stopgrad_vhat: bool = True
    use_rank1_precond: bool = True
    use_alpha: bool = True
    momentum: float = 0.0
    qk_l2_norm: bool = True
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.eta_scale <= 0:
            raise ValueError("eta_scale must be positive")
        if self.chunk_size_other <= 0:
            raise ValueError("chunk_size_other must be positive")
        if self.chunk_size_memory is not None and self.chunk_size_memory <= 0:
            raise ValueError("chunk_size_memory must be positive")
        if self.objective not in {"l2", "dot"}:
            raise ValueError("objective must be one of {'l2', 'dot'}")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if self.chunk_size_memory is None:
            object.__setattr__(self, "chunk_size_memory", int(self.chunk_size_other))


@dataclass
class ResidualMLPMemoryState:
    w1: torch.Tensor
    w2: torch.Tensor
    w_skip: torch.Tensor | None = None
    m_w1: torch.Tensor | None = None
    m_w2: torch.Tensor | None = None
    m_w_skip: torch.Tensor | None = None

    def clone(self) -> "ResidualMLPMemoryState":
        return ResidualMLPMemoryState(
            w1=self.w1.detach().clone(),
            w2=self.w2.detach().clone(),
            w_skip=None if self.w_skip is None else self.w_skip.detach().clone(),
            m_w1=None if self.m_w1 is None else self.m_w1.detach().clone(),
            m_w2=None if self.m_w2 is None else self.m_w2.detach().clone(),
            m_w_skip=None if self.m_w_skip is None else self.m_w_skip.detach().clone(),
        )


@dataclass
class SelfModifyingTitansState:
    """
    Fast state for self-modifying Titans.

    Each memory M_□ is a residual MLP (Eq. 91) whose initial parameters are meta-learned
    (stored in the module) and cloned into this fast state per context.
    """

    k: ResidualMLPMemoryState
    v: ResidualMLPMemoryState
    q: ResidualMLPMemoryState
    eta: ResidualMLPMemoryState
    alpha: ResidualMLPMemoryState
    memory: ResidualMLPMemoryState

    def clone(self) -> "SelfModifyingTitansState":
        return SelfModifyingTitansState(
            k=self.k.clone(),
            v=self.v.clone(),
            q=self.q.clone(),
            eta=self.eta.clone(),
            alpha=self.alpha.clone(),
            memory=self.memory.clone(),
        )


class ResidualMLPMemory(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or hidden_dim <= 0:
            raise ValueError("in_dim/out_dim/hidden_dim must be positive")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation = activation
        self.w2 = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        self.w1 = nn.Linear(self.hidden_dim, self.out_dim, bias=False)
        self.w_skip: nn.Linear | None = None
        if self.in_dim != self.out_dim:
            self.w_skip = nn.Linear(self.in_dim, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x if self.w_skip is None else self.w_skip(x)
        hidden = self.activation(self.w2(x))
        return residual + self.w1(hidden)


class SelfModifyingTitans(nn.Module):
    """
    Self-modifying Titans (Nested Learning paper, Eqs. 83–93), correctness-first.

    - Multiple memories: M_k, M_v, M_q, M_eta, M_alpha, M_memory.
    - Each memory is a 2-layer residual MLP (Eq. 91).
    - Updates are performed on fast state using chunked DGD-like rule (Eq. 90/93).

    Note: This implementation prioritizes semantic fidelity and testability over speed.
    """

    def __init__(self, config: SelfModifyingTitansConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        hidden = dim
        act = F.gelu
        self.m_k = ResidualMLPMemory(in_dim=dim, out_dim=dim, hidden_dim=hidden, activation=act)
        self.m_v = ResidualMLPMemory(in_dim=dim, out_dim=dim, hidden_dim=hidden, activation=act)
        self.m_q = ResidualMLPMemory(in_dim=dim, out_dim=dim, hidden_dim=hidden, activation=act)
        self.m_eta = ResidualMLPMemory(in_dim=dim, out_dim=1, hidden_dim=hidden, activation=act)
        self.m_alpha = ResidualMLPMemory(in_dim=dim, out_dim=1, hidden_dim=hidden, activation=act)
        self.m_memory = ResidualMLPMemory(
            in_dim=dim, out_dim=dim, hidden_dim=hidden, activation=act
        )

    def init_fast_state(self) -> SelfModifyingTitansState:
        return SelfModifyingTitansState(
            k=self._init_memory_state(self.m_k),
            v=self._init_memory_state(self.m_v),
            q=self._init_memory_state(self.m_q),
            eta=self._init_memory_state(self.m_eta),
            alpha=self._init_memory_state(self.m_alpha),
            memory=self._init_memory_state(self.m_memory),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        q = self.m_q(x)
        if self.config.qk_l2_norm:
            q = F.normalize(q, dim=-1, eps=self.config.eps)
        return self.m_memory(q)

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: SelfModifyingTitansState,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected x to have shape (B, T, D)")
        batch, _seq_len, dim = x.shape
        if dim != self.config.dim:
            raise ValueError(f"Expected dim={self.config.dim}, got {dim}")
        state = self._ensure_batched_state(state, batch)
        q = self._memory_forward(x, state.q)
        if self.config.qk_l2_norm:
            q = F.normalize(q, dim=-1, eps=self.config.eps)
        return self._memory_forward(q, state.memory)

    def forward_with_updates(
        self,
        x: torch.Tensor,
        state: SelfModifyingTitansState,
        *,
        chunk_size_other: int | None = None,
        chunk_size_memory: int | None = None,
    ) -> tuple[torch.Tensor, SelfModifyingTitansState]:
        if x.ndim != 3:
            raise ValueError("Expected x to have shape (B, T, D)")
        batch, seq_len, dim = x.shape
        if dim != self.config.dim:
            raise ValueError(f"Expected dim={self.config.dim}, got {dim}")
        state = self._ensure_batched_state(state, batch)
        other_chunk = int(
            self.config.chunk_size_other if chunk_size_other is None else chunk_size_other
        )
        memory_chunk = int(
            self.config.chunk_size_memory if chunk_size_memory is None else chunk_size_memory
        )
        if other_chunk <= 0 or memory_chunk <= 0:
            raise ValueError("chunk sizes must be positive")

        outputs: list[torch.Tensor] = []
        other_k: list[torch.Tensor] = []
        other_v: list[torch.Tensor] = []
        other_eta: list[torch.Tensor] = []
        other_alpha: list[torch.Tensor] = []
        memory_k: list[torch.Tensor] = []
        memory_v: list[torch.Tensor] = []
        memory_eta: list[torch.Tensor] = []
        memory_alpha: list[torch.Tensor] = []

        def _next_boundary(idx: int, *, chunk_size: int) -> int:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
            return min(((idx // chunk_size) + 1) * chunk_size, seq_len)

        with torch.no_grad():
            idx = 0
            while idx < seq_len:
                next_other = _next_boundary(idx, chunk_size=other_chunk)
                next_memory = _next_boundary(idx, chunk_size=memory_chunk)
                end = min(next_other, next_memory, seq_len)
                x_chunk = x[:, idx:end, :]

                k_chunk = self._memory_forward(x_chunk, state.k)
                v_chunk = self._memory_forward(x_chunk, state.v)
                q_chunk = self._memory_forward(x_chunk, state.q)
                if self.config.qk_l2_norm:
                    k_chunk = F.normalize(k_chunk, dim=-1, eps=self.config.eps)
                    q_chunk = F.normalize(q_chunk, dim=-1, eps=self.config.eps)
                eta_chunk = self._memory_forward(x_chunk, state.eta).squeeze(-1)
                eta_chunk = F.softplus(eta_chunk) * self.config.eta_scale
                if self.config.use_alpha:
                    alpha_chunk = self._memory_forward(x_chunk, state.alpha).squeeze(-1)
                    alpha_chunk = torch.sigmoid(alpha_chunk)
                else:
                    alpha_chunk = torch.ones_like(eta_chunk)
                o_chunk = self._memory_forward(q_chunk, state.memory)
                outputs.append(o_chunk)

                other_k.append(k_chunk)
                other_v.append(v_chunk)
                other_eta.append(eta_chunk)
                other_alpha.append(alpha_chunk)
                memory_k.append(k_chunk)
                memory_v.append(v_chunk)
                memory_eta.append(eta_chunk)
                memory_alpha.append(alpha_chunk)

                idx = end

                if idx == next_other and other_k:
                    other_memories: tuple[str, ...] = ("k", "v", "q", "eta")
                    if self.config.use_alpha:
                        other_memories = (*other_memories, "alpha")
                    self._apply_chunk_update_seq(
                        state,
                        k_seq=torch.cat(other_k, dim=1),
                        v_seq=torch.cat(other_v, dim=1),
                        eta_seq=torch.cat(other_eta, dim=1),
                        alpha_seq=torch.cat(other_alpha, dim=1),
                        memories=other_memories,
                    )
                    other_k.clear()
                    other_v.clear()
                    other_eta.clear()
                    other_alpha.clear()

                if idx == next_memory and memory_k:
                    self._apply_chunk_update_seq(
                        state,
                        k_seq=torch.cat(memory_k, dim=1),
                        v_seq=torch.cat(memory_v, dim=1),
                        eta_seq=torch.cat(memory_eta, dim=1),
                        alpha_seq=torch.cat(memory_alpha, dim=1),
                        memories=("memory",),
                    )
                    memory_k.clear()
                    memory_v.clear()
                    memory_eta.clear()
                    memory_alpha.clear()

            if other_k:
                other_memories = ("k", "v", "q", "eta")
                if self.config.use_alpha:
                    other_memories = (*other_memories, "alpha")
                self._apply_chunk_update_seq(
                    state,
                    k_seq=torch.cat(other_k, dim=1),
                    v_seq=torch.cat(other_v, dim=1),
                    eta_seq=torch.cat(other_eta, dim=1),
                    alpha_seq=torch.cat(other_alpha, dim=1),
                    memories=other_memories,
                )
            if memory_k:
                self._apply_chunk_update_seq(
                    state,
                    k_seq=torch.cat(memory_k, dim=1),
                    v_seq=torch.cat(memory_v, dim=1),
                    eta_seq=torch.cat(memory_eta, dim=1),
                    alpha_seq=torch.cat(memory_alpha, dim=1),
                    memories=("memory",),
                )

        return torch.cat(outputs, dim=1), state

    def _apply_chunk_update(
        self,
        state: SelfModifyingTitansState,
        buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        *,
        memories: tuple[str, ...],
    ) -> None:
        if not buffer:
            return
        k_seq = torch.stack([item[0] for item in buffer], dim=1)
        v_seq = torch.stack([item[1] for item in buffer], dim=1)
        eta_seq = torch.stack([item[2] for item in buffer], dim=1)
        alpha_seq = torch.stack([item[3] for item in buffer], dim=1)
        self._apply_chunk_update_seq(
            state,
            k_seq=k_seq,
            v_seq=v_seq,
            eta_seq=eta_seq,
            alpha_seq=alpha_seq,
            memories=memories,
        )

    def _apply_chunk_update_seq(
        self,
        state: SelfModifyingTitansState,
        *,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        eta_seq: torch.Tensor,
        alpha_seq: torch.Tensor,
        memories: tuple[str, ...],
    ) -> None:
        steps = k_seq.size(1)
        dim = self.config.dim
        eye = (
            torch.eye(dim, device=k_seq.device, dtype=k_seq.dtype)
            .unsqueeze(0)
            .expand(k_seq.size(0), -1, -1)
        )

        boundary: dict[str, ResidualMLPMemoryState] = {
            name: getattr(state, name).clone() for name in memories
        }
        grads = {name: self._memory_grads_chunk(boundary[name], k_seq, v_seq) for name in memories}

        for t in range(steps):
            k_t = k_seq[:, t, :]
            eta_t = eta_seq[:, t]
            alpha_t = alpha_seq[:, t]
            kk = torch.einsum("bi,bj->bij", k_t, k_t)
            precond = alpha_t[:, None, None] * eye - eta_t[:, None, None] * kk
            for name in memories:
                fast = getattr(state, name)
                g1, g2, gskip = grads[name]
                self._apply_param_update(
                    fast,
                    (
                        g1[:, t, ...],
                        g2[:, t, ...],
                        None if gskip is None else gskip[:, t, ...],
                    ),
                    eta_t,
                    alpha_t,
                    precond,
                )

    def _memory_grads(
        self,
        frozen: ResidualMLPMemoryState,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.enable_grad():
            w1 = frozen.w1.detach().requires_grad_(True)
            w2 = frozen.w2.detach().requires_grad_(True)
            w_skip = None
            if frozen.w_skip is not None:
                w_skip = frozen.w_skip.detach().requires_grad_(True)

            pred = self._memory_forward(k_t, ResidualMLPMemoryState(w1=w1, w2=w2, w_skip=w_skip))
            vhat = self._memory_forward(v_t, ResidualMLPMemoryState(w1=w1, w2=w2, w_skip=w_skip))
            if self.config.stopgrad_vhat:
                vhat = vhat.detach()

            if self.config.objective == "dot":
                loss = -(pred * vhat).sum(dim=-1)
            else:
                loss = (pred - vhat).pow(2).sum(dim=-1)
            loss_scalar = loss.sum()

            grads = torch.autograd.grad(
                loss_scalar,
                (w1, w2, w_skip) if w_skip is not None else (w1, w2),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        if w_skip is None:
            g1, g2 = grads
            return g1, g2, None
        g1, g2, gskip = grads
        return g1, g2, gskip

    def _memory_grads_chunk(
        self,
        frozen: ResidualMLPMemoryState,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Compute per-token gradients for an entire chunk in parallel (paper §8.2).

        Returns gradients with leading shape (B, T, ...).
        """
        w1 = frozen.w1.detach()
        w2 = frozen.w2.detach()
        w_skip = None if frozen.w_skip is None else frozen.w_skip.detach()

        k_tokens = k_seq.transpose(0, 1)
        v_tokens = v_seq.transpose(0, 1)

        if w_skip is None:

            def loss_fn(
                w1_t: torch.Tensor,
                w2_t: torch.Tensor,
                k_t: torch.Tensor,
                v_t: torch.Tensor,
            ) -> torch.Tensor:
                mem = ResidualMLPMemoryState(w1=w1_t, w2=w2_t)
                pred = self._memory_forward(k_t, mem)
                vhat = self._memory_forward(v_t, mem)
                if self.config.stopgrad_vhat:
                    vhat = vhat.detach()
                if self.config.objective == "dot":
                    loss = -(pred * vhat).sum(dim=-1)
                else:
                    loss = (pred - vhat).pow(2).sum(dim=-1)
                return loss.sum()

            grad_fn = grad(loss_fn, argnums=(0, 1))
            g1_tokens, g2_tokens = vmap(grad_fn, in_dims=(None, None, 0, 0))(
                w1,
                w2,
                k_tokens,
                v_tokens,
            )
            return g1_tokens.transpose(0, 1), g2_tokens.transpose(0, 1), None

        def loss_fn(
            w1_t: torch.Tensor,
            w2_t: torch.Tensor,
            w_skip_t: torch.Tensor,
            k_t: torch.Tensor,
            v_t: torch.Tensor,
        ) -> torch.Tensor:
            mem = ResidualMLPMemoryState(w1=w1_t, w2=w2_t, w_skip=w_skip_t)
            pred = self._memory_forward(k_t, mem)
            vhat = self._memory_forward(v_t, mem)
            if self.config.stopgrad_vhat:
                vhat = vhat.detach()
            if self.config.objective == "dot":
                loss = -(pred * vhat).sum(dim=-1)
            else:
                loss = (pred - vhat).pow(2).sum(dim=-1)
            return loss.sum()

        grad_fn = grad(loss_fn, argnums=(0, 1, 2))
        g1_tokens, g2_tokens, gskip_tokens = vmap(
            grad_fn,
            in_dims=(None, None, None, 0, 0),
        )(w1, w2, w_skip, k_tokens, v_tokens)
        return (
            g1_tokens.transpose(0, 1),
            g2_tokens.transpose(0, 1),
            gskip_tokens.transpose(0, 1),
        )

    def _apply_param_update(
        self,
        fast: ResidualMLPMemoryState,
        grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        eta_t: torch.Tensor,
        alpha_t: torch.Tensor,
        precond: torch.Tensor,
    ) -> None:
        g1, g2, gskip = grads
        g1 = self._apply_momentum(fast, "m_w1", g1)
        g2 = self._apply_momentum(fast, "m_w2", g2)
        if self.config.use_rank1_precond:
            fast.w2 = torch.matmul(fast.w2, precond) - eta_t[:, None, None] * g2
        else:
            fast.w2 = alpha_t[:, None, None] * fast.w2 - eta_t[:, None, None] * g2
        fast.w1 = alpha_t[:, None, None] * fast.w1 - eta_t[:, None, None] * g1

        if fast.w_skip is None:
            return
        if gskip is None:
            raise RuntimeError("Expected w_skip grad to be present")
        gskip = self._apply_momentum(fast, "m_w_skip", gskip)
        if self.config.use_rank1_precond:
            fast.w_skip = torch.matmul(fast.w_skip, precond) - eta_t[:, None, None] * gskip
        else:
            fast.w_skip = alpha_t[:, None, None] * fast.w_skip - eta_t[:, None, None] * gskip

    def _apply_momentum(
        self,
        fast: ResidualMLPMemoryState,
        attr_name: str,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        beta = float(self.config.momentum)
        if beta <= 0.0:
            return grad
        buf = getattr(fast, attr_name)
        if buf is None:
            buf = torch.zeros_like(grad)
        buf = beta * buf + grad
        setattr(fast, attr_name, buf)
        return buf

    def _init_memory_state(self, module: ResidualMLPMemory) -> ResidualMLPMemoryState:
        skip = None if module.w_skip is None else module.w_skip.weight.detach().clone()
        return ResidualMLPMemoryState(
            w1=module.w1.weight.detach().clone(),
            w2=module.w2.weight.detach().clone(),
            w_skip=skip,
        )

    def _ensure_batched_state(
        self, state: SelfModifyingTitansState, batch: int
    ) -> SelfModifyingTitansState:
        if state.k.w1.ndim == 2:
            return SelfModifyingTitansState(
                k=self._expand_memory_state(state.k, batch),
                v=self._expand_memory_state(state.v, batch),
                q=self._expand_memory_state(state.q, batch),
                eta=self._expand_memory_state(state.eta, batch),
                alpha=self._expand_memory_state(state.alpha, batch),
                memory=self._expand_memory_state(state.memory, batch),
            )
        if state.k.w1.ndim != 3:
            raise ValueError("SelfModifyingTitansState weights must be 2D or 3D tensors")
        if state.k.w1.size(0) != batch:
            raise ValueError(
                f"State batch mismatch: expected batch={batch}, got {state.k.w1.size(0)}"
            )
        return state

    def _expand_memory_state(
        self, mem: ResidualMLPMemoryState, batch: int
    ) -> ResidualMLPMemoryState:
        def _expand(t: torch.Tensor) -> torch.Tensor:
            return t.detach().clone().unsqueeze(0).repeat(batch, 1, 1)

        def _expand_opt(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else _expand(t)

        return ResidualMLPMemoryState(
            w1=_expand(mem.w1),
            w2=_expand(mem.w2),
            w_skip=_expand_opt(mem.w_skip),
            m_w1=_expand_opt(mem.m_w1),
            m_w2=_expand_opt(mem.m_w2),
            m_w_skip=_expand_opt(mem.m_w_skip),
        )

    def _memory_forward(self, x: torch.Tensor, mem: ResidualMLPMemoryState) -> torch.Tensor:
        if x.ndim == 2:
            x_seq = x.unsqueeze(1)
            squeeze = True
        else:
            x_seq = x
            squeeze = False
        w2_t = mem.w2.transpose(-1, -2)
        hidden = torch.matmul(x_seq, w2_t)
        hidden = F.gelu(hidden)
        w1_t = mem.w1.transpose(-1, -2)
        out = torch.matmul(hidden, w1_t)
        if mem.w_skip is None:
            out = out + x_seq
        else:
            w_skip_t = mem.w_skip.transpose(-1, -2)
            out = out + torch.matmul(x_seq, w_skip_t)
        if squeeze:
            return out.squeeze(1)
        return out
