from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class AssocMemory(nn.Module):
    """Base class for associative memories with explicit update hooks."""

    def forward(self, query: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError

    @torch.no_grad()
    def update(self, *, key: torch.Tensor, value: torch.Tensor, error_signal: torch.Tensor | None = None) -> None:
        raise NotImplementedError


class SupportsReset(Protocol):
    def reset_state(self) -> None:
        ...
