from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.func import functional_call

ParamDict = Dict[str, torch.Tensor]


def params_with_deltas(module: nn.Module, deltas: ParamDict) -> ParamDict:
    params: ParamDict = {}
    missing: list[str] = []
    for name, param in module.named_parameters():
        delta = deltas.get(name)
        if delta is None:
            missing.append(name)
            continue
        params[name] = param + delta
    if missing:
        raise KeyError(
            f"Missing fast-state delta(s) for {module.__class__.__name__}: {sorted(missing)[:10]}"
        )
    return params


def module_buffers(module: nn.Module) -> ParamDict:
    return {name: buf for name, buf in module.named_buffers()}


def call_with_params(
    module: nn.Module,
    params: ParamDict,
    *args: Any,
    **kwargs: Any,
) -> Any:
    buffers = module_buffers(module)
    return functional_call(module, (params, buffers), args, kwargs, strict=True)


def call_with_deltas(
    module: nn.Module,
    deltas: ParamDict,
    *args: Any,
    **kwargs: Any,
) -> Any:
    return call_with_params(module, params_with_deltas(module, deltas), *args, **kwargs)


def require_grad_params(params: ParamDict) -> ParamDict:
    return {name: value.detach().requires_grad_(True) for name, value in params.items()}


def grads_to_dict(params: ParamDict, grads: Tuple[torch.Tensor | None, ...]) -> ParamDict:
    out: ParamDict = {}
    for (name, _), grad in zip(params.items(), grads, strict=True):
        if grad is None:
            continue
        out[name] = grad
    return out
