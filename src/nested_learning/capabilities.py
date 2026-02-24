from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass
class RuntimeCapabilities:
    python_version: str
    platform: str
    machine: str
    torch_version: str
    cuda_available: bool
    cuda_device_count: int
    cuda_devices: list[str] = field(default_factory=list)
    mps_available: bool = False
    mps_built: bool = False
    distributed_available: bool = False
    compile_available: bool = False
    sdpa_flash_available: bool = False
    sdpa_mem_efficient_available: bool = False
    sdpa_math_available: bool = True
    bf16_supported: bool = False
    fp16_supported: bool = False
    default_device: str = "cpu"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_runtime_capabilities() -> RuntimeCapabilities:
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count() if cuda_available else 0)
    cuda_devices: list[str] = []
    warnings: list[str] = []

    if cuda_available:
        for idx in range(cuda_device_count):
            try:
                name = torch.cuda.get_device_name(idx)
                cuda_devices.append(f"cuda:{idx} {name}")
            except Exception as err:  # pragma: no cover - backend specific
                warnings.append(f"failed to query cuda:{idx}: {err}")

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    mps_built = bool(mps_backend and mps_backend.is_built())

    distributed_available = bool(torch.distributed.is_available())
    compile_available = bool(hasattr(torch, "compile"))

    flash_enabled = False
    mem_eff_enabled = False
    math_enabled = True
    if hasattr(torch.backends, "cuda") and torch.backends.cuda.is_built():
        try:
            flash_enabled = bool(torch.backends.cuda.flash_sdp_enabled())
            mem_eff_enabled = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
            math_enabled = bool(torch.backends.cuda.math_sdp_enabled())
        except Exception as err:  # pragma: no cover - backend specific
            warnings.append(f"failed to query SDPA backend flags: {err}")

    bf16_supported = False
    fp16_supported = False
    if cuda_available:
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
            fp16_supported = True
        except Exception as err:  # pragma: no cover
            warnings.append(f"failed to query CUDA dtype support: {err}")
    elif mps_available:
        fp16_supported = True

    default_device = "cpu"
    if cuda_available:
        default_device = "cuda:0"
    elif mps_available:
        default_device = "mps"

    return RuntimeCapabilities(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_devices=cuda_devices,
        mps_available=mps_available,
        mps_built=mps_built,
        distributed_available=distributed_available,
        compile_available=compile_available,
        sdpa_flash_available=flash_enabled,
        sdpa_mem_efficient_available=mem_eff_enabled,
        sdpa_math_available=math_enabled,
        bf16_supported=bf16_supported,
        fp16_supported=fp16_supported,
        default_device=default_device,
        warnings=warnings,
    )
