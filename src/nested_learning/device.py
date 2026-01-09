from __future__ import annotations

import torch


def resolve_device(device_str: str) -> torch.device:
    normalized = str(device_str).strip().lower()
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        parts = normalized.split(":")
        idx = int(parts[1]) if len(parts) > 1 else 0
        if idx >= torch.cuda.device_count():
            idx = max(torch.cuda.device_count() - 1, 0)
        return torch.device(f"cuda:{idx}")
    if normalized.startswith("mps"):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device(device_str)

