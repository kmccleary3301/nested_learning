from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from nested_learning.training import run_training_loop, unwrap_config


@hydra.main(config_path="configs", config_name="pilot", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    device = _resolve_device(cfg.train.device)
    run_training_loop(cfg, device=device, distributed=False)


def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        parts = device_str.split(":")
        idx = int(parts[1]) if len(parts) > 1 else 0
        if idx >= torch.cuda.device_count():
            idx = torch.cuda.device_count() - 1
        return torch.device(f"cuda:{idx}")
    if device_str.startswith("mps"):
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device(device_str)


if __name__ == "__main__":
    main()
