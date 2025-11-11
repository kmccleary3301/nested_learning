from __future__ import annotations

import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from nested_learning.training import DistributedContext, run_training_loop, unwrap_config


def setup_distributed(backend: str | None = None) -> DistributedContext:
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return DistributedContext(rank=dist.get_rank(), world_size=world_size, device=device)


@hydra.main(config_path="configs", config_name="hope/mid", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    dist_ctx = setup_distributed()
    run_training_loop(cfg, device=dist_ctx.device, distributed=True, dist_ctx=dist_ctx)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
