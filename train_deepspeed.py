from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from nested_learning.logging_utils import NullLogger, init_logger
from nested_learning.training import (
    DistributedContext,
    build_dataloader,
    build_model_from_cfg,
    compute_teach_signal,
)

try:
    import deepspeed
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "DeepSpeed is not installed. Install it in this environment to use train_deepspeed.py."
    ) from exc


def setup_distributed() -> DistributedContext:
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    return DistributedContext(rank=rank, world_size=world_size, device=device)


def load_ds_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@hydra.main(config_path="configs", config_name="hope/target", version_base=None)
def main(cfg: DictConfig) -> None:
    dist_ctx = setup_distributed()
    model = build_model_from_cfg(cfg.model)
    ds_config = load_ds_config(cfg.deepspeed.config)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_config,
    )

    dataloader, sampler = build_dataloader(cfg.data, distributed=True, dist_ctx=dist_ctx)
    logger = init_logger(getattr(cfg, "logging", None), cfg) if engine.global_rank == 0 else NullLogger()
    steps = cfg.train.steps
    log_interval = cfg.train.get("log_interval", 10)
    checkpoint_cfg = cfg.train.get("checkpoint", {})
    ckpt_dir = Path(checkpoint_cfg.get("dir", "checkpoints/deepspeed"))

    if checkpoint_cfg.get("resume_tag"):
        tag = checkpoint_cfg["resume_tag"]
        engine.load_checkpoint(str(ckpt_dir), tag=tag)
        if engine.global_rank == 0:
            print(f"[DeepSpeed] Resumed from {ckpt_dir} tag={tag}")

    step_iter = iter(dataloader)
    epoch = 0
    for step in range(steps):
        if sampler is not None and step % len(dataloader) == 0:
            sampler.set_epoch(epoch)
            epoch += 1
        try:
            batch = next(step_iter)
        except StopIteration:
            step_iter = iter(dataloader)
            batch = next(step_iter)
        tokens = batch.to(dist_ctx.device)
        logits = engine(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1)
        )
        engine.backward(loss)
        engine.step()
        with torch.no_grad():
            teach_signal = compute_teach_signal(engine.module, logits, tokens)
            engine.module(tokens, teach_signal=teach_signal)
        if step % log_interval == 0 and engine.global_rank == 0:
            ppl = torch.exp(loss.detach()).item()
            logger.log({"loss": loss.item(), "ppl": ppl}, step=step)
            print(f"[DeepSpeed] step={step} loss={loss.item():.4f} ppl={ppl:.2f}")
        if (
            checkpoint_cfg.get("enable", False)
            and step % checkpoint_cfg.get("save_interval", 100) == 0
            and engine.global_rank == 0
        ):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            engine.save_checkpoint(str(ckpt_dir), tag=f"step_{step:06d}")

    logger.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
