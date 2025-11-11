from __future__ import annotations

import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    state_dict_type,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from nested_learning.logging_utils import NullLogger, init_logger
from nested_learning.training import (
    DistributedContext,
    build_dataloader,
    build_model_from_cfg,
    compute_teach_signal,
    _build_optimizer,
    _make_autocast_factory,
    _maybe_compile_model,
    _seed_everything,
    unwrap_config,
)


def setup_distributed() -> DistributedContext:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    return DistributedContext(rank=rank, world_size=world_size, device=device)


def build_fsdp_model(cfg: DictConfig, device: torch.device) -> tuple[FSDP, torch.nn.Module]:
    base_model = build_model_from_cfg(cfg.model).to(device)
    base_model = _maybe_compile_model(base_model, cfg.train.get("compile"))
    fsdp_cfg = cfg.train.get("fsdp", {})
    min_params = fsdp_cfg.get("auto_wrap_min_params", 2_000_000)
    auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=min_params)
    cpu_offload = CPUOffload(offload_params=fsdp_cfg.get("cpu_offload", False))
    model = FSDP(
        base_model,
        device_id=device.index,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload,
    )
    return model, base_model


def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    if hasattr(module, "_fsdp_wrapped_module"):
        return module._fsdp_wrapped_module  # type: ignore[attr-defined]
    if hasattr(module, "module"):
        return module.module  # type: ignore[attr-defined]
    return module


def save_checkpoint(cfg: DictConfig, model: FSDP, optimizer: torch.optim.Optimizer, step: int, rank: int) -> None:
    ckpt_cfg = cfg.train.get("checkpoint")
    if not ckpt_cfg or not ckpt_cfg.get("enable", False) or step % ckpt_cfg.get("save_interval", 100) != 0:
        return
    if rank != 0:
        return
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/fsdp"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        model_state = model.state_dict()
    state = {"model": model_state, "optimizer": optimizer.state_dict(), "step": step}
    torch.save(state, ckpt_path)


def maybe_resume(cfg: DictConfig, model: FSDP, optimizer: torch.optim.Optimizer, rank: int) -> int:
    ckpt_cfg = cfg.train.get("checkpoint")
    if not ckpt_cfg:
        return 0
    resume_path = ckpt_cfg.get("resume_path")
    if not resume_path:
        return 0
    if not Path(resume_path).exists():
        raise FileNotFoundError(f"Resume checkpoint {resume_path} not found")
    map_location = "cpu"
    state = torch.load(resume_path, map_location=map_location)
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if rank == 0:
        print(f"[FSDP] Resumed from {resume_path} at step {state.get('step', 0)}")
    return state.get("step", 0)


@hydra.main(config_path="configs", config_name="hope/mid", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    dist_ctx = setup_distributed()
    train_seed = cfg.train.get("seed")
    deterministic = cfg.train.get("deterministic", False)
    if train_seed is not None:
        _seed_everything(int(train_seed), deterministic=bool(deterministic))
    model, _ = build_fsdp_model(cfg, dist_ctx.device)
    inner_model = unwrap_model(model)
    train_seed = cfg.train.get("seed")
    loader_seed = None if train_seed is None else int(train_seed) + dist_ctx.rank
    dataloader, sampler = build_dataloader(
        cfg.data,
        distributed=True,
        dist_ctx=dist_ctx,
        seed=loader_seed,
    )
    optimizer = _build_optimizer(model, cfg, device=dist_ctx.device)
    start_step = maybe_resume(cfg, model, optimizer, dist_ctx.rank)
    logger = init_logger(getattr(cfg, "logging", None), cfg) if dist_ctx.rank == 0 else NullLogger()
    autocast_factory = _make_autocast_factory(dist_ctx.device, cfg.train.get("mixed_precision"))

    steps = cfg.train.steps
    log_interval = cfg.train.get("log_interval", 10)
    step_iter = iter(dataloader)
    epoch = 0
    for step in range(start_step, steps):
        if sampler is not None and step % len(dataloader) == 0:
            sampler.set_epoch(epoch)
            epoch += 1
        try:
            batch = next(step_iter)
        except StopIteration:
            step_iter = iter(dataloader)
            batch = next(step_iter)
        tokens = batch.to(dist_ctx.device)
        with autocast_factory():
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1)
            )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            teach_signal = compute_teach_signal(inner_model, logits, tokens)
            inner_model(tokens, teach_signal=teach_signal)
        if step % log_interval == 0 and dist_ctx.rank == 0:
            ppl = torch.exp(loss.detach()).item()
            logger.log({"loss": loss.item(), "ppl": ppl}, step=step)
            print(f"[fsdp] step={step} loss={loss.item():.4f} ppl={ppl:.2f}")
        save_checkpoint(cfg, model, optimizer, step, dist_ctx.rank)

    logger.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
