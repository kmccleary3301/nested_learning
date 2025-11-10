from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from .data import (
    MixtureShardDataset,
    ShardSourceConfig,
    SyntheticTextConfig,
    SyntheticTextDataset,
    TokenShardDataset,
    collate_batch,
)
from .levels import LevelSpec
from .logging_utils import NullLogger, init_logger
from .model import HOPEModel, ModelConfig
from .titan.model import TitanOnlyModel, TitanOnlyModelConfig


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device


def build_model_from_cfg(model_cfg: DictConfig) -> torch.nn.Module:
    model_type = model_cfg.get("type", "hope")
    optimizer_cfg = {}
    if "optimizers" in model_cfg:
        optimizer_cfg = OmegaConf.to_container(model_cfg.optimizers, resolve=True)
    teach_scale = model_cfg.get("teach_scale", 1.0)
    teach_clip = model_cfg.get("teach_clip", 0.0)
    teach_schedule = {}
    if "teach_schedule" in model_cfg:
        teach_schedule = OmegaConf.to_container(model_cfg.teach_schedule, resolve=True)  # type: ignore[arg-type]
    if model_type == "titan":
        titan_spec = LevelSpec(**model_cfg.titan_level)
        titan_cfg = TitanOnlyModelConfig(
            vocab_size=model_cfg.vocab_size,
            dim=model_cfg.dim,
            num_layers=model_cfg.num_layers,
            heads=model_cfg.heads,
            titan_level=titan_spec,
            optimizers=optimizer_cfg,
            teach_scale=teach_scale,
            teach_clip=teach_clip,
            teach_schedule=teach_schedule,
        )
        return TitanOnlyModel(titan_cfg)
    titan_spec = LevelSpec(**model_cfg.titan_level)
    cms_specs = [LevelSpec(**entry) for entry in model_cfg.cms_levels]
    hope_cfg = ModelConfig(
        vocab_size=model_cfg.vocab_size,
        dim=model_cfg.dim,
        num_layers=model_cfg.num_layers,
        heads=model_cfg.heads,
        titan_level=titan_spec,
        cms_levels=cms_specs,
        optimizers=optimizer_cfg,
        teach_scale=teach_scale,
        teach_clip=teach_clip,
        teach_schedule=teach_schedule,
    )
    return HOPEModel(hope_cfg)


def build_dataloader(
    data_cfg: DictConfig,
    *,
    distributed: bool,
    dist_ctx: DistributedContext | None,
) -> Tuple[DataLoader, DistributedSampler | None]:
    dataset = _build_dataset(data_cfg)
    use_sampler = distributed and not isinstance(dataset, IterableDataset)
    if use_sampler:
        assert dist_ctx is not None
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=False,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    if isinstance(dataset, IterableDataset):
        shuffle = False
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_batch,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
    )
    return dataloader, sampler


def _build_dataset(data_cfg: DictConfig):
    source = data_cfg.source
    if source == "synthetic":
        synth_cfg = SyntheticTextConfig(
            vocab_size=data_cfg.vocab_size,
            seq_len=data_cfg.seq_len,
            dataset_size=data_cfg.dataset_size,
        )
        return SyntheticTextDataset(synth_cfg)
    if source == "shards":
        shard_dir = data_cfg.shards_dir
        return TokenShardDataset(shard_dir)
    if source == "mixture":
        mixture_cfg = data_cfg.mixture
        sources = [
            ShardSourceConfig(
                name=entry.name,
                shards_dir=entry.shards_dir,
                weight=entry.weight,
            )
            for entry in mixture_cfg.sources
        ]
        samples_per_epoch = mixture_cfg.samples_per_epoch
        seed = mixture_cfg.get("seed", 0)
        return MixtureShardDataset(
            sources,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )
    msg = f"Unsupported data source {source}"
    raise ValueError(msg)


def compute_teach_signal(model: HOPEModel, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.detach(), dim=-1)
    targets = torch.nn.functional.one_hot(tokens, probs.size(-1)).float()
    residual = probs - targets
    embed = model.embed.weight.detach()
    return residual @ embed


def maybe_save_checkpoint(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    distributed: bool,
    dist_ctx: DistributedContext | None,
) -> None:
    ckpt_cfg = cfg.train.get("checkpoint")
    if not ckpt_cfg or not ckpt_cfg.get("enable", False):
        return
    if distributed and dist_ctx is not None and dist_ctx.rank != 0:
        return
    save_interval = ckpt_cfg.get("save_interval", total_steps)
    save_last = ckpt_cfg.get("save_last", True)
    is_last_step = (step + 1) >= total_steps
    should_save = ((step + 1) % max(1, save_interval) == 0) or (save_last and is_last_step)
    if not should_save:
        return
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/default"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"step_{step + 1:06d}.pt"
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step + 1,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(state, ckpt_path)
    prefix = "[checkpoint]"
    if distributed and dist_ctx is not None:
        prefix = f"[checkpoint rank={dist_ctx.rank}]"
    print(f"{prefix} saved {ckpt_path}")


def run_training_loop(
    cfg: DictConfig,
    *,
    device: torch.device,
    distributed: bool = False,
    dist_ctx: DistributedContext | None = None,
) -> Dict[str, float]:
    model = build_model_from_cfg(cfg.model).to(device)
    if distributed:
        assert dist_ctx is not None
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )
        base_model = model.module
    else:
        base_model = model

    dataloader, sampler = build_dataloader(cfg.data, distributed=distributed, dist_ctx=dist_ctx)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=cfg.optim.lr)
    logger = init_logger(getattr(cfg, "logging", None), cfg)
    if distributed and dist_ctx is not None and dist_ctx.rank != 0:
        logger = NullLogger()
    steps = cfg.train.steps
    log_interval = cfg.train.get("log_interval", 1)
    step_iter = iter(dataloader)
    epoch = 0
    metrics: Dict[str, float] = {}
    for step in range(steps):
        if sampler is not None and step % len(dataloader) == 0:
            sampler.set_epoch(epoch)
            epoch += 1
        try:
            batch = next(step_iter)
        except StopIteration:
            step_iter = iter(dataloader)
            batch = next(step_iter)
        tokens = batch.to(device)
        _apply_teach_schedule(base_model, cfg, step)
        logits = model(tokens) if not isinstance(model, torch.nn.parallel.DistributedDataParallel) else model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            teach_signal = compute_teach_signal(base_model, logits, tokens)
            base_model(tokens, teach_signal=teach_signal)
        if step % log_interval == 0:
            ppl = torch.exp(loss.detach()).item()
            logger.log({"loss": loss.item(), "ppl": ppl}, step=step)
            if (not distributed) or (dist_ctx and dist_ctx.rank == 0):
                print(f"[train] step={step} loss={loss.item():.4f} ppl={ppl:.2f}")
            metrics = {"loss": loss.item(), "ppl": ppl}
        maybe_save_checkpoint(
            cfg,
            base_model,
            optimizer,
            step=step,
            total_steps=steps,
            distributed=distributed,
            dist_ctx=dist_ctx,
        )
    logger.finish()
    return metrics


def _apply_teach_schedule(model: HOPEModel, cfg: DictConfig, step: int) -> None:
    schedule = cfg.model.get("teach_schedule")
    base_scale = cfg.model.get("teach_scale", 1.0)
    scale = base_scale
    if schedule:
        warmup = schedule.get("warmup_steps", 0)
        if warmup and warmup > 0:
            scale *= min(1.0, (step + 1) / warmup)
        decay_start = schedule.get("decay_start")
        decay_duration = schedule.get("decay_duration")
        if decay_start is not None and decay_duration and decay_duration > 0 and (step + 1) > decay_start:
            progress = min(1.0, (step + 1 - decay_start) / decay_duration)
            scale *= max(0.0, 1.0 - progress)
    model.set_teach_runtime(scale=scale)
