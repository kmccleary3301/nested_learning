from __future__ import annotations

import base64
import json
import os
import pickle
import random
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
from .logging_utils import BaseLogger, NullLogger, init_logger
from .model import HOPEModel, ModelConfig
from .titan.model import TitanOnlyModel, TitanOnlyModelConfig


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device


def unwrap_config(cfg: DictConfig) -> DictConfig:
    """Hydra can wrap grouped configs (e.g., hope/pilot) under the group name."""
    if "model" in cfg:
        return cfg
    if "hope" in cfg:
        return cfg.hope
    if "ablations" in cfg:
        return cfg.ablations
    return cfg


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
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
    )
    return HOPEModel(hope_cfg)


def build_dataloader(
    data_cfg: DictConfig,
    *,
    distributed: bool,
    dist_ctx: DistributedContext | None,
    seed: int | None = None,
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
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _make_worker_init_fn(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_batch,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
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


def compute_teach_signal(model: torch.nn.Module, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Approximate dL/dh where h is the hidden state before the LM head.
    Aligns with CE(logits[:, :-1], tokens[:, 1:]) used in training.
    """
    logits_detached = logits.detach()
    probs = torch.softmax(logits_detached, dim=-1)
    target_tokens = tokens[:, 1:]
    targets = F.one_hot(target_tokens, probs.size(-1)).float()
    residual = probs[:, :-1] - targets  # only positions used in loss
    denom = max(1, tokens.size(0) * max(1, tokens.size(1) - 1))
    residual = residual / denom
    head_weight = model.lm_head.weight.detach()
    grad = residual @ head_weight
    pad = torch.zeros(
        grad.size(0),
        1,
        grad.size(-1),
        device=grad.device,
        dtype=grad.dtype,
    )
    return torch.cat([grad, pad], dim=1)


def _checksum_path(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maybe_save_checkpoint(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    distributed: bool,
    dist_ctx: DistributedContext | None,
    step_offset: int = 0,
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
    global_step = step + 1 + int(step_offset)
    ckpt_path = ckpt_dir / f"step_{global_step:06d}.pt"
    tmp_path = ckpt_path.with_suffix(".tmp")
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step + 1,
        "config": resolved_cfg,
    }
    torch.save(state, tmp_path)
    os.replace(tmp_path, ckpt_path)
    write_checkpoint_metadata(cfg, ckpt_path, global_step)
    prefix = "[checkpoint]"
    if distributed and dist_ctx is not None:
        prefix = f"[checkpoint rank={dist_ctx.rank}]"
    print(f"{prefix} saved {ckpt_path} (global_step={global_step})")


def run_training_loop(
    cfg: DictConfig,
    *,
    device: torch.device,
    distributed: bool = False,
    dist_ctx: DistributedContext | None = None,
) -> Dict[str, float]:
    model = build_model_from_cfg(cfg.model).to(device)
    train_seed = cfg.train.get("seed")
    deterministic = cfg.train.get("deterministic", False)
    if train_seed is not None:
        _seed_everything(int(train_seed), deterministic=bool(deterministic))
    model = _maybe_compile_model(model, cfg.train.get("compile"))
    if distributed:
        assert dist_ctx is not None
        ddp_kwargs = {"find_unused_parameters": True}
        if device.type == "cuda":
            idx = device.index if device.index is not None else 0
            ddp_kwargs["device_ids"] = [idx]
            ddp_kwargs["output_device"] = idx
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        base_model = model.module
    else:
        base_model = model

    seed_offset = 0
    if train_seed is not None and dist_ctx is not None:
        seed_offset = dist_ctx.rank
    dataloader_seed = None if train_seed is None else int(train_seed) + seed_offset
    dataloader, sampler = build_dataloader(
        cfg.data,
        distributed=distributed,
        dist_ctx=dist_ctx,
        seed=dataloader_seed,
    )
    optimizer = _build_optimizer(base_model, cfg, device=device)
    autocast_factory = _make_autocast_factory(device, cfg.train.get("mixed_precision"))
    logger = init_logger(getattr(cfg, "logging", None), cfg)
    if distributed and dist_ctx is not None and dist_ctx.rank != 0:
        logger = NullLogger()
    _log_run_features(logger, base_model, cfg, optimizer, device)
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
        with autocast_factory():
            logits = model(tokens) if not isinstance(model, torch.nn.parallel.DistributedDataParallel) else model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1)
            )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
        optimizer.step()
        update_metrics: Dict[str, float] = {}
        with torch.no_grad():
            teach_signal = compute_teach_signal(base_model, logits, tokens)
            teach_signal_norm = teach_signal.norm(dim=-1).mean().item()
            base_model(tokens, teach_signal=teach_signal)
            if hasattr(base_model, "pop_update_metrics"):
                update_metrics = base_model.pop_update_metrics()
        if step % log_interval == 0:
            ppl = torch.exp(loss.detach()).item()
            metrics_payload = {"loss": loss.item(), "ppl": ppl, "teach_signal_norm": teach_signal_norm}
            metrics_payload.update(update_metrics)
            logger.log(metrics_payload, step=step)
            if (not distributed) or (dist_ctx and dist_ctx.rank == 0):
                print(
                    f"[train] step={step} loss={loss.item():.4f} "
                    f"ppl={ppl:.2f} teach_norm={teach_signal_norm:.4f}"
                )
            metrics = metrics_payload
        maybe_save_checkpoint(
            cfg,
            base_model,
            optimizer,
            step=step,
            total_steps=steps,
            distributed=distributed,
            dist_ctx=dist_ctx,
            step_offset=int(cfg.train.get("step_offset", 0) or 0),
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


def _maybe_compile_model(model: torch.nn.Module, compile_cfg: dict | None) -> torch.nn.Module:
    if not compile_cfg or not compile_cfg.get("enable", False):
        return model
    kwargs = {}
    if "mode" in compile_cfg:
        kwargs["mode"] = compile_cfg["mode"]
    if "backend" in compile_cfg:
        kwargs["backend"] = compile_cfg["backend"]
    try:
        return torch.compile(model, **kwargs)  # type: ignore[attr-defined]
    except Exception as err:  # pragma: no cover - compile is optional
        if compile_cfg.get("strict", False):
            raise
        print(f"[compile] fallback to eager due to: {err}")
        return model


def _make_autocast_factory(device: torch.device, mp_cfg: dict | None):
    if not mp_cfg or not mp_cfg.get("enabled", False):
        return lambda: nullcontext()
    dtype = _resolve_autocast_dtype(mp_cfg.get("dtype", "bf16"))
    device_type = "cuda" if device.type == "cuda" else "cpu"

    def factory():
        return torch.autocast(device_type=device_type, dtype=dtype)

    return factory


def _resolve_autocast_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    msg = f"Unsupported autocast dtype {name}"
    raise ValueError(msg)


def _build_optimizer(model: torch.nn.Module, cfg: DictConfig, *, device: torch.device) -> torch.optim.Optimizer:
    optimizer_cfg = getattr(cfg, "optim", {})
    optim_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if optim_type == "muon":
        return _build_muon_optimizer(model, optimizer_cfg, device=device)
    lr = optimizer_cfg.get("lr", 1e-3)
    betas = optimizer_cfg.get("betas", (0.9, 0.999))
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    fused_cfg = optimizer_cfg.get("fused", "auto")
    fused = False
    if fused_cfg == "auto":
        fused = device.type == "cuda" and torch.cuda.is_available()
    else:
        fused = bool(fused_cfg)
    kwargs = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
    if fused:
        kwargs["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kwargs)


def _build_muon_optimizer(model: torch.nn.Module, optimizer_cfg: DictConfig, *, device: torch.device):
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch build")
    lr = optimizer_cfg.get("lr", 1e-3)
    weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    momentum = optimizer_cfg.get("momentum", 0.95)
    ns_coefficients = optimizer_cfg.get("ns_coefficients")
    ns_steps = optimizer_cfg.get("ns_steps")
    eps = optimizer_cfg.get("eps", 1e-7)
    fused_cfg = optimizer_cfg.get("fused", "auto")
    fused = False
    if fused_cfg == "auto":
        fused = device.type == "cuda" and torch.cuda.is_available()
    else:
        fused = bool(fused_cfg)
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_muon_candidate(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    muon_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "eps": eps,
    }
    if ns_coefficients is not None:
        muon_kwargs["ns_coefficients"] = tuple(ns_coefficients)
    if ns_steps is not None:
        muon_kwargs["ns_steps"] = int(ns_steps)
    muon_opt = torch.optim.Muon(muon_params, **muon_kwargs) if muon_params else None  # type: ignore[attr-defined]
    adamw_kwargs = {"lr": lr, "betas": optimizer_cfg.get("betas", (0.9, 0.999)), "weight_decay": weight_decay}
    if fused:
        adamw_kwargs["fused"] = True
    adamw_opt = torch.optim.AdamW(adamw_params, **adamw_kwargs) if adamw_params else None
    muon_elems = int(sum(p.numel() for p in muon_params))
    adamw_elems = int(sum(p.numel() for p in adamw_params))
    return _HybridOptimizer(muon_opt, adamw_opt, muon_elems, adamw_elems)


def _is_muon_candidate(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim < 2:
        return False
    lowered = name.lower()
    if "norm" in lowered or "embed" in lowered:
        return False
    return True


class _HybridOptimizer:
    def __init__(
        self,
        muon_opt: torch.optim.Optimizer | None,
        adamw_opt: torch.optim.Optimizer | None,
        muon_param_elems: int,
        adamw_param_elems: int,
    ):
        self.muon_opt = muon_opt
        self.adamw_opt = adamw_opt
        self.muon_param_elems = muon_param_elems
        self.adamw_param_elems = adamw_param_elems

    def zero_grad(self) -> None:
        if self.muon_opt:
            self.muon_opt.zero_grad()
        if self.adamw_opt:
            self.adamw_opt.zero_grad()

    def step(self) -> None:
        if self.muon_opt:
            self.muon_opt.step()
        if self.adamw_opt:
            self.adamw_opt.step()

    def state_dict(self) -> dict:
        return {
            "muon": self.muon_opt.state_dict() if self.muon_opt else None,
            "adamw": self.adamw_opt.state_dict() if self.adamw_opt else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if self.muon_opt and state.get("muon") is not None:
            self.muon_opt.load_state_dict(state["muon"])
        if self.adamw_opt and state.get("adamw") is not None:
            self.adamw_opt.load_state_dict(state["adamw"])

    @property
    def param_groups(self):
        groups = []
        if self.muon_opt:
            groups.extend(self.muon_opt.param_groups)
        if self.adamw_opt:
            groups.extend(self.adamw_opt.param_groups)
        return groups

    def get_param_split(self) -> dict[str, int]:
        return {
            "muon": self.muon_param_elems,
            "adamw": self.adamw_param_elems,
        }


def _log_run_features(
    logger: BaseLogger,
    model: HOPEModel,
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    mp_cfg = cfg.train.get("mixed_precision", {})
    compile_cfg = cfg.train.get("compile", {})
    features: dict[str, object] = {
        "train.mixed_precision_enabled": bool(mp_cfg.get("enabled", False)),
        "train.mixed_precision_dtype": str(mp_cfg.get("dtype", "bf16")),
        "train.compile_enabled": bool(compile_cfg.get("enable", False)),
        "train.compile_mode": str(compile_cfg.get("mode", "default")) if compile_cfg else "default",
        "attention.flash_enabled": _detect_flash_attention(model),
        "device": device.type,
    }
    split_fn = getattr(optimizer, "get_param_split", None)
    if callable(split_fn):
        split = split_fn()
        features["optim.muon_param_elems"] = int(split.get("muon", 0))
        features["optim.adamw_param_elems"] = int(split.get("adamw", 0))
    logger.log(features, step=-1)
    print(f"[train] run_features {features}")


def _detect_flash_attention(model: HOPEModel) -> bool:
    blocks = getattr(model, "blocks", [])
    for block in blocks:
        attn = getattr(block, "attn", None)
        config = getattr(attn, "config", None)
        if config is not None and hasattr(config, "use_flash"):
            return bool(config.use_flash)
    return False


def write_checkpoint_metadata(cfg: DictConfig, ckpt_path: Path, step: int) -> None:
    config_yaml = OmegaConf.to_yaml(cfg)
    config_path = ckpt_path.with_suffix(".yaml")
    config_path.write_text(config_yaml)
    config_hash = sha256(config_yaml.encode("utf-8")).hexdigest()
    ckpt_hash = _checksum_path(str(ckpt_path))
    sha_path = ckpt_path.with_suffix(".sha256")
    if ckpt_hash:
        sha_path.write_text(f"{ckpt_hash}  {ckpt_path.name}\n")
    tokenizer_path = cfg.data.get("tokenizer_path") if hasattr(cfg, "data") else None
    metadata = {
        "step": step,
        "checkpoint_sha256": ckpt_hash,
        "config_sha256": config_hash,
        "tokenizer_hash": _checksum_path(tokenizer_path) if tokenizer_path else None,
        "config_path": str(config_path),
        "rng_states": _capture_rng_states(),
    }
    ckpt_path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))


def verify_checkpoint_integrity(ckpt_path: Path) -> Dict[str, object]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    meta_path = ckpt_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file {meta_path} missing")
    metadata = json.loads(meta_path.read_text())
    computed_sha = _checksum_path(str(ckpt_path))
    recorded_sha = metadata.get("checkpoint_sha256")
    if recorded_sha and computed_sha and recorded_sha != computed_sha:
        raise ValueError(f"Checkpoint SHA mismatch: recorded {recorded_sha} vs computed {computed_sha}")
    sha_file = ckpt_path.with_suffix(".sha256")
    if sha_file.exists() and computed_sha:
        recorded_line = sha_file.read_text().strip().split()
        if recorded_line:
            recorded = recorded_line[0]
            if recorded != computed_sha:
                raise ValueError(f".sha256 mismatch: {recorded} vs {computed_sha}")
    config_path = ckpt_path.with_suffix(".yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} missing")
    config_hash = sha256(config_path.read_text().encode("utf-8")).hexdigest()
    recorded_cfg_hash = metadata.get("config_sha256")
    if recorded_cfg_hash and recorded_cfg_hash != config_hash:
        raise ValueError(f"Config SHA mismatch: recorded {recorded_cfg_hash} vs computed {config_hash}")
    if "rng_states" not in metadata:
        raise ValueError("Metadata missing rng_states")
    return metadata


def _capture_rng_states() -> Dict[str, object]:
    payload: Dict[str, object] = {
        "python": _encode_pickle(random.getstate()),
        "numpy": _encode_pickle(np.random.get_state()),
        "torch": _tensor_state_to_hex(torch.random.get_rng_state()),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = [_tensor_state_to_hex(state) for state in torch.cuda.get_rng_state_all()]  # type: ignore[attr-defined]
    return payload


def _encode_pickle(obj: object) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _tensor_state_to_hex(state: torch.Tensor) -> str:
    return state.cpu().numpy().tobytes().hex()


def _seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]


def _make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_fn
