from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .tokenizer import SentencePieceTokenizer
from .training import compute_teach_signal


@dataclass
class MemorizeConfig:
    enabled: bool = False
    steps: int = 1
    reset: bool = True
    use_correct_answer: bool = False
    surprise_threshold: float | None = None
    paths: tuple[str, ...] | None = None
    online_chunk_size: int | None = None  # If set, use online chunked updates


def snapshot_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=False)


def _setup_memorization_context(model, cfg: MemorizeConfig):
    """Helper to setup model state for memorization."""
    prev_allowed = getattr(model, "get_allowed_update_levels", lambda: None)()
    prev_threshold = getattr(model, "get_surprise_threshold", lambda: None)()

    if hasattr(model, "set_allowed_update_levels"):
        allowed = None
        if cfg.paths is not None:
            allowed = {path.strip() for path in cfg.paths if path.strip()}
        getattr(model, "set_allowed_update_levels")(allowed)

    if cfg.surprise_threshold is not None and hasattr(model, "set_surprise_threshold"):
        getattr(model, "set_surprise_threshold")(cfg.surprise_threshold)

    return prev_allowed, prev_threshold


def _teardown_memorization_context(model, prev_allowed, prev_threshold):
    """Helper to restore model state after memorization."""
    if hasattr(model, "set_allowed_update_levels"):
        restored = None if prev_allowed is None else set(prev_allowed)
        getattr(model, "set_allowed_update_levels")(restored)
    if hasattr(model, "set_surprise_threshold"):
        getattr(model, "set_surprise_threshold")(prev_threshold)


def _collect_metrics(model, stats: dict[str, float]):
    """Helper to collect and aggregate update metrics."""
    if hasattr(model, "pop_update_metrics"):
        metrics = model.pop_update_metrics()
        titan_updates = sum(
            value for key, value in metrics.items() if key.endswith("titan.titan.grad_norm")
        )
        cms_fast_updates = sum(
            value for key, value in metrics.items() if "cms.cms_fast.grad_norm" in key
        )
        stats["cms_fast_updates"] += cms_fast_updates
        stats["titan_mem_updates"] += titan_updates


def memorize_tokens(model, token_batch: torch.Tensor, cfg: MemorizeConfig) -> dict[str, float]:
    if not cfg.enabled:
        return {}
    if token_batch.size(1) < 2:
        return {}

    with torch.no_grad():
        stats: dict[str, float] = {"titan_mem_updates": 0.0, "cms_fast_updates": 0.0}
        prev_allowed, prev_threshold = _setup_memorization_context(model, cfg)

        if cfg.online_chunk_size and cfg.online_chunk_size > 0:
            seq_len = token_batch.size(1)
            chunk_size = int(cfg.online_chunk_size)

            # Signal index i corresponds to the error for token i+1. The final
            # position is always padding (no next-token target), so we mask that
            # out by keeping indices [target_start-1 : target_end-1].
            target_start = 1
            while target_start < seq_len:
                target_end = min(target_start + chunk_size, seq_len)
                context_tokens = token_batch[:, :target_end]

                logits = model(context_tokens)
                full_signal = compute_teach_signal(model, logits, context_tokens)

                mask = torch.zeros_like(full_signal)
                mask_start = target_start - 1
                mask_end = target_end - 1
                mask[:, mask_start:mask_end, :] = 1.0

                masked_signal = full_signal * mask
                model(context_tokens, teach_signal=masked_signal)
                _collect_metrics(model, stats)

                target_start = target_end

        else:
            # Batch Mode (Default)
            for _ in range(cfg.steps):
                logits = model(token_batch)
                teach_signal = compute_teach_signal(model, logits, token_batch)
                surprise = float(torch.norm(teach_signal))
                if cfg.surprise_threshold is not None and surprise < cfg.surprise_threshold:
                    continue
                model(token_batch, teach_signal=teach_signal)
                _collect_metrics(model, stats)

        _teardown_memorization_context(model, prev_allowed, prev_threshold)
        return stats


def memorize_sequence(
    model,
    tokenizer: SentencePieceTokenizer,
    text: str,
    device: torch.device,
    cfg: MemorizeConfig,
) -> dict[str, float]:
    if not text:
        return {}
    tokens = tokenizer.encode(text)
    if tokens.size(0) < 2:
        return {}
    batch = tokens.to(device).unsqueeze(0)
    return memorize_tokens(model, batch, cfg)
