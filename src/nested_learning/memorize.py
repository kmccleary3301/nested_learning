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
    use_fast_state: bool = True
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
        getattr(model, "set_allowed_update_levels")(
            prev_allowed if prev_allowed is None else set(prev_allowed)
        )
    if hasattr(model, "set_surprise_threshold"):
        getattr(model, "set_surprise_threshold")(prev_threshold)


def _collect_metrics(model, stats: dict[str, float]):
    """Helper to collect and aggregate update metrics."""
    if hasattr(model, "pop_update_metrics"):
        metrics = model.pop_update_metrics()
        titan_updates = sum(
            value for key, value in metrics.items() if key.endswith("titan.titan.grad_norm")
        )
        titan_hits = sum(
            value for key, value in metrics.items() if key.endswith("titan.titan.gate_hit")
        )
        cms_fast_updates = sum(
            value for key, value in metrics.items() if "cms.cms_fast.grad_norm" in key
        )
        cms_fast_hits = sum(
            value for key, value in metrics.items() if key.endswith("cms.cms_fast.gate_hit")
        )
        stats["cms_fast_updates"] += cms_fast_updates
        stats["titan_mem_updates"] += titan_updates
        stats["titan_update_events"] += titan_hits
        stats["cms_fast_update_events"] += cms_fast_hits


def memorize_tokens(
    model,
    token_batch: torch.Tensor,
    cfg: MemorizeConfig,
    *,
    fast_state=None,
) -> dict[str, float]:
    if token_batch.size(1) < 2:
        return {}

    if cfg.use_fast_state and fast_state is None:
        raise ValueError("cfg.use_fast_state=True requires passing fast_state")

    with torch.no_grad():
        stats: dict[str, float] = {
            "titan_mem_updates": 0.0,
            "cms_fast_updates": 0.0,
            "titan_update_events": 0.0,
            "cms_fast_update_events": 0.0,
        }
        prev_allowed, prev_threshold = _setup_memorization_context(model, cfg)

        if cfg.online_chunk_size and cfg.online_chunk_size > 0:
            # Online / Chunked Learning Mode
            seq_len = token_batch.size(1)
            chunk_size = cfg.online_chunk_size

            # We process the sequence in increasing windows
            # But to avoid O(N^2) cost for very long sequences, this is an approximation
            # where we re-process the history. For faithful online learning, this is necessary
            # without external KV cache management.

            # Note: compute_teach_signal computes gradients for predicting tokens[1:]
            # token_batch: [t0, t1, t2, t3]
            # logits: [p1, p2, p3, p4] (aligned with t0..t3 input)
            # teach_signal index i corresponds to error on token[i+1]

            # We iterate over target token indices (1..seq_len-1) in chunks.
            # For targets up to index K (exclusive end), feed tokens[:, :K] as context.
            target_start = 1
            while target_start < seq_len:
                target_end = min(target_start + chunk_size, seq_len)
                # We want to learn targets [target_start ... target_end]
                # (python slice style end index).
                # Range: target_start until target_end.

                # To compute error for target at index K, we need input 0..K.
                # So we need input up to target_end-1? No, up to target_end.
                # Because compute_teach_signal aligns logits[:-1] with tokens[1:].
                # If tokens is [A, B], logits[:-1] is preds for [B].
                # So if we have input [A, B], we get error for B.
                # If we have input [A, B, C], we get error for B, C.

                # So to get error for targets up to target_end-1 (python slice),
                # we need input tokens[:, :target_end].

                context_tokens = token_batch[:, :target_end]

                logits = (
                    model(context_tokens, fast_state=fast_state)
                    if cfg.use_fast_state
                    else model(context_tokens)
                )
                full_signal = compute_teach_signal(model, logits, context_tokens)

                # full_signal length is target_end.
                # indices correspond to errors for targets at 1 ... target_end.
                # idx 0 -> target 1.
                # idx k -> target k+1.

                # We want to keep errors for targets [target_start ... target_end-1].
                # These correspond to signal indices [target_start-1 ... target_end-2].

                # Example: [A, B, C]. target_start=1 (B). target_end=2 (up to B).
                # chunk=1.
                # context [A, B].
                # signal len 2. idx 0->B. idx 1->pad.
                # We want B. idx 0.
                # signal indices: target_start-1 (0) to target_end-1 (1)?
                # Wait, if target_end is 2 (slice), we processed B.
                # signal indices: 1-1=0. 2-2=0. Range 0:1.

                mask = torch.zeros_like(full_signal)
                mask_start = target_start - 1
                mask_end = target_end - 1
                mask[:, mask_start:mask_end, :] = 1.0

                masked_signal = full_signal * mask
                if cfg.use_fast_state:
                    model(context_tokens, teach_signal=masked_signal, fast_state=fast_state)
                else:
                    model(context_tokens, teach_signal=masked_signal)
                _collect_metrics(model, stats)

                target_start = target_end

        else:
            # Batch Mode (Default)
            for _ in range(cfg.steps):
                logits = (
                    model(token_batch, fast_state=fast_state)
                    if cfg.use_fast_state
                    else model(token_batch)
                )
                teach_signal = compute_teach_signal(model, logits, token_batch)
                surprise = float(torch.norm(teach_signal))
                if cfg.surprise_threshold is not None and surprise < cfg.surprise_threshold:
                    continue
                if cfg.use_fast_state:
                    model(token_batch, teach_signal=teach_signal, fast_state=fast_state)
                else:
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
    *,
    fast_state=None,
) -> dict[str, float]:
    if not text:
        return {}
    tokens = tokenizer.encode(text)
    if tokens.size(0) < 2:
        return {}
    batch = tokens.to(device).unsqueeze(0)
    return memorize_tokens(model, batch, cfg, fast_state=fast_state)
