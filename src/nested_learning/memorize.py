from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .training import compute_teach_signal
from .tokenizer import SentencePieceTokenizer


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
        getattr(model, "set_allowed_update_levels")(prev_allowed if prev_allowed is None else set(prev_allowed))
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
    if token_batch.size(1) < 2:
        return {}

    with torch.no_grad():
        stats: dict[str, float] = {"titan_mem_updates": 0.0, "cms_fast_updates": 0.0}
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
            
            # We start from a minimal context.
            start_idx = 0
            current_idx = 0
            
            while current_idx < seq_len - 1:
                # Define the active window we want to learn from
                # We predict tokens from current_idx+1 to end_idx
                end_idx = min(current_idx + chunk_size, seq_len - 1)
                
                # Context includes everything up to end_idx
                # We feed tokens[0 ... end_idx]
                # This produces logits for tokens[1 ... end_idx+1]
                
                # Actually, to predict token at K, we need input at K-1.
                # Inputs: tokens[:, :end_idx] -> Length L
                # Logits: Length L
                # Logits[-1] predicts tokens[end_idx]
                
                # We want to update based on errors for tokens[current_idx+1 : end_idx+1]
                # These correspond to logits at indices [current_idx : end_idx]
                
                # Wait, indices:
                # Tokens: [A, B, C, D, E]
                # Inputs: [A, B, C] (len 3) -> Logits for [B, C, D]
                # teach_signal has len 3. 
                # idx 0: error for B. idx 1: error for C. idx 2: error for D.
                
                # We want to update on new tokens.
                # If previous loop covered up to B. current_idx points to index of B (1)?
                # Let's say current_idx is the number of tokens "already learned".
                
                # Iteration 1: Learn [B, C]. (chunk=2).
                # Feed [A, B, C]. 
                # Teach signal for [B, C]. (Indices 0, 1 of teach_signal? No, indices of full seq?)
                
                # Let's stick to the "Sub-batch" logic.
                # Sub-batch inputs: tokens[:, :end_idx+1] (Input includes the token we want to predict? No.)
                # Standard causal LM: Input [0..N] predicts [1..N+1].
                # So to learn token at index K, we need input up to K-1.
                
                # Loop:
                # We want to learn tokens at indices [current_idx+1 ... end_idx+1] (1-based target indices)
                # Corresponding inputs are at [0 ... end_idx]
                
                # Let's simplify: We iterate through the TARGET tokens.
                # Target indices to learn: range(current_target_start, min(current_target_start + chunk, seq_len))
                # Start: 1.
                
                pass_end_idx = min(current_idx + chunk_size, seq_len - 1)
                # We want to learn targets at indices [current_idx+1 ... pass_end_idx+1]
                # But compute_teach_signal works on the sequence provided.
                
                # We simply feed the accumulated context + new chunk.
                # context_tokens = token_batch[:, :pass_end_idx+1] (Input tokens)
                # Targets will be context_tokens[:, 1:]
                
                context_tokens = token_batch[:, :pass_end_idx+1]
                logits = model(context_tokens)
                
                # Full signal for the current context
                full_signal = compute_teach_signal(model, logits, context_tokens)
                
                # Mask out the part we already learned (indices < current_idx)
                # teach_signal has length pass_end_idx+1.
                # Indices 0..current_idx-1 are for targets 1..current_idx (already learned)
                # We zero them out.
                
                mask = torch.zeros_like(full_signal)
                if current_idx > 0:
                    mask[:, current_idx:, :] = 1.0
                else:
                    mask[:, :, :] = 1.0
                    
                masked_signal = full_signal * mask
                
                # Update
                model(context_tokens, teach_signal=masked_signal)
                _collect_metrics(model, stats)
                
                current_idx = pass_end_idx + 1 # Advance
                # Wait, if pass_end_idx was 1 (targets[1] learned).
                # Next start is 2. Correct.
                
                # Because of loop condition while current_idx < seq_len - 1:
                # If seq_len=5. indices 0..4.
                # Targets 1..4.
                # if current_idx = 4. 4 < 4 False. Done.
                
                # Logic check:
                # Batch [A, B, C]. Seq len 3. Targets B, C.
                # current=0. chunk=1.
                # pass_end=0. context=[A]. targets=[?]. 
                # compute_teach_signal on [A]:
                #   targets=[]. residual=[]. grad=[pad].
                #   This does nothing useful?
                #   Wait, compute_teach_signal requires at least 2 tokens to compute useful grad for the last one?
                #   If input is [A], logits predict next. But we don't have next target in context_tokens.
                #   So compute_teach_signal pads the last position.
                #   So we need input [A, B] to get error for B.
                
                # Correct logic:
                # To learn target B (index 1), we need input [A, B].
                # Then compute_teach_signal uses B as target for A's output.
                
                # So to learn up to index K (target), we need inputs 0..K.
                # pass_end_idx should be the index of the LAST TARGET TOKEN.
                
                # Let's restart loop index logic.
                # We want to cover target indices 1 to seq_len-1.
                # target_start = 1.
                
            target_start = 1
            while target_start < seq_len:
                target_end = min(target_start + chunk_size, seq_len)
                # We want to learn targets [target_start ... target_end] (exclusive end? no, python slice style)
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
                
                logits = model(context_tokens)
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
