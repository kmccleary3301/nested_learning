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


def snapshot_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=False)


def memorize_tokens(model, token_batch: torch.Tensor, steps: int) -> None:
    if token_batch.size(1) < 2:
        return
    with torch.no_grad():
        for _ in range(steps):
            logits = model(token_batch)
            teach_signal = compute_teach_signal(model, logits, token_batch)
            model(token_batch, teach_signal=teach_signal)


def memorize_sequence(
    model,
    tokenizer: SentencePieceTokenizer,
    text: str,
    device: torch.device,
    steps: int,
) -> None:
    if not text:
        return
    tokens = tokenizer.encode(text)
    if tokens.size(0) < 2:
        return
    batch = tokens.to(device).unsqueeze(0)
    memorize_tokens(model, batch, steps)
