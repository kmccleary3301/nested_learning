from __future__ import annotations

from pathlib import Path
from typing import Sequence

import sentencepiece as spm
import torch


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path):
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> torch.Tensor:
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.processor.bos_id())
        tokens.extend(self.processor.encode(text))
        if add_eos:
            tokens.append(self.processor.eos_id())
        return torch.tensor(tokens, dtype=torch.long)

    def batch_encode(self, texts: Sequence[str]) -> list[torch.Tensor]:
        return [self.encode(text) for text in texts]
