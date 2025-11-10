from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


@dataclass
class SyntheticTextConfig:
    vocab_size: int
    seq_len: int
    dataset_size: int


class SyntheticTextDataset(Dataset[torch.Tensor]):
    def __init__(self, config: SyntheticTextConfig):
        self.config = config

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(idx)
        return torch.randint(0, self.config.vocab_size, (self.config.seq_len,), generator=g)


class TokenShardDataset(Dataset[torch.Tensor]):
    """Memory-mapped dataset over NumPy shards produced by shard_corpus.py."""

    def __init__(self, shard_dir: str | Path):
        self.shard_dir = Path(shard_dir)
        if not self.shard_dir.exists():
            msg = f"Shard directory {self.shard_dir} does not exist"
            raise FileNotFoundError(msg)
        self.paths = sorted(self.shard_dir.glob("*.npy"))
        if not self.paths:
            msg = f"No shard files found in {self.shard_dir}"
            raise ValueError(msg)
        self.metadata: List[tuple[int, int]] = []
        self._cache: dict[int, np.memmap] = {}
        total = 0
        for idx, path in enumerate(self.paths):
            arr = np.load(path, mmap_mode="r")
            length = arr.shape[0]
            self.metadata.append((total, length))
            total += length
        self.total_sequences = total

    def __len__(self) -> int:
        return self.total_sequences

    def _load_array(self, shard_idx: int) -> np.memmap:
        if shard_idx not in self._cache:
            self._cache[shard_idx] = np.load(self.paths[shard_idx], mmap_mode="r")
        return self._cache[shard_idx]

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(idx)
        shard_idx = self._find_shard(idx)
        start_offset = self.metadata[shard_idx][0]
        arr = self._load_array(shard_idx)
        local_idx = idx - start_offset
        tokens = torch.from_numpy(arr[local_idx])
        return tokens.long()

    def _find_shard(self, idx: int) -> int:
        lo, hi = 0, len(self.metadata) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, length = self.metadata[mid]
            if idx < start:
                hi = mid - 1
            elif idx >= start + length:
                lo = mid + 1
            else:
                return mid
        return len(self.metadata) - 1


@dataclass
class ShardSourceConfig:
    name: str
    shards_dir: str
    weight: float


class ShardSource:
    def __init__(self, config: ShardSourceConfig):
        self.name = config.name
        self.weight = config.weight
        self.dir = Path(config.shards_dir)
        if not self.dir.exists():
            msg = f"Shard directory {self.dir} missing for source {self.name}"
            raise FileNotFoundError(msg)
        self.paths = sorted(self.dir.glob("*.npy"))
        if not self.paths:
            raise ValueError(f"No shard files in {self.dir}")
        self._cache: dict[Path, np.memmap] = {}

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        shard_path = self.paths[rng.integers(0, len(self.paths))]
        if shard_path not in self._cache:
            self._cache[shard_path] = np.load(shard_path, mmap_mode="r")
        shard = self._cache[shard_path]
        idx = rng.integers(0, shard.shape[0])
        return shard[idx]


class MixtureShardDataset(IterableDataset[torch.Tensor]):
    def __init__(
        self,
        sources: Sequence[ShardSourceConfig],
        *,
        samples_per_epoch: int,
        seed: int = 0,
    ):
        super().__init__()
        self.sources = [ShardSource(cfg) for cfg in sources]
        total_weight = sum(max(src.weight, 0.0) for src in self.sources)
        if total_weight <= 0:
            raise ValueError("Mixture weights must sum to > 0")
        self.weights = np.array([max(src.weight, 0.0) / total_weight for src in self.sources])
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        if worker is None:
            start = 0
            end = self.samples_per_epoch
            worker_seed = self.seed
        else:
            per_worker = (self.samples_per_epoch + worker.num_workers - 1) // worker.num_workers
            start = worker.id * per_worker
            end = min(start + per_worker, self.samples_per_epoch)
            worker_seed = self.seed + worker.id
        rng = np.random.default_rng(worker_seed)
        for _ in range(start, end):
            idx = rng.choice(len(self.sources), p=self.weights)
            sample = np.array(self.sources[idx].sample(rng), copy=True)
            yield torch.from_numpy(sample).long()


def collate_batch(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)
